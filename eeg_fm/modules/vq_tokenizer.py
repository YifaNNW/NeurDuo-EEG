"""Per-channel spectral VQ tokenizer that assigns each 0.5 s chunk a code from its FFT amplitude and phase."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from eeg_fm.state import EEGMetadataBatch
from eeg_fm.tokenizer_config import TokenizerConfig


def _all_reduce_sum(tensor: torch.Tensor) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def _broadcast_from_src0(tensor: torch.Tensor) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(tensor, src=0)


class NormEMAVectorQuantizer(nn.Module):
    def __init__(self, cfg: TokenizerConfig) -> None:
        super().__init__()
        self.num_codes = cfg.codebook_size
        self.code_dim = cfg.code_dim
        self.decay = cfg.ema_decay
        self.eps = cfg.codebook_eps
        self.dead_code_reset = cfg.dead_code_reset
        self.dead_code_threshold = cfg.dead_code_threshold

        embedding = F.normalize(torch.randn(self.num_codes, self.code_dim), dim=-1)
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(self.num_codes))
        self.register_buffer("embed_avg", embedding.clone())
        self.register_buffer("initted", torch.zeros((), dtype=torch.bool))
        self.kmeans_init = cfg.kmeans_init

    @torch.no_grad()
    def _init_codebook(self, z: torch.Tensor) -> None:
        n = z.shape[0]
        idx = torch.randint(0, n, (self.num_codes,), device=z.device)
        sampled = z[idx].clone()
        self.embedding.copy_(sampled)
        self.embed_avg.copy_(sampled)
        self.cluster_size.fill_(1.0)
        _broadcast_from_src0(self.embedding)
        _broadcast_from_src0(self.embed_avg)
        _broadcast_from_src0(self.cluster_size)
        self.initted.fill_(True)

    def forward(self, z_e: torch.Tensor) -> dict[str, torch.Tensor]:
        z = F.normalize(z_e, dim=-1)
        if self.training and self.kmeans_init and not bool(self.initted):
            self._init_codebook(z)

        embed_norm = F.normalize(self.embedding, dim=-1).to(z.dtype)
        cos = z @ embed_norm.t()
        idx = cos.argmax(dim=-1)
        z_q = embed_norm[idx]

        commit_loss = F.mse_loss(z_q.detach(), z)
        z_q_st = z + (z_q - z).detach()

        if self.training:
            unused = self._ema_update(z, idx)
        else:
            unused = (self.cluster_size < self.dead_code_threshold).sum().float()

        with torch.no_grad():
            probs = torch.bincount(idx, minlength=self.num_codes).float()
            probs = probs / probs.sum().clamp_min(1.0)
            perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())

        return {
            "z_q": z_q_st,
            "commit": commit_loss,
            "unused_codes": unused,
            "perplexity": perplexity,
        }

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        nonfinite = (~torch.isfinite(z).all()).float().reshape(1)
        _all_reduce_sum(nonfinite)
        if nonfinite.item() > 0:
            return (self.cluster_size < self.dead_code_threshold).sum().float()
        z = z.float()
        onehot = F.one_hot(idx, self.num_codes).type_as(z)
        cluster_batch = onehot.sum(dim=0)
        embed_sum = onehot.t() @ z
        _all_reduce_sum(cluster_batch)
        _all_reduce_sum(embed_sum)

        self.cluster_size.mul_(self.decay).add_(cluster_batch, alpha=1.0 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(-1)
        self.embedding.copy_(F.normalize(embed_normalized, dim=-1))

        unused = (self.cluster_size < self.dead_code_threshold).sum().float()
        if self.dead_code_reset:
            self._revive_dead_codes(z)
        return unused

    @torch.no_grad()
    def _revive_dead_codes(self, z: torch.Tensor) -> None:
        dead = self.cluster_size < self.dead_code_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return
        n = z.shape[0]
        pick = torch.randint(0, n, (n_dead,), device=z.device)
        revived = z[pick]
        _broadcast_from_src0(revived)
        self.embedding[dead] = revived
        self.embed_avg[dead] = revived
        self.cluster_size[dead] = 1.0


class SpectralVQTokenizer(nn.Module):
    def __init__(self, cfg: TokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_freq = cfg.n_freq_bins
        feat_dim = 2 * self.n_freq

        self.input_proj = nn.Linear(feat_dim, cfg.enc_dim)
        self.per_channel_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(cfg.enc_dim),
                nn.Linear(cfg.enc_dim, cfg.enc_dim * 4),
                nn.GELU(),
                nn.Linear(cfg.enc_dim * 4, cfg.enc_dim),
            )
            for _ in range(cfg.enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(cfg.enc_dim)
        self.to_code = nn.Linear(cfg.enc_dim, cfg.code_dim)

        self.vq = NormEMAVectorQuantizer(cfg)

        self.code_proj = nn.Linear(cfg.code_dim, cfg.enc_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.enc_dim, cfg.dec_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.dec_hidden_dim, feat_dim),
        )

    def _spectra(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast("cuda", enabled=False):
            fft = torch.fft.rfft(x.float(), dim=-1)
            amp = self._std_norm(fft.abs())
            phase = self._std_norm(torch.angle(fft))
        return amp, phase

    @staticmethod
    def _std_norm(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-5)
        return (x - mean) / std

    def _encode(
        self, chunks: torch.Tensor, metadata: EEGMetadataBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C, L = chunks.shape
        N = B * T
        device = chunks.device
        flat = chunks.reshape(N, C, L)

        if metadata.missing_mask is not None:
            miss = metadata.missing_mask.to(device)
        else:
            miss = torch.zeros(B, C, device=device)
        miss = miss[:, None, :].expand(B, T, C).reshape(N, C)
        pad_mask = miss > 0.5

        amp, phase = self._spectra(flat)
        feat = torch.cat([amp, phase], dim=-1)
        x = self.input_proj(feat)

        for blk in self.per_channel_blocks:
            x = x + blk(x)
        enc = self.enc_norm(x)
        z_e = self.to_code(enc)
        return z_e, amp, phase, pad_mask

    def forward(self, chunks: torch.Tensor, metadata: EEGMetadataBatch) -> dict:
        if chunks.dim() != 4:
            raise ValueError(
                f"Expected chunks [B, T, C, L], got {tuple(chunks.shape)}."
            )
        B, T, C, L = chunks.shape
        z_e, amp, phase, pad_mask = self._encode(chunks, metadata)

        Dc = z_e.size(-1)
        keep = (~pad_mask).reshape(-1)
        z_flat = z_e.reshape(-1, Dc)
        vq = self.vq(z_flat[keep])
        z_q_full = z_flat.new_zeros(z_flat.shape)
        z_q_full[keep] = vq["z_q"].to(z_q_full.dtype)
        dec_in = self.code_proj(z_q_full.view(B * T, C, Dc))
        recon = self.decoder(dec_in)
        amp_hat, phase_hat = recon.split(self.n_freq, dim=-1)
        valid = (~pad_mask).float().unsqueeze(-1)
        denom = valid.sum().clamp_min(1.0) * self.n_freq
        rec_amp = (((amp_hat - amp) ** 2) * valid).sum() / denom
        rec_phase = (((phase_hat - phase) ** 2) * valid).sum() / denom
        commit = vq["commit"]
        loss = (self.cfg.amp_weight * rec_amp
                + self.cfg.phase_weight * rec_phase
                + self.cfg.commitment_beta * commit)
        return {
            "loss": loss,
            "metrics": {
                "loss_total": loss.detach(), "rec_amp": rec_amp.detach(),
                "rec_phase": rec_phase.detach(), "commit": commit.detach(),
                "unused_codes": vq["unused_codes"], "perplexity": vq["perplexity"],
            },
        }

    @torch.no_grad()
    def get_codebook_indices(self, chunks: torch.Tensor, metadata: EEGMetadataBatch) -> torch.Tensor:
        B, T, C = chunks.shape[0], chunks.shape[1], chunks.shape[2]
        z_e, _, _, pad_mask = self._encode(chunks, metadata)
        embed_norm = F.normalize(self.vq.embedding, dim=-1)
        z = F.normalize(z_e, dim=-1)
        idx = (z @ embed_norm.t()).argmax(dim=-1)
        idx = idx.masked_fill(pad_mask, -100)
        return idx.reshape(B, T, C)
