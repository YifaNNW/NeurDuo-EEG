"""Pretrained backbone with input chunking, window normalization, channel metadata and the shared readout head."""


import torch
import torch.nn as nn

from eeg_fm.config import EEGFMConfig
from eeg_fm.data.normalize import normalize_sequence
from eeg_fm.electrodes import coord_of, strip_reference
from eeg_fm.finetune.heads import STHead
from eeg_fm.model import EEGFM
from eeg_fm.pretrain_config import PretrainConfig
from eeg_fm.state import EEGMetadataBatch

NATIVE_FS = 256
REFERENCE_ID = 0


def load_model_config(checkpoint="", pretrain_config=""):
    if checkpoint:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg = EEGFMConfig.from_dict(payload['config']['model'])
        # Some paper-stage ablations removed modules with a runtime patch, so the stored config
        # does not say which ones are present. The weights do.
        state_dict = payload.get('student', payload)
        cfg.use_slow_stream = any(k.startswith('slow_backbone.') for k in state_dict)
        cfg.use_memory_context = any(k.startswith('memory_context.') for k in state_dict)
        return cfg
    if pretrain_config:
        return PretrainConfig.from_json(pretrain_config).model
    raise ValueError("A pretrained checkpoint or a pretraining config is required to build the model.")


def _coords(channel_names):
    out = []
    for name in channel_names:
        site, _ = strip_reference(name)
        xyz = coord_of(site) if site else None
        if xyz is None:
            raise ValueError(
                f"channel {name!r} has no standard_1005 coordinate; "
                f"pretraining metadata cannot be reproduced for it")
        out.append(xyz)
    return torch.tensor(out, dtype=torch.float32)


class Model(nn.Module):
    def __init__(self, params):
        super().__init__()
        channels = list(params.channels)
        self.C = len(channels)
        self.S = params.seg_sec

        cfg = params.model_config
        cfg.use_gradient_checkpointing = True
        self.cfg = cfg
        self.chunk = cfg.chunk_samples
        self.stride = cfg.chunk_stride_samples
        self.cont_normalize = getattr(params, 'cont_normalize', 'window')

        n_samples = int(round(self.S * NATIVE_FS))
        if (n_samples - self.chunk) % self.stride or n_samples < self.chunk:
            raise ValueError(
                f"{self.S}s @{NATIVE_FS}Hz = {n_samples} samples does not tile "
                f"with chunk={self.chunk} stride={self.stride}")
        self.n_steps = (n_samples - self.chunk) // self.stride + 1
        self.n_samples = n_samples
        self.n_tokens = self.n_steps * self.C
        self.d_model = cfg.readout_dim

        self.backbone = EEGFM(cfg)
        if params.use_pretrained_weights:
            payload = torch.load(params.foundation_dir, map_location='cpu',
                                 weights_only=False)
            state_dict = payload.get('student', payload)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            assert not missing, (
                f"{len(missing)} pretrained tensors were NOT loaded, e.g. {missing[:5]}. "
                f"The model configuration does not match the checkpoint.")
            print(f"[neurduo] loaded {params.foundation_dir} "
                  f"(missing={len(missing)} unexpected={len(unexpected)}, "
                  f"global_step={payload.get('global_step')})", flush=True)
        del self.backbone.future_latent_predictor

        self.register_buffer('_pos', _coords(channels), persistent=False)
        self.register_buffer('_ref', torch.full((self.C,), REFERENCE_ID,
                                                dtype=torch.long), persistent=False)
        self.register_buffer('_missing', torch.zeros(self.C), persistent=False)

        self.classifier = STHead(self.n_tokens, self.d_model, params.num_of_classes,
                                 params.dropout, budget=params.readout_budget)

    def _metadata(self, batch_size, device):
        e = lambda t, *shape: t.unsqueeze(0).expand(batch_size, *shape)
        return EEGMetadataBatch(
            channel_positions=e(self._pos, self.C, 3),
            reference_ids=e(self._ref, self.C),
            missing_mask=e(self._missing, self.C),
        )

    def _grid(self, fused, b):
        assert fused.size(0) == b * self.C, (
            f"expected backbone rows = B*C = {b}*{self.C}, got {fused.size(0)}")
        return fused.reshape(b, self.C * fused.size(1), fused.size(2))

    def tokens(self, x):
        b, c, t = x.shape
        assert c == self.C and t == self.n_samples, \
            f"expects (B,{self.C},{self.n_samples}) @{NATIVE_FS}Hz, got {tuple(x.shape)}"
        chunks = x.unfold(-1, self.chunk, self.stride).permute(0, 2, 1, 3).contiguous()
        chunks = normalize_sequence(chunks)
        fused = self.backbone(chunks, self._metadata(b, x.device))
        return self._grid(fused, b)

    def tokens_continuous(self, x):
        b, l, c, t = x.shape
        assert self.stride == self.chunk, (
            f"tokens_continuous requires stride==chunk (no overlap), got "
            f"stride={self.stride} chunk={self.chunk}")
        assert c == self.C and t == self.n_samples, \
            f"expects (B,L,{self.C},{self.n_samples}), got {tuple(x.shape)}"
        flat = x.reshape(b * l, c, t)
        chunks = flat.unfold(-1, self.chunk, self.stride).permute(0, 2, 1, 3).contiguous()
        if self.cont_normalize == 'sequence':
            chunks = chunks.view(b, l * self.n_steps, c, self.chunk)
            chunks = normalize_sequence(chunks)
        elif self.cont_normalize == 'window':
            chunks = normalize_sequence(chunks)
            chunks = chunks.view(b, l * self.n_steps, c, self.chunk)
        else:
            raise ValueError(f"unknown cont_normalize {self.cont_normalize!r}")
        fused = self.backbone(chunks, self._metadata(b, x.device))
        d = fused.size(2)
        return (fused.view(b, self.C, l, self.n_steps, d)
                     .permute(0, 2, 1, 3, 4)
                     .reshape(b * l, self.C * self.n_steps, d))

    def forward(self, x):
        return self.classifier(self.tokens(x))
