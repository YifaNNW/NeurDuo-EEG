"""Mamba stack with optional attention across channels, and the fast stream built from it."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from eeg_fm.config import EEGFMConfig
from eeg_fm.state import MambaBlockState, MambaStackState

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


def _require_mamba() -> None:
    if Mamba is None:
        raise ImportError(
            "Real Mamba backbone requested but `mamba_ssm` is not installed. "
            "Install PyTorch first, then install `mamba-ssm` and optionally `causal-conv1d`."
        )


class ChannelAttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, num_channels: int,
                missing: torch.Tensor | None = None) -> torch.Tensor:
        bc, s, d = x.shape
        c = num_channels
        b = bc // c
        h = x.view(b, c, s, d).permute(0, 2, 1, 3).reshape(b * s, c, d)
        h = self.norm(h)
        kpm = None
        if missing is not None:
            kpm = missing.bool().unsqueeze(1).expand(b, s, c).reshape(b * s, c)
            kpm = torch.where(kpm.all(dim=1, keepdim=True), torch.zeros_like(kpm), kpm)
        out, _ = self.attn(h, h, h, key_padding_mask=kpm, need_weights=False)
        return out.view(b, s, c, d).permute(0, 2, 1, 3).reshape(bc, s, d)


class _RealMambaStack(nn.Module):
    def __init__(
        self,
        model_dim: int,
        d_state: int,
        num_layers: int,
        expand_factor: int,
        conv_kernel_size: int,
        use_checkpoint: bool = False,
        channel_attn_every: int = 0,
        channel_attn_heads: int = 4,
    ) -> None:
        super().__init__()
        _require_mamba()
        self.model_dim = model_dim
        self.use_checkpoint = use_checkpoint

        self.norms = nn.ModuleList(nn.LayerNorm(model_dim) for _ in range(num_layers))
        self.blocks = nn.ModuleList(
            Mamba(
                d_model=model_dim,
                d_state=d_state,
                d_conv=conv_kernel_size,
                expand=expand_factor,
                layer_idx=layer_idx,
            )
            for layer_idx in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.channel_attn_every = int(channel_attn_every or 0)
        if self.channel_attn_every > 0:
            n_blocks = num_layers // self.channel_attn_every
            self.channel_attn = nn.ModuleList(
                [ChannelAttentionBlock(model_dim, channel_attn_heads) for _ in range(n_blocks)]
            )
        else:
            self.channel_attn = None

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MambaStackState:
        layer_states = []
        for block in self.blocks:
            conv_state, ssm_state = block.allocate_inference_cache(
                batch_size=batch_size,
                max_seqlen=1,
                dtype=dtype,
            )
            layer_states.append(
                MambaBlockState(
                    conv_state=conv_state.to(device=device),
                    ssm_state=ssm_state.to(device=device),
                )
            )
        last_output = torch.zeros(batch_size, self.model_dim, device=device, dtype=dtype)
        return MambaStackState(layer_states=tuple(layer_states), last_output=last_output)

    def forward_sequence(self, x: torch.Tensor, num_channels: int | None = None,
                         missing: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"forward_sequence expects [B, S, D], got {tuple(x.shape)}")
        hidden = x
        for i, (norm, block) in enumerate(zip(self.norms, self.blocks)):
            normed = norm(hidden)
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                block_out = checkpoint(block, normed, use_reentrant=False)
            else:
                block_out = block(normed)
            hidden = hidden + block_out
            if self.channel_attn is not None and (i + 1) % self.channel_attn_every == 0:
                if num_channels is None:
                    raise ValueError("Channel attention requires num_channels in forward_sequence.")
                j = (i + 1) // self.channel_attn_every - 1
                hidden = hidden + self.channel_attn[j](hidden, num_channels, missing)
        return self.final_norm(hidden)

    def forward_step(
        self,
        x_t: torch.Tensor,
        state: MambaStackState,
    ) -> tuple[torch.Tensor, MambaStackState]:
        if (
            self.training
            and torch.is_grad_enabled()
            and (x_t.requires_grad or any(p.requires_grad for p in self.parameters()))
        ):
            raise RuntimeError(
                "_RealMambaStack.forward_step was called in a training forward pass that needs "
                "gradients. Mamba.step has no backward, so train with forward_sequence and use "
                "forward_step only under torch.no_grad() for streaming inference."
            )
        if torch.is_autocast_enabled():
            try:
                dtype = torch.get_autocast_dtype("cuda")
            except (AttributeError, TypeError):
                dtype = torch.get_autocast_gpu_dtype()
        else:
            dtype = x_t.dtype
        hidden = x_t
        next_layer_states = []

        for norm, block, layer_state in zip(self.norms, self.blocks, state.layer_states):
            residual = hidden
            block_input = norm(hidden).unsqueeze(1)
            conv_state_in = layer_state.conv_state
            ssm_state_in = layer_state.ssm_state
            if conv_state_in.dtype != dtype:
                conv_state_in = conv_state_in.to(dtype=dtype)
            if ssm_state_in.dtype != dtype:
                ssm_state_in = ssm_state_in.to(dtype=dtype)
            block_output, conv_state, ssm_state = block.step(
                block_input,
                conv_state_in,
                ssm_state_in,
            )
            hidden = residual + block_output.squeeze(1)
            next_layer_states.append(MambaBlockState(conv_state=conv_state, ssm_state=ssm_state))

        final_output = self.final_norm(hidden)
        next_state = MambaStackState(
            layer_states=tuple(next_layer_states),
            last_output=final_output,
        )
        return final_output, next_state


class FastCausalMamba(nn.Module):
    def __init__(self, config: EEGFMConfig) -> None:
        super().__init__()
        self.input_proj = (
            nn.Linear(config.token_dim, config.fast_dim) if config.token_dim != config.fast_dim else nn.Identity()
        )
        self.stack = _RealMambaStack(
            model_dim=config.fast_dim,
            d_state=config.fast_d_state,
            num_layers=config.fast_num_layers,
            expand_factor=config.fast_expand_factor,
            conv_kernel_size=config.fast_conv_kernel_size,
            use_checkpoint=config.use_gradient_checkpointing,
            channel_attn_every=config.channel_attn_every,
            channel_attn_heads=config.channel_attn_heads,
        )

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MambaStackState:
        return self.stack.init_state(batch_size, device=device, dtype=dtype)

    def forward_sequence(self, chunk_tokens: torch.Tensor, num_channels: int | None = None,
                         missing: torch.Tensor | None = None) -> torch.Tensor:
        return self.stack.forward_sequence(
            self.input_proj(chunk_tokens), num_channels=num_channels, missing=missing)
