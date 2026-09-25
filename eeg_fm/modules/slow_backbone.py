"""Slow stream: a Mamba stack over each channel's memory tokens."""

import torch
from torch import nn

from eeg_fm.config import EEGFMConfig
from eeg_fm.modules.fast_backbone import _RealMambaStack
from eeg_fm.state import MambaStackState


class SlowCausalSSM(nn.Module):
    def __init__(self, config: EEGFMConfig) -> None:
        super().__init__()
        self.stack = _RealMambaStack(
            model_dim=config.slow_dim,
            d_state=config.slow_d_state,
            num_layers=config.slow_num_layers,
            expand_factor=config.slow_expand_factor,
            conv_kernel_size=config.slow_conv_kernel_size,
            use_checkpoint=config.use_gradient_checkpointing,
        )

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MambaStackState:
        return self.stack.init_state(batch_size, device=device, dtype=dtype)

    def forward_sequence(self, memory_tokens: torch.Tensor) -> torch.Tensor:
        return self.stack.forward_sequence(memory_tokens)

    def forward_step(
        self,
        memory_token: torch.Tensor,
        state: MambaStackState,
    ) -> tuple[torch.Tensor, MambaStackState]:
        return self.stack.forward_step(memory_token, state)
