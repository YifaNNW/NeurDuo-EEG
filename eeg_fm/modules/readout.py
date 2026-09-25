"""Concatenation of the fast and slow states into the backbone output."""

import torch
from torch import nn

from eeg_fm.config import EEGFMConfig


class SplicedReadout(nn.Module):
    def __init__(self, config: EEGFMConfig) -> None:
        super().__init__()

    def forward(
        self,
        fast_state: torch.Tensor,
        slow_state: torch.Tensor,
    ) -> torch.Tensor:
        if slow_state.dtype != fast_state.dtype:
            slow_state = slow_state.to(dtype=fast_state.dtype)
        return torch.cat([fast_state, slow_state], dim=-1)
