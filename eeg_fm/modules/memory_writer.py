"""Memory writer that pools recent fast states into one memory token with learned attention weights."""

import torch
from torch import nn

from eeg_fm.config import EEGFMConfig


class MemoryWriter(nn.Module):
    def __init__(self, config: EEGFMConfig) -> None:
        super().__init__()
        self.score = nn.Linear(config.fast_dim, 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(config.fast_dim),
            nn.Linear(config.fast_dim, config.slow_dim),
        )

    def forward(self, recent_fast_states: torch.Tensor) -> torch.Tensor:
        if recent_fast_states.size(1) == 0:
            raise ValueError("MemoryWriter requires at least one fast state.")
        logits = self.score(recent_fast_states).squeeze(-1)
        weights = logits.softmax(dim=-1).unsqueeze(-1)
        summary = torch.sum(recent_fast_states * weights, dim=1)
        return self.proj(summary)
