"""Next-code prediction head, used only in pretraining."""

import torch
from torch import nn

from eeg_fm.config import EEGFMConfig


class FutureTokenPredictor(nn.Module):
    def __init__(self, config: EEGFMConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(config.readout_dim),
            nn.Linear(config.readout_dim, config.predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(config.predictor_hidden_dim, config.codebook_size),
        )

    def forward(self, fused_states: torch.Tensor) -> torch.Tensor:
        return self.net(fused_states)
