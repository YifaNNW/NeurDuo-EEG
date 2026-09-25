"""Multi-kernel convolutional encoder that turns each channel's 0.5 s chunk into a token."""

import torch
from torch import nn

from eeg_fm.config import EEGFMConfig


class SignalPatchEncoder(nn.Module):
    def __init__(self, config: EEGFMConfig) -> None:
        super().__init__()
        hidden_dim = config.patch_hidden_dim
        kernel_sizes = config.patch_kernel_sizes

        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=ks, padding=ks // 2),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            for ks in kernel_sizes
        )
        self.merge = nn.Sequential(
            nn.Conv1d(hidden_dim * len(kernel_sizes), hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.temporal_gate = nn.Linear(hidden_dim, 1)
        self.proj = nn.Linear(hidden_dim, config.token_dim)

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_samples = chunk.shape
        x = chunk.reshape(batch_size * num_channels, 1, num_samples)

        branch_outputs = [branch(x) for branch in self.branches]
        x = self.merge(torch.cat(branch_outputs, dim=1))

        x = x.transpose(1, 2)
        logits = self.temporal_gate(x).squeeze(-1)
        weights = logits.softmax(dim=-1).unsqueeze(-1)
        pooled = torch.sum(x * weights, dim=1)

        encoded = self.proj(pooled)
        return encoded.reshape(batch_size, num_channels, -1)
