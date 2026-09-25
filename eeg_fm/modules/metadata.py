"""Embedding of electrode positions and reference IDs that is added to every channel token."""

import torch
from torch import nn

from eeg_fm.config import EEGFMConfig
from eeg_fm.state import EEGMetadataBatch


class MetadataEncoder(nn.Module):
    def __init__(self, config: EEGFMConfig) -> None:
        super().__init__()
        hidden_dim = config.metadata_hidden_dim
        self.token_dim = config.token_dim

        self.coord_proj = nn.Sequential(
            nn.Linear(config.coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.token_dim),
        )
        self.reference_embed = nn.Embedding(config.max_reference_types, self.token_dim)

    def forward(self, metadata: EEGMetadataBatch, num_channels: int, *, device: torch.device) -> torch.Tensor:
        batch_size = metadata.batch_size
        output = torch.zeros(batch_size, num_channels, self.token_dim, device=device)

        if metadata.channel_positions is not None:
            output = output + self.coord_proj(metadata.channel_positions.to(device))

        if metadata.reference_ids is not None:
            output = output + self.reference_embed(metadata.reference_ids.to(device))

        return output
