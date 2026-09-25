"""Layer normalization of the channel tokens before the fast stream."""

from torch import nn

from eeg_fm.config import EEGFMConfig


class LightweightSpatialMixer(nn.Module):
    def __init__(self, config: EEGFMConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.token_dim)
