"""Fixed-size bank of the latest memory tokens, used in streaming inference."""

import torch
from torch import nn

from eeg_fm.config import EEGFMConfig
from eeg_fm.state import SequenceBuffer


class MemoryBank(nn.Module):
    def __init__(self, config: EEGFMConfig) -> None:
        super().__init__()
        self.capacity = config.memory_bank_size
        self.dim = config.slow_dim

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> SequenceBuffer:
        return SequenceBuffer(
            values=torch.zeros(batch_size, self.capacity, self.dim, device=device, dtype=dtype),
            length=0,
        )

    def append(self, state: SequenceBuffer, value: torch.Tensor) -> SequenceBuffer:
        values = torch.roll(state.values, shifts=-1, dims=1)
        values[:, -1] = value
        length = min(state.length + 1, self.capacity)
        return SequenceBuffer(values=values, length=length)
