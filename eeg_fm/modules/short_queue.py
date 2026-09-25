"""Fixed-size queue of recent fast states that the memory writer pools during streaming inference."""

import torch
from torch import nn

from eeg_fm.config import EEGFMConfig
from eeg_fm.state import SequenceBuffer


class ShortQueue(nn.Module):
    def __init__(self, config: EEGFMConfig) -> None:
        super().__init__()
        self.capacity = config.queue_size
        self.dim = config.fast_dim

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

    def get_window(self, state: SequenceBuffer, window_size: int) -> torch.Tensor:
        size = min(state.length, window_size)
        if size == 0:
            return state.values[:, :0]
        return state.values[:, -size:]
