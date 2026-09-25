"""Gated attention retrieval from each channel's memory tokens, for whole windows and single streaming steps."""

import torch
from torch import nn

from eeg_fm.config import EEGFMConfig
from eeg_fm.state import SequenceBuffer


class MemoryContext(nn.Module):
    def __init__(self, config: EEGFMConfig) -> None:
        super().__init__()
        self.query = nn.Linear(config.fast_dim, config.slow_dim)
        self.key = nn.Linear(config.slow_dim, config.slow_dim)
        self.value = nn.Linear(config.slow_dim, config.slow_dim)
        self.gate = nn.Sequential(
            nn.Linear(config.fast_dim + config.slow_dim, config.slow_dim),
            nn.Sigmoid(),
        )
        self.scale = config.slow_dim ** -0.5

    def forward_sequence(
        self,
        fast_states: torch.Tensor,
        memory_tokens: torch.Tensor,
        slot_index: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        b, s, _ = fast_states.shape
        k_total = memory_tokens.size(1)
        if k_total == 0:
            return torch.zeros(b, s, self.value.out_features,
                               device=fast_states.device, dtype=fast_states.dtype)

        memory = memory_tokens
        if memory.dtype != fast_states.dtype:
            memory = memory.to(dtype=fast_states.dtype)

        q = self.query(fast_states)
        k = self.key(memory)
        v = self.value(memory)
        scores = (q @ k.transpose(1, 2)) * self.scale

        slot = slot_index.to(device=fast_states.device).view(1, s, 1)
        arange = torch.arange(k_total, device=fast_states.device).view(1, 1, k_total)
        valid = (arange <= slot) & (arange > slot - capacity)
        has_memory = (slot_index.to(device=fast_states.device) >= 0).view(1, s, 1)

        scores = scores.masked_fill(~valid, float("-inf"))
        scores = torch.where(has_memory, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1)
        context = (weights @ v) * has_memory.to(dtype=v.dtype)

        gate = self.gate(torch.cat([fast_states, context], dim=-1))
        return gate * context

    def forward(self, fast_state: torch.Tensor, memory_bank: SequenceBuffer) -> torch.Tensor:
        if memory_bank.length == 0:
            return torch.zeros(
                fast_state.size(0),
                self.value.out_features,
                device=fast_state.device,
                dtype=fast_state.dtype,
            )

        memory = memory_bank.values[:, -memory_bank.length :]
        if memory.dtype != fast_state.dtype:
            memory = memory.to(dtype=fast_state.dtype)
        q = self.query(fast_state).unsqueeze(1)
        k = self.key(memory)
        v = self.value(memory)
        weights = torch.softmax((q @ k.transpose(1, 2)) * self.scale, dim=-1)
        context = (weights @ v).squeeze(1)
        gate = self.gate(torch.cat([fast_state, context], dim=-1))
        return gate * context
