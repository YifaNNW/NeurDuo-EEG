"""Window normalization: per-channel centering and one scale shared by all channels."""

from __future__ import annotations

import torch


def normalize_window(chunks: torch.Tensor) -> torch.Tensor:
    if chunks.numel() == 0:
        return chunks
    if chunks.dim() != 3:
        raise ValueError(f"normalize_window expects [S, C, L], got {tuple(chunks.shape)}")

    centered = chunks - chunks.mean(dim=(0, 2), keepdim=True)

    per_channel_std = centered.std(dim=(0, 2))
    positive = per_channel_std[per_channel_std > 0]
    if positive.numel() == 0:
        return centered
    scale = positive.median().clamp_min(1e-12)
    return centered / scale
