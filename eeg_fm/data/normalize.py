"""Amplitude normalization of chunked EEG.

One rule, applied to a span of consecutive chunks ``[S, C, L]``:

    centered = x - mean_c                     # per channel, over all chunks and samples in the span
    scale    = median_c( std_c(centered) )    # one scale for all channels, over channels with std > 0
    out      = centered / scale

The channel mean removes per-electrode offsets; the shared scale keeps the relative amplitude
between electrodes; the median makes the scale insensitive to a dead or saturated electrode.
A span with no channel of positive std is only centered.

What the span is, is set by the mode, named as in the fine-tuning ``cont_normalize`` option:

    "sequence"  the whole sequence of S chunks is one span           -> normalize_sequence
    "window"    consecutive spans of ``window_steps`` chunks, each   -> normalize_window
                normalized on its own; if S is not a multiple, the
                last shorter span is normalized on its own
    "none"      no normalization
"""

from __future__ import annotations

import torch

NORMALIZE_MODES = ("sequence", "window", "none")


def normalize_sequence(x: torch.Tensor) -> torch.Tensor:
    """Normalize a whole sequence ``[S, C, L]``, or each sequence of a batch ``[N, S, C, L]``."""
    if x.dim() not in (3, 4):
        raise ValueError(f"normalize_sequence expects [S, C, L] or [N, S, C, L], got {tuple(x.shape)}")
    if x.numel() == 0:
        return x
    if x.dim() == 3:
        return normalize_sequence(x.unsqueeze(0)).squeeze(0)

    centered = x - x.mean(dim=(1, 3), keepdim=True)
    std = centered.std(dim=(1, 3))                                        # [N, C]
    positive = std > 0
    n = positive.sum(dim=1)                                               # [N]
    ordered, _ = torch.sort(torch.where(positive, std, torch.inf), dim=1)
    # Lower median over the positive channels, the value torch.median returns.
    idx = torch.div(n - 1, 2, rounding_mode="floor").clamp_min(0)
    scale = ordered.gather(1, idx.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
    scale = torch.where(n > 0, scale, torch.ones_like(scale))
    return centered / scale.view(-1, 1, 1, 1)


def normalize_window(x: torch.Tensor, window_steps: int) -> torch.Tensor:
    """Normalize a sequence ``[S, C, L]`` in independent windows of ``window_steps`` chunks."""
    if x.dim() != 3:
        raise ValueError(f"normalize_window expects [S, C, L], got {tuple(x.shape)}")
    window_steps = int(window_steps)
    if window_steps < 1:
        raise ValueError(f"window_steps must be >= 1, got {window_steps}")
    if x.numel() == 0:
        return x

    S, C, L = x.shape
    w = min(window_steps, S)
    n_full = S // w
    out = normalize_sequence(x[: n_full * w].reshape(n_full, w, C, L)).reshape(n_full * w, C, L)
    if n_full * w < S:
        out = torch.cat([out, normalize_sequence(x[n_full * w :])], dim=0)
    return out


def normalize(x: torch.Tensor, mode: str = "sequence", window_steps: int = 8) -> torch.Tensor:
    """Normalize a sequence ``[S, C, L]`` by ``mode``: "sequence", "window" or "none"."""
    if mode == "sequence":
        return normalize_sequence(x)
    if mode == "window":
        return normalize_window(x, window_steps)
    if mode == "none":
        return x
    raise ValueError(f"normalize mode must be one of {NORMALIZE_MODES}, got {mode!r}")

