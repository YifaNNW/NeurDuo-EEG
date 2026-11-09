"""Sub-window normalization for pretraining.

The original ``normalize.normalize_window`` normalizes one whole pretraining
sequence ``[S, C, L]`` (S = 64 chunks = 32 s) with a single set of statistics:
per-channel mean over all S*L samples, and one scale shared by all channels
(the median of the per-channel std). Downstream fine-tuning with
``cont_normalize='window'`` instead normalizes every labelling unit on its own,
so a continuous downstream stream contains statistic resets at unit boundaries
that pretraining never shows the model.

``normalize_window_con`` splits the S axis into consecutive blocks of
``window_steps`` chunks and applies exactly the same rule to each block
independently:

    for each block b of chunks [b*w : (b+1)*w]:
        centered = block - mean_over(chunks in block, samples)   # per channel
        scale    = median_c( std_c(centered) ), over channels with std > 0
        out      = centered / scale

``window_steps >= S`` reproduces ``normalize_window`` bit-for-bit in behaviour.
If S is not a multiple of ``window_steps`` the last, shorter block is
normalized on its own. Blocks with no channel of positive std are only
centered, as in the original.

``ContinuousEEGWindowDatasetCon`` is a drop-in replacement for
``ContinuousEEGWindowDataset`` that uses this function, so pretraining can
switch to sub-window normalization without editing the original files.
"""

from __future__ import annotations

import torch

from eeg_fm.data.manifest_dataset import ContinuousEEGWindowDataset


def _normalize_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """Normalize each block of ``[N, w, C, L]`` independently."""
    centered = blocks - blocks.mean(dim=(1, 3), keepdim=True)
    std = centered.std(dim=(1, 3))                                # [N, C]
    positive = std > 0
    n = positive.sum(dim=1)                                       # [N]
    ordered, _ = torch.sort(torch.where(positive, std, torch.inf), dim=1)
    # Lower median over the positive channels, same as torch.median on the
    # positive subset that normalize_window uses.
    idx = torch.div(n - 1, 2, rounding_mode="floor").clamp_min(0)
    scale = ordered.gather(1, idx.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
    scale = torch.where(n > 0, scale, torch.ones_like(scale))
    return centered / scale.view(-1, 1, 1, 1)


def normalize_window_con(chunks: torch.Tensor, window_steps: int) -> torch.Tensor:
    """Normalize ``[S, C, L]`` chunks in independent blocks of ``window_steps`` chunks."""
    if chunks.numel() == 0:
        return chunks
    if chunks.dim() != 3:
        raise ValueError(f"normalize_window_con expects [S, C, L], got {tuple(chunks.shape)}")
    window_steps = int(window_steps)
    if window_steps < 1:
        raise ValueError(f"window_steps must be >= 1, got {window_steps}")

    S, C, L = chunks.shape
    w = min(window_steps, S)
    n_full = S // w
    head = chunks[: n_full * w].reshape(n_full, w, C, L)
    out = [_normalize_blocks(head).reshape(n_full * w, C, L)]
    if n_full * w < S:
        tail = chunks[n_full * w :].unsqueeze(0)
        out.append(_normalize_blocks(tail).squeeze(0))
    return torch.cat(out, dim=0) if len(out) > 1 else out[0]


class ContinuousEEGWindowDatasetCon(ContinuousEEGWindowDataset):
    """``ContinuousEEGWindowDataset`` with sub-window normalization of every sequence."""

    def __init__(self, manifest_path, *, norm_window_steps: int, **kwargs) -> None:
        if int(norm_window_steps) < 1:
            raise ValueError(f"norm_window_steps must be >= 1, got {norm_window_steps}")
        self.norm_window_steps = int(norm_window_steps)
        kwargs["normalize_per_window"] = False  # the parent must not normalize again
        super().__init__(manifest_path, **kwargs)

    def _slice_window_into_chunks(self, signal_window):
        chunks = super()._slice_window_into_chunks(signal_window)
        return normalize_window_con(chunks, self.norm_window_steps)
