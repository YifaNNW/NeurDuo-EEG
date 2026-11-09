"""Data side of the CHB-MIT perturbation test (rebuttal, reviewer ReVg Q1/W1). No GPU needed.

A 10 min window is 20 consecutive 30 s units of one recording. The target is the last unit
(570-600 s). A perturbed copy differs from the original only in one span that ends ``lag`` seconds
before the target starts; the span is overwritten with other real EEG of the same patient.

Normalisation is the fine-tuning one ("sequence": one span over the whole window), but its
statistics are computed once on the original window and reused for every copy. The perturbation
therefore cannot reach the target unit through the statistics; the target unit's input is
identical in every copy, so a change in its prediction can only come through the backbone state.
"""

from dataclasses import dataclass

import numpy as np
import torch

FS = 256
CHUNK = 128                        # samples per chunk (0.5 s)
CHUNKS_PER_SEC = FS // CHUNK       # 2
UNIT_SEC = 30
UNIT_CHUNKS = UNIT_SEC * CHUNKS_PER_SEC   # 60
CONTEXT_LEN = 20                   # units per window
TARGET = CONTEXT_LEN - 1           # index of the target unit
TARGET_START_SEC = TARGET * UNIT_SEC      # 570
TARGET_START_CHUNK = TARGET_START_SEC * CHUNKS_PER_SEC   # 1140
WRITE_EVERY_SEC = 2                # write_every_n_steps (4) x 0.5 s
PREICTAL_UNITS = 10                # a non-seizure target within 5 min before an onset is "pre-ictal"


def to_chunks(x):
    """[L, C, T] units -> [L * T/CHUNK, C, CHUNK] chunks, in the order tokens_continuous uses."""
    L, C, T = x.shape
    assert T % CHUNK == 0, f"{T} samples do not tile into {CHUNK}-sample chunks"
    n = T // CHUNK
    return x.reshape(L, C, n, CHUNK).permute(0, 2, 1, 3).reshape(L * n, C, CHUNK)


def seq_stats(chunks):
    """Statistics of eeg_fm.data.normalize.normalize_sequence for one span [S, C, K].

    Same operations in the same order, so apply_stats(chunks, *seq_stats(chunks)) is bit-identical
    to normalize_sequence(chunks).
    """
    x = chunks.unsqueeze(0)
    mean = x.mean(dim=(1, 3), keepdim=True)                               # [1, 1, C, 1]
    centered = x - mean
    std = centered.std(dim=(1, 3))                                        # [1, C]
    positive = std > 0
    n = positive.sum(dim=1)
    ordered, _ = torch.sort(torch.where(positive, std, torch.inf), dim=1)
    idx = torch.div(n - 1, 2, rounding_mode="floor").clamp_min(0)
    scale = ordered.gather(1, idx.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
    scale = torch.where(n > 0, scale, torch.ones_like(scale))             # [1]
    return mean, scale


def apply_stats(chunks, mean, scale):
    """Normalise any chunks [S, C, K] with statistics taken from another span."""
    return ((chunks.unsqueeze(0) - mean) / scale.view(-1, 1, 1, 1)).squeeze(0)


@dataclass(frozen=True)
class Span:
    test: str          # "A": background EEG, "B": planted seizure
    dur_sec: int
    lag_sec: int

    @property
    def start_sec(self):
        return TARGET_START_SEC - self.lag_sec - self.dur_sec

    @property
    def start_chunk(self):
        return self.start_sec * CHUNKS_PER_SEC

    @property
    def n_chunks(self):
        return self.dur_sec * CHUNKS_PER_SEC

    def check(self):
        assert self.start_sec >= 0, f"{self} starts before the window"
        assert self.lag_sec >= 0, f"{self} overlaps the target unit"
        assert self.start_sec % WRITE_EVERY_SEC == 0, f"{self} is not on a memory-write boundary"
        return self


def make_spans(lags, durations_a, dur_b):
    spans = [Span("A", d, lag).check() for d in durations_a for lag in lags]
    spans += [Span("B", dur_b, lag).check() for lag in lags]
    return spans


def perturb(base, span, material):
    """Copy of the normalised window ``base`` [S, C, K] with ``span`` overwritten by ``material``."""
    assert material.shape == (span.n_chunks,) + tuple(base.shape[1:]), \
        f"material {tuple(material.shape)} does not fit {span}"
    out = base.clone()
    out[span.start_chunk:span.start_chunk + span.n_chunks] = material
    return out


def check_only_span_differs(base, pert, span):
    """The copies must agree element by element outside the span, the target unit included."""
    a, b = span.start_chunk, span.start_chunk + span.n_chunks
    assert torch.equal(base[:a], pert[:a]) and torch.equal(base[b:], pert[b:]), \
        f"{span}: inputs differ outside the replaced span"
    assert torch.equal(base[TARGET_START_CHUNK:], pert[TARGET_START_CHUNK:]), \
        f"{span}: target unit input changed"


class WindowIndex:
    """Which windows qualify for each test, and where their replacement EEG comes from.

    All arrays are the per-unit arrays of the segmented data (y, subject, recording, epoch_idx);
    rows of one recording are contiguous and in time order.
    """

    def __init__(self, y, subject, recording, epoch_idx, seed=0):
        self.y, self.subject, self.recording, self.epoch_idx = y, subject, recording, epoch_idx
        self.seed = seed
        n = len(y)
        prev_same = np.zeros(n, bool)
        prev_same[1:] = (recording[1:] == recording[:-1]) & (epoch_idx[1:] == epoch_idx[:-1] + 1)
        next_same = np.zeros(n, bool)
        next_same[:-1] = prev_same[1:]
        # "Core" seizure units: seizure, with seizure units right before and after in the same
        # recording, so the whole 30 s lies inside one seizure.
        ys = y.astype(bool)
        core = ys.copy()
        core[1:] &= prev_same[1:] & ys[:-1]
        core[:-1] &= next_same[:-1] & ys[1:]
        core[0] = core[-1] = False
        self.core_rows = np.flatnonzero(core)
        self.prev_same = prev_same

    def previous_row(self, rows):
        """Row of the unit right before the window in the same recording, or None."""
        first = rows[0]
        return first - 1 if first > 0 and self.prev_same[first] else None

    def target_info(self, rows):
        t = rows[TARGET]
        label = int(self.y[t])
        pre = False
        if label == 0:
            r = t
            for _ in range(PREICTAL_UNITS):
                if r + 1 >= len(self.y) or not self.prev_same[r + 1]:
                    break
                r += 1
                if self.y[r]:
                    pre = True
                    break
        return label, pre

    def eligible_a(self, rows):
        """Test A needs a full window and a non-seizure unit right before it."""
        if len(rows) != CONTEXT_LEN:
            return None
        prev = self.previous_row(rows)
        if prev is None or self.y[prev]:
            return None
        return prev

    def donor_b(self, rows, window_key):
        """Test B needs a seizure-free window; the donor is a core seizure unit of the same subject
        from another recording, drawn with a fixed seed. None if the window does not qualify."""
        if len(rows) != CONTEXT_LEN or self.y[rows].any():
            return None
        sub, rec = self.subject[rows[0]], self.recording[rows[0]]
        cand = self.core_rows[(self.subject[self.core_rows] == sub)
                              & (self.recording[self.core_rows] != rec)]
        if len(cand) == 0:
            return None
        rng = np.random.default_rng([self.seed, *window_key])
        return int(rng.choice(cand))
