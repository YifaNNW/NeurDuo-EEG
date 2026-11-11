"""Alarm scoring for CHB-MIT from saved per-unit predictions (rebuttal, ReVg Q4 and vZMz Q2).

numpy only, no model and no GPU. A unit is one 30 s segment with a label and a score.

For one model and one fold:
  1. the threshold is the median score of the seizure units of the validation subjects, then fixed;
  2. inside each test recording, consecutive units at or above the threshold form one alarm, and
     consecutive seizure units form one seizure; nothing is joined across two recordings;
  3. a seizure is detected if an alarm overlaps it, an alarm that overlaps no seizure is false;
  4. the delay of a detected seizure runs from its onset to the end of the first alarmed unit.

``Rule`` holds the event tolerances. ``STRICT`` has none. ``SZCORE`` uses the defaults advised by
SzCORE (Dan et al. 2024, arXiv 2402.13005) rounded to 30 s units: 30 s before and 60 s after a
seizure still count as overlap, events less than 90 s apart are merged, events longer than 5 min are
split. It follows the published description, not the reference implementation line by line.
"""

from dataclasses import dataclass

import numpy as np

UNIT_SEC = 30.0


@dataclass(frozen=True)
class Rule:
    pre_units: int = 0     # units before a seizure onset that still count as overlap
    post_units: int = 0    # units after a seizure end that still count as overlap
    merge_gap: int = 0     # events separated by fewer than this many units are merged (0: off)
    max_units: int = 0     # events longer than this are split (0: off)


STRICT = Rule()
SZCORE = Rule(pre_units=1, post_units=2, merge_gap=3, max_units=10)


def runs(mask):
    """[(start, stop)] of each run of True, stop exclusive."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    stops = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        stops = stops + [len(mask)]
    return list(zip(starts, stops))


def events(mask, rule=STRICT):
    """Runs of True, then merged and split as the rule says."""
    ev = runs(mask)
    if rule.merge_gap > 0 and ev:
        merged = [ev[0]]
        for a, b in ev[1:]:
            if a - merged[-1][1] < rule.merge_gap:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        ev = merged
    if rule.max_units > 0:
        ev = [(s, min(s + rule.max_units, b)) for a, b in ev for s in range(a, b, rule.max_units)]
    return ev


def threshold_at_sensitivity(y, score, target=0.5):
    """Threshold reached by ``target`` of the seizure units (same rule as ChbmitEvaluator)."""
    pos = np.asarray(score)[np.asarray(y) == 1]
    return float(np.quantile(pos, 1.0 - target)) if len(pos) else float('nan')


def score_sequence(y, pred, rule=STRICT):
    """Score one sequence of consecutive units. Returns counts and the delays in seconds."""
    y = np.asarray(y) == 1
    alarms = events(pred, rule)
    seizures = events(y, rule)
    false = [True] * len(alarms)
    delays, already_on = [], 0
    for a, b in seizures:
        lo, hi = a - rule.pre_units, b + rule.post_units
        hit = [i for i, (s, e) in enumerate(alarms) if s < hi and e > lo]
        for i in hit:
            false[i] = False
        if hit:
            s = alarms[hit[0]][0]
            already_on += s < a
            delays.append((max(s, lo) + 1 - a) * UNIT_SEC)
    return dict(n_alarms=len(alarms), n_false=int(sum(false)), n_seizures=len(seizures),
                n_detected=len(delays), n_already_on=int(already_on), delays=delays)


def _dense(idx, *arrays):
    """Place units at their position in the recording; a missing unit is not alarmed, not seizure."""
    idx = np.asarray(idx, np.int64) - int(np.min(idx))
    assert len(np.unique(idx)) == len(idx), 'duplicate unit in a recording'
    out = []
    for a in arrays:
        d = np.zeros(int(idx.max()) + 1, bool)
        d[idx] = a
        out.append(d)
    return out


def score_fold(pred, thr, rule=STRICT, per_recording=True):
    """Score one test fold at a fixed threshold.

    ``pred`` holds subject, recording, epoch_idx, y_true, y_score, one entry per unit.
    ``per_recording=False`` reproduces the paper: the fold is one sequence in the saved order, so an
    alarm can run from the last unit of a recording into the first unit of the next one.
    """
    y = np.asarray(pred['y_true']) == 1
    hit = np.asarray(pred['y_score']) >= thr
    sub, rec, idx = (np.asarray(pred[k]) for k in ('subject', 'recording', 'epoch_idx'))
    tot = dict(n_alarms=0, n_false=0, n_seizures=0, n_detected=0, n_already_on=0, delays=[])
    by_subject = {}
    if per_recording:
        keys = np.stack([sub, rec], 1)
        _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        groups = [np.flatnonzero(inv == g) for g in np.argsort(first)]
    else:
        groups = [np.arange(len(y))]
    for rows in groups:
        if per_recording:
            yy, pp = _dense(idx[rows], y[rows], hit[rows])
        else:
            yy, pp = y[rows], hit[rows]
        r = score_sequence(yy, pp, rule)
        for k, v in r.items():
            tot[k] += v
        if per_recording:
            s = by_subject.setdefault(int(sub[rows[0]]), dict(n_false=0, n_seizures=0, n_detected=0,
                                                             hours=0.0))
            s['n_false'] += r['n_false']
            s['n_seizures'] += r['n_seizures']
            s['n_detected'] += r['n_detected']
            s['hours'] += len(rows) * UNIT_SEC / 3600.0
    tot.update(hours=len(y) * UNIT_SEC / 3600.0, n_sz_units=int(y.sum()),
               n_sz_units_hit=int((hit & y).sum()), by_subject=by_subject)
    return tot


def pool(parts):
    """Add the counts of several folds."""
    out = dict(n_alarms=0, n_false=0, n_seizures=0, n_detected=0, n_already_on=0, delays=[],
               hours=0.0, n_sz_units=0, n_sz_units_hit=0, by_subject={})
    for p in parts:
        for k in out:
            if k == 'by_subject':
                out[k].update(p[k])
            else:
                out[k] += p[k]
    return out


def rates(t):
    """Counts -> the reported numbers."""
    d = np.asarray(t['delays'], float)
    nan = float('nan')
    return dict(
        fp_per_hour=t['n_false'] / t['hours'] if t['hours'] else nan,
        unit_sens=t['n_sz_units_hit'] / t['n_sz_units'] if t['n_sz_units'] else nan,
        event_sens=t['n_detected'] / t['n_seizures'] if t['n_seizures'] else nan,
        delay_median_s=float(np.median(d)) if len(d) else nan,
        delay_le60_share=float((d <= 60).mean()) if len(d) else nan,
    )
