"""Summarise the CHB-MIT perturbation test: one row per test x duration x lag.

Reads <in_dir>/fold*.npz written by perturb_chbmit.py. Each subject is a test subject in exactly one
fold, so intervals come from resampling subjects with replacement (all their windows together).

Columns, all on the target unit (9:30-10:00), "bb" = backbone path:
  dp_abs     mean |p_pert - p_orig| in percentage points
  dp_signed  mean (p_pert - p_orig) in percentage points (test B: does a planted seizure raise it?)
  dlo_abs    mean |log-odds change|
  share      dlo_abs / mean |log-odds change| when all history is removed (reset)
  flips      share of targets whose alarm decision at the fold's validation threshold changes
  repeat     mean |log-odds| difference of the same input run twice (floor)
  d_fast, d_slow_all, d_emb   mean relative change of the stream outputs / target embedding
"""

import argparse
import glob
import json
import os

import numpy as np

RUN = '/projects/u6xc/haiping/neurduo/runs/finetune/small/chbmit'


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load(in_dir):
    parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(in_dir, 'fold*.npz')))]
    if not parts:
        raise SystemExit(f'no fold*.npz in {in_dir}')
    keys = [k for k in parts[0] if k not in ('args', 'model')]
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def boot_ci(values, subjects, n_boot=2000, seed=0):
    subs = np.unique(subjects)
    groups = [values[subjects == s] for s in subs]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        stats.append(np.concatenate([groups[i] for i in pick]).mean())
    return np.percentile(stats, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', required=True)
    ap.add_argument('--path', choices=['bb', 'total'], default='bb')
    ap.add_argument('--stratum', choices=['all', 'other', 'preictal', 'seizure'], default='all')
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()

    d = load(args.in_dir)
    thr = {}
    for k in np.unique(d['fold']):
        r = json.load(open(f'{RUN}/fold{k}_seed42.json'))
        thr[int(k)] = r['test_threshold_valthr_sens50']
    if args.stratum != 'all':
        keep = {'seizure': d['target_label'] == 1,
                'preictal': (d['target_label'] == 0) & d['target_preictal'],
                'other': (d['target_label'] == 0) & ~d['target_preictal']}[args.stratum]
        d = {k: v[keep] for k, v in d.items()}

    lo_p = d['lo_bb'] if args.path == 'bb' else d['lo_total']
    p0, p1 = sigmoid(d['lo_orig']), sigmoid(lo_p)
    t = np.array([thr[int(k)] for k in d['fold']])
    rows = []
    for test in ('A', 'B'):
        for dur in sorted(np.unique(d['dur'][d['test'] == test])):
            for lag in sorted(np.unique(d['lag'][(d['test'] == test) & (d['dur'] == dur)])):
                m = (d['test'] == test) & (d['dur'] == dur) & (d['lag'] == lag)
                sub = d['subject'][m]
                dp = 100 * (p1[m] - p0[m])
                dlo = np.abs(lo_p[m] - d['lo_orig'][m])
                dlo_reset = np.abs(d['lo_reset'][m] - d['lo_orig'][m])
                lo_ci = boot_ci(dlo, sub)
                dp_ci = boot_ci(np.abs(dp), sub)
                rows.append(dict(
                    test=test, dur=int(dur), lag=int(lag), n=int(m.sum()), subjects=len(np.unique(sub)),
                    dp_abs=np.abs(dp).mean(), dp_abs_lo=dp_ci[0], dp_abs_hi=dp_ci[1],
                    dp_signed=dp.mean(), dlo_abs=dlo.mean(), dlo_lo=lo_ci[0], dlo_hi=lo_ci[1],
                    share=dlo.mean() / max(dlo_reset.mean(), 1e-12),
                    flips=((p1[m] >= t[m]) != (p0[m] >= t[m])).mean(),
                    repeat=np.abs(d['lo_repeat'][m] - d['lo_orig'][m]).mean(),
                    d_fast=np.nanmean(d['d_fast'][m]), d_slow_all=np.nanmean(d['d_slow_all'][m]),
                    d_emb=np.nanmean(d['d_emb'][m])))

    print(f"path={args.path} stratum={args.stratum} folds={sorted(thr)} "
          f"windows={len(np.unique(d['fold'] * 100000 + d['window']))}")
    hdr = (f"{'test':<4}{'dur':>4}{'lag':>5}{'n':>6}{'subj':>5}  {'|dp| pp [95% CI]':<22}{'dp pp':>7}"
           f"{'|dlo| [95% CI]':>24}{'share':>7}{'flips':>7}{'repeat':>9}{'d_fast':>8}{'d_slow':>8}{'d_emb':>8}")
    print(hdr)
    for r in rows:
        print(f"{r['test']:<4}{r['dur']:>4}{r['lag']:>5}{r['n']:>6}{r['subjects']:>5}  "
              f"{r['dp_abs']:6.3f} [{r['dp_abs_lo']:.3f},{r['dp_abs_hi']:.3f}]".ljust(28)
              + f"{r['dp_signed']:>7.3f}   {r['dlo_abs']:7.4f} [{r['dlo_lo']:.4f},{r['dlo_hi']:.4f}]"
              f"{r['share']:>7.3f}{r['flips']:>7.3f}{r['repeat']:>9.1e}{r['d_fast']:>8.4f}"
              f"{r['d_slow_all']:>8.4f}{r['d_emb']:>8.4f}")
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print('saved', args.csv)


if __name__ == '__main__':
    main()
