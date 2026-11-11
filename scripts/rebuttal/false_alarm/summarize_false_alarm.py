"""CHB-MIT false alarms with the threshold fixed on validation subjects (rebuttal, ReVg Q4, vZMz Q2).

Reads, for every fold of a fine-tuning run written by scripts/finetune/run_finetune.py,
  <run_dir>/predictions/fold{k}_seed{s}.npz       test units
  <run_dir>/predictions/fold{k}_seed{s}_val.npz   validation units
  <run_dir>/fold{k}_seed{s}.json                  the run's own numbers, used as a check
and scores them offline with alarm_lib. Nothing is retrained and no inference is run.

Rows per run, each per fold and pooled over folds (counts and hours are added, then divided):
  val / recording / strict   threshold from validation, alarms inside a recording   <- the answer
  val / recording / szcore   the same with the SzCORE tolerances
  val / sequence  / strict   the same with the paper's merging across recordings
  test / sequence / strict   the paper's Table 3 rule: threshold from the test fold itself
  test / recording / strict  the paper's threshold, alarms inside a recording

Columns:
  fp_per_hour   false alarms / recording hours
  unit_sens     seizure units at or above the threshold / seizure units
  event_sens    seizures overlapped by an alarm / seizures
  delay         seizure onset to the end of the first alarmed unit, median in seconds, and the
                share of detected seizures with a delay of at most 60 s

Writes to results/rebuttal/false_alarm/ (git-ignored):
  summary.csv    one row per run and rule, pooled over folds, with AUC-PR mean and s.d. over folds
  folds.csv      the same per fold, with the threshold
  subjects.csv   per test subject, for val / recording / strict

Usage:
  python scripts/rebuttal/false_alarm/summarize_false_alarm.py
  python scripts/rebuttal/false_alarm/summarize_false_alarm.py \
      --run small=/projects/u6xc/haiping/neurduo/runs/finetune/small/chbmit
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import alarm_lib as A

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / 'results' / 'rebuttal' / 'false_alarm'
RUNS = '/projects/u6xc/haiping/neurduo/runs/finetune'
DEFAULT_RUNS = [
    f'small={RUNS}/small/chbmit',
    f'small_pure_mamba_scratch={RUNS}/small_pure_mamba_scratch/chbmit',
    f'small_pure_mamba_scratch_2ch={RUNS}/small_pure_mamba_scratch/chbmit_2ch',
]
RULES = [   # name, threshold from, alarms inside a recording, tolerances
    ('val/recording/strict', 'val', True, A.STRICT),
    ('val/recording/szcore', 'val', True, A.SZCORE),
    ('val/sequence/strict', 'val', False, A.STRICT),
    ('test/sequence/strict', 'test', False, A.STRICT),
    ('test/recording/strict', 'test', True, A.STRICT),
]
COLS = ['run', 'rule', 'fold', 'threshold', 'hours', 'n_false', 'fp_per_hour', 'n_sz_units',
        'unit_sens', 'n_seizures', 'n_detected', 'event_sens', 'delay_median_s',
        'delay_le60_share', 'n_already_on']


def load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def check(name, mine, theirs, tol=1e-9):
    if theirs is None:
        return
    if not abs(mine - theirs) <= tol * max(1.0, abs(theirs)):
        raise SystemExit(f'check failed: {name}: recomputed {mine!r}, run reported {theirs!r}')


def score_run(name, run_dir, folds, seed, target):
    rows, pooled, subjects, aucpr = [], {r[0]: [] for r in RULES}, [], []
    k50 = int(target * 100)
    for k in folds:
        stem = f'fold{k}_seed{seed}'
        test = load(f'{run_dir}/predictions/{stem}.npz')
        val = load(f'{run_dir}/predictions/{stem}_val.npz')
        js = json.load(open(f'{run_dir}/{stem}.json'))
        aucpr.append(js['test_auc_pr'])
        thr = dict(val=A.threshold_at_sensitivity(val['y_true'], val['y_score'], target),
                   test=A.threshold_at_sensitivity(test['y_true'], test['y_score'], target))
        check(f'{stem} validation threshold', thr['val'], js.get(f'test_threshold_valthr_sens{k50}'))
        check(f'{stem} test threshold', thr['test'], js.get(f'test_threshold_at_sens{k50}'))
        for rule, src, per_rec, tol in RULES:
            t = A.score_fold(test, thr[src], tol, per_recording=per_rec)
            if rule == 'val/sequence/strict':     # the run's own evaluator computes exactly this
                check(f'{stem} FP/h', t['n_false'] / t['hours'],
                      js.get(f'test_fp_per_hour_valthr_sens{k50}'))
            if rule == 'test/sequence/strict':
                check(f'{stem} FP/h', t['n_false'] / t['hours'],
                      js.get(f'test_fp_per_hour_at_sens{k50}'))
            pooled[rule].append(t)
            rows.append(dict(run=name, rule=rule, fold=k, threshold=thr[src], **row(t)))
            if rule == 'val/recording/strict':
                for s, v in sorted(t['by_subject'].items()):
                    subjects.append(dict(run=name, fold=k, subject=s, hours=v['hours'],
                                         n_false=v['n_false'],
                                         fp_per_hour=v['n_false'] / v['hours'],
                                         n_seizures=v['n_seizures'], n_detected=v['n_detected']))
    for rule, parts in pooled.items():
        rows.append(dict(run=name, rule=rule, fold='all', threshold=float('nan'),
                         **row(A.pool(parts))))
    return rows, subjects, (float(np.mean(aucpr)), float(np.std(aucpr)))


def row(t):
    r = A.rates(t)
    return dict(hours=t['hours'], n_false=t['n_false'], n_sz_units=t['n_sz_units'],
                n_seizures=t['n_seizures'], n_detected=t['n_detected'],
                n_already_on=t['n_already_on'], **r)


def write_csv(path, rows, cols):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', action='append', default=None, metavar='NAME=DIR',
                    help='fine-tuning run folder; repeat for several (default: the three CHB-MIT '
                         'runs that have validation predictions)')
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--target', type=float, default=0.5,
                    help='unit sensitivity the threshold reaches on the set it is chosen on')
    ap.add_argument('--out_dir', default=str(OUT_DIR))
    args = ap.parse_args()

    rows, subjects, aucpr = [], [], {}
    for spec in args.run or DEFAULT_RUNS:
        name, run_dir = spec.split('=', 1)
        r, s, aucpr[name] = score_run(name, run_dir, args.folds, args.seed, args.target)
        rows += r
        subjects += s
    print('checks against the runs\' own json: ok\n')

    print('| run | AUC-PR | threshold | alarms | FP/h | unit sens. | event sens. | median delay '
          '| delay <= 60 s |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        if r['fold'] != 'all':
            continue
        src, grp, tol = r['rule'].split('/')
        m, sd = aucpr[r['run']]
        print(f"| {r['run']} | {m:.3f} ± {sd:.3f} | {src} | {grp}, {tol} | {r['fp_per_hour']:.2f} "
              f"| {r['unit_sens']:.2f} | {r['n_detected']}/{r['n_seizures']} = {r['event_sens']:.2f} "
              f"| {r['delay_median_s']:.0f} s | {r['delay_le60_share']:.2f} |")

    os.makedirs(args.out_dir, exist_ok=True)
    summary = [dict(r, auc_pr_mean=aucpr[r['run']][0], auc_pr_sd=aucpr[r['run']][1])
               for r in rows if r['fold'] == 'all']
    drop = ('fold', 'threshold')
    write_csv(f'{args.out_dir}/summary.csv', [{k: v for k, v in r.items() if k not in drop}
                                              for r in summary],
              ['run', 'rule', 'auc_pr_mean', 'auc_pr_sd'] + [c for c in COLS[2:] if c not in drop])
    write_csv(f'{args.out_dir}/folds.csv', [r for r in rows if r['fold'] != 'all'], COLS)
    write_csv(f'{args.out_dir}/subjects.csv', subjects,
              ['run', 'fold', 'subject', 'hours', 'n_false', 'fp_per_hour', 'n_seizures',
               'n_detected'])
    print(f'\nwrote summary.csv, folds.csv and subjects.csv to {args.out_dir}')


if __name__ == '__main__':
    main()
