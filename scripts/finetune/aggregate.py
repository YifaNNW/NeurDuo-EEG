"""Mean and standard deviation of the test metrics over repeated fine-tuning runs."""

import argparse
import glob
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_dataset_specs():
    spec = importlib.util.spec_from_file_location(
        "dataset_specs", ROOT / "eeg_fm" / "finetune" / "dataset_specs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DATASET_SPECS


METRICS = {
    'multiclass': [('test_acc', 'Bal-Acc'), ('test_kappa', 'Cohen-κ'), ('test_f1', 'W-F1')],
    'binary': [('test_roc_auc', 'ROC-AUC'), ('test_pr_auc', 'AUC-PR'), ('test_acc', 'Bal-Acc')],
    'regression': [('test_corrcoef', 'Pearson r'), ('test_r2', 'R²'), ('test_rmse', 'RMSE')],
    'seq2seq': [('test_macro_f1', 'Macro-F1'), ('test_kappa', 'Cohen-κ'), ('test_accuracy', 'Acc')],
    'seizure': [('test_auc_pr', 'AUC-PR'), ('test_auroc', 'AUROC')],
}


def collect(res_dir, rep, tag):
    pat = re.compile(rf'^{rep}\d+(_seed\d+)?{re.escape(tag)}\.json$')
    rows = []
    for f in sorted(glob.glob(os.path.join(res_dir, f'{rep}*.json'))):
        if pat.match(os.path.basename(f)):
            with open(f) as fh:
                rows.append(json.load(fh))
    return rows


def ms(vals):
    a = np.array(vals, float)
    if not len(a):
        return "—"
    return f"{a[0]:.3f}" if len(a) == 1 else f"{a.mean():.3f} ± {a.std():.3f}"


def main():
    specs = load_dataset_specs()
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(specs), required=True)
    ap.add_argument('--output_dir', default='outputs/finetune')
    ap.add_argument('--tag', default='', help="variant suffix passed to run_finetune.py --tag")
    args = ap.parse_args()

    ds = specs[args.dataset]
    seizure = args.dataset.startswith('chbmit')
    metrics = METRICS['seizure' if seizure else ds['task']]
    rep = 'fold' if ds['task'] == 'seq2seq' or 'n_folds' in ds else 'seed'
    rows = collect(os.path.join(args.output_dir, args.dataset), rep, args.tag)
    if not rows:
        raise SystemExit(f"no results for {args.dataset} (tag '{args.tag}') under {args.output_dir}")

    print(f"{args.dataset} ({ds['task']}): {len(rows)} {rep}s")
    for key, name in metrics:
        print(f"  {name:<10} {ms([r[key] for r in rows if key in r])}")
    if seizure:
        false_alarms = sum(r['test_fp_per_hour_at_sens50'] * r['test_hours'] for r in rows)
        hours = sum(r['test_hours'] for r in rows)
        print(f"  {'FP/h':<10} {false_alarms / hours:.2f}  (pooled over folds, "
              f"{false_alarms:.0f} false alarms in {hours:.1f} h)")


if __name__ == '__main__':
    main()
