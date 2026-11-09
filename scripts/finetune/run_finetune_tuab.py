"""Fine-tuning on TUAB (abnormal EEG detection) without editing the original sources.

Registers a 'tuab' task in DATASET_SPECS and then runs run_finetune.main(). The arrays from
prep_tuab.py are already at 256 Hz, so they are memory-mapped instead of being loaded and
copied (the training split is ~58 GB). Use exactly like run_finetune.py, e.g.

    python scripts/finetune/run_finetune_tuab.py --dataset tuab --seed 0 \
        --data_dir /path/to/TUAB/segmented --checkpoint model_weights/neurduo_eeg_small.pt

Aggregate with:  python scripts/finetune/run_finetune_tuab.py --aggregate --output_dir <dir>
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from eeg_fm.finetune import dataset as ft_dataset  # noqa: E402
from eeg_fm.finetune.dataset_specs import DATASET_SPECS  # noqa: E402

TUAB_CHANNELS = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']

DATASET_SPECS['tuab'] = dict(
    base_fs=256,
    seg_sec=10,
    task='binary',
    n_outputs=1,
    label_dtype='float',
    channels=TUAB_CHANNELS,
    readout_budget=19200,
)


def _load_mmap(self, split):
    d = self.ds['seg_dir']
    X = np.load(os.path.join(d, f'{split}_X.npy'), mmap_mode='r')
    y = np.load(os.path.join(d, f'{split}_y.npy'))
    return self._make(X, y, split)


_orig_make = ft_dataset.LoadDataset._make


def _make_mmap(self, X, y, what):
    if self.ds['base_fs'] != ft_dataset.NATIVE_FS or X.dtype != np.float32:
        return _orig_make(self, X, y, what)
    assert X.shape[1] == self.ds['n_channels'], \
        f"{self.ds['name']} {what}: expected {self.ds['n_channels']} channels, got {X.shape[1]}"
    assert X.shape[2] == self.ds['base_fs'] * self.ds['seg_sec'], \
        f"{self.ds['name']} {what}: expected {self.ds['base_fs'] * self.ds['seg_sec']} samples, got {X.shape[2]}"
    return ft_dataset.SegmentSplit(X, y, self.ds['label_dtype'])


ft_dataset.LoadDataset._load = _load_mmap
ft_dataset.LoadDataset._make = _make_mmap


def aggregate(output_dir, tag=''):
    import scripts.finetune.aggregate as agg
    agg.load_dataset_specs = lambda: DATASET_SPECS
    sys.argv = [sys.argv[0], '--dataset', 'tuab', '--output_dir', output_dir, '--tag', tag]
    agg.main()


if __name__ == '__main__':
    if '--aggregate' in sys.argv:
        argv = sys.argv[1:]
        argv.remove('--aggregate')
        out = argv[argv.index('--output_dir') + 1] if '--output_dir' in argv else 'outputs/finetune'
        tag = argv[argv.index('--tag') + 1] if '--tag' in argv else ''
        aggregate(out, tag)
    else:
        import scripts.finetune.run_finetune as rf
        rf.main()
