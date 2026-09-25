"""Loading of downstream datasets, resampling to 256 Hz and splitting into window or sequence samples."""

import json
import os

import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import DataLoader, Dataset

NATIVE_FS = 256


def apply_transform(X, base_fs):
    if base_fs != NATIVE_FS:
        X = resample(X, int(round(X.shape[-1] * NATIVE_FS / base_fs)), axis=-1)
    return X.astype(np.float32)


class SegmentSplit(Dataset):
    def __init__(self, X, y, label_dtype):
        self.X = X
        self.y = y
        self.label_dtype = label_dtype

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def collate(self, batch):
        xs = np.stack([b[0] for b in batch])
        ys = np.array([b[1] for b in batch])
        y = torch.from_numpy(ys)
        y = y.long() if self.label_dtype == 'long' else y.float()
        return torch.from_numpy(xs).float(), y


class SequenceSplit(Dataset):
    IGNORE = -100

    def __init__(self, X, rows_per_window, y, ds, meta):
        self.X = X
        self.windows = rows_per_window
        self.y = y
        self.ds = ds
        self.L = ds['context_len']
        self.meta = meta

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        rows = self.windows[i]
        n = len(rows)
        raw = np.asarray(self.X[rows[0]:rows[-1] + 1][rows - rows[0]], np.float32)
        subset = self.ds.get('channel_subset')
        if subset is not None:
            raw = raw[:, list(subset)]
        x = apply_transform(raw, self.ds['base_fs'])
        if n < self.L:
            pad = np.zeros((self.L - n,) + x.shape[1:], x.dtype)
            x = np.concatenate([x, pad], 0)
        lab = np.full(self.L, self.IGNORE, np.int64)
        lab[:n] = self.y[rows]
        mask = np.zeros(self.L, bool)
        mask[:n] = True
        return x, lab, mask

    def collate(self, batch):
        x = torch.from_numpy(np.stack([b[0] for b in batch])).float()
        y = torch.from_numpy(np.stack([b[1] for b in batch])).long()
        m = torch.from_numpy(np.stack([b[2] for b in batch]))
        return x, y, m


def cut_windows(rows, L):
    return [rows[s:s + L] for s in range(0, len(rows), L)]


def fold_groups(subject, n_folds):
    subs = np.unique(subject)
    assert len(subs) % n_folds == 0, \
        f"{len(subs)} subjects do not divide into {n_folds} folds"
    return np.split(subs, n_folds)


def check_channels(ds):
    path = os.path.join(ds['seg_dir'], 'channels.txt')
    if not os.path.exists(path):
        return
    with open(path) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if ds.get('channel_subset') is not None:
        names = [names[i] for i in ds['channel_subset']]
    assert [n.upper() for n in names] == [c.upper() for c in ds['channels']], \
        f"{ds['name']}: channels.txt {names} does not match the spec {ds['channels']}"


class LoadDataset(object):
    def __init__(self, dataset, batch_size=64, num_workers=8, fold=0):
        self.ds = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = fold
        check_channels(self.ds)

    def _make(self, X, y, what):
        assert X.shape[1] == self.ds['n_channels'], \
            f"{self.ds['name']} {what}: expected {self.ds['n_channels']} channels, got {X.shape[1]}"
        assert X.shape[2] == self.ds['base_fs'] * self.ds['seg_sec'], \
            f"{self.ds['name']} {what}: expected {self.ds['base_fs'] * self.ds['seg_sec']} samples, got {X.shape[2]}"
        X = apply_transform(X, self.ds['base_fs'])
        return SegmentSplit(X, y, self.ds['label_dtype'])

    def _load(self, split):
        X = np.load(os.path.join(self.ds['seg_dir'], f'{split}_X.npy'))
        y = np.load(os.path.join(self.ds['seg_dir'], f'{split}_y.npy'))
        return self._make(X, y, split)

    def _load_folds(self):
        d = self.ds['seg_dir']
        X = np.load(f'{d}/X.npy')
        y = np.load(f'{d}/y.npy')
        subject = np.load(f'{d}/subject.npy')
        groups = fold_groups(subject, self.ds['n_folds'])

        dropped = groups[self.fold]
        train_subs = np.setdiff1d(np.unique(subject), dropped)
        pool = np.flatnonzero(np.isin(subject, train_subs))

        rng = np.random.RandomState(1000 + self.fold)
        val_idx = []
        for s in train_subs:
            for c in (0.0, 1.0):
                cell = pool[(subject[pool] == s) & (y[pool] == c)]
                k = int(round(self.ds['val_frac'] * len(cell)))
                val_idx.append(rng.choice(cell, size=k, replace=False))
        val_idx = np.concatenate(val_idx)
        is_val = np.zeros(len(y), bool)
        is_val[val_idx] = True

        Xt = np.load(f'{d}/test_X.npy')
        yt = np.load(f'{d}/test_y.npy')
        print(f"[fold {self.fold}/{self.ds['n_folds']}] dropped subjects "
              f"{dropped.tolist()} | train subjects {train_subs.tolist()} "
              f"({self.ds['val_frac']:.0%} of their trials held out for epoch "
              f"selection) | test = the fixed test subjects",
              flush=True)
        train_m = np.isin(subject, train_subs) & ~is_val
        return {'train': self._make(X[train_m], y[train_m], 'train'),
                'val': self._make(X[is_val], y[is_val], 'val'),
                'test': self._make(Xt, yt, 'test')}

    def _load_sequences(self):
        d = self.ds['seg_dir']
        X = np.load(f'{d}/X.npy', mmap_mode='r')
        y = np.load(f'{d}/y.npy')
        subject = np.load(f'{d}/subject.npy')
        recording = np.load(f'{d}/recording.npy')
        epoch_idx = np.load(f'{d}/epoch_idx.npy')
        with open(f'{d}/split_subjects.json') as f:
            splits = json.load(f)
        assert splits.get('n_folds', self.ds['n_folds']) == self.ds['n_folds'], \
            (f"split_subjects.json was written with n_folds={splits['n_folds']}, "
             f"but the spec asks for n_folds={self.ds['n_folds']}")
        fold = splits['folds'][str(self.fold)]

        L = self.ds['context_len']
        sets = {}
        for split in ('train', 'val', 'test'):
            subs = set(fold[split])
            rows_by_rec, meta, windows = {}, [], []
            sel = np.flatnonzero(np.isin(subject, list(subs)))
            for r in np.unique(recording[sel]):
                rows_by_rec[int(r)] = sel[recording[sel] == r]
            for r in sorted(rows_by_rec):
                for w, rows in enumerate(cut_windows(rows_by_rec[r], L)):
                    windows.append(rows)
                    meta.append(dict(recording=int(r), subject=int(subject[rows[0]]),
                                     window=w, n_valid=len(rows),
                                     epoch_idx=epoch_idx[rows].tolist()))
            sets[split] = SequenceSplit(X, windows, y, self.ds, meta)
        print(f"[fold {self.fold}/{self.ds['n_folds']}] subjects "
              f"train {len(fold['train'])} / val {len(fold['val'])} / test {len(fold['test'])}"
              f" | windows " + str({k: len(v) for k, v in sets.items()}), flush=True)
        return sets

    def get_data_loader(self):
        if self.ds['task'] == 'seq2seq':
            sets = self._load_sequences()
        elif 'n_folds' in self.ds:
            sets = self._load_folds()
        else:
            sets = {s: self._load(s) for s in ['train', 'val', 'test']}
        pos = {s: float(np.mean(v.y)) for s, v in sets.items()} \
            if self.ds['task'] == 'binary' else None
        print({s: len(v) for s, v in sets.items()}, 'input shape:', sets['train'][0][0].shape,
              ('| positive rate ' + str({k: round(v, 4) for k, v in pos.items()})) if pos else '')
        return {
            'train': DataLoader(sets['train'], batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, collate_fn=sets['train'].collate, drop_last=False),
            'val':   DataLoader(sets['val'], batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, collate_fn=sets['val'].collate),
            'test':  DataLoader(sets['test'], batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, collate_fn=sets['test'].collate),
        }
