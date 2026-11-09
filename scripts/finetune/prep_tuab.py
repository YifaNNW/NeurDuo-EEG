"""Segment the TUH Abnormal EEG corpus (TUAB v3.0.1) into the arrays run_finetune expects.

Protocol (BIOT / LaBraM / CBraMod style, adapted to NeurDuo's pretraining preprocessing):
  * 19 monopolar 10-20 channels (AR reference), in the pretraining TUH order:
    FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ PZ.
    (A1/A2 are reference electrodes and T1/T2 have no standard_1005 position, so the
    23-channel LaBraM set reduces to these 19, the same set the TUH pretraining data uses.)
  * Filtering as in pretraining (preprocess_tusz/tuar.py): band-pass 0.5-75 Hz and notch
    50/60 Hz at the native rate, then resample to 256 Hz. Stored in microvolts.
  * Non-overlapping 10 s windows (2560 samples); the remainder of each file is dropped.
  * Split: official train -> train/val by subject (80/20, RandomState(seed)); official eval ->
    test, with every subject that appears in the pretraining corpus (TUSZ/TUEP/TUAR) removed
    ("clean" test set, no subject overlap with pretraining).

Output in --out_dir: {train,val,test}_X.npy [N, 19, 2560] float32, {split}_y.npy (0 normal,
1 abnormal), {split}_subject.npy, {split}_files.txt, channels.txt, meta.json.
"""

import argparse
import glob
import json
import os
import random
import shutil
import sys
import traceback
from multiprocessing import Pool

import mne
import numpy as np

mne.set_log_level("ERROR")

CHANNELS = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]
TARGET_SR = 256
SEG_SEC = 10
SEG = TARGET_SR * SEG_SEC


def subject_of(path):
    return os.path.basename(path).split("_")[0]


def pretrain_tuh_subjects(manifests, extra_txt):
    subs = set()
    for m in manifests:
        with open(m) as f:
            for line in f:
                p = json.loads(line)["signal_path"]
                ds = p.split("/processed/")[-1].split("/")[0]
                if ds.startswith("TU"):
                    subs.add(subject_of(p))
    if extra_txt and os.path.exists(extra_txt):
        with open(extra_txt) as f:
            subs |= {ln.strip() for ln in f if ln.strip()}
    return subs


def process(args):
    path, tmp_dir = args
    out = os.path.join(tmp_dir, os.path.basename(path).replace(".edf", ".npy"))
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        names = {}
        for ch in raw.ch_names:
            short = ch.upper().replace("EEG ", "").split("-")[0].strip()
            if short in CHANNELS and short not in names:
                names[short] = ch
        missing = [c for c in CHANNELS if c not in names]
        if missing:
            return path, 0, f"missing channels {missing}"
        raw.pick([names[c] for c in CHANNELS])
        sr = raw.info["sfreq"]
        nyq = sr / 2.0
        raw.filter(l_freq=0.5, h_freq=min(75.0, nyq * 0.9), verbose=False)
        notches = [f for f in (50.0, 60.0) if f < nyq]
        if notches:
            raw.notch_filter(freqs=notches, verbose=False)
        if int(round(sr)) != TARGET_SR:
            raw.resample(TARGET_SR, verbose=False)
        x = raw.get_data(units="uV").astype(np.float32)
        n = x.shape[1] // SEG
        if n == 0:
            return path, 0, "shorter than one window"
        segs = x[:, : n * SEG].reshape(len(CHANNELS), n, SEG).transpose(1, 0, 2)
        if not np.isfinite(segs).all():
            return path, 0, "non-finite values"
        np.save(out, np.ascontiguousarray(segs))
        return path, n, ""
    except Exception:
        return path, 0, traceback.format_exc(limit=1).strip().splitlines()[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuab_root", required=True, help=".../TUAB/edf")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pretrain_manifests", nargs="+", required=True)
    ap.add_argument("--pretrain_subjects_txt", default="")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    files = {}
    for split in ("train", "eval"):
        for label, name in ((0, "normal"), (1, "abnormal")):
            for p in sorted(glob.glob(os.path.join(args.tuab_root, split, name, "01_tcp_ar", "*.edf"))):
                files[p] = (split, label)
    print(f"{len(files)} EDF files", flush=True)

    overlap = pretrain_tuh_subjects(args.pretrain_manifests, args.pretrain_subjects_txt)
    train_subs = sorted({subject_of(p) for p, (s, _) in files.items() if s == "train"})
    eval_subs = sorted({subject_of(p) for p, (s, _) in files.items() if s == "eval"})
    excluded = sorted(set(eval_subs) & overlap)
    rng = random.Random(args.seed)
    shuffled = train_subs[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * args.val_frac))
    val_set = set(shuffled[:n_val])
    print(f"train subjects {len(train_subs)} -> train {len(train_subs) - n_val} / val {n_val}; "
          f"eval subjects {len(eval_subs)}, excluded (in pretraining) {len(excluded)}, "
          f"clean test {len(eval_subs) - len(excluded)}", flush=True)

    def target(p):
        split, _ = files[p]
        sub = subject_of(p)
        if split == "eval":
            return None if sub in overlap else "test"
        return "val" if sub in val_set else "train"

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "_per_file")
    os.makedirs(tmp_dir, exist_ok=True)
    todo = [p for p in files if target(p) is not None]
    print(f"processing {len(todo)} files with {args.workers} workers", flush=True)

    counts, failed = {}, {}
    with Pool(args.workers) as pool:
        for i, (p, n, err) in enumerate(pool.imap_unordered(process, [(p, tmp_dir) for p in todo],
                                                            chunksize=1), 1):
            counts[p] = n
            if err:
                failed[p] = err
            if i % 200 == 0:
                print(f"  [{i}/{len(todo)}] failed={len(failed)}", flush=True)
    print(f"done: {len(todo) - len(failed)} ok, {len(failed)} skipped", flush=True)
    for p, e in sorted(failed.items()):
        print(f"  skip {os.path.basename(p)}: {e}")

    meta = dict(protocol=__doc__.strip().splitlines()[0], channels=CHANNELS, fs=TARGET_SR,
                seg_sec=SEG_SEC, seed=args.seed, val_frac=args.val_frac,
                excluded_eval_subjects=excluded, skipped_files={os.path.basename(k): v for k, v in failed.items()},
                splits={})
    for split in ("train", "val", "test"):
        ps = sorted(p for p in todo if target(p) == split and counts.get(p, 0) > 0)
        n_total = sum(counts[p] for p in ps)
        X = np.lib.format.open_memmap(os.path.join(args.out_dir, f"{split}_X.npy"), mode="w+",
                                      dtype=np.float32, shape=(n_total, len(CHANNELS), SEG))
        y = np.empty(n_total, np.int64)
        subj = np.empty(n_total, dtype="<U16")
        k = 0
        with open(os.path.join(args.out_dir, f"{split}_files.txt"), "w") as fl:
            for p in ps:
                seg = np.load(os.path.join(tmp_dir, os.path.basename(p).replace(".edf", ".npy")))
                X[k:k + len(seg)] = seg
                y[k:k + len(seg)] = files[p][1]
                subj[k:k + len(seg)] = subject_of(p)
                fl.write(f"{os.path.relpath(p, args.tuab_root)}\t{k}\t{len(seg)}\n")
                k += len(seg)
        X.flush()
        del X
        np.save(os.path.join(args.out_dir, f"{split}_y.npy"), y)
        np.save(os.path.join(args.out_dir, f"{split}_subject.npy"), subj)
        meta["splits"][split] = dict(files=len(ps), subjects=len(set(subj.tolist())), segments=int(n_total),
                                     abnormal_rate=round(float(y.mean()), 4) if n_total else None)
        print(f"{split}: {meta['splits'][split]}", flush=True)

    with open(os.path.join(args.out_dir, "channels.txt"), "w") as f:
        f.write("\n".join(CHANNELS) + "\n")
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    shutil.rmtree(tmp_dir)
    print("wrote", args.out_dir, flush=True)


if __name__ == "__main__":
    sys.exit(main())
