"""CHB-MIT perturbation test of how long early EEG still affects a later prediction (rebuttal,
reviewer ReVg Q1/W1). Inference only, on one fine-tuned model per fold.

For each 10 min test window the target is the last 30 s unit (9:30-10:00). Each perturbed copy
overwrites one span that ends ``lag`` seconds before 9:30 with other real EEG of the same patient:
  test A  background EEG from the unit right before the window (4 s and 30 s spans)
  test B  a 30 s seizure unit of the same patient from another recording (planted event)
Normalisation statistics come from the original window and are reused for every copy (see
perturb_lib). The context head gets the original embeddings of units 0-18 and the perturbed copy's
embedding of the target unit only ("backbone path"), so the effect reaches the prediction only
through the backbone state; the effect with every embedding perturbed is kept as "total".

Per row of the output (one window x one perturbation) we keep the seizure log-odds of the target
unit for the original, backbone-path and total predictions, the log-odds with the target unit run
alone from an empty state ("reset", all history removed), and the relative change of the fast and
slow stream outputs and of the target embedding just before 9:30.
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import perturb_lib as P
from eeg_fm.config import EEGFMConfig
from eeg_fm.data.normalize import normalize_sequence
from eeg_fm.finetune.dataset import LoadDataset
from eeg_fm.finetune.dataset_specs import get_spec
from eeg_fm.finetune.net_eegfm import Model
from eeg_fm.finetune.net_seq import SeqModel

RUN = '/projects/u6xc/haiping/neurduo/runs/finetune/small/chbmit'


def load_model(path, ds, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    a = payload['args']
    known = {f.name for f in dataclasses.fields(EEGFMConfig)}
    cfg = EEGFMConfig(**{k: v for k, v in payload['model_config'].items() if k in known})
    params = SimpleNamespace(
        channels=ds['channels'], seg_sec=ds['seg_sec'], model_config=cfg,
        cont_normalize=a.get('cont_normalize') or ds.get('cont_normalize', 'window'),
        use_pretrained_weights=False, foundation_dir=None,
        readout_budget=ds['readout_budget'], num_of_classes=ds['n_outputs'],
        dropout=a['dropout'], context_len=a.get('context_len') or ds['context_len'])
    assert params.cont_normalize == 'sequence', \
        f"this test assumes whole-sequence normalisation, the model used {params.cont_normalize!r}"
    model = SeqModel(Model(params), params)
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval(), a


@torch.no_grad()
def run(model, chunks):
    """Backbone, readout and unit embeddings for already-normalised chunks [B, S, C, K].

    Same steps as EEGFM.forward and Model.tokens_continuous, minus the normalisation.
    Returns the unit embeddings [B, S/60, 256] and the fast stream, slow stream and slow stream
    plus memory retrieval outputs at the last step, each [B, C, D] (None for a fast-only model).
    """
    base, bb = model.base, model.base.backbone
    B, S, C, _ = chunks.shape
    md = base._metadata(B, chunks.device)
    fast = bb._encode_fast_sequence(chunks, md)
    if bb.config.fast_only:
        fused, slow_scan, slow_all = fast, None, None
    else:
        mem = bb._write_memory(fast)
        slot = bb._memory_slot_index(S, chunks.device)
        slow_scan = bb._scan_slow_sequence(fast, mem, slot)
        slow_all = bb._add_memory_context(fast, slow_scan, mem, slot)
        fused = bb.readout(fast, slow_all)
    l, d = S // base.n_steps, fused.size(2)
    tokens = (fused.view(B, C, l, base.n_steps, d).permute(0, 2, 1, 3, 4)
                   .reshape(B * l, C * base.n_steps, d))
    h = model.st_head.readout(tokens).view(B, l, -1)
    last = lambda t: None if t is None else t.view(B, C, S, -1)[:, :, -1]
    return h, last(fast), last(slow_scan), last(slow_all)


@torch.no_grad()
def target_log_odds(model, h):
    z = model.context(h)[:, P.TARGET]
    return (z[:, 1] - z[:, 0]).float()


def rel_change(a, b):
    """||b - a|| / ||a|| per batch row, over channels and features."""
    if a is None:
        return torch.full((b.shape[0],), float('nan'), device=b.device)
    return (b - a).flatten(1).norm(dim=1) / a.flatten(1).norm(dim=1).clamp_min(1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--data_dir', default='finetune_data/chbmit')
    ap.add_argument('--model', default=None, help=f'default {RUN}/models/fold<k>_seed42.pt')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--lags', type=int, nargs='+', default=[0, 4, 16, 32, 60, 120, 300, 540])
    ap.add_argument('--durations_a', type=int, nargs='+', default=[4, 30])
    ap.add_argument('--duration_b', type=int, default=30)
    ap.add_argument('--max_per_subject', type=int, default=150,
                    help='windows drawn per test subject and test (fixed seed); 0 = all')
    ap.add_argument('--max_windows', type=int, default=0, help='stop after this many windows (smoke)')
    ap.add_argument('--batch', type=int, default=13, help='copies per forward pass')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda')
    ds = get_spec('chbmit', args.data_dir)
    assert ds.get('channel_subset') is None, 'material rows are read without a channel subset'
    model_path = args.model or f'{RUN}/models/fold{args.fold}_seed42.pt'
    model, margs = load_model(model_path, ds, device)
    assert margs['fold'] == args.fold, f"{model_path} is fold {margs['fold']}, not {args.fold}"
    print(f'[perturb] fold {args.fold} model {model_path} (seed {margs["seed"]})', flush=True)

    test = LoadDataset(ds, fold=args.fold, num_workers=0)._load_sequences()['test']
    arr = {k: np.load(f'{args.data_dir}/{k}.npy') for k in ('y', 'subject', 'recording', 'epoch_idx')}
    index = P.WindowIndex(arr['y'], arr['subject'], arr['recording'], arr['epoch_idx'], seed=args.seed)
    spans_a = [s for s in P.make_spans(args.lags, args.durations_a, args.duration_b) if s.test == 'A']
    spans_b = [s for s in P.make_spans(args.lags, args.durations_a, args.duration_b) if s.test == 'B']

    # Which windows take part in which test, capped per subject with a fixed seed.
    plan = {}
    for test_name in ('A', 'B'):
        by_sub = {}
        for wi, rows in enumerate(test.windows):
            src = (index.eligible_a(rows) if test_name == 'A'
                   else index.donor_b(rows, (args.fold, wi)))
            if src is not None:
                by_sub.setdefault(int(arr['subject'][rows[0]]), []).append((wi, src))
        rng = np.random.default_rng([args.seed, args.fold, ord(test_name)])
        for sub, items in sorted(by_sub.items()):
            if args.max_per_subject and len(items) > args.max_per_subject:
                items = [items[i] for i in sorted(rng.choice(len(items), args.max_per_subject, replace=False))]
            for wi, src in items:
                plan.setdefault(wi, {})[test_name] = src
        print(f'[perturb] test {test_name}: ' + ', '.join(
            f'subject {s} {min(len(v), args.max_per_subject or len(v))}/{len(v)}'
            for s, v in sorted(by_sub.items())), flush=True)
    order = sorted(plan)
    if args.max_windows:
        order = order[:args.max_windows]
    print(f'[perturb] {len(order)} windows, spans A {len(spans_a)} B {len(spans_b)}', flush=True)

    out = {k: [] for k in ('fold', 'window', 'subject', 'recording', 'target_label', 'target_preictal',
                           'test', 'dur', 'lag', 'source_row',
                           'lo_orig', 'lo_bb', 'lo_total', 'lo_reset', 'lo_repeat',
                           'd_fast', 'd_slow_scan', 'd_slow_all', 'd_emb')}
    t0, checked_path = time.time(), False
    for n_done, wi in enumerate(order):
        rows = test.windows[wi]
        x = torch.from_numpy(test[wi][0]).to(device)                    # [20, C, 7680], raw
        chunks = P.to_chunks(x)                                         # [1200, C, 128]
        mean, scale = P.seq_stats(chunks)
        base = P.apply_stats(chunks, mean, scale)
        assert torch.equal(base, normalize_sequence(chunks.unsqueeze(0)).squeeze(0))

        copies, meta = [base], [None]
        for test_name, spans in (('A', spans_a), ('B', spans_b)):
            src = plan[wi].get(test_name)
            if src is None:
                continue
            unit = torch.from_numpy(np.asarray(test.X[src], np.float32)).to(device)   # [C, 7680]
            mat_unit = P.apply_stats(P.to_chunks(unit.unsqueeze(0)), mean, scale)   # [60, C, 128]
            for span in spans:
                mat = mat_unit[P.UNIT_CHUNKS - span.n_chunks:]           # last dur seconds of the unit
                pert = P.perturb(base, span, mat)
                P.check_only_span_differs(base, pert, span)
                copies.append(pert)
                meta.append((span, src))
        batch = torch.stack(copies)                                      # [V, 1200, C, 128]

        hs, fs, ss, sa = [], [], [], []
        for i in range(0, len(batch), args.batch):
            h, f, s1, s2 = run(model, batch[i:i + args.batch])
            hs.append(h); fs.append(f); ss.append(s1); sa.append(s2)
        h = torch.cat(hs)
        f = torch.cat(fs)
        s1 = None if ss[0] is None else torch.cat(ss)
        s2 = None if sa[0] is None else torch.cat(sa)

        h0 = h[:1]
        lo_orig = target_log_odds(model, h0)
        # The same original input again, alone in its batch: the run-to-run floor.
        lo_repeat = target_log_odds(model, run(model, base.unsqueeze(0))[0])
        # All history removed: the target unit alone from an empty state, same statistics.
        h_reset = run(model, base[P.TARGET_START_CHUNK:].unsqueeze(0))[0]
        mix = h0.clone(); mix[:, P.TARGET] = h_reset[:, 0]
        lo_reset = target_log_odds(model, mix)

        if not checked_path:
            # Our path must give the model's own prediction on the raw window.
            m = torch.ones(1, P.CONTEXT_LEN, dtype=torch.bool, device=device)
            z = model(x.unsqueeze(0), m)[:, P.TARGET]
            ref = (z[:, 1] - z[:, 0]).float()
            diff = (ref - lo_orig).abs().item()
            print(f'[check] model() vs our path, target log-odds diff {diff:.2e}', flush=True)
            assert diff < 1e-3, 'our forward path does not reproduce the model'
            checked_path = True

        mixes = h0.repeat(len(h) - 1, 1, 1)
        mixes[:, P.TARGET] = h[1:, P.TARGET]
        lo_bb = target_log_odds(model, mixes)
        lo_total = target_log_odds(model, h[1:])
        d_fast = rel_change(f[:1], f[1:])
        d_s1 = rel_change(None, f[1:]) if s1 is None else rel_change(s1[:1], s1[1:])
        d_s2 = rel_change(None, f[1:]) if s2 is None else rel_change(s2[:1], s2[1:])
        d_emb = rel_change(h0[:, P.TARGET], h[1:, P.TARGET])

        label, pre = index.target_info(rows)
        for j, (span, src) in enumerate(meta[1:]):
            vals = dict(fold=args.fold, window=wi, subject=int(arr['subject'][rows[0]]),
                        recording=int(arr['recording'][rows[0]]), target_label=label,
                        target_preictal=pre, test=span.test, dur=span.dur_sec, lag=span.lag_sec,
                        source_row=src, lo_orig=lo_orig.item(), lo_bb=lo_bb[j].item(),
                        lo_total=lo_total[j].item(), lo_reset=lo_reset.item(),
                        lo_repeat=lo_repeat.item(), d_fast=d_fast[j].item(),
                        d_slow_scan=d_s1[j].item(), d_slow_all=d_s2[j].item(),
                        d_emb=d_emb[j].item())
            for k, v in vals.items():
                out[k].append(v)
        if n_done % 50 == 0 or n_done == len(order) - 1:
            el = time.time() - t0
            print(f'[perturb] {n_done + 1}/{len(order)} windows, {el / 60:.1f} min, '
                  f'last |orig-repeat| {abs(lo_orig.item() - lo_repeat.item()):.2e}', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'fold{args.fold}.npz')
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in out.items()},
                        args=json.dumps(vars(args)), model=model_path)
    print(f'[perturb] saved {len(out["window"])} rows to {path} in {(time.time() - t0) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
