"""Command-line entry point for fine-tuning a pretrained backbone on one downstream task."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from eeg_fm.finetune.dataset import LoadDataset
from eeg_fm.finetune.dataset_specs import DATASET_SPECS, get_spec
from eeg_fm.finetune.net_eegfm import Model, load_model_config
from eeg_fm.finetune.net_seq import SeqModel, SingleEpochModel
from eeg_fm.finetune.trainer import Trainer


def setup_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    ap = argparse.ArgumentParser(description="NeurDuo-EEG downstream fine-tuning")
    ap.add_argument('--dataset', choices=list(DATASET_SPECS), required=True)
    ap.add_argument('--data_dir', required=True, help='directory with the segmented arrays')
    ap.add_argument('--checkpoint', default=None, help='pretrained backbone checkpoint')
    ap.add_argument('--no_pretrained', action='store_true',
                    help='train from random initialization (architecture from --model_config)')
    ap.add_argument('--model_config', default=None,
                    help='pretraining JSON that defines the architecture with --no_pretrained')
    ap.add_argument('--output_dir', default='outputs/finetune')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=None,
                    help='default: 50 for window tasks, 20 for sequence tasks')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4, help='backbone learning rate')
    ap.add_argument('--weight_decay', type=float, default=5e-2)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--label_smoothing', type=float, default=0.1)
    ap.add_argument('--clip_value', type=float, default=1.0)
    ap.add_argument('--warmup_epochs', type=int, default=3)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--context_len', type=int, default=None)
    ap.add_argument('--cont_normalize', choices=['window', 'sequence'], default=None)
    ap.add_argument('--grad_accum', type=int, default=1)
    ap.add_argument('--tag', default='', help='suffix for the result filename')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--keep_ckpt', action='store_true',
                    help='keep the resume checkpoint after a successful run')
    args = ap.parse_args()

    setup_seed(args.seed)

    ds = get_spec(args.dataset, args.data_dir)
    if 'n_folds' in ds:
        assert 0 <= args.fold < ds['n_folds'], \
            f"--fold {args.fold} out of range for {args.dataset} (n_folds={ds['n_folds']})"
    if args.context_len is not None:
        ds['context_len'] = args.context_len
    seq = ds['task'] == 'seq2seq'
    epochs = args.epochs or (20 if seq else 50)

    if args.no_pretrained:
        if not args.model_config:
            ap.error('--no_pretrained needs --model_config to define the architecture')
        model_config = load_model_config(pretrain_config=args.model_config)
    else:
        if not args.checkpoint:
            ap.error('--checkpoint is required unless --no_pretrained is given')
        model_config = load_model_config(checkpoint=args.checkpoint)

    bs = args.batch_size
    loader_bs = bs
    if seq:
        epoch_budget = int(round(160 * bs / 64))
        loader_bs = max(1, int(round(epoch_budget / ds['context_len'])))
        bs = loader_bs * ds['context_len']
        if args.grad_accum > 1:
            accum = max(d for d in range(1, args.grad_accum + 1) if loader_bs % d == 0)
            print(f"[main] grad_accum: requested {args.grad_accum} -> using {accum} "
                  f"(micro-batch {loader_bs // accum} sequences, effective batch {bs} units)",
                  flush=True)
            loader_bs = loader_bs // accum
            args.grad_accum = accum
        print(f"[main] seq2seq: L={ds['context_len']} -> {loader_bs} sequences per micro-batch "
              f"x {args.grad_accum} = {bs} units per step (budget {epoch_budget})", flush=True)

    params = SimpleNamespace(
        lr=args.lr, weight_decay=args.weight_decay, epochs=epochs, batch_size=bs,
        dropout=args.dropout, label_smoothing=args.label_smoothing, clip_value=args.clip_value,
        use_pretrained_weights=not args.no_pretrained, foundation_dir=args.checkpoint,
        model_config=model_config, warmup_epochs=args.warmup_epochs,
        dataset=args.dataset, task=ds['task'], num_of_classes=ds['n_outputs'],
        channels=ds['channels'], seg_sec=ds['seg_sec'], readout_budget=ds['readout_budget'],
        context_len=ds.get('context_len', 1),
        cont_normalize=args.cont_normalize or ds.get('cont_normalize', 'window'),
        seq_select_key=ds.get('seq_select_key', 'kappa'),
        grad_accum=args.grad_accum if seq else 1,
    )

    if seq:
        rep = f'fold{args.fold}_seed{args.seed}'
    else:
        rep = f'fold{args.fold}' if 'n_folds' in ds else f'seed{args.seed}'
    stem = f'{rep}{args.tag}'
    res_dir = os.path.join(args.output_dir, args.dataset)
    params.ckpt_path = os.path.join(res_dir, 'checkpoints', f'{stem}.pt')
    params.resume = args.resume
    print(f"[{args.dataset} {rep}] {params}", flush=True)

    loaders = LoadDataset(ds, batch_size=loader_bs, fold=args.fold,
                          num_workers=args.num_workers).get_data_loader()
    model = Model(params)
    if seq:
        model = (SeqModel if params.context_len > 1 else SingleEpochModel)(model, params)
    n_bb = sum(p.numel() for n, p in model.named_parameters() if 'backbone' in n)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[neurduo] params: backbone {n_bb/1e6:.2f}M | head {(n_all-n_bb)/1e6:.2f}M "
          f"| total {n_all/1e6:.2f}M", flush=True)

    torch.cuda.reset_peak_memory_stats()
    _t0 = time.time()
    trainer = Trainer(params, loaders, model)
    res = trainer.train()
    res['training_time_min'] = round((time.time() - _t0) / 60, 2)
    res['peak_gpu_memory_gb'] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    res['checkpoint'] = args.checkpoint
    res.update(dataset=args.dataset, task=ds['task'], seed=args.seed, fold=args.fold,
               pretrained=not args.no_pretrained, readout_budget=ds['readout_budget'],
               warmup_epochs=args.warmup_epochs, epochs=epochs, batch_size=bs,
               windows_per_batch=loader_bs, grad_accum=params.grad_accum,
               n_params_backbone=int(n_bb), n_params_head=int(n_all - n_bb))
    if seq:
        res.update(context_len=params.context_len,
                   context_seconds=params.context_len * ds['seg_sec'],
                   cont_normalize=params.cont_normalize,
                   channels=ds['channels'], sampling_rate=ds['base_fs'])
    os.makedirs(res_dir, exist_ok=True)
    with open(f'{res_dir}/{stem}.json', 'w') as f:
        json.dump(res, f, indent=2)

    if seq:
        meta = loaders['test'].dataset.meta
        pred_dir = os.path.join(res_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        np.savez_compressed(
            f'{pred_dir}/{stem}.npz',
            subject=np.concatenate([[m['subject']] * m['n_valid'] for m in meta]),
            recording=np.concatenate([[m['recording']] * m['n_valid'] for m in meta]),
            window=np.concatenate([[m['window']] * m['n_valid'] for m in meta]),
            epoch_idx=np.concatenate([m['epoch_idx'] for m in meta]),
            **{k: v for k, v in trainer.test_predictions.items() if v is not None})
        print('saved predictions:', f'{pred_dir}/{stem}.npz', flush=True)
    print('saved result:', {k: v for k, v in res.items() if k != 'confusion_matrix'},
          flush=True)
    if not args.keep_ckpt and params.ckpt_path and os.path.exists(params.ckpt_path):
        os.remove(params.ckpt_path)


if __name__ == '__main__':
    main()
