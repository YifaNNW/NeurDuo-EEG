"""Check that streaming inference reproduces the whole-window forward pass."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from eeg_fm.finetune.dataset_specs import DATASET_SPECS, get_spec
from eeg_fm.finetune.net_eegfm import Model, load_model_config
from eeg_fm.streaming import StreamRunner


def use_reference_kernels():
    import mamba_ssm.modules.mamba_simple as mamba_simple
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    mamba_simple.causal_conv1d_fn = None
    mamba_simple.causal_conv1d_update = None
    mamba_simple.selective_scan_fn = selective_scan_ref
    mamba_simple.selective_state_update = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None, help='pretrained backbone checkpoint')
    ap.add_argument('--model_config', default=None,
                    help='pretraining JSON; random weights when no checkpoint is given')
    ap.add_argument('--dataset', default='chbmit', choices=list(DATASET_SPECS),
                    help='takes the channel montage of this dataset')
    ap.add_argument('--steps', type=int, default=24)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                    choices=['cuda', 'cpu'])
    args = ap.parse_args()
    if not (args.checkpoint or args.model_config):
        ap.error('pass --checkpoint or --model_config')
    if args.device == 'cpu':
        use_reference_kernels()
    dev = torch.device(args.device)

    ds = get_spec(args.dataset)
    C = len(ds['channels'])
    p = SimpleNamespace(
        dropout=0.1, use_pretrained_weights=bool(args.checkpoint), foundation_dir=args.checkpoint,
        model_config=load_model_config(args.checkpoint or '', args.model_config or ''),
        num_of_classes=ds['n_outputs'], channels=ds['channels'], seg_sec=ds['seg_sec'],
        readout_budget=ds['readout_budget'])
    w = Model(p).eval().to(dev)
    bb, L, S = w.backbone, w.chunk, args.steps
    md = w._metadata(1, dev)
    torch.manual_seed(0)
    chunks = torch.randn(1, S, C, L, device=dev)

    with torch.no_grad():
        ref = bb(chunks, md)
    print(f"  reference (whole window) {tuple(ref.shape)}", flush=True)

    r = StreamRunner(bb, md, C, batch=1, device=dev)
    got = torch.stack([r.step(chunks[:, i]) for i in range(S)], 1)
    d = (got - ref).abs()
    rel = (d.max() / ref.abs().max()).item()
    ok = rel < 1e-4
    print(f"  {'PASS' if ok else 'FAIL'} ({args.device}) streaming vs whole-window scan  max|Δ|={d.max():.3e}  "
          f"rel={rel:.3e}  mean|Δ|={d.mean():.3e}", flush=True)
    print(f"  persistent state: {r.state_bytes()/2**20:.2f} MB ({C} channels)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
