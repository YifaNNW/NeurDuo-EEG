"""Backbone pretraining with sub-window normalization (eeg_fm/data/normalize_con.py).

Same as run_pretrain.py, except that every training/validation sequence is
normalized in independent blocks of --norm_window_steps chunks (default 8 = 4 s)
instead of once over the whole sequence. All other arguments are passed through.
Pass the tokenizer trained with run_train_tokenizer_con.py as --tokenizer_ckpt.
"""

import argparse
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.pretrain.run_pretrain as base  # noqa: E402
from eeg_fm.data.normalize_con import ContinuousEEGWindowDatasetCon  # noqa: E402


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--norm_window_steps", type=int, default=8)
    known, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0], *rest]

    # base.main() looks this name up when it builds the datasets.
    base.ContinuousEEGWindowDataset = functools.partial(
        ContinuousEEGWindowDatasetCon, norm_window_steps=known.norm_window_steps
    )
    print(f"[normalize_con] sub-window normalization: {known.norm_window_steps} chunks per block",
          flush=True)
    base.main()


if __name__ == "__main__":
    main()
