"""Configuration of spectral tokenizer training, loaded from a JSON file."""

from dataclasses import dataclass
import json
from pathlib import Path

from eeg_fm.data.normalize import NORMALIZE_MODES


@dataclass
class TokenizerConfig:
    manifest_path: str = ""
    val_manifest_path: str = ""
    output_dir: str = "outputs/tokenizer_v1"
    sequence_length_steps: int = 256
    sequence_stride_steps: int = 128
    chunk_samples: int = 128
    chunk_stride_samples: int = 32
    normalize: str = "sequence"          # "sequence" | "window" | "none"; see eeg_fm/data/normalize.py
    norm_window_steps: int = 8           # chunks per window in "window" mode (8 = 4 s)

    batch_size: int = 16
    num_workers: int = 4
    max_epochs: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    warmup_steps: int = 200
    min_lr_ratio: float = 0.05
    log_every_steps: int = 50
    save_every_steps: int = 1000
    eval_every_steps: int = 200
    shuffle: bool = True
    amp: bool = True
    device: str = "cuda"
    seed: int = 1337
    resume_from: str | None = None

    codebook_size: int = 2048
    code_dim: int = 64
    ema_decay: float = 0.99
    commitment_beta: float = 1.0
    codebook_eps: float = 1e-5
    kmeans_init: bool = True
    dead_code_reset: bool = True
    dead_code_threshold: float = 1.0

    enc_dim: int = 128
    enc_layers: int = 2
    dec_hidden_dim: int = 128

    amp_weight: float = 1.0
    phase_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.normalize not in NORMALIZE_MODES:
            raise ValueError(f"normalize must be one of {NORMALIZE_MODES}, got {self.normalize!r}")

    @property
    def n_freq_bins(self) -> int:
        return self.chunk_samples // 2 + 1

    @classmethod
    def from_json(cls, path: str | Path) -> "TokenizerConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        return cls(**data)
