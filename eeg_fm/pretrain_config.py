"""Configuration of backbone pretraining, loaded from a JSON file."""

from dataclasses import dataclass, field
import json
from pathlib import Path

from eeg_fm.config import EEGFMConfig


@dataclass
class PretrainConfig:
    manifest_path: str = ""
    val_manifest_path: str = ""
    output_dir: str = "outputs/pretrain"
    sequence_length_steps: int = 256
    sequence_stride_steps: int = 128
    batch_size: int = 4
    num_workers: int = 0
    normalize_per_window: bool = True
    shuffle: bool = True

    max_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0
    frontend_lr_scale: float = 0.1
    warmup_steps: int = 500
    min_lr_ratio: float = 0.05

    log_every_steps: int = 10
    save_every_steps: int = 500
    eval_every_steps: int = 500
    resume_from: str | None = None
    warm_start_from: str | None = None

    amp: bool = True
    amp_dtype: str = "bf16"
    device: str = "cuda"
    seed: int = 1337

    tokenizer_ckpt: str = ""

    model: EEGFMConfig = field(default_factory=EEGFMConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "PretrainConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data = {key: value for key, value in data.items() if not key.startswith("_")}
        model_data = {
            key: value
            for key, value in data.pop("model", {}).items()
            if not key.startswith("_")
        }
        return cls(model=EEGFMConfig(**model_data), **data)
