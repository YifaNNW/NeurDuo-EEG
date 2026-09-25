"""Dataset of fixed-length windows of 0.5 s chunks cut from the recordings listed in a JSONL manifest."""

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eeg_fm.data.normalize import normalize_window
from torch.utils.data import Dataset


@dataclass
class ManifestRecording:
    signal_path: str
    channel_positions: list[list[float]]
    reference_ids: list[int]
    missing_mask: list[float] | None = None
    num_samples: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestRecording":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


_MMAP_CACHE_SIZE = 128


class ContinuousEEGWindowDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        chunk_samples: int,
        chunk_stride_samples: int,
        sequence_length_steps: int,
        sequence_stride_steps: int,
        normalize_per_window: bool = True,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.chunk_samples = chunk_samples
        self.chunk_stride_samples = chunk_stride_samples
        self.sequence_length_steps = sequence_length_steps
        self.sequence_stride_steps = sequence_stride_steps
        self.normalize_per_window = normalize_per_window
        self.window_num_samples = chunk_samples + (sequence_length_steps - 1) * chunk_stride_samples
        self.window_stride_samples = sequence_stride_steps * chunk_stride_samples
        self._mmap_cache: dict[str, np.ndarray] = {}
        self.records = self._load_manifest()
        self.window_index = self._build_window_index()
        self._mmap_cache.clear()

    def _load_manifest(self) -> list[ManifestRecording]:
        records = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(ManifestRecording.from_dict(json.loads(line)))
        if not records:
            raise ValueError(f"No recordings were found in manifest {self.manifest_path}.")
        return records

    def _resolve_signal_path(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute() and p.exists():
            return p
        if p.exists():
            return p
        manifest_dir = self.manifest_path.parent
        candidates = [
            manifest_dir / p,
            manifest_dir.parent / p,
        ]
        for c in candidates:
            if c.exists():
                return c
        matches = list(manifest_dir.rglob(p.name))
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(
            f"Cannot resolve signal_path '{path}' (tried as-is, relative to "
            f"manifest dir '{manifest_dir}' and its parent)."
        )

    def _read_signal(self, path: str) -> np.ndarray:
        cached = self._mmap_cache.get(path)
        if cached is not None:
            return cached
        signal_path = self._resolve_signal_path(path)
        if signal_path.suffix == ".npy":
            array = np.load(signal_path, mmap_mode="r")
        elif signal_path.suffix in {".pt", ".pth"}:
            tensor = torch.load(signal_path, map_location="cpu")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected a tensor in {signal_path}, got {type(tensor)}.")
            array = tensor.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported signal file format: {signal_path.suffix}.")
        if len(self._mmap_cache) >= _MMAP_CACHE_SIZE:
            self._mmap_cache.pop(next(iter(self._mmap_cache)))
        self._mmap_cache[path] = array
        return array

    def _build_window_index(self) -> list[tuple[int, int]]:
        index: list[tuple[int, int]] = []
        for record_idx, record in enumerate(self.records):
            num_samples = record.num_samples
            if num_samples is None:
                num_samples = int(self._read_signal(record.signal_path).shape[1])
                record.num_samples = num_samples
            if num_samples < self.window_num_samples:
                continue
            max_start = num_samples - self.window_num_samples
            for start_sample in range(0, max_start + 1, self.window_stride_samples):
                index.append((record_idx, start_sample))
        if not index:
            raise ValueError("The manifest did not yield any valid continuous windows.")
        return index

    def __len__(self) -> int:
        return len(self.window_index)

    def _slice_window_into_chunks(self, signal_window: np.ndarray) -> torch.Tensor:
        sw = np.lib.stride_tricks.sliding_window_view(
            signal_window, self.chunk_samples, axis=-1
        )
        chunks = sw[:, :: self.chunk_stride_samples][:, : self.sequence_length_steps]
        chunk_array = np.transpose(chunks, (1, 0, 2))
        chunk_tensor = torch.from_numpy(
            np.ascontiguousarray(chunk_array, dtype=np.float32)
        )
        if self.normalize_per_window:
            chunk_tensor = normalize_window(chunk_tensor)
        return chunk_tensor

    def _build_metadata_dict(self, record: ManifestRecording, num_channels: int) -> dict[str, torch.Tensor]:
        missing_mask = record.missing_mask if record.missing_mask is not None else [0.0] * num_channels
        if len(record.channel_positions) != num_channels:
            raise ValueError(
                f"Recording {record.signal_path} metadata channel_positions has length "
                f"{len(record.channel_positions)} but signal has {num_channels} channels."
            )
        if len(record.reference_ids) != num_channels:
            raise ValueError(
                f"Recording {record.signal_path} metadata reference_ids has length "
                f"{len(record.reference_ids)} but signal has {num_channels} channels."
            )
        if len(missing_mask) != num_channels:
            raise ValueError(
                f"Recording {record.signal_path} metadata missing_mask has length "
                f"{len(missing_mask)} but signal has {num_channels} channels."
            )
        return {
            "channel_positions": torch.tensor(record.channel_positions, dtype=torch.float32),
            "reference_ids": torch.tensor(record.reference_ids, dtype=torch.long),
            "missing_mask": torch.tensor(missing_mask, dtype=torch.float32),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_idx, start_sample = self.window_index[index]
        record = self.records[record_idx]
        signal = self._read_signal(record.signal_path)
        signal_window = signal[:, start_sample : start_sample + self.window_num_samples]
        chunks = self._slice_window_into_chunks(signal_window)
        metadata = self._build_metadata_dict(record, signal.shape[0])
        return {
            "chunks": chunks,
            "metadata": metadata,
        }
