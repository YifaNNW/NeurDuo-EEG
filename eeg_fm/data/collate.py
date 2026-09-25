"""Collate function that stacks windows and their channel metadata into a batch."""

from typing import Any

import torch

from eeg_fm.state import stack_metadata_dicts


def collate_pretrain_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    chunks = torch.stack([sample["chunks"] for sample in samples], dim=0)
    metadata = stack_metadata_dicts([sample["metadata"] for sample in samples])
    return {
        "chunks": chunks,
        "metadata": metadata,
    }
