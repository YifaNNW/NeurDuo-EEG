"""Containers for channel metadata and for the recurrent state used in streaming inference."""

from dataclasses import dataclass, fields
from typing import Optional

import torch


@dataclass
class EEGMetadataBatch:
    channel_positions: Optional[torch.Tensor] = None
    reference_ids: Optional[torch.Tensor] = None
    missing_mask: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        for value in self.__dict__.values():
            if value is not None:
                return int(value.shape[0])
        raise ValueError("Cannot infer batch size from an empty metadata batch.")

    def to(self, device: torch.device | str) -> "EEGMetadataBatch":
        moved = {}
        for item in fields(self):
            value = getattr(self, item.name)
            moved[item.name] = value.to(device) if value is not None else None
        return EEGMetadataBatch(**moved)


@dataclass
class SequenceBuffer:
    values: torch.Tensor
    length: int


@dataclass
class MambaBlockState:
    conv_state: torch.Tensor
    ssm_state: torch.Tensor


@dataclass
class MambaStackState:
    layer_states: tuple[MambaBlockState, ...]
    last_output: torch.Tensor


def stack_metadata_dicts(items: list[dict[str, Optional[torch.Tensor]]]) -> EEGMetadataBatch:
    if not items:
        raise ValueError("Cannot stack an empty metadata list.")

    stacked: dict[str, Optional[torch.Tensor]] = {}
    for name in EEGMetadataBatch().__dict__.keys():
        values = [item.get(name) for item in items]
        if all(value is None for value in values):
            stacked[name] = None
            continue
        if any(value is None for value in values):
            raise ValueError(f"Metadata field {name} is missing for part of the batch.")
        stacked[name] = torch.stack(values, dim=0)
    return EEGMetadataBatch(**stacked)
