"""Distributed batch sampler that builds every batch from windows with the same channel count."""

from __future__ import annotations

import collections
import math
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class ChannelBucketedDistributedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not (0 <= rank < num_replicas):
            raise ValueError(f"rank {rank} is not in [0, {num_replicas})")
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        channels_of = [len(r.channel_positions) for r in dataset.records]
        buckets: dict[int, list[int]] = collections.defaultdict(list)
        for widx, (record_idx, _) in enumerate(dataset.window_index):
            buckets[channels_of[record_idx]].append(widx)
        self.buckets: list[list[int]] = [buckets[c] for c in sorted(buckets)]
        self.bucket_channels: list[int] = sorted(buckets)

        total = 0
        full_groups = 0
        leftover = 0
        for b in self.buckets:
            nb = len(b) // batch_size if drop_last else math.ceil(len(b) / batch_size)
            total += nb
            full_groups += nb // num_replicas
            leftover += nb % num_replicas
        self.total_batches = total
        self.num_aligned_groups = full_groups
        self.num_mixed_groups = leftover // num_replicas
        self.num_batches = full_groups + self.num_mixed_groups

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        groups: list[list[list[int]]] = []
        leftover: list[list[int]] = []
        for bucket in self.buckets:
            idx: Sequence[int] = bucket
            if self.shuffle:
                perm = torch.randperm(len(bucket), generator=g).tolist()
                idx = [bucket[i] for i in perm]
            batches: list[list[int]] = []
            for start in range(0, len(idx), self.batch_size):
                batch = list(idx[start : start + self.batch_size])
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
            cut = len(batches) - len(batches) % self.num_replicas
            for start in range(0, cut, self.num_replicas):
                groups.append(batches[start : start + self.num_replicas])
            leftover.extend(batches[cut:])

        cut = len(leftover) - len(leftover) % self.num_replicas
        for start in range(0, cut, self.num_replicas):
            groups.append(leftover[start : start + self.num_replicas])

        if self.shuffle:
            order = torch.randperm(len(groups), generator=g).tolist()
            groups = [groups[i] for i in order]

        assert len(groups) == self.num_batches, (
            f"{len(groups)} groups != precomputed num_batches {self.num_batches}"
        )
        return iter(group[self.rank] for group in groups)

    def describe(self) -> str:
        rows = [f"    {c:>4} channels: {len(b):>8,} windows" for c, b in zip(self.bucket_channels, self.buckets)]
        aligned = self.num_aligned_groups
        mixed = self.num_mixed_groups
        pct = 100.0 * aligned / self.num_batches if self.num_batches else 0.0
        return (
            f"[bucket-sampler] {len(self.buckets)} channel buckets, {self.total_batches:,} batches in total, "
            f"{self.num_batches:,} steps per rank "
            f"({aligned:,} same-bucket steps = {pct:.1f}%, {mixed:,} mixed leftover steps)\n" + "\n".join(rows)
        )
