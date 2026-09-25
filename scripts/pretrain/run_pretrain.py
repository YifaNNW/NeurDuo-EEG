"""Command-line entry point for backbone pretraining."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from eeg_fm.data import ContinuousEEGWindowDataset, collate_pretrain_batch
from eeg_fm.data.channel_bucket_sampler import ChannelBucketedDistributedBatchSampler
from eeg_fm.pretrain_config import PretrainConfig
from eeg_fm.pretraining import EEGFMPretrainer
from eeg_fm.trainer import PretrainTrainer


def build_config(args: argparse.Namespace) -> PretrainConfig:
    if args.config:
        config = PretrainConfig.from_json(args.config)
    else:
        config = PretrainConfig()

    if args.manifest:
        config.manifest_path = args.manifest
    if args.val_manifest:
        config.val_manifest_path = args.val_manifest
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.tokenizer_ckpt:
        config.tokenizer_ckpt = args.tokenizer_ckpt
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_epochs:
        config.max_epochs = args.max_epochs
    if args.eval_every_steps:
        config.eval_every_steps = args.eval_every_steps
    if args.resume_from:
        config.resume_from = args.resume_from
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.device:
        config.device = args.device

    return config


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    return True, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="NeurDuo-EEG Pretraining")
    parser.add_argument("--config", type=str, default=None, help="JSON config file path")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tokenizer_ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None, help="per-GPU batch size")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--eval_every_steps", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    distributed, rank, world_size, local_rank = setup_distributed()

    config = build_config(args)
    if not config.manifest_path:
        parser.error("--manifest is required (or set manifest_path in config JSON)")
    if not config.tokenizer_ckpt:
        parser.error("--tokenizer_ckpt is required (or set tokenizer_ckpt in config JSON)")

    if rank == 0:
        print("=" * 60)
        print("NeurDuo-EEG Pretraining")
        print("=" * 60)
        print(f"  distributed:    {distributed} (world_size={world_size})")
        print(f"  manifest:       {config.manifest_path}")
        print(f"  val_manifest:   {config.val_manifest_path or '(none)'}")
        print(f"  output_dir:     {config.output_dir}")
        print(f"  batch_size/GPU: {config.batch_size}")
        if distributed:
            print(f"  effective batch:{config.batch_size * world_size}")
        print(f"  learning_rate:  {config.learning_rate}")
        print(f"  max_epochs:     {config.max_epochs}")
        print(f"  device:         {config.device}")
        print(f"  amp:            {config.amp} dtype={config.amp_dtype}")
        print(f"  seq_len_steps:  {config.sequence_length_steps}")
        print(f"  chunk_samples:  {config.model.chunk_samples}")
        print(f"  token_dim:      {config.model.token_dim}")
        print(f"  fast_dim:       {config.model.fast_dim}")
        print(f"  slow_dim:       {config.model.slow_dim}")
        print("=" * 60, flush=True)

    def build_dataset(manifest_path: str) -> ContinuousEEGWindowDataset:
        return ContinuousEEGWindowDataset(
            manifest_path=manifest_path,
            chunk_samples=config.model.chunk_samples,
            chunk_stride_samples=config.model.chunk_stride_samples,
            sequence_length_steps=config.sequence_length_steps,
            sequence_stride_steps=config.sequence_stride_steps,
            normalize_per_window=config.normalize_per_window,
        )

    dataset = build_dataset(config.manifest_path)
    if rank == 0:
        print(f"Dataset: {len(dataset)} windows", flush=True)

    common_loader_kwargs = dict(
        num_workers=config.num_workers,
        collate_fn=collate_pretrain_batch,
        pin_memory=(config.device != "cpu") and torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )

    batch_sampler = ChannelBucketedDistributedBatchSampler(
        dataset,
        batch_size=config.batch_size,
        num_replicas=world_size if distributed else 1,
        rank=rank if distributed else 0,
        shuffle=config.shuffle,
        seed=config.seed,
        drop_last=True,
    )
    if rank == 0:
        print(batch_sampler.describe(), flush=True)
    if len(batch_sampler) == 0:
        parser.error("the training manifest does not fill one batch per GPU; add recordings or lower batch_size")
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, **common_loader_kwargs)

    val_dataloader = None
    if config.val_manifest_path:
        val_dataset = build_dataset(config.val_manifest_path)
        val_sampler = ChannelBucketedDistributedBatchSampler(
            val_dataset,
            batch_size=config.batch_size,
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
            shuffle=False,
            seed=config.seed,
            drop_last=False,
        )
        if len(val_sampler) == 0:
            parser.error("the validation manifest does not give every GPU a batch; add recordings or use fewer GPUs")
        val_dataloader = DataLoader(
            val_dataset, batch_sampler=val_sampler, **common_loader_kwargs
        )
        if rank == 0:
            print(f"Val dataset: {len(val_dataset)} windows", flush=True)

    model = EEGFMPretrainer(
        config.model,
        tokenizer_ckpt=config.tokenizer_ckpt,
    )
    if rank == 0:
        print(
            f"Objective: future-token CE (codebook={config.model.codebook_size}), "
            f"tokenizer_ckpt={config.tokenizer_ckpt}",
            flush=True,
        )
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {total_params:,} total params, {trainable_params:,} trainable", flush=True)

    trainer = PretrainTrainer(model, dataloader, config, val_dataloader=val_dataloader)
    try:
        trainer.fit()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
