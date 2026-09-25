"""Command-line entry point for spectral tokenizer training."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from eeg_fm.data import ContinuousEEGWindowDataset, collate_pretrain_batch
from eeg_fm.data.channel_bucket_sampler import ChannelBucketedDistributedBatchSampler
from eeg_fm.modules.vq_tokenizer import SpectralVQTokenizer
from eeg_fm.tokenizer_config import TokenizerConfig
from eeg_fm.tokenizer_trainer import TokenizerTrainer
from scripts.pretrain.run_pretrain import cleanup_distributed, setup_distributed


def build_config(args: argparse.Namespace) -> TokenizerConfig:
    config = TokenizerConfig.from_json(args.config) if args.config else TokenizerConfig()
    if args.manifest:
        config.manifest_path = args.manifest
    if args.val_manifest:
        config.val_manifest_path = args.val_manifest
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_epochs:
        config.max_epochs = args.max_epochs
    if args.resume_from:
        config.resume_from = args.resume_from
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.device:
        config.device = args.device
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="NeurDuo-EEG VQ Tokenizer Training (Stage 1)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None, help="per-GPU batch size")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    distributed, rank, world_size, local_rank = setup_distributed()

    config = build_config(args)
    if not config.manifest_path:
        parser.error("--manifest is required (or set manifest_path in config JSON)")

    if rank == 0:
        print("=" * 60)
        print("NeurDuo-EEG VQ Tokenizer (Stage 1)")
        print("=" * 60)
        print(f"  distributed:    {distributed} (world_size={world_size})")
        print(f"  manifest:       {config.manifest_path}")
        print(f"  val_manifest:   {config.val_manifest_path or '(none)'}")
        print(f"  output_dir:     {config.output_dir}")
        print(f"  batch_size/GPU: {config.batch_size}")
        if distributed:
            print(f"  effective batch:{config.batch_size * world_size}")
        print(f"  codebook:       {config.codebook_size} x {config.code_dim}")
        print(f"  chunk_samples:  {config.chunk_samples} (n_freq={config.n_freq_bins})")
        print(f"  learning_rate:  {config.learning_rate}")
        print("=" * 60, flush=True)

    def build_dataset(manifest_path: str) -> ContinuousEEGWindowDataset:
        return ContinuousEEGWindowDataset(
            manifest_path=manifest_path,
            chunk_samples=config.chunk_samples,
            chunk_stride_samples=config.chunk_stride_samples,
            sequence_length_steps=config.sequence_length_steps,
            sequence_stride_steps=config.sequence_stride_steps,
            normalize_per_window=config.normalize_per_window,
        )

    dataset = build_dataset(config.manifest_path)
    if rank == 0:
        print(f"Dataset: {len(dataset)} windows", flush=True)

    loader_kwargs = dict(
        num_workers=config.num_workers,
        collate_fn=collate_pretrain_batch,
        pin_memory=(config.device != "cpu") and torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

    sampler = ChannelBucketedDistributedBatchSampler(
        dataset,
        batch_size=config.batch_size,
        num_replicas=world_size if distributed else 1,
        rank=rank if distributed else 0,
        shuffle=config.shuffle,
        seed=config.seed,
        drop_last=True,
    )
    if rank == 0:
        print(sampler.describe(), flush=True)
    if len(sampler) == 0:
        parser.error("the training manifest does not fill one batch per GPU; add recordings or lower batch_size")
    dataloader = DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)

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
            val_dataset, batch_sampler=val_sampler, **loader_kwargs
        )
        if rank == 0:
            print(f"Val dataset: {len(val_dataset)} windows", flush=True)

    model = SpectralVQTokenizer(config)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Tokenizer: {total_params:,} total params, {trainable:,} trainable", flush=True)

    trainer = TokenizerTrainer(model, dataloader, config, val_dataloader=val_dataloader)
    try:
        trainer.fit()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
