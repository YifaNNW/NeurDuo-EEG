"""Distributed training loop for the spectral VQ tokenizer."""

import os
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from eeg_fm.modules.vq_tokenizer import SpectralVQTokenizer
from eeg_fm.tokenizer_config import TokenizerConfig
from eeg_fm.trainer import (
    _build_scheduler,
    get_rank,
    get_world_size,
    is_dist_initialized,
    is_main_process,
    set_random_seed,
)


class TokenizerTrainer:
    def __init__(
        self,
        model: SpectralVQTokenizer,
        dataloader,
        config: TokenizerConfig,
        val_dataloader=None,
    ) -> None:
        self.raw_model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.config = config

        self.distributed = is_dist_initialized()
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if self.distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(
                config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu"
            )

        self.raw_model.to(self.device)
        if self.distributed:
            self.model = DDP(
                self.raw_model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                output_device=self.local_rank if self.device.type == "cuda" else None,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
        else:
            self.model = self.raw_model

        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.raw_model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.total_steps = config.max_epochs * len(dataloader)
        self.scheduler = _build_scheduler(self.optimizer, config, self.total_steps)
        self.scaler = torch.amp.GradScaler(enabled=config.amp and self.device.type == "cuda")
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss: float | None = None

    def _autocast_context(self):
        if self.config.amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def save_config_snapshot(self) -> None:
        if not is_main_process():
            return
        config_path = self.output_dir / "tokenizer_config.snapshot.json"
        config_path.write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    def _checkpoint_payload(self, epoch: int) -> dict:
        return {
            "tokenizer": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
        }

    def save_checkpoint(self, epoch: int, *, tag: str | None = None) -> Path | None:
        if not is_main_process():
            return None
        payload = self._checkpoint_payload(epoch)
        checkpoint_path = self.checkpoint_dir / (f"{tag}.pt" if tag else f"step_{self.global_step:08d}.pt")
        torch.save(payload, checkpoint_path)
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(payload, latest_path)
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> None:
        map_location = {"cuda:0": f"cuda:{self.local_rank}"} if self.device.type == "cuda" else "cpu"
        payload = torch.load(path, map_location=map_location)
        self.raw_model.load_state_dict(payload["tokenizer"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
        if "scaler" in payload:
            self.scaler.load_state_dict(payload["scaler"])
        self.start_epoch = int(payload["epoch"])
        self.global_step = int(payload["global_step"])
        self.best_val_loss = payload.get("best_val_loss")

    def _log(self, msg: str) -> None:
        if is_main_process():
            print(msg, flush=True)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if self.val_dataloader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        metric_sums: dict[str, float] = {}
        for batch in self.val_dataloader:
            chunks = batch["chunks"].to(self.device, non_blocking=True)
            metadata = batch["metadata"].to(self.device)
            with self._autocast_context():
                outputs = self.model(chunks, metadata)
                loss = outputs["loss"]
            total_loss += float(loss.detach().item())
            total_batches += 1
            for name, value in outputs["metrics"].items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value.detach().item())

        if self.distributed:
            packed = torch.tensor(
                [total_loss, float(total_batches), *metric_sums.values()],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            total_loss = float(packed[0].item())
            total_batches = int(packed[1].item())
            for idx, name in enumerate(metric_sums.keys(), start=2):
                metric_sums[name] = float(packed[idx].item())

        denom = max(1, total_batches)
        metrics = {"val_loss": total_loss / denom}
        metrics.update({f"val_{name}": value / denom for name, value in metric_sums.items()})
        self.model.train()
        return metrics

    def _maybe_save_best(self, epoch: int, metrics: dict[str, float]) -> None:
        val_loss = metrics.get("val_loss")
        if val_loss is None:
            return
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            if self.distributed:
                dist.barrier()
            path = self.save_checkpoint(epoch, tag="best")
            if path is not None:
                self._log(f"new best val_loss={val_loss:.4f}, checkpoint: {path}")

    def fit(self) -> None:
        set_random_seed(self.config.seed + self.rank)
        self.save_config_snapshot()
        if self.config.resume_from is not None:
            self.load_checkpoint(self.config.resume_from)
            if self.distributed:
                dist.barrier()

        self._log(
            f"[tokenizer-trainer] distributed={self.distributed} world_size={self.world_size} "
            f"rank={self.rank} device={self.device} total_steps={self.total_steps}"
        )

        for epoch in range(self.start_epoch, self.config.max_epochs):
            self.dataloader.batch_sampler.set_epoch(epoch)

            self.model.train()
            for batch in self.dataloader:
                chunks = batch["chunks"].to(self.device, non_blocking=True)
                metadata = batch["metadata"].to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    outputs = self.model(chunks, metadata)
                    loss = outputs["loss"]
                nonfinite = (~torch.isfinite(loss)).float().reshape(1)
                if self.distributed:
                    dist.all_reduce(nonfinite)
                if nonfinite.item() > 0:
                    self._nonfinite_streak = getattr(self, "_nonfinite_streak", 0) + 1
                    if self._nonfinite_streak >= 20:
                        raise RuntimeError(
                            f"Loss was nonfinite for {self._nonfinite_streak} consecutive steps "
                            f"(step {self.global_step}). The codebook EMA is guarded, so check the "
                            f"forward pass and inputs: batch_size, learning rate, and AMP dtype.")
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.global_step += 1
                    continue
                self._nonfinite_streak = 0
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1

                if self.global_step % self.config.log_every_steps == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    metric_text = " ".join(
                        f"{name}={value.item():.4f}" for name, value in outputs["metrics"].items()
                    )
                    self._log(
                        f"[epoch {epoch + 1} step {self.global_step}] "
                        f"loss={loss.item():.4f} lr={lr:.2e} {metric_text}"
                    )

                if self.global_step % self.config.save_every_steps == 0:
                    if self.distributed:
                        dist.barrier()
                    path = self.save_checkpoint(epoch)
                    if path is not None:
                        self._log(f"saved checkpoint: {path}")

                if (
                    self.val_dataloader is not None
                    and self.config.eval_every_steps > 0
                    and self.global_step % self.config.eval_every_steps == 0
                ):
                    metrics = self.evaluate()
                    metric_text = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    self._log(f"[eval step {self.global_step}] {metric_text}")
                    self._maybe_save_best(epoch, metrics)

            if self.distributed:
                dist.barrier()
            path = self.save_checkpoint(epoch)
            if path is not None:
                self._log(f"finished epoch {epoch + 1}, checkpoint: {path}")
            if self.val_dataloader is not None:
                metrics = self.evaluate()
                metric_text = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                self._log(f"[eval epoch {epoch + 1}] {metric_text}")
                self._maybe_save_best(epoch, metrics)
