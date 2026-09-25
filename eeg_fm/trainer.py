"""Distributed training loop for backbone pretraining with bf16 autocast, validation and checkpointing."""

import math
import os
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR

from eeg_fm.pretrain_config import PretrainConfig
from eeg_fm.pretraining import EEGFMPretrainer


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
    if getattr(param, "_no_weight_decay", False):
        return True
    if param.ndim <= 1:
        return True
    return False


def _build_param_groups(model: EEGFMPretrainer, config: PretrainConfig) -> list[dict]:
    frontend_names = {"signal_patch_encoder", "metadata_encoder", "spatial_mixer"}
    buckets: dict[tuple[bool, bool], list] = {
        (False, False): [], (False, True): [], (True, False): [], (True, True): [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_frontend = any(fn in name for fn in frontend_names)
        buckets[(is_frontend, _is_no_decay(name, param))].append(param)

    groups = []
    for (is_frontend, no_decay), params in buckets.items():
        if not params:
            continue
        groups.append({
            "params": params,
            "lr": config.learning_rate * (config.frontend_lr_scale if is_frontend else 1.0),
            "weight_decay": 0.0 if no_decay else config.weight_decay,
            "name": f"{'frontend' if is_frontend else 'backbone'}_{'nodecay' if no_decay else 'decay'}",
        })
    return groups


def _build_scheduler(optimizer: torch.optim.Optimizer, config: PretrainConfig, total_steps: int) -> LambdaLR:
    warmup = config.warmup_steps
    min_ratio = config.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return max(min_ratio, step / max(1, warmup))
        progress = (step - warmup) / max(1, total_steps - warmup)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


class PretrainTrainer:
    def __init__(
        self,
        model: EEGFMPretrainer,
        dataloader,
        config: PretrainConfig,
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
                static_graph=True,
            )
        else:
            self.model = self.raw_model

        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        param_groups = _build_param_groups(self.raw_model, config)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=config.weight_decay,
        )
        self.total_steps = config.max_epochs * len(dataloader)
        self.scheduler = _build_scheduler(self.optimizer, config, self.total_steps)
        self.amp_dtype = self._resolve_amp_dtype()
        self.scaler = torch.amp.GradScaler(enabled=self.amp_dtype == torch.float16)
        self.global_step = 0
        self.best_val_loss: float | None = None

    def _assert_grad_coverage(self) -> None:
        if getattr(self, "_grad_coverage_checked", False):
            return
        self._grad_coverage_checked = True

        trainable = [(n, p) for n, p in self.raw_model.named_parameters() if p.requires_grad]
        in_optimizer = {id(p) for g in self.optimizer.param_groups for p in g["params"]}

        missing_grad = [(n, p) for n, p in trainable if p.grad is None]
        missing_opt = [(n, p) for n, p in trainable if id(p) not in in_optimizer]

        problems = []
        total = sum(p.numel() for _, p in trainable)
        if missing_grad:
            dead = sum(p.numel() for _, p in missing_grad)
            problems.append(
                f"· {len(missing_grad)}/{len(trainable)} parameters have no gradient "
                f"({dead:,}/{total:,} = {100 * dead / total:.1f}%).\n    "
                + "\n    ".join(n for n, _ in missing_grad[:20])
            )
        if missing_opt:
            dead = sum(p.numel() for _, p in missing_opt)
            problems.append(
                f"· {len(missing_opt)}/{len(trainable)} parameters are missing from the optimizer "
                f"({dead:,}/{total:,} = {100 * dead / total:.1f}%).\n    "
                + "\n    ".join(n for n, _ in missing_opt[:20])
            )
        if problems:
            raise RuntimeError("Gradient coverage check failed:\n" + "\n".join(problems))
        self._log(
            f"[grad-check] all {len(trainable)} trainable parameters "
            f"({total:,} values) have gradients and optimizer coverage"
        )

    def _resolve_amp_dtype(self):
        if not self.config.amp or self.device.type != "cuda":
            return None
        name = str(getattr(self.config, "amp_dtype", "bf16")).lower()
        if name in ("fp32", "float32", "none"):
            return None
        if name in ("fp16", "float16", "half"):
            return torch.float16
        if name in ("bf16", "bfloat16"):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "amp_dtype='bf16' requires a GPU with bf16 support; "
                    "select fp16 or fp32 explicitly."
                )
            return torch.bfloat16
        raise ValueError(f"Unsupported amp_dtype: {self.config.amp_dtype!r}")

    def _autocast_context(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _save_config(self) -> None:
        if not is_main_process():
            return
        config_path = self.output_dir / "pretrain_config.snapshot.json"
        config_path.write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    def _build_checkpoint(self, epoch: int) -> dict:
        checkpoint = {
            "student": self.raw_model.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
        }
        return checkpoint

    def save_checkpoint(self, epoch: int, *, tag: str | None = None) -> Path | None:
        if not is_main_process():
            return None
        checkpoint = self._build_checkpoint(epoch)
        checkpoint_path = self.checkpoint_dir / (f"{tag}.pt" if tag else f"step_{self.global_step:08d}.pt")
        torch.save(checkpoint, checkpoint_path)
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> None:
        map_location = {"cuda:0": f"cuda:{self.local_rank}"} if self.device.type == "cuda" else "cpu"
        payload = torch.load(path, map_location=map_location)
        self.raw_model.model.load_state_dict(payload["student"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
        if "scaler" in payload:
            self.scaler.load_state_dict(payload["scaler"])
        self.global_step = int(payload["global_step"])
        self.best_val_loss = payload.get("best_val_loss")

    def warm_start(self, path: str | Path) -> None:
        map_location = {"cuda:0": f"cuda:{self.local_rank}"} if self.device.type == "cuda" else "cpu"
        payload = torch.load(path, map_location=map_location)
        self.raw_model.model.load_state_dict(payload["student"])
        self._log(
            f"[trainer] warm-started weights from {path}; "
            "optimizer, step, epoch, and schedule remain reset"
        )

    def _log(self, msg: str) -> None:
        if is_main_process():
            print(msg, flush=True)

    def _set_dataloader_epoch(self, epoch: int) -> None:
        self.dataloader.batch_sampler.set_epoch(epoch)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if self.val_dataloader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        for batch in self.val_dataloader:
            chunks = batch["chunks"].to(self.device, non_blocking=True)
            metadata = batch["metadata"].to(self.device)
            with self._autocast_context():
                outputs = self.model(chunks, metadata)
                loss = outputs["loss"]
            total_loss += float(loss.detach().item())
            total_batches += 1

        if self.distributed:
            packed = torch.tensor(
                [total_loss, float(total_batches)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            total_loss = float(packed[0].item())
            total_batches = int(packed[1].item())

        denom = max(1, total_batches)
        metrics = {"val_loss": total_loss / denom}
        self.model.train()
        return metrics

    def fit(self) -> None:
        set_random_seed(self.config.seed + self.rank)
        self._save_config()
        if self.config.resume_from is not None:
            self.load_checkpoint(self.config.resume_from)
            if self.distributed:
                dist.barrier()
        elif self.config.warm_start_from is not None:
            self.warm_start(self.config.warm_start_from)
            if self.distributed:
                dist.barrier()

        self._log(
            f"[trainer] distributed={self.distributed} world_size={self.world_size} "
            f"rank={self.rank} device={self.device}"
        )

        if self.global_step >= self.total_steps:
            self._log(
                f"[trainer] global_step={self.global_step} >= total_steps={self.total_steps}, "
                "training is already complete"
            )
            return

        steps_per_epoch = max(len(self.dataloader), 1)
        epoch = self.global_step // steps_per_epoch
        while self.global_step < self.total_steps:
            self._set_dataloader_epoch(epoch)

            self.model.train()
            for batch in self.dataloader:
                chunks = batch["chunks"].to(self.device, non_blocking=True)
                metadata = batch["metadata"].to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    outputs = self.model(chunks, metadata)
                    loss = outputs["loss"]
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self._assert_grad_coverage()
                clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1

                if self.global_step % self.config.log_every_steps == 0:
                    current_lrs = [f"{group['lr']:.2e}" for group in self.optimizer.param_groups]
                    self._log(
                        f"[epoch {epoch + 1} step {self.global_step}] "
                        f"loss={loss.item():.4f} lr=[{','.join(current_lrs)}]"
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
                    val_loss = metrics["val_loss"]
                    if self.best_val_loss is None or val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        if self.distributed:
                            dist.barrier()
                        path = self.save_checkpoint(epoch, tag="best")
                        if path is not None:
                            self._log(f"new best val_loss={val_loss:.4f}, checkpoint: {path}")

                if self.global_step >= self.total_steps:
                    break

            if self.distributed:
                dist.barrier()
            path = self.save_checkpoint(epoch)
            if path is not None:
                self._log(
                    f"finished epoch {epoch + 1} (step {self.global_step}/{self.total_steps}), "
                    f"checkpoint: {path}"
                )
            if self.val_dataloader is not None:
                metrics = self.evaluate()
                metric_text = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                self._log(f"[eval epoch {epoch + 1}] {metric_text}")
                val_loss = metrics["val_loss"]
                if self.best_val_loss is None or val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.distributed:
                        dist.barrier()
                    path = self.save_checkpoint(epoch, tag="best")
                    if path is not None:
                        self._log(f"new best val_loss={val_loss:.4f}, checkpoint: {path}")
            epoch += 1
