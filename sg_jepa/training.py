"""Paper-aligned trainer shared by Original LeWM and Semigroup-JEPA."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from sg_jepa.config import ExperimentConfig
from sg_jepa.losses import SIGReg, compute_objective
from sg_jepa.models import build_world_model
from sg_jepa.optim import MuonAdamW, split_named_parameters
from sg_jepa.train_utils import (
    atomic_json,
    atomic_torch_save,
    epoch_loader,
    load_training_checkpoint,
    partition_window_datasets,
    restore_rng_state,
    rng_state,
    seed_everything,
)


def _optimizer(model: torch.nn.Module, config: ExperimentConfig) -> MuonAdamW:
    muon, adamw = split_named_parameters(model.named_parameters())
    groups: list[dict[str, Any]] = []
    if muon:
        groups.append(
            {
                "params": [parameter for _, parameter in muon],
                "use_muon": True,
                "lr": config.train.muon_learning_rate,
                "weight_decay": config.train.weight_decay,
                "momentum": config.train.muon_momentum,
                "nesterov": True,
                "ns_steps": config.train.muon_ns_steps,
                "lr_adjustment": "match_rms_adamw",
                "rms_match_scale": config.train.muon_rms_match_scale,
            }
        )
    if adamw:
        groups.append(
            {
                "params": [parameter for _, parameter in adamw],
                "use_muon": False,
                "lr": config.train.learning_rate,
                "weight_decay": config.train.weight_decay,
                "betas": (0.9, 0.999),
                "eps": 1.0e-8,
            }
        )
    return MuonAdamW(groups)


class _WarmupCosineCooldown:
    """Historical v10 linear warmup, plateau, then cosine cooldown."""

    def __init__(
        self,
        *,
        total_steps: int,
        warmup_steps: int,
        cooldown_start_fraction: float,
        min_factor: float,
    ) -> None:
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.cooldown_start_step = int(round(self.total_steps * float(cooldown_start_fraction)))
        self.cooldown_start_step = min(
            max(self.warmup_steps, self.cooldown_start_step),
            max(0, self.total_steps - 1),
        )
        self.min_factor = min(max(float(min_factor), 0.0), 1.0)

    def __call__(self, step: int) -> float:
        step = int(step)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return min(1.0, max(0.0, (step + 1) / self.warmup_steps))
        if step <= self.cooldown_start_step:
            return 1.0
        denominator = max(1, self.total_steps - self.cooldown_start_step)
        progress = min(
            1.0,
            max(0.0, (step - self.cooldown_start_step) / denominator),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_factor + (1.0 - self.min_factor) * cosine


def _scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    *,
    total_steps: int,
    updates_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if config.train.lr_schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda _step: 1.0,
        )
    schedule = _WarmupCosineCooldown(
        total_steps=total_steps,
        warmup_steps=int(round(updates_per_epoch * config.train.lr_warmup_epochs)),
        cooldown_start_fraction=config.train.lr_cooldown_start_fraction,
        min_factor=config.train.lr_min_factor,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
        if torch.is_tensor(value)
    }


def _autocast(device: torch.device, precision: str):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bf16" and device.type == "cuda",
    )


@torch.inference_mode()
def _validate(
    model: torch.nn.Module,
    sigreg: SIGReg,
    dataset: torch.utils.data.Dataset,
    config: ExperimentConfig,
    *,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    loader = epoch_loader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        seed=config.train.seed,
        epoch=epoch,
        num_workers=config.train.num_workers,
        device=device,
        drop_last=config.train.drop_last,
    )
    totals: dict[str, float] = defaultdict(float)
    samples = 0
    model.eval()
    for batch_index, batch in enumerate(loader):
        if config.train.max_val_batches is not None and batch_index >= config.train.max_val_batches:
            break
        tensor_batch = _move_batch(batch, device)
        batch_size = int(tensor_batch["pixels"].shape[0])
        with _autocast(device, config.train.precision):
            losses = compute_objective(model, tensor_batch, config.objective, sigreg)
        for name, value in losses.items():
            totals[name] += float(value.detach().cpu()) * batch_size
        samples += batch_size
    if samples == 0:
        raise RuntimeError("validation loader produced no batches")
    model.train()
    return {name: total / samples for name, total in totals.items()}


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: ExperimentConfig,
    action_statistics: dict[str, Any],
    data_split: dict[str, Any],
    step: int,
    next_epoch: int,
    next_microbatch: int,
    completed_epochs: int,
    best_validation_loss: float | None,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "kind": "native_world_model_training",
        "step": int(step),
        "cursor": {
            "next_epoch": int(next_epoch),
            "next_microbatch": int(next_microbatch),
            "completed_epochs": int(completed_epochs),
        },
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config.to_dict(),
        "action_statistics": action_statistics,
        "data_split": data_split,
        "best_validation_loss": best_validation_loss,
        "history": history,
        "rng_state": rng_state(),
    }


def run_training(
    config: ExperimentConfig,
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    resume: str | Path | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Train with an episode split, validation, accumulation, and exact resume."""

    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    seed_everything(config.train.seed)
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    required_frames = config.model.history_size + max(1, config.objective.rollout_horizon)
    train_dataset, val_dataset, data_split = partition_window_datasets(
        dataset_path,
        num_steps=required_frames,
        train_fraction=config.train.train_fraction,
        seed=config.train.seed,
    )
    if train_dataset.action_dim != config.model.action_dim:
        raise ValueError(
            f"dataset action_dim={train_dataset.action_dim} but config expects "
            f"{config.model.action_dim}"
        )
    model = build_world_model(config.model).to(selected_device)
    sigreg = SIGReg(config.objective.sigreg_knots, config.objective.sigreg_num_proj).to(
        selected_device
    )
    optimizer = _optimizer(model, config)
    batch_size = min(config.train.batch_size, len(train_dataset))
    if config.train.drop_last:
        microbatches_per_epoch = len(train_dataset) // batch_size
    else:
        microbatches_per_epoch = math.ceil(len(train_dataset) / batch_size)
    if microbatches_per_epoch == 0:
        raise ValueError("training dataset does not contain one complete batch")
    updates_per_epoch = math.ceil(microbatches_per_epoch / config.train.gradient_accumulation_steps)
    total_steps = config.train.epochs * updates_per_epoch
    if config.train.max_steps is not None:
        total_steps = min(total_steps, config.train.max_steps)
    scheduler = _scheduler(
        optimizer,
        config,
        total_steps=total_steps,
        updates_per_epoch=updates_per_epoch,
    )
    start_step = 0
    next_epoch = 0
    next_microbatch = 0
    completed_epochs = 0
    best_validation_loss: float | None = None
    history: list[dict[str, Any]] = []
    if resume is not None:
        checkpoint = load_training_checkpoint(resume)
        if checkpoint.get("schema_version") != 2 or checkpoint.get("kind") != (
            "native_world_model_training"
        ):
            raise ValueError("resume requires a schema-v2 native training checkpoint")
        if checkpoint.get("config") != config.to_dict():
            raise ValueError("resume checkpoint configuration differs")
        if checkpoint.get("data_split") != data_split:
            raise ValueError("resume checkpoint episode split differs")
        if checkpoint.get("action_statistics") != train_dataset.action_statistics.to_dict():
            raise ValueError("resume checkpoint action statistics differ")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" not in checkpoint:
            raise ValueError("resume checkpoint has no learning-rate scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler"])
        cursor = checkpoint["cursor"]
        start_step = int(checkpoint["step"])
        next_epoch = int(cursor["next_epoch"])
        next_microbatch = int(cursor["next_microbatch"])
        completed_epochs = int(cursor["completed_epochs"])
        stored_best = checkpoint.get("best_validation_loss")
        best_validation_loss = None if stored_best is None else float(stored_best)
        history = [dict(row) for row in checkpoint.get("history", [])]
        restore_rng_state(checkpoint["rng_state"])

    stop_step = total_steps if max_steps is None else min(total_steps, start_step + max_steps)
    if start_step > total_steps:
        raise ValueError("resume step exceeds the configured training schedule")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(data_split, output / "train_val_split.json")
    atomic_json(train_dataset.action_statistics.to_dict(), output / "normalization_stats.json")
    started = time.perf_counter()
    global_step = start_step
    model.train()
    stop_requested = global_step >= stop_step
    for epoch in range(next_epoch, config.train.epochs):
        if stop_requested:
            break
        loader = epoch_loader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            seed=config.train.seed,
            epoch=epoch,
            num_workers=config.train.num_workers,
            device=selected_device,
            drop_last=config.train.drop_last,
        )
        iterator = iter(loader)
        batch_index = 0
        if epoch == next_epoch and next_microbatch:
            if next_microbatch >= len(loader):
                raise ValueError("resume microbatch cursor exceeds the epoch")
            for _ in range(next_microbatch):
                next(iterator)
            batch_index = next_microbatch
        while batch_index < len(loader) and global_step < stop_step:
            optimizer.zero_grad(set_to_none=True)
            pending = min(
                config.train.gradient_accumulation_steps,
                len(loader) - batch_index,
            )
            step_totals: dict[str, float] = defaultdict(float)
            for _ in range(pending):
                batch = next(iterator)
                tensor_batch = _move_batch(batch, selected_device)
                with _autocast(selected_device, config.train.precision):
                    losses = compute_objective(model, tensor_batch, config.objective, sigreg)
                loss = losses["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss before step {global_step + 1}")
                (loss / pending).backward()
                for name, value in losses.items():
                    step_totals[name] += float(value.detach().cpu()) / pending
                batch_index += 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.gradient_clip
            )
            learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
            optimizer.step()
            scheduler.step()
            global_step += 1
            cursor_epoch = epoch + 1 if batch_index == len(loader) else epoch
            cursor_batch = 0 if batch_index == len(loader) else batch_index
            history.append(
                {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "learning_rates": learning_rates,
                    **dict(step_totals),
                }
            )
            next_epoch, next_microbatch = cursor_epoch, cursor_batch

        epoch_complete = batch_index == len(loader)
        if epoch_complete:
            completed_epochs = epoch + 1
            validation = _validate(
                model,
                sigreg,
                val_dataset,
                config,
                device=selected_device,
                epoch=epoch,
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "split": "validation",
                    **{f"validation_{name}": value for name, value in validation.items()},
                }
            )
            validation_loss = validation["loss"]
            if best_validation_loss is None or validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                action_statistics=train_dataset.action_statistics.to_dict(),
                data_split=data_split,
                step=global_step,
                next_epoch=next_epoch,
                next_microbatch=next_microbatch,
                completed_epochs=completed_epochs,
                best_validation_loss=best_validation_loss,
                history=history,
            )
            if validation_loss == best_validation_loss:
                atomic_torch_save(payload, output / "best.pt")
            if completed_epochs % config.train.checkpoint_every_epochs == 0:
                atomic_torch_save(
                    payload,
                    output / f"checkpoint_epoch_{completed_epochs:03d}.pt",
                )
        stop_requested = global_step >= stop_step

    final_payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        action_statistics=train_dataset.action_statistics.to_dict(),
        data_split=data_split,
        step=global_step,
        next_epoch=next_epoch,
        next_microbatch=next_microbatch,
        completed_epochs=completed_epochs,
        best_validation_loss=best_validation_loss,
        history=history,
    )
    checkpoint_path = output / "checkpoint.pt"
    atomic_torch_save(final_payload, checkpoint_path)
    reloaded = build_world_model(config.model)
    reloaded.load_state_dict(load_training_checkpoint(checkpoint_path)["model"], strict=True)
    report: dict[str, Any] = {
        "schema_version": 2,
        "kind": "training",
        "config": config.name,
        "device": str(selected_device),
        "steps_before": start_step,
        "steps_after": global_step,
        "configured_epochs": config.train.epochs,
        "completed_epochs": completed_epochs,
        "steps_this_run": global_step - start_step,
        "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
        "lr_schedule": config.train.lr_schedule,
        "learning_rates_after": [float(value) for value in scheduler.get_last_lr()],
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": str(output / "best.pt") if (output / "best.pt").is_file() else None,
        "best_validation_loss": best_validation_loss,
        "data_split": data_split,
        "action_statistics": train_dataset.action_statistics.to_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "strict_reload": "pass",
        "status": "pass",
    }
    atomic_json(report, output / "train_report.json")
    return report


__all__ = ["run_training"]
