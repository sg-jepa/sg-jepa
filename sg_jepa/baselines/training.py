"""Paper-aligned training for the frozen-DINOv2 DINO-WM baseline."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import yaml

from sg_jepa.config import TrainConfig
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

from .dino_wm import PredictorConfig, build_dino_world_model
from .dinov2 import Native128Preprocessor, load_dinov2_encoder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dino_training_config(
    path: str | Path,
) -> tuple[str, PredictorConfig, TrainConfig, dict[str, Any]]:
    """Load the small public DINO-WM training schema."""

    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("DINO-WM training config requires schema_version=1")
    required = {"schema_version", "name", "predictor", "encoder", "train"}
    if set(payload) != required:
        raise ValueError(f"DINO-WM training config must contain exactly {sorted(required)}")
    predictor = payload["predictor"]
    train = payload["train"]
    encoder = payload["encoder"]
    if not all(isinstance(value, dict) for value in (predictor, train, encoder)):
        raise TypeError("predictor, encoder, and train must be mappings")
    if encoder.get("model") != "dinov2_vits14":
        raise ValueError("the released DINO-WM path supports only dinov2_vits14")
    allowed_train = {field.name for field in fields(TrainConfig)}
    if extra := set(train) - allowed_train:
        raise ValueError(f"unknown train keys: {sorted(extra)}")
    return (
        str(payload["name"]),
        PredictorConfig.from_dict(predictor),
        TrainConfig(**train),
        payload,
    )


def _raw_unit_pixels(normalized: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=normalized.dtype, device=normalized.device)[
        None, None, :, None, None
    ]
    std = torch.tensor(IMAGENET_STD, dtype=normalized.dtype, device=normalized.device)[
        None, None, :, None, None
    ]
    return (normalized * std + mean).clamp_(0.0, 1.0)


def _autocast(device: torch.device, precision: str):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bf16" and device.type == "cuda",
    )


def _forward_loss(
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    preprocessor: Native128Preprocessor,
    batch: dict[str, Any],
    predictor_config: PredictorConfig,
    *,
    device: torch.device,
    precision: str,
):
    pixels = batch["pixels"].to(device, non_blocking=device.type == "cuda")
    actions = batch["action"].to(device, non_blocking=device.type == "cuda")
    batch_size, frames = pixels.shape[:2]
    unit_pixels = _raw_unit_pixels(pixels).reshape(-1, *pixels.shape[2:])
    with torch.inference_mode():
        encoder_pixels = preprocessor(unit_pixels).to(device)
        tokens = (
            encoder(encoder_pixels)
            .float()
            .reshape(
                batch_size,
                frames,
                predictor_config.num_patches,
                predictor_config.visual_dim,
            )
        )
    with _autocast(device, precision):
        return model.teacher_forced_loss(tokens, actions)


@torch.inference_mode()
def _validate(
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    preprocessor: Native128Preprocessor,
    dataset: torch.utils.data.Dataset,
    predictor_config: PredictorConfig,
    train_config: TrainConfig,
    *,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    loader = epoch_loader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        seed=train_config.seed,
        epoch=epoch,
        num_workers=train_config.num_workers,
        device=device,
    )
    totals: dict[str, float] = defaultdict(float)
    samples = 0
    model.eval()
    for batch_index, batch in enumerate(loader):
        if train_config.max_val_batches is not None and batch_index >= train_config.max_val_batches:
            break
        result = _forward_loss(
            model,
            encoder,
            preprocessor,
            batch,
            predictor_config,
            device=device,
            precision=train_config.precision,
        )
        batch_size = int(batch["pixels"].shape[0])
        for name, value in {
            "loss": result.loss,
            "visual_loss": result.visual_loss,
            "proprio_loss": result.proprio_loss,
        }.items():
            totals[name] += float(value.detach().cpu()) * batch_size
        samples += batch_size
    if samples == 0:
        raise RuntimeError("validation loader produced no DINO-WM batches")
    model.train()
    return {name: total / samples for name, total in totals.items()}


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    raw_config: dict[str, Any],
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
        "kind": "dino_world_model_training",
        "step": int(step),
        "cursor": {
            "next_epoch": int(next_epoch),
            "next_microbatch": int(next_microbatch),
            "completed_epochs": int(completed_epochs),
        },
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": raw_config,
        "action_statistics": action_statistics,
        "data_split": data_split,
        "best_validation_loss": best_validation_loss,
        "history": history,
        "rng_state": rng_state(),
    }


def run_dino_training(
    config_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    dinov2_root: str | Path,
    device: str = "cuda",
    resume: str | Path | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Train and validate the predictor while keeping DINOv2 frozen."""

    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    name, predictor_config, train_config, raw_config = load_dino_training_config(config_path)
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(train_config.seed)
    train_dataset, val_dataset, data_split = partition_window_datasets(
        dataset_path,
        num_steps=predictor_config.history_size + 1,
        train_fraction=train_config.train_fraction,
        seed=train_config.seed,
    )
    if train_dataset.action_dim != predictor_config.action_dim:
        raise ValueError(
            f"dataset action_dim={train_dataset.action_dim} but config expects "
            f"{predictor_config.action_dim}"
        )
    encoder = load_dinov2_encoder(dinov2_root, device=selected_device)
    preprocessor = Native128Preprocessor()
    model = build_dino_world_model(predictor_config).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
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
            "dino_world_model_training"
        ):
            raise ValueError("resume requires a schema-v2 DINO-WM training checkpoint")
        if checkpoint.get("config") != raw_config:
            raise ValueError("resume checkpoint configuration differs")
        if checkpoint.get("data_split") != data_split:
            raise ValueError("resume checkpoint episode split differs")
        if checkpoint.get("action_statistics") != train_dataset.action_statistics.to_dict():
            raise ValueError("resume checkpoint action statistics differ")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        cursor = checkpoint["cursor"]
        start_step = int(checkpoint["step"])
        next_epoch = int(cursor["next_epoch"])
        next_microbatch = int(cursor["next_microbatch"])
        completed_epochs = int(cursor["completed_epochs"])
        stored_best = checkpoint.get("best_validation_loss")
        best_validation_loss = None if stored_best is None else float(stored_best)
        history = [dict(row) for row in checkpoint.get("history", [])]
        restore_rng_state(checkpoint["rng_state"])

    batch_size = min(train_config.batch_size, len(train_dataset))
    microbatches_per_epoch = math.ceil(len(train_dataset) / batch_size)
    updates_per_epoch = math.ceil(microbatches_per_epoch / train_config.gradient_accumulation_steps)
    total_steps = train_config.epochs * updates_per_epoch
    if train_config.max_steps is not None:
        total_steps = min(total_steps, train_config.max_steps)
    stop_step = total_steps if max_steps is None else min(total_steps, start_step + max_steps)
    if start_step > total_steps:
        raise ValueError("resume step exceeds the configured DINO-WM schedule")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(data_split, output / "train_val_split.json")
    atomic_json(train_dataset.action_statistics.to_dict(), output / "normalization_stats.json")
    started = time.perf_counter()
    global_step = start_step
    model.train()
    stop_requested = global_step >= stop_step
    for epoch in range(next_epoch, train_config.epochs):
        if stop_requested:
            break
        loader = epoch_loader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            seed=train_config.seed,
            epoch=epoch,
            num_workers=train_config.num_workers,
            device=selected_device,
        )
        iterator = iter(loader)
        batch_index = 0
        if epoch == next_epoch and next_microbatch:
            if next_microbatch >= len(loader):
                raise ValueError("resume microbatch cursor exceeds the DINO-WM epoch")
            for _ in range(next_microbatch):
                next(iterator)
            batch_index = next_microbatch
        while batch_index < len(loader) and global_step < stop_step:
            optimizer.zero_grad(set_to_none=True)
            pending = min(
                train_config.gradient_accumulation_steps,
                len(loader) - batch_index,
            )
            totals = {"loss": 0.0, "visual_loss": 0.0, "proprio_loss": 0.0}
            for _ in range(pending):
                batch = next(iterator)
                result = _forward_loss(
                    model,
                    encoder,
                    preprocessor,
                    batch,
                    predictor_config,
                    device=selected_device,
                    precision=train_config.precision,
                )
                if not torch.isfinite(result.loss):
                    raise FloatingPointError(
                        f"non-finite DINO-WM loss before step {global_step + 1}"
                    )
                (result.loss / pending).backward()
                totals["loss"] += float(result.loss.detach().cpu()) / pending
                totals["visual_loss"] += float(result.visual_loss.detach().cpu()) / pending
                totals["proprio_loss"] += float(result.proprio_loss.detach().cpu()) / pending
                batch_index += 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.gradient_clip
            )
            optimizer.step()
            global_step += 1
            next_epoch = epoch + 1 if batch_index == len(loader) else epoch
            next_microbatch = 0 if batch_index == len(loader) else batch_index
            history.append(
                {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    **totals,
                }
            )

        if batch_index == len(loader):
            completed_epochs = epoch + 1
            validation = _validate(
                model,
                encoder,
                preprocessor,
                val_dataset,
                predictor_config,
                train_config,
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
                raw_config=raw_config,
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
            if completed_epochs % train_config.checkpoint_every_epochs == 0:
                atomic_torch_save(
                    payload,
                    output / f"checkpoint_epoch_{completed_epochs:03d}.pt",
                )
        stop_requested = global_step >= stop_step

    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        raw_config=raw_config,
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
    atomic_torch_save(payload, checkpoint_path)
    reloaded = build_dino_world_model(predictor_config)
    reloaded.load_state_dict(
        load_training_checkpoint(checkpoint_path)["model_state_dict"],
        strict=True,
    )
    report = {
        "schema_version": 2,
        "kind": "dino_wm_training",
        "config": name,
        "device": str(selected_device),
        "steps_before": start_step,
        "steps_after": global_step,
        "steps_this_run": global_step - start_step,
        "completed_epochs": completed_epochs,
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": str(output / "best.pt") if (output / "best.pt").is_file() else None,
        "best_validation_loss": best_validation_loss,
        "data_split": data_split,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "strict_reload": "pass",
        "status": "pass",
    }
    atomic_json(report, output / "train_report.json")
    return report


__all__ = ["load_dino_training_config", "run_dino_training"]
