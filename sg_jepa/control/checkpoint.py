"""Resumable Diffusion Policy checkpoints and top-k validation retention."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sg_jepa.train_utils import atomic_json, atomic_torch_save, load_training_checkpoint

from .ema import EMAModel

CHECKPOINT_FORMAT_VERSION = 2


def save_policy_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    policy_config: dict[str, Any],
    training_config: dict[str, Any],
    data_contract: dict[str, Any],
    world_model: dict[str, Any],
    evaluation: dict[str, Any],
    metrics: dict[str, float | None],
    training_state: dict[str, Any],
) -> Path:
    destination = Path(path).expanduser().resolve()
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": "diffusion_policy_training",
        "step": int(step),
        "policy_state_dict": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "policy_config": policy_config,
        "training_config": training_config,
        "data_contract": data_contract,
        "world_model": world_model,
        "evaluation": evaluation,
        "metrics": metrics,
        "training_state": training_state,
        "inference_only": False,
    }
    atomic_torch_save(payload, destination)
    return destination


def load_policy_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = load_training_checkpoint(path)
    if payload.get("kind") != "diffusion_policy_training":
        raise ValueError("resume checkpoint is not a public Diffusion Policy training checkpoint")
    if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported Diffusion Policy training-checkpoint version")
    if payload.get("inference_only"):
        raise ValueError("an inference-only policy cannot resume training")
    return payload


def export_ema_policy(source: str | Path, destination: str | Path) -> Path:
    payload = load_policy_checkpoint(source)
    ema = payload.get("ema")
    if not isinstance(ema, dict) or not isinstance(ema.get("shadow"), dict):
        raise ValueError("training checkpoint does not contain EMA weights")
    exported = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": "diffusion_policy",
        "step": int(payload["step"]),
        "policy_state_dict": ema["shadow"],
        "policy_config": payload["policy_config"],
        "data_contract": payload["data_contract"],
        "world_model": payload["world_model"],
        "evaluation": payload["evaluation"],
        "metrics": payload["metrics"],
        "inference_only": True,
    }
    target = Path(destination).expanduser().resolve()
    atomic_torch_save(exported, target)
    return target


def replace_alias(source: str | Path, destination: str | Path) -> None:
    """Atomically update a same-filesystem checkpoint alias."""

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(f".{destination_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(source_path, temporary)
        except OSError:
            shutil.copy2(source_path, temporary)
        os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)


class TopKCheckpointManager:
    def __init__(self, directory: str | Path, *, k: int) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.k = int(k)
        self.index_path = self.directory / "top_k.json"

    def entries(self) -> list[dict[str, Any]]:
        if not self.index_path.is_file():
            return []
        payload = json.loads(self.index_path.read_text())
        return [
            entry
            for entry in payload.get("checkpoints", [])
            if (self.directory / entry["file"]).is_file()
        ]

    def consider(self, checkpoint: str | Path, *, validation_loss: float, step: int) -> bool:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if checkpoint.parent != self.directory:
            raise ValueError("top-k checkpoint must be inside its manager directory")
        entries = [entry for entry in self.entries() if entry["file"] != checkpoint.name]
        entries.append(
            {
                "file": checkpoint.name,
                "validation_loss": float(validation_loss),
                "step": int(step),
            }
        )
        entries.sort(key=lambda entry: (entry["validation_loss"], entry["step"]))
        kept = entries[: self.k]
        for entry in entries[self.k :]:
            candidate = self.directory / entry["file"]
            if re.fullmatch(r"policy_step_\d{6,9}\.pt", candidate.name):
                candidate.unlink(missing_ok=True)
        atomic_json(
            {"metric": "validation_loss", "mode": "min", "checkpoints": kept},
            self.index_path,
        )
        return checkpoint.name in {entry["file"] for entry in kept}


__all__ = [
    "TopKCheckpointManager",
    "export_ema_policy",
    "load_policy_checkpoint",
    "replace_alias",
    "save_policy_checkpoint",
]
