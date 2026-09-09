"""Deterministic data partitions and resumable training-state helpers."""

from __future__ import annotations

import json
import os
import random
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sg_jepa.data import (
    TrajectoryWindowDataset,
    open_trajectory_store,
    split_development_episodes,
)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if not state:
        raise ValueError("resume checkpoint has no RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        cuda_state = list(state.get("torch_cuda", []))
        if len(cuda_state) != torch.cuda.device_count():
            raise ValueError("resume checkpoint CUDA topology differs")
        torch.cuda.set_rng_state_all(cuda_state)


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError("training checkpoint must contain a mapping")
    return value


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def partition_window_datasets(
    dataset_path: str | Path,
    *,
    num_steps: int,
    train_fraction: float,
    seed: int,
) -> tuple[TrajectoryWindowDataset, TrajectoryWindowDataset, dict[str, Any]]:
    """Create leakage-free train/validation windows and train-only statistics."""

    store = open_trajectory_store(dataset_path)
    split = split_development_episodes(
        store,
        train_fraction=float(train_fraction),
        seed=int(seed),
    )
    train = TrajectoryWindowDataset(
        dataset_path,
        split="train",
        num_steps=int(num_steps),
        store=store,
        episode_indices=split.train_indices,
    )
    validation = TrajectoryWindowDataset(
        dataset_path,
        split="train",
        num_steps=int(num_steps),
        action_statistics=train.action_statistics,
        store=store,
        episode_indices=split.val_indices,
    )
    summary = split.to_dict()
    summary.update(
        {
            "train_window_count": len(train),
            "val_window_count": len(validation),
        }
    )
    return train, validation, summary


def epoch_loader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    num_workers: int,
    device: torch.device,
    drop_last: bool = False,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": min(int(batch_size), len(dataset)),
        "shuffle": bool(shuffle),
        "drop_last": bool(drop_last),
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
        "generator": torch.Generator().manual_seed(int(seed) + int(epoch)),
    }
    if num_workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
            multiprocessing_context="spawn",
        )
    return DataLoader(dataset, **kwargs)


__all__ = [
    "atomic_json",
    "atomic_torch_save",
    "epoch_loader",
    "load_training_checkpoint",
    "partition_window_datasets",
    "restore_rng_state",
    "rng_state",
    "seed_everything",
]
