"""Paper MLP state probes and symmetry-aware planar rollout metrics."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from sg_jepa.baselines import (
    Native128Preprocessor,
    build_dino_world_model,
    load_dino_config,
    load_dinov2_encoder,
    strict_load_dino,
)
from sg_jepa.checkpoints import load_state_dict, sha256_file, strict_load
from sg_jepa.config import load_experiment_config
from sg_jepa.data import (
    ActionStatistics,
    EpisodeSplit,
    TrajectoryWindowDataset,
    open_trajectory_store,
    split_development_episodes,
)
from sg_jepa.models import build_world_model
from sg_jepa.train_utils import (
    atomic_json,
    atomic_torch_save,
    load_training_checkpoint,
    restore_rng_state,
    rng_state,
    seed_everything,
)

from .frozen_approach import _encode_dino, _encode_native, _rollout_native

TASK_TARGETS: dict[str, tuple[int | None, tuple[str, ...], str]] = {
    "right_triangle": (
        1,
        ("x", "z", "vx", "vz", "sin1theta", "cos1theta", "omega"),
        "full_state_sym1",
    ),
    "square": (
        4,
        ("x", "z", "vx", "vz", "sin4theta", "cos4theta", "omega"),
        "full_state_sym4",
    ),
    "approach_ball": (
        None,
        ("x", "y", "z", "vx", "vy", "vz"),
        "ball3d_posvel",
    ),
}


@dataclass(frozen=True)
class ProbeTargetSpec:
    task: str
    symmetry_order: int | None
    target_names: tuple[str, ...]
    name: str

    @classmethod
    def for_task(cls, task: str) -> ProbeTargetSpec:
        canonical = str(task).strip().lower().replace("-", "_")
        try:
            order, names, name = TASK_TARGETS[canonical]
        except KeyError as exc:
            raise ValueError(f"unsupported probe task {task!r}: {sorted(TASK_TARGETS)}") from exc
        return cls(canonical, order, names, name)

    @property
    def output_dim(self) -> int:
        return len(self.target_names)


@dataclass(frozen=True)
class ProbeTrainConfig:
    temporal_window: int | None = None
    max_train_windows: int = 50_000
    max_val_windows: int = 50_000
    feature_batch_size: int = 32
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    max_epochs: int = 50
    patience: int = 3
    seed: int = 42
    num_workers: int = 0

    def __post_init__(self) -> None:
        positive = (
            self.feature_batch_size,
            self.batch_size,
            self.max_epochs,
            self.patience,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("probe batch sizes, epochs, and patience must be positive")
        if self.temporal_window is not None and self.temporal_window <= 0:
            raise ValueError("temporal_window must be positive")
        if self.max_train_windows == 0 or self.max_val_windows == 0:
            raise ValueError("probe window limits must be positive or negative for all")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.num_workers < 0:
            raise ValueError("invalid probe optimizer or worker configuration")


@dataclass(frozen=True)
class FrozenStateProbe:
    model: nn.Module
    target_spec: ProbeTargetSpec
    feature_dim: int
    temporal_window: int
    target_mean: torch.Tensor
    target_std: torch.Tensor

    @torch.inference_mode()
    def predict_features(self, features: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        normalized = self.model(features.to(device=device, dtype=torch.float32))
        return normalized * self.target_std + self.target_mean


def build_state_probe(feature_dim: int, output_dim: int) -> nn.Module:
    """LN-512-256 MLP used for the paper state-probe comparisons."""

    return nn.Sequential(
        nn.LayerNorm(int(feature_dim)),
        nn.Linear(int(feature_dim), 512),
        nn.GELU(),
        nn.Dropout(0.05),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.05),
        nn.Linear(256, int(output_dim)),
    )


def transform_state_targets(
    state: torch.Tensor | np.ndarray,
    spec: ProbeTargetSpec | str,
) -> torch.Tensor:
    target_spec = spec if isinstance(spec, ProbeTargetSpec) else ProbeTargetSpec.for_task(spec)
    values = torch.as_tensor(state, dtype=torch.float32)
    if values.shape[-1] < 6:
        raise ValueError(f"state must have at least six coordinates, got {tuple(values.shape)}")
    if target_spec.task == "approach_ball":
        return values[..., :6]
    base = [values[..., index] for index in range(4)]
    assert target_spec.symmetry_order is not None
    theta = values[..., 4]
    order = float(target_spec.symmetry_order)
    return torch.stack(
        (*base, torch.sin(order * theta), torch.cos(order * theta), values[..., 5]),
        dim=-1,
    )


def _task_from_payload(payload: dict[str, Any]) -> str:
    shape = payload.get("shape")
    if isinstance(shape, str):
        return shape
    target = payload.get("target_spec")
    aliases = {
        "full_state_sym1": "right_triangle",
        "full_state_sym4": "square",
        "ball3d_posvel": "approach_ball",
    }
    if isinstance(target, dict):
        target = target.get("name") or target.get("target_spec")
    try:
        return aliases[str(target)]
    except KeyError as exc:
        raise ValueError("probe artifact does not identify a supported target") from exc


def _is_dino_config(payload: Any) -> bool:
    return isinstance(payload, dict) and (
        "predictor" in payload
        or payload.get("kind") == "dino_world_model"
        or payload.get("checkpoint_kind") == "dino_world_model"
    )


def _load_dino_predictor(
    checkpoint_path: str | Path,
    config_path: str | Path,
) -> tuple[nn.Module, Any]:
    predictor_config, _ = load_dino_config(config_path)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if checkpoint.suffix == ".safetensors":
        model = build_dino_world_model(predictor_config)
        model.load_state_dict(load_state_dict(checkpoint), strict=True)
    else:
        model, _report = strict_load_dino(checkpoint, config_path)
    return model, predictor_config


def load_state_probe(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    config_path: str | Path | None = None,
) -> FrozenStateProbe:
    source = Path(path).expanduser().resolve()
    if source.suffix == ".safetensors":
        sidecar_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else source.with_suffix(".json")
        )
        sidecar = json.loads(sidecar_path.read_text())
        architecture = dict(sidecar["architecture"])
        payload = {
            **architecture,
            "target_stats": sidecar["target_stats"],
            "probe_state_dict": load_state_dict(source),
        }
    else:
        payload = load_training_checkpoint(source)
    if payload.get("probe_kind") != "mlp":
        raise ValueError("paper evaluation requires an MLP state probe")
    spec = ProbeTargetSpec.for_task(_task_from_payload(payload))
    feature_dim = int(payload["feature_dim"])
    temporal_window = int(payload["temporal_window"])
    output_dim = int(payload.get("output_dim", spec.output_dim))
    if output_dim != spec.output_dim:
        raise ValueError("probe output dimension differs from its target spec")
    stats = payload["target_stats"]
    if tuple(stats["target_definition"]) != spec.target_names:
        raise ValueError("probe target names differ from the paper contract")
    target_mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device)
    target_std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device)
    if target_mean.shape != (output_dim,) or target_std.shape != (output_dim,):
        raise ValueError("probe normalization statistics have the wrong shape")
    if not torch.isfinite(target_mean).all() or not torch.isfinite(target_std).all():
        raise ValueError("probe normalization statistics must be finite")
    if torch.any(target_std <= 0):
        raise ValueError("probe target standard deviations must be positive")
    model = build_state_probe(feature_dim, output_dim)
    model.load_state_dict(payload["probe_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return FrozenStateProbe(
        model=model,
        target_spec=spec,
        feature_dim=feature_dim,
        temporal_window=temporal_window,
        target_mean=target_mean,
        target_std=target_std,
    )


def _loader(features: torch.Tensor, targets: torch.Tensor, config: ProbeTrainConfig, epoch: int):
    return DataLoader(
        TensorDataset(features, targets),
        batch_size=min(config.batch_size, len(features)),
        shuffle=epoch > 0,
        generator=torch.Generator().manual_seed(config.seed + epoch),
        num_workers=config.num_workers,
        drop_last=False,
    )


@torch.inference_mode()
def _probe_loss(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    config: ProbeTrainConfig,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for x, y in _loader(features, targets, config, 0):
        x, y = x.to(device), y.to(device)
        loss = F.mse_loss(model(x), (y - mean) / std)
        total += float(loss) * len(x)
        count += len(x)
    if count == 0:
        raise ValueError("probe validation set is empty")
    return total / count


def fit_state_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    output_dir: str | Path,
    *,
    task: str,
    temporal_window: int,
    config: ProbeTrainConfig | None = None,
    provenance: dict[str, Any] | None = None,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Fit and select the paper MLP by validation normalized MSE."""

    config = config or ProbeTrainConfig(temporal_window=temporal_window)
    spec = ProbeTargetSpec.for_task(task)
    train_features = torch.as_tensor(train_features, dtype=torch.float32)
    val_features = torch.as_tensor(val_features, dtype=torch.float32)
    train_targets = torch.as_tensor(train_targets, dtype=torch.float32)
    val_targets = torch.as_tensor(val_targets, dtype=torch.float32)
    if train_features.ndim != 2 or val_features.ndim != 2:
        raise ValueError("probe features must be matrices")
    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError("train and validation feature dimensions differ")
    if train_targets.shape != (len(train_features), spec.output_dim):
        raise ValueError("training probe targets have the wrong shape")
    if val_targets.shape != (len(val_features), spec.output_dim):
        raise ValueError("validation probe targets have the wrong shape")
    mean = train_targets.mean(dim=0)
    std = train_targets.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    device_value = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if provenance is None or provenance.get("device") == "auto"
        else torch.device(provenance.get("device", "cpu"))
    )
    seed_everything(config.seed)
    model = build_state_probe(train_features.shape[1], spec.output_dim).to(device_value)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    start_epoch = 1
    best_epoch = 0
    best_loss: float | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    if resume is not None:
        checkpoint = load_training_checkpoint(resume)
        if checkpoint.get("kind") != "paper_mlp_state_probe_training":
            raise ValueError("resume checkpoint is not a paper MLP state probe")
        if checkpoint.get("train_config") != asdict(config):
            raise ValueError("probe resume configuration differs")
        if checkpoint.get("target_spec") != asdict(spec):
            raise ValueError("probe resume target differs")
        if not torch.equal(torch.as_tensor(checkpoint["target_mean"]), mean):
            raise ValueError("probe resume target mean differs")
        if not torch.equal(torch.as_tensor(checkpoint["target_std"]), std):
            raise ValueError("probe resume target std differs")
        model.load_state_dict(checkpoint["probe_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        stored_best = checkpoint.get("best_validation_normalized_mse")
        best_loss = None if stored_best is None else float(stored_best)
        stale = int(checkpoint["stale_epochs"])
        history = [dict(row) for row in checkpoint["history"]]
        restore_rng_state(checkpoint["rng_state"])

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    mean_device, std_device = mean.to(device_value), std.to(device_value)
    for epoch in range(start_epoch, config.max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for features, targets in _loader(train_features, train_targets, config, epoch):
            features, targets = features.to(device_value), targets.to(device_value)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(features), (targets - mean_device) / std_device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite probe loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(features)
            count += len(features)
        validation_loss = _probe_loss(
            model,
            val_features,
            val_targets,
            mean_device,
            std_device,
            config,
            device_value,
        )
        history.append(
            {
                "epoch": epoch,
                "train_normalized_mse": total / count,
                "validation_normalized_mse": validation_loss,
            }
        )
        improved = best_loss is None or validation_loss < best_loss
        if improved:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
        else:
            stale += 1
        checkpoint = {
            "schema_version": 2,
            "kind": "paper_mlp_state_probe_training",
            "probe_kind": "mlp",
            "feature_dim": int(train_features.shape[1]),
            "output_dim": spec.output_dim,
            "temporal_window": int(temporal_window),
            "target_spec": asdict(spec),
            "target_mean": mean,
            "target_std": std,
            "target_stats": {
                "target_definition": list(spec.target_names),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "normalization": "train_sample_zscore",
                "sample_count": len(train_targets),
            },
            "probe_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_normalized_mse": best_loss,
            "stale_epochs": stale,
            "train_config": asdict(config),
            "provenance": dict(provenance or {}),
            "history": history,
            "rng_state": rng_state(),
        }
        atomic_torch_save(checkpoint, output / "probe_latest.pt")
        if improved:
            atomic_torch_save(checkpoint, output / "probe_best.pt")
            inference = {
                key: checkpoint[key]
                for key in (
                    "kind",
                    "probe_kind",
                    "feature_dim",
                    "output_dim",
                    "temporal_window",
                    "target_spec",
                    "target_stats",
                    "probe_state_dict",
                    "provenance",
                )
            }
            inference["kind"] = "mlp_state_probe"
            inference["shape"] = spec.task
            inference["target_spec"] = spec.name
            atomic_torch_save(inference, output / "probe_weights.pt")
        if stale >= config.patience:
            break
    report = {
        "schema_version": 2,
        "kind": "paper_mlp_state_probe",
        "task": spec.task,
        "target_spec": asdict(spec),
        "feature_dim": int(train_features.shape[1]),
        "temporal_window": int(temporal_window),
        "train_windows": len(train_features),
        "validation_windows": len(val_features),
        "best_epoch": best_epoch,
        "best_validation_normalized_mse": best_loss,
        "checkpoint": str(output / "probe_weights.pt"),
        "history": history,
        "status": "pass",
    }
    atomic_json(report, output / "probe_summary.json")
    return report


def _indices_from_split_file(store, split_path: str | Path) -> EpisodeSplit:
    payload = json.loads(Path(split_path).read_text())
    train_ids = {int(value) for value in payload["train_episode_indices"]}
    val_ids = {int(value) for value in payload["val_episode_indices"]}
    if train_ids & val_ids:
        raise ValueError("probe split train and validation IDs overlap")
    by_source = {int(record.source_id): index for index, record in enumerate(store.episodes)}
    missing = (train_ids | val_ids) - set(by_source)
    if missing:
        raise ValueError(f"probe split IDs are absent from the dataset: {sorted(missing)[:8]}")
    train_indices = tuple(sorted(by_source[value] for value in train_ids))
    val_indices = tuple(sorted(by_source[value] for value in val_ids))
    if any(store.split_id[index] != 0 for index in (*train_indices, *val_indices)):
        raise ValueError("probe split includes held-out test episodes")
    return EpisodeSplit(
        train_indices,
        val_indices,
        tuple(sorted(train_ids)),
        tuple(sorted(val_ids)),
        int(payload.get("seed", 42)),
        float(payload.get("train_fraction", len(train_ids) / (len(train_ids) + len(val_ids)))),
    )


def _sample_dataset(dataset: TrajectoryWindowDataset, maximum: int, seed: int) -> Subset:
    count = len(dataset) if maximum < 0 else min(len(dataset), int(maximum))
    generator = np.random.default_rng(int(seed))
    indices = np.arange(len(dataset))
    if count < len(indices):
        indices = np.sort(generator.choice(indices, size=count, replace=False))
    return Subset(dataset, indices.tolist())


@torch.inference_mode()
def _extract_probe_features(
    dataset: torch.utils.data.Dataset,
    *,
    model_kind: str,
    native_model: nn.Module | None,
    dino_encoder: nn.Module | None,
    preprocessor: Native128Preprocessor | None,
    spec: ProbeTargetSpec,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader_kwargs: dict[str, Any] = {"num_workers": num_workers}
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
            multiprocessing_context="spawn",
        )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        **loader_kwargs,
    )
    feature_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for batch in loader:
        pixels = batch["pixels"].to(device)
        if model_kind == "native":
            assert native_model is not None
            encoded = native_model.encode({"pixels": pixels})["emb"].float()
        else:
            assert dino_encoder is not None and preprocessor is not None
            batch_size_value, frames = pixels.shape[:2]
            mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, None, :, None, None]
            std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, None, :, None, None]
            raw = (pixels * std + mean).clamp(0.0, 1.0).reshape(-1, *pixels.shape[2:])
            tokens = dino_encoder(preprocessor(raw)).to(torch.float16).float()
            encoded = tokens.mean(dim=1).reshape(batch_size_value, frames, -1)
        feature_chunks.append(encoded.flatten(start_dim=1).cpu())
        target_chunks.append(transform_state_targets(batch["state"][:, -1], spec).cpu())
    return torch.cat(feature_chunks), torch.cat(target_chunks)


def run_probe_training(
    *,
    task: str,
    config_path: str | Path,
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    dinov2_root: str | Path | None = None,
    split_path: str | Path | None = None,
    resume: str | Path | None = None,
    config: ProbeTrainConfig | None = None,
) -> dict[str, Any]:
    """Extract frozen temporal features and train the paper MLP probe."""

    spec = ProbeTargetSpec.for_task(task)
    config = config or ProbeTrainConfig()
    temporal_window = config.temporal_window or (2 if spec.task == "approach_ball" else 4)
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    raw_config = yaml.safe_load(Path(config_path).read_text())
    model_kind = "dino" if _is_dino_config(raw_config) else "native"
    native_model: nn.Module | None = None
    dino_encoder: nn.Module | None = None
    preprocessor: Native128Preprocessor | None = None
    if model_kind == "native":
        experiment = load_experiment_config(config_path)
        native_model = build_world_model(experiment.model)
        strict_load(native_model, checkpoint_path)
        native_model.to(selected_device).eval().requires_grad_(False)
    else:
        predictor, _predictor_config = _load_dino_predictor(checkpoint_path, config_path)
        predictor.to(selected_device).eval().requires_grad_(False)
        if dinov2_root is None:
            raise ValueError("DINO probe training requires --dinov2-root")
        dino_encoder = load_dinov2_encoder(dinov2_root, device=selected_device)
        preprocessor = Native128Preprocessor()

    store = open_trajectory_store(dataset_path)
    split = (
        _indices_from_split_file(store, split_path)
        if split_path is not None
        else split_development_episodes(store, train_fraction=0.9, seed=config.seed)
    )
    train_dataset = TrajectoryWindowDataset(
        dataset_path,
        num_steps=temporal_window,
        split="train",
        store=store,
        episode_indices=split.train_indices,
    )
    val_dataset = TrajectoryWindowDataset(
        dataset_path,
        num_steps=temporal_window,
        split="train",
        store=store,
        episode_indices=split.val_indices,
        action_statistics=train_dataset.action_statistics,
    )
    sampled_train = _sample_dataset(train_dataset, config.max_train_windows, config.seed)
    sampled_val = _sample_dataset(val_dataset, config.max_val_windows, config.seed + 1)
    train_features, train_targets = _extract_probe_features(
        sampled_train,
        model_kind=model_kind,
        native_model=native_model,
        dino_encoder=dino_encoder,
        preprocessor=preprocessor,
        spec=spec,
        device=selected_device,
        batch_size=config.feature_batch_size,
        num_workers=config.num_workers,
    )
    val_features, val_targets = _extract_probe_features(
        sampled_val,
        model_kind=model_kind,
        native_model=native_model,
        dino_encoder=dino_encoder,
        preprocessor=preprocessor,
        spec=spec,
        device=selected_device,
        batch_size=config.feature_batch_size,
        num_workers=config.num_workers,
    )
    return fit_state_probe(
        train_features,
        train_targets,
        val_features,
        val_targets,
        output_dir,
        task=spec.task,
        temporal_window=temporal_window,
        config=config,
        provenance={
            "device": str(selected_device),
            "model_kind": model_kind,
            "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "dataset": str(Path(dataset_path).expanduser().resolve()),
            "data_split": split.to_dict(),
        },
        resume=resume,
    )


def _action_statistics(
    checkpoint_path: Path,
    store,
    normalization_path: str | Path | None,
) -> tuple[ActionStatistics, str]:
    candidates: list[tuple[dict[str, Any], str]] = []
    if normalization_path is not None:
        candidates.append((json.loads(Path(normalization_path).read_text()), "explicit_file"))
    if checkpoint_path.suffix == ".safetensors" and checkpoint_path.with_suffix(".json").is_file():
        candidates.append(
            (json.loads(checkpoint_path.with_suffix(".json").read_text()), "checkpoint_sidecar")
        )
    elif checkpoint_path.suffix != ".safetensors":
        payload = load_training_checkpoint(checkpoint_path)
        if isinstance(payload, dict):
            candidates.append((payload, "checkpoint"))
    for payload, source in candidates:
        stats = payload.get("action_statistics", payload.get("action"))
        if isinstance(stats, dict) and {"mean", "std"}.issubset(stats):
            return ActionStatistics.from_dict(stats), source
    development = np.flatnonzero(store.split_id == 0)
    return ActionStatistics.fit(store.actions(development)), "fitted_development_fallback"


def _probe_trajectory(
    probe: FrozenStateProbe,
    sequence: torch.Tensor,
    *,
    history: int,
    horizon: int,
) -> torch.Tensor:
    pooled = sequence.mean(dim=1) if sequence.ndim == 3 else sequence
    windows = torch.stack(
        [
            pooled[index - probe.temporal_window + 1 : index + 1].reshape(-1)
            for index in range(history, history + horizon)
        ]
    )
    if windows.shape[-1] != probe.feature_dim:
        raise ValueError(f"probe expects feature_dim={probe.feature_dim}, got {windows.shape[-1]}")
    return probe.predict_features(windows)


def _planar_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spec: ProbeTargetSpec,
    std: torch.Tensor,
) -> dict[str, float]:
    difference = prediction - target
    result = {
        "target_mse": float(difference.square().mean()),
        "normalized_target_mse": float((difference / std).square().mean()),
        "position_l2": float(difference[:2].norm()),
        "velocity_l2": float(difference[2:4].norm()),
        "angular_velocity_mae": float(difference[6].abs()),
    }
    assert spec.symmetry_order is not None
    phase_pred = torch.atan2(prediction[4], prediction[5])
    phase_target = torch.atan2(target[4], target[5])
    delta = torch.atan2(torch.sin(phase_pred - phase_target), torch.cos(phase_pred - phase_target))
    result["angle_symmetry_mae_rad"] = float(delta.abs() / spec.symmetry_order)
    result["angle_symmetry_mae_deg"] = result["angle_symmetry_mae_rad"] * 180.0 / math.pi
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _planar_evaluation_indices(
    store: Any,
    *,
    maximum: int,
    manifest_path: str | Path | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if manifest_path is None:
        indices = np.flatnonzero(store.split_id == 1)[:maximum]
        return indices, {"kind": "first_held_out", "maximum": int(maximum)}

    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text())
    gravities = payload.get("gravity_values")
    per_gravity = payload.get("per_gravity")
    if not isinstance(gravities, list) or not isinstance(per_gravity, dict):
        raise ValueError("planar evaluation manifest has an invalid cohort mapping")
    groups_by_gravity = {float(key): value for key, value in per_gravity.items()}
    source_ids: list[int] = []
    for gravity in gravities:
        group = groups_by_gravity.get(float(gravity))
        if not isinstance(group, dict) or not isinstance(group.get("source_ids"), list):
            raise ValueError(f"planar manifest lacks source IDs for gravity {gravity}")
        declared = int(group.get("count", len(group["source_ids"])))
        if declared != len(group["source_ids"]):
            raise ValueError(f"planar manifest count differs at gravity {gravity}")
        source_ids.extend(int(value) for value in group["source_ids"])
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("planar evaluation manifest repeats source episode IDs")
    by_source = {int(record.source_id): index for index, record in enumerate(store.episodes)}
    selected_ids = source_ids[:maximum]
    missing = [value for value in selected_ids if value not in by_source]
    if missing:
        raise ValueError(f"planar dataset lacks manifest source IDs: {missing[:8]}")
    indices = np.asarray([by_source[value] for value in selected_ids], dtype=np.int64)
    if np.any(store.split_id[indices] != 1):
        raise ValueError("planar evaluation manifest selected development episodes")
    return indices, {
        "kind": "source_episode_manifest",
        "path": str(path),
        "sha256": sha256_file(path),
        "available_episode_count": len(source_ids),
        "selected_episode_count": len(indices),
    }


def run_planar_probe_evaluation(
    *,
    task: str,
    method: str,
    config_path: str | Path,
    checkpoint_path: str | Path,
    probe_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    probe_config_path: str | Path | None = None,
    dinov2_root: str | Path | None = None,
    normalization_path: str | Path | None = None,
    episode_manifest_path: str | Path | None = None,
    max_episodes: int = 8,
    horizon: int = 44,
    spin_dt: float = 1.0 / 16.0,
    frame_batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate paper state metrics on a bounded 2D held-out cohort."""

    spec = ProbeTargetSpec.for_task(task)
    if spec.task == "approach_ball":
        raise ValueError("Approach uses run_frozen_approach_evaluation")
    if max_episodes <= 0 or horizon <= 0 or spin_dt <= 0:
        raise ValueError("evaluation bounds and spin_dt must be positive")
    selected_device = torch.device(device)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    probe = load_state_probe(probe_path, device=selected_device, config_path=probe_config_path)
    if probe.target_spec != spec:
        raise ValueError("probe target task differs from evaluation task")
    raw_config = yaml.safe_load(Path(config_path).read_text())
    model_kind = "dino" if _is_dino_config(raw_config) else "native"
    encoder = None
    if model_kind == "native":
        experiment = load_experiment_config(config_path)
        model = build_world_model(experiment.model)
        strict_load(model, checkpoint_path)
        history = experiment.model.history_size
        image_size = experiment.model.image_size
    else:
        model, predictor_config = _load_dino_predictor(checkpoint_path, config_path)
        history = predictor_config.history_size
        image_size = 128
        if dinov2_root is None:
            raise ValueError("DINO evaluation requires --dinov2-root")
        encoder = load_dinov2_encoder(dinov2_root, device=selected_device)
    model.to(selected_device).eval().requires_grad_(False)
    store = open_trajectory_store(dataset_path)
    statistics, statistics_source = _action_statistics(
        checkpoint_path,
        store,
        normalization_path,
    )
    indices, cohort = _planar_evaluation_indices(
        store,
        maximum=max_episodes,
        manifest_path=episode_manifest_path,
    )
    if not len(indices):
        raise ValueError("planar evaluation requires held-out episodes")
    maximum = min(int(store.lengths[index]) - history for index in indices)
    if horizon > maximum:
        raise ValueError(f"requested horizon {horizon} exceeds cohort maximum {maximum}")

    rows: list[dict[str, Any]] = []
    for store_index in indices:
        episode = store.episode(int(store_index))
        actions = torch.as_tensor(
            (episode.action - np.asarray(statistics.mean)) / np.asarray(statistics.std),
            dtype=torch.float32,
            device=selected_device,
        )
        if model_kind == "native":
            true_sequence = _encode_native(
                model,
                episode.pixels,
                image_size=image_size,
                device=selected_device,
                frame_batch_size=frame_batch_size,
            )
            predicted_sequence = _rollout_native(
                model,
                true_sequence,
                actions,
                history=history,
                horizon=horizon,
            )
        else:
            assert encoder is not None
            tokens = _encode_dino(
                encoder,
                episode.pixels,
                device=selected_device,
                frame_batch_size=frame_batch_size,
            )
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.bfloat16,
                enabled=selected_device.type == "cuda",
            ):
                continuation = model.rollout_visual_tokens(
                    tokens.unsqueeze(0), actions.unsqueeze(0), horizon=horizon
                )[0].float()
            true_sequence = tokens
            predicted_sequence = torch.cat((tokens[:history], continuation), dim=0)
        targets = transform_state_targets(episode.state[history : history + horizon], spec).to(
            selected_device
        )
        decoded = {
            "oracle": _probe_trajectory(probe, true_sequence, history=history, horizon=horizon),
            "predicted": _probe_trajectory(
                probe,
                predicted_sequence,
                history=history,
                horizon=horizon,
            ),
        }
        target_signed = torch.cumsum(targets[:, 6] * spin_dt, dim=0)
        target_absolute = torch.cumsum(targets[:, 6].abs() * spin_dt, dim=0)
        gravity = float(episode.gravity[0])
        for source, values in decoded.items():
            predicted_signed = torch.cumsum(values[:, 6] * spin_dt, dim=0)
            predicted_absolute = torch.cumsum(values[:, 6].abs() * spin_dt, dim=0)
            for offset in range(horizon):
                rows.append(
                    {
                        "episode_id": episode.episode_id,
                        "gravity": gravity,
                        "source": source,
                        "horizon": offset + 1,
                        **_planar_metrics(values[offset], targets[offset], spec, probe.target_std),
                        "signed_spin_mae_rad": float(
                            (predicted_signed[offset] - target_signed[offset]).abs()
                        ),
                        "signed_spin_mae_turns": float(
                            (predicted_signed[offset] - target_signed[offset]).abs()
                            / (2.0 * math.pi)
                        ),
                        "absolute_spin_mae_rad": float(
                            (predicted_absolute[offset] - target_absolute[offset]).abs()
                        ),
                        "absolute_spin_mae_turns": float(
                            (predicted_absolute[offset] - target_absolute[offset]).abs()
                            / (2.0 * math.pi)
                        ),
                    }
                )
    numeric = [key for key in rows[0] if key not in {"episode_id", "gravity", "source", "horizon"}]
    aggregate: list[dict[str, Any]] = []
    for source in ("oracle", "predicted"):
        for step in range(1, horizon + 1):
            selected = [row for row in rows if row["source"] == source and row["horizon"] == step]
            aggregate.append(
                {
                    "source": source,
                    "horizon": step,
                    "n": len(selected),
                    **{key: float(np.mean([row[key] for row in selected])) for key in numeric},
                }
            )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "per_episode_horizon.csv", rows)
    _write_csv(output / "horizon_metrics.csv", aggregate)
    report = {
        "schema_version": 2,
        "kind": "planar_state_probe_rollout_evaluation",
        "claim": "bounded_paper_metric_implementation",
        "paper_metric_definition": True,
        "paper_result_reproduction": False,
        "task": spec.task,
        "method": method,
        "model_kind": model_kind,
        "history": history,
        "horizon": horizon,
        "spin_dt": spin_dt,
        "evaluated_episode_count": len(indices),
        "cohort": cohort,
        "action_statistics": statistics.to_dict(),
        "action_statistics_source": statistics_source,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "probe_sha256": sha256_file(probe_path),
        "outputs": {
            "per_episode_horizon": str(output / "per_episode_horizon.csv"),
            "horizon_metrics": str(output / "horizon_metrics.csv"),
        },
        "status": "pass",
    }
    atomic_json(report, output / "evaluation.json")
    return report


__all__ = [
    "FrozenStateProbe",
    "ProbeTargetSpec",
    "ProbeTrainConfig",
    "build_state_probe",
    "fit_state_probe",
    "load_state_probe",
    "run_planar_probe_evaluation",
    "run_probe_training",
    "transform_state_targets",
]
