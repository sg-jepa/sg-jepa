"""Bounded Approach-Ball evaluation with historical models and probes."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sg_jepa.baselines import (
    Native128Preprocessor,
    build_dino_world_model,
    load_dino_config,
    load_dinov2_encoder,
    strict_load_dino,
)
from sg_jepa.checkpoints import load_state_dict, sha256_file, strict_load
from sg_jepa.config import load_experiment_config
from sg_jepa.data import open_trajectory_store
from sg_jepa.models import build_world_model

APPROACH_TARGETS = ("x", "y", "z", "vx", "vy", "vz")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class FrozenProbe:
    model: nn.Module
    feature_dim: int
    temporal_window: int
    target_mean: torch.Tensor
    target_std: torch.Tensor
    checkpoint_kind: str


def build_approach_probe(feature_dim: int, output_dim: int = 6) -> nn.Module:
    """The released two-frame MLP architecture for Approach state decoding."""

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


def strict_load_approach_probe(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    config_path: str | Path | None = None,
) -> FrozenProbe:
    """Strictly load either the native-v10 or corrected DINO-v11 probe."""

    path = Path(path).expanduser().resolve()
    if path.suffix == ".safetensors":
        metadata_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else path.with_suffix(".json")
        )
        payload = json.loads(metadata_path.read_text())
        payload = {
            **payload["architecture"],
            "target_stats": payload["target_stats"],
            "kind": payload.get("checkpoint_kind", "mlp_state_probe"),
            "probe_state_dict": load_state_dict(path),
        }
    else:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:  # pragma: no cover - old PyTorch compatibility.
            payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("probe_state_dict"), dict):
        raise TypeError("probe checkpoint has no probe_state_dict")
    if payload.get("probe_kind") != "mlp":
        raise ValueError("Approach release supports only the locked MLP probe")
    if payload.get("target_spec") != "ball3d_posvel":
        raise ValueError("probe target must be ball3d_posvel")
    feature_dim = int(payload.get("feature_dim", 0))
    temporal_window = int(payload.get("temporal_window", 0))
    output_dim = int(payload.get("output_dim", 0))
    if temporal_window != 2 or output_dim != len(APPROACH_TARGETS):
        raise ValueError("Approach probe must use a two-frame, six-coordinate contract")
    if feature_dim not in {512, 768}:
        raise ValueError(f"unexpected Approach probe feature dimension {feature_dim}")
    stats = payload.get("target_stats")
    if not isinstance(stats, dict):
        raise TypeError("probe checkpoint has no embedded target_stats")
    if tuple(stats.get("target_definition", ())) != APPROACH_TARGETS:
        raise ValueError("probe target definition differs from the release contract")
    mean = torch.as_tensor(stats.get("mean"), dtype=torch.float32)
    std = torch.as_tensor(stats.get("std"), dtype=torch.float32)
    if mean.shape != (6,) or std.shape != (6,) or not torch.isfinite(mean).all():
        raise ValueError("probe target statistics are invalid")
    if not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("probe target standard deviations must be positive and finite")
    state = payload["probe_state_dict"]
    non_finite = [
        name
        for name, value in state.items()
        if torch.is_floating_point(value) and not torch.isfinite(value).all()
    ]
    if non_finite:
        raise ValueError(f"probe checkpoint contains non-finite tensors: {non_finite[:5]}")
    model = build_approach_probe(feature_dim, output_dim)
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device)).eval().requires_grad_(False)
    return FrozenProbe(
        model=model,
        feature_dim=feature_dim,
        temporal_window=temporal_window,
        target_mean=mean.to(device),
        target_std=std.to(device),
        checkpoint_kind=str(payload.get("kind", "native_v10_vector_probe")),
    )


def _native_pixels(pixels: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(pixels))
    if value.ndim != 4:
        raise ValueError("episode pixels must have four dimensions")
    if value.shape[-1] == 3:
        value = value.permute(0, 3, 1, 2)
    if value.shape[1] != 3:
        raise ValueError("episode pixels must be RGB")
    value = value.to(device=device, dtype=torch.float32).div_(255.0)
    if tuple(value.shape[-2:]) != (image_size, image_size):
        value = F.interpolate(
            value,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    mean = torch.tensor(IMAGENET_MEAN, dtype=value.dtype, device=device)[None, :, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=value.dtype, device=device)[None, :, None, None]
    return (value - mean) / std


@torch.inference_mode()
def _encode_native(
    model: nn.Module,
    pixels: np.ndarray,
    *,
    image_size: int,
    device: torch.device,
    frame_batch_size: int,
) -> torch.Tensor:
    images = _native_pixels(pixels, image_size, device)
    chunks = []
    for first in range(0, len(images), frame_batch_size):
        output = model.encode({"pixels": images[first : first + frame_batch_size].unsqueeze(0)})
        chunks.append(output["emb"][0].float())
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def _rollout_native(
    model: nn.Module,
    true_features: torch.Tensor,
    normalized_actions: torch.Tensor,
    *,
    history: int,
    horizon: int,
) -> torch.Tensor:
    action_features = model.action_encoder(normalized_actions.unsqueeze(0))
    generated = [true_features[index] for index in range(history)]
    for _ in range(horizon):
        current = len(generated)
        first = max(0, current - history)
        latent_window = torch.stack(generated[first:current], dim=0).unsqueeze(0)
        action_window = action_features[:, first:current]
        generated.append(model.predict(latent_window, action_window)[0, -1].float())
    return torch.stack(generated, dim=0)


@torch.inference_mode()
def _encode_dino(
    encoder: nn.Module,
    pixels: np.ndarray,
    *,
    device: torch.device,
    frame_batch_size: int,
) -> torch.Tensor:
    preprocessor = Native128Preprocessor()
    chunks = []
    for first in range(0, len(pixels), frame_batch_size):
        images = preprocessor(np.asarray(pixels[first : first + frame_batch_size])).to(device)
        # The historical cache rounded every patch token through fp16 before
        # predictor/probe use. Preserve that contract even under bf16 autocast.
        tokens = encoder(images).to(torch.float16).to(torch.float32)
        chunks.append(tokens)
    return torch.cat(chunks, dim=0)


def _decoded_probe(
    probe: FrozenProbe,
    sequence: torch.Tensor,
    *,
    history: int,
    horizon: int,
) -> torch.Tensor:
    previous = sequence[history - 1 : history + horizon - 1]
    current = sequence[history : history + horizon]
    inputs = torch.cat((previous, current), dim=-1)
    if inputs.shape[-1] != probe.feature_dim:
        raise ValueError(f"probe expects feature_dim={probe.feature_dim}, got {inputs.shape[-1]}")
    normalized = probe.model(inputs.float())
    return normalized * probe.target_std + probe.target_mean


def _metrics(
    prediction: torch.Tensor, target: torch.Tensor, std: torch.Tensor
) -> dict[str, np.ndarray]:
    difference = prediction.float() - target.float()
    return {
        "target_mse": difference.square().mean(dim=-1).cpu().numpy(),
        "normalized_target_mse": (difference / std).square().mean(dim=-1).cpu().numpy(),
        "position_l2": difference[:, :3].norm(dim=-1).cpu().numpy(),
        "velocity_l2": difference[:, 3:].norm(dim=-1).cpu().numpy(),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), int(row["horizon"]))].append(row)
    output = []
    for (source, horizon), items in sorted(grouped.items()):
        output.append(
            {
                "source": source,
                "horizon": horizon,
                "n": len(items),
                **{
                    key: float(np.mean([float(item[key]) for item in items]))
                    for key in (
                        "target_mse",
                        "normalized_target_mse",
                        "position_l2",
                        "velocity_l2",
                    )
                },
            }
        )
    return output


def run_frozen_approach_evaluation(
    *,
    method: str,
    model_kind: str,
    config_path: str | Path,
    checkpoint_path: str | Path,
    probe_path: str | Path,
    probe_config_path: str | Path | None = None,
    dataset_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    max_episodes: int = 8,
    horizon: int = 44,
    action_mean: float = 9.759801140957409,
    action_std: float = 1.9904320653880276,
    dinov2_root: str | Path | None = None,
    frame_batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate a historical checkpoint on a small, user-supplied test cohort."""

    if model_kind not in {"native", "dino"}:
        raise ValueError("model_kind must be 'native' or 'dino'")
    if max_episodes <= 0 or horizon <= 0 or frame_batch_size <= 0:
        raise ValueError("max_episodes, horizon, and frame_batch_size must be positive")
    if not np.isfinite([action_mean, action_std]).all() or action_std <= 0:
        raise ValueError("action normalization must be finite with positive std")
    selected_device = torch.device(device)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    probe_path = Path(probe_path).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    probe = strict_load_approach_probe(
        probe_path,
        device=selected_device,
        config_path=probe_config_path,
    )
    encoder = None
    if model_kind == "native":
        config = load_experiment_config(config_path)
        if config.model.action_dim != 1 or config.model.history_size != 20:
            raise ValueError("native Approach checkpoint requires action_dim=1 and history=20")
        model = build_world_model(config.model)
        strict_load(model, checkpoint_path)
        model.to(selected_device).eval().requires_grad_(False)
        if probe.feature_dim != 2 * config.model.embed_dim:
            raise ValueError("native probe dimension differs from model embedding dimension")
        history = config.model.history_size
        image_size = config.model.image_size
        checkpoint_metadata: dict[str, Any] = {}
    else:
        if checkpoint_path.suffix == ".safetensors":
            predictor_config, raw_config = load_dino_config(config_path)
            model = build_dino_world_model(predictor_config)
            model.load_state_dict(load_state_dict(checkpoint_path), strict=True)
            checkpoint_metadata = raw_config
        else:
            model, checkpoint_metadata = strict_load_dino(checkpoint_path, config_path)
        model.to(selected_device).eval().requires_grad_(False)
        history = model.config.history_size
        image_size = 128
        if probe.feature_dim != 2 * model.config.visual_dim:
            raise ValueError("DINO probe dimension differs from predictor visual dimension")
        if dinov2_root is None:
            raise ValueError("DINO evaluation requires dinov2_root")
        encoder = load_dinov2_encoder(dinov2_root, device=selected_device)

    store = open_trajectory_store(dataset_path)
    indices = np.flatnonzero(store.split_id == 1)[:max_episodes]
    if not len(indices):
        raise ValueError("Approach qualitative evaluation requires held-out episodes")
    maximum_horizon = min(int(store.lengths[int(index)]) - history for index in indices)
    if horizon > maximum_horizon:
        raise ValueError(f"requested horizon {horizon} exceeds cohort maximum {maximum_horizon}")

    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    with torch.inference_mode():
        for store_index in indices:
            episode = store.episode(int(store_index))
            actions = torch.as_tensor(
                (np.asarray(episode.action, dtype=np.float32) - action_mean) / action_std,
                device=selected_device,
                dtype=torch.float32,
            )
            if actions.shape != (int(store.lengths[int(store_index)]), 1):
                raise ValueError(f"Approach action contract differs: {tuple(actions.shape)}")
            target = torch.as_tensor(
                np.asarray(episode.state[history : history + horizon, :6], dtype=np.float32),
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
                autocast = torch.autocast(
                    device_type=selected_device.type,
                    dtype=torch.bfloat16,
                    enabled=selected_device.type == "cuda",
                )
                with autocast:
                    predicted_tokens = model.rollout_visual_tokens(
                        tokens.unsqueeze(0),
                        actions.unsqueeze(0),
                        horizon=horizon,
                    )[0].float()
                true_sequence = tokens.mean(dim=1)
                predicted_sequence = predicted_tokens.mean(dim=1)

            real_decoded = _decoded_probe(
                probe,
                true_sequence,
                history=history,
                horizon=horizon,
            )
            predicted_decoded = _decoded_probe(
                probe,
                predicted_sequence,
                history=history,
                horizon=horizon,
            )
            gravity = float(np.asarray(episode.gravity).reshape(-1)[0])
            for source, decoded in (("real", real_decoded), ("predicted", predicted_decoded)):
                values = _metrics(decoded, target, probe.target_std)
                for offset in range(horizon):
                    rows.append(
                        {
                            "episode_id": episode.episode_id,
                            "gravity": gravity,
                            "source": source,
                            "horizon": offset + 1,
                            **{key: float(value[offset]) for key, value in values.items()},
                        }
                    )
            if len(examples) < 4:
                examples.append(
                    {
                        "episode_id": episode.episode_id,
                        "gravity": gravity,
                        "target": target.cpu().numpy(),
                        "predicted": predicted_decoded.cpu().numpy(),
                        "real_probe": real_decoded.cpu().numpy(),
                    }
                )

    aggregates = _aggregate(rows)
    per_episode_path = output_dir / "per_episode_horizon.csv"
    aggregate_path = output_dir / "horizon_metrics.csv"
    _write_csv(per_episode_path, rows)
    _write_csv(aggregate_path, aggregates)
    np.savez_compressed(
        output_dir / "trajectory_examples.npz",
        episode_id=np.asarray([item["episode_id"] for item in examples]),
        gravity=np.asarray([item["gravity"] for item in examples]),
        target=np.stack([item["target"] for item in examples]),
        predicted=np.stack([item["predicted"] for item in examples]),
        real_probe=np.stack([item["real_probe"] for item in examples]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "approach_ball_frozen_checkpoint_qualitative_evaluation",
        "claim": "qualitative_reproduction",
        "paper_metric": False,
        "exact_reproduction": False,
        "warning": (
            "This bounded regenerated cohort checks qualitative rollout behavior only; "
            "it is not the paper cohort and cannot estimate the reported aggregate metric."
        ),
        "method": method,
        "model_kind": model_kind,
        "dataset": str(dataset_path),
        "cohort": "bounded held-out subset from user-supplied or regenerated data",
        "evaluated_episode_count": len(indices),
        "history": history,
        "horizon": horizon,
        "action_normalization": {"mean": [action_mean], "std": [action_std]},
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "strict_load": "pass",
            **checkpoint_metadata,
        },
        "probe": {
            "path": str(probe_path),
            "sha256": sha256_file(probe_path),
            "strict_load": "pass",
            "kind": probe.checkpoint_kind,
            "feature_dim": probe.feature_dim,
            "temporal_window": probe.temporal_window,
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "outputs": {
            "per_episode_horizon": str(per_episode_path),
            "horizon_metrics": str(aggregate_path),
            "trajectory_examples": str(output_dir / "trajectory_examples.npz"),
        },
        "status": "pass",
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "APPROACH_TARGETS",
    "FrozenProbe",
    "build_approach_probe",
    "run_frozen_approach_evaluation",
    "strict_load_approach_probe",
]
