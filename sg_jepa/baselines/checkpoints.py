"""Strict loaders for separately packaged baseline checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from .dino_wm import DinoWorldModel, PredictorConfig


def load_dino_config(path: str | Path) -> tuple[PredictorConfig, dict[str, Any]]:
    """Load a DINO-WM architecture from a training YAML or release sidecar."""

    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("DINO-WM config must be a mapping")
    if payload.get("schema_version") != 1 and payload.get("format_version") != 1:
        raise ValueError("DINO-WM config requires schema_version=1 or format_version=1")
    predictor = payload.get("predictor", payload.get("architecture"))
    if not isinstance(predictor, dict):
        raise ValueError("DINO-WM config requires a predictor or architecture mapping")
    return PredictorConfig.from_dict(predictor), payload


def strict_load_dino(
    checkpoint: str | Path,
    config: str | Path,
) -> tuple[DinoWorldModel, dict[str, Any]]:
    """Construct and strictly load a paper DINO-WM predictor."""

    predictor_config, sidecar = load_dino_config(config)
    try:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            # The historical full-state file contains NumPy RNG objects, which
            # the restricted unpickler rejects. Callers verify the manifest
            # SHA-256 before reaching this trusted legacy loader.
            weights_only=False,
            mmap=True,
        )
    except TypeError:  # pragma: no cover - old PyTorch compatibility.
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise TypeError("DINO-WM checkpoint has no model_state_dict")
    embedded = payload.get("config", {}).get("predictor", {})
    expected = sidecar.get("predictor", sidecar.get("architecture"))
    if embedded and embedded != expected:
        raise ValueError("checkpoint predictor config differs from its release sidecar")
    state = payload["model_state_dict"]
    non_finite = [
        name
        for name, value in state.items()
        if torch.is_floating_point(value) and not torch.isfinite(value).all()
    ]
    if non_finite:
        raise ValueError(f"checkpoint contains non-finite tensors: {non_finite[:5]}")
    model = DinoWorldModel(predictor_config)
    model.load_state_dict(state, strict=True)
    return model, {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "tensor_count": len(model.state_dict()),
        "action_dim": predictor_config.action_dim,
        "history_size": predictor_config.history_size,
    }


__all__ = ["load_dino_config", "strict_load_dino"]
