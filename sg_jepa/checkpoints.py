"""Content-addressed artifact resolution and strict checkpoint loading."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload
        for key in (
            "model",
            "model_state_dict",
            "state_dict",
            "world_model",
            "policy",
            "policy_state_dict",
        ):
            candidate = payload.get(key)
            if (
                isinstance(candidate, Mapping)
                and candidate
                and all(torch.is_tensor(value) for value in candidate.values())
            ):
                return candidate
    raise TypeError("checkpoint does not contain a recognizable tensor state dictionary")


def _weights_only_load(path: Path) -> Any:
    """Safely admit only the NumPy objects used by our captured RNG state."""

    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:  # pragma: no cover - older supported PyTorch.
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    uint32 = np.asarray([], dtype=np.uint32)
    allowed = [
        uint32.__reduce__()[0],
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    with safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def load_state_dict(path: str | Path) -> Mapping[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise ImportError("install sg-jepa[hub] to read safetensors") from exc
        state = load_file(str(path), device="cpu")
    else:
        payload = _weights_only_load(path)
        state = _extract_state_dict(payload)
    non_finite = [
        name
        for name, value in state.items()
        if torch.is_floating_point(value) and not torch.isfinite(value).all()
    ]
    if non_finite:
        raise ValueError(f"checkpoint contains non-finite tensors: {non_finite[:5]}")
    return state


def strict_load(model: nn.Module, path: str | Path) -> nn.Module:
    """Load a checkpoint with no missing/unexpected keys and finite tensors."""

    state = load_state_dict(path)
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "load_state_dict",
    "sha256_file",
    "strict_load",
]
