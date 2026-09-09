"""Task-aware control conversion and train-only normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

CONTROL_DIM = 5
RAW_ACTION_DIM = 6
DIRECT_CONTROL_CODEC = "direct"
PADDLE_TILT_VECTOR_CODEC = "paddle_tilt_vector"
CONTROL_CODECS = {DIRECT_CONTROL_CODEC, PADDLE_TILT_VECTOR_CODEC}
TRAIN_ZSCORE_GRAVITY = "successful_train_zscore"
FIXED_PADDLE_GRAVITY = "fixed_physical_range_0_20"
GRAVITY_NORMALIZATIONS = {TRAIN_ZSCORE_GRAVITY, FIXED_PADDLE_GRAVITY}
PADDLE_TILT_SCALE = 0.5
DEFAULT_MAX_TILT_THETA = math.pi / 6.0
CONTROL_SCHEMAS = {
    3: ("dx", "dy", "dz"),
    5: ("dx", "dy", "dz", "phi", "theta"),
}


def _is_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


def _check_last_dim(value: Any, expected: int, name: str) -> None:
    if value.ndim < 1 or int(value.shape[-1]) != expected:
        raise ValueError(f"{name} must end in dimension {expected}, got {tuple(value.shape)}")


def raw_actions_to_controls(
    raw_actions: np.ndarray | torch.Tensor,
    *,
    control_codec: str = DIRECT_CONTROL_CODEC,
) -> np.ndarray | torch.Tensor:
    """Decode simulator actions into the policy's continuous controls.

    Franka and Catcher policies directly model the non-gravity action columns.
    The paper's Arm Paddle Ball policy instead models a two-vector encoding of
    paddle tilt: ``sin(theta) * [cos(phi), sin(phi)] / 0.5``.
    """

    raw_dim = int(raw_actions.shape[-1]) if raw_actions.ndim else 0
    if raw_dim - 1 not in CONTROL_SCHEMAS:
        raise ValueError(
            f"raw_actions must end in dimension 4 or 6, got {tuple(raw_actions.shape)}"
        )
    if control_codec not in CONTROL_CODECS:
        raise ValueError(f"unsupported control codec: {control_codec!r}")
    if control_codec == PADDLE_TILT_VECTOR_CODEC:
        if raw_dim != RAW_ACTION_DIM:
            raise ValueError("paddle_tilt_vector requires [g,dx,dy,dz,phi,theta] actions")
        xyz = raw_actions[..., 1:4]
        phi = raw_actions[..., 4]
        theta = raw_actions[..., 5]
        if _is_tensor(raw_actions):
            tilt = (
                torch.stack(
                    (torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi)),
                    dim=-1,
                )
                / PADDLE_TILT_SCALE
            )
            return torch.cat((xyz, tilt), dim=-1)
        values = np.asarray(raw_actions)
        tilt = (
            np.stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)), axis=-1)
            / PADDLE_TILT_SCALE
        )
        return np.concatenate((values[..., 1:4], tilt), axis=-1)
    return raw_actions[..., 1:]


def _gravity_column(
    gravity: float | np.ndarray | torch.Tensor,
    controls: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    target_shape = tuple(controls.shape[:-1]) + (1,)
    if _is_tensor(controls):
        value = torch.as_tensor(gravity, dtype=controls.dtype, device=controls.device)
        while value.ndim < controls.ndim:
            value = value.unsqueeze(-1)
        return torch.broadcast_to(value, target_shape)
    array = np.asarray(controls)
    value = np.asarray(gravity, dtype=array.dtype)
    while value.ndim < array.ndim:
        value = np.expand_dims(value, -1)
    return np.broadcast_to(value, target_shape)


def controls_to_raw_actions(
    controls: np.ndarray | torch.Tensor,
    gravity: float | np.ndarray | torch.Tensor,
    *,
    xyz_limit: float | None = None,
    control_codec: str = DIRECT_CONTROL_CODEC,
    max_tilt_theta: float = DEFAULT_MAX_TILT_THETA,
) -> np.ndarray | torch.Tensor:
    """Encode policy controls as simulator actions and prepend gravity."""

    control_dim = int(controls.shape[-1]) if controls.ndim else 0
    if control_dim not in CONTROL_SCHEMAS:
        raise ValueError(f"controls must end in dimension 3 or 5, got {tuple(controls.shape)}")
    if control_codec not in CONTROL_CODECS:
        raise ValueError(f"unsupported control codec: {control_codec!r}")
    if control_codec == PADDLE_TILT_VECTOR_CODEC:
        if control_dim != CONTROL_DIM:
            raise ValueError("paddle_tilt_vector requires five policy controls")
        limit = 1.0 if xyz_limit is None else float(xyz_limit)
        if limit <= 0 or not 0 < max_tilt_theta <= math.pi / 2:
            raise ValueError("xyz_limit and max_tilt_theta are outside their legal ranges")
        if _is_tensor(controls):
            xyz = controls[..., :3].clamp(-limit, limit)
            tilt = controls[..., 3:5]
            radius = torch.linalg.vector_norm(tilt, dim=-1, keepdim=True)
            unit_tilt = tilt / torch.clamp(radius, min=torch.finfo(tilt.dtype).eps)
            legal_radius = min(1.0, float(math.sin(max_tilt_theta) / PADDLE_TILT_SCALE))
            radius_safe = radius.clamp(max=legal_radius)
            projected = unit_tilt * radius_safe
            sin_theta = (PADDLE_TILT_SCALE * radius_safe[..., 0]).clamp(
                0.0, math.sin(max_tilt_theta)
            )
            theta = torch.asin(sin_theta)
            phi = torch.atan2(projected[..., 1], projected[..., 0])
            phi = torch.where(
                radius[..., 0] > torch.finfo(tilt.dtype).eps,
                phi,
                torch.zeros_like(phi),
            )
            gravity_column = _gravity_column(gravity, controls)
            return torch.cat((gravity_column, xyz, phi.unsqueeze(-1), theta.unsqueeze(-1)), dim=-1)

        values = np.asarray(controls)
        xyz = np.clip(values[..., :3], -limit, limit)
        tilt = values[..., 3:5]
        radius = np.linalg.norm(tilt, axis=-1, keepdims=True)
        eps = np.finfo(tilt.dtype).eps if np.issubdtype(tilt.dtype, np.floating) else 1.0e-12
        unit_tilt = tilt / np.maximum(radius, eps)
        legal_radius = min(1.0, float(math.sin(max_tilt_theta) / PADDLE_TILT_SCALE))
        radius_safe = np.minimum(radius, legal_radius)
        projected = unit_tilt * radius_safe
        sin_theta = np.clip(
            PADDLE_TILT_SCALE * radius_safe[..., 0],
            0.0,
            math.sin(max_tilt_theta),
        )
        theta = np.arcsin(sin_theta)
        phi = np.arctan2(projected[..., 1], projected[..., 0])
        phi = np.where(radius[..., 0] > eps, phi, 0.0)
        gravity_column = _gravity_column(gravity, values)
        return np.concatenate((gravity_column, xyz, phi[..., None], theta[..., None]), axis=-1)
    if xyz_limit is not None and xyz_limit <= 0:
        raise ValueError("xyz_limit must be positive")
    value_controls = controls
    if xyz_limit is not None:
        if _is_tensor(controls):
            value_controls = controls.clone()
            value_controls[..., :3].clamp_(-xyz_limit, xyz_limit)
        else:
            value_controls = np.asarray(controls).copy()
            value_controls[..., :3] = np.clip(value_controls[..., :3], -xyz_limit, xyz_limit)
    gravity_column = _gravity_column(gravity, value_controls)
    if _is_tensor(value_controls):
        return torch.cat((gravity_column, value_controls), dim=-1)
    return np.concatenate((gravity_column, np.asarray(value_controls)), axis=-1)


def full_range_translation_delta(
    translation_action: np.ndarray | torch.Tensor,
    action_scale: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Apply the corrected Franka delta contract without a legacy unit cap.

    The factorized calculation intentionally mirrors the historical evaluator:
    normalize into the unit cube, then enlarge the task scale by the same
    factor.  Algebraically this preserves the complete requested delta.
    """

    _check_last_dim(translation_action, 3, "translation_action")
    _check_last_dim(action_scale, 3, "action_scale")
    if _is_tensor(translation_action):
        scale = torch.as_tensor(
            action_scale,
            dtype=translation_action.dtype,
            device=translation_action.device,
        )
        factor = torch.maximum(
            torch.ones_like(translation_action[..., :1]),
            translation_action.abs().amax(dim=-1, keepdim=True),
        )
        return (translation_action / factor) * (scale * factor)
    action = np.asarray(translation_action)
    scale = np.asarray(action_scale)
    factor = np.maximum(
        np.asarray(1.0, dtype=action.dtype),
        np.max(np.abs(action), axis=-1, keepdims=True),
    )
    mapped = (action / factor).astype(action.dtype, copy=False)
    return mapped * (scale * factor.astype(scale.dtype, copy=False))


@dataclass(frozen=True)
class ActionNormalizer:
    """Component-wise min/max map for Catcher or paddle controls."""

    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if len(self.minimum) not in CONTROL_SCHEMAS or len(self.maximum) != len(self.minimum):
            raise ValueError("ActionNormalizer requires three or five components")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not all(math.isfinite(value) for value in (*self.minimum, *self.maximum)):
            raise ValueError("action normalization bounds must be finite")
        if any(hi < lo for lo, hi in zip(self.minimum, self.maximum, strict=True)):
            raise ValueError("maximum must be greater than or equal to minimum")

    @classmethod
    def fit(
        cls,
        controls: np.ndarray | torch.Tensor,
        epsilon: float = 1.0e-6,
    ) -> ActionNormalizer:
        control_dim = int(controls.shape[-1]) if controls.ndim else 0
        if control_dim not in CONTROL_SCHEMAS:
            raise ValueError("controls must have three or five components")
        values = controls.detach().cpu().numpy() if _is_tensor(controls) else np.asarray(controls)
        flat = np.asarray(values, dtype=np.float64).reshape(-1, control_dim)
        if flat.shape[0] == 0 or not np.isfinite(flat).all():
            raise ValueError("cannot fit ActionNormalizer on empty or non-finite controls")
        return cls(
            tuple(float(value) for value in flat.min(axis=0)),
            tuple(float(value) for value in flat.max(axis=0)),
            float(epsilon),
        )

    def _bounds(self, value: np.ndarray | torch.Tensor):
        if _is_tensor(value):
            lo = torch.as_tensor(self.minimum, dtype=value.dtype, device=value.device)
            hi = torch.as_tensor(self.maximum, dtype=value.dtype, device=value.device)
            return lo, torch.clamp(hi - lo, min=self.epsilon)
        array = np.asarray(value)
        lo = np.asarray(self.minimum, dtype=array.dtype)
        hi = np.asarray(self.maximum, dtype=array.dtype)
        return lo, np.maximum(hi - lo, self.epsilon)

    def normalize(self, controls: np.ndarray | torch.Tensor, *, clip: bool = False):
        _check_last_dim(controls, len(self.minimum), "controls")
        lo, span = self._bounds(controls)
        result = 2.0 * (controls - lo) / span - 1.0
        if clip:
            return result.clamp(-1.0, 1.0) if _is_tensor(result) else np.clip(result, -1.0, 1.0)
        return result

    def denormalize(self, normalized: np.ndarray | torch.Tensor, *, clip: bool = True):
        _check_last_dim(normalized, len(self.minimum), "normalized")
        value = normalized
        if clip:
            value = value.clamp(-1.0, 1.0) if _is_tensor(value) else np.clip(value, -1.0, 1.0)
        lo, span = self._bounds(value)
        return (value + 1.0) * 0.5 * span + lo

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "componentwise_minmax_minus_one_to_one",
            "control_schema": list(CONTROL_SCHEMAS[len(self.minimum)]),
            "minimum": list(self.minimum),
            "maximum": list(self.maximum),
            "epsilon": self.epsilon,
            "fit_population": "successful_world_model_training_partition_only",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ActionNormalizer:
        return cls(
            tuple(float(value) for value in payload["minimum"]),
            tuple(float(value) for value in payload["maximum"]),
            float(payload.get("epsilon", 1.0e-6)),
        )


@dataclass(frozen=True)
class FeatureNormalizer:
    """Frozen policy-side z-score for embeddings and episode gravity."""

    embedding_mean: tuple[float, ...]
    embedding_std: tuple[float, ...]
    gravity_mean: float
    gravity_std: float
    epsilon: float = 1.0e-6
    gravity_kind: str = TRAIN_ZSCORE_GRAVITY

    def __post_init__(self) -> None:
        if not self.embedding_mean or len(self.embedding_mean) != len(self.embedding_std):
            raise ValueError("embedding normalization dimensions differ or are empty")
        values = (*self.embedding_mean, *self.embedding_std, self.gravity_mean, self.gravity_std)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("feature normalization values must be finite")
        if any(value <= 0 for value in self.embedding_std) or self.gravity_std <= 0:
            raise ValueError("feature normalization standard deviations must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.gravity_kind not in GRAVITY_NORMALIZATIONS:
            raise ValueError(f"unsupported gravity normalization: {self.gravity_kind!r}")

    def normalize_embeddings(self, value: np.ndarray | torch.Tensor):
        if _is_tensor(value):
            mean = torch.as_tensor(self.embedding_mean, dtype=value.dtype, device=value.device)
            std = torch.as_tensor(self.embedding_std, dtype=value.dtype, device=value.device)
            return (value - mean) / std.clamp_min(self.epsilon)
        array = np.asarray(value)
        mean = np.asarray(self.embedding_mean, dtype=array.dtype)
        std = np.asarray(self.embedding_std, dtype=array.dtype)
        return (array - mean) / np.maximum(std, self.epsilon)

    def normalize_gravity(self, value: float | np.ndarray | torch.Tensor):
        return (value - self.gravity_mean) / max(self.gravity_std, self.epsilon)

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray | torch.Tensor,
        gravity: np.ndarray | torch.Tensor,
        epsilon: float = 1.0e-6,
        gravity_kind: str = TRAIN_ZSCORE_GRAVITY,
    ) -> FeatureNormalizer:
        features = (
            embeddings.detach().cpu().numpy() if _is_tensor(embeddings) else np.asarray(embeddings)
        )
        values = gravity.detach().cpu().numpy() if _is_tensor(gravity) else np.asarray(gravity)
        features = np.asarray(features, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if features.ndim != 2 or not len(features) or len(values) == 0:
            raise ValueError("normalization inputs must be non-empty [N,D] arrays")
        if not np.isfinite(features).all() or not np.isfinite(values).all():
            raise ValueError("normalization inputs must be finite")
        if gravity_kind not in GRAVITY_NORMALIZATIONS:
            raise ValueError(f"unsupported gravity normalization: {gravity_kind!r}")
        if gravity_kind == FIXED_PADDLE_GRAVITY:
            gravity_mean, gravity_std = 10.0, 10.0
        else:
            gravity_mean = float(values.mean())
            gravity_std = float(max(values.std(), epsilon))
        return cls(
            tuple(float(value) for value in features.mean(axis=0)),
            tuple(float(value) for value in np.maximum(features.std(axis=0), epsilon)),
            gravity_mean,
            gravity_std,
            float(epsilon),
            gravity_kind,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureNormalizer:
        return cls(
            tuple(float(value) for value in payload["embedding_mean"]),
            tuple(float(value) for value in payload["embedding_std"]),
            float(payload["gravity_mean"]),
            float(payload["gravity_std"]),
            float(payload.get("epsilon", 1.0e-6)),
            str(payload.get("gravity_kind", TRAIN_ZSCORE_GRAVITY)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_mean": list(self.embedding_mean),
            "embedding_std": list(self.embedding_std),
            "gravity_mean": self.gravity_mean,
            "gravity_std": self.gravity_std,
            "gravity_kind": self.gravity_kind,
            "gravity_formula": (
                "(g-10)/10"
                if self.gravity_kind == FIXED_PADDLE_GRAVITY
                else "(g-successful_train_mean)/successful_train_std"
            ),
            "epsilon": self.epsilon,
        }
