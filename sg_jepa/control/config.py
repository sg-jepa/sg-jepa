"""Latent-GRU Diffusion Policy configuration for public control tasks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from .actions import FIXED_PADDLE_GRAVITY, GRAVITY_NORMALIZATIONS, TRAIN_ZSCORE_GRAVITY

PROJECTED_CLS_FEATURE_MODE = "projected_cls"
DIRECT_CONDITION = "projected_cls_gravity"
DINO_MEAN_PATCH_FEATURE_MODE = "dinov2_mean_patch"
DINO_DIRECT_CONDITION = "dinov2_mean_patch_gravity"
FEATURE_CONDITIONS = {
    PROJECTED_CLS_FEATURE_MODE: DIRECT_CONDITION,
    DINO_MEAN_PATCH_FEATURE_MODE: DINO_DIRECT_CONDITION,
}
UNIFORM_BATCH_SAMPLING = "uniform_epoch_without_replacement"
PAPER_PADDLE_BATCH_SAMPLING = "paper_paddle_two_stream_v1"
PAPER_FRANKA_BATCH_SAMPLING = "franka_strike_balanced_v1"
STORED_ACTION_RECONSTRUCTION = "stored"
FRANKA_ACTION_RECONSTRUCTION = "full_range_pose_delta_v2"


@dataclass(frozen=True)
class PolicyConfig:
    observation_horizon: int = 20
    action_horizon: int = 16
    execution_horizon: int = 8
    embedding_dim: int = 256
    action_dim: int = 5
    diffusion_steps: int = 100
    diffusion_step_embed_dim: int = 256
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    clip_sample: bool = True
    feature_mode: str = PROJECTED_CLS_FEATURE_MODE
    condition_spec: str = DIRECT_CONDITION
    control_codec: str = "direct"
    max_tilt_theta: float = math.pi / 6.0
    history_contract: str = "repeat_frame_zero_v1"
    observation_adapter: str = "latent_gru"
    adapter_hidden_dim: int = 64
    adapter_output_dim: int = 16
    gru_hidden_dim: int = 256
    gru_num_layers: int = 2
    gru_dropout: float = 0.0
    cross_attention_memory_dim: int = 256
    cross_attention_heads: int = 4
    cross_attention_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.observation_horizon <= 0 or self.action_horizon <= 0:
            raise ValueError("observation_horizon and action_horizon must be positive")
        if not 0 < self.execution_horizon <= self.action_horizon:
            raise ValueError("execution_horizon must lie in [1, action_horizon]")
        if self.embedding_dim <= 0 or self.action_dim not in {3, 5}:
            raise ValueError("control policies require embedding_dim > 0 and action_dim in {3,5}")
        if self.feature_mode not in FEATURE_CONDITIONS:
            raise ValueError(f"unsupported feature mode: {self.feature_mode}")
        expected_condition = FEATURE_CONDITIONS[self.feature_mode]
        if self.condition_spec != expected_condition:
            raise ValueError(f"{self.feature_mode} requires condition_spec={expected_condition}")
        if self.control_codec not in {"direct", "paddle_tilt_vector"}:
            raise ValueError(f"unsupported control_codec: {self.control_codec!r}")
        if self.control_codec == "paddle_tilt_vector" and self.action_dim != 5:
            raise ValueError("paddle_tilt_vector requires action_dim=5")
        if not 0 < self.max_tilt_theta <= math.pi / 2.0:
            raise ValueError("max_tilt_theta must lie in (0, pi/2]")
        if self.observation_adapter != "latent_gru":
            raise ValueError("this workflow is locked to the latent_gru adapter")
        if self.gru_hidden_dim <= 0 or self.gru_num_layers <= 0:
            raise ValueError("GRU dimensions must be positive")
        if not 0.0 <= self.gru_dropout < 1.0:
            raise ValueError("gru_dropout must lie in [0,1)")
        if self.gru_num_layers == 1 and self.gru_dropout != 0.0:
            raise ValueError("single-layer GRU requires gru_dropout=0")
        if self.history_contract != "repeat_frame_zero_v1":
            raise ValueError("unsupported history contract")
        if self.diffusion_steps <= 1:
            raise ValueError("diffusion_steps must be greater than one")
        if self.diffusion_step_embed_dim < 4 or self.diffusion_step_embed_dim % 2:
            raise ValueError("diffusion_step_embed_dim must be even and at least four")
        if len(self.down_dims) < 2 or any(value <= 0 for value in self.down_dims):
            raise ValueError("down_dims must contain at least two positive widths")
        if any(value % self.n_groups for value in self.down_dims):
            raise ValueError("every down_dims value must be divisible by n_groups")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        factor = 2 ** (len(self.down_dims) - 1)
        if self.action_horizon % factor:
            raise ValueError(f"action_horizon={self.action_horizon} must be divisible by {factor}")

    @property
    def observation_condition_dim(self) -> int:
        return self.gru_hidden_dim + 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["down_dims"] = list(self.down_dims)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyConfig:
        values = dict(payload)
        aliases = {
            "dino_mean_patch_384": DINO_MEAN_PATCH_FEATURE_MODE,
            "dino_mean_patch": DINO_MEAN_PATCH_FEATURE_MODE,
        }
        values["feature_mode"] = aliases.get(values.get("feature_mode"), values.get("feature_mode"))
        if values.get("history_contract") == "legacy_repeat_masked_v1":
            values["history_contract"] = "repeat_frame_zero_v1"
        rename = {
            "temporal_hidden_dim": "gru_hidden_dim",
            "temporal_num_layers": "gru_num_layers",
            "temporal_layers": "gru_num_layers",
            "temporal_dropout": "gru_dropout",
        }
        for old, new in rename.items():
            if old in values and new not in values:
                values[new] = values[old]
        if values.get("feature_mode") in FEATURE_CONDITIONS:
            values["condition_spec"] = FEATURE_CONDITIONS[values["feature_mode"]]
        allowed = {field.name for field in fields(cls)}
        values = {key: value for key, value in values.items() if key in allowed}
        values["down_dims"] = tuple(int(value) for value in values["down_dims"])
        return cls(**values)


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 300_000
    batch_size: int = 256
    learning_rate: float = 1.0e-4
    betas: tuple[float, float] = (0.95, 0.999)
    weight_decay: float = 1.0e-6
    warmup_steps: int = 500
    validate_every: int = 25_000
    checkpoint_every: int = 25_000
    max_val_batches: int | None = 16
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75
    ema_max_decay: float = 0.9999
    top_k: int = 12
    include_validation_in_gradient: bool = True
    batch_sampling: str = UNIFORM_BATCH_SAMPLING
    action_reconstruction: str = STORED_ACTION_RECONSTRUCTION
    quality_weights: str | None = None
    quality_weights_sha256: str | None = None
    seed: int = 42
    num_workers: int = 8
    precision: str = "bf16"
    gravity_normalization: str = TRAIN_ZSCORE_GRAVITY
    feature_cache_dtype: str = "fp16"

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if len(self.betas) != 2 or any(not 0.0 <= value < 1.0 for value in self.betas):
            raise ValueError("AdamW betas must contain two values in [0,1)")
        if self.warmup_steps < 0 or self.validate_every <= 0 or self.checkpoint_every <= 0:
            raise ValueError("invalid scheduling configuration")
        if self.top_k <= 0 or self.num_workers < 0:
            raise ValueError("top_k must be positive and num_workers non-negative")
        if not isinstance(self.include_validation_in_gradient, bool):
            raise TypeError("include_validation_in_gradient must be boolean")
        if self.batch_sampling not in {
            UNIFORM_BATCH_SAMPLING,
            PAPER_PADDLE_BATCH_SAMPLING,
            PAPER_FRANKA_BATCH_SAMPLING,
        }:
            raise ValueError(f"unsupported batch_sampling: {self.batch_sampling!r}")
        if self.action_reconstruction not in {
            STORED_ACTION_RECONSTRUCTION,
            FRANKA_ACTION_RECONSTRUCTION,
        }:
            raise ValueError(f"unsupported action_reconstruction: {self.action_reconstruction!r}")
        if self.batch_sampling == PAPER_PADDLE_BATCH_SAMPLING:
            if (
                self.batch_size != 256
                or not self.include_validation_in_gradient
                or self.precision != "bf16"
                or self.gravity_normalization != FIXED_PADDLE_GRAVITY
                or self.feature_cache_dtype != "fp16"
            ):
                raise ValueError(
                    "paper Paddle sampling requires batch_size=256, validation episodes in "
                    "the gradient population, BF16 training, FP16 cached features, and fixed "
                    "physical-range gravity normalization"
                )
            if not self.quality_weights or not self.quality_weights_sha256:
                raise ValueError(
                    "paper Paddle two-stream sampling requires quality weights and SHA256"
                )
            if len(self.quality_weights_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in self.quality_weights_sha256
            ):
                raise ValueError("quality_weights_sha256 must be a lowercase SHA256 digest")
        elif self.quality_weights is not None or self.quality_weights_sha256 is not None:
            raise ValueError("quality weights are only valid for paper Paddle sampling")
        if self.batch_sampling == PAPER_FRANKA_BATCH_SAMPLING:
            if (
                self.batch_size != 256
                or not self.include_validation_in_gradient
                or self.precision != "bf16"
                or self.gravity_normalization != TRAIN_ZSCORE_GRAVITY
                or self.feature_cache_dtype != "fp16"
                or self.action_reconstruction != FRANKA_ACTION_RECONSTRUCTION
            ):
                raise ValueError(
                    "paper Franka sampling requires batch_size=256, validation episodes in "
                    "the gradient population, BF16 training, FP16 cached features, train-zscore "
                    "gravity, and full-range pose-delta action reconstruction"
                )
        elif self.action_reconstruction == FRANKA_ACTION_RECONSTRUCTION:
            raise ValueError(
                "full-range Franka action reconstruction requires strike-balanced sampling"
            )
        if self.max_val_batches is not None and self.max_val_batches <= 0:
            raise ValueError("max_val_batches must be positive when provided")
        if self.ema_inv_gamma <= 0 or self.ema_power <= 0:
            raise ValueError("EMA inverse gamma and power must be positive")
        if not 0.0 <= self.ema_max_decay < 1.0:
            raise ValueError("ema_max_decay must lie in [0,1)")
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be bf16 or fp32")
        if self.gravity_normalization not in GRAVITY_NORMALIZATIONS:
            raise ValueError(f"unsupported gravity_normalization: {self.gravity_normalization!r}")
        if self.feature_cache_dtype not in {"fp16", "fp32"}:
            raise ValueError("feature_cache_dtype must be fp16 or fp32")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["betas"] = list(self.betas)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingConfig:
        values = dict(payload)
        values["betas"] = tuple(float(value) for value in values["betas"])
        return cls(**values)


def load_policy_bundle_config(
    path: str | Path,
) -> tuple[PolicyConfig, TrainingConfig, dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text())
    required = {"policy", "training", "evaluation"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"policy config must contain exactly {sorted(required)}")
    if not all(isinstance(payload[key], dict) for key in required):
        raise ValueError("policy, training, and evaluation sections must be mappings")
    evaluation = dict(payload["evaluation"])
    if evaluation.get("sampler") not in {"ddim", "ddpm"}:
        raise ValueError("evaluation sampler must be ddim or ddpm")
    evaluation.setdefault("precision", "fp32")
    if evaluation["precision"] not in {"bf16", "fp32"}:
        raise ValueError("evaluation precision must be bf16 or fp32")
    policy = PolicyConfig.from_dict(payload["policy"])
    training = TrainingConfig.from_dict(payload["training"])
    if training.batch_sampling == PAPER_PADDLE_BATCH_SAMPLING:
        if policy.control_codec != "paddle_tilt_vector":
            raise ValueError("paper Paddle sampling requires the paddle_tilt_vector codec")
        if (
            evaluation["sampler"] != "ddpm"
            or int(evaluation.get("inference_steps", -1)) != policy.diffusion_steps
            or evaluation["precision"] != "bf16"
        ):
            raise ValueError("paper Paddle evaluation requires full BF16 DDPM denoising")
    if training.batch_sampling == PAPER_FRANKA_BATCH_SAMPLING:
        if policy.control_codec != "direct" or policy.action_dim != 5:
            raise ValueError("paper Franka sampling requires five direct controls")
        if (
            evaluation["sampler"] != "ddim"
            or int(evaluation.get("inference_steps", -1)) != 10
            or evaluation["precision"] != "fp32"
        ):
            raise ValueError("paper Franka evaluation requires 10-step FP32 DDIM denoising")
    return policy, training, evaluation
