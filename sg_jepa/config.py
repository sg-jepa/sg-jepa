"""Typed experiment configuration with strict, small YAML contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(frozen=True)
class WorldModelConfig:
    image_size: int = 128
    patch_size: int = 8
    encoder_scale: str = "tiny"
    encoder_hidden_size: int | None = None
    encoder_heads: int | None = None
    encoder_layers: int | None = None
    embed_dim: int = 256
    action_dim: int = 3
    history_size: int = 20
    predictor_kind: str = "gru"
    predictor_hidden_dim: int = 512
    predictor_depth: int = 3
    predictor_heads: int = 16
    predictor_mlp_dim: int = 2048
    predictor_dim_head: int = 64
    predictor_dropout: float = 0.1
    predictor_emb_dropout: float = 0.0
    predictor_conditioning: str = "concat"
    predictor_residual_init: float = 0.1
    ssm_state_dim: int = 16
    ssm_conv_kernel: int = 4
    ssm_expand: int = 2
    ssm_dt_rank: int | None = None
    projector_hidden_dim: int = 2048

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.image_size % self.patch_size:
            raise ValueError("image_size must be positive and divisible by patch_size")
        if self.action_dim <= 0 or self.embed_dim <= 0 or self.history_size <= 0:
            raise ValueError("action_dim, embed_dim, and history_size must be positive")
        if self.predictor_kind not in {"transformer", "gru", "ssm"}:
            raise ValueError("predictor_kind must be transformer, gru, or ssm")


@dataclass(frozen=True)
class ObjectiveConfig:
    prediction_weight: float = 0.0
    rollout_weight: float = 1.0
    rollout_horizon: int = 5
    rollout_gamma: float = 0.95
    target_detach: bool = False
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024

    def __post_init__(self) -> None:
        if min(self.prediction_weight, self.rollout_weight, self.sigreg_weight) < 0:
            raise ValueError("objective weights must be non-negative")
        if self.rollout_weight > 0 and self.rollout_horizon <= 0:
            raise ValueError("positive rollout_weight requires rollout_horizon > 0")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    max_steps: int | None = None
    batch_size: int = 2
    learning_rate: float = 5.0e-5
    muon_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-3
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_rms_match_scale: float = 0.4
    lr_schedule: str = "constant"
    lr_warmup_epochs: float = 0.0
    lr_cooldown_start_fraction: float = 0.8
    lr_min_factor: float = 0.1
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    train_fraction: float = 0.9
    num_workers: int = 0
    drop_last: bool = False
    max_val_batches: int | None = None
    checkpoint_every_epochs: int = 1
    seed: int = 42
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.lr_schedule not in {"constant", "warmup_cosine_cooldown"}:
            raise ValueError("lr_schedule must be constant or warmup_cosine_cooldown")
        if self.lr_warmup_epochs < 0.0:
            raise ValueError("lr_warmup_epochs must be non-negative")
        if not 0.0 <= self.lr_cooldown_start_fraction <= 1.0:
            raise ValueError("lr_cooldown_start_fraction must lie in [0,1]")
        if not 0.0 <= self.lr_min_factor <= 1.0:
            raise ValueError("lr_min_factor must lie in [0,1]")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie strictly between zero and one")
        if self.num_workers < 0 or self.checkpoint_every_epochs <= 0:
            raise ValueError("num_workers must be non-negative and checkpoint cadence positive")
        if self.max_val_batches is not None and self.max_val_batches <= 0:
            raise ValueError("max_val_batches must be positive when provided")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: WorldModelConfig
    objective: ObjectiveConfig
    train: TrainConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _strict_dataclass(cls: type[T], values: dict[str, Any], section: str) -> T:
    allowed = {field.name for field in fields(cls)}
    extra = set(values) - allowed
    if extra:
        raise ValueError(f"unknown {section} keys: {sorted(extra)}")
    return cls(**values)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    extra = set(payload) - {"name", "model", "objective", "train"}
    if extra:
        raise ValueError(f"unknown top-level configuration keys: {sorted(extra)}")
    return ExperimentConfig(
        name=str(payload["name"]),
        model=_strict_dataclass(WorldModelConfig, payload["model"], "model"),
        objective=_strict_dataclass(ObjectiveConfig, payload["objective"], "objective"),
        train=_strict_dataclass(TrainConfig, payload["train"], "train"),
    )
