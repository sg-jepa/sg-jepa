"""Optional, separately attributed baselines used by the paper."""

from .checkpoints import load_dino_config, strict_load_dino
from .dino_wm import DinoWorldModel, PredictorConfig, build_dino_world_model
from .dinov2 import (
    DINOV2_SOURCE_COMMIT,
    DINOV2_WEIGHTS_SHA256,
    FrozenDinoV2Encoder,
    Native128Preprocessor,
    load_dinov2_encoder,
    provision_dinov2_artifact,
    verify_dinov2_artifact,
)

__all__ = [
    "DinoWorldModel",
    "DINOV2_SOURCE_COMMIT",
    "DINOV2_WEIGHTS_SHA256",
    "FrozenDinoV2Encoder",
    "Native128Preprocessor",
    "PredictorConfig",
    "build_dino_world_model",
    "load_dino_config",
    "load_dinov2_encoder",
    "provision_dinov2_artifact",
    "strict_load_dino",
    "verify_dinov2_artifact",
]
