"""Frozen world-model plus diffusion-policy inference for control tasks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sg_jepa.baselines import (
    DINOV2_WEIGHTS_SHA256,
    Native128Preprocessor,
    load_dinov2_encoder,
)
from sg_jepa.checkpoints import load_state_dict, sha256_file, strict_load
from sg_jepa.config import load_experiment_config
from sg_jepa.models import build_world_model

from .actions import (
    CONTROL_SCHEMAS,
    ActionNormalizer,
    FeatureNormalizer,
    controls_to_raw_actions,
)
from .config import PolicyConfig, load_policy_bundle_config
from .diffusion import GaussianDiffusion1D
from .policy import build_policy_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_policy_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        metadata = json.loads(path.with_suffix(".json").read_text())
        payload = {
            "kind": "diffusion_policy",
            "step": metadata["checkpoint_step"],
            "policy_config": metadata["architecture"],
            "policy_state_dict": load_state_dict(path),
            "data_contract": metadata["data_contract"],
            "world_model": {
                "weights_sha256": metadata.get("encoder_source_sha256"),
            },
        }
    else:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:  # pragma: no cover - old PyTorch compatibility.
            payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("policy checkpoint must contain a mapping")
    required = {"kind", "step", "policy_config", "policy_state_dict", "data_contract"}
    if not required.issubset(payload):
        raise TypeError(f"policy checkpoint is missing {sorted(required - set(payload))}")
    accepted = {
        "diffusion_policy",
        "arm_catcher_ball_diffusion_policy",
        "franka_paddle_hit_ball_to_basket_diffusion_policy",
        "arm_paddle_ball_diffusion_policy",
    }
    if payload["kind"] not in accepted:
        raise ValueError(f"unsupported diffusion-policy kind: {payload['kind']!r}")
    return payload


def _checkpoint_source_identity(path: Path) -> str:
    """Return the pre-conversion identity recorded by a safetensors sidecar."""

    if path.suffix == ".safetensors":
        metadata_path = path.with_suffix(".json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            source = metadata.get("source")
            if isinstance(source, dict) and isinstance(source.get("sha256"), str):
                return source["sha256"]
    return sha256_file(path)


def _native_images(
    frames: np.ndarray,
    *,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(frames))
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"expected RGB frames [B,H,W,3], got {tuple(value.shape)}")
    value = value.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).div_(255.0)
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


@dataclass(frozen=True)
class FrozenObservationEncoder:
    model: nn.Module
    feature_mode: str
    embedding_dim: int
    device: torch.device
    checkpoint_sha256: str
    image_size: int = 256

    @torch.inference_mode()
    def encode_rgb_frames(self, frames: np.ndarray) -> torch.Tensor:
        pixels = np.asarray(frames)
        if pixels.ndim != 4 or pixels.shape[-1] != 3 or not len(pixels):
            raise ValueError(f"expected non-empty RGB frames [B,H,W,3], got {pixels.shape}")
        if self.feature_mode == "projected_cls":
            images = _native_images(pixels, image_size=self.image_size, device=self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                features = self.model.encode({"pixels": images.unsqueeze(1)})["emb"][:, 0]
            features = features.float()
        elif self.feature_mode == "dinov2_mean_patch":
            images = Native128Preprocessor()(pixels).to(self.device)
            patch_tokens = self.model(images).to(torch.float16)
            features = patch_tokens.mean(dim=1, dtype=torch.float16).float()
        else:  # pragma: no cover - construction prevents this path.
            raise ValueError(f"unsupported feature mode {self.feature_mode}")
        expected = (len(pixels), self.embedding_dim)
        if tuple(features.shape) != expected or not torch.isfinite(features).all():
            raise RuntimeError(
                f"observation encoder returned {tuple(features.shape)}, expected {expected}"
            )
        return features


@dataclass
class PolicyBundle:
    model: nn.Module
    diffusion: GaussianDiffusion1D
    observation_encoder: FrozenObservationEncoder
    config: PolicyConfig
    feature_normalizer: FeatureNormalizer
    action_normalizer: ActionNormalizer
    device: torch.device
    policy_checkpoint: Path
    policy_checkpoint_sha256: str
    checkpoint_step: int
    sampler: str = "ddim"
    inference_steps: int = 10
    eta: float = 0.0
    inference_precision: str = "fp32"

    @torch.inference_mode()
    def sample_features(
        self,
        features: torch.Tensor,
        gravity: float,
        *,
        seed: int,
    ) -> np.ndarray:
        horizon = self.config.observation_horizon
        features = features[-horizon:]
        if features.ndim != 2 or features.shape[1] != self.config.embedding_dim:
            raise ValueError("policy features have an incompatible shape")
        if features.shape[0] < horizon:
            features = torch.cat(
                (features[0:1].expand(horizon - features.shape[0], -1), features), dim=0
            )
        embeddings = self.feature_normalizer.normalize_embeddings(features).unsqueeze(0)
        embeddings = embeddings.to(self.device)
        normalized_gravity = self.feature_normalizer.normalize_gravity(float(gravity))
        gravity_condition = torch.tensor(
            [[normalized_gravity]], dtype=embeddings.dtype, device=self.device
        )
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.inference_precision == "bf16" and self.device.type == "cuda",
        ):
            normalized = self.diffusion.sample(
                self.model,
                embeddings,
                gravity_condition,
                action_horizon=self.config.action_horizon,
                action_dim=self.config.action_dim,
                generator=generator,
                sampler=self.sampler,
                num_inference_steps=self.inference_steps,
                eta=self.eta,
            )[0]
        controls = self.action_normalizer.denormalize(normalized.float(), clip=True)
        raw = controls_to_raw_actions(
            controls,
            float(gravity),
            xyz_limit=1.0 if self.config.action_dim == 3 else None,
            control_codec=self.config.control_codec,
            max_tilt_theta=self.config.max_tilt_theta,
        )
        result = raw.detach().cpu().numpy().astype(np.float32, copy=False)
        expected = (self.config.action_horizon, self.config.action_dim + 1)
        if result.shape != expected or not np.isfinite(result).all():
            raise RuntimeError(f"policy returned invalid raw actions {result.shape}")
        return result

    def start_episode(self, gravity: float) -> PolicyEpisodeState:
        return PolicyEpisodeState(self, float(gravity))

    def provenance(self) -> dict[str, Any]:
        return {
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "checkpoint_step": self.checkpoint_step,
            "feature_mode": self.config.feature_mode,
            "world_model_checkpoint_sha256": self.observation_encoder.checkpoint_sha256,
            "sampler": self.sampler,
            "inference_steps": self.inference_steps,
            "eta": self.eta,
            "inference_precision": self.inference_precision,
            "control_codec": self.config.control_codec,
            "max_tilt_theta": self.config.max_tilt_theta,
            "control_schema": list(CONTROL_SCHEMAS[self.config.action_dim]),
        }


@dataclass
class PolicyEpisodeState:
    bundle: PolicyBundle
    gravity: float

    def __post_init__(self) -> None:
        self.features: list[torch.Tensor] = []

    @torch.inference_mode()
    def observe(self, frame: np.ndarray) -> None:
        value = np.asarray(frame)
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError("RGB observation must have shape [H,W,3]")
        feature = self.bundle.observation_encoder.encode_rgb_frames(value[None])[0]
        self.features.append(feature.float())
        self.features = self.features[-self.bundle.config.observation_horizon :]

    def advance(self, raw_action: np.ndarray | Sequence[float]) -> None:
        value = np.asarray(raw_action, dtype=np.float32).reshape(self.bundle.config.action_dim + 1)
        if not np.isclose(value[0], self.gravity, atol=1.0e-6, rtol=0.0):
            raise ValueError("executed action gravity differs from episode gravity")

    def predict_action(self, *, seed: int) -> np.ndarray:
        if not self.features:
            raise RuntimeError("observe the initial frame before requesting an action")
        return self.bundle.sample_features(torch.stack(self.features), self.gravity, seed=seed)


def load_policy_bundle(
    policy_checkpoint: str | Path,
    policy_config: str | Path,
    *,
    world_model_checkpoint: str | Path | None = None,
    world_model_config: str | Path | None = None,
    device: str | torch.device = "cuda",
    dinov2_root: str | Path | None = None,
) -> PolicyBundle:
    """Compose a released Franka or Paddle policy with its frozen encoder."""

    policy_checkpoint = Path(policy_checkpoint).expanduser().resolve()
    payload = _load_policy_payload(policy_checkpoint)
    sidecar, _training, evaluation = load_policy_bundle_config(policy_config)
    embedded = PolicyConfig.from_dict(payload["policy_config"])
    # Execution horizon is a deployment choice and does not alter policy weights.
    embedded_architecture = embedded.to_dict()
    sidecar_architecture = sidecar.to_dict()
    # Execution horizon and the simulator-facing control codec are deployment
    # contracts; neither changes the policy tensor architecture. Historical
    # release sidecars predate the explicit Paddle codec field.
    for deployment_field in ("execution_horizon", "control_codec"):
        embedded_architecture.pop(deployment_field)
        sidecar_architecture.pop(deployment_field)
    if embedded_architecture != sidecar_architecture:
        raise ValueError("policy sidecar architecture differs from the checkpoint")
    state = payload["policy_state_dict"]
    non_finite = [
        name
        for name, value in state.items()
        if torch.is_floating_point(value) and not torch.isfinite(value).all()
    ]
    if non_finite:
        raise ValueError(f"policy contains non-finite tensors: {non_finite[:5]}")
    selected_device = torch.device(device)
    policy_model = build_policy_model(embedded)
    policy_model.load_state_dict(state, strict=True)
    policy_model.to(selected_device).eval().requires_grad_(False)

    expected_world = payload.get("world_model")
    if not isinstance(expected_world, dict):
        raise TypeError("policy checkpoint has no world_model provenance")
    feature_mode = sidecar.feature_mode
    if feature_mode == "projected_cls":
        if world_model_checkpoint is None or world_model_config is None:
            raise ValueError("native policy inference requires an encoder checkpoint and config")
        world_model_checkpoint = Path(world_model_checkpoint).expanduser().resolve()
        world_hash = _checkpoint_source_identity(world_model_checkpoint)
        if world_hash != expected_world.get("weights_sha256"):
            raise ValueError("policy and native world-model checkpoint hashes differ")
        experiment = load_experiment_config(world_model_config)
        expected_raw_action_dim = embedded.action_dim + 1
        if (
            experiment.model.action_dim != expected_raw_action_dim
            or experiment.model.embed_dim != embedded.embedding_dim
        ):
            raise ValueError("native world-model sidecar differs from policy features")
        world_model = build_world_model(experiment.model)
        strict_load(world_model, world_model_checkpoint)
        world_model.to(selected_device).eval().requires_grad_(False)
        observation_encoder = FrozenObservationEncoder(
            model=world_model,
            feature_mode=feature_mode,
            embedding_dim=embedded.embedding_dim,
            device=selected_device,
            checkpoint_sha256=world_hash,
            image_size=experiment.model.image_size,
        )
    elif feature_mode == "dinov2_mean_patch":
        if dinov2_root is None:
            raise ValueError("DINO policy inference requires dinov2_root")
        dino_encoder = load_dinov2_encoder(dinov2_root, device=selected_device)
        world_hash = DINOV2_WEIGHTS_SHA256
        observation_encoder = FrozenObservationEncoder(
            model=dino_encoder,
            feature_mode=feature_mode,
            embedding_dim=embedded.embedding_dim,
            device=selected_device,
            checkpoint_sha256=world_hash,
            image_size=112,
        )
    else:  # pragma: no cover - PolicyConfig rejects other modes.
        raise ValueError(f"unsupported feature mode {feature_mode}")

    contract = payload["data_contract"]
    if not isinstance(contract, dict):
        raise TypeError("policy data_contract must be a mapping")
    if "world_model" in contract and contract["world_model"] != expected_world:
        raise ValueError("policy data contract has inconsistent world-model provenance")
    feature_normalizer = FeatureNormalizer.from_dict(contract["feature_normalizer"])
    action_normalizer = ActionNormalizer.from_dict(contract["action_normalizer"])
    if len(feature_normalizer.embedding_mean) != embedded.embedding_dim:
        raise ValueError("feature normalizer dimension differs from policy config")
    sampler = str(evaluation["sampler"])
    inference_steps = int(evaluation["inference_steps"])
    eta = float(evaluation["eta"])
    diffusion = GaussianDiffusion1D(embedded.diffusion_steps, clip_sample=embedded.clip_sample).to(
        selected_device
    )
    return PolicyBundle(
        model=policy_model,
        diffusion=diffusion,
        observation_encoder=observation_encoder,
        config=sidecar,
        feature_normalizer=feature_normalizer,
        action_normalizer=action_normalizer,
        device=selected_device,
        policy_checkpoint=policy_checkpoint,
        policy_checkpoint_sha256=sha256_file(policy_checkpoint),
        checkpoint_step=int(payload["step"]),
        sampler=sampler,
        inference_steps=inference_steps,
        eta=eta,
        inference_precision=str(evaluation["precision"]),
    )


__all__ = [
    "FrozenObservationEncoder",
    "PolicyBundle",
    "PolicyEpisodeState",
    "load_policy_bundle",
]
