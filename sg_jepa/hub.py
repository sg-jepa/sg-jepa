"""Small Hugging Face interface for released Semigroup-JEPA artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sg_jepa.baselines import PredictorConfig, build_dino_world_model
from sg_jepa.checkpoints import load_state_dict
from sg_jepa.config import WorldModelConfig
from sg_jepa.control.config import PolicyConfig
from sg_jepa.control.policy import build_policy_model
from sg_jepa.evaluation.frozen_approach import FrozenProbe, build_approach_probe
from sg_jepa.models import build_world_model


@dataclass(frozen=True)
class PretrainedPolicy:
    """Loaded policy network and the metadata required for inference."""

    model: nn.Module
    config: PolicyConfig
    data_contract: dict[str, Any]
    evaluation: dict[str, Any]
    encoder: nn.Module | None
    encoder_id: str
    artifact_id: str


def _manifest() -> dict[str, Any]:
    path = files("sg_jepa").joinpath("pretrained.json")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("artifacts"), dict):
        raise RuntimeError("bundled pretrained manifest is invalid")
    return payload


def available_pretrained() -> tuple[str, ...]:
    """Return stable IDs accepted by the pretrained loaders."""

    return tuple(sorted(_manifest()["artifacts"]))


def _record(artifact_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _manifest()
    try:
        record = manifest["artifacts"][artifact_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown artifact {artifact_id!r}; choose from {', '.join(available_pretrained())}"
        ) from exc
    return manifest, record


def _download_pair(
    artifact_id: str,
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError("install sg-jepa[hub] to download pretrained artifacts") from exc
    manifest, record = _record(artifact_id)
    organization = str(manifest["organization"])
    repo_id = str(record["repo_id"])
    repo_type = str(record.get("repo_type", manifest.get("repo_type", "model")))
    revisions = manifest.get("repository_revisions", {})
    revision = str(revisions.get(repo_id, manifest.get("revision", "")))
    if organization == "ORG" or not revision or revision.startswith("HF_COMMIT"):
        raise RuntimeError(
            "Hugging Face release coordinates are not finalized; replace the HF_COMMIT "
            "value in sg_jepa/pretrained.json after uploading the repository"
        )
    common = {
        "repo_id": repo_id.replace("ORG", organization, 1),
        "repo_type": repo_type,
        "revision": revision,
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "local_files_only": local_files_only,
    }
    weights = Path(hf_hub_download(filename=record["filename"], **common))
    config_path = Path(hf_hub_download(filename=record["config_filename"], **common))
    sidecar = json.loads(config_path.read_text())
    if sidecar.get("artifact_id") != artifact_id:
        raise ValueError("downloaded sidecar artifact ID differs from the manifest")
    if sidecar.get("source", {}).get("sha256") != record["source"]["sha256"]:
        raise ValueError("downloaded sidecar source hash differs from the manifest")
    return weights, sidecar, record


def load_pretrained(
    artifact_id: str,
    *,
    device: str | torch.device = "cpu",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> nn.Module:
    """Download and strictly load a released world model."""

    weights, sidecar, record = _download_pair(
        artifact_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    architecture = sidecar.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("world-model sidecar has no architecture mapping")
    if record["kind"] == "world_model":
        model = build_world_model(WorldModelConfig(**architecture))
    elif record["kind"] == "dino_world_model":
        model = build_dino_world_model(PredictorConfig.from_dict(architecture))
    else:
        raise ValueError(f"{artifact_id!r} is not a world-model artifact")
    model.load_state_dict(load_state_dict(weights), strict=True)
    model.to(torch.device(device)).eval().requires_grad_(False)
    model.pretrained_metadata = sidecar  # type: ignore[attr-defined]
    return model


def load_probe(
    artifact_id: str,
    *,
    device: str | torch.device = "cpu",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> FrozenProbe:
    """Load the paper MLP state probe associated with a model ID."""

    _, record = _record(artifact_id)
    if record["kind"] != "probe":
        artifact_id = str(record.get("probe", ""))
        if not artifact_id:
            raise ValueError("this artifact has no released probe")
    weights, sidecar, record = _download_pair(
        artifact_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if record["kind"] != "probe":
        raise ValueError(f"{artifact_id!r} is not a probe artifact")
    architecture = sidecar["architecture"]
    feature_dim = int(architecture["feature_dim"])
    output_dim = int(architecture["output_dim"])
    temporal_window = int(architecture["temporal_window"])
    model = build_approach_probe(feature_dim, output_dim)
    model.load_state_dict(load_state_dict(weights), strict=True)
    selected_device = torch.device(device)
    model.to(selected_device).eval().requires_grad_(False)
    statistics = sidecar["target_stats"]
    return FrozenProbe(
        model=model,
        feature_dim=feature_dim,
        temporal_window=temporal_window,
        target_mean=torch.as_tensor(
            statistics["mean"], dtype=torch.float32, device=selected_device
        ),
        target_std=torch.as_tensor(statistics["std"], dtype=torch.float32, device=selected_device),
        checkpoint_kind=str(sidecar.get("checkpoint_kind", "mlp_state_probe")),
    )


def load_policy(
    artifact_id: str,
    *,
    device: str | torch.device = "cpu",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> PretrainedPolicy:
    """Load a released diffusion policy and its native encoder when available."""

    _, record = _record(artifact_id)
    if record["kind"] != "policy":
        artifact_id = str(record.get("policy", ""))
        if not artifact_id:
            raise ValueError("this artifact has no released policy")
    weights, sidecar, record = _download_pair(
        artifact_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    config = PolicyConfig.from_dict(sidecar["architecture"])
    model = build_policy_model(config)
    model.load_state_dict(load_state_dict(weights), strict=True)
    selected_device = torch.device(device)
    model.to(selected_device).eval().requires_grad_(False)
    encoder_id = str(record["encoder"])
    encoder = None
    if encoder_id != "dinov2-vits14-upstream":
        encoder = load_pretrained(
            encoder_id,
            device=selected_device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    return PretrainedPolicy(
        model=model,
        config=config,
        data_contract=dict(sidecar["data_contract"]),
        evaluation=dict(sidecar["evaluation"]),
        encoder=encoder,
        encoder_id=encoder_id,
        artifact_id=artifact_id,
    )


__all__ = [
    "PretrainedPolicy",
    "available_pretrained",
    "load_policy",
    "load_pretrained",
    "load_probe",
]
