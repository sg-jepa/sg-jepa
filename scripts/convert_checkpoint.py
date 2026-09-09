"""Convert one verified project checkpoint into an inference-only artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file

from sg_jepa.checkpoints import sha256_file
from sg_jepa.control.config import PolicyConfig


def _load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # pragma: no cover - older PyTorch.
        return torch.load(path, map_location="cpu")


def _state(payload: Any, kind: str) -> Mapping[str, torch.Tensor]:
    if kind == "world_model" and isinstance(payload, Mapping):
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(payload.get(key), Mapping):
                return payload[key]
    keys = {
        "dino_world_model": "model_state_dict",
        "probe": "probe_state_dict",
        "policy": "policy_state_dict",
    }
    key = keys.get(kind)
    if key and isinstance(payload, Mapping) and isinstance(payload.get(key), Mapping):
        return payload[key]
    raise TypeError(f"cannot find the {kind} tensor state dictionary")


def _yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration must be a YAML mapping")
    return value


def _sidecar(
    artifact_id: str,
    record: dict[str, Any],
    payload: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    kind = record["kind"]
    result: dict[str, Any] = {
        "format_version": 1,
        "artifact_id": artifact_id,
        "kind": kind,
        "source": record["source"],
        "conversion_command": (
            "python scripts/convert_checkpoint.py "
            f"--artifact {artifact_id} --source /path/to/{record['source']['filename']} "
            "--output-dir /path/to/huggingface-repo"
        ),
        "external_dependencies": [],
    }
    if kind == "world_model":
        if not isinstance(config.get("model"), dict):
            raise ValueError("native world-model conversion requires --config with a model section")
        result["architecture"] = config["model"]
        if isinstance(record.get("action_statistics"), Mapping):
            result["action_statistics"] = dict(record["action_statistics"])
    elif kind == "dino_world_model":
        if not isinstance(config.get("predictor"), dict):
            raise ValueError("DINO-WM conversion requires --config with a predictor section")
        result["architecture"] = config["predictor"]
        result["external_dependencies"] = [
            {
                "name": "DINOv2 ViT-S/14",
                "source_revision": "85a24602099d397264d5b30461ad7f3bfd726ca1",
                "weights_sha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
            }
        ]
    elif kind == "probe":
        if not isinstance(payload, Mapping):
            raise TypeError("probe payload must be a mapping")
        target_spec = payload.get("target_spec", payload.get("target_profile"))
        if target_spec is None:
            raise ValueError("probe payload does not identify its target contract")
        architecture = {
            "probe_kind": payload["probe_kind"],
            "feature_dim": int(payload["feature_dim"]),
            "output_dim": int(payload["output_dim"]),
            "temporal_window": int(payload["temporal_window"]),
            "target_spec": target_spec,
        }
        for key in ("shape", "symmetry_order", "target_profile", "target_definition"):
            if key in payload:
                architecture[key] = payload[key]
        result["architecture"] = architecture
        result["target_stats"] = payload["target_stats"]
        result["checkpoint_kind"] = payload.get("kind", "mlp_state_probe")
    elif kind == "policy":
        if not isinstance(payload, Mapping):
            raise TypeError("policy payload must be a mapping")
        policy = PolicyConfig.from_dict(payload["policy_config"])
        evaluation = dict(record["evaluation"])
        policy_values = policy.to_dict()
        policy_values["execution_horizon"] = int(evaluation["execution_horizon"])
        contract = payload["data_contract"]
        if not isinstance(contract, Mapping):
            raise TypeError("policy data_contract must be a mapping")
        inference_contract = {
            key: contract[key]
            for key in (
                "feature_normalizer",
                "action_normalizer",
                "history_contract",
                "feature_mode",
                "controller_calibration",
            )
            if key in contract
        }
        if not {"feature_normalizer", "action_normalizer"}.issubset(inference_contract):
            raise ValueError("policy data contract is missing inference normalizers")
        result.update(
            {
                "architecture": policy_values,
                "data_contract": inference_contract,
                "evaluation": evaluation,
                "checkpoint_step": int(payload["step"]),
            }
        )
        if record["encoder"] == "dinov2-vits14-upstream":
            result["external_dependencies"] = [
                {
                    "name": "DINOv2 ViT-S/14",
                    "source_revision": "85a24602099d397264d5b30461ad7f3bfd726ca1",
                    "weights_sha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
                }
            ]
    else:  # pragma: no cover - manifest validation catches this.
        raise ValueError(f"unsupported artifact kind {kind!r}")
    return result


def convert(
    artifact_id: str,
    source: Path,
    output_dir: Path,
    *,
    config_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path or Path(__file__).parents[1] / "sg_jepa" / "pretrained.json"
    manifest = json.loads(manifest_path.read_text())
    try:
        record = manifest["artifacts"][artifact_id]
    except KeyError as exc:
        raise KeyError(f"unknown artifact {artifact_id!r}") from exc
    source = source.expanduser().resolve()
    expected = record["source"]
    if source.stat().st_size != int(expected["bytes"]):
        raise ValueError("source checkpoint byte size differs from the release manifest")
    if sha256_file(source) != expected["sha256"]:
        raise ValueError("source checkpoint SHA-256 differs from the release manifest")
    payload = _load(source)
    source_state = _state(payload, record["kind"])
    tensors = {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in source_state.items()
        if torch.is_tensor(tensor)
    }
    if len(tensors) != len(source_state):
        raise TypeError("state dictionary contains non-tensor values")
    if any(
        torch.is_floating_point(tensor) and not torch.isfinite(tensor).all()
        for tensor in tensors.values()
    ):
        raise ValueError("source state dictionary contains non-finite tensors")
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = output_dir / record["filename"]
    sidecar_path = output_dir / record["config_filename"]
    weights.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    if weights.exists() or sidecar_path.exists():
        raise FileExistsError("refusing to overwrite an existing converted artifact")
    save_file(tensors, str(weights))
    restored = load_file(str(weights), device="cpu")
    if set(restored) != set(tensors):
        raise RuntimeError("converted tensor names differ from the source")
    for name, expected_tensor in tensors.items():
        if not torch.equal(restored[name], expected_tensor):
            raise RuntimeError(f"converted tensor differs from source: {name}")
    sidecar = _sidecar(artifact_id, record, payload, _yaml(config_path))
    if config_path is not None:
        sidecar["conversion_command"] += " --config /path/to/config.yaml"
    if record["kind"] == "policy":
        encoder_id = str(record["encoder"])
        sidecar["encoder"] = encoder_id
        if encoder_id == "dinov2-vits14-upstream":
            sidecar["encoder_source_sha256"] = manifest["dinov2"]["weights_sha256"]
        else:
            sidecar["encoder_source_sha256"] = manifest["artifacts"][encoder_id]["source"]["sha256"]
    sidecar["converted"] = {
        "filename": weights.name,
        "bytes": weights.stat().st_size,
        "sha256": sha256_file(weights),
        "tensor_count": len(tensors),
        "tensor_equality": "exact",
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    report = convert(
        args.artifact,
        args.source,
        args.output_dir,
        config_path=args.config,
        manifest_path=args.manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
