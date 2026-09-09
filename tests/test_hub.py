import json
from pathlib import Path

import pytest

import sg_jepa.hub as hub
from sg_jepa.config import WorldModelConfig
from sg_jepa.models import build_world_model


def _tiny_architecture() -> dict[str, object]:
    return {
        "image_size": 16,
        "patch_size": 8,
        "encoder_scale": "tiny",
        "encoder_hidden_size": 8,
        "encoder_heads": 2,
        "encoder_layers": 1,
        "embed_dim": 8,
        "action_dim": 1,
        "history_size": 2,
        "predictor_kind": "gru",
        "predictor_hidden_dim": 8,
        "predictor_depth": 1,
        "predictor_heads": 2,
        "predictor_mlp_dim": 16,
        "predictor_dim_head": 4,
        "predictor_dropout": 0.0,
        "predictor_emb_dropout": 0.0,
        "predictor_conditioning": "concat",
        "predictor_residual_init": 0.1,
        "projector_hidden_dim": 16,
    }


def test_hub_loader_downloads_sidecar_and_strict_loads(tmp_path: Path, monkeypatch) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    architecture = _tiny_architecture()
    source = build_world_model(WorldModelConfig(**architecture))
    safetensors.save_file(source.state_dict(), str(tmp_path / "weights.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "artifact_id": "tiny",
                "architecture": architecture,
                "source": {"sha256": "source-hash"},
            }
        )
    )
    manifest = {
        "schema_version": 1,
        "organization": "example",
        "repo_type": "dataset",
        "repository_revisions": {"ORG/tiny": "0123456789abcdef"},
        "artifacts": {
            "tiny": {
                "kind": "world_model",
                "repo_id": "ORG/tiny",
                "filename": "weights.safetensors",
                "config_filename": "config.json",
                "source": {"sha256": "source-hash"},
            }
        },
    }
    monkeypatch.setattr(hub, "_manifest", lambda: manifest)

    def fake_download(*, filename, repo_type, **_kwargs):
        assert repo_type == "dataset"
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    loaded = hub.load_pretrained("tiny")
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert set(loaded.state_dict()) == set(source.state_dict())
