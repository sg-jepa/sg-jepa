import json
from pathlib import Path

import torch
import yaml

from sg_jepa.baselines import PredictorConfig, build_dino_world_model, load_dino_config


def _tiny_model():
    config = PredictorConfig(
        history_size=3,
        num_patches=4,
        visual_dim=8,
        action_dim=2,
        action_emb_dim=3,
        proprio_emb_dim=3,
        depth=2,
        heads=2,
        mlp_dim=24,
        dim_head=4,
        dropout=0.0,
    )
    return build_dino_world_model(config).eval()


def test_dino_wm_shapes_and_loss() -> None:
    torch.manual_seed(3)
    model = _tiny_model()
    tokens = torch.randn(2, 4, 4, 8)
    actions = torch.randn(2, 4, 2)
    output = model.teacher_forced_loss(tokens, actions)
    assert output.loss.ndim == 0
    assert output.predicted_visual_tokens.shape == (2, 3, 4, 8)
    assert model.rollout_visual_tokens(tokens, actions, horizon=1).shape == (2, 4, 4, 8)


def test_dino_wm_is_frame_causal() -> None:
    torch.manual_seed(4)
    model = _tiny_model()
    tokens = torch.randn(1, 3, 4, 8)
    actions = torch.randn(1, 3, 2)
    combined = model.encode_inputs(tokens, actions)
    altered = combined.clone()
    altered[:, 2] += 100.0
    with torch.no_grad():
        before = model.predictor(combined)
        after = model.predictor(altered)
    torch.testing.assert_close(before[:, :2], after[:, :2], rtol=0.0, atol=0.0)


def test_dino_config_accepts_training_yaml_and_release_json(tmp_path: Path) -> None:
    predictor = {
        "history_size": 3,
        "num_patches": 4,
        "visual_dim": 8,
        "action_dim": 2,
        "proprio_dim": 1,
        "action_emb_dim": 3,
        "proprio_emb_dim": 3,
        "depth": 2,
        "heads": 2,
        "mlp_dim": 24,
        "dim_head": 4,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        "attention_backend": "sdpa",
    }
    training_config = tmp_path / "training.yaml"
    release_sidecar = tmp_path / "release.json"
    training_config.write_text(yaml.safe_dump({"schema_version": 1, "predictor": predictor}))
    release_sidecar.write_text(
        json.dumps({"format_version": 1, "kind": "dino_world_model", "architecture": predictor})
    )

    from_training, training_payload = load_dino_config(training_config)
    from_release, release_payload = load_dino_config(release_sidecar)

    assert from_training == from_release == PredictorConfig.from_dict(predictor)
    assert training_payload["predictor"] == release_payload["architecture"]
