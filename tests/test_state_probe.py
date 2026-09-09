from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from sg_jepa.evaluation.state_probe import (
    ProbeTargetSpec,
    ProbeTrainConfig,
    _is_dino_config,
    _planar_metrics,
    fit_state_probe,
    load_state_probe,
    transform_state_targets,
)


def test_symmetry_aware_targets() -> None:
    state = torch.tensor([[1.0, 2.0, 3.0, 4.0, math.pi / 8.0, -2.0]])
    square = transform_state_targets(state, "square")
    triangle = transform_state_targets(state, "right_triangle")
    assert square[0, :4].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert square[0, 4].item() == pytest.approx(1.0)
    assert square[0, 5].item() == pytest.approx(0.0, abs=1.0e-6)
    assert square[0, 6].item() == -2.0
    assert triangle[0, 4].item() == pytest.approx(math.sin(math.pi / 8.0))
    assert triangle[0, 5].item() == pytest.approx(math.cos(math.pi / 8.0))


def test_square_angle_metric_respects_quarter_turn_symmetry() -> None:
    spec = ProbeTargetSpec.for_task("square")
    target = transform_state_targets(torch.zeros(1, 6), spec)[0]
    equivalent = transform_state_targets(
        torch.tensor([[0.0, 0.0, 0.0, 0.0, math.pi / 2.0, 0.0]]), spec
    )[0]
    metrics = _planar_metrics(equivalent, target, spec, torch.ones(7))
    assert metrics["angle_symmetry_mae_rad"] == pytest.approx(0.0, abs=1.0e-6)


def test_probe_training_writes_strictly_loadable_inference_checkpoint(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(7)
    train_features = torch.randn(24, 12, generator=generator)
    val_features = torch.randn(8, 12, generator=generator)
    projection = torch.randn(12, 7, generator=generator)
    train_targets = train_features @ projection
    val_targets = val_features @ projection
    report = fit_state_probe(
        train_features,
        train_targets,
        val_features,
        val_targets,
        tmp_path,
        task="square",
        temporal_window=4,
        config=ProbeTrainConfig(
            temporal_window=4,
            batch_size=8,
            max_epochs=1,
            patience=1,
        ),
        provenance={"device": "cpu", "fixture": True},
    )
    assert report["status"] == "pass"
    loaded = load_state_probe(tmp_path / "probe_weights.pt")
    assert loaded.feature_dim == 12
    assert loaded.temporal_window == 4
    assert loaded.target_spec == ProbeTargetSpec.for_task("square")
    assert loaded.predict_features(val_features[:2]).shape == (2, 7)


def test_release_sidecar_is_detected_as_dino() -> None:
    assert _is_dino_config({"kind": "dino_world_model", "architecture": {}})
    assert _is_dino_config({"schema_version": 1, "predictor": {}})
    assert not _is_dino_config({"kind": "world_model", "architecture": {}})
