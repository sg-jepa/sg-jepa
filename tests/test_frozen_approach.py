from pathlib import Path

import pytest
import torch

from sg_jepa.baselines import Native128Preprocessor
from sg_jepa.evaluation import strict_load_approach_probe
from sg_jepa.evaluation.frozen_approach import APPROACH_TARGETS, build_approach_probe


def test_native128_preprocessor_preserves_paper_contract() -> None:
    pixels = torch.zeros(2, 256, 192, 3, dtype=torch.uint8)
    output = Native128Preprocessor()(pixels)
    assert output.shape == (2, 3, 112, 112)
    assert output.dtype == torch.float32
    assert torch.allclose(output, torch.full_like(output, -1.0), atol=1.0e-6)
    assert Native128Preprocessor().to_dict()["encoder_resize"] == 112


def test_historical_probe_loads_strictly(tmp_path: Path) -> None:
    source = build_approach_probe(512)
    path = tmp_path / "probe.pt"
    torch.save(
        {
            "probe_kind": "mlp",
            "target_spec": "ball3d_posvel",
            "feature_dim": 512,
            "temporal_window": 2,
            "output_dim": 6,
            "target_stats": {
                "target_definition": list(APPROACH_TARGETS),
                "mean": [0.0] * 6,
                "std": [1.0] * 6,
            },
            "probe_state_dict": source.state_dict(),
        },
        path,
    )
    loaded = strict_load_approach_probe(path)
    assert loaded.feature_dim == 512
    assert loaded.temporal_window == 2
    assert torch.equal(loaded.model[1].weight, source[1].weight)


def test_probe_rejects_wrong_temporal_window(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "probe_kind": "mlp",
            "target_spec": "ball3d_posvel",
            "feature_dim": 512,
            "temporal_window": 1,
            "output_dim": 6,
            "target_stats": {
                "target_definition": list(APPROACH_TARGETS),
                "mean": [0.0] * 6,
                "std": [1.0] * 6,
            },
            "probe_state_dict": build_approach_probe(512).state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="two-frame"):
        strict_load_approach_probe(path)
