from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_control_evaluations import aggregate_control_evaluations


@pytest.mark.parametrize(
    ("report_kind", "success_field"),
    [
        ("franka_frozen_checkpoint_qualitative_evaluation", "success"),
        (
            "paddle_frozen_checkpoint_qualitative_evaluation",
            "paper_success",
        ),
        ("catcher_frozen_checkpoint_qualitative_evaluation", "captured"),
    ],
)
def test_aggregate_uses_task_specific_success_metric(
    tmp_path: Path, report_kind: str, success_field: str
) -> None:
    for seed in range(5):
        output = tmp_path / f"seed_{seed}" / "evaluation.json"
        output.parent.mkdir()
        output.write_text(
            json.dumps(
                {
                    "kind": report_kind,
                    "status": "pass",
                    "evaluation": {"seed": seed},
                    "records": [
                        {"gravity": 1.0, success_field: True},
                        {"gravity": 2.0, success_field: False},
                    ],
                }
            )
        )

    result = aggregate_control_evaluations(tmp_path)

    assert result["status"] == "pass"
    assert result["success_field"] == success_field
    assert result["seed_count"] == 5
    assert result["episodes"] == 10
    assert result["success_rate"] == 0.5
    assert [row["success_rate"] for row in result["per_gravity"]] == [1.0, 0.0]
