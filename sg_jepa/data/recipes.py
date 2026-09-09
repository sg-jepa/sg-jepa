"""Validation for immutable paper dataset recipe metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

TASK_ACTION_SCHEMAS = {
    "right_triangle": ("impulse_x", "impulse_z", "g"),
    "square": ("impulse_x", "impulse_z", "g"),
    "approach_ball": ("g",),
    "arm_catcher_ball": ("g", "dx", "dy", "dz"),
    "arm_paddle_ball": ("g", "dx", "dy", "dz", "phi", "theta"),
    "franka_basket": ("g", "dx", "dy", "dz", "phi", "theta"),
}


def load_recipes(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("dataset recipes require schema_version=1")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(TASK_ACTION_SCHEMAS):
        raise ValueError(
            f"recipes must cover exactly {sorted(TASK_ACTION_SCHEMAS)}, got {sorted(tasks or {})}"
        )
    for task, recipe in tasks.items():
        if not isinstance(recipe, dict):
            raise ValueError(f"recipe {task} must be a mapping")
        if recipe.get("action_schema") != list(TASK_ACTION_SCHEMAS[task]):
            raise ValueError(f"recipe {task} action schema differs from the public contract")
        for key in ("image_size", "frames_per_episode", "fps", "source_lock_group"):
            if key not in recipe:
                raise ValueError(f"recipe {task} is missing {key}")
    return tasks


def validate_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    """Validate the fixed 25-gravity, 200-per-gravity planar cohort."""

    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text())
    gravities = [float(value) for value in payload.get("gravity_values", [])]
    expected_gravities = [-2.0 + 0.5 * index for index in range(25)]
    if gravities != expected_gravities:
        raise ValueError("evaluation manifest must use the locked -2:0.5:10 gravity grid")
    if payload.get("samples_per_gravity") != 200 or payload.get("total_source_ids") != 5000:
        raise ValueError("evaluation manifest must contain 200 x 25 = 5,000 source IDs")
    if payload.get("split") != "test" or payload.get("split_id") != 1:
        raise ValueError("evaluation manifest must select split=test/split_id=1")

    per_gravity = payload.get("per_gravity")
    if not isinstance(per_gravity, dict):
        raise ValueError("evaluation manifest is missing per_gravity records")
    grouped_ids: list[int] = []
    for gravity in expected_gravities:
        key = f"{gravity:g}"
        record = per_gravity.get(key)
        if not isinstance(record, dict) or float(record.get("gravity", float("nan"))) != gravity:
            raise ValueError(f"evaluation manifest is missing gravity {key}")
        source_ids = [int(value) for value in record.get("source_ids", [])]
        if record.get("count") != 200 or len(source_ids) != 200 or len(set(source_ids)) != 200:
            raise ValueError(f"gravity {key} must contain 200 unique source IDs")
        grouped_ids.extend(source_ids)

    primary = payload.get("primary_evaluation")
    primary_ids = (
        [int(value) for value in primary.get("source_ids", [])] if isinstance(primary, dict) else []
    )
    if len(grouped_ids) != len(set(grouped_ids)) or len(primary_ids) != 5000:
        raise ValueError("evaluation source IDs must be unique across gravity groups")
    if set(primary_ids) != set(grouped_ids):
        raise ValueError("primary_evaluation differs from the per-gravity source-ID union")
    return {
        "schema_version": 1,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "gravity_count": len(gravities),
        "source_ids": len(primary_ids),
        "samples_per_gravity": 200,
        "seed": int(payload["seed"]),
        "status": "pass",
    }


__all__ = ["load_recipes", "validate_evaluation_manifest"]
