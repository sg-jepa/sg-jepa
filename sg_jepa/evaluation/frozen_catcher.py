"""Bounded qualitative evaluation for arm-catcher diffusion policies."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from sg_jepa.control import load_policy_bundle
from sg_jepa.envs.catcher import ArmCatcherBallEnv, CatcherRuntime, _metadata_gravity


def _resolve_lance_dataset(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    for candidate in (root, root / "data.lance"):
        if candidate.is_dir() and candidate.name.endswith(".lance"):
            return candidate
    raise FileNotFoundError(f"could not find a Lance dataset under {root}")


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record["episode_metadata"]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("episode_metadata must decode to a mapping")
    return value


def load_catcher_test_tasks(
    dataset_path: str | Path,
    *,
    max_episodes: int,
) -> tuple[Path, list[dict[str, Any]]]:
    """Load frame zero from a bounded held-out Catcher cohort."""

    if max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    try:
        import lance
    except ImportError as exc:  # pragma: no cover - optional dependency guard.
        raise RuntimeError("Catcher evaluation requires sg-jepa[data]") from exc
    resolved = _resolve_lance_dataset(dataset_path)
    dataset = lance.dataset(str(resolved))
    required = {
        "episode_idx",
        "step_idx",
        "split_id",
        "gravity",
        "source_episode_index",
        "state",
        "catcher_state",
        "arm_state",
        "episode_metadata",
        "success",
    }
    if missing := required - set(dataset.schema.names):
        raise ValueError(f"Catcher dataset is missing columns: {sorted(missing)}")
    table = dataset.to_table(
        columns=sorted(required),
        filter="split_id = 1 AND step_idx = 0",
    )
    tasks = table.to_pylist()
    tasks.sort(key=lambda row: int(row["episode_idx"]))
    selected = tasks[:max_episodes]
    if not selected:
        raise ValueError("the Lance dataset contains no held-out frame-zero records")
    for task in selected:
        metadata = _metadata(task)
        if int(metadata["episode_index"]) != int(task["source_episode_index"]):
            raise ValueError("Catcher metadata/source episode index mismatch")
        if not math.isclose(
            _metadata_gravity(metadata),
            float(task["gravity"]),
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("Catcher metadata/scalar gravity mismatch")
        complete = ("ball_initial_state", "catcher", "arm", "camera_parameters")
        if any(name not in metadata for name in complete):
            raise ValueError(
                "Catcher metadata-only reconstruction requires complete episode metadata"
            )
    return resolved, selected


def action_smoothness(raw_actions: np.ndarray) -> dict[str, float]:
    """Summarize first and second differences of executed XYZ controls."""

    actions = np.asarray(raw_actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError(f"raw_actions must have shape [T,4], got {actions.shape}")
    controls = actions[:, 1:]

    def values(array: np.ndarray, prefix: str) -> dict[str, float]:
        if array.size == 0:
            return {f"{prefix}_rms": 0.0, f"{prefix}_max": 0.0}
        norms = np.linalg.norm(array, axis=-1)
        return {
            f"{prefix}_rms": float(np.sqrt(np.mean(np.square(norms)))),
            f"{prefix}_max": float(np.max(norms)),
        }

    return {
        **values(np.diff(controls, axis=0), "action_step"),
        **values(np.diff(controls, n=2, axis=0), "action_second_difference"),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty rollout set")
    captured = [record for record in records if bool(record["captured"])]
    capture_times = [
        float(record["time_to_capture_seconds"])
        for record in captured
        if record.get("time_to_capture_seconds") is not None
    ]
    result: dict[str, Any] = {
        "episodes": len(records),
        "captures": len(captured),
        "capture_rate": len(captured) / len(records),
        "misses": sum(bool(record["missed"]) for record in records),
        "terminal_reasons": dict(
            sorted(Counter(str(record["terminal_reason"]) for record in records).items())
        ),
        "mean_time_to_capture_seconds": (float(np.mean(capture_times)) if capture_times else None),
        "median_time_to_capture_seconds": (
            float(np.median(capture_times)) if capture_times else None
        ),
    }
    for key in (
        "action_step_rms",
        "action_step_max",
        "action_second_difference_rms",
        "action_second_difference_max",
    ):
        values = [float(record[key]) for record in records if math.isfinite(float(record[key]))]
        result[f"mean_{key}"] = float(np.mean(values)) if values else None
    return result


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run_frozen_catcher_evaluation(
    *,
    method: str,
    policy_checkpoint: str | Path,
    policy_config: str | Path,
    world_model_checkpoint: str | Path | None,
    world_model_config: str | Path | None,
    dataset_path: str | Path,
    menagerie_root: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    dinov2_root: str | Path | None = None,
    max_episodes: int = 1,
    seed: int = 42,
    render_size: int = 256,
) -> dict[str, Any]:
    """Run a few closed-loop episodes as an integration check, not a paper metric."""

    if render_size < 32:
        raise ValueError("render_size must be at least 32")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    records_dir = output / "records"
    records_dir.mkdir()
    dataset, tasks = load_catcher_test_tasks(dataset_path, max_episodes=max_episodes)
    bundle = load_policy_bundle(
        policy_checkpoint,
        policy_config,
        world_model_checkpoint=world_model_checkpoint,
        world_model_config=world_model_config,
        device=device,
        dinov2_root=dinov2_root,
    )
    if bundle.config.action_dim != 3:
        raise ValueError(
            f"Catcher evaluation requires action_dim=3, got {bundle.config.action_dim}"
        )

    records: list[dict[str, Any]] = []
    with CatcherRuntime(menagerie_root) as runtime:
        for number, task in enumerate(tasks, start=1):
            started = time.perf_counter()
            episode = int(task["episode_idx"])
            environment = ArmCatcherBallEnv.from_episode_record(
                task,
                runtime=runtime,
                width=render_size,
                height=render_size,
            )
            policy_state = bundle.start_episode(float(task["gravity"]))
            replans = 0
            try:
                policy_state.observe(environment.render_current())
                while not environment.terminal:
                    plan = policy_state.predict_action(seed=seed + 1_000_003 * episode + replans)
                    for action in plan[: bundle.config.execution_horizon]:
                        if environment.terminal:
                            break
                        environment.step(action)
                        policy_state.advance(environment.last_executed_action())
                        if not environment.terminal:
                            policy_state.observe(environment.render_current())
                    replans += 1
                raw_actions = np.asarray(environment.executed_actions, dtype=np.float32).reshape(
                    -1, 4
                )
                metadata = _metadata(task)
                record = {
                    "episode_idx": episode,
                    "source_episode_index": int(task["source_episode_index"]),
                    "gravity": float(task["gravity"]),
                    "source_success": bool(task["success"]),
                    "source_terminal_reason": metadata.get("terminal_reason"),
                    "method": method,
                    "seed": seed + 1_000_003 * episode,
                    "replan_count": replans,
                    "wall_seconds": float(time.perf_counter() - started),
                    **environment.result(),
                    **action_smoothness(raw_actions),
                }
            finally:
                environment.close()
            _write_json(records_dir / f"episode_{episode:05d}.json", record)
            records.append(record)
            print(
                json.dumps(
                    {
                        "completed": number,
                        "episodes": len(tasks),
                        "method": method,
                        "episode": episode,
                        "captured": record["captured"],
                        "terminal_reason": record["terminal_reason"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    _write_csv(output / "episodes.csv", records)
    report = {
        "schema_version": 1,
        "kind": "catcher_frozen_checkpoint_qualitative_evaluation",
        "claim": "qualitative_reproduction",
        "paper_metric": False,
        "exact_reproduction": False,
        "warning": (
            "This bounded cohort validates policy inference and environment execution; "
            "it is not a paper capture-rate estimate."
        ),
        "method": method,
        "dataset": str(dataset),
        "policy": bundle.provenance(),
        "evaluation": {
            "selected_episodes": len(tasks),
            "frames_per_episode": 64,
            "executed_transitions": 63,
            "planning_horizon": bundle.config.action_horizon,
            "execution_horizon": bundle.config.execution_horizon,
            "seed": seed,
            "render_size": render_size,
        },
        "summary": summarize_records(records),
        "records": records,
        "status": "pass",
    }
    _write_json(output / "evaluation.json", report)
    return report


__all__ = [
    "action_smoothness",
    "load_catcher_test_tasks",
    "run_frozen_catcher_evaluation",
    "summarize_records",
]
