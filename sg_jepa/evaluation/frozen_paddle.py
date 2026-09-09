"""Bounded qualitative evaluation for arm-paddle diffusion policies."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from sg_jepa.control import load_policy_bundle
from sg_jepa.envs.paddle import ArmPaddleBallEnv, PaddleRuntime
from sg_jepa.envs.paddle_metrics import (
    PAPER_METRIC_NAME,
    PAPER_SUCCESS_CRITERION,
    summarize_records,
)

PAPER_TEST_EPISODES = 1_600
PAPER_EPISODE_SEED_MULTIPLIER = 1_009


def _resolve_lance_dataset(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    for candidate in (root, root / "data.lance"):
        if candidate.is_dir() and candidate.name.endswith(".lance"):
            return candidate
    raise FileNotFoundError(f"could not find a Lance dataset under {root}")


def load_paddle_test_tasks(
    dataset_path: str | Path,
    *,
    max_episodes: int,
    episode_start: int = 0,
) -> tuple[Path, list[dict[str, Any]]]:
    """Load frame zero from a bounded held-out Paddle cohort."""

    if max_episodes <= 0 or episode_start < 0:
        raise ValueError("max_episodes must be positive and episode_start non-negative")
    try:
        import lance
    except ImportError as exc:  # pragma: no cover - optional dependency guard.
        raise RuntimeError("Paddle evaluation requires sg-jepa[data]") from exc
    resolved = _resolve_lance_dataset(dataset_path)
    dataset = lance.dataset(str(resolved))
    required = {
        "episode_idx",
        "step_idx",
        "split_id",
        "gravity",
        "source_episode_index",
        "state",
        "paddle_state",
        "arm_state",
        "episode_metadata",
        "success",
    }
    if missing := required - set(dataset.schema.names):
        raise ValueError(f"Paddle dataset is missing columns: {sorted(missing)}")
    table = dataset.to_table(
        columns=sorted(required),
        filter="split_id = 1 AND step_idx = 0",
    )
    tasks = table.to_pylist()
    tasks.sort(key=lambda row: int(row["episode_idx"]))
    selected = tasks[episode_start : episode_start + max_episodes]
    if not selected:
        raise ValueError("the Lance dataset contains no held-out frame-zero records")
    return resolved, selected


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


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record["episode_metadata"]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("episode_metadata must decode to a mapping")
    return value


def run_frozen_paddle_evaluation(
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
    episode_start: int = 0,
    seed: int = 42,
    render_size: int = 256,
    paper_cohort: bool = False,
) -> dict[str, Any]:
    """Run closed-loop Paddle evaluation with the final paper success metric."""

    if render_size < 32:
        raise ValueError("render_size must be at least 32")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    records_dir = output / "records"
    records_dir.mkdir()
    dataset, tasks = load_paddle_test_tasks(
        dataset_path,
        max_episodes=max_episodes,
        episode_start=episode_start,
    )
    if paper_cohort and (episode_start != 0 or len(tasks) != PAPER_TEST_EPISODES):
        raise ValueError(
            "Paddle --paper-cohort requires all 1,600 held-out episodes from offset zero"
        )
    bundle = load_policy_bundle(
        policy_checkpoint,
        policy_config,
        world_model_checkpoint=world_model_checkpoint,
        world_model_config=world_model_config,
        device=device,
        dinov2_root=dinov2_root,
    )

    records: list[dict[str, Any]] = []
    with PaddleRuntime(menagerie_root, frames_per_episode=64) as runtime:
        for number, task in enumerate(tasks, start=1):
            started = time.perf_counter()
            episode = int(task["episode_idx"])
            source_episode = int(task["source_episode_index"])
            policy_seed = seed + PAPER_EPISODE_SEED_MULTIPLIER * source_episode
            gravity = float(task["gravity"])
            expected_initial = {
                "state": task["state"],
                "paddle_state": task["paddle_state"],
                "arm_state": task["arm_state"],
            }
            environment = ArmPaddleBallEnv.from_episode_metadata(
                _metadata(task),
                runtime=runtime,
                expected_initial=expected_initial,
                width=render_size,
                height=render_size,
                expected_frames_per_episode=64,
            )
            policy_state = bundle.start_episode(gravity)
            replans = 0
            try:
                policy_state.observe(environment.render_current())
                while not environment.terminal:
                    plan = policy_state.predict_action(seed=policy_seed + replans)
                    for action in plan[: bundle.config.execution_horizon]:
                        if environment.terminal:
                            break
                        environment.step(action)
                        policy_state.advance(environment.last_executed_action())
                        if not environment.terminal:
                            policy_state.observe(environment.render_current())
                    replans += 1
                record = {
                    "episode_idx": episode,
                    "source_episode_index": source_episode,
                    "gravity": gravity,
                    "source_success": bool(task["success"]),
                    "method": method,
                    "seed": policy_seed,
                    "policy_seed": policy_seed,
                    "replan_count": replans,
                    "wall_seconds": float(time.perf_counter() - started),
                    **environment.metrics(max_frames=63),
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
                        "contacts": record["distinct_paddle_contacts"],
                        "drop": record["drop"],
                        "paper_success": record["paper_success"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    _write_csv(output / "episodes.csv", records)
    report = {
        "schema_version": 2,
        "kind": "paddle_frozen_checkpoint_qualitative_evaluation",
        "claim": (
            "paper_subset_reproduction_attempt" if paper_cohort else "qualitative_reproduction"
        ),
        "paper_metric": bool(paper_cohort),
        "paper_metric_definition": True,
        "paper_metric_name": PAPER_METRIC_NAME,
        "success_field": "paper_success",
        "success_criterion": dict(PAPER_SUCCESS_CRITERION),
        "exact_reproduction": False,
        "warning": None
        if paper_cohort
        else (
            "This bounded cohort validates policy inference and environment execution; "
            "it is not a paper success-rate estimate."
        ),
        "method": method,
        "dataset": str(dataset),
        "policy": bundle.provenance(),
        "evaluation": {
            "selected_episodes": len(tasks),
            "selection": "contiguous held-out records in dataset order",
            "episode_start": episode_start,
            "frames_per_episode": 64,
            "executed_transitions": 63,
            "planning_horizon": bundle.config.action_horizon,
            "execution_horizon": bundle.config.execution_horizon,
            "seed": seed,
            "policy_seed_rule": (f"seed + {PAPER_EPISODE_SEED_MULTIPLIER} * source_episode_index"),
            "render_size": render_size,
        },
        "summary": summarize_records(records),
        "records": records,
        "status": "pass",
    }
    _write_json(output / "evaluation.json", report)
    return report


__all__ = ["load_paddle_test_tasks", "run_frozen_paddle_evaluation"]
