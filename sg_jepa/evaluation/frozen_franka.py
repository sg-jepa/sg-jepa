"""Bounded qualitative Franka evaluation with frozen historical artifacts."""

from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from sg_jepa.control import load_policy_bundle
from sg_jepa.envs.franka import CONTROL_CONTRACT, BasketRuntime, BasketSimulator


def _resolve_lance_dataset(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    candidates = (root, root / "data.lance", root / "dataset" / "data.lance")
    for candidate in candidates:
        if candidate.is_dir() and candidate.name.endswith(".lance"):
            return candidate
    raise FileNotFoundError(f"could not find a Lance dataset under {root}")


def load_franka_test_tasks(
    dataset_path: str | Path,
    *,
    max_episodes: int,
    episode_start: int = 0,
) -> tuple[Path, list[dict[str, Any]]]:
    """Load only step zero of a bounded held-out cohort."""

    if max_episodes <= 0 or episode_start < 0:
        raise ValueError("max_episodes must be positive and episode_start non-negative")
    try:
        import lance
    except ImportError as exc:  # pragma: no cover - optional dependency guard.
        raise RuntimeError("Franka evaluation requires sg-jepa[data]") from exc
    resolved = _resolve_lance_dataset(dataset_path)
    dataset = lance.dataset(str(resolved))
    columns = [
        "episode_idx",
        "step_idx",
        "split_id",
        "gravity",
        "source_episode_index",
        "state",
        "action",
        "paddle_state",
        "arm_state",
        "basket_state",
        "episode_metadata",
        "success",
        "failure_reason",
    ]
    table = dataset.to_table(columns=columns, filter="split_id = 1 AND step_idx = 0")
    tasks = table.to_pylist()
    tasks.sort(key=lambda row: int(row["episode_idx"]))
    selected = tasks[episode_start : episode_start + max_episodes]
    if not selected:
        raise ValueError("the Lance dataset contains no held-out step-zero records")
    return resolved, selected


def _smoothness(actions: np.ndarray) -> dict[str, float]:
    controls = np.asarray(actions, dtype=np.float64)[:, 1:]
    first = np.diff(controls, axis=0)
    second = np.diff(controls, n=2, axis=0)
    return {
        "action_step_rms": float(np.sqrt(np.mean(np.square(first)))) if first.size else 0.0,
        "action_step_max": float(np.max(np.abs(first))) if first.size else 0.0,
        "action_second_difference_rms": (
            float(np.sqrt(np.mean(np.square(second)))) if second.size else 0.0
        ),
        "action_second_difference_max": (float(np.max(np.abs(second))) if second.size else 0.0),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = len(records)
    blade = sum(bool(row["valid_paddle_blade_contact"]) for row in records)
    entries = sum(bool(row["basket_entry"]) for row in records)
    successes = sum(bool(row["success"]) for row in records)
    relaxed = sum(bool(row["relaxed_success"]) for row in records)
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "valid_blade_contacts": blade,
        "valid_blade_contact_rate": blade / episodes,
        "basket_entries": entries,
        "basket_entry_rate": entries / episodes,
        "relaxed_successes": relaxed,
        "relaxed_success_rate": relaxed / episodes,
        "failure_reasons": dict(
            sorted(Counter(str(row["failure_reason"]) for row in records).items())
        ),
        "mean_episode_seconds": float(np.mean([float(row["wall_seconds"]) for row in records])),
    }


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in records for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run_frozen_franka_evaluation(
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
    max_episodes: int = 2,
    episode_start: int = 0,
    seed: int = 42,
    render_size: int = 256,
    paper_cohort: bool = False,
) -> dict[str, Any]:
    """Run a few closed-loop episodes without estimating a paper success rate."""

    if render_size < 32:
        raise ValueError("render_size must be at least 32")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    records_dir = output / "records"
    records_dir.mkdir()
    dataset, tasks = load_franka_test_tasks(
        dataset_path,
        max_episodes=max_episodes,
        episode_start=episode_start,
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
    with BasketRuntime(menagerie_root, width=render_size, height=render_size) as runtime:
        for number, task_record in enumerate(tasks):
            started = time.perf_counter()
            episode = int(task_record["episode_idx"])
            gravity = float(task_record["gravity"])
            simulator = BasketSimulator(task_record, runtime=runtime)
            state = bundle.start_episode(gravity)
            current = simulator.render_current()
            state.observe(current)
            replans = 0
            while simulator.frame_idx < 63:
                plan = state.predict_action(seed=seed + 1_000_003 * episode + replans)
                for action in plan[: bundle.config.execution_horizon]:
                    if simulator.frame_idx >= 63:
                        break
                    simulator.step(action)
                    state.advance(action)
                    current = simulator.render_current()
                    state.observe(current)
                replans += 1
            actions = np.asarray(simulator.executed_actions, dtype=np.float32)
            result = simulator.result()
            record: dict[str, Any] = {
                "episode_idx": episode,
                "source_episode_index": int(task_record["source_episode_index"]),
                "gravity": gravity,
                "source_success": bool(task_record["success"]),
                "source_failure_reason": str(task_record["failure_reason"]),
                "method": method,
                "replan_count": replans,
                "seed": seed + 1_000_003 * episode,
                "wall_seconds": float(time.perf_counter() - started),
                **result,
                **_smoothness(actions),
            }
            _write_json(records_dir / f"episode_{episode:05d}.json", record)
            records.append(record)
            print(
                json.dumps(
                    {
                        "completed": number + 1,
                        "episodes": len(tasks),
                        "method": method,
                        "episode": episode,
                        "blade": result["valid_paddle_blade_contact"],
                        "entry": result["basket_entry"],
                        "success": result["success"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    _write_csv(output / "episodes.csv", records)
    provenance = bundle.provenance()
    report = {
        "schema_version": 1,
        "kind": "franka_frozen_checkpoint_qualitative_evaluation",
        "claim": (
            "paper_subset_reproduction_attempt" if paper_cohort else "qualitative_reproduction"
        ),
        "paper_metric": bool(paper_cohort),
        "exact_reproduction": False,
        "warning": None
        if paper_cohort
        else (
            "This bounded cohort checks closed-loop behavior and integration only; "
            "it is not a success-rate estimate for the paper test set."
        ),
        "method": method,
        "dataset": str(dataset),
        "policy": provenance,
        "world_model_config": (
            None
            if world_model_config is None
            else str(Path(world_model_config).expanduser().resolve())
        ),
        "policy_config": str(Path(policy_config).expanduser().resolve()),
        "evaluation": {
            "selected_episodes": len(tasks),
            "selection": "contiguous held-out records in dataset order",
            "episode_start": episode_start,
            "frames_per_episode": 64,
            "executed_transitions": 63,
            "seed": seed,
            "render_size": render_size,
            "control_contract": CONTROL_CONTRACT,
            "legacy_translation_clip_removed": True,
            "reachability_projection": "5 mm / 5 degrees",
        },
        "summary": _summarize(records),
        "records": records,
        "status": "pass",
    }
    _write_json(output / "evaluation.json", report)
    return report


__all__ = ["load_franka_test_tasks", "run_frozen_franka_evaluation"]
