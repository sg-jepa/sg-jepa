"""Public entry point for all six Semigroup-JEPA datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from data_generation.franka import generate_franka_basket_dataset
from data_generation.mujoco import TASKS as MUJOCO_TASKS
from data_generation.mujoco import generate_mujoco_dataset
from data_generation.planar import generate_planar_dataset

TASKS = (
    "right_triangle",
    "square",
    "approach_ball",
    "arm_catcher_ball",
    "arm_paddle_ball",
    "franka_basket",
)


def _recipe(path: str | Path, task: str) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("recipe must be a schema_version=1 mapping")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(TASKS):
        raise ValueError(f"recipe must define exactly: {', '.join(TASKS)}")
    value = tasks[task]
    if not isinstance(value, dict):
        raise ValueError(f"task recipe {task!r} must be a mapping")
    return value


def _episode_default(recipe: dict[str, Any], split: str) -> int:
    generation = recipe.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("task recipe has no generation mapping")
    train = int(generation["train_episodes"])
    test = int(generation["test_episodes"])
    return {"train": train, "test": test, "all": train + test}[split]


def generate_dataset(
    *,
    recipe_path: str | Path,
    task: str,
    split: str,
    output: str | Path,
    episodes: int | None = None,
    seed: int | None = None,
    menagerie_root: str | Path | None = None,
    source_start: int | None = None,
) -> dict[str, Any]:
    """Generate one task/split using paper defaults or a bounded episode override."""

    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {', '.join(TASKS)}")
    if split not in {"train", "test", "all"}:
        raise ValueError("split must be train, test, or all")
    if source_start is not None and task != "franka_basket":
        raise ValueError("source_start is only supported for franka_basket")
    if source_start is not None and episodes is None:
        raise ValueError("source_start requires an explicit episode count")
    recipe = _recipe(recipe_path, task)
    requested_episodes = _episode_default(recipe, split) if episodes is None else int(episodes)
    if requested_episodes <= 0:
        raise ValueError("episodes must be positive")
    selected_seed = int(recipe["base_seed"] if seed is None else seed)
    paper_scale = episodes is None
    common = {
        "episodes": requested_episodes,
        "image_size": int(recipe["image_size"]),
        "seed": selected_seed,
        "split": split,
        "paper_scale": paper_scale,
    }
    if task in {"right_triangle", "square"}:
        return generate_planar_dataset(
            task,
            output,
            frames=int(recipe["frames_per_episode"]),
            **common,
        )
    if task in MUJOCO_TASKS:
        return generate_mujoco_dataset(
            task,
            output,
            frames=int(recipe["frames_per_episode"]),
            fps=int(recipe["fps"]),
            menagerie_root=menagerie_root,
            **common,
        )
    return generate_franka_basket_dataset(
        output,
        episodes=requested_episodes,
        image_size=int(recipe["image_size"]),
        seed=selected_seed,
        split=split,
        paper_scale=paper_scale,
        menagerie_root=menagerie_root,
        source_start=source_start,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        help="override the recipe episode count for a bounded validation run",
    )
    parser.add_argument("--seed", type=int, help="override the task's deterministic base seed")
    parser.add_argument(
        "--source-start",
        type=int,
        help=(
            "first canonical source episode for a bounded Franka shard; "
            "requires --task franka_basket --split all --episodes"
        ),
    )
    parser.add_argument(
        "--menagerie-root",
        type=Path,
        help="reuse a checkout of the pinned MuJoCo Menagerie revision",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = generate_dataset(
        recipe_path=args.recipe,
        task=args.task,
        split=args.split,
        output=args.output,
        episodes=args.episodes,
        seed=args.seed,
        menagerie_root=args.menagerie_root,
        source_start=args.source_start,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
