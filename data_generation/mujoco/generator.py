"""Shared MuJoCo generator for Approach, Catcher, and Paddle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_generation.common.assets import (
    provision_menagerie_assets,
    verify_menagerie_component,
)

SOURCE_FILES = (
    "common.py",
    "generate.py",
    "sim_support.py",
    "render.py",
    "render_support.py",
    "robot_assets.py",
    "pack_lance.py",
)


@dataclass(frozen=True)
class TaskSpec:
    source_revision: str
    train_gravity_mean: float
    train_gravity_std: float
    train_gravity_min: float
    test_gravities: tuple[float, ...]


_APPROACH_GRAVITIES = tuple(float(value) for value in range(21)) + (8.87, 3.721, 1.625, 0.62)

TASKS = {
    "approach_ball": TaskSpec(
        "ecd0759ac2d5fe65b60be99048c761aaca4bd92f",
        9.8,
        2.0,
        0.0,
        _APPROACH_GRAVITIES,
    ),
    "arm_catcher_ball": TaskSpec(
        "2f7db523e6aadbbcb5425d9793862e52a0032ef3",
        4.0,
        0.5,
        0.1,
        tuple(-1.0 + 0.5 * index for index in range(23)),
    ),
    "arm_paddle_ball": TaskSpec(
        "a87e0e19e19d7f28657ea48941168eb9ab22b99a",
        9.8,
        2.0,
        0.0,
        _APPROACH_GRAVITIES,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _engine_root() -> Path:
    root = Path(__file__).resolve().parent / "engine"
    missing = [name for name in SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"MuJoCo generator is missing source files: {missing}")
    return root


def source_manifest(task: str) -> dict[str, Any]:
    """Return the content identity of the shared generator implementation."""

    if task not in TASKS:
        raise ValueError(f"unknown MuJoCo task {task!r}")
    engine = _engine_root()
    return {
        "source_repository": "semigroup-jepa-data-generator",
        "task_source_revision": TASKS[task].source_revision,
        "shared_engine": "approach-catcher-paddle",
        "source_sha256": {name: _sha256(engine / name) for name in SOURCE_FILES},
    }


def _generation_environment(
    task: str,
    work_dir: Path,
    *,
    episodes: int,
    frames: int,
    image_size: int,
    fps: int,
    seed: int,
    split: str,
    paper_scale: bool,
) -> dict[str, str]:
    if split not in {"train", "test", "all"}:
        raise ValueError("split must be train, test, or all")
    spec = TASKS[task]
    if paper_scale:
        expected_episodes = {"all": 9_600, "train": 8_000, "test": 1_600}[split]
        if (episodes, frames, image_size, fps) != (expected_episodes, 64, 256, 16):
            raise ValueError(
                f"paper-scale {split} generation requires "
                f"episodes={expected_episodes}, frames=64, image_size=256, fps=16"
            )
        train_per_batch, test_per_batch, batch_size = 25, 5, 30
        canonical_episodes = 9_600
        supersample = 2
    else:
        if episodes < 1 or frames < 2 or image_size < 32 or fps <= 0:
            raise ValueError(
                "generation requires episodes>=1, frames>=2, image_size>=32, and fps>0"
            )
        if split == "train":
            train_per_batch, test_per_batch = episodes, 0
        elif split == "test":
            train_per_batch, test_per_batch = 0, episodes
        else:
            if episodes < 2:
                raise ValueError("split=all requires at least two episodes")
            train_per_batch = max(1, episodes // 2)
            test_per_batch = episodes - train_per_batch
        batch_size = episodes
        canonical_episodes = episodes
        supersample = 1

    env = os.environ.copy()
    env.update(
        {
            "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
            "PYTHONUNBUFFERED": "1",
            "DEMO_MUJOCO_WORK_DIR": str(work_dir),
            "DEMO_MUJOCO_TASKS": task,
            "DEMO_MUJOCO_OUTPUT_VERSIONS": "v0",
            "DEMO_MUJOCO_EPISODES_PER_TASK": str(episodes),
            "DEMO_MUJOCO_CANONICAL_EPISODES_PER_TASK": str(canonical_episodes),
            "DEMO_MUJOCO_SELECTED_SPLIT": split,
            "DEMO_MUJOCO_BASE_SEED": str(seed),
            "DEMO_MUJOCO_DYNAMIC_SHAPE_MODE": "ball_only",
            "DEMO_MUJOCO_RENDER_ENV": "natural",
            "DEMO_MUJOCO_IMAGE_WIDTH": str(image_size),
            "DEMO_MUJOCO_IMAGE_HEIGHT": str(image_size),
            "DEMO_MUJOCO_FPS": str(fps),
            "DEMO_MUJOCO_FRAMES_PER_EPISODE": str(frames),
            "DEMO_MUJOCO_RENDER_SUPERSAMPLE": str(supersample),
            "DEMO_MUJOCO_SHARD_EPISODES": str(min(8, episodes)),
            "DEMO_MUJOCO_RENDER_WORKERS": "1",
            "DEMO_MUJOCO_TRAIN_GAUSSIAN_MEAN": str(spec.train_gravity_mean),
            "DEMO_MUJOCO_TRAIN_GAUSSIAN_STD": str(spec.train_gravity_std),
            "DEMO_MUJOCO_TRAIN_G_MIN": str(spec.train_gravity_min),
            "DEMO_MUJOCO_TEST_GRAVITIES": ",".join(f"{value:g}" for value in spec.test_gravities),
            "DEMO_MUJOCO_GRAVITY_BATCH_EPISODES": str(batch_size),
            "DEMO_MUJOCO_TRAIN_EPISODES_PER_BATCH": str(train_per_batch),
            "DEMO_MUJOCO_TEST_EPISODES_PER_BATCH": str(test_per_batch),
            "DEMO_MUJOCO_ARM_CATCHER_CAPTURE_RETRY_LIMIT": "96",
        }
    )
    return env


def generate_mujoco_dataset(
    task: str,
    output_dir: str | Path,
    *,
    episodes: int = 4,
    frames: int = 64,
    image_size: int = 64,
    fps: int = 16,
    seed: int = 20260525,
    split: str = "all",
    paper_scale: bool = False,
    menagerie_root: str | Path | None = None,
) -> dict[str, Any]:
    """Simulate, render, and pack one of the three shared-engine tasks."""

    if task not in TASKS:
        raise ValueError(f"unknown MuJoCo task {task!r}")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    source = source_manifest(task)

    asset_manifest = None
    if task in {"arm_catcher_ball", "arm_paddle_ball"}:
        if menagerie_root is None:
            menagerie_root = output_dir / "mujoco_menagerie"
            asset_manifest = provision_menagerie_assets(
                menagerie_root,
                components=("unitree_z1",),
            )
        else:
            menagerie_root = Path(menagerie_root).expanduser().resolve()
            asset_manifest = verify_menagerie_component(menagerie_root, "unitree_z1")

    work_dir = output_dir / "_work"
    env = _generation_environment(
        task,
        work_dir,
        episodes=episodes,
        frames=frames,
        image_size=image_size,
        fps=fps,
        seed=seed,
        split=split,
        paper_scale=paper_scale,
    )
    with tempfile.TemporaryDirectory(prefix="sgjepa-mujoco-") as runtime_text:
        runtime = Path(runtime_text)
        shutil.copytree(_engine_root(), runtime / "engine")
        workflow = runtime / "engine"
        if menagerie_root is not None:
            target = workflow / "third_party" / "mujoco_menagerie" / "unitree_z1"
            target.parent.mkdir(parents=True)
            shutil.copytree(Path(menagerie_root) / "unitree_z1", target)
        for command in ("generate.py", "render.py", "pack_lance.py"):
            subprocess.run([sys.executable, command], cwd=workflow, env=env, check=True)

    generated = work_dir / "data" / task / "v0" / "data.lance"
    if not generated.is_dir():
        raise FileNotFoundError(f"generator did not create {generated}")
    dataset = output_dir / "data.lance"
    shutil.move(str(generated), dataset)
    shutil.rmtree(work_dir)
    train_episodes = episodes if split == "train" else 0
    test_episodes = episodes if split == "test" else 0
    if split == "all":
        if paper_scale:
            train_episodes, test_episodes = 8_000, 1_600
        else:
            train_episodes = max(1, episodes // 2)
            test_episodes = episodes - train_episodes
    manifest = {
        "schema_version": 1,
        "task": task,
        "split": split,
        "seed": seed,
        "paper_scale": paper_scale,
        "episodes": episodes,
        "frames_per_episode": frames,
        "image_size": image_size,
        "fps": fps,
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
        "dataset": dataset.name,
        "source": source,
        "asset_manifest": asset_manifest,
        "qualitative_reproduction": True,
        "status": "pass",
    }
    (output_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


__all__ = ["TASKS", "generate_mujoco_dataset", "source_manifest"]
