"""Franka paddle-to-basket dataset generation."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from data_generation.common.assets import (
    MENAGERIE_REPOSITORY,
    MENAGERIE_REVISION,
    provision_menagerie_assets,
    verify_menagerie_component,
)

SOURCE_REVISION = "franka_paddle_hit_ball_to_basket_v1"
PANDA_UPSTREAM_TEXT = """{
  "commit": "c1a4eeb85694ae1dffe33ff1797d4e528928a133",
  "license": "Apache-2.0",
  "repository": "https://github.com/google-deepmind/mujoco_menagerie.git",
  "subdirectory": "franka_emika_panda"
}
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _engine_root() -> Path:
    root = Path(__file__).resolve().parent / "engine"
    required = ("base_scene.py", "basket_task.py", "generation.py", "dataset.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Franka generator is missing source files: {missing}")
    return root


def source_manifest() -> dict[str, Any]:
    root = _engine_root()
    return {
        "source_identity": SOURCE_REVISION,
        "source_sha256": {
            path.name: _sha256(path) for path in sorted(root.iterdir()) if path.is_file()
        },
        "menagerie_repository": MENAGERIE_REPOSITORY,
        "menagerie_revision": MENAGERIE_REVISION,
    }


def _install_panda(runtime: Path, menagerie_root: Path) -> dict[str, Any]:
    verified = verify_menagerie_component(menagerie_root, "franka_emika_panda")
    target = runtime / "third_party" / "mujoco_menagerie" / "franka_emika_panda"
    target.parent.mkdir(parents=True)
    shutil.copytree(menagerie_root / "franka_emika_panda", target)
    (target / "UPSTREAM.json").write_text(PANDA_UPSTREAM_TEXT)
    return verified


def _source_indices(
    split: str,
    episodes: int,
    paper_scale: bool,
    source_start: int | None = None,
) -> tuple[list[int], int, int]:
    if source_start is not None:
        if split != "all":
            raise ValueError("a Franka source range requires split=all")
        if paper_scale:
            raise ValueError("a Franka source range is a bounded shard, not paper_scale")
        stop = source_start + episodes
        if source_start < 0 or stop > 13_000:
            raise ValueError("Franka source range must lie within [0, 13000)")
        indices = list(range(source_start, stop))
        train_count = sum(index < 8_000 for index in indices)
        return indices, train_count, episodes - train_count
    if split == "train":
        if episodes > 8_000:
            raise ValueError("Franka train split contains 8,000 episodes")
        return list(range(episodes)), episodes, 0
    if split == "test":
        if episodes > 5_000:
            raise ValueError("Franka test split contains 5,000 episodes")
        return list(range(8_000, 8_000 + episodes)), 0, episodes
    if paper_scale and episodes != 13_000:
        raise ValueError("paper-scale split=all requires 13,000 episodes")
    if episodes < 2:
        raise ValueError("split=all requires at least two episodes")
    train_count = 8_000 if paper_scale else max(1, episodes // 2)
    test_count = episodes - train_count
    return [*range(train_count), *range(8_000, 8_000 + test_count)], train_count, test_count


def _release_schema(schema: Any, train_count: int, test_count: int, seed: int) -> Any:
    import pyarrow as pa

    fields = []
    names: set[str] = set()
    for field in schema:
        if field.name not in names:
            fields.append(field)
            names.add(field.name)
    fields.append(pa.field("task", pa.string(), nullable=False))
    metadata = dict(schema.metadata or {})
    metadata.update(
        {
            b"schema_version": b"1",
            b"public_task_alias": b"franka_basket",
            b"train_episodes": str(train_count).encode(),
            b"test_episodes": str(test_count).encode(),
            b"base_seed": str(seed).encode(),
        }
    )
    return pa.schema(fields, metadata=metadata)


def generate_franka_basket_dataset(
    output_dir: str | Path,
    *,
    episodes: int = 4,
    image_size: int = 256,
    seed: int = 20260807,
    split: str = "all",
    paper_scale: bool = False,
    menagerie_root: str | Path | None = None,
    source_start: int | None = None,
) -> dict[str, Any]:
    """Stream selected deterministic Franka episodes into Lance."""

    if split not in {"train", "test", "all"}:
        raise ValueError("split must be train, test, or all")
    if episodes <= 0 or image_size < 32:
        raise ValueError("episodes must be positive and image_size must be at least 32")
    if paper_scale and image_size != 256:
        raise ValueError("paper-scale Franka generation requires image_size=256")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    indices, train_count, test_count = _source_indices(
        split,
        episodes,
        paper_scale,
        source_start,
    )

    try:
        import lance
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise ImportError("install sg-jepa[data] to generate Franka datasets") from exc

    source = source_manifest()
    with tempfile.TemporaryDirectory(prefix="sgjepa-franka-") as temporary_text:
        temporary = Path(temporary_text)
        runtime = temporary / "engine"
        shutil.copytree(_engine_root(), runtime)
        if menagerie_root is None:
            menagerie = temporary / "mujoco_menagerie"
            asset_provision = provision_menagerie_assets(
                menagerie,
                components=("franka_emika_panda",),
            )
        else:
            menagerie = Path(menagerie_root).expanduser().resolve()
            asset_provision = verify_menagerie_component(menagerie, "franka_emika_panda")
        asset_manifest = _install_panda(runtime, menagerie)

        os.environ.setdefault("MUJOCO_GL", "egl")
        sys.path.insert(0, str(runtime))
        try:
            for name in ("base_scene", "basket_task", "generation"):
                sys.modules.pop(name, None)
            full = importlib.import_module("generation")
            task = full.task
            full.BASE_SEED = seed
            task.WIDTH = image_size
            task.HEIGHT = image_size
            task.base.WIDTH = image_size
            task.base.HEIGHT = image_size
            schema = _release_schema(full.schema(), train_count, test_count, seed)
            metrics: list[dict[str, Any]] = []
            cache = full.ModelCache()
            data_path = output_dir / "data.lance"
            try:
                for chunk_start in range(0, len(indices), 8):
                    record_batches = []
                    chunk = indices[chunk_start : chunk_start + 8]
                    for local_index, source_index in enumerate(chunk, start=chunk_start):
                        assignment = full.assignment_for_episode(source_index)
                        model, prepared, _ = full.prepare_episode(cache, assignment)
                        renderer = cache.renderer(prepared.request.basket, assignment.gravity)
                        result, arrays, _frames, pixels = task.simulate(
                            model,
                            prepared,
                            renderer=renderer,
                        )
                        if not task.outcome_matches(result, prepared.request):
                            raise RuntimeError(f"episode {source_index} changed planned outcome")
                        if not result["finite_arrays"] or len(pixels) != task.FRAMES:
                            raise RuntimeError(f"episode {source_index} produced invalid arrays")
                        columns: dict[str, list[Any]] = {field.name: [] for field in schema}
                        full.append_episode_columns(
                            columns,
                            assignment,
                            prepared,
                            result,
                            arrays,
                            pixels,
                        )
                        if source_start is None:
                            columns["episode_idx"] = [local_index] * task.FRAMES
                        columns["task"] = ["franka_basket"] * task.FRAMES
                        metrics.append(
                            {
                                "source_episode_index": source_index,
                                "split": assignment.split_name,
                                "gravity": float(assignment.gravity),
                                "success": bool(result["success"]),
                                "failure_reason": result["failure_reason"],
                            }
                        )
                        record_batches.append(pa.RecordBatch.from_pydict(columns, schema=schema))
                    table = pa.Table.from_batches(record_batches, schema=schema)
                    lance.write_dataset(
                        table,
                        str(data_path),
                        mode="create" if chunk_start == 0 else "append",
                        max_rows_per_group=1024,
                        max_rows_per_file=65536,
                    )
            finally:
                cache.close()
        finally:
            sys.path.remove(str(runtime))
            for name in ("base_scene", "basket_task", "generation"):
                sys.modules.pop(name, None)

    if lance.dataset(str(data_path)).count_rows() != episodes * 64:
        raise RuntimeError("Franka Lance row count differs from the episode contract")
    metrics_path = output_dir / "episode_metrics.json"
    metrics_path.write_text(json.dumps({"episodes": metrics}, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "task": "franka_basket",
        "split": split,
        "seed": seed,
        "paper_scale": paper_scale,
        "episodes": episodes,
        "train_episodes": train_count,
        "test_episodes": test_count,
        "frames_per_episode": 64,
        "fps": 16,
        "image_size": image_size,
        "dataset": data_path.name,
        "source_episode_indices": indices,
        "source_start": source_start,
        "source": source,
        "asset_provision": asset_provision,
        "asset_manifest": asset_manifest,
        "metrics_sha256": _sha256(metrics_path),
        "qualitative_reproduction": True,
        "status": "pass",
    }
    (output_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


__all__ = ["generate_franka_basket_dataset", "source_manifest"]
