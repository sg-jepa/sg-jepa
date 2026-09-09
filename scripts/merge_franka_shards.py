#!/usr/bin/env python3
"""Validate and atomically merge deterministic Franka Lance shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "generation_manifest.json"
    if not path.is_file():
        raise ValueError(f"Franka shard has no generation manifest: {root}")
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "pass" or manifest.get("task") != "franka_basket":
        raise ValueError(f"Franka shard manifest did not pass: {path}")
    if manifest.get("split") != "all" or manifest.get("paper_scale") is not False:
        raise ValueError(f"expected a bounded split=all Franka shard: {path}")
    return manifest


def _validate_lance_shard(
    root: Path,
    manifest: dict[str, Any],
) -> tuple[Any, list[int], list[dict[str, Any]]]:
    dataset_path = root / str(manifest.get("dataset", ""))
    if not dataset_path.is_dir() or dataset_path.suffix != ".lance":
        raise ValueError(f"invalid Lance path in Franka shard: {dataset_path}")
    dataset = lance.dataset(str(dataset_path))
    source_indices = [int(value) for value in manifest["source_episode_indices"]]
    if len(source_indices) != int(manifest["episodes"]):
        raise ValueError(f"source index count differs in {root}")
    if len(set(source_indices)) != len(source_indices):
        raise ValueError(f"source indices are not unique in {root}")

    observed_sources: list[int] = []
    previous_episode: int | None = None
    expected_step = 0
    rows = 0
    for batch in dataset.to_batches(
        columns=["episode_idx", "step_idx", "split_id", "source_episode_index"],
        batch_size=8192,
    ):
        values = [column.to_pylist() for column in batch.columns]
        for episode, step, split_id, source in zip(*values, strict=True):
            episode = int(episode)
            step = int(step)
            split_id = int(split_id)
            source = int(source)
            if episode != source:
                raise ValueError(f"episode/source ID mismatch in {root}: {episode} != {source}")
            if split_id != (0 if source < 8_000 else 1):
                raise ValueError(f"canonical split mismatch for source {source} in {root}")
            if episode != previous_episode:
                if previous_episode is not None and expected_step != 64:
                    raise ValueError(f"episode {previous_episode} does not contain 64 frames")
                if step != 0:
                    raise ValueError(f"episode {episode} does not begin at step zero")
                if observed_sources and episode <= observed_sources[-1]:
                    raise ValueError(f"episodes are not strictly ordered in {root}")
                observed_sources.append(source)
                previous_episode = episode
                expected_step = 0
            if step != expected_step:
                raise ValueError(
                    f"non-contiguous steps in episode {episode}: {step} != {expected_step}"
                )
            expected_step += 1
            rows += 1
    if previous_episode is not None and expected_step != 64:
        raise ValueError(f"episode {previous_episode} does not contain 64 frames")
    if observed_sources != source_indices:
        raise ValueError(f"Lance source order differs from the manifest in {root}")
    if rows != 64 * len(source_indices) or dataset.count_rows() != rows:
        raise ValueError(f"Franka shard row count differs in {root}")

    metrics_path = root / "episode_metrics.json"
    metrics_payload = json.loads(metrics_path.read_text())
    metrics = metrics_payload.get("episodes")
    if not isinstance(metrics, list):
        raise ValueError(f"invalid episode metrics in {root}")
    metric_sources = [int(item["source_episode_index"]) for item in metrics]
    if metric_sources != source_indices:
        raise ValueError(f"metric source order differs from Lance in {root}")
    if _sha256(metrics_path) != manifest.get("metrics_sha256"):
        raise ValueError(f"episode metrics checksum differs in {root}")
    return dataset, source_indices, metrics


def _target_schema(schema: pa.Schema, train_episodes: int, test_episodes: int, seed: int):
    metadata = dict(schema.metadata or {})
    metadata.update(
        {
            b"schema_version": b"1",
            b"public_task_alias": b"franka_basket",
            b"train_episodes": str(train_episodes).encode(),
            b"test_episodes": str(test_episodes).encode(),
            b"base_seed": str(seed).encode(),
        }
    )
    return schema.with_metadata(metadata)


def _batches(datasets: Sequence[Any], schema: pa.Schema) -> Iterator[pa.RecordBatch]:
    for dataset in datasets:
        for batch in dataset.to_batches(batch_size=4096):
            yield pa.RecordBatch.from_arrays(batch.columns, schema=schema)


def _staging_path(output: Path) -> Path:
    job = os.environ.get("SLURM_JOB_ID", "interactive")
    return output.with_name(f".{output.name}.assembling-{job}-{os.getpid()}")


def merge(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    if args.episodes != args.train_episodes + args.test_episodes:
        raise ValueError("train/test episode counts do not sum to episodes")
    if args.paper_scale and (
        args.episodes != 13_000 or args.train_episodes != 8_000 or args.test_episodes != 5_000
    ):
        raise ValueError("paper-scale Franka merge requires 8000 train + 5000 test episodes")

    shard_records = []
    for raw_root in args.shard:
        root = raw_root.expanduser().resolve()
        manifest = _load_manifest(root)
        if int(manifest.get("seed", -1)) != args.seed:
            raise ValueError(f"base seed differs in {root}")
        dataset, sources, metrics = _validate_lance_shard(root, manifest)
        shard_records.append((sources[0], root, manifest, dataset, sources, metrics))
    shard_records.sort(key=lambda item: item[0])
    datasets = [item[3] for item in shard_records]
    first_schema = datasets[0].schema
    if any(
        not first_schema.equals(dataset.schema, check_metadata=False) for dataset in datasets[1:]
    ):
        raise ValueError("Franka shard field schemas differ")

    source_indices = [source for item in shard_records for source in item[4]]
    if len(source_indices) != args.episodes or len(set(source_indices)) != args.episodes:
        raise ValueError("merged Franka source IDs are missing or duplicated")
    if source_indices != sorted(source_indices):
        raise ValueError("merged Franka source IDs are not strictly ordered")
    train_count = sum(source < 8_000 for source in source_indices)
    if train_count != args.train_episodes or args.episodes - train_count != args.test_episodes:
        raise ValueError("merged Franka split counts differ from the requested counts")
    if args.paper_scale and source_indices != list(range(13_000)):
        raise ValueError("paper-scale Franka shards must cover canonical sources 0..12999")

    reference = shard_records[0][2]
    for _, root, manifest, _, _, _ in shard_records[1:]:
        for key in ("source", "asset_manifest"):
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"{key} differs in {root}")

    staging = _staging_path(output)
    if staging.exists():
        raise FileExistsError(f"refusing to replace stale staging directory: {staging}")
    staging.mkdir(parents=True)
    data_path = staging / "data.lance"
    schema = _target_schema(first_schema, args.train_episodes, args.test_episodes, args.seed)
    reader = pa.RecordBatchReader.from_batches(schema, _batches(datasets, schema))
    lance.write_dataset(
        reader,
        str(data_path),
        mode="create",
        max_rows_per_group=1024,
        max_rows_per_file=65536,
    )
    final_dataset = lance.dataset(str(data_path))
    if final_dataset.count_rows() != args.episodes * 64:
        raise RuntimeError("merged Franka Lance row count differs from the episode contract")

    metrics = [metric for item in shard_records for metric in item[5]]
    metrics_path = staging / "episode_metrics.json"
    metrics_path.write_text(json.dumps({"episodes": metrics}, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "task": "franka_basket",
        "split": "all",
        "seed": args.seed,
        "paper_scale": args.paper_scale,
        "episodes": args.episodes,
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "frames_per_episode": 64,
        "fps": int(reference["fps"]),
        "image_size": int(reference["image_size"]),
        "dataset": data_path.name,
        "source_episode_indices": source_indices,
        "source": reference["source"],
        "asset_provision": reference["asset_provision"],
        "asset_manifest": reference["asset_manifest"],
        "metrics_sha256": _sha256(metrics_path),
        "shards": [
            {
                "path": str(root),
                "episodes": len(sources),
                "source_start": sources[0],
                "source_stop": sources[-1] + 1,
                "manifest_sha256": _sha256(root / "generation_manifest.json"),
            }
            for _, root, _, _, sources, _ in shard_records
        ],
        "qualitative_reproduction": True,
        "status": "pass",
    }
    (staging / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if output.exists():
        raise FileExistsError(f"output appeared while merging: {output}")
    staging.rename(output)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--train-episodes", type=int, required=True)
    parser.add_argument("--test-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--paper-scale", action="store_true")
    return parser


def main() -> None:
    manifest = merge(build_parser().parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
