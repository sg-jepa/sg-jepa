"""Shared one-row-per-frame Lance writer."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EpisodeRecord:
    """One rendered episode before it is encoded into Lance rows."""

    episode_index: int
    split: str
    pixels: np.ndarray
    state: np.ndarray
    action: np.ndarray
    gravity: float
    physics: np.ndarray
    source_episode_index: int | None = None
    success: bool = False


def _imports() -> tuple[Any, Any]:
    try:
        import lance
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise ImportError("install sg-jepa[data] to write Lance datasets") from exc
    return lance, pa


def _jpeg(frame: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(frame, mode="RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _metadata(values: Mapping[str, Any]) -> dict[bytes, bytes]:
    encoded: dict[bytes, bytes] = {}
    for key, value in values.items():
        text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        encoded[str(key).encode()] = text.encode()
    return encoded


def write_lance_episodes(
    episodes: Iterable[EpisodeRecord],
    output: str | Path,
    *,
    task: str,
    fps: int,
    image_size: int,
    frames_per_episode: int,
    action_names: Sequence[str],
    state_names: Sequence[str],
    physics_names: Sequence[str],
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Stream episodes into the common Semigroup-JEPA Lance schema."""

    lance, pa = _imports()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "task_name": task,
        "fps": fps,
        "frames_per_episode": frames_per_episode,
        "pixels_shape": [image_size, image_size, 3],
        "pixels_dtype": "uint8",
        "pixels_encoding": "jpeg_rgb",
        "state_schema": list(state_names),
        "action_schema": list(action_names),
        "phys_schema": list(physics_names),
        "split_to_id": {"train": 0, "test": 1},
    }
    metadata.update(extra_metadata or {})
    schema = pa.schema(
        [
            pa.field("episode_idx", pa.int32(), nullable=False),
            pa.field("step_idx", pa.int32(), nullable=False),
            pa.field("split_id", pa.int8(), nullable=False),
            pa.field("pixels", pa.binary(), nullable=False),
            pa.field("state", pa.list_(pa.float32(), len(state_names)), nullable=False),
            pa.field("action", pa.list_(pa.float32(), len(action_names)), nullable=False),
            pa.field("reward", pa.float32(), nullable=False),
            pa.field("phys", pa.list_(pa.float32(), len(physics_names)), nullable=False),
            pa.field("gravity", pa.float32(), nullable=False),
            pa.field("source_episode_index", pa.int32(), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("episode_metadata", pa.string(), nullable=False),
            pa.field("success", pa.bool_(), nullable=False),
        ],
        metadata=_metadata(metadata),
    )

    def batches():
        for episode in episodes:
            if episode.split not in {"train", "test"}:
                raise ValueError(f"invalid split {episode.split!r}")
            frame_count = int(episode.pixels.shape[0])
            if frame_count != frames_per_episode:
                raise ValueError("episode length differs from frames_per_episode")
            if episode.state.shape != (frame_count, len(state_names)):
                raise ValueError("state shape differs from schema")
            if episode.action.shape != (frame_count, len(action_names)):
                raise ValueError("action shape differs from schema")
            physics = np.asarray(episode.physics, dtype=np.float32).reshape(1, -1)
            if physics.shape[1] != len(physics_names):
                raise ValueError("physics shape differs from schema")
            physics = np.repeat(physics, frame_count, axis=0)
            source_episode_index = (
                episode.episode_index
                if episode.source_episode_index is None
                else episode.source_episode_index
            )
            episode_metadata = json.dumps(
                {
                    "task": task,
                    "episode_index": episode.episode_index,
                    "source_episode_index": source_episode_index,
                    "split": episode.split,
                    "gravity": episode.gravity,
                    "success": bool(episode.success),
                },
                sort_keys=True,
            )
            arrays = [
                pa.array([episode.episode_index] * frame_count, type=pa.int32()),
                pa.array(np.arange(frame_count), type=pa.int32()),
                pa.array([0 if episode.split == "train" else 1] * frame_count, type=pa.int8()),
                pa.array([_jpeg(frame) for frame in episode.pixels], type=pa.binary()),
                pa.array(episode.state.tolist(), type=schema.field("state").type),
                pa.array(episode.action.tolist(), type=schema.field("action").type),
                pa.array(np.zeros(frame_count, dtype=np.float32), type=pa.float32()),
                pa.array(physics.tolist(), type=schema.field("phys").type),
                pa.array([episode.gravity] * frame_count, type=pa.float32()),
                pa.array([source_episode_index] * frame_count, type=pa.int32()),
                pa.array([task] * frame_count, type=pa.string()),
                pa.array([episode_metadata] * frame_count, type=pa.string()),
                pa.array([bool(episode.success)] * frame_count, type=pa.bool_()),
            ]
            yield pa.RecordBatch.from_arrays(arrays, schema=schema)

    reader = pa.RecordBatchReader.from_batches(schema, batches())
    lance.write_dataset(reader, str(output), mode="create")
    return output


__all__ = ["EpisodeRecord", "write_lance_episodes"]
