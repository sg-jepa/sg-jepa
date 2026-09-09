"""Streaming reader for Semigroup-JEPA Lance trajectory datasets."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ActionStatistics:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @classmethod
    def fit(cls, action: np.ndarray) -> ActionStatistics:
        flat = np.asarray(action, dtype=np.float64).reshape(-1, action.shape[-1])
        mean = flat.mean(axis=0)
        std = np.maximum(flat.std(axis=0), 1.0e-6)
        return cls(tuple(float(value) for value in mean), tuple(float(value) for value in std))

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> ActionStatistics:
        return cls(tuple(payload["mean"]), tuple(payload["std"]))


@dataclass(frozen=True)
class Episode:
    pixels: np.ndarray
    action: np.ndarray
    state: np.ndarray
    gravity: np.ndarray
    episode_id: str
    success: bool | None = None
    paddle_state: np.ndarray | None = None
    task_event: np.ndarray | None = None


@dataclass(frozen=True)
class EpisodeSplit:
    """Deterministic development-set partition expressed in source episode IDs."""

    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    train_episode_ids: tuple[int, ...]
    val_episode_ids: tuple[int, ...]
    seed: int
    train_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "episode",
            "source_split": "train",
            "seed": self.seed,
            "train_fraction": self.train_fraction,
            "episode_count": len(self.train_indices) + len(self.val_indices),
            "train_episode_count": len(self.train_indices),
            "val_episode_count": len(self.val_indices),
            "train_episode_indices": list(self.train_episode_ids),
            "val_episode_indices": list(self.val_episode_ids),
        }


@dataclass(frozen=True)
class _EpisodeRows:
    row_ids: np.ndarray
    source_id: int


def _decode_metadata(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


class LanceTrajectoryStore:
    """On-demand episode reader for the public JPEG-in-Lance contract."""

    def __init__(self, path: str | Path) -> None:
        try:
            import lance
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise ImportError("install sg-jepa[data] to read Lance datasets") from exc
        self.path = Path(path).expanduser().resolve()
        self.dataset = lance.dataset(str(self.path))
        raw_metadata = self.dataset.schema.metadata or {}
        self.metadata = {
            key.decode() if isinstance(key, bytes) else str(key): _decode_metadata(value)
            for key, value in raw_metadata.items()
        }
        required = {
            "episode_idx",
            "step_idx",
            "split_id",
            "pixels",
            "state",
            "action",
            "gravity",
            "source_episode_index",
        }
        names = set(self.dataset.schema.names)
        if missing := required - names:
            raise ValueError(f"Lance dataset is missing required columns: {sorted(missing)}")
        self.has_success = "success" in names
        self.has_paddle_state = "paddle_state" in names
        self.has_task_event = "task_event" in names
        index_columns = ["episode_idx", "step_idx", "split_id", "source_episode_index"]
        if self.has_success:
            index_columns.append("success")
        table = self.dataset.to_table(columns=index_columns)
        episode_idx = self._numpy(table.column("episode_idx"), np.int64)
        step_idx = self._numpy(table.column("step_idx"), np.int64)
        split_rows = self._numpy(table.column("split_id"), np.int8)
        source_rows = self._numpy(table.column("source_episode_index"), np.int64)
        success_rows = self._numpy(table.column("success"), np.bool_) if self.has_success else None
        if not len(episode_idx):
            raise ValueError(f"empty Lance dataset: {self.path}")
        boundaries = np.r_[
            0,
            np.flatnonzero(episode_idx[1:] != episode_idx[:-1]) + 1,
            len(episode_idx),
        ]
        episodes: list[_EpisodeRows] = []
        splits: list[int] = []
        lengths: list[int] = []
        successes: list[bool] = []
        for first, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            if not np.array_equal(step_idx[first:stop], np.arange(stop - first)):
                raise ValueError("Lance rows are not contiguous, ordered episodes")
            if np.unique(split_rows[first:stop]).size != 1:
                raise ValueError("split_id changes within an episode")
            if np.unique(source_rows[first:stop]).size != 1:
                raise ValueError("source_episode_index changes within an episode")
            if success_rows is not None:
                if np.unique(success_rows[first:stop]).size != 1:
                    raise ValueError("success changes within an episode")
                successes.append(bool(success_rows[first]))
            episodes.append(
                _EpisodeRows(np.arange(first, stop, dtype=np.int64), int(source_rows[first]))
            )
            splits.append(int(split_rows[first]))
            lengths.append(stop - first)
        self.episodes = episodes
        self.split_id = np.asarray(splits, dtype=np.int8)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.success = np.asarray(successes, dtype=bool) if self.has_success else None
        self.action_dim = int(self.dataset.schema.field("action").type.list_size)

    def __getstate__(self) -> dict[str, Any]:
        """Drop the native Lance handle when a spawn worker serializes the store."""

        state = dict(self.__dict__)
        state["dataset"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Open an independent Lance handle inside each spawned worker."""

        import lance

        self.__dict__.update(state)
        self.dataset = lance.dataset(str(self.path))

    @staticmethod
    def _numpy(column: Any, dtype: np.dtype) -> np.ndarray:
        if hasattr(column, "combine_chunks"):
            column = column.combine_chunks()
        return np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype)

    @staticmethod
    def _fixed_list(table: Any, name: str, dtype: np.dtype) -> np.ndarray:
        return np.asarray(table.column(name).to_pylist(), dtype=dtype)

    def episode(self, index: int, rows: slice = slice(None)) -> Episode:
        record = self.episodes[index]
        selected_rows = record.row_ids[rows]
        if not len(selected_rows):
            raise ValueError("episode slice must contain at least one frame")
        columns = ["pixels", "action", "state", "gravity", "step_idx"]
        if self.has_paddle_state:
            columns.append("paddle_state")
        if self.has_task_event:
            columns.append("task_event")
        table = self.dataset.take(
            selected_rows.tolist(),
            columns=columns,
        )
        expected_steps = np.arange(len(record.row_ids), dtype=np.int64)[rows]
        if not np.array_equal(self._numpy(table.column("step_idx"), np.int64), expected_steps):
            raise ValueError("Lance take changed episode row order")
        pixels = np.stack(
            [
                np.asarray(Image.open(BytesIO(value)).convert("RGB"), dtype=np.uint8)
                for value in table.column("pixels").to_pylist()
            ]
        )
        success = None if self.success is None else bool(self.success[index])
        paddle_state = (
            self._fixed_list(table, "paddle_state", np.float32) if self.has_paddle_state else None
        )
        task_event = (
            self._fixed_list(table, "task_event", np.uint8) if self.has_task_event else None
        )
        return Episode(
            pixels=pixels,
            action=self._fixed_list(table, "action", np.float32),
            state=self._fixed_list(table, "state", np.float32),
            gravity=self._numpy(table.column("gravity"), np.float32),
            episode_id=str(record.source_id),
            success=success,
            paddle_state=paddle_state,
            task_event=task_event,
        )

    def actions(self, indices: np.ndarray) -> np.ndarray:
        if not len(indices):
            raise ValueError("cannot fit action statistics without training episodes")
        rows = np.concatenate([self.episodes[int(index)].row_ids for index in indices])
        table = self.dataset.take(rows.tolist(), columns=["action"])
        return self._fixed_list(table, "action", np.float32)


def open_trajectory_store(path: str | Path) -> LanceTrajectoryStore:
    path = Path(path).expanduser().resolve()
    if (path / "generation_manifest.json").is_file():
        declared = json.loads((path / "generation_manifest.json").read_text()).get("dataset")
        if not isinstance(declared, str) or not declared:
            raise ValueError("generation manifest does not declare a dataset")
        candidate = Path(declared).expanduser()
        if not candidate.is_absolute():
            candidate = path / candidate
        path = candidate.resolve()
    elif (path / "data.lance").is_dir():
        path = path / "data.lance"
    if not path.is_dir() or path.suffix != ".lance":
        raise ValueError(f"expected a Lance dataset or generator output directory: {path}")
    return LanceTrajectoryStore(path)


def split_development_episodes(
    store: LanceTrajectoryStore,
    *,
    train_fraction: float = 0.9,
    seed: int = 42,
) -> EpisodeSplit:
    """Split the development population without allowing episode leakage."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    development = np.flatnonzero(store.split_id == 0)
    if len(development) < 2:
        raise ValueError("episode-level train/validation splitting requires two episodes")
    shuffled = development.copy()
    np.random.default_rng(int(seed)).shuffle(shuffled)
    train_count = int(np.floor(len(shuffled) * float(train_fraction)))
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    train_indices = tuple(sorted(int(value) for value in shuffled[:train_count]))
    val_indices = tuple(sorted(int(value) for value in shuffled[train_count:]))
    train_ids = tuple(int(store.episodes[index].source_id) for index in train_indices)
    val_ids = tuple(int(store.episodes[index].source_id) for index in val_indices)
    if set(train_ids) & set(val_ids):
        raise ValueError("source episode IDs overlap across train and validation")
    return EpisodeSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        train_episode_ids=train_ids,
        val_episode_ids=val_ids,
        seed=int(seed),
        train_fraction=float(train_fraction),
    )


TASK_ACTION_DIMS = {
    "right_triangle": 3,
    "square": 3,
    "approach_ball": 1,
    "arm_catcher_ball": 4,
    "arm_paddle_ball": 6,
    "franka_basket": 6,
    "franka_paddle_hit_ball_to_basket": 6,
}


def validate_trajectory_dataset(
    path: str | Path,
    *,
    expected_task: str | None = None,
    expected_split: str | None = None,
    sample_episodes: int = 8,
) -> dict[str, Any]:
    """Validate schema and representative episode payloads without loading all pixels."""

    if expected_split not in {None, "train", "test", "all"}:
        raise ValueError("expected_split must be train, test, all, or None")
    store = open_trajectory_store(path)
    unique_splits = set(int(value) for value in np.unique(store.split_id))
    if not unique_splits.issubset({0, 1}):
        raise ValueError("split_id must use train=0 and test=1")
    if expected_split in {"train", "test"}:
        wanted = {0 if expected_split == "train" else 1}
        if unique_splits != wanted:
            raise ValueError(f"dataset does not contain only the requested {expected_split} split")
    metadata = store.metadata
    task = metadata.get("public_task_alias", metadata.get("task_name", metadata.get("task")))
    aliases = {"franka_paddle_hit_ball_to_basket": "franka_basket"}
    canonical_task = aliases.get(task, task)
    canonical_expected = aliases.get(expected_task, expected_task)
    if canonical_expected is not None and canonical_task != canonical_expected:
        raise ValueError(f"expected task {expected_task!r}, got {task!r}")
    if canonical_expected in TASK_ACTION_DIMS:
        expected_dim = TASK_ACTION_DIMS[canonical_expected]
        if store.action_dim != expected_dim:
            raise ValueError(f"{canonical_expected} requires action_dim={expected_dim}")
    selected = np.unique(
        np.linspace(0, len(store.lengths) - 1, min(sample_episodes, len(store.lengths))).astype(int)
    )
    image_shape = None
    state_dim = None
    for index in selected:
        episode = store.episode(int(index))
        length = int(store.lengths[index])
        if episode.pixels.shape[0] != length:
            raise ValueError("pixel time axis differs from episode length")
        if episode.action.shape != (length, store.action_dim):
            raise ValueError("action shape differs from schema")
        if episode.state.shape[0] != length or episode.gravity.shape != (length,):
            raise ValueError("state/gravity time axes differ from episode length")
        if not all(
            np.isfinite(value).all() for value in (episode.action, episode.state, episode.gravity)
        ):
            raise ValueError("trajectory contains non-finite values")
        image_shape = list(episode.pixels.shape[1:])
        state_dim = int(episode.state.shape[-1])
    return {
        "schema_version": 1,
        "task": canonical_task,
        "episodes": len(store.lengths),
        "train_episodes": int((store.split_id == 0).sum()),
        "test_episodes": int((store.split_id == 1).sum()),
        "frames_min": int(store.lengths.min()),
        "frames_max": int(store.lengths.max()),
        "action_dim": store.action_dim,
        "state_dim": state_dim,
        "image_shape": image_shape,
        "sampled_episodes": len(selected),
        "status": "pass",
    }


def _tensor_episode(
    episode: Episode,
    statistics: ActionStatistics,
    rows: slice,
) -> dict[str, torch.Tensor | str]:
    pixels = np.asarray(episode.pixels[rows])
    if pixels.ndim != 4:
        raise ValueError("pixels must have shape [T,H,W,C] or [T,C,H,W]")
    if pixels.shape[-1] in {1, 3}:
        pixels = np.moveaxis(pixels, -1, 1)
    if pixels.shape[1] != 3:
        raise ValueError("only RGB trajectories are supported")
    pixels_tensor = torch.from_numpy(np.ascontiguousarray(pixels)).float().div_(255.0)
    image_mean = torch.tensor((0.485, 0.456, 0.406))[None, :, None, None]
    image_std = torch.tensor((0.229, 0.224, 0.225))[None, :, None, None]
    mean = torch.tensor(statistics.mean, dtype=torch.float32)
    std = torch.tensor(statistics.std, dtype=torch.float32)
    action = (torch.from_numpy(np.asarray(episode.action[rows]).copy()) - mean) / std
    return {
        "pixels": (pixels_tensor - image_mean) / image_std,
        "action": action,
        "state": torch.from_numpy(np.asarray(episode.state[rows], dtype=np.float32).copy()),
        "gravity": torch.from_numpy(np.asarray(episode.gravity[rows], dtype=np.float32).copy()),
        "episode_id": episode.episode_id,
    }


class TrajectoryDataset(Dataset):
    """One item per episode, normalized with train-split action statistics."""

    def __init__(
        self,
        path: str | Path,
        *,
        split: str = "train",
        action_statistics: ActionStatistics | None = None,
        store: LanceTrajectoryStore | None = None,
        episode_indices: Sequence[int] | None = None,
    ) -> None:
        if split not in {"train", "test", "all"}:
            raise ValueError("split must be train, test, or all")
        self.path = Path(path)
        self.store = store or open_trajectory_store(path)
        wanted = np.ones(len(self.store.lengths), dtype=bool)
        if split != "all":
            wanted = self.store.split_id == (0 if split == "train" else 1)
        if episode_indices is not None:
            requested = np.asarray(tuple(int(value) for value in episode_indices), dtype=np.int64)
            if requested.ndim != 1 or len(np.unique(requested)) != len(requested):
                raise ValueError("episode_indices must be a one-dimensional unique sequence")
            if len(requested) and (requested.min() < 0 or requested.max() >= len(wanted)):
                raise IndexError("episode_indices contain an out-of-range store index")
            selected = np.zeros(len(wanted), dtype=bool)
            selected[requested] = True
            if np.any(selected & ~wanted):
                raise ValueError("episode_indices select episodes outside the requested split")
            wanted &= selected
        self.indices = np.flatnonzero(wanted)
        if not len(self.indices):
            raise ValueError(f"dataset has no {split} episodes")
        if action_statistics is None:
            train = self.indices if split != "test" else np.flatnonzero(self.store.split_id == 0)
            action_statistics = ActionStatistics.fit(self.store.actions(train))
        self.action_statistics = action_statistics
        self.action_dim = self.store.action_dim
        self.frames_per_episode = int(self.store.lengths[self.indices].min())

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return _tensor_episode(
            self.store.episode(int(self.indices[index])),
            self.action_statistics,
            slice(None),
        )


class TrajectoryWindowDataset(Dataset):
    """All contiguous H+K training windows, matching the paper sampler."""

    def __init__(
        self,
        path: str | Path,
        *,
        num_steps: int,
        split: str = "train",
        action_statistics: ActionStatistics | None = None,
        store: LanceTrajectoryStore | None = None,
        episode_indices: Sequence[int] | None = None,
    ) -> None:
        if num_steps <= 1:
            raise ValueError("num_steps must exceed one")
        self.episodes = TrajectoryDataset(
            path,
            split=split,
            action_statistics=action_statistics,
            store=store,
            episode_indices=episode_indices,
        )
        self.action_statistics = self.episodes.action_statistics
        self.action_dim = self.episodes.action_dim
        self.num_steps = int(num_steps)
        self.windows: list[tuple[int, int]] = []
        for local_index, store_index in enumerate(self.episodes.indices):
            length = int(self.episodes.store.lengths[int(store_index)])
            self.windows.extend(
                (local_index, start) for start in range(length - self.num_steps + 1)
            )
        if not self.windows:
            raise ValueError(f"dataset has no windows of {self.num_steps} frames")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        local_index, start = self.windows[index]
        store_index = int(self.episodes.indices[local_index])
        return _tensor_episode(
            self.episodes.store.episode(
                store_index,
                slice(start, start + self.num_steps),
            ),
            self.action_statistics,
            slice(None),
        )


__all__ = [
    "ActionStatistics",
    "EpisodeSplit",
    "LanceTrajectoryStore",
    "TrajectoryDataset",
    "TrajectoryWindowDataset",
    "open_trajectory_store",
    "split_development_episodes",
    "validate_trajectory_dataset",
]
