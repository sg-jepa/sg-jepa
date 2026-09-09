"""Deterministic policy-window sampling contracts used by paper checkpoints."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler

from sg_jepa.checkpoints import sha256_file

from .config import (
    PAPER_FRANKA_BATCH_SAMPLING,
    PAPER_PADDLE_BATCH_SAMPLING,
    UNIFORM_BATCH_SAMPLING,
)

PAPER_PADDLE_SAMPLER_KIND = "deterministic_two_stream_with_replacement_v1"
PAPER_PADDLE_BATCH_SIZE = 256
PAPER_PADDLE_LEGACY_COUNT = 205
PAPER_PADDLE_QUALITY_COUNT = 51
PAPER_FRANKA_SAMPLER_KIND = "deterministic_strike_balanced_v1"
PAPER_FRANKA_APPROACH_HORIZON = 16


@dataclass(frozen=True)
class PaddleQualityWeights:
    """Source-episode keyed sampling weights from the final Paddle workflow."""

    source_episode_ids: np.ndarray
    window_starts: np.ndarray
    weights: np.ndarray
    sha256: str

    def provenance(self, configured_path: str) -> dict[str, Any]:
        return {
            "kind": "all_success_soft_quality_phase_local_v1",
            "configured_path": configured_path,
            "sha256": self.sha256,
            "episode_count": int(self.source_episode_ids.size),
            "windows_per_episode": int(self.window_starts.size),
        }


def load_paddle_quality_weights(
    path: str | Path,
    *,
    expected_sha256: str,
) -> PaddleQualityWeights:
    """Load and strictly validate the compact final-paper quality artifact."""

    artifact = Path(path).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"Paddle quality weights do not exist: {artifact}")
    digest = sha256_file(artifact)
    if digest != expected_sha256:
        raise RuntimeError(f"Paddle quality weights SHA256 differs: {digest} != {expected_sha256}")
    with np.load(artifact, allow_pickle=False) as payload:
        required = {
            "schema_version",
            "source_episode_ids",
            "window_starts",
            "quality_sampling_weights",
            "gradient_eligible",
        }
        if missing := required - set(payload.files):
            raise ValueError(f"Paddle quality artifact lacks arrays: {sorted(missing)}")
        schema_version = np.asarray(payload["schema_version"])
        source_ids = np.asarray(payload["source_episode_ids"], dtype=np.int64)
        starts = np.asarray(payload["window_starts"], dtype=np.int64)
        weights = np.asarray(payload["quality_sampling_weights"], dtype=np.float64)
        eligible = np.asarray(payload["gradient_eligible"], dtype=bool)
    if schema_version.shape != (1,) or int(schema_version[0]) != 1:
        raise ValueError("unsupported Paddle quality artifact schema")
    if (
        source_ids.ndim != 1
        or len(source_ids) == 0
        or len(set(source_ids.tolist())) != len(source_ids)
    ):
        raise ValueError("Paddle quality source episode IDs must be non-empty and unique")
    if starts.ndim != 1 or not np.array_equal(starts, np.arange(len(starts))):
        raise ValueError("Paddle quality window starts must be contiguous from zero")
    if weights.shape != (len(source_ids), len(starts)):
        raise ValueError("Paddle quality weight matrix has an incompatible shape")
    if eligible.shape != (len(source_ids),) or not eligible.all():
        raise ValueError("Paddle quality artifact must mark every source episode eligible")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("Paddle quality weights must be finite and strictly positive")
    return PaddleQualityWeights(source_ids, starts, weights, digest)


def align_paddle_quality_weights(
    dataset: Any,
    artifact: PaddleQualityWeights,
    *,
    require_complete_population: bool = False,
) -> np.ndarray:
    """Align source-episode/window weights to a policy-window dataset."""

    lookup = {
        int(source_id): row for row, source_id in enumerate(artifact.source_episode_ids.tolist())
    }
    dataset_source_ids = {int(episode.source_id) for episode in dataset.episodes}
    artifact_source_ids = set(lookup)
    if not dataset_source_ids.issubset(artifact_source_ids) or (
        require_complete_population and dataset_source_ids != artifact_source_ids
    ):
        absent = sorted(dataset_source_ids - artifact_source_ids)[:8]
        extra = sorted(artifact_source_ids - dataset_source_ids)[:8]
        raise RuntimeError(
            "Paddle quality population differs from successful policy demonstrations: "
            f"absent={absent}, extra={extra}"
        )
    result = np.empty(len(dataset), dtype=np.float64)
    for index, (episode_index, start) in enumerate(dataset.windows):
        source_id = int(dataset.episodes[int(episode_index)].source_id)
        if not 0 <= int(start) < artifact.weights.shape[1]:
            raise ValueError(f"Paddle policy window start is outside the quality artifact: {start}")
        result[index] = artifact.weights[lookup[source_id], int(start)]
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise RuntimeError("aligned Paddle quality weights are not strictly positive")
    return result / float(result.sum())


def validate_paddle_quality_population(
    source_episode_ids: Iterator[int] | list[int] | tuple[int, ...],
    artifact: PaddleQualityWeights,
    *,
    require_complete_population: bool = False,
) -> None:
    """Fail before feature caching when demonstration identities do not align."""

    dataset_source_ids = {int(value) for value in source_episode_ids}
    artifact_source_ids = {int(value) for value in artifact.source_episode_ids.tolist()}
    if not dataset_source_ids.issubset(artifact_source_ids) or (
        require_complete_population and dataset_source_ids != artifact_source_ids
    ):
        absent = sorted(dataset_source_ids - artifact_source_ids)[:8]
        extra = sorted(artifact_source_ids - dataset_source_ids)[:8]
        raise RuntimeError(
            "Paddle quality population differs from successful policy demonstrations: "
            f"absent={absent}, extra={extra}"
        )


class DeterministicPaperPaddleBatchSampler(Sampler[list[int]]):
    """Exact 205 uniform-replay + 51 soft-quality batches with replacement."""

    CHUNK_STEPS = 1024

    def __init__(
        self,
        dataset_size: int,
        *,
        quality_weights: np.ndarray,
        seed: int,
        start_step: int,
        total_steps: int,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.total_steps = int(total_steps)
        values = np.asarray(quality_weights, dtype=np.float64)
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if values.shape != (self.dataset_size,) or not np.isfinite(values).all():
            raise ValueError("quality_weights must be finite and dataset-aligned")
        if np.any(values <= 0.0):
            raise ValueError("every Paddle quality weight must be strictly positive")
        if not 0 <= self.start_step <= self.total_steps:
            raise ValueError("invalid start/total step interval")
        self.quality_weights = torch.as_tensor(values / float(values.sum()), dtype=torch.float64)

    def __len__(self) -> int:
        return self.total_steps - self.start_step

    def __iter__(self) -> Iterator[list[int]]:
        cached_chunk = -1
        legacy: torch.Tensor | None = None
        quality: torch.Tensor | None = None
        for step in range(self.start_step, self.total_steps):
            chunk, offset = divmod(step, self.CHUNK_STEPS)
            if chunk != cached_chunk:
                legacy_generator = torch.Generator().manual_seed(self.seed + 2_000_003 * chunk)
                quality_generator = torch.Generator().manual_seed(
                    self.seed + 1_000_003 + 2_000_003 * chunk
                )
                legacy = torch.randint(
                    self.dataset_size,
                    (self.CHUNK_STEPS, PAPER_PADDLE_LEGACY_COUNT),
                    generator=legacy_generator,
                )
                quality = torch.multinomial(
                    self.quality_weights,
                    self.CHUNK_STEPS * PAPER_PADDLE_QUALITY_COUNT,
                    replacement=True,
                    generator=quality_generator,
                ).reshape(self.CHUNK_STEPS, PAPER_PADDLE_QUALITY_COUNT)
                cached_chunk = chunk
            assert legacy is not None and quality is not None
            yield torch.cat((legacy[offset], quality[offset])).tolist()

    def contract(self) -> dict[str, int | str]:
        return {
            "kind": PAPER_PADDLE_SAMPLER_KIND,
            "batch_size": PAPER_PADDLE_BATCH_SIZE,
            "legacy_replay_count": PAPER_PADDLE_LEGACY_COUNT,
            "soft_quality_count": PAPER_PADDLE_QUALITY_COUNT,
            "seed": self.seed,
            "chunk_steps": self.CHUNK_STEPS,
        }


class DeterministicFrankaStrikeBatchSampler(Sampler[list[int]]):
    """Exact half-uniform, half-final-approach sampling from the paper run."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        *,
        seed: int,
        start_step: int,
        total_steps: int,
        approach_horizon: int = PAPER_FRANKA_APPROACH_HORIZON,
    ) -> None:
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("strike-balanced sampling requires a positive even batch")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.total_steps = int(total_steps)
        self.approach_horizon = int(approach_horizon)
        if not 0 <= self.start_step <= self.total_steps or self.approach_horizon <= 0:
            raise ValueError("invalid strike-balanced sampler schedule")

        uniform: list[int] = []
        approach: list[int] = []
        for window_index, (episode_index, start) in enumerate(dataset.windows):
            episode = dataset.episodes[int(episode_index)]
            strike = episode.strike_frame
            if strike is None or int(strike) < 0:
                raise ValueError("strike-balanced episodes must contain valid blade contact")
            last = min(len(episode.controls) - dataset.config.action_horizon - 1, int(strike) - 1)
            if last < 0:
                raise ValueError("strike occurs before the first policy transition")
            if int(start) <= last:
                uniform.append(window_index)
            if max(0, int(strike) - self.approach_horizon) <= int(start) <= last:
                approach.append(window_index)
        self.uniform_indices = torch.as_tensor(uniform, dtype=torch.long)
        self.approach_indices = torch.as_tensor(approach, dtype=torch.long)
        if self.uniform_indices.numel() == 0 or self.approach_indices.numel() == 0:
            raise ValueError("strike-balanced sampler populations are empty")

    def __len__(self) -> int:
        return self.total_steps - self.start_step

    def __iter__(self) -> Iterator[list[int]]:
        half = self.batch_size // 2
        for global_step in range(self.start_step, self.total_steps):
            generator = torch.Generator().manual_seed(self.seed + 1_000_003 * global_step)
            uniform = self.uniform_indices[
                torch.randint(self.uniform_indices.numel(), (half,), generator=generator)
            ]
            approach = self.approach_indices[
                torch.randint(self.approach_indices.numel(), (half,), generator=generator)
            ]
            order = torch.randperm(self.batch_size, generator=generator)
            yield torch.cat((uniform, approach))[order].tolist()

    def contract(self) -> dict[str, int | str]:
        return {
            "kind": PAPER_FRANKA_SAMPLER_KIND,
            "batch_size": self.batch_size,
            "uniform_per_batch": self.batch_size // 2,
            "approach_per_batch": self.batch_size // 2,
            "approach_horizon": self.approach_horizon,
            "uniform_population": int(self.uniform_indices.numel()),
            "approach_population": int(self.approach_indices.numel()),
            "seed": self.seed,
        }


__all__ = [
    "DeterministicFrankaStrikeBatchSampler",
    "DeterministicPaperPaddleBatchSampler",
    "PAPER_FRANKA_APPROACH_HORIZON",
    "PAPER_FRANKA_BATCH_SAMPLING",
    "PAPER_FRANKA_SAMPLER_KIND",
    "PAPER_PADDLE_BATCH_SAMPLING",
    "PAPER_PADDLE_BATCH_SIZE",
    "PAPER_PADDLE_LEGACY_COUNT",
    "PAPER_PADDLE_QUALITY_COUNT",
    "PAPER_PADDLE_SAMPLER_KIND",
    "PaddleQualityWeights",
    "UNIFORM_BATCH_SAMPLING",
    "align_paddle_quality_weights",
    "load_paddle_quality_weights",
    "validate_paddle_quality_population",
]
