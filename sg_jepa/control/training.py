"""Paper-aligned training for low-dimensional diffusion policies."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from sg_jepa.baselines import load_dinov2_encoder
from sg_jepa.checkpoints import sha256_file, strict_load
from sg_jepa.config import load_experiment_config
from sg_jepa.data import open_trajectory_store, split_development_episodes
from sg_jepa.models import build_world_model
from sg_jepa.train_utils import atomic_json, restore_rng_state, rng_state, seed_everything

from .actions import (
    FIXED_PADDLE_GRAVITY,
    ActionNormalizer,
    FeatureNormalizer,
    raw_actions_to_controls,
)
from .checkpoint import (
    TopKCheckpointManager,
    export_ema_policy,
    load_policy_checkpoint,
    replace_alias,
    save_policy_checkpoint,
)
from .config import (
    FRANKA_ACTION_RECONSTRUCTION,
    PAPER_FRANKA_BATCH_SAMPLING,
    STORED_ACTION_RECONSTRUCTION,
    UNIFORM_BATCH_SAMPLING,
    PolicyConfig,
    TrainingConfig,
    load_policy_bundle_config,
)
from .diffusion import GaussianDiffusion1D
from .ema import EMAModel
from .inference import FrozenObservationEncoder
from .policy import build_policy_model
from .sampling import (
    PAPER_FRANKA_SAMPLER_KIND,
    PAPER_PADDLE_BATCH_SAMPLING,
    PAPER_PADDLE_SAMPLER_KIND,
    DeterministicFrankaStrikeBatchSampler,
    DeterministicPaperPaddleBatchSampler,
    align_paddle_quality_weights,
    load_paddle_quality_weights,
    validate_paddle_quality_population,
)

FRANKA_ACTION_SCALE_M = np.asarray([0.04, 0.04, 0.04], dtype=np.float64)
FRANKA_VALID_BLADE_CONTACT_INDEX = 1


@dataclass(frozen=True)
class _CachedEpisode:
    features: torch.Tensor
    controls: torch.Tensor
    gravity: float
    source_id: int
    raw_actions: torch.Tensor | None = None
    strike_frame: int | None = None
    reconstructed_rows: int = 0
    rows_above_legacy_cap: int = 0
    max_pose_reconstruction_error_m: float = 0.0


@dataclass(frozen=True)
class PolicyPartitions:
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    train_episode_ids: tuple[int, ...]
    val_episode_ids: tuple[int, ...]

    def gradient_indices(self, *, include_validation: bool) -> tuple[int, ...]:
        if include_validation:
            return (*self.train_indices, *self.val_indices)
        return self.train_indices

    def to_dict(self) -> dict[str, Any]:
        return {
            "successful_only": True,
            "train_episode_indices": list(self.train_episode_ids),
            "val_episode_indices": list(self.val_episode_ids),
            "train_success_episode_count": len(self.train_indices),
            "val_success_episode_count": len(self.val_indices),
        }


class _PolicyWindows(Dataset):
    """Causal histories and actions that advance the current frame onward."""

    def __init__(
        self,
        episodes: Sequence[_CachedEpisode],
        config: PolicyConfig,
        feature_normalizer: FeatureNormalizer,
        action_normalizer: ActionNormalizer,
    ) -> None:
        self.episodes = tuple(episodes)
        self.config = config
        self.feature_normalizer = feature_normalizer
        self.action_normalizer = action_normalizer
        self.windows = [
            (episode_index, frame)
            for episode_index, episode in enumerate(self.episodes)
            # The final stored action has no observed successor and is never a target.
            for frame in range(len(episode.controls) - config.action_horizon)
        ]
        if not self.windows:
            raise ValueError("dataset contains no complete policy transition horizons")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, frame = self.windows[index]
        episode = self.episodes[episode_index]
        first = max(0, frame - self.config.observation_horizon + 1)
        history = episode.features[first : frame + 1]
        if len(history) < self.config.observation_horizon:
            history = torch.cat(
                (
                    history[0:1].expand(self.config.observation_horizon - len(history), -1),
                    history,
                )
            )
        history = history.float()
        actions = episode.controls[frame : frame + self.config.action_horizon]
        return {
            "features": torch.as_tensor(
                self.feature_normalizer.normalize_embeddings(history), dtype=torch.float32
            ),
            "gravity": torch.tensor(
                [self.feature_normalizer.normalize_gravity(episode.gravity)], dtype=torch.float32
            ),
            "actions": torch.as_tensor(
                self.action_normalizer.normalize(actions, clip=True), dtype=torch.float32
            ),
        }


class DeterministicStepBatchSampler(Sampler[list[int]]):
    """Address full batches by optimizer step so prefetch cannot alter resume."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        seed: int,
        start_step: int,
        total_steps: int,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.total_steps = int(total_steps)
        self.batches_per_epoch = self.dataset_size // self.batch_size
        if self.batches_per_epoch <= 0:
            raise ValueError("policy dataset is smaller than one complete batch")
        if not 0 <= self.start_step <= self.total_steps:
            raise ValueError("start_step must lie in [0,total_steps]")

    def __len__(self) -> int:
        return self.total_steps - self.start_step

    def __iter__(self) -> Iterator[list[int]]:
        cached_epoch = -1
        permutation: torch.Tensor | None = None
        for global_step in range(self.start_step, self.total_steps):
            epoch, batch_index = divmod(global_step, self.batches_per_epoch)
            if epoch != cached_epoch:
                generator = torch.Generator().manual_seed(self.seed + 1_000_003 * epoch)
                permutation = torch.randperm(self.dataset_size, generator=generator)
                cached_epoch = epoch
            assert permutation is not None
            first = batch_index * self.batch_size
            yield permutation[first : first + self.batch_size].tolist()


def _lr_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _worker_seed(_worker_id: int) -> None:
    seed = int(torch.initial_seed() % 2**32)
    random.seed(seed)
    np.random.seed(seed)


def _resolve_config_artifact(config_path: str | Path, value: str) -> Path:
    """Resolve a portable auxiliary path from cwd or a config ancestor."""

    configured = Path(value).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    candidates = [Path.cwd() / configured]
    candidates.extend(parent / configured for parent in Path(config_path).resolve().parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"configured policy-training artifact does not exist: {value!r}")


def _encoder(
    config: PolicyConfig,
    *,
    world_model_config: str | Path | None,
    world_model_checkpoint: str | Path | None,
    dinov2_root: str | Path | None,
    device: torch.device,
) -> tuple[FrozenObservationEncoder, dict[str, Any]]:
    if config.feature_mode == "projected_cls":
        if world_model_config is None or world_model_checkpoint is None:
            raise ValueError("native policy training requires world-model config and checkpoint")
        config_path = Path(world_model_config).expanduser().resolve()
        checkpoint_path = Path(world_model_checkpoint).expanduser().resolve()
        experiment = load_experiment_config(config_path)
        if experiment.model.action_dim != config.action_dim + 1:
            raise ValueError("policy controls and native world-model raw actions differ")
        if experiment.model.embed_dim != config.embedding_dim:
            raise ValueError("policy and native world-model embedding dimensions differ")
        model = build_world_model(experiment.model)
        strict_load(model, checkpoint_path)
        model.to(device).eval().requires_grad_(False)
        digest = sha256_file(checkpoint_path)
        provenance = {
            "artifact": str(checkpoint_path),
            "weights_sha256": digest,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
        }
        return (
            FrozenObservationEncoder(
                model,
                config.feature_mode,
                config.embedding_dim,
                device,
                digest,
                experiment.model.image_size,
            ),
            provenance,
        )
    if dinov2_root is None:
        raise ValueError("DINO policy training requires --dinov2-root")
    model = load_dinov2_encoder(dinov2_root, device=device)
    return (
        FrozenObservationEncoder(
            model,
            config.feature_mode,
            config.embedding_dim,
            device,
            "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
            112,
        ),
        {
            "artifact": "DINOv2 ViT-S/14 upstream",
            "source_revision": "85a24602099d397264d5b30461ad7f3bfd726ca1",
            "weights_sha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
        },
    )


def _split_ids(payload: dict[str, Any], key: str) -> set[int]:
    aliases = {
        "train_episode_indices": "train_group_values",
        "val_episode_indices": "val_group_values",
    }
    values = payload.get(key, payload.get(aliases[key]))
    if not isinstance(values, list):
        raise ValueError(f"world-model split lacks {key}")
    return {int(value) for value in values}


def _cap_partitions(
    train: list[int],
    validation: list[int],
    maximum: int | None,
) -> tuple[list[int], list[int]]:
    if maximum is None or maximum >= len(train) + len(validation):
        return train, validation
    if maximum < 2:
        raise ValueError("max_episodes must leave at least one train and validation episode")
    val_count = round(maximum * len(validation) / (len(train) + len(validation)))
    val_count = min(max(val_count, 1), len(validation), maximum - 1)
    train_count = min(maximum - val_count, len(train))
    val_count = min(maximum - train_count, len(validation))
    if train_count < 1 or val_count < 1:
        raise ValueError("bounded policy cohort must contain train and validation successes")
    return train[:train_count], validation[:val_count]


def successful_policy_partitions(
    store: Any,
    *,
    split_path: str | Path | None,
    seed: int,
    max_episodes: int | None = None,
) -> PolicyPartitions:
    """Intersect the world-model split with successful development episodes."""

    if not store.has_success or store.success is None:
        raise ValueError(
            "policy training requires an episode-constant success column; regenerate or repack data"
        )
    development = np.flatnonzero(store.split_id == 0)
    source_to_index: dict[int, int] = {}
    for index in development:
        source_id = int(store.episodes[int(index)].source_id)
        if source_id in source_to_index:
            raise ValueError("development source_episode_index values must be unique")
        source_to_index[source_id] = int(index)
    if split_path is None:
        split = split_development_episodes(store, train_fraction=0.9, seed=seed)
        train_ids = set(split.train_episode_ids)
        val_ids = set(split.val_episode_ids)
    else:
        payload = json.loads(Path(split_path).expanduser().resolve().read_text())
        train_ids = _split_ids(payload, "train_episode_indices")
        val_ids = _split_ids(payload, "val_episode_indices")
        if train_ids & val_ids:
            raise ValueError("world-model train and validation episode IDs overlap")
    successful_ids = {
        int(store.episodes[int(index)].source_id)
        for index in development
        if bool(store.success[int(index)])
    }
    missing = successful_ids - (train_ids | val_ids)
    if missing:
        raise ValueError(
            f"successful development episodes are absent from split: {sorted(missing)[:8]}"
        )
    train = sorted(source_to_index[value] for value in successful_ids & train_ids)
    validation = sorted(source_to_index[value] for value in successful_ids & val_ids)
    if not train or not validation:
        raise ValueError("successful policy train and validation partitions must both be non-empty")
    train, validation = _cap_partitions(train, validation, max_episodes)
    return PolicyPartitions(
        tuple(train),
        tuple(validation),
        tuple(int(store.episodes[index].source_id) for index in train),
        tuple(int(store.episodes[index].source_id) for index in validation),
    )


def _cache_episodes(
    store: Any,
    indices: Sequence[int],
    encoder: FrozenObservationEncoder,
    *,
    action_dim: int,
    control_codec: str,
    action_reconstruction: str,
    require_strike_frame: bool,
    feature_cache_dtype: str,
    frame_batch_size: int,
) -> dict[int, _CachedEpisode]:
    cached: dict[int, _CachedEpisode] = {}
    for index in indices:
        episode = store.episode(int(index))
        if episode.success is not True:
            raise ValueError("policy cache received a non-success episode")
        chunks = [
            encoder.encode_rgb_frames(episode.pixels[first : first + frame_batch_size]).cpu()
            for first in range(0, len(episode.pixels), frame_batch_size)
        ]
        stored_actions = np.asarray(episode.action, dtype=np.float32)
        raw_actions = stored_actions
        reconstructed_rows = 0
        rows_above_legacy_cap = 0
        max_pose_reconstruction_error_m = 0.0
        if action_reconstruction == FRANKA_ACTION_RECONSTRUCTION:
            if episode.paddle_state is None:
                raise ValueError("full-range Franka action reconstruction requires paddle_state")
            raw_actions = stored_actions.copy()
            positions = np.asarray(episode.paddle_state[:, :3], dtype=np.float64)
            if positions.shape != (len(raw_actions), 3) or len(raw_actions) < 2:
                raise ValueError("Franka paddle_state must contain one xyz pose per frame")
            raw_actions[:-1, 1:4] = (
                (positions[1:] - positions[:-1]) / FRANKA_ACTION_SCALE_M
            ).astype(np.float32)
            raw_actions[-1] = raw_actions[-2]
            raw_actions[:, 0] = np.asarray(episode.gravity, dtype=np.float32)[0]
            replayed = (
                positions[:-1] + raw_actions[:-1, 1:4].astype(np.float64) * FRANKA_ACTION_SCALE_M
            )
            max_pose_reconstruction_error_m = float(np.max(np.abs(replayed - positions[1:])))
            if max_pose_reconstruction_error_m > 2.0e-7:
                raise RuntimeError(
                    "Franka pose-delta reconstruction error exceeds 2e-7 m: "
                    f"{max_pose_reconstruction_error_m:g}"
                )
            reconstructed_rows = int(
                np.count_nonzero(np.max(np.abs(raw_actions - stored_actions), axis=1) > 1.0e-7)
            )
            magnitude = np.max(np.abs(raw_actions[:-1, 1:4]), axis=1)
            rows_above_legacy_cap = int(np.count_nonzero(magnitude > 1.0 + 1.0e-7))
        elif action_reconstruction != STORED_ACTION_RECONSTRUCTION:
            raise ValueError(f"unsupported action reconstruction: {action_reconstruction!r}")

        strike_frame = None
        if require_strike_frame:
            if episode.task_event is None:
                raise ValueError("strike-balanced Franka sampling requires task_event")
            events = np.asarray(episode.task_event)
            if events.ndim != 2 or events.shape[1] <= FRANKA_VALID_BLADE_CONTACT_INDEX:
                raise ValueError("Franka task_event lacks valid_paddle_blade_contact")
            blade_contact = events[:, FRANKA_VALID_BLADE_CONTACT_INDEX].astype(bool)
            if not blade_contact.any():
                raise ValueError("successful Franka episode lacks valid paddle-blade contact")
            strike_frame = int(blade_contact.argmax())
        controls = np.asarray(
            raw_actions_to_controls(raw_actions, control_codec=control_codec),
            dtype=np.float32,
        )
        if controls.shape[-1] != action_dim:
            raise ValueError(
                f"dataset provides {controls.shape[-1]} controls but policy expects {action_dim}"
            )
        cached[int(index)] = _CachedEpisode(
            features=torch.cat(chunks).to(
                torch.float16 if feature_cache_dtype == "fp16" else torch.float32
            ),
            controls=torch.from_numpy(controls),
            gravity=float(episode.gravity[0]),
            source_id=int(episode.episode_id),
            raw_actions=torch.from_numpy(raw_actions),
            strike_frame=strike_frame,
            reconstructed_rows=reconstructed_rows,
            rows_above_legacy_cap=rows_above_legacy_cap,
            max_pose_reconstruction_error_m=max_pose_reconstruction_error_m,
        )
    return cached


def _action_reconstruction_contract(
    episodes: Sequence[_CachedEpisode],
    *,
    kind: str,
) -> dict[str, Any]:
    if kind == STORED_ACTION_RECONSTRUCTION:
        return {"contract": STORED_ACTION_RECONSTRUCTION}
    if kind != FRANKA_ACTION_RECONSTRUCTION:
        raise ValueError(f"unsupported action reconstruction: {kind!r}")
    return {
        "contract": FRANKA_ACTION_RECONSTRUCTION,
        "formula": "(paddle_position[t+1]-paddle_position[t])/ACTION_SCALE",
        "action_scale_m": FRANKA_ACTION_SCALE_M.tolist(),
        "source_column": "paddle_state[:3]",
        "orientation": "logged phi/theta retained",
        "final_action": "repeat step 62",
        "legacy_translation_clip_removed": True,
        "cached_episode_count": len(episodes),
        "changed_action_rows": int(sum(item.reconstructed_rows for item in episodes)),
        "rows_above_legacy_cap": int(sum(item.rows_above_legacy_cap for item in episodes)),
        "max_pose_reconstruction_error_m": float(
            max(item.max_pose_reconstruction_error_m for item in episodes)
        ),
    }


def _normalizers(
    episodes: Sequence[_CachedEpisode],
    *,
    gravity_normalization: str,
    control_codec: str,
) -> tuple[FeatureNormalizer, ActionNormalizer]:
    if not episodes:
        raise ValueError("normalizers require successful policy episodes")
    feature_dim = int(episodes[0].features.shape[-1])
    total = np.zeros(feature_dim, dtype=np.float64)
    total_sq = np.zeros(feature_dim, dtype=np.float64)
    count = 0
    for episode in episodes:
        values = np.asarray(episode.features, dtype=np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        count += len(values)
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1.0e-12)
    std = np.sqrt(variance)
    gravities = np.asarray([episode.gravity for episode in episodes], dtype=np.float32)
    if gravity_normalization == FIXED_PADDLE_GRAVITY:
        gravity_mean, gravity_std = 10.0, 10.0
    else:
        gravity_mean = float(np.asarray(gravities, dtype=np.float64).mean())
        gravity_std = float(max(np.asarray(gravities, dtype=np.float64).std(), 1.0e-6))
    feature_normalizer = FeatureNormalizer(
        tuple(float(value) for value in mean),
        tuple(float(value) for value in std),
        gravity_mean,
        gravity_std,
        1.0e-6,
        gravity_normalization,
    )
    normalizer_controls = []
    for episode in episodes:
        # Match the source cache path: action bounds use float64 trigonometry,
        # while policy targets below use float32 controls.
        if episode.raw_actions is None:
            normalizer_controls.append(episode.controls[:-1].double())
        else:
            normalizer_controls.append(
                raw_actions_to_controls(
                    episode.raw_actions[:-1].double(),
                    control_codec=control_codec,
                )
            )
    return (
        feature_normalizer,
        ActionNormalizer.fit(torch.cat(normalizer_controls)),
    )


def _loader_kwargs(config: TrainingConfig, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _worker_seed,
    }
    if config.num_workers > 0:
        result.update(
            persistent_workers=True,
            prefetch_factor=2,
            multiprocessing_context="spawn",
        )
    return result


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()
    }


@torch.inference_mode()
def validate_policy(
    model: nn.Module,
    ema: EMAModel,
    diffusion: GaussianDiffusion1D,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    max_batches: int | None,
    seed: int,
) -> float:
    """Evaluate EMA weights with fixed timesteps and diffusion noise."""

    eval_model = build_policy_model(model.config).to(device)
    ema.copy_to(eval_model)
    eval_model.eval().requires_grad_(False)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    losses: list[float] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _move(batch, device)
        actions = batch["actions"]
        timesteps = torch.randint(
            0,
            diffusion.timesteps,
            (actions.shape[0],),
            device=device,
            generator=generator,
        )
        noise = torch.randn(
            actions.shape,
            device=device,
            dtype=actions.dtype,
            generator=generator,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16" and device.type == "cuda",
        ):
            loss = diffusion.training_loss(
                eval_model,
                actions,
                batch["features"],
                batch["gravity"],
                noise=noise,
                timesteps=timesteps,
            )
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("policy validation loader produced no batches")
    return float(np.mean(losses))


def run_policy_training(
    config_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    world_model_config: str | Path | None = None,
    world_model_checkpoint: str | Path | None = None,
    dinov2_root: str | Path | None = None,
    split_path: str | Path | None = None,
    resume: str | Path | None = None,
    device: str = "cuda",
    max_steps: int | None = None,
    max_episodes: int | None = None,
    frame_batch_size: int = 16,
    batch_sampling_override: str | None = None,
) -> dict[str, Any]:
    """Train with successful demonstrations, validation, top-k, and exact resume."""

    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    policy_config, training, evaluation = load_policy_bundle_config(config_path)
    configured_batch_sampling = training.batch_sampling
    if batch_sampling_override is not None:
        if batch_sampling_override != UNIFORM_BATCH_SAMPLING:
            raise ValueError(
                "batch_sampling_override only supports the bounded smoke sampler: "
                f"{UNIFORM_BATCH_SAMPLING}"
            )
        training = replace(
            training,
            batch_sampling=batch_sampling_override,
            quality_weights=None,
            quality_weights_sha256=None,
        )
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(training.seed)
    encoder, world_model = _encoder(
        policy_config,
        world_model_config=world_model_config,
        world_model_checkpoint=world_model_checkpoint,
        dinov2_root=dinov2_root,
        device=selected_device,
    )
    store = open_trajectory_store(dataset_path)
    if store.action_dim != policy_config.action_dim + 1:
        raise ValueError("dataset raw-action dimension differs from the policy")
    partitions = successful_policy_partitions(
        store,
        split_path=split_path,
        seed=training.seed,
        max_episodes=max_episodes,
    )
    gradient_indices = partitions.gradient_indices(
        include_validation=training.include_validation_in_gradient
    )
    quality_artifact = None
    if training.batch_sampling == PAPER_PADDLE_BATCH_SAMPLING:
        assert training.quality_weights is not None
        assert training.quality_weights_sha256 is not None
        quality_path = _resolve_config_artifact(config_path, training.quality_weights)
        quality_artifact = load_paddle_quality_weights(
            quality_path,
            expected_sha256=training.quality_weights_sha256,
        )
        validate_paddle_quality_population(
            (store.episodes[index].source_id for index in gradient_indices),
            quality_artifact,
            require_complete_population=(
                max_episodes is None and int((store.split_id == 0).sum()) == 8_000
            ),
        )
    unique_indices = tuple(dict.fromkeys((*gradient_indices, *partitions.val_indices)))
    cache = _cache_episodes(
        store,
        unique_indices,
        encoder,
        action_dim=policy_config.action_dim,
        control_codec=policy_config.control_codec,
        action_reconstruction=training.action_reconstruction,
        require_strike_frame=training.batch_sampling == PAPER_FRANKA_BATCH_SAMPLING,
        feature_cache_dtype=training.feature_cache_dtype,
        frame_batch_size=frame_batch_size,
    )
    gradient_episodes = [cache[index] for index in gradient_indices]
    val_episodes = [cache[index] for index in partitions.val_indices]
    feature_normalizer, action_normalizer = _normalizers(
        gradient_episodes,
        gravity_normalization=training.gravity_normalization,
        control_codec=policy_config.control_codec,
    )
    train_dataset = _PolicyWindows(
        gradient_episodes,
        policy_config,
        feature_normalizer,
        action_normalizer,
    )
    val_dataset = _PolicyWindows(
        val_episodes,
        policy_config,
        feature_normalizer,
        action_normalizer,
    )
    paper_quality_weights: np.ndarray | None = None
    batch_sampling_contract: dict[str, Any] = {
        "kind": training.batch_sampling,
        "seed": training.seed,
    }
    if training.batch_sampling != configured_batch_sampling:
        batch_sampling_contract.update(
            {
                "configured_kind": configured_batch_sampling,
                "override_reason": "bounded_smoke_cohort",
            }
        )
    if training.batch_sampling == PAPER_PADDLE_BATCH_SAMPLING:
        assert quality_artifact is not None
        assert training.quality_weights is not None
        paper_quality_weights = align_paddle_quality_weights(
            train_dataset,
            quality_artifact,
        )
        batch_sampling_contract.update(
            {
                **quality_artifact.provenance(training.quality_weights),
                "kind": PAPER_PADDLE_BATCH_SAMPLING,
                "algorithm": PAPER_PADDLE_SAMPLER_KIND,
                "legacy_replay_count": 205,
                "soft_quality_count": 51,
                "batch_size": 256,
                "aligned_window_count": len(train_dataset),
            }
        )
    elif training.batch_sampling == PAPER_FRANKA_BATCH_SAMPLING:
        preview_sampler = DeterministicFrankaStrikeBatchSampler(
            train_dataset,
            training.batch_size,
            seed=training.seed,
            start_step=0,
            total_steps=0,
        )
        batch_sampling_contract.update(preview_sampler.contract())
        batch_sampling_contract["configured_kind"] = training.batch_sampling
        batch_sampling_contract["algorithm"] = PAPER_FRANKA_SAMPLER_KIND
    fit_population = (
        "all_successful_development_episodes"
        if training.include_validation_in_gradient
        else "successful_world_model_training_partition_only"
    )
    feature_payload = feature_normalizer.to_dict()
    action_payload = action_normalizer.to_dict()
    feature_payload["fit_population"] = fit_population
    action_payload["fit_population"] = fit_population
    action_payload["control_codec"] = policy_config.control_codec
    if policy_config.control_codec == "paddle_tilt_vector":
        action_payload["control_schema"] = ["dx", "dy", "dz", "tilt_x", "tilt_y"]
    split_source = (
        {"kind": "explicit", "path": str(Path(split_path).expanduser().resolve())}
        if split_path is not None
        else {"kind": "deterministic_fallback", "seed": training.seed, "train_fraction": 0.9}
    )
    if split_path is not None:
        split_source["sha256"] = sha256_file(split_path)
    data_contract = {
        "dataset": str(Path(dataset_path).expanduser().resolve()),
        "successful_only": True,
        "split": split_source,
        "partitions": partitions.to_dict(),
        "feature_mode": policy_config.feature_mode,
        "feature_cache_dtype": training.feature_cache_dtype,
        "control_codec": policy_config.control_codec,
        "action_reconstruction": _action_reconstruction_contract(
            gradient_episodes,
            kind=training.action_reconstruction,
        ),
        "max_tilt_theta": policy_config.max_tilt_theta,
        "feature_normalizer": feature_payload,
        "action_normalizer": action_payload,
        "world_model": world_model,
        "action_alignment": "action[t] advances frame[t] to frame[t+1]",
        "unused_action_index": int(min(store.lengths[index] for index in unique_indices) - 1),
        "optimization_population": {
            "include_world_model_validation_partition": (training.include_validation_in_gradient),
            "gradient_episode_count": len(gradient_indices),
            "validation_episode_count": len(partitions.val_indices),
            "validation_loss_role": (
                "diagnostic_only; validation episodes also receive gradients"
                if training.include_validation_in_gradient
                else "held_out_checkpoint_selection"
            ),
        },
        "batch_sampling": batch_sampling_contract,
        "target_alignment": {
            "observation": "frames ending at t, left-pad frame zero",
            "target": f"action[t:t+{policy_config.action_horizon}] advances frame t onward",
            "unused_final_action_supervised": False,
        },
    }

    output = Path(output_dir).expanduser().resolve()
    if output.exists() and resume is None:
        raise FileExistsError(f"refusing to overwrite policy output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    atomic_json(data_contract, output / "data_contract.json")

    model = build_policy_model(policy_config).to(selected_device)
    diffusion = GaussianDiffusion1D(
        policy_config.diffusion_steps,
        clip_sample=policy_config.clip_sample,
    ).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        betas=training.betas,
        weight_decay=training.weight_decay,
    )
    ema = EMAModel(
        model,
        inv_gamma=training.ema_inv_gamma,
        power=training.ema_power,
        max_decay=training.ema_max_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_lambda(
            step,
            warmup_steps=training.warmup_steps,
            total_steps=training.steps,
        ),
    )
    start_step = 0
    recent_losses: list[float] = []
    validation_history: list[dict[str, float | int]] = []
    last_validation_loss: float | None = None
    restored_rng: dict[str, Any] | None = None
    if resume is not None:
        checkpoint = load_policy_checkpoint(resume)
        if checkpoint["policy_config"] != policy_config.to_dict():
            raise ValueError("resume policy configuration differs")
        if checkpoint["training_config"] != training.to_dict():
            raise ValueError("resume training configuration differs")
        if checkpoint["data_contract"] != data_contract:
            raise ValueError("resume policy data contract differs")
        if checkpoint["world_model"] != world_model:
            raise ValueError("resume world-model provenance differs")
        model.load_state_dict(checkpoint["policy_state_dict"], strict=True)
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = int(checkpoint["step"])
        state = checkpoint.get("training_state", {})
        recent_losses = [float(value) for value in state.get("recent_losses", [])]
        validation_history = [dict(value) for value in state.get("validation_history", [])]
        stored_validation = state.get("last_validation_loss")
        last_validation_loss = None if stored_validation is None else float(stored_validation)
        restored_rng = state.get("rng")
    if start_step > training.steps:
        raise ValueError("resume step exceeds configured policy training")
    stop_step = (
        training.steps if max_steps is None else min(training.steps, start_step + int(max_steps))
    )
    if training.batch_sampling == PAPER_FRANKA_BATCH_SAMPLING:
        batch_sampler = DeterministicFrankaStrikeBatchSampler(
            train_dataset,
            training.batch_size,
            seed=training.seed,
            start_step=start_step,
            total_steps=stop_step,
        )
    elif paper_quality_weights is None:
        batch_sampler: Sampler[list[int]] = DeterministicStepBatchSampler(
            len(train_dataset),
            training.batch_size,
            seed=training.seed,
            start_step=start_step,
            total_steps=stop_step,
        )
    else:
        batch_sampler = DeterministicPaperPaddleBatchSampler(
            len(train_dataset),
            quality_weights=paper_quality_weights,
            seed=training.seed,
            start_step=start_step,
            total_steps=stop_step,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        generator=torch.Generator().manual_seed(training.seed + 17),
        **_loader_kwargs(training, selected_device),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(training.seed + 29),
        **_loader_kwargs(training, selected_device),
    )
    if restored_rng is not None:
        restore_rng_state(restored_rng)

    manager = TopKCheckpointManager(checkpoint_dir, k=training.top_k)
    model.train()
    completed_step = start_step
    last_checkpoint = Path(resume).expanduser().resolve() if resume is not None else None
    for zero_based_step, batch in enumerate(train_loader, start=start_step):
        step = zero_based_step + 1
        batch = _move(batch, selected_device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=selected_device.type,
            dtype=torch.bfloat16,
            enabled=training.precision == "bf16" and selected_device.type == "cuda",
        ):
            loss = diffusion.training_loss(
                model,
                batch["actions"],
                batch["features"],
                batch["gravity"],
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite policy loss at step {step}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        decay = ema.update(model)
        completed_step = step
        recent_losses.append(float(loss.detach().cpu()))
        recent_losses = recent_losses[-100:]

        validate = step % training.validate_every == 0 or step == training.steps
        checkpoint_now = (
            step % training.checkpoint_every == 0 or step == training.steps or step == stop_step
        )
        validation_loss: float | None = None
        if validate:
            validation_loss = validate_policy(
                model,
                ema,
                diffusion,
                val_loader,
                device=selected_device,
                precision=training.precision,
                max_batches=training.max_val_batches,
                seed=training.seed + step,
            )
            last_validation_loss = validation_loss
            validation_history.append({"step": step, "validation_loss": validation_loss})
            atomic_json({"history": validation_history}, output / "validation_history.json")
        if checkpoint_now or validate:
            checkpoint_path = checkpoint_dir / f"policy_step_{step:06d}.pt"
            save_policy_checkpoint(
                checkpoint_path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                policy_config=policy_config.to_dict(),
                training_config=training.to_dict(),
                data_contract=data_contract,
                world_model=world_model,
                evaluation=evaluation,
                metrics={
                    "train_loss": float(loss.detach().cpu()),
                    "validation_loss": validation_loss,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "ema_decay": float(decay),
                },
                training_state={
                    "rng": rng_state(),
                    "recent_losses": recent_losses,
                    "validation_history": validation_history,
                    "last_validation_loss": last_validation_loss,
                },
            )
            last_checkpoint = checkpoint_path
            replace_alias(checkpoint_path, checkpoint_dir / "last_policy.pt")
            if validation_loss is not None:
                manager.consider(
                    checkpoint_path,
                    validation_loss=validation_loss,
                    step=step,
                )
                best_source = checkpoint_dir / manager.entries()[0]["file"]
                best_inference = checkpoint_dir / "best_policy_ema_inference.pt"
                export_ema_policy(best_source, best_inference)
                replace_alias(best_inference, checkpoint_dir / "best_policy.pt")

    if last_checkpoint is None:
        raise RuntimeError("policy training produced no resumable checkpoint")
    last_alias = checkpoint_dir / "last_policy.pt"
    if last_alias.is_file():
        last_checkpoint = last_alias
    final_inference = checkpoint_dir / "policy_ema_inference.pt"
    export_ema_policy(last_checkpoint, final_inference)
    reloaded = build_policy_model(policy_config)
    reloaded.load_state_dict(load_policy_checkpoint(last_checkpoint)["ema"]["shadow"], strict=True)
    if world_model_checkpoint is not None:
        if sha256_file(world_model_checkpoint) != world_model["weights_sha256"]:
            raise RuntimeError("world-model checkpoint changed during policy training")
    report = {
        "schema_version": 2,
        "kind": "policy_training",
        "device": str(selected_device),
        "steps_before": start_step,
        "steps_after": completed_step,
        "steps_this_run": completed_step - start_step,
        "configured_steps": training.steps,
        "interrupted_for_smoke": completed_step < training.steps,
        "successful_only": True,
        "gradient_episodes": len(gradient_indices),
        "validation_episodes": len(partitions.val_indices),
        "train_windows": len(train_dataset),
        "validation_windows": len(val_dataset),
        "last_training_checkpoint": str(checkpoint_dir / "last_policy.pt"),
        "policy_checkpoint": str(final_inference),
        "best_policy_checkpoint": (
            str(checkpoint_dir / "best_policy.pt")
            if (checkpoint_dir / "best_policy.pt").is_file()
            else None
        ),
        "last_validation_loss": last_validation_loss,
        "validation_history": validation_history,
        "recent_training_loss": float(np.mean(recent_losses)) if recent_losses else None,
        "schedule": "linear_warmup_cosine_decay",
        "batch_sampling": batch_sampling_contract,
        "strict_reload": "pass",
        "status": "pass",
    }
    atomic_json(report, output / "training.json")
    return report


__all__ = [
    "FRANKA_ACTION_SCALE_M",
    "DeterministicStepBatchSampler",
    "PAPER_FRANKA_SAMPLER_KIND",
    "PolicyPartitions",
    "run_policy_training",
    "successful_policy_partitions",
    "validate_policy",
]
