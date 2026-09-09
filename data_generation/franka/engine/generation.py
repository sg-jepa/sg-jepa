#!/usr/bin/env python3
"""Episode assignment and simulation helpers for Franka paddle-to-basket."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
from pathlib import Path

import basket_task as task
import mujoco
import numpy as np
import pyarrow as pa

TASK_NAME = "franka_paddle_hit_ball_to_basket"
DATASET_VERSION = "v1"
TOTAL_EPISODES = 13000
TRAIN_EPISODES = 8000
TEST_EPISODES_PER_GRAVITY = 200
CHUNK_EPISODES = 8
BASE_SEED = 20260807
DEFAULT_OUTCOME_PATTERN = (
    *("success" for _ in range(14)),
    "undershoot",
    "overshoot",
    "left_right_miss",
    "rim_hit",
    "invalid_robot_contact",
    "no_paddle_contact",
)
TEST_GRAVITIES = [
    2.0,
    2.7,
    3.4,
    4.1,
    4.8,
    5.5,
    6.2,
    6.9,
    7.6,
    8.3,
    9.0,
    9.7,
    10.4,
    11.1,
    11.8,
    12.5,
    13.2,
    13.9,
    14.6,
    15.3,
    16.0,
    8.87,
    3.721,
    1.625,
    0.62,
]
TEST_EPISODES = len(TEST_GRAVITIES) * TEST_EPISODES_PER_GRAVITY


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generation_fingerprint() -> tuple[dict, str]:
    """Fingerprint every input that can change physics or rendered pixels."""
    payload = {
        "dataset_version": DATASET_VERSION,
        "generator_sha256": _sha256(Path(__file__)),
        "task_core_sha256": _sha256(Path(task.__file__)),
        "base_scene_sha256": _sha256(Path(task.base.__file__)),
        "dataset_config_sha256": _sha256(task.ROOT / "dataset.json"),
        "franka_asset_tree_sha256": task.base._tree_hash(task.PANDA_DIR),
        "mujoco_version": mujoco.__version__,
        "render_backend": os.environ.get("MUJOCO_GL", ""),
        "renderer_fingerprint": os.environ.get("RENDERER_FINGERPRINT", ""),
        "base_seed": BASE_SEED,
        "fixed_basket": task.FIXED_BASKET.as_array().tolist(),
        "format": {
            "fps": task.FPS,
            "frames": task.FRAMES,
            "width": task.WIDTH,
            "height": task.HEIGHT,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return payload, hashlib.sha256(encoded).hexdigest()


def parse_outcome_pattern(value: str | None) -> tuple[str, ...]:
    if value is None:
        return tuple(DEFAULT_OUTCOME_PATTERN)
    # ``+`` is an Slurm --export-safe separator; commas remain convenient for
    # direct/local CLI use.
    pattern = tuple(item.strip() for item in value.replace("+", ",").split(",") if item.strip())
    allowed = {
        "success",
        "undershoot",
        "overshoot",
        "left_right_miss",
        "rim_hit",
        "invalid_robot_contact",
        "no_paddle_contact",
    }
    if not pattern or any(item not in allowed for item in pattern):
        raise ValueError(f"outcome pattern must contain only {sorted(allowed)}")
    return pattern


def schema(outcome_pattern: tuple[str, ...] = DEFAULT_OUTCOME_PATTERN) -> pa.Schema:
    metadata = {
        "task_name": TASK_NAME,
        "version": DATASET_VERSION,
        "fps": str(task.FPS),
        "frames_per_episode": str(task.FRAMES),
        "pixels_shape": json.dumps([task.WIDTH, task.HEIGHT, 3]),
        "pixels_dtype": "uint8",
        "pixels_encoding": "jpeg_rgb",
        "split_to_id": json.dumps({"train": 0, "test": 1}, sort_keys=True),
        "shape_id_map": json.dumps({"ball": 0}, sort_keys=True),
        "policy_id_map": json.dumps({"gravity_aware_analytic_strike": 0}, sort_keys=True),
        "state_schema": json.dumps(task.STATE_SCHEMA),
        "action_schema": json.dumps(task.ACTION_SCHEMA),
        "paddle_state_schema": json.dumps(task.PADDLE_STATE_SCHEMA),
        "contact_schema": json.dumps(task.CONTACT_SCHEMA),
        "arm_state_schema": json.dumps(task.ARM_STATE_SCHEMA),
        "task_event_schema": json.dumps(task.TASK_EVENT_SCHEMA),
        "basket_state_schema": json.dumps(task.BASKET_STATE_SCHEMA),
        "strike_state_schema": json.dumps(task.STRIKE_STATE_SCHEMA),
        "outcome_pattern": json.dumps(list(outcome_pattern)),
        "train_episodes": str(TRAIN_EPISODES),
        "test_episodes": str(TEST_EPISODES),
        "test_episodes_per_gravity": str(TEST_EPISODES_PER_GRAVITY),
        "phys_schema": json.dumps(
            [
                "g",
                "mass",
                "size_x",
                "size_y",
                "size_z",
                "shape_id",
                "ground_half_x",
                "ground_half_y",
                "wall_height",
                "wall_thickness",
            ]
        ),
        "task_param_meanings": json.dumps(
            ["basket_x", "basket_y", "basket_rim_z", "basket_opening_radius"]
        ),
    }
    encoded = {key.encode(): value.encode() for key, value in metadata.items()}
    return pa.schema(
        [
            pa.field("episode_idx", pa.int32(), nullable=False),
            pa.field("step_idx", pa.int32(), nullable=False),
            pa.field("split_id", pa.int8(), nullable=False),
            pa.field("pixels", pa.binary(), nullable=False),
            pa.field("state", pa.list_(pa.float32(), len(task.STATE_SCHEMA)), nullable=False),
            pa.field("action", pa.list_(pa.float32(), len(task.ACTION_SCHEMA)), nullable=False),
            pa.field("reward", pa.float32(), nullable=False),
            pa.field("phys", pa.list_(pa.float32(), 10), nullable=False),
            pa.field("gravity", pa.float32(), nullable=False),
            pa.field("source_episode_index", pa.int32(), nullable=False),
            pa.field(
                "paddle_state",
                pa.list_(pa.float32(), len(task.PADDLE_STATE_SCHEMA)),
                nullable=False,
            ),
            pa.field("contact", pa.list_(pa.uint8(), len(task.CONTACT_SCHEMA)), nullable=False),
            pa.field("success", pa.bool_(), nullable=False),
            pa.field("policy_type_id", pa.int8(), nullable=False),
            pa.field("episode_metadata", pa.string(), nullable=False),
            pa.field(
                "arm_state", pa.list_(pa.float32(), len(task.ARM_STATE_SCHEMA)), nullable=False
            ),
            pa.field(
                "basket_state",
                pa.list_(pa.float32(), len(task.BASKET_STATE_SCHEMA)),
                nullable=False,
            ),
            pa.field(
                "task_event",
                pa.list_(pa.uint8(), len(task.TASK_EVENT_SCHEMA)),
                nullable=False,
            ),
            pa.field(
                "strike_state",
                pa.list_(pa.float32(), len(task.STRIKE_STATE_SCHEMA)),
                nullable=False,
            ),
            pa.field("failure_reason", pa.string(), nullable=False),
        ],
        metadata=encoded,
    )


SCHEMA = schema()


@dataclasses.dataclass(frozen=True)
class Assignment:
    episode_index: int
    split_name: str
    split_id: int
    split_rank: int
    gravity: float
    planned_success: bool
    planned_detail: str


def assignment_for_episode(
    episode_index: int,
    outcome_pattern: tuple[str, ...] = DEFAULT_OUTCOME_PATTERN,
) -> Assignment:
    if not 0 <= episode_index < TOTAL_EPISODES:
        raise ValueError(f"episode_index must be in [0, {TOTAL_EPISODES})")
    if episode_index < TRAIN_EPISODES:
        split_name = "train"
        split_id = 0
        split_rank = episode_index
        gravity_rng = np.random.default_rng(BASE_SEED + 104729 * (episode_index + 1))
        gravity = max(float(gravity_rng.normal(9.8, 2.0)), 0.62)
    else:
        split_name = "test"
        split_id = 1
        split_rank = episode_index - TRAIN_EPISODES
        gravity_index = split_rank // TEST_EPISODES_PER_GRAVITY
        gravity = TEST_GRAVITIES[gravity_index]
    planned_detail = outcome_pattern[split_rank % len(outcome_pattern)]
    return Assignment(
        episode_index=episode_index,
        split_name=split_name,
        split_id=split_id,
        split_rank=split_rank,
        gravity=gravity,
        planned_success=planned_detail == "success",
        planned_detail=planned_detail,
    )


class ModelCache:
    def __init__(self) -> None:
        self.models: dict[tuple[float, float, float], mujoco.MjModel] = {}
        self.renderers: dict[tuple[float, float, float], mujoco.Renderer] = {}

    @staticmethod
    def key(basket: task.BasketSpec) -> tuple[float, float, float]:
        return (basket.opening_radius, basket.depth, basket.rim_tube_radius)

    def model(self, basket: task.BasketSpec, gravity: float) -> mujoco.MjModel:
        key = self.key(basket)
        if key not in self.models:
            self.models[key] = task.build_model(
                np.asarray([-0.72, 0.0, task.BALL_RADIUS + 0.003]),
                np.zeros(3),
                gravity,
                basket,
            )
        return self.models[key]

    def renderer(self, basket: task.BasketSpec, gravity: float) -> mujoco.Renderer:
        key = self.key(basket)
        self.model(basket, gravity)
        if key not in self.renderers:
            self.renderers[key] = mujoco.Renderer(
                self.models[key], height=task.HEIGHT, width=task.WIDTH
            )
        return self.renderers[key]

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()


def _sample_request(assignment: Assignment, attempt: int) -> task.EpisodeRequest:
    rng = random.Random(BASE_SEED + 1000003 * (assignment.episode_index + 1) + 7919 * attempt)
    gravity = assignment.gravity
    hit_frame = rng.randint(24, 28)
    initial_vz = rng.uniform(3.25, 4.35) * math.sqrt(max(gravity, 0.62) / 9.8)
    # Consume the legacy v0 draws so every non-basket random variable remains
    # seed-aligned with v0.  The v1 scene itself always uses one fixed basket.
    rng.uniform(-0.66, -0.52)
    rng.uniform(-0.12, 0.12)
    rng.uniform(0.72, 0.88)
    rng.choice((0.23, 0.245, 0.26))
    rng.choice((0.22, 0.25, 0.28))
    basket = task.FIXED_BASKET
    flight_time = float(
        np.clip(
            rng.uniform(0.56, 0.66) * math.sqrt(9.8 / max(gravity, 0.62)),
            0.44,
            1.62,
        )
    )
    target_offset = (0.0, 0.0, 0.0)
    if assignment.planned_detail == "left_right_miss":
        target_offset = (0.0, rng.choice((-0.32, 0.32)), 0.0)
    request = task.EpisodeRequest(
        name=f"production_{assignment.planned_detail}",
        label=assignment.planned_detail.replace("_", " "),
        gravity=gravity,
        planned_success=assignment.planned_success,
        planned_detail=assignment.planned_detail,
        hit_frame=hit_frame,
        initial_vz=initial_vz,
        ball_y=rng.uniform(-0.06, 0.06),
        initial_z=task.BALL_RADIUS + rng.uniform(0.003, 0.045),
        basket=basket,
        flight_time=flight_time,
        target_offset=target_offset,
        impact_offset=(rng.uniform(-0.025, 0.025), rng.uniform(-0.018, 0.018)),
        roll=math.radians(rng.uniform(-12.0, 12.0)),
    )
    # Outcome-specific search replaces these offsets where required; successes
    # retain small off-center impacts for useful contact diversity.
    if not assignment.planned_success:
        request = dataclasses.replace(request, impact_offset=(0.0, 0.0), roll=0.0)
    return request


def prepare_episode(
    cache: ModelCache,
    assignment: Assignment,
) -> tuple[mujoco.MjModel, task.PreparedEpisode, dict]:
    failures: list[str] = []
    for attempt in range(18):
        request = _sample_request(assignment, attempt)
        model = cache.model(request.basket, request.gravity)
        try:
            prepared, metrics = task.prepare_matching_episode(model, request, require_ik=True)
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        if task.outcome_matches(metrics, prepared.request):
            return model, prepared, metrics
    tail = failures[-1] if failures else "no candidates"
    raise RuntimeError(
        f"unable to realize episode {assignment.episode_index} "
        f"g={assignment.gravity:.6g} plan={assignment.planned_detail}: {tail}"
    )


def episode_metadata(assignment: Assignment, prepared: task.PreparedEpisode, metrics: dict) -> str:
    payload = {
        "scenario": TASK_NAME,
        "description": (
            "An incoming bouncing red ball is struck by a Franka-mounted blue paddle "
            "and follows a gravity-dependent projectile path toward an elevated basket."
        ),
        "fps": task.FPS,
        "num_frames": task.FRAMES,
        "gravity": [0.0, 0.0, -float(assignment.gravity)],
        "episode_index": assignment.episode_index,
        "split_name": assignment.split_name,
        "split_id": assignment.split_id,
        "planned_outcome": "success" if assignment.planned_success else "failure",
        "planned_outcome_detail": assignment.planned_detail,
        "success": metrics["success"],
        "failure_reason": metrics["failure_reason"],
        "events": {
            "paddle_contact": metrics["paddle_contact"],
            "valid_paddle_blade_contact": metrics["valid_paddle_blade_contact"],
            "basket_entry": metrics["basket_entry"],
            "ball_retained_after_entry": metrics["ball_retained_after_entry"],
            "basket_escape_after_entry": metrics["basket_escape_after_entry"],
            "rim_hit": metrics["rim_hit"],
        },
        "state_schema": task.STATE_SCHEMA,
        "action_schema": task.ACTION_SCHEMA,
        "paddle_state_schema": task.PADDLE_STATE_SCHEMA,
        "contact_schema": task.CONTACT_SCHEMA,
        "arm_state_schema": task.ARM_STATE_SCHEMA,
        "basket_state_schema": task.BASKET_STATE_SCHEMA,
        "task_event_schema": task.TASK_EVENT_SCHEMA,
        "strike_state_schema": task.STRIKE_STATE_SCHEMA,
        "basket": metrics["basket"],
        "strike": {
            key: metrics[key]
            for key in (
                "predicted_interception_point",
                "actual_paddle_ball_contact_point",
                "incoming_ball_velocity",
                "outgoing_ball_velocity",
                "desired_outgoing_ball_velocity",
                "desired_basket_target_point",
                "paddle_blade_normal_at_impact",
                "paddle_linear_velocity_at_impact",
                "paddle_angular_velocity_at_impact",
                "desired_flight_time",
                "desired_projectile_apex_z",
                "impact_to_entry_or_closest_time",
                "minimum_distance_to_basket_opening",
                "minimum_ball_z_after_entry",
                "opening_cross_position",
                "minimum_paddle_floor_clearance_m",
                "maximum_paddle_floor_lift_m",
            )
        },
        "controller": {
            "type": "gravity_aware_ballistic_target_with_actual_mujoco_impact",
            "ballistic_relation": "v_out=(p_target-p_hit-0.5*g_vec*T^2)/T",
            "planned_hit_frame": prepared.request.hit_frame,
            "swing_gain": prepared.request.swing_gain,
            "target_offset": prepared.request.target_offset,
            "impact_offset": prepared.request.impact_offset,
            "roll": prepared.request.roll,
            "normal_correction": prepared.request.normal_correction,
            "swing_lead_frames": prepared.request.swing_lead_frames,
            "follow_through_frames": prepared.request.follow_through_frames,
            "minimum_swing_x": prepared.request.minimum_swing_x,
        },
        "physics": {
            "gravity": assignment.gravity,
            "ball_radius": task.BALL_RADIUS,
            "ball_mass": task.BALL_MASS,
            "ball_restitution": 0.92,
            "paddle_restitution": 0.92,
        },
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def append_episode_columns(
    columns: dict[str, list],
    assignment: Assignment,
    prepared: task.PreparedEpisode,
    metrics: dict,
    arrays: dict[str, np.ndarray],
    pixels: list[bytes],
) -> None:
    metadata = episode_metadata(assignment, prepared, metrics)
    phys = np.asarray(
        [
            assignment.gravity,
            task.BALL_MASS,
            task.BALL_RADIUS,
            task.BALL_RADIUS,
            task.BALL_RADIUS,
            0.0,
            4.0,
            4.0,
            1.68,
            0.025,
        ],
        dtype=np.float32,
    ).tolist()
    for frame in range(task.FRAMES):
        columns["episode_idx"].append(assignment.episode_index)
        columns["step_idx"].append(frame)
        columns["split_id"].append(assignment.split_id)
        columns["pixels"].append(pixels[frame])
        columns["state"].append(arrays["state"][frame].tolist())
        columns["action"].append(arrays["action"][frame].tolist())
        columns["reward"].append(0.0)
        columns["phys"].append(phys)
        columns["gravity"].append(float(assignment.gravity))
        columns["source_episode_index"].append(assignment.episode_index)
        columns["paddle_state"].append(arrays["paddle_state"][frame].tolist())
        columns["contact"].append(arrays["contact"][frame].tolist())
        columns["success"].append(bool(metrics["success"]))
        columns["policy_type_id"].append(0)
        columns["episode_metadata"].append(metadata)
        columns["arm_state"].append(arrays["arm_state"][frame].tolist())
        columns["basket_state"].append(arrays["basket_state"][frame].tolist())
        columns["task_event"].append(arrays["task_event"][frame].tolist())
        columns["strike_state"].append(arrays["strike_state"][frame].tolist())
        columns["failure_reason"].append(metrics["failure_reason"])
