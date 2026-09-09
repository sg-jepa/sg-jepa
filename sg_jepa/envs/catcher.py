"""Closed-loop MuJoCo adapter for the 16 Hz arm-catcher task."""

from __future__ import annotations

import importlib
import math
import os
import random
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_generation.common.assets import (
    provision_menagerie_assets,
    verify_menagerie_component,
)
from data_generation.mujoco.generator import _engine_root

TEST_GRAVITIES = ",".join(str(-1.0 + 0.5 * index) for index in range(23))
PRODUCTION_ENV = {
    "MUJOCO_GL": "egl",
    "DEMO_MUJOCO_TASKS": "arm_catcher_ball",
    "DEMO_MUJOCO_OUTPUT_VERSIONS": "v0",
    "DEMO_MUJOCO_EPISODES_PER_TASK": "9600",
    "DEMO_MUJOCO_CANONICAL_EPISODES_PER_TASK": "9600",
    "DEMO_MUJOCO_SELECTED_SPLIT": "all",
    "DEMO_MUJOCO_BASE_SEED": "20260525",
    "DEMO_MUJOCO_DYNAMIC_SHAPE_MODE": "ball_only",
    "DEMO_MUJOCO_RENDER_ENV": "natural",
    "DEMO_MUJOCO_IMAGE_WIDTH": "256",
    "DEMO_MUJOCO_IMAGE_HEIGHT": "256",
    "DEMO_MUJOCO_FRAMES_PER_EPISODE": "64",
    "DEMO_MUJOCO_FPS": "16",
    "DEMO_MUJOCO_STEPS_PER_FRAME": "20",
    "DEMO_MUJOCO_STEP_RATE": "320",
    "DEMO_MUJOCO_TRAIN_GAUSSIAN_MEAN": "4.0",
    "DEMO_MUJOCO_TRAIN_GAUSSIAN_STD": "0.5",
    "DEMO_MUJOCO_TRAIN_G_MIN": "0.0",
    "DEMO_MUJOCO_TEST_GRAVITIES": TEST_GRAVITIES,
    "DEMO_MUJOCO_GRAVITY_BATCH_EPISODES": "30",
    "DEMO_MUJOCO_TRAIN_EPISODES_PER_BATCH": "25",
    "DEMO_MUJOCO_TEST_EPISODES_PER_BATCH": "5",
    "DEMO_MUJOCO_ARM_CATCHER_CAPTURE_RETRY_LIMIT": "96",
    "DEMO_MUJOCO_RENDER_SUPERSAMPLE": "2",
    "DEMO_MUJOCO_RENDER_SOFTEN": "0.14",
    "DEMO_MUJOCO_OPEN_BOX_SHADOW_LIFT": "0.24",
    "DEMO_MUJOCO_ARM_CATCHER_WALL_EMISSION": "0.34",
    "DEMO_MUJOCO_SHARD_EPISODES": "8",
    "DEMO_MUJOCO_RENDER_WORKERS": "1",
}


def _metadata_gravity(metadata: Mapping[str, Any]) -> float:
    """Return the signed scalar gravity used by the policy contract."""

    episode_phys = metadata.get("episode_phys")
    if isinstance(episode_phys, list) and episode_phys:
        return float(episode_phys[0])
    return -float(metadata["gravity"][2])


def configure_production_environment(*, work_dir: Path | str) -> dict[str, str]:
    """Install the immutable FPS16 Catcher generation profile before imports."""

    os.environ.pop("DEMO_MUJOCO_FIXED_BALL_RADIUS", None)
    os.environ.pop("DEMO_MUJOCO_FIXED_CIRCLE_RADIUS", None)
    values = {
        **PRODUCTION_ENV,
        "DEMO_MUJOCO_WORK_DIR": str(Path(work_dir).expanduser().resolve()),
    }
    for key, value in values.items():
        os.environ[key] = value
    return values


def _module_belongs_to(module: Any, root: Path) -> bool:
    module_file = Path(getattr(module, "__file__", "")).resolve(strict=False)
    try:
        module_file.relative_to(root.resolve())
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class CatcherSimulatorRuntime:
    source_root: Path
    sim: Any
    render: Any
    generate: Any


def _load_simulator_runtime(
    source_root: Path | str,
    *,
    work_dir: Path | str,
) -> CatcherSimulatorRuntime:
    """Load an isolated bundled simulator under the Catcher production profile."""

    source_root = Path(source_root).expanduser().resolve()
    required = ("common.py", "generate.py", "sim_support.py", "render_support.py")
    missing = [name for name in required if not (source_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Simulator source root {source_root} is missing required files: {missing}"
        )
    configured = configure_production_environment(work_dir=work_dir)
    for name in ("common", "generate", "sim_support", "render_support"):
        existing = sys.modules.get(name)
        if existing is not None and not _module_belongs_to(existing, source_root):
            raise RuntimeError(
                f"Refusing to reuse {name} from {getattr(existing, '__file__', None)}; "
                f"the production simulator must be imported from {source_root}"
            )
    source_text = str(source_root)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)
    importlib.invalidate_caches()
    sim = importlib.import_module("sim_support")
    render = importlib.import_module("render_support")
    generate = importlib.import_module("generate")
    for name, module in (("sim_support", sim), ("render_support", render), ("generate", generate)):
        if not _module_belongs_to(module, source_root):
            raise RuntimeError(
                f"Resolved {name} from {getattr(module, '__file__', None)}, expected {source_root}"
            )
    render.GENERATED_ASSET_DIR = (
        Path(configured["DEMO_MUJOCO_WORK_DIR"]).resolve() / "runtime/generated_render_assets"
    )
    common = importlib.import_module("common")
    observed = {
        "TASKS": list(common.TASKS),
        "EPISODES_PER_TASK": int(common.EPISODES_PER_TASK),
        "CANONICAL_EPISODES_PER_TASK": int(common.CANONICAL_EPISODES_PER_TASK),
        "FRAMES_PER_EPISODE": int(common.FRAMES_PER_EPISODE),
        "FPS": int(common.FPS),
        "STEPS_PER_FRAME": int(common.STEPS_PER_FRAME),
        "STEP_RATE": int(common.STEP_RATE),
    }
    expected = {
        "TASKS": ["arm_catcher_ball"],
        "EPISODES_PER_TASK": 9600,
        "CANONICAL_EPISODES_PER_TASK": 9600,
        "FRAMES_PER_EPISODE": 64,
        "FPS": 16,
        "STEPS_PER_FRAME": 20,
        "STEP_RATE": 320,
    }
    if observed != expected:
        raise RuntimeError(
            f"Simulator imported with the wrong production profile: {observed} != {expected}"
        )
    return CatcherSimulatorRuntime(source_root, sim, render, generate)


class CatcherRuntime:
    """Own a temporary generator runtime and its pinned Unitree assets."""

    def __init__(self, menagerie_root: str | Path | None = None) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="sgjepa-catcher-eval-")
        temporary = Path(self._temporary.name)
        self.engine_root = temporary / "engine"
        shutil.copytree(_engine_root(), self.engine_root)
        if menagerie_root is None:
            menagerie = temporary / "mujoco_menagerie"
            provision_menagerie_assets(menagerie, components=("unitree_z1",))
        else:
            menagerie = Path(menagerie_root).expanduser().resolve()
            verify_menagerie_component(menagerie, "unitree_z1")
        target = self.engine_root / "third_party" / "mujoco_menagerie" / "unitree_z1"
        target.parent.mkdir(parents=True)
        shutil.copytree(menagerie / "unitree_z1", target)
        self.runtime = _load_simulator_runtime(
            self.engine_root,
            work_dir=temporary / "work",
        )

    def close(self) -> None:
        source = str(self.engine_root)
        if source in sys.path:
            sys.path.remove(source)
        for name in ("common", "generate", "sim_support", "render_support"):
            module = sys.modules.get(name)
            if module is not None and _module_belongs_to(module, self.engine_root):
                sys.modules.pop(name, None)
        self._temporary.cleanup()

    def __enter__(self) -> CatcherSimulatorRuntime:
        return self.runtime

    def __exit__(self, *_args: Any) -> None:
        self.close()


class ArmCatcherBallEnv:
    """Validated production environment with raw ``[g, dx, dy, dz]`` actions."""

    def __init__(
        self,
        *,
        episode_metadata: Mapping[str, Any],
        runtime: CatcherSimulatorRuntime,
        expected_initial: Mapping[str, Sequence[float]],
        width: int = 256,
        height: int = 256,
        initial_tolerance: float = 2.0e-4,
    ) -> None:
        self.runtime = runtime
        self.sim = runtime.sim
        self.render = runtime.render
        self.generate = runtime.generate
        self.width = int(width)
        self.height = int(height)
        self.source_metadata = dict(episode_metadata)
        self.scene = self._rebuild_generator_scene(self.source_metadata)
        metadata = self.scene["metadata"]
        self.fps = int(metadata["fps"])
        self.frames_per_episode = int(metadata["num_frames"])
        self.step_rate = int(metadata["step_rate"])
        self.steps_per_frame = int(round(self.step_rate / self.fps))
        if (self.fps, self.frames_per_episode, self.step_rate, self.steps_per_frame) != (
            16,
            64,
            320,
            20,
        ):
            raise RuntimeError(
                "Rebuilt scene does not use the FPS16 production timing: "
                f"fps={self.fps} frames={self.frames_per_episode} "
                f"step_rate={self.step_rate} steps_per_frame={self.steps_per_frame}"
            )
        self.model, self.data = self.sim.build_arm_catcher_model(
            self.scene, step_rate=self.step_rate
        )
        self.dynamic_spec = self.scene["dynamic_objects"][0]
        self.catcher_spec = self.scene["catcher"]
        self.arm_spec = self.scene["arm"]
        self.sim.initialize_dynamic_state(self.model, self.data, self.dynamic_spec)
        self.target_position, self.joint_targets = self.sim.initialize_arm_catcher_state(
            self.model, self.data, self.catcher_spec, self.arm_spec
        )
        self.gravity = _metadata_gravity(metadata)
        self.radius = float(self.dynamic_spec["radius"])
        self.control_axes = list(self.catcher_spec.get("control_axes", ["x", "y", "z"]))
        self.action_scale = np.asarray(
            self.catcher_spec.get("action_scale_xyz", [1.0] * len(self.control_axes)),
            dtype=np.float32,
        )
        self.frame_idx = 0
        self.caught_flags: list[bool] = []
        self.captured_flags: list[bool] = []
        self.missed_flags: list[bool] = []
        self.reached_catch_plane = False
        self.captured = False
        self.missed = False
        self.capture_frame: int | None = None
        self.miss_frame: int | None = None
        self.terminal_reason: str | None = None
        self.latch_offset: list[float] | None = None
        self.capture_window = 3
        self.last_distance = math.nan
        self.frozen_arm_joint_positions: list[float] | None = None
        self.zero_arm_joint_velocities = [0.0] * len(self.arm_spec["joint_names"])
        self.renderer = self.render.OffscreenRenderer(
            width=self.width,
            height=self.height,
            render_scale=self.render.render_supersample_scale(),
        )
        self.executed_actions: list[list[float]] = []
        self._closed = False
        self._restore_initial_state(expected_initial)
        self.initial_state_max_errors = self.verify_initial_state(
            expected_initial, atol=float(initial_tolerance)
        )
        self._observe_current()

    @classmethod
    def from_episode_record(
        cls,
        record: Mapping[str, Any],
        *,
        runtime: CatcherSimulatorRuntime,
        width: int = 256,
        height: int = 256,
        initial_tolerance: float = 2.0e-4,
    ) -> ArmCatcherBallEnv:
        metadata = record["episode_metadata"]
        if isinstance(metadata, str):
            metadata = __import__("json").loads(metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("episode_metadata must decode to a mapping")
        return cls(
            episode_metadata=metadata,
            runtime=runtime,
            expected_initial={
                "state": record["state"],
                "catcher_state": record["catcher_state"],
                "arm_state": record["arm_state"],
            },
            width=width,
            height=height,
            initial_tolerance=initial_tolerance,
        )

    def _rebuild_generator_scene(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        episode_index = int(metadata["episode_index"])
        split_name = str(metadata.get("split_name", "test"))
        gravity = _metadata_gravity(metadata)
        generated = self.generate.build_episode_scene(
            "arm_catcher_ball",
            random.Random(int(metadata["random_seed"])),
            gravity,
            episode_index,
            split_name,
        )
        generated["metadata"] = dict(metadata)
        camera = metadata.get("camera_parameters")
        if isinstance(camera, Mapping):
            generated["camera"].update(camera)
        ball = metadata.get("ball_initial_state")
        if isinstance(ball, Mapping):
            generated["dynamic_objects"][0].update(
                {
                    "position": ball["position"],
                    "mass": ball["mass"],
                    "radius": ball["radius"],
                    "initial_velocity": ball["linear_velocity"],
                }
            )
        catcher = metadata.get("catcher")
        if isinstance(catcher, Mapping):
            generated["catcher"].update(catcher)
        arm = metadata.get("arm")
        if isinstance(arm, Mapping):
            generated["arm"].update(arm)
        return generated

    def _restore_initial_state(self, expected: Mapping[str, Sequence[float]]) -> None:
        arm = np.asarray(expected["arm_state"], dtype=np.float64)
        catcher = np.asarray(expected["catcher_state"], dtype=np.float64)
        ball = np.asarray(expected["state"], dtype=np.float64)
        if arm.shape != (18,) or catcher.shape != (9,) or ball.shape[0] < 13:
            raise ValueError(
                "Catcher frame zero requires state>=13, catcher_state=9 and arm_state=18"
            )
        self.sim.set_arm_joint_state(
            self.model,
            self.data,
            self.arm_spec,
            arm[:6],
            arm[6:12],
        )
        self.joint_targets = [float(value) for value in arm[12:18]]
        self.sim.set_arm_ctrl(self.data, self.joint_targets)
        self.target_position = [float(value) for value in catcher[6:9]]
        self._end_effector_state()
        self.sim.set_locked_ball_state(
            self.model,
            self.data,
            self.dynamic_spec,
            ball[:3],
            ball[3:6],
            angular_velocity=ball[10:13],
        )
        self.sim.mujoco.mj_forward(self.model, self.data)
        self._end_effector_state()

    def _end_effector_state(self):
        # The production Unitree model exposes the flange as a site on
        # ``link06`` and the catcher itself as a world-level mocap body.  The
        # historical evaluator queried an ``arm_catcher_ee`` body from an
        # older simulator snapshot; that body does not exist in the pinned
        # public generator.  Keep the state contract in net-back-face
        # coordinates by using the same synchronization helper as dataset
        # generation.
        position, velocity = self.sim.sync_arm_catcher_mount_for_scene(
            self.model,
            self.data,
            self.catcher_spec,
            self.arm_spec,
        )
        return (
            position,
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            velocity,
            np.zeros(3, dtype=np.float64),
        )

    def _tip_target_for_mount(self, mount_target: Sequence[float]) -> np.ndarray:
        return self.sim.arm_catcher_tip_target_for_mount(
            mount_target,
            self.catcher_spec.get("mount_back_offset", 0.0),
            self.arm_spec.get("tip_normal_world", (0.0, 1.0, 0.0)),
        )

    def state_vectors(self) -> dict[str, list[float]]:
        ball_position, quaternion, ball_velocity, angular_velocity = self.sim.body_state(
            self.data, self.dynamic_spec["name"]
        )
        catcher_position, _quat, catcher_velocity, _angular = self._end_effector_state()
        joints, joint_velocities = self.sim.arm_joint_state(self.model, self.data, self.arm_spec)
        return {
            "state": [
                *map(float, ball_position),
                *map(float, ball_velocity),
                *map(float, quaternion),
                *map(float, angular_velocity),
            ],
            "catcher_state": [
                *map(float, catcher_position),
                *map(float, catcher_velocity),
                *map(float, self.target_position),
            ],
            "arm_state": [
                *map(float, joints),
                *map(float, joint_velocities),
                *map(float, self.joint_targets),
            ],
        }

    def verify_initial_state(
        self,
        expected: Mapping[str, Sequence[float]],
        *,
        atol: float = 2.0e-4,
    ) -> dict[str, float]:
        actual = self.state_vectors()
        wanted = {
            "state": [float(value) for value in expected["state"][:13]],
            "catcher_state": [float(value) for value in expected["catcher_state"]],
            "arm_state": [float(value) for value in expected["arm_state"]],
        }
        errors: dict[str, float] = {}
        for name, expected_values in wanted.items():
            actual_values = actual[name]
            if len(actual_values) != len(expected_values):
                raise ValueError(
                    f"Initial {name} width mismatch: {len(actual_values)} != {len(expected_values)}"
                )
            error = max(
                (
                    abs(observed - target)
                    for observed, target in zip(actual_values, expected_values, strict=True)
                ),
                default=0.0,
            )
            errors[name] = float(error)
        if failed := {name: value for name, value in errors.items() if value > atol}:
            self.close()
            raise RuntimeError(f"simulator reconstruction differs from Lance frame zero: {failed}")
        return errors

    def _freeze_captured_arm_and_ball(self) -> None:
        if self.frozen_arm_joint_positions is None or self.latch_offset is None:
            return
        self.sim.set_arm_ctrl(self.data, self.frozen_arm_joint_positions)
        self.sim.set_arm_joint_state(
            self.model,
            self.data,
            self.arm_spec,
            self.frozen_arm_joint_positions,
            self.zero_arm_joint_velocities,
        )
        ee_position, _quat, _velocity, _angular = self._end_effector_state()
        self.sim.set_locked_ball_state(
            self.model,
            self.data,
            self.dynamic_spec,
            self.sim.add_vectors(ee_position, self.latch_offset),
            (0.0, 0.0, 0.0),
        )

    def _observe_current(self) -> None:
        if self.captured:
            self._freeze_captured_arm_and_ball()
        ee_position, _quat, ee_velocity, _angular = self._end_effector_state()
        ball_position, _ball_quat, ball_velocity, _ball_angular = self.sim.body_state(
            self.data, self.dynamic_spec["name"]
        )
        self.last_distance = float(
            np.linalg.norm(np.asarray(ball_position) - np.asarray(ee_position))
        )
        flags = self.sim.contact_flags(self.model, self.data)
        rel_y = float(ball_position[1] - ee_position[1])
        half_depth = 0.5 * float(self.catcher_spec["net_depth"])
        if rel_y <= half_depth + 0.08:
            self.reached_catch_plane = True
        candidate_caught = (
            self.sim.caught_frame(
                ball_position,
                ball_velocity,
                ee_position,
                ee_velocity,
                flags,
                self.catcher_spec,
                self.radius,
            )
            if not self.missed
            else False
        )
        recent = [*self.caught_flags, candidate_caught][-self.capture_window :]
        if (
            not self.captured
            and not self.missed
            and self.sim.has_consecutive_true(recent, self.capture_window)
        ):
            rel_pos = [float(ball_position[index] - ee_position[index]) for index in range(3)]
            half_width = float(self.catcher_spec["net_half_width"])
            half_height = float(self.catcher_spec["net_half_height"])
            self.latch_offset = [
                self.sim.clip_value(rel_pos[0], -0.45 * half_width, 0.45 * half_width),
                self.sim.clip_value(rel_pos[1], -0.50 * half_depth, 0.25 * half_depth),
                self.sim.clip_value(rel_pos[2], -0.45 * half_height, 0.45 * half_height),
            ]
            self.captured = True
            self.capture_frame = self.frame_idx
            self.terminal_reason = self.terminal_reason or "captured"
            frozen, _velocities = self.sim.arm_joint_state(self.model, self.data, self.arm_spec)
            self.frozen_arm_joint_positions = [float(value) for value in frozen]
            self.joint_targets = list(self.frozen_arm_joint_positions)
            self._freeze_captured_arm_and_ball()
            ee_position, _quat, _velocity, _angular = self._end_effector_state()
            self.target_position = self.sim.clip_vector(
                list(ee_position),
                self.catcher_spec["workspace_min"],
                self.catcher_spec["workspace_max"],
            )
            candidate_caught = True
        if not self.captured and not self.missed:
            reason = self.sim.miss_reason(
                ball_position,
                ee_position,
                flags,
                self.catcher_spec,
                self.radius,
                self.reached_catch_plane,
            )
            if reason is not None:
                self.missed = True
                self.miss_frame = self.frame_idx
                self.terminal_reason = str(reason)
                candidate_caught = False
        self.caught_flags.append(bool(candidate_caught))
        self.captured_flags.append(bool(self.captured))
        self.missed_flags.append(bool(self.missed))

    @property
    def terminal(self) -> bool:
        return bool(self.captured or self.missed or self.frame_idx >= self.frames_per_episode - 1)

    def _arm_render_payload(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "name": self.arm_spec["name"],
            "robot_model": self.arm_spec.get("robot_model"),
            "model_scale": self.arm_spec.get("model_scale", 1.0),
            "frames": [dict(frame)],
            "base_position": self.arm_spec["base_position"],
            "base_euler": self.arm_spec.get("base_euler", [0.0, 0.0, 0.0]),
            "shoulder_height": self.arm_spec["shoulder_height"],
            "upper_link_length": self.arm_spec["upper_link_length"],
            "lower_link_length": self.arm_spec["lower_link_length"],
            "joint_names": self.arm_spec["joint_names"],
            "joint_limits": self.arm_spec["joint_limits"],
            "ik_controller": "analytic_yaw_shoulder_elbow",
        }

    def render_current(self) -> np.ndarray:
        ball_position, quaternion, ball_velocity, angular_velocity = self.sim.body_state(
            self.data, self.dynamic_spec["name"]
        )
        catcher_position, _quat, catcher_velocity, _angular = self._end_effector_state()
        joints, joint_velocities = self.sim.arm_joint_state(self.model, self.data, self.arm_spec)
        dynamic_payload = self.sim.make_dynamic_payload(self.dynamic_spec)
        dynamic_payload["frames"].append(
            self.sim.frame_payload(
                0,
                self.fps,
                ball_position,
                quaternion,
                ball_velocity,
                angular_velocity,
            )
        )
        catcher_frame = {
            "frame": 1,
            "time_seconds": 0.0,
            "position": [float(value) for value in catcher_position],
            "linear_velocity": [float(value) for value in catcher_velocity],
            "target_position": [float(value) for value in self.target_position],
        }
        arm_frame = {
            "frame": 1,
            "time_seconds": 0.0,
            "joint_position": [float(value) for value in joints],
            "joint_velocity": [float(value) for value in joint_velocities],
            "joint_target": [float(value) for value in self.joint_targets],
        }
        payload = {
            "metadata": {**self.scene["metadata"], "num_frames": 1},
            "camera": self.scene["camera"],
            "static_objects": [
                self.sim.serialize_object(spec) for spec in self.scene["static_objects"]
            ],
            "dynamic_objects": [
                dynamic_payload,
                *self.sim.make_catcher_render_payloads(
                    self.catcher_spec, [catcher_frame], self.fps
                ),
            ],
            "arm": self._arm_render_payload(arm_frame),
        }
        frames = self.renderer.render_episode_frames(
            payload,
            version="planning",
            task_name="arm_catcher_ball",
            episode_index=int(self.scene["metadata"]["episode_index"]),
        )
        return np.asarray(frames[0], dtype=np.uint8)

    def step(self, raw_action: Sequence[float]) -> None:
        if self.terminal:
            return
        action = np.asarray(raw_action, dtype=np.float32).reshape(4).copy()
        if not np.isfinite(action).all():
            raise ValueError("raw_action must contain four finite values [g,dx,dy,dz]")
        if not np.isclose(float(action[0]), self.gravity, atol=1.0e-6, rtol=0.0):
            raise ValueError(f"raw action gravity {action[0]} differs from scene {self.gravity}")
        action[0] = self.gravity
        bounds = self.scene["metadata"].get("normalized_action_bounds", [-1.0, 1.0])
        action[1:] = np.clip(action[1:], float(bounds[0]), float(bounds[1]))
        self.executed_actions.append(action.tolist())
        if self.captured and self.frozen_arm_joint_positions is not None:
            for _ in range(self.steps_per_frame):
                self._freeze_captured_arm_and_ball()
                self.sim.mujoco.mj_step(self.model, self.data)
            self.frame_idx += 1
            self._observe_current()
            return
        if not self.missed:
            physical_delta = [float(value) for value in action[1:] * self.action_scale]
            self.target_position = self.sim.apply_control_delta(
                self.target_position,
                physical_delta,
                self.control_axes,
                self.catcher_spec,
            )
        current_joints, _velocities = self.sim.arm_joint_state(self.model, self.data, self.arm_spec)
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        self.joint_targets = self.sim.solve_unitree_z1_ik(
            self.model,
            self.data,
            self.arm_spec,
            self._tip_target_for_mount(self.target_position),
            current_joints,
            allow_fallback_seeds=False,
        )
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.sim.mujoco.mj_forward(self.model, self.data)
        for _ in range(self.steps_per_frame):
            self.sim.set_arm_ctrl(self.data, self.joint_targets)
            self.sim.mujoco.mj_step(self.model, self.data)
        self.frame_idx += 1
        self._observe_current()

    def last_executed_action(self) -> list[float]:
        if not self.executed_actions:
            raise RuntimeError("no action has been executed")
        return [float(value) for value in self.executed_actions[-1]]

    def result(self) -> dict[str, Any]:
        terminal_reason = self.terminal_reason
        if terminal_reason is None and self.frame_idx >= self.frames_per_episode - 1:
            terminal_reason = "full_horizon"
        return {
            "captured": bool(self.captured),
            "missed": bool(self.missed),
            "capture_frame": self.capture_frame,
            "miss_frame": self.miss_frame,
            "terminal_reason": str(terminal_reason or "evaluation_budget"),
            "time_to_capture_seconds": (
                None if self.capture_frame is None else float(self.capture_frame / self.fps)
            ),
            "final_ball_catcher_distance": float(self.last_distance),
            "final_frame": int(self.frame_idx),
            "executed_action_count": len(self.executed_actions),
            "initial_state_max_errors": dict(self.initial_state_max_errors),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if getattr(self, "renderer", None) is not None:
            self.renderer.free()

    def __enter__(self) -> ArmCatcherBallEnv:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


__all__ = [
    "ArmCatcherBallEnv",
    "CatcherRuntime",
    "CatcherSimulatorRuntime",
    "PRODUCTION_ENV",
    "_metadata_gravity",
]
