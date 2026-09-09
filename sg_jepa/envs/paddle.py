"""Closed-loop MuJoCo adapter for the 16 Hz arm-paddle task."""

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

from data_generation.common.assets import (
    provision_menagerie_assets,
    verify_menagerie_component,
)
from data_generation.mujoco.generator import _engine_root

from .paddle_metrics import ContactCounter, episode_metrics

TEST_GRAVITIES = (
    "0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,"
    "13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,8.87,3.721,1.625,0.62"
)
PRODUCTION_ENV = {
    "MUJOCO_GL": "egl",
    "DEMO_MUJOCO_TASKS": "arm_paddle_ball",
    "DEMO_MUJOCO_OUTPUT_VERSIONS": "v0",
    "DEMO_MUJOCO_EPISODES_PER_TASK": "9600",
    "DEMO_MUJOCO_CANONICAL_EPISODES_PER_TASK": "9600",
    "DEMO_MUJOCO_SELECTED_SPLIT": "all",
    "DEMO_MUJOCO_BASE_SEED": "20260525",
    "DEMO_MUJOCO_DYNAMIC_SHAPE_MODE": "ball_only",
    "DEMO_MUJOCO_RENDER_ENV": "natural",
    "DEMO_MUJOCO_IMAGE_WIDTH": "256",
    "DEMO_MUJOCO_IMAGE_HEIGHT": "256",
    "DEMO_MUJOCO_FRAMES_PER_EPISODE": "256",
    "DEMO_MUJOCO_FPS": "16",
    "DEMO_MUJOCO_STEPS_PER_FRAME": "20",
    "DEMO_MUJOCO_STEP_RATE": "320",
    "DEMO_MUJOCO_TRAIN_GAUSSIAN_MEAN": "9.8",
    "DEMO_MUJOCO_TRAIN_GAUSSIAN_STD": "2.0",
    "DEMO_MUJOCO_TRAIN_G_MIN": "0.0",
    "DEMO_MUJOCO_TEST_GRAVITIES": TEST_GRAVITIES,
    "DEMO_MUJOCO_GRAVITY_BATCH_EPISODES": "30",
    "DEMO_MUJOCO_TRAIN_EPISODES_PER_BATCH": "25",
    "DEMO_MUJOCO_TEST_EPISODES_PER_BATCH": "5",
    "DEMO_MUJOCO_RENDER_SUPERSAMPLE": "2",
    "DEMO_MUJOCO_RENDER_SOFTEN": "0.0",
    "DEMO_MUJOCO_OPEN_BOX_SHADOW_LIFT": "0.0",
    "DEMO_MUJOCO_SHARD_EPISODES": "8",
    "DEMO_MUJOCO_RENDER_WORKERS": "1",
}


def configure_production_environment(
    *, frames_per_episode: int = 256, work_dir: Path | str | None = None
) -> dict[str, str]:
    """Force an explicit immutable FPS16 production timing profile.

    The historical FPS16 workflow used 256 frames.  FPS16 v2 uses the same
    simulator and physical timing with 64 recorded frames, so callers must
    opt into that profile explicitly while old callers retain their original
    default.
    """

    # A stale radius override silently changes production physics.
    os.environ.pop("DEMO_MUJOCO_FIXED_BALL_RADIUS", None)
    os.environ.pop("DEMO_MUJOCO_FIXED_CIRCLE_RADIUS", None)
    if int(frames_per_episode) not in (64, 256):
        raise ValueError("FPS16 production frames_per_episode must be 64 or 256")
    values = dict(PRODUCTION_ENV)
    values["DEMO_MUJOCO_FRAMES_PER_EPISODE"] = str(int(frames_per_episode))
    if work_dir is not None:
        values["DEMO_MUJOCO_WORK_DIR"] = str(Path(work_dir).expanduser().resolve())
    for key, value in values.items():
        os.environ[key] = value
    return values


@dataclass(frozen=True)
class SimulatorRuntime:
    source_root: Path
    np: Any
    sim: Any
    render: Any
    generate: Any


def _module_belongs_to(module: Any, root: Path) -> bool:
    module_file = Path(getattr(module, "__file__", "")).resolve(strict=False)
    try:
        module_file.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_simulator_runtime(
    source_root: Path | str,
    *,
    frames_per_episode: int = 256,
    work_dir: Path | str | None = None,
) -> SimulatorRuntime:
    """Load an isolated runtime copy after installing the locked task constants."""

    source_root = Path(source_root).expanduser().resolve()
    required = ("common.py", "generate.py", "sim_support.py", "render_support.py")
    missing = [name for name in required if not (source_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Simulator source root {source_root} is missing required files: {missing}"
        )
    configured = configure_production_environment(
        frames_per_episode=frames_per_episode,
        work_dir=work_dir,
    )
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
    numpy = importlib.import_module("numpy")
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
        "FRAMES_PER_EPISODE": int(common.FRAMES_PER_EPISODE),
        "FPS": int(common.FPS),
        "STEPS_PER_FRAME": int(common.STEPS_PER_FRAME),
        "STEP_RATE": int(common.STEP_RATE),
    }
    expected = {
        "TASKS": ["arm_paddle_ball"],
        "EPISODES_PER_TASK": 9600,
        "FRAMES_PER_EPISODE": int(frames_per_episode),
        "FPS": 16,
        "STEPS_PER_FRAME": 20,
        "STEP_RATE": 320,
    }
    if observed != expected:
        raise RuntimeError(
            f"Simulator imported with the wrong production profile: {observed} != {expected}"
        )
    return SimulatorRuntime(source_root, numpy, sim, render, generate)


class PaddleRuntime:
    """Own a temporary generator runtime and its pinned Unitree assets."""

    def __init__(
        self,
        menagerie_root: str | Path | None = None,
        *,
        frames_per_episode: int = 64,
    ) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="sgjepa-paddle-eval-")
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
            frames_per_episode=frames_per_episode,
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

    def __enter__(self) -> SimulatorRuntime:
        return self.runtime

    def __exit__(self, *_args: Any) -> None:
        self.close()


@dataclass(frozen=True)
class StepResult:
    frame_index: int
    paddle_contact: bool
    counted_paddle_contact: bool
    floor_contact: bool
    wall_contact: bool
    terminated: bool


class PersistentSceneRenderer:
    """Exact source renderer with one model/context allocation per episode."""

    def __init__(
        self, *, runtime: SimulatorRuntime, payload: Mapping[str, Any], width: int, height: int
    ):
        self.runtime = runtime
        self.render = runtime.render
        self.mujoco = runtime.render.mujoco
        self.np = runtime.np
        self.width = int(width)
        self.height = int(height)
        self.scale = int(self.render.render_supersample_scale())
        self.render_width = self.width * self.scale
        self.render_height = self.height * self.scale
        self.gl_context = self.mujoco.GLContext(self.render_width, self.render_height)
        self.gl_context.make_current()
        self.model = self.render.build_render_model(
            payload,
            self.render_width,
            self.render_height,
            version="policy_evaluation",
            task_name="arm_paddle_ball",
            episode_index=int(payload["metadata"]["episode_index"]),
        )
        self.data = self.mujoco.MjData(self.model)
        self.scene = self.mujoco.MjvScene(self.model, maxgeom=2048)
        self.option = self.mujoco.MjvOption()
        self.camera = self.mujoco.MjvCamera()
        self.camera.type = self.mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera.fixedcamid = int(
            self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_CAMERA,
                "render_camera",
            )
        )
        self.context = self.mujoco.MjrContext(self.model, self.mujoco.mjtFontScale.mjFONTSCALE_100)
        self.mujoco.mjr_setBuffer(self.mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        self.viewport = self.mujoco.MjrRect(0, 0, self.render_width, self.render_height)
        self.pixels = self.np.empty((self.render_height, self.render_width, 3), dtype=self.np.uint8)
        self.pose_offsets = self.render.freejoint_qpos_offsets(
            self.model, payload["dynamic_objects"]
        )
        self.arm_offsets = self.render.arm_joint_qpos_offsets(self.model, payload)
        self.dynamic_names = tuple(spec["name"] for spec in payload["dynamic_objects"])
        self.use_default_render_env = self.render.render_env_mode() == "default"
        self.soften_strength = self.render.render_soften_strength()
        self.closed = False

    def render_frame(self, payload: Mapping[str, Any]):
        if self.closed:
            raise RuntimeError("PersistentSceneRenderer is closed")
        self.gl_context.make_current()
        names = tuple(spec["name"] for spec in payload["dynamic_objects"])
        if names != self.dynamic_names:
            raise ValueError(
                f"Dynamic render object order changed: {names} != {self.dynamic_names}"
            )
        if self.arm_offsets is not None:
            self.render.set_arm_joint_positions(
                self.data,
                self.arm_offsets,
                payload["arm"]["frames"][0]["joint_position"],
            )
        for dynamic_spec in payload["dynamic_objects"]:
            frame = dynamic_spec["frames"][0]
            self.render.set_freejoint_pose(
                self.data,
                self.pose_offsets[dynamic_spec["name"]],
                frame["position"],
                frame["quaternion_xyzw"],
            )
        self.mujoco.mj_forward(self.model, self.data)
        self.mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            None,
            self.camera,
            self.mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        if not self.use_default_render_env:
            self.scene.flags[int(self.mujoco.mjtRndFlag.mjRND_SHADOW)] = 1
        self.mujoco.mjr_render(self.viewport, self.scene, self.context)
        self.mujoco.mjr_readPixels(self.pixels, None, self.viewport, self.context)
        frame = self.np.flipud(self.pixels)
        frame = self.render.downsample_frame(frame, self.width, self.height, self.scale)
        return self.render.soften_frame(frame, self.soften_strength).copy()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if getattr(self, "context", None) is not None:
            self.context.free()
        if getattr(self, "gl_context", None) is not None:
            self.gl_context.free()


class ArmPaddleBallEnv:
    """Closed-loop adapter for production high-level paddle actions."""

    def __init__(
        self,
        *,
        episode_metadata: Mapping[str, Any],
        runtime: SimulatorRuntime,
        expected_initial: Mapping[str, Sequence[float]] | None = None,
        width: int = 256,
        height: int = 256,
        initial_tolerance: float = 5e-4,
        expected_frames_per_episode: int = 256,
    ) -> None:
        self.runtime = runtime
        self.np = runtime.np
        self.sim = runtime.sim
        self.generate = runtime.generate
        self.width = int(width)
        self.height = int(height)
        self.source_metadata = dict(episode_metadata)
        self.scene = self._rebuild_scene(self.source_metadata)
        metadata = self.scene["metadata"]
        self.fps = int(metadata["fps"])
        self.frames_per_episode = int(metadata["num_frames"])
        self.step_rate = int(metadata["step_rate"])
        self.steps_per_frame = int(round(self.step_rate / self.fps))
        if (self.fps, self.frames_per_episode, self.step_rate, self.steps_per_frame) != (
            16,
            int(expected_frames_per_episode),
            320,
            20,
        ):
            raise RuntimeError(
                "Rebuilt scene does not use the FPS16 production timing: "
                f"fps={self.fps} frames={self.frames_per_episode} "
                f"step_rate={self.step_rate} steps_per_frame={self.steps_per_frame}"
            )
        self.model, self.data = self.sim.build_arm_paddle_model(
            self.scene, step_rate=self.step_rate
        )
        self.dynamic_spec = self.scene["dynamic_objects"][0]
        self.paddle_spec = self.scene["paddle"]
        self.arm_spec = self.scene["arm"]
        self.sim.initialize_dynamic_state(self.model, self.data, self.dynamic_spec)
        self.sim.initialize_paddle_state(self.model, self.data, self.paddle_spec)
        self.gravity = abs(float(metadata["gravity"][2]))
        self.target_position = [
            float(value) for value in self.paddle_spec["initial_target_position"]
        ]
        self.target_phi = float(self.paddle_spec.get("initial_target_phi", 0.0))
        self.target_theta = float(self.paddle_spec.get("initial_target_theta", 0.0))
        self.max_tilt_theta = float(self.paddle_spec["max_tilt_theta"])
        self.action_scale = [float(value) for value in self.paddle_spec["action_scale_xyz"]]
        self.frame_idx = 0
        self.counter = ContactCounter()
        self.executed_actions: list[list[float]] = []
        self.contact_intervals: list[dict[str, bool]] = []
        self.zero_arm_velocities = [0.0] * len(self.arm_spec["joint_names"])
        initial_mount, initial_axis = self.sim.arm_paddle_mount_pose(
            self.target_position,
            self.sim.paddle_command_quaternion(
                self.target_phi, self.target_theta, self.max_tilt_theta
            ),
            self.paddle_spec,
        )
        self.joint_targets = self.sim.solve_arm_paddle_tip_ik(
            self.model,
            self.data,
            self.arm_spec,
            initial_mount,
            initial_axis,
            self.arm_spec.get("home_qpos", self.zero_arm_velocities),
        )
        self.arm_joint_positions = list(self.joint_targets)
        self.arm_joint_velocities = list(self.zero_arm_velocities)
        self.sim.set_arm_joint_state(
            self.model,
            self.data,
            self.arm_spec,
            self.arm_joint_positions,
            self.zero_arm_velocities,
        )
        self.sim.set_named_arm_ctrl(self.model, self.data, self.arm_spec, self.joint_targets)
        self._last_arm_sync_frame: int | None = None
        self._renderer: PersistentSceneRenderer | None = None
        self._closed = False
        self._sync_arm_to_actual_paddle()
        if expected_initial is not None:
            self.verify_initial_state(expected_initial, atol=float(initial_tolerance))
        self.ball_z_trace: list[float] = []
        self.paddle_z_trace: list[float] = []
        self._record_vertical_state()

    @classmethod
    def from_episode_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        runtime: SimulatorRuntime,
        expected_initial: Mapping[str, Sequence[float]] | None = None,
        width: int = 256,
        height: int = 256,
        expected_frames_per_episode: int = 256,
    ) -> ArmPaddleBallEnv:
        return cls(
            episode_metadata=metadata,
            runtime=runtime,
            expected_initial=expected_initial,
            width=width,
            height=height,
            expected_frames_per_episode=expected_frames_per_episode,
        )

    def _rebuild_scene(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        episode_index = int(metadata["episode_index"])
        split_name = str(metadata["split_name"])
        gravity = abs(float(metadata["gravity"][2]))
        random_seed = int(metadata["random_seed"])
        scene = self.generate.build_episode_scene(
            "arm_paddle_ball",
            random.Random(random_seed),
            gravity,
            episode_index,
            split_name,
        )
        rebuilt = scene["metadata"]
        checks = {
            "scenario": (rebuilt["scenario"], "arm_paddle_ball"),
            "episode_index": (int(rebuilt["episode_index"]), episode_index),
            "split_name": (rebuilt["split_name"], split_name),
            "random_seed": (int(rebuilt["random_seed"]), random_seed),
        }
        mismatches = {key: values for key, values in checks.items() if values[0] != values[1]}
        if mismatches:
            raise ValueError(f"Rebuilt scene metadata mismatch: {mismatches}")
        if not math.isclose(abs(float(rebuilt["gravity"][2])), gravity, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"Rebuilt gravity mismatch: {rebuilt['gravity']} vs {metadata['gravity']}"
            )

        # Bounded public datasets preserve canonical source episode IDs while
        # changing the number of generated episodes.  Membership in the
        # generator's deterministic rest-start cohort depends on that episode
        # count, so rebuilding solely from the seed can select a different
        # initial-condition branch.  The Lance metadata is authoritative for
        # the sampled initial state; restore it before constructing MuJoCo.
        ball = metadata["ball_initial_state"]
        dynamic = scene["dynamic_objects"][0]
        dynamic["position"] = [float(value) for value in ball["position"]]
        dynamic["initial_velocity"] = [float(value) for value in ball["linear_velocity"]]
        dynamic["initial_angular_velocity"] = [float(value) for value in ball["angular_velocity"]]
        if "quaternion_xyzw" in ball:
            dynamic["initial_quaternion"] = [float(value) for value in ball["quaternion_xyzw"]]

        paddle = metadata["paddle_initial_state"]
        scene["paddle"]["initial_position"] = [float(value) for value in paddle["position"]]
        scene["paddle"]["initial_target_position"] = [
            float(value) for value in paddle["target_position"]
        ]
        scene["paddle"]["initial_target_phi"] = float(paddle["target_phi"])
        scene["paddle"]["initial_target_theta"] = float(paddle["target_theta"])
        scene["metadata"] = dict(metadata)
        return scene

    def _sync_arm_to_actual_paddle(self) -> None:
        if self._last_arm_sync_frame == self.frame_idx:
            return
        position, _velocity = self.sim.paddle_joint_state(self.model, self.data, self.paddle_spec)
        orientation = self.sim.paddle_orientation_state(
            self.model,
            self.data,
            self.paddle_spec,
            fallback_phi=self.target_phi,
        )
        mount, axis = self.sim.arm_paddle_mount_pose(
            position, orientation["quaternion_xyzw"], self.paddle_spec
        )
        previous = list(self.arm_joint_positions)
        current = self.sim.solve_arm_paddle_tip_ik(
            self.model,
            self.data,
            self.arm_spec,
            mount,
            axis,
            previous,
        )
        self.sim.set_arm_joint_state(
            self.model,
            self.data,
            self.arm_spec,
            current,
            self.zero_arm_velocities,
        )
        self.arm_joint_positions = [float(value) for value in current]
        self.arm_joint_velocities = (
            [
                (float(current_value) - float(previous_value)) * self.fps
                for current_value, previous_value in zip(current, previous, strict=False)
            ]
            if self.frame_idx > 0
            else list(self.zero_arm_velocities)
        )
        self._last_arm_sync_frame = self.frame_idx

    def state_vectors(self) -> dict[str, list[float]]:
        """Return current ball, paddle and arm state vectors."""

        self._sync_arm_to_actual_paddle()
        position, quaternion, velocity, angular_velocity = self.sim.body_state(
            self.data, self.dynamic_spec["name"]
        )
        paddle_position, paddle_velocity = self.sim.paddle_joint_state(
            self.model, self.data, self.paddle_spec
        )
        return {
            "state": [
                *[float(value) for value in position],
                *[float(value) for value in velocity],
                *[float(value) for value in quaternion],
                *[float(value) for value in angular_velocity],
                *[float(value) for value in self.scene["metadata"]["state_anchor"]],
            ],
            "paddle_state": [
                *[float(value) for value in paddle_position],
                *[float(value) for value in paddle_velocity],
                *[float(value) for value in self.target_position],
            ],
            "arm_state": [
                *self.arm_joint_positions,
                *self.arm_joint_velocities,
                *[float(value) for value in self.joint_targets],
            ],
        }

    def _record_vertical_state(self) -> None:
        """Record one state sample for the paper's T+1 true-flight gate."""

        position, _quaternion, _velocity, _angular_velocity = self.sim.body_state(
            self.data, self.dynamic_spec["name"]
        )
        paddle_position, _paddle_velocity = self.sim.paddle_joint_state(
            self.model, self.data, self.paddle_spec
        )
        self.ball_z_trace.append(float(position[2]))
        self.paddle_z_trace.append(float(paddle_position[2]))

    def initial_state_vectors(self) -> dict[str, list[float]]:
        """Compatibility alias for frame-zero callers."""

        return self.state_vectors()

    def verify_initial_state(
        self, expected: Mapping[str, Sequence[float]], *, atol: float = 5e-4
    ) -> dict[str, float]:
        actual = self.state_vectors()
        aliases = {
            "state": "state",
            "initial_state": "state",
            "paddle_state": "paddle_state",
            "initial_paddle_state": "paddle_state",
            "arm_state": "arm_state",
            "initial_arm_state": "arm_state",
        }
        errors: dict[str, float] = {}
        compared = set()
        for supplied_name, canonical_name in aliases.items():
            if supplied_name not in expected or canonical_name in compared:
                continue
            compared.add(canonical_name)
            wanted = [float(value) for value in expected[supplied_name]]
            observed = actual[canonical_name]
            if len(wanted) != len(observed):
                raise ValueError(
                    f"Initial {canonical_name} width mismatch: {len(observed)} != {len(wanted)}"
                )
            max_error = max(
                (abs(left - right) for left, right in zip(observed, wanted, strict=False)),
                default=0.0,
            )
            errors[canonical_name] = max_error
            if max_error > atol:
                raise ValueError(
                    f"Reconstructed initial {canonical_name} differs from Lance by "
                    f"{max_error:.6g} (tolerance {atol:.6g})"
                )
        required = {"state", "paddle_state", "arm_state"}
        if compared != required:
            raise ValueError(
                f"Expected Lance initial vectors {sorted(required)}, received {sorted(compared)}"
            )
        return errors

    @property
    def max_transition_frames(self) -> int:
        return self.frames_per_episode - 1

    def controller_calibration(self) -> dict[str, Any]:
        """Return actuator calibration without measured state."""

        return {
            "initial_target_position": [
                float(value) for value in self.paddle_spec["initial_target_position"]
            ],
            "initial_target_phi": float(self.paddle_spec.get("initial_target_phi", 0.0)),
            "initial_target_theta": float(self.paddle_spec.get("initial_target_theta", 0.0)),
            "workspace_min": [float(value) for value in self.paddle_spec["workspace_min"]],
            "workspace_max": [float(value) for value in self.paddle_spec["workspace_max"]],
            "action_scale_xyz": [float(value) for value in self.action_scale],
            "max_tilt_theta": float(self.max_tilt_theta),
        }

    def last_executed_action(self) -> list[float]:
        if not self.executed_actions:
            raise RuntimeError("no action has been executed")
        return [float(value) for value in self.executed_actions[-1]]

    @property
    def terminal(self) -> bool:
        return bool(self.counter.floor_failure or self.frame_idx >= self.max_transition_frames)

    def _render_payload(self) -> dict[str, Any]:
        self._sync_arm_to_actual_paddle()
        ball_position, ball_quaternion, ball_velocity, ball_angular = self.sim.body_state(
            self.data, self.dynamic_spec["name"]
        )
        paddle_position, paddle_velocity = self.sim.paddle_joint_state(
            self.model, self.data, self.paddle_spec
        )
        orientation = self.sim.paddle_orientation_state(
            self.model,
            self.data,
            self.paddle_spec,
            fallback_phi=self.target_phi,
        )
        ball_payload = self.sim.make_dynamic_payload(self.dynamic_spec)
        ball_payload["frames"].append(
            self.sim.frame_payload(
                self.frame_idx,
                self.fps,
                ball_position,
                ball_quaternion,
                ball_velocity,
                ball_angular,
            )
        )
        paddle_frame = {
            "frame": self.frame_idx + 1,
            "time_seconds": self.frame_idx / self.fps,
            "position": [float(value) for value in paddle_position],
            "linear_velocity": [float(value) for value in paddle_velocity],
            "target_position": list(self.target_position),
            "orientation_phi": float(orientation["orientation_phi"]),
            "orientation_theta": float(orientation["orientation_theta"]),
            "target_phi": float(self.target_phi),
            "target_theta": float(self.target_theta),
            "quaternion_xyzw": [float(value) for value in orientation["quaternion_xyzw"]],
            "angular_velocity": [float(value) for value in orientation["angular_velocity"]],
        }
        paddle_payloads = self.sim.make_paddle_render_payloads(
            self.paddle_spec, [paddle_frame], self.fps
        )
        arm_frame = {
            "frame": self.frame_idx + 1,
            "time_seconds": self.frame_idx / self.fps,
            "joint_position": list(self.arm_joint_positions),
            "joint_velocity": list(self.arm_joint_velocities),
            "joint_target": [float(value) for value in self.joint_targets],
        }
        arm_payload = {
            "name": self.arm_spec["name"],
            "robot_model": self.arm_spec.get("robot_model"),
            "model_scale": self.arm_spec.get("model_scale", 1.0),
            "frames": [arm_frame],
            "base_position": self.arm_spec["base_position"],
            "base_euler": self.arm_spec.get("base_euler", [0.0, 0.0, 0.0]),
            "joint_names": self.arm_spec["joint_names"],
            "joint_limits": self.arm_spec["joint_limits"],
            "home_qpos": self.arm_spec.get("home_qpos"),
            "tool_tip_local_x": self.arm_spec.get("tool_tip_local_x"),
            "paddle_handle": self.paddle_spec.get("handle"),
        }
        return {
            "metadata": {**self.scene["metadata"], "num_frames": 1},
            "camera": self.scene["camera"],
            "static_objects": [
                self.sim.serialize_object(spec) for spec in self.scene["static_objects"]
            ],
            "dynamic_objects": [ball_payload, *paddle_payloads],
            "arm": arm_payload,
        }

    def render_current(self):
        payload = self._render_payload()
        if self._renderer is None:
            self._renderer = PersistentSceneRenderer(
                runtime=self.runtime,
                payload=payload,
                width=self.width,
                height=self.height,
            )
        return self._renderer.render_frame(payload)

    def step(self, raw_action: Sequence[float]) -> StepResult:
        if self.terminal:
            return StepResult(
                frame_index=self.frame_idx,
                paddle_contact=False,
                counted_paddle_contact=False,
                floor_contact=self.counter.floor_failure,
                wall_contact=False,
                terminated=True,
            )
        values = [float(value) for value in raw_action]
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"raw_action must contain six finite values [g,dx,dy,dz,phi,theta], got {values}"
            )
        gravity_tolerance = max(1e-3, 1e-4 * max(1.0, self.gravity))
        if not math.isclose(values[0], self.gravity, rel_tol=0.0, abs_tol=gravity_tolerance):
            raise ValueError(
                "Policy attempted to control the exogenous gravity channel: "
                f"action g={values[0]} scene g={self.gravity}"
            )
        bounds = self.scene["metadata"].get("normalized_action_bounds", [-1.0, 1.0])
        normalized_xyz = [
            self.sim.clip_value(value, float(bounds[0]), float(bounds[1])) for value in values[1:4]
        ]
        physical_delta = [normalized_xyz[index] * self.action_scale[index] for index in range(3)]
        self.target_position = self.sim.clip_vector(
            [self.target_position[index] + physical_delta[index] for index in range(3)],
            self.paddle_spec["workspace_min"],
            self.paddle_spec["workspace_max"],
        )
        self.target_phi = float(self.sim.wrap_angle_pi(values[4]))
        self.target_theta = float(self.sim.clip_value(values[5], 0.0, self.max_tilt_theta))
        executed = [
            self.gravity,
            *[float(value) for value in normalized_xyz],
            self.target_phi,
            self.target_theta,
        ]
        self.executed_actions.append(executed)

        self._sync_arm_to_actual_paddle()
        target_mount, target_axis = self.sim.arm_paddle_mount_pose(
            self.target_position,
            self.sim.paddle_command_quaternion(
                self.target_phi, self.target_theta, self.max_tilt_theta
            ),
            self.paddle_spec,
        )
        self.joint_targets = self.sim.solve_arm_paddle_tip_ik(
            self.model,
            self.data,
            self.arm_spec,
            target_mount,
            target_axis,
            self.joint_targets,
        )
        self.sim.set_arm_joint_state(
            self.model,
            self.data,
            self.arm_spec,
            self.arm_joint_positions,
            self.zero_arm_velocities,
        )
        self.sim.set_named_arm_ctrl(self.model, self.data, self.arm_spec, self.joint_targets)
        interval = {
            key: bool(value)
            for key, value in self.sim.paddle_contact_flags(self.model, self.data).items()
        }
        for _ in range(self.steps_per_frame):
            self.sim.set_paddle_ctrl(
                self.data,
                self.target_position,
                self.paddle_spec,
                self.target_phi,
                self.target_theta,
            )
            self.sim.mujoco.mj_step(self.model, self.data)
            flags = self.sim.paddle_contact_flags(self.model, self.data)
            for name, value in flags.items():
                interval[name] = bool(interval[name] or value)
        counted = self.counter.observe(
            self.frame_idx,
            paddle_contact=interval["ball_paddle_contact"],
            floor_contact=interval["ball_floor_contact"],
            wall_contact=interval["ball_wall_contact"],
        )
        self.contact_intervals.append(dict(interval))
        self.frame_idx += 1
        self._last_arm_sync_frame = None
        self._record_vertical_state()
        return StepResult(
            frame_index=self.frame_idx,
            paddle_contact=interval["ball_paddle_contact"],
            counted_paddle_contact=counted,
            floor_contact=interval["ball_floor_contact"],
            wall_contact=interval["ball_wall_contact"],
            terminated=self.terminal,
        )

    def metrics(self, *, max_frames: int | None = None) -> dict[str, Any]:
        maximum = (
            self.max_transition_frames
            if max_frames is None
            else min(int(max_frames), self.max_transition_frames)
        )
        return episode_metrics(
            counter=self.counter,
            actions=self.executed_actions,
            frames_executed=self.frame_idx,
            max_frames=maximum,
            fps=self.fps,
            ball_z=self.ball_z_trace,
            paddle_z=self.paddle_z_trace,
            paddle_contact=[interval["ball_paddle_contact"] for interval in self.contact_intervals],
            floor_contact=[interval["ball_floor_contact"] for interval in self.contact_intervals],
            max_tilt_theta=self.max_tilt_theta,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> ArmPaddleBallEnv:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


__all__ = [
    "ArmPaddleBallEnv",
    "PaddleRuntime",
    "SimulatorRuntime",
    "StepResult",
]
