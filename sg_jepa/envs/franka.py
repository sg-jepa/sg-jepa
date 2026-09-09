"""Closed-loop MuJoCo adapter for the Franka basket task."""

from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from data_generation.common.assets import verify_menagerie_component
from data_generation.franka.generator import PANDA_UPSTREAM_TEXT, _engine_root
from sg_jepa.control.actions import full_range_translation_delta

IK_POSITION_TOLERANCE_M = 0.005
IK_ANGLE_TOLERANCE_DEG = 5.0
IK_PROJECTION_ITERATIONS = 12
CONTROL_CONTRACT = "full_range_pose_delta_v2"


def _load_franka_task(runtime: Path):
    sys.path.insert(0, str(runtime))
    for name in ("base_scene", "basket_task"):
        sys.modules.pop(name, None)
    return importlib.import_module("basket_task")


def _normal_from_angles(phi: float, theta: float) -> np.ndarray:
    sine = math.sin(float(theta))
    normal = np.asarray(
        [-math.cos(float(theta)), sine * math.cos(float(phi)), sine * math.sin(float(phi))],
        dtype=np.float64,
    )
    return normal / np.linalg.norm(normal)


def _ball_basket_contact_geoms(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[str, ...]:
    result: list[str] = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        if first == "ball_geom":
            other = second or ""
        elif second == "ball_geom":
            other = first or ""
        else:
            continue
        if (
            other == "basket_bottom"
            or other.startswith("basket_rim_")
            or other.startswith("basket_net_")
        ):
            result.append(other)
    return tuple(result)


class BasketRuntime:
    """One compiled scene and renderer reused by sequential episodes."""

    def __init__(
        self,
        menagerie_root: str | Path,
        *,
        width: int = 256,
        height: int = 256,
    ) -> None:
        menagerie_root = Path(menagerie_root).expanduser().resolve()
        verify_menagerie_component(menagerie_root, "franka_emika_panda")
        self._temporary = tempfile.TemporaryDirectory(prefix="sgjepa-franka-eval-")
        runtime = Path(self._temporary.name) / "engine"
        shutil.copytree(_engine_root(), runtime)
        target = runtime / "third_party" / "mujoco_menagerie" / "franka_emika_panda"
        target.parent.mkdir(parents=True)
        shutil.copytree(menagerie_root / "franka_emika_panda", target)
        (target / "UPSTREAM.json").write_text(PANDA_UPSTREAM_TEXT)
        self._runtime = runtime
        self.task = _load_franka_task(runtime)
        self.basket = self.task.FIXED_BASKET
        self.model = self.task.build_model(
            np.asarray([-0.72, 0.0, self.task.BALL_RADIUS + 0.01]),
            np.zeros(3),
            9.8,
            self.basket,
        )
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.ik_data = mujoco.MjData(self.model)

    def close(self) -> None:
        self.renderer.close()
        if str(self._runtime) in sys.path:
            sys.path.remove(str(self._runtime))
        for name in ("base_scene", "basket_task"):
            sys.modules.pop(name, None)
        self._temporary.cleanup()

    def __enter__(self) -> BasketRuntime:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class BasketSimulator:
    """Execute full-range ``[g,dx,dy,dz,phi,theta]`` actions at 16 FPS."""

    def __init__(self, record: dict[str, Any], *, runtime: BasketRuntime) -> None:
        self.record = record
        self.runtime = runtime
        self.task = runtime.task
        self.model = runtime.model
        self.data = mujoco.MjData(self.model)
        self.gravity = float(record["gravity"])
        self.metadata = (
            json.loads(record["episode_metadata"])
            if isinstance(record["episode_metadata"], str)
            else dict(record["episode_metadata"])
        )
        self.basket = self.task.BasketSpec(*map(float, record["basket_state"]))
        if not np.allclose(self.basket.as_array(), self.task.FIXED_BASKET.as_array(), atol=1.0e-6):
            raise ValueError("evaluation record does not use the locked v1 basket")
        state = np.asarray(record["state"], dtype=np.float64)
        paddle = np.asarray(record["paddle_state"], dtype=np.float64)
        arm = np.asarray(record["arm_state"], dtype=np.float64)
        first_action = np.asarray(record["action"], dtype=np.float64)
        self.task.set_basket_pose(self.model, self.basket)
        self.task.reset_data(self.model, self.data, state[:3], state[3:6], self.gravity)
        ball_qpos = self.task.base._joint_qpos_adr(self.model, "ball_free")
        ball_dof = self.task.base._joint_dof_adr(self.model, "ball_free")
        self.data.qpos[ball_qpos + 3 : ball_qpos + 7] = [
            state[9],
            state[6],
            state[7],
            state[8],
        ]
        self.data.qvel[ball_dof + 3 : ball_dof + 6] = state[10:13]
        self.arm_q = arm[:7].copy()
        self.arm_joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.task.JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self.minimum_joint_limit_margin = self._joint_limit_margin(self.arm_q)
        self.task.base._set_arm_state(self.model, self.data, self.arm_q)
        self.roll = float(self.metadata["controller"].get("roll", 0.0))
        self.position = paddle[:3].copy()
        self.rotation = self.task._rotation_from_normal(
            _normal_from_angles(first_action[4], first_action[5]), self.roll
        )
        self.quaternion = self.task.base._matrix_to_quat_wxyz(self.rotation)
        self.data.mocap_pos[0] = self.position
        self.data.mocap_quat[0] = self.quaternion
        mujoco.mj_forward(self.model, self.data)

        self.frame_idx = 0
        self.total_steps = 0
        self.executed_actions: list[np.ndarray] = []
        self.first_paddle_contact_step: int | None = None
        self.first_blade_contact_step: int | None = None
        self.first_blade_contact_frame: int | None = None
        self.basket_entry_step: int | None = None
        self.basket_entry_frame: int | None = None
        self.first_rim_step: int | None = None
        self.first_post_strike_basket_contact_step: int | None = None
        self.first_post_strike_basket_contact_frame: int | None = None
        self.first_post_strike_basket_contact_geom: str | None = None
        self.post_strike_rim_contact = False
        self.post_strike_net_contact = False
        self.post_strike_bottom_contact = False
        self.invalid_contact = False
        self.bounce_steps: list[int] = []
        self.floor_active = False
        self.previous_ball_position = self.data.body("ball").xpos.copy()
        self.minimum_opening_distance = float("inf")
        self.opening_cross_position: np.ndarray | None = None
        self.post_contact_min_x = float("inf")
        self.basket_escape_after_entry = False
        self.minimum_ball_z_after_entry = float("inf")
        self.max_ik_position_error = 0.0
        self.max_ik_angle_error_deg = 0.0
        self.max_requested_ik_position_error = 0.0
        self.max_requested_ik_angle_error_deg = 0.0
        self.reachability_fractions: list[float] = []
        self.max_requested_translation_action = 0.0
        self.minimum_paddle_floor_clearance = self.task.paddle_lowest_z(
            self.position, self.rotation
        )

    @property
    def fps(self) -> int:
        return int(self.task.FPS)

    def render_current(self) -> np.ndarray:
        self.runtime.renderer.update_scene(self.data, camera="observation")
        return self.runtime.renderer.render().copy()

    def _ik_solution(
        self, position: np.ndarray, rotation: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        handle_axis = -rotation[:, 0]
        ik_command = position + (
            (self.task.HANDLE_LENGTH - self.task.base.HANDLE_LENGTH) * handle_axis
        )
        target_position, target_rotation = self.task.base._hand_target(ik_command, rotation)
        q, position_error, angle_error = self.task.base.solve_ik(
            self.model,
            self.runtime.ik_data,
            target_position,
            target_rotation,
            self.arm_q,
        )
        return q, float(position_error), float(angle_error)

    def _joint_limit_margin(self, q: np.ndarray) -> float:
        lower = self.model.jnt_range[self.arm_joint_ids, 0]
        upper = self.model.jnt_range[self.arm_joint_ids, 1]
        return float(np.min(np.minimum(np.asarray(q) - lower, upper - np.asarray(q))))

    @staticmethod
    def _rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
        flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(flat, np.asarray(quaternion, dtype=np.float64))
        return flat.reshape(3, 3)

    def _project_reachable_pose(
        self, requested_position: np.ndarray, requested_rotation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        requested_q, position_error, angle_error = self._ik_solution(
            requested_position, requested_rotation
        )
        self.max_requested_ik_position_error = max(
            self.max_requested_ik_position_error, position_error
        )
        self.max_requested_ik_angle_error_deg = max(
            self.max_requested_ik_angle_error_deg, angle_error
        )
        if position_error <= IK_POSITION_TOLERANCE_M and angle_error <= IK_ANGLE_TOLERANCE_DEG:
            self.max_ik_position_error = max(self.max_ik_position_error, position_error)
            self.max_ik_angle_error_deg = max(self.max_ik_angle_error_deg, angle_error)
            return requested_position, requested_rotation, requested_q, 1.0

        requested_quaternion = self.task.base._matrix_to_quat_wxyz(requested_rotation)
        low, high = 0.0, 1.0
        best_position = self.position.copy()
        best_rotation = self.rotation.copy()
        best_q = self.arm_q.copy()
        best_position_error = 0.0
        best_angle_error = 0.0
        for _ in range(IK_PROJECTION_ITERATIONS):
            alpha = 0.5 * (low + high)
            position = (1.0 - alpha) * self.position + alpha * requested_position
            quaternion = self.task.base._quat_nlerp(self.quaternion, requested_quaternion, alpha)
            rotation = self._rotation_from_quaternion(quaternion)
            lifted = position[None].copy()
            self.task.enforce_paddle_floor_clearance(lifted, rotation[None])
            position = lifted[0]
            q, candidate_position_error, candidate_angle_error = self._ik_solution(
                position, rotation
            )
            feasible = (
                candidate_position_error <= IK_POSITION_TOLERANCE_M
                and candidate_angle_error <= IK_ANGLE_TOLERANCE_DEG
            )
            if feasible:
                low = alpha
                best_position = position
                best_rotation = rotation
                best_q = q
                best_position_error = candidate_position_error
                best_angle_error = candidate_angle_error
            else:
                high = alpha
        self.max_ik_position_error = max(self.max_ik_position_error, best_position_error)
        self.max_ik_angle_error_deg = max(self.max_ik_angle_error_deg, best_angle_error)
        return best_position, best_rotation, best_q, low

    def step(self, raw_action: np.ndarray) -> None:
        """Execute a corrected full-range Cartesian delta without legacy clipping."""

        action = np.asarray(raw_action, dtype=np.float32).reshape(6).copy()
        if not np.isclose(action[0], self.gravity, atol=1.0e-6, rtol=0.0):
            raise ValueError("action gravity differs from evaluation episode")
        if not np.isfinite(action).all():
            raise ValueError("action contains a non-finite value")
        action[0] = self.gravity
        self.max_requested_translation_action = max(
            self.max_requested_translation_action, float(np.max(np.abs(action[1:4])))
        )
        # Match the historical corrected-action adapter exactly: map the
        # direction into the legacy unit cube and enlarge ACTION_SCALE by the
        # same factor. The product is the original full-range delta.
        requested_position = self.position + full_range_translation_delta(
            action[1:4], np.asarray(self.task.ACTION_SCALE, dtype=np.float64)
        )
        requested_rotation = self.task._rotation_from_normal(
            _normal_from_angles(action[4], action[5]), self.roll
        )
        single_position = requested_position[None].copy()
        self.task.enforce_paddle_floor_clearance(single_position, requested_rotation[None])
        requested_position = single_position[0]
        next_position, next_rotation, next_arm_q, reachability_fraction = (
            self._project_reachable_pose(requested_position, requested_rotation)
        )
        self.reachability_fractions.append(reachability_fraction)
        next_quaternion = self.task.base._matrix_to_quat_wxyz(next_rotation)
        ball_dof = self.task.base._joint_dof_adr(self.model, "ball_free")
        for substep in range(self.task.STEPS_PER_FRAME):
            alpha = (substep + 1) / self.task.STEPS_PER_FRAME
            self.data.mocap_pos[0] = (1.0 - alpha) * self.position + alpha * next_position
            self.data.mocap_quat[0] = self.task.base._quat_nlerp(
                self.quaternion, next_quaternion, alpha
            )
            arm_q = (1.0 - alpha) * self.arm_q + alpha * next_arm_q
            arm_qvel = (next_arm_q - self.arm_q) * self.fps
            self.task.base._set_arm_state(self.model, self.data, arm_q, arm_qvel)
            mujoco.mj_step(self.model, self.data)
            self.total_steps += 1
            flags = self.task.contact_flags(self.model, self.data)
            if flags.floor and not self.floor_active:
                self.bounce_steps.append(self.total_steps)
            self.floor_active = flags.floor
            self.invalid_contact |= flags.invalid
            if flags.paddle and self.first_paddle_contact_step is None:
                self.first_paddle_contact_step = self.total_steps
            if flags.blade and self.first_blade_contact_step is None:
                self.first_blade_contact_step = self.total_steps
                self.first_blade_contact_frame = self.frame_idx
            if flags.rim and self.first_rim_step is None:
                self.first_rim_step = self.total_steps
            if self.first_blade_contact_step is not None:
                basket_geoms = _ball_basket_contact_geoms(self.model, self.data)
                if basket_geoms:
                    self.post_strike_rim_contact |= any(
                        name.startswith("basket_rim_") for name in basket_geoms
                    )
                    self.post_strike_net_contact |= any(
                        name.startswith("basket_net_") for name in basket_geoms
                    )
                    self.post_strike_bottom_contact |= "basket_bottom" in basket_geoms
                    if self.first_post_strike_basket_contact_step is None:
                        self.first_post_strike_basket_contact_step = self.total_steps
                        self.first_post_strike_basket_contact_frame = self.frame_idx
                        self.first_post_strike_basket_contact_geom = basket_geoms[0]
            ball_position = self.data.body("ball").xpos.copy()
            if self.first_blade_contact_step is not None:
                self.post_contact_min_x = min(self.post_contact_min_x, float(ball_position[0]))
                self.minimum_opening_distance = min(
                    self.minimum_opening_distance,
                    self.task._opening_distance(ball_position, self.basket),
                )
                crossed_down = (
                    self.previous_ball_position[2] > self.basket.rim_z
                    and ball_position[2] <= self.basket.rim_z
                    and self.data.qvel[ball_dof + 2] < 0.0
                )
                if crossed_down and self.opening_cross_position is None:
                    fraction = (self.previous_ball_position[2] - self.basket.rim_z) / max(
                        self.previous_ball_position[2] - ball_position[2], 1.0e-12
                    )
                    self.opening_cross_position = self.previous_ball_position + fraction * (
                        ball_position - self.previous_ball_position
                    )
                    radial = np.linalg.norm(
                        self.opening_cross_position[:2] - self.basket.center[:2]
                    )
                    if radial <= self.basket.entry_radius:
                        self.basket_entry_step = self.total_steps
                        self.basket_entry_frame = self.frame_idx
                if self.basket_entry_step is not None:
                    self.minimum_ball_z_after_entry = min(
                        self.minimum_ball_z_after_entry, float(ball_position[2])
                    )
                    radial = np.linalg.norm(ball_position[:2] - self.basket.center[:2])
                    penetrated = ball_position[2] < self.basket.rim_z - self.basket.depth - 0.01
                    escaped = (
                        ball_position[2] < self.basket.rim_z
                        and radial > self.basket.opening_radius + self.task.BALL_RADIUS + 0.03
                    )
                    self.basket_escape_after_entry |= bool(penetrated or escaped)
            self.previous_ball_position = ball_position
        self.position = next_position
        self.rotation = next_rotation
        self.quaternion = next_quaternion
        self.arm_q = next_arm_q
        self.minimum_joint_limit_margin = min(
            self.minimum_joint_limit_margin, self._joint_limit_margin(self.arm_q)
        )
        self.frame_idx += 1
        self.executed_actions.append(action)
        self.minimum_paddle_floor_clearance = min(
            self.minimum_paddle_floor_clearance,
            self.task.paddle_lowest_z(self.position, self.rotation),
        )

    def result(self) -> dict[str, Any]:
        valid_contact = self.first_blade_contact_step is not None
        entry = self.basket_entry_step is not None
        success = bool(valid_contact and entry)
        basket_contact = self.first_post_strike_basket_contact_step is not None
        relaxed_success = bool(valid_contact and (entry or basket_contact))
        if success:
            reason = "none"
        elif not valid_contact:
            reason = "invalid_robot_contact" if self.invalid_contact else "no_paddle_contact"
        elif self.first_rim_step is not None:
            reason = "rim_hit"
        elif self.opening_cross_position is not None:
            delta = self.opening_cross_position - self.basket.center
            reason = "left_right_miss" if abs(delta[1]) > self.basket.entry_radius else "other"
        elif self.post_contact_min_x > self.basket.center_x + self.basket.entry_radius:
            reason = "undershoot"
        else:
            reason = "overshoot"
        return {
            "success": success,
            "strict_success": success,
            "relaxed_success": relaxed_success,
            "failure_reason": reason,
            "paddle_contact": self.first_paddle_contact_step is not None,
            "valid_paddle_blade_contact": valid_contact,
            "basket_entry": entry,
            "rim_hit": self.first_rim_step is not None,
            "invalid_robot_contact": self.invalid_contact,
            "post_strike_basket_contact": basket_contact,
            "post_strike_rim_contact": self.post_strike_rim_contact,
            "post_strike_net_contact": self.post_strike_net_contact,
            "post_strike_bottom_contact": self.post_strike_bottom_contact,
            "first_post_strike_basket_contact_frame": self.first_post_strike_basket_contact_frame,
            "first_post_strike_basket_contact_geom": self.first_post_strike_basket_contact_geom,
            "ball_retained_after_entry": bool(entry and not self.basket_escape_after_entry),
            "basket_escape_after_entry": self.basket_escape_after_entry,
            "first_blade_contact_frame": self.first_blade_contact_frame,
            "basket_entry_frame": self.basket_entry_frame,
            "pre_contact_bounce_count": sum(
                step < (self.first_blade_contact_step or math.inf) for step in self.bounce_steps
            ),
            "minimum_distance_to_basket_opening": (
                None
                if not math.isfinite(self.minimum_opening_distance)
                else self.minimum_opening_distance
            ),
            "minimum_paddle_floor_clearance_m": self.minimum_paddle_floor_clearance,
            "max_ik_position_error_m": self.max_ik_position_error,
            "max_ik_angle_error_deg": self.max_ik_angle_error_deg,
            "max_requested_ik_position_error_m": self.max_requested_ik_position_error,
            "max_requested_ik_angle_error_deg": self.max_requested_ik_angle_error_deg,
            "minimum_joint_limit_margin_rad": self.minimum_joint_limit_margin,
            "projected_action_count": sum(
                fraction < 1.0 for fraction in self.reachability_fractions
            ),
            "minimum_reachability_fraction": min(self.reachability_fractions, default=1.0),
            "mean_reachability_fraction": float(
                np.mean(self.reachability_fractions) if self.reachability_fractions else 1.0
            ),
            "executed_action_count": len(self.executed_actions),
            "final_frame": self.frame_idx,
            "policy_control_contract": CONTROL_CONTRACT,
            "legacy_translation_clip_removed": True,
            "max_requested_translation_action": self.max_requested_translation_action,
        }


__all__ = [
    "BasketRuntime",
    "BasketSimulator",
    "CONTROL_CONTRACT",
]
