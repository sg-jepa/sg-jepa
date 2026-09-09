#!/usr/bin/env python3
"""Shared MuJoCo scene, controller, and logging for paddle-to-basket episodes.

The collision-bearing paddle remains a mocap body, exactly as in the source
``franka_paddle_intercept_ball`` task.  The rendered, collision-disabled Panda
tracks its rear handle through the source task's seven-joint IK.  Ball motion,
paddle impact, rim collisions, and the post-impact projectile are all advanced
by MuJoCo; no post-impact ball state is written by this module.
"""

from __future__ import annotations

import dataclasses
import io
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

import base_scene as base
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


FPS = base.FPS
FRAMES = base.FRAMES
WIDTH = base.WIDTH
HEIGHT = base.HEIGHT
STEP_RATE = base.STEP_RATE
STEPS_PER_FRAME = base.STEPS_PER_FRAME
DT = base.DT
BALL_RADIUS = base.BALL_RADIUS
BALL_MASS = base.BALL_MASS
PADDLE_HALF = base.PADDLE_HALF
# Keep the Panda hand visibly behind the blade in the side camera.  The source
# task's shorter mount projects the wrist mesh over the center of the paddle,
# which looks like a raised face even though the blade itself is flat.
HANDLE_LENGTH = 0.32
PADDLE_HANDLE_RADIUS = 0.018
PADDLE_ADAPTER_RADIUS = 0.031
PADDLE_ADAPTER_HALF_LENGTH = 0.018
# Keep a visible gap above the floor rather than merely preventing a negative
# z value.  This covers the complete tilted blade/mount envelope, not just the
# mocap body's origin at the blade center.
PADDLE_FLOOR_CLEARANCE_M = 0.012
ACTION_SCALE = base.ACTION_SCALE
HOME_QPOS = base.HOME_QPOS
JOINT_NAMES = base.JOINT_NAMES
PANDA_DIR = base.PANDA_DIR
PANDA_UPSTREAM = base.PANDA_UPSTREAM

STATE_SCHEMA = base.STATE_SCHEMA
ACTION_SCHEMA = base.ACTION_SCHEMA
PADDLE_STATE_SCHEMA = base.PADDLE_STATE_SCHEMA
ARM_STATE_SCHEMA = base.ARM_STATE_SCHEMA
CONTACT_SCHEMA = [
    "ball_paddle_blade_contact",
    "ball_floor_contact",
    "ball_wall_contact",
    "ball_basket_rim_contact",
    "ball_basket_bottom_contact",
    "invalid_paddle_or_robot_contact",
]
TASK_EVENT_SCHEMA = [
    "paddle_contact",
    "valid_paddle_blade_contact",
    "basket_entry",
    "rim_hit",
]
BASKET_STATE_SCHEMA = [
    "basket_x",
    "basket_y",
    "basket_rim_z",
    "basket_opening_radius",
    "basket_depth",
    "basket_rim_tube_radius",
]
STRIKE_STATE_SCHEMA = [
    *[f"predicted_intercept_{axis}" for axis in "xyz"],
    *[f"actual_contact_{axis}" for axis in "xyz"],
    *[f"incoming_velocity_{axis}" for axis in "xyz"],
    *[f"outgoing_velocity_{axis}" for axis in "xyz"],
    *[f"desired_outgoing_velocity_{axis}" for axis in "xyz"],
    *[f"desired_target_{axis}" for axis in "xyz"],
    *[f"blade_normal_{axis}" for axis in "xyz"],
    *[f"paddle_linear_velocity_{axis}" for axis in "xyz"],
    *[f"paddle_angular_velocity_{axis}" for axis in "xyz"],
    "impact_to_entry_or_closest_time",
    "minimum_distance_to_basket_opening",
]


@dataclasses.dataclass(frozen=True)
class BasketSpec:
    center_x: float = -0.59
    center_y: float = 0.0
    rim_z: float = 0.80
    opening_radius: float = 0.245
    depth: float = 0.25
    rim_tube_radius: float = 0.018

    @property
    def center(self) -> np.ndarray:
        return np.asarray([self.center_x, self.center_y, self.rim_z], dtype=np.float64)

    @property
    def entry_radius(self) -> float:
        return max(self.opening_radius - BALL_RADIUS, 0.02)

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.center_x,
                self.center_y,
                self.rim_z,
                self.opening_radius,
                self.depth,
                self.rim_tube_radius,
            ],
            dtype=np.float32,
        )


# Fixed across every training and test episode in v1. Keep
# this as one immutable task-level value so scene geometry, controller targets,
# logged basket state, and validation cannot silently drift apart.
FIXED_BASKET = BasketSpec()


@dataclasses.dataclass(frozen=True)
class EpisodeRequest:
    name: str
    label: str
    gravity: float
    planned_success: bool
    planned_detail: str
    hit_frame: int
    initial_vz: float
    ball_y: float
    initial_z: float
    basket: BasketSpec
    flight_time: float
    target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    impact_offset: tuple[float, float] = (0.0, 0.0)
    swing_gain: float = 1.0
    normal_correction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    roll: float = 0.0
    swing_lead_frames: int = 1
    follow_through_frames: int = 3
    minimum_swing_x: float = -0.22


@dataclasses.dataclass
class PreparedEpisode:
    request: EpisodeRequest
    initial_position: np.ndarray
    initial_velocity: np.ndarray
    probe_positions: np.ndarray
    probe_velocities: np.ndarray
    probe_bounces: list[int]
    predicted_intercept: np.ndarray
    desired_target: np.ndarray
    desired_outgoing_velocity: np.ndarray
    commands: np.ndarray
    rotations: np.ndarray
    quaternions: np.ndarray
    angles: np.ndarray
    floor_clearance_lifts: np.ndarray
    arm_targets: np.ndarray | None = None
    ik_position_errors: np.ndarray | None = None
    ik_angle_errors: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class ContactFlags:
    blade: bool = False
    floor: bool = False
    wall: bool = False
    rim: bool = False
    bottom: bool = False
    invalid: bool = False

    @property
    def paddle(self) -> bool:
        return self.blade or self.invalid

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.blade, self.floor, self.wall, self.rim, self.bottom, self.invalid],
            dtype=np.uint8,
        )


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _geom(root: ET.Element, name: str) -> ET.Element:
    for geom in root.iter("geom"):
        if geom.get("name") == name:
            return geom
    raise RuntimeError(f"missing geom {name!r}")


def _basket_customizer(basket: BasketSpec):
    def customize(root: ET.Element) -> None:
        asset = root.find("asset")
        worldbody = root.find("worldbody")
        contact = root.find("contact")
        assert asset is not None and worldbody is not None and contact is not None

        ET.SubElement(
            asset,
            "material",
            name="basket_rim_orange",
            rgba="1.0 0.48 0.04 1",
            specular="0.10",
            shininess="0.20",
        )
        ET.SubElement(
            asset,
            "material",
            name="basket_net_teal",
            rgba="0.12 0.82 0.70 0.62",
            specular="0.03",
        )
        ET.SubElement(
            asset,
            "material",
            name="basket_support_dark",
            rgba="0.10 0.14 0.17 1",
            specular="0.04",
        )

        camera = next(
            camera for camera in worldbody.findall("camera") if camera.get("name") == "observation"
        )
        camera.set("pos", "0.06 -2.72 1.08")
        for orientation_key in ("quat", "axisangle", "euler", "zaxis"):
            camera.attrib.pop(orientation_key, None)
        camera.set("xyaxes", "1 0 0 0 0.245 0.9695")
        camera.set("fovy", "45")

        # Keep invalid handle contacts observable while the Panda links and
        # adapter remain collision-disabled.  Such contacts never count as a
        # valid strike.
        handle = _geom(root, "paddle_handle")
        handle.set("contype", "1")
        handle.set("conaffinity", "1")
        adapter = _geom(root, "paddle_adapter")
        adapter.set("pos", _fmt([-HANDLE_LENGTH, 0.0, 0.0]))

        # The source interception task has a decorative target cylinder on the
        # blade face.  Remove it here so this task presents one flat blue
        # paddle while retaining the proven blade and handle geometry.
        paddle_body = next(body for body in root.iter("body") if body.get("name") == "paddle")
        for paddle_geom in list(paddle_body.findall("geom")):
            if paddle_geom.get("name") == "paddle_target_visual":
                paddle_body.remove(paddle_geom)
        ET.SubElement(
            paddle_body,
            "geom",
            name="paddle_mount_extension",
            type="capsule",
            fromto=_fmt([-base.HANDLE_LENGTH, 0.0, 0.0, -HANDLE_LENGTH, 0.0, 0.0]),
            size="0.018",
            material="handle_wood",
            contype="0",
            conaffinity="0",
        )

        basket_body = ET.SubElement(
            worldbody,
            "body",
            name="basket",
            pos=_fmt([basket.center_x, basket.center_y, basket.rim_z]),
        )
        segments = 20
        for index in range(segments):
            a0 = 2.0 * math.pi * index / segments
            a1 = 2.0 * math.pi * (index + 1) / segments
            p0 = [basket.opening_radius * math.cos(a0), basket.opening_radius * math.sin(a0), 0.0]
            p1 = [basket.opening_radius * math.cos(a1), basket.opening_radius * math.sin(a1), 0.0]
            ET.SubElement(
                basket_body,
                "geom",
                name=f"basket_rim_{index:02d}",
                type="capsule",
                fromto=_fmt([*p0, *p1]),
                size=f"{basket.rim_tube_radius:.10g}",
                material="basket_rim_orange",
                friction="0.22 0.004 0.0005",
                solimp="0.99 0.99 0.001",
            )

        bottom_radius = max(basket.opening_radius * 0.63, BALL_RADIUS * 1.4)
        bottom_z = -basket.depth
        ET.SubElement(
            basket_body,
            "geom",
            name="basket_bottom",
            type="cylinder",
            pos=_fmt([0.0, 0.0, bottom_z]),
            size=_fmt([bottom_radius, 0.014]),
            material="basket_net_teal",
            friction="0.32 0.006 0.0005",
        )
        # Physical tapered strands leave the opening unobstructed while making
        # every gap narrower than the ball diameter.  Entered balls are thus
        # contained instead of passing through a visual-only net.
        net_segments = 16
        for index in range(net_segments):
            angle = 2.0 * math.pi * index / net_segments
            top = [
                basket.opening_radius * math.cos(angle),
                basket.opening_radius * math.sin(angle),
                -0.012,
            ]
            bottom = [
                bottom_radius * math.cos(angle),
                bottom_radius * math.sin(angle),
                bottom_z,
            ]
            ET.SubElement(
                basket_body,
                "geom",
                name=f"basket_net_{index:02d}",
                type="capsule",
                fromto=_fmt([*top, *bottom]),
                size="0.008",
                material="basket_net_teal",
                contype="1",
                conaffinity="1",
                friction="0.30 0.006 0.0005",
                solimp="0.99 0.99 0.001",
            )

        # Run the pole from the floor to the rim and visibly brace it to the
        # lower basket.  The bracket is basket-relative, so random poses stay
        # connected without recompiling the model.
        support_height = max(basket.rim_z * 0.5, 0.05)
        ET.SubElement(
            worldbody,
            "geom",
            name="basket_support",
            type="box",
            pos=_fmt(
                [
                    basket.center_x - basket.opening_radius - 0.045,
                    basket.center_y,
                    support_height,
                ]
            ),
            size=_fmt([0.025, 0.025, support_height]),
            material="basket_support_dark",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            basket_body,
            "geom",
            name="basket_bracket",
            type="capsule",
            fromto=_fmt(
                [
                    -basket.opening_radius - 0.045,
                    0.0,
                    -0.012,
                    -basket.opening_radius * 0.70,
                    0.0,
                    -basket.depth * 0.46,
                ]
            ),
            size="0.018",
            material="basket_support_dark",
            contype="0",
            conaffinity="0",
        )

        ET.SubElement(
            contact,
            "pair",
            geom1="ball_geom",
            geom2="paddle_handle",
            solref=base._elastic_solref(0.45),
            solimp="0.99 0.99 0.001",
            friction="0.25 0.005 0.0005",
        )
        for index in range(segments):
            ET.SubElement(
                contact,
                "pair",
                geom1="ball_geom",
                geom2=f"basket_rim_{index:02d}",
                solref=base._elastic_solref(0.52),
                solimp="0.99 0.99 0.001",
                friction="0.20 0.004 0.0005",
            )
        ET.SubElement(
            contact,
            "pair",
            geom1="ball_geom",
            geom2="basket_bottom",
            solref=base._elastic_solref(0.25),
            solimp="0.99 0.99 0.001",
            friction="0.35 0.006 0.0005",
        )
        for index in range(net_segments):
            ET.SubElement(
                contact,
                "pair",
                geom1="ball_geom",
                geom2=f"basket_net_{index:02d}",
                solref=base._elastic_solref(0.18),
                solimp="0.99 0.99 0.001",
                friction="0.30 0.006 0.0005",
            )

    return customize


def build_model(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    gravity: float,
    basket: BasketSpec,
) -> mujoco.MjModel:
    # Compile through the source task first, then round-trip MuJoCo's canonical
    # XML and extend it.  This reuses the complete proven scene construction
    # without changing or duplicating the source task.
    source_model = base.build_model(initial_position, initial_velocity, gravity)
    descriptor, temporary_name = tempfile.mkstemp(prefix="franka_intercept_", suffix=".xml")
    os.close(descriptor)
    try:
        mujoco.mj_saveLastXML(temporary_name, source_model)
        root = ET.parse(temporary_name).getroot()
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str(base.ASSET_DIR))
    _basket_customizer(basket)(root)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.body_gravcomp[: source_model.nbody] = source_model.body_gravcomp
    return model


def set_basket_pose(model: mujoco.MjModel, basket: BasketSpec) -> None:
    """Move the compiled basket and its support without rebuilding the model."""
    basket_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "basket"))
    model.body_pos[basket_body] = basket.center
    support = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "basket_support"))
    support_half_height = max(basket.rim_z * 0.5, 0.05)
    model.geom_pos[support] = [
        basket.center_x - basket.opening_radius - 0.045,
        basket.center_y,
        support_half_height,
    ]
    model.geom_size[support, 2] = support_half_height


def reset_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    gravity: float,
    *,
    paddle_far: bool = False,
) -> None:
    model.opt.gravity[:] = [0.0, 0.0, -float(gravity)]
    mujoco.mj_resetData(model, data)
    ball_qpos = base._joint_qpos_adr(model, "ball_free")
    ball_dof = base._joint_dof_adr(model, "ball_free")
    data.qpos[ball_qpos : ball_qpos + 3] = initial_position
    data.qpos[ball_qpos + 3 : ball_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[ball_dof : ball_dof + 3] = initial_velocity
    base._set_arm_state(model, data, HOME_QPOS)
    if paddle_far:
        data.mocap_pos[0] = [1.8, 0.0, 1.4]
        data.mocap_quat[0] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def contact_flags(model: mujoco.MjModel, data: mujoco.MjData) -> ContactFlags:
    flags = dict(blade=False, floor=False, wall=False, rim=False, bottom=False, invalid=False)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
        }
        if "ball_geom" not in names:
            continue
        other = next(name for name in names if name != "ball_geom")
        flags["blade"] |= other == "paddle_blade"
        flags["floor"] |= other == "floor_geom"
        flags["wall"] |= other == "back_wall_visual"
        flags["rim"] |= other.startswith("basket_rim_")
        flags["bottom"] |= other == "basket_bottom"
        flags["invalid"] |= other in {"paddle_handle", "paddle_adapter"} or other.startswith("link")
    return ContactFlags(**flags)


def probe_ball(
    model: mujoco.MjModel,
    gravity: float,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    data = mujoco.MjData(model)
    reset_data(model, data, initial_position, initial_velocity, gravity, paddle_far=True)
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    bounces: list[int] = []
    floor_active = False
    ball_dof = base._joint_dof_adr(model, "ball_free")
    for frame in range(FRAMES):
        positions.append(data.body("ball").xpos.copy())
        velocities.append(data.qvel[ball_dof : ball_dof + 3].copy())
        for _ in range(STEPS_PER_FRAME):
            mujoco.mj_step(model, data)
            floor = contact_flags(model, data).floor
            if floor and not floor_active:
                bounces.append(frame)
            floor_active = floor
    return np.asarray(positions), np.asarray(velocities), bounces


def _rotation_from_normal(normal: np.ndarray, roll: float = 0.0) -> np.ndarray:
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(reference, normal))) > 0.94:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    local_z = reference - normal * float(np.dot(reference, normal))
    local_z /= np.linalg.norm(local_z)
    local_y = np.cross(local_z, normal)
    local_y /= np.linalg.norm(local_y)
    if roll:
        local_y, local_z = (
            math.cos(roll) * local_y + math.sin(roll) * local_z,
            -math.sin(roll) * local_y + math.cos(roll) * local_z,
        )
    return np.column_stack((normal, local_y, local_z))


def _normal_angles(normal: np.ndarray) -> tuple[float, float]:
    normal = np.asarray(normal, dtype=np.float64) / np.linalg.norm(normal)
    theta = math.acos(float(np.clip(-normal[0], -1.0, 1.0)))
    phi = math.atan2(float(normal[2]), float(normal[1]))
    return phi, theta


def desired_ballistic_velocity(
    intercept: np.ndarray,
    target: np.ndarray,
    gravity: float,
    flight_time: float,
) -> np.ndarray:
    gravity_vector = np.asarray([0.0, 0.0, -float(gravity)], dtype=np.float64)
    return (target - intercept - 0.5 * gravity_vector * flight_time * flight_time) / flight_time


def paddle_lowest_z(position: np.ndarray, rotation: np.ndarray) -> float:
    """Return the lowest world-z point of every visible paddle/mount geom."""
    position = np.asarray(position, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)

    # Exact axis-aligned projection of the oriented blade box onto world z.
    blade_half_height = float(np.dot(np.abs(rotation[2, :]), PADDLE_HALF))
    blade_lowest = float(position[2] - blade_half_height)

    # The handle and its visual extension form one local-x capsule from the
    # blade rear face to the adapter center.
    handle_endpoint_z = position[2] + rotation[2, 0] * np.asarray(
        [-PADDLE_HALF[0], -HANDLE_LENGTH], dtype=np.float64
    )
    handle_lowest = float(np.min(handle_endpoint_z) - PADDLE_HANDLE_RADIUS)

    # MuJoCo's adapter cylinder is aligned with local x.  Project its axial
    # half-length and circular cross-section separately onto world z.
    adapter_center_z = float(position[2] - HANDLE_LENGTH * rotation[2, 0])
    adapter_half_height = float(
        abs(rotation[2, 0]) * PADDLE_ADAPTER_HALF_LENGTH
        + np.linalg.norm(rotation[2, 1:3]) * PADDLE_ADAPTER_RADIUS
    )
    adapter_lowest = adapter_center_z - adapter_half_height
    return min(blade_lowest, handle_lowest, adapter_lowest)


def enforce_paddle_floor_clearance(
    commands: np.ndarray,
    rotations: np.ndarray,
    *,
    clearance: float = PADDLE_FLOOR_CLEARANCE_M,
) -> np.ndarray:
    """Lift each commanded pose enough to keep all paddle geometry clear."""
    lifts = np.zeros(len(commands), dtype=np.float64)
    for frame in range(len(commands)):
        required_lift = float(clearance) - paddle_lowest_z(commands[frame], rotations[frame])
        if required_lift > 0.0:
            commands[frame, 2] += required_lift
            lifts[frame] = required_lift
    return lifts


def build_prepared(
    model: mujoco.MjModel,
    request: EpisodeRequest,
    *,
    initial_x: float = -0.72,
    contact_x: float = 0.52,
    solve_arm: bool = False,
) -> PreparedEpisode:
    set_basket_pose(model, request.basket)
    hit_time = request.hit_frame / FPS
    vx = (contact_x - initial_x) / hit_time
    initial_position = np.asarray([initial_x, request.ball_y, request.initial_z], dtype=np.float64)
    initial_velocity = np.asarray([vx, 0.0, request.initial_vz], dtype=np.float64)
    probe_positions, probe_velocities, probe_bounces = probe_ball(
        model, request.gravity, initial_position, initial_velocity
    )
    intercept = probe_positions[request.hit_frame].copy()
    incoming = probe_velocities[request.hit_frame].copy()
    target = request.basket.center + np.asarray(request.target_offset, dtype=np.float64)
    desired_outgoing = desired_ballistic_velocity(
        intercept, target, request.gravity, request.flight_time
    )

    delta_velocity = desired_outgoing - incoming
    delta_velocity += np.asarray(request.normal_correction, dtype=np.float64)
    normal = delta_velocity / np.linalg.norm(delta_velocity)
    if normal[0] > -0.08:
        normal[0] = -0.08
        normal /= np.linalg.norm(normal)
    rotation = _rotation_from_normal(normal, request.roll)
    local_y = rotation[:, 1]
    local_z = rotation[:, 2]
    impact_y, impact_z = request.impact_offset
    face_distance = PADDLE_HALF[0] + BALL_RADIUS
    contact_center = (
        intercept - normal * face_distance - local_y * float(impact_y) - local_z * float(impact_z)
    )

    restitution = 0.92
    incoming_normal = float(np.dot(incoming, normal))
    desired_normal = float(np.dot(desired_outgoing, normal))
    paddle_normal_speed = (
        (desired_normal + restitution * incoming_normal) / (1.0 + restitution)
    ) * request.swing_gain
    # A genuine return must visibly move left; the small tangential component
    # preserves this even for steep gravity-aware normals.
    swing_velocity = paddle_normal_speed * normal
    minimum_swing_x = min(float(request.minimum_swing_x), -1.0e-4)
    if swing_velocity[0] > minimum_swing_x:
        swing_velocity[0] = minimum_swing_x

    commands = np.zeros((FRAMES, 3), dtype=np.float64)
    rotations = np.repeat(rotation[None, :, :], FRAMES, axis=0)
    # One frame of high-rate interpolation is enough to establish the impact
    # velocity (32 MuJoCo steps at 16 FPS) and avoids placing a steeply tilted
    # paddle/handle below the floor several frames before contact.
    swing_start = max(request.hit_frame - max(int(request.swing_lead_frames), 1), 1)
    high_low_gravity_contact = bool(request.gravity <= 1.0 and contact_center[2] > 0.90)
    # At the lowest test gravities the ball can still be near its apex at
    # impact.  Two post-impact swing frames keep these unusually high returns
    # inside the Franka workspace while retaining a clearly active follow-through.
    follow_through_frames = max(int(request.follow_through_frames), 1)
    if high_low_gravity_contact:
        follow_through_frames = min(follow_through_frames, 2)
    follow_end = min(request.hit_frame + follow_through_frames, FRAMES - 1)
    rest = contact_center - swing_velocity * ((request.hit_frame - swing_start) / FPS)
    rest[2] = max(rest[2], 0.20)
    for frame in range(FRAMES):
        if frame < swing_start:
            alpha = base._smoothstep(frame / max(swing_start - 5, 1))
            position = rest.copy()
            position[1] = (1.0 - alpha) * -0.16 + alpha * rest[1]
            position[2] = (1.0 - alpha) * 0.50 + alpha * rest[2]
        elif frame <= follow_end:
            position = contact_center + swing_velocity * ((frame - request.hit_frame) / FPS)
        else:
            follow_position = contact_center + swing_velocity * (
                (follow_end - request.hit_frame) / FPS
            )
            alpha = base._smoothstep((frame - follow_end) / 8.0)
            recovery_offset = (
                np.asarray([0.02, -0.10, -0.08])
                if high_low_gravity_contact
                else np.asarray([0.10, -0.10, 0.02])
            )
            position = (1.0 - alpha) * follow_position + alpha * (contact_center + recovery_offset)
        commands[frame] = position

    if request.planned_detail == "invalid_robot_contact":
        # Deliberately realize an exposed-handle strike instead of relying on
        # an accidental, gravity-sensitive miss around the blade.  The ball is
        # allowed to pass the blade plane while the mount is displaced upward;
        # during the final frame the rear handle moves down into the ball.  The
        # first offset controls that clear approach and the magnitude of the
        # second selects a contact point along the physical source handle.
        approach_offset = abs(float(request.impact_offset[0]))
        handle_depth = abs(float(request.impact_offset[1]))
        handle_contact_center = intercept + handle_depth * normal
        handle_approach_center = handle_contact_center + approach_offset * local_z
        commands[: request.hit_frame] = handle_approach_center
        commands[request.hit_frame :] = handle_contact_center
    elif request.planned_detail == "overshoot":
        # Extreme-gravity overshoots can otherwise place the one-frame windup
        # and late recovery below the robot's reachable workspace.  These
        # clamps affect only the pre-contact windup endpoint and post-contact
        # recovery; the actual contact center and physical collision remain
        # simulator-driven.
        commands[swing_start, 2] = max(commands[swing_start, 2], 0.20)
        commands[follow_end + 1 :, 2] = np.maximum(commands[follow_end + 1 :, 2], 0.10)

    # A center-height floor clamp is insufficient for a tilted 40 cm blade:
    # its lower corner or rear mount can still pass through z=0.  Constrain the
    # full oriented geometry after every outcome-specific trajectory edit.
    floor_clearance_lifts = enforce_paddle_floor_clearance(commands, rotations)

    quaternions = np.stack([base._matrix_to_quat_wxyz(item) for item in rotations])
    phi, theta = _normal_angles(normal)
    angles = np.repeat(np.asarray([[phi, theta]], dtype=np.float64), FRAMES, axis=0)
    prepared = PreparedEpisode(
        request=request,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        probe_positions=probe_positions,
        probe_velocities=probe_velocities,
        probe_bounces=probe_bounces,
        predicted_intercept=intercept,
        desired_target=target,
        desired_outgoing_velocity=desired_outgoing,
        commands=commands,
        rotations=rotations,
        quaternions=quaternions,
        angles=angles,
        floor_clearance_lifts=floor_clearance_lifts,
    )
    if solve_arm:
        attach_arm_trajectory(model, prepared)
    return prepared


def attach_arm_trajectory(model: mujoco.MjModel, prepared: PreparedEpisode) -> None:
    # The inherited solver targets the source mount length.  Offset only its
    # virtual blade centers so the solved hand pose lands at our longer rear
    # adapter; the physical blade commands and all logged paddle poses remain
    # unchanged.
    handle_axes = -prepared.rotations[:, :, 0]
    ik_commands = prepared.commands + ((HANDLE_LENGTH - base.HANDLE_LENGTH) * handle_axes)
    arm_targets, position_errors, angle_errors = base.solve_arm_trajectory(
        model, ik_commands, prepared.rotations
    )
    prepared.arm_targets = arm_targets
    prepared.ik_position_errors = position_errors
    prepared.ik_angle_errors = angle_errors


def _quat_angular_velocity(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    difference = np.zeros(3, dtype=np.float64)
    mujoco.mju_subQuat(difference, q1, q0)
    return difference * FPS


def _encode_jpeg(frame: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=95, subsampling=0)
    return buffer.getvalue()


def _opening_distance(position: np.ndarray, basket: BasketSpec) -> float:
    horizontal = float(np.linalg.norm(position[:2] - basket.center[:2]))
    horizontal_excess = max(horizontal - basket.entry_radius, 0.0)
    return math.hypot(horizontal_excess, float(position[2] - basket.rim_z))


def _contact_point_for_pair(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    allowed: set[str],
) -> np.ndarray | None:
    for index in range(data.ncon):
        contact = data.contact[index]
        names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
        }
        if "ball_geom" in names and names.intersection(allowed):
            return np.asarray(contact.pos, dtype=np.float64).copy()
    return None


def simulate(
    model: mujoco.MjModel,
    prepared: PreparedEpisode,
    *,
    renderer: mujoco.Renderer | None = None,
) -> tuple[dict, dict[str, np.ndarray], list[np.ndarray], list[bytes]]:
    request = prepared.request
    set_basket_pose(model, request.basket)
    if prepared.arm_targets is None:
        arm_targets = np.repeat(HOME_QPOS[None, :], FRAMES, axis=0)
        ik_position_errors = np.zeros(FRAMES, dtype=np.float64)
        ik_angle_errors = np.zeros(FRAMES, dtype=np.float64)
    else:
        arm_targets = prepared.arm_targets
        assert prepared.ik_position_errors is not None and prepared.ik_angle_errors is not None
        ik_position_errors = prepared.ik_position_errors
        ik_angle_errors = prepared.ik_angle_errors

    data = mujoco.MjData(model)
    reset_data(
        model,
        data,
        prepared.initial_position,
        prepared.initial_velocity,
        request.gravity,
    )
    base._set_arm_state(model, data, arm_targets[0])
    data.mocap_pos[0] = prepared.commands[0]
    data.mocap_quat[0] = prepared.quaternions[0]
    mujoco.mj_forward(model, data)

    states = np.zeros((FRAMES, len(STATE_SCHEMA)), dtype=np.float32)
    actions = np.zeros((FRAMES, len(ACTION_SCHEMA)), dtype=np.float32)
    paddle_states = np.zeros((FRAMES, len(PADDLE_STATE_SCHEMA)), dtype=np.float32)
    arm_states = np.zeros((FRAMES, len(ARM_STATE_SCHEMA)), dtype=np.float32)
    contacts = np.zeros((FRAMES, len(CONTACT_SCHEMA)), dtype=np.uint8)
    task_events = np.zeros((FRAMES, len(TASK_EVENT_SCHEMA)), dtype=np.uint8)
    frames: list[np.ndarray] = []
    pixels: list[bytes] = []
    ball_dof = base._joint_dof_adr(model, "ball_free")

    first_paddle_contact_step: int | None = None
    first_blade_contact_step: int | None = None
    first_blade_contact_frame: int | None = None
    basket_entry_step: int | None = None
    basket_entry_frame: int | None = None
    first_rim_step: int | None = None
    invalid_contact = False
    actual_contact_point: np.ndarray | None = None
    incoming_velocity: np.ndarray | None = None
    outgoing_velocity: np.ndarray | None = None
    blade_normal: np.ndarray | None = None
    paddle_linear_velocity: np.ndarray | None = None
    paddle_angular_velocity: np.ndarray | None = None
    opening_cross_position: np.ndarray | None = None
    min_opening_distance = float("inf")
    min_opening_step: int | None = None
    post_contact_min_x = float("inf")
    minimum_ball_z_after_entry = float("inf")
    basket_escape_after_entry = False
    bounce_steps: list[int] = []
    floor_active = False
    blade_active = False
    total_step = 0
    previous_ball_position = data.body("ball").xpos.copy()

    for frame in range(FRAMES):
        arm_velocity = (
            np.zeros(7) if frame == 0 else (arm_targets[frame] - arm_targets[frame - 1]) * FPS
        )
        base._set_arm_state(model, data, arm_targets[frame], arm_velocity)
        data.mocap_pos[0] = prepared.commands[frame]
        data.mocap_quat[0] = prepared.quaternions[frame]
        mujoco.mj_forward(model, data)
        if renderer is not None:
            renderer.update_scene(data, camera="observation")
            rendered = renderer.render().copy()
            frames.append(rendered)
            pixels.append(_encode_jpeg(rendered))

        states[frame] = base._ball_state(model, data)
        if frame < FRAMES - 1:
            delta = (prepared.commands[frame + 1] - prepared.commands[frame]) / ACTION_SCALE
            actions[frame] = [
                request.gravity,
                *np.clip(delta, -1.0, 1.0),
                *prepared.angles[frame + 1],
            ]
        frame_paddle_velocity = (
            np.zeros(3)
            if frame == 0
            else (prepared.commands[frame] - prepared.commands[frame - 1]) * FPS
        )
        paddle_states[frame] = [
            *prepared.commands[frame],
            *frame_paddle_velocity,
            *prepared.commands[frame],
        ]
        arm_states[frame] = [*arm_targets[frame], *arm_velocity, *arm_targets[frame]]

        interval_flags = np.zeros(len(CONTACT_SCHEMA), dtype=np.uint8)
        interval_events = np.zeros(len(TASK_EVENT_SCHEMA), dtype=np.uint8)
        if frame == FRAMES - 1:
            contacts[frame] = contact_flags(model, data).as_array()
            continue

        for substep in range(STEPS_PER_FRAME):
            alpha = (substep + 1) / STEPS_PER_FRAME
            paddle_position = (1.0 - alpha) * prepared.commands[frame] + alpha * prepared.commands[
                frame + 1
            ]
            paddle_quaternion = base._quat_nlerp(
                prepared.quaternions[frame], prepared.quaternions[frame + 1], alpha
            )
            arm_q = (1.0 - alpha) * arm_targets[frame] + alpha * arm_targets[frame + 1]
            arm_qvel = (arm_targets[frame + 1] - arm_targets[frame]) * FPS
            data.mocap_pos[0] = paddle_position
            data.mocap_quat[0] = paddle_quaternion
            base._set_arm_state(model, data, arm_q, arm_qvel)
            velocity_before_step = data.qvel[ball_dof : ball_dof + 3].copy()
            mujoco.mj_step(model, data)
            total_step += 1
            flags = contact_flags(model, data)
            interval_flags |= flags.as_array()
            interval_events[0] |= flags.paddle
            interval_events[1] |= flags.blade
            interval_events[3] |= flags.rim

            if flags.floor and not floor_active:
                bounce_steps.append(total_step)
            floor_active = flags.floor
            if flags.invalid:
                invalid_contact = True
            if flags.paddle and first_paddle_contact_step is None:
                first_paddle_contact_step = total_step
                actual_contact_point = _contact_point_for_pair(
                    model, data, {"paddle_blade", "paddle_handle", "paddle_adapter"}
                )
            if flags.blade and first_blade_contact_step is None:
                first_blade_contact_step = total_step
                first_blade_contact_frame = frame
                incoming_velocity = velocity_before_step
                blade_normal = prepared.rotations[frame][:, 0].copy()
                paddle_linear_velocity = (
                    prepared.commands[frame + 1] - prepared.commands[frame]
                ) * FPS
                paddle_angular_velocity = _quat_angular_velocity(
                    prepared.quaternions[frame], prepared.quaternions[frame + 1]
                )
                contact_point = _contact_point_for_pair(model, data, {"paddle_blade"})
                if contact_point is not None:
                    actual_contact_point = contact_point
            if flags.blade:
                outgoing_velocity = data.qvel[ball_dof : ball_dof + 3].copy()
            elif blade_active and first_blade_contact_step is not None:
                outgoing_velocity = data.qvel[ball_dof : ball_dof + 3].copy()
            blade_active = flags.blade
            if flags.rim and first_rim_step is None:
                first_rim_step = total_step

            ball_position = data.body("ball").xpos.copy()
            if first_blade_contact_step is not None:
                post_contact_min_x = min(post_contact_min_x, float(ball_position[0]))
                distance = _opening_distance(ball_position, request.basket)
                if distance < min_opening_distance:
                    min_opening_distance = distance
                    min_opening_step = total_step
                crossed_down = (
                    previous_ball_position[2] > request.basket.rim_z
                    and ball_position[2] <= request.basket.rim_z
                    and data.qvel[ball_dof + 2] < 0.0
                )
                if crossed_down and opening_cross_position is None:
                    fraction = (previous_ball_position[2] - request.basket.rim_z) / max(
                        previous_ball_position[2] - ball_position[2], 1.0e-12
                    )
                    opening_cross_position = previous_ball_position + fraction * (
                        ball_position - previous_ball_position
                    )
                    radial = float(
                        np.linalg.norm(opening_cross_position[:2] - request.basket.center[:2])
                    )
                    if radial <= request.basket.entry_radius:
                        basket_entry_step = total_step
                        basket_entry_frame = frame
                        interval_events[2] = 1
                if basket_entry_step is not None:
                    minimum_ball_z_after_entry = min(
                        minimum_ball_z_after_entry, float(ball_position[2])
                    )
                    radial_from_basket = float(
                        np.linalg.norm(ball_position[:2] - request.basket.center[:2])
                    )
                    basket_bottom_z = request.basket.rim_z - request.basket.depth
                    penetrated_bottom = ball_position[2] < basket_bottom_z - 0.01
                    escaped_side = (
                        ball_position[2] < request.basket.rim_z
                        and radial_from_basket > request.basket.opening_radius + BALL_RADIUS + 0.03
                    )
                    basket_escape_after_entry |= bool(penetrated_bottom or escaped_side)
            previous_ball_position = ball_position
        contacts[frame] = interval_flags
        task_events[frame] = interval_events

    actions[-1] = actions[-2]
    valid_blade_contact = first_blade_contact_step is not None
    basket_entry = basket_entry_step is not None
    success = bool(valid_blade_contact and basket_entry)
    ball_retained_after_entry = bool(basket_entry and not basket_escape_after_entry)
    if first_blade_contact_step is None:
        min_opening_distance = float(np.linalg.norm(data.body("ball").xpos - request.basket.center))
        min_opening_step = None

    if success:
        failure_reason = "none"
    elif not valid_blade_contact:
        failure_reason = "invalid_robot_contact" if invalid_contact else "no_paddle_contact"
    elif first_rim_step is not None:
        failure_reason = "rim_hit"
    elif opening_cross_position is None:
        failure_reason = (
            "undershoot"
            if post_contact_min_x > request.basket.center_x + request.basket.entry_radius
            else "other"
        )
    else:
        delta = opening_cross_position - request.basket.center
        if delta[0] > request.basket.entry_radius:
            failure_reason = "undershoot"
        elif delta[0] < -request.basket.entry_radius:
            failure_reason = "overshoot"
        elif abs(delta[1]) > request.basket.entry_radius:
            failure_reason = "left_right_miss"
        else:
            failure_reason = "other"

    pre_contact_limit = (
        first_blade_contact_step
        if first_blade_contact_step is not None
        else request.hit_frame * STEPS_PER_FRAME
    )
    pre_contact_bounces = sum(step < pre_contact_limit for step in bounce_steps)
    impact_reference = first_blade_contact_step
    event_reference = basket_entry_step if basket_entry_step is not None else min_opening_step
    impact_to_event = (
        0.0
        if impact_reference is None or event_reference is None
        else (event_reference - impact_reference) * DT
    )
    actual_contact = np.zeros(3) if actual_contact_point is None else actual_contact_point
    incoming = np.zeros(3) if incoming_velocity is None else incoming_velocity
    outgoing = np.zeros(3) if outgoing_velocity is None else outgoing_velocity
    normal = np.zeros(3) if blade_normal is None else blade_normal
    paddle_linear = np.zeros(3) if paddle_linear_velocity is None else paddle_linear_velocity
    paddle_angular = np.zeros(3) if paddle_angular_velocity is None else paddle_angular_velocity
    strike_state = np.asarray(
        [
            *prepared.predicted_intercept,
            *actual_contact,
            *incoming,
            *outgoing,
            *prepared.desired_outgoing_velocity,
            *prepared.desired_target,
            *normal,
            *paddle_linear,
            *paddle_angular,
            impact_to_event,
            min_opening_distance,
        ],
        dtype=np.float32,
    )
    if len(strike_state) != len(STRIKE_STATE_SCHEMA):
        raise RuntimeError("strike state/schema length mismatch")

    joint_ids = [base._joint_id(model, name) for name in JOINT_NAMES]
    q_margin = np.minimum(
        arm_targets - model.jnt_range[joint_ids, 0],
        model.jnt_range[joint_ids, 1] - arm_targets,
    )
    desired_apex_z = float(
        prepared.predicted_intercept[2]
        + max(float(prepared.desired_outgoing_velocity[2]), 0.0) ** 2
        / (2.0 * max(float(request.gravity), 1.0e-9))
    )
    minimum_paddle_floor_clearance = min(
        paddle_lowest_z(position, rotation)
        for position, rotation in zip(prepared.commands, prepared.rotations, strict=False)
    )
    metrics = {
        "name": request.name,
        "label": request.label,
        "gravity": request.gravity,
        "planned_success": request.planned_success,
        "planned_detail": request.planned_detail,
        "paddle_contact": first_paddle_contact_step is not None,
        "valid_paddle_blade_contact": valid_blade_contact,
        "basket_entry": basket_entry,
        "ball_retained_after_entry": ball_retained_after_entry,
        "basket_escape_after_entry": basket_escape_after_entry,
        "minimum_ball_z_after_entry": (
            None if not math.isfinite(minimum_ball_z_after_entry) else minimum_ball_z_after_entry
        ),
        "success": success,
        "failure_reason": failure_reason,
        "planned_hit_frame": request.hit_frame,
        "first_paddle_contact_frame": (
            None
            if first_paddle_contact_step is None
            else (first_paddle_contact_step - 1) // STEPS_PER_FRAME
        ),
        "first_blade_contact_frame": first_blade_contact_frame,
        "basket_entry_frame": basket_entry_frame,
        "rim_hit": first_rim_step is not None,
        "pre_contact_bounce_count": int(pre_contact_bounces),
        "basket": dataclasses.asdict(request.basket),
        "predicted_interception_point": prepared.predicted_intercept.tolist(),
        "actual_paddle_ball_contact_point": (
            None if actual_contact_point is None else actual_contact_point.tolist()
        ),
        "incoming_ball_velocity": (
            None if incoming_velocity is None else incoming_velocity.tolist()
        ),
        "outgoing_ball_velocity": (
            None if outgoing_velocity is None else outgoing_velocity.tolist()
        ),
        "desired_outgoing_ball_velocity": prepared.desired_outgoing_velocity.tolist(),
        "desired_basket_target_point": prepared.desired_target.tolist(),
        "paddle_blade_normal_at_impact": (None if blade_normal is None else blade_normal.tolist()),
        "paddle_linear_velocity_at_impact": (
            None if paddle_linear_velocity is None else paddle_linear_velocity.tolist()
        ),
        "paddle_angular_velocity_at_impact": (
            None if paddle_angular_velocity is None else paddle_angular_velocity.tolist()
        ),
        "desired_flight_time": request.flight_time,
        "desired_projectile_apex_z": desired_apex_z,
        "impact_to_entry_or_closest_time": impact_to_event,
        "minimum_distance_to_basket_opening": float(min_opening_distance),
        "opening_cross_position": (
            None if opening_cross_position is None else opening_cross_position.tolist()
        ),
        "post_contact_min_x": (
            None if not math.isfinite(post_contact_min_x) else post_contact_min_x
        ),
        "max_ik_position_error_m": float(np.max(ik_position_errors)),
        "max_ik_angle_error_deg": float(np.max(ik_angle_errors)),
        "min_joint_limit_margin_rad": float(np.min(q_margin)),
        "minimum_paddle_floor_clearance_m": float(minimum_paddle_floor_clearance),
        "maximum_paddle_floor_lift_m": float(np.max(prepared.floor_clearance_lifts)),
        "finite_arrays": bool(
            all(
                np.isfinite(array).all()
                for array in (
                    states,
                    actions,
                    paddle_states,
                    arm_states,
                    strike_state,
                )
            )
        ),
        "initial_position": prepared.initial_position.tolist(),
        "initial_velocity": prepared.initial_velocity.tolist(),
        "target_offset": list(request.target_offset),
        "impact_offset": list(request.impact_offset),
        "swing_gain": request.swing_gain,
        "normal_correction": list(request.normal_correction),
        "roll": request.roll,
        "swing_lead_frames": request.swing_lead_frames,
        "follow_through_frames": request.follow_through_frames,
        "minimum_swing_x": request.minimum_swing_x,
    }
    arrays = {
        "state": states,
        "action": actions,
        "reward": np.zeros(FRAMES, dtype=np.float32),
        "paddle_state": paddle_states,
        "arm_state": arm_states,
        "contact": contacts,
        "task_event": task_events,
        "basket_state": np.repeat(request.basket.as_array()[None, :], FRAMES, axis=0),
        "strike_state": np.repeat(strike_state[None, :], FRAMES, axis=0),
    }
    return metrics, arrays, frames, pixels


def candidate_requests(request: EpisodeRequest) -> list[EpisodeRequest]:
    """Return deterministic control candidates for outcome-conditioned search."""
    if request.planned_success:
        gains = (0.72, 0.84, 0.96, 1.08, 1.20, 1.34, 1.50)
        flight_scales = (0.88, 1.0, 1.12)
        corrections = (
            (0.0, 0.0, 0.0),
            (-0.25, 0.0, 0.0),
            (0.0, 0.0, 0.35),
            (0.0, 0.0, -0.30),
        )
        standard = [
            dataclasses.replace(
                request,
                swing_gain=gain,
                flight_time=request.flight_time * flight_scale,
                normal_correction=correction,
            )
            for flight_scale in flight_scales
            for correction in corrections
            for gain in gains
        ]
        if request.gravity > 1.0:
            high_gravity_fallbacks = []
            wide_high_gravity_fallbacks = []
            canonical_low_gravity_fallbacks = []
            if 2.5 <= request.gravity <= 3.0:
                # The g=2.7 test block spans many incoming seeds.  A centered
                # two-frame wind-up provides a seed-independent physical
                # return while desired_ballistic_velocity still uses the
                # episode's exact gravity.
                canonical_low_gravity_fallbacks = [
                    dataclasses.replace(
                        request,
                        hit_frame=28,
                        initial_vz=2.02,
                        initial_z=BALL_RADIUS + 0.009,
                        ball_y=0.0,
                        flight_time=1.00,
                        impact_offset=(0.0, 0.0),
                        swing_gain=1.34,
                        normal_correction=(0.0, 0.0, 0.0),
                        roll=0.0,
                        swing_lead_frames=2,
                        follow_through_frames=3,
                    )
                ]
            if request.gravity >= 12.0:
                # A three-frame wind-up gives the paddle enough travel to
                # establish a hard, visibly active return before contact.  A
                # short gravity-aware flight request supplies the vertical
                # speed required by the upper gravity tail without increasing
                # the horizontal gain until it skips over the basket.  This
                # family is continuous across the sampled training gravities,
                # rather than being tied to the 25 fixed test values.
                wide_high_gravity_fallbacks = [
                    dataclasses.replace(
                        request,
                        hit_frame=21,
                        initial_vz=4.50,
                        initial_z=BALL_RADIUS + 0.003,
                        ball_y=0.0,
                        flight_time=flight_time,
                        impact_offset=(0.0, 0.0),
                        swing_gain=1.20,
                        normal_correction=(0.0, 0.0, correction_z),
                        roll=0.0,
                        swing_lead_frames=3,
                        follow_through_frames=2,
                    )
                    for flight_time in (0.25, 0.28, 0.30, 0.32)
                    for correction_z in (0.0, 0.5, 1.0, 1.5, 2.0)
                ]
                # Canonical high-contact returns cover rare seeded trajectories
                # whose low bounce phase forces every standard strike onto the
                # floor-clearance boundary.  These still use MuJoCo contact and
                # the episode gravity; only the incoming phase/controller seed
                # is replaced after the diverse standard grid is exhausted.
                high_gravity_fallbacks = [
                    dataclasses.replace(
                        request,
                        hit_frame=hit_frame,
                        initial_vz=initial_vz,
                        initial_z=BALL_RADIUS + 0.003,
                        ball_y=0.0,
                        flight_time=flight_time,
                        impact_offset=(0.0, 0.0),
                        swing_gain=gain,
                        normal_correction=(0.0, 0.0, correction_z),
                        roll=0.0,
                        follow_through_frames=follow_through_frames,
                    )
                    for hit_frame, initial_vz, flight_time, gain, correction_z, follow_through_frames in (
                        (24, 4.50, 0.352, 1.50, 0.50, 3),
                        (24, 4.70, 0.360, 1.50, 0.50, 3),
                        (24, 4.30, 0.340, 1.50, 0.50, 3),
                        (24, 4.90, 0.380, 1.50, 0.50, 3),
                        (24, 4.50, 0.370, 1.30, 0.50, 3),
                        (24, 4.50, 0.340, 1.70, 0.30, 3),
                        (24, 5.00, 0.400, 1.50, 0.70, 3),
                        # At g=16 the hit-frame-24 bounce phase is nearly on
                        # the floor.  Intercept the preceding, higher bounce
                        # and stop after two follow-through frames: the ball
                        # has already separated, while a third frame would
                        # put only the visual arm's recovery outside its IK
                        # workspace.
                        (21, 4.50, 0.370, 1.50, 0.00, 2),
                    )
                ]
            return (
                standard
                + canonical_low_gravity_fallbacks
                + wide_high_gravity_fallbacks
                + high_gravity_fallbacks
            )
        # The 0.62 m/s^2 test tail can leave the incoming ball close to its
        # apex at contact.  Aim slightly above the opening with a shorter
        # flight request so the physical rebound descends through the basket
        # within the episode and remains reachable by the arm.
        low_gravity_fallbacks = [
            dataclasses.replace(
                request,
                flight_time=request.flight_time * flight_scale,
                target_offset=(target_x, 0.0, 0.10),
                swing_gain=gain,
            )
            for flight_scale in (0.50, 0.60)
            for target_x in (-0.10, 0.0, 0.10, 0.20)
            for gain in (0.50, 0.60, 0.70, 0.80, 0.90, 1.0)
        ]
        return standard + low_gravity_fallbacks
    if request.planned_detail == "no_paddle_contact":
        return [
            dataclasses.replace(request, impact_offset=(offset, 0.0))
            for offset in (0.24, 0.27, 0.30, 0.34, 0.40, 0.48)
        ]
    if request.planned_detail == "invalid_robot_contact":
        return [
            dataclasses.replace(request, impact_offset=(approach, -depth))
            for approach, depth in (
                (0.26, 0.13),
                (0.30, 0.13),
                (0.22, 0.11),
                (0.55, 0.14),
                (0.42, 0.12),
                (0.48, 0.13),
                (0.62, 0.16),
                (0.36, 0.10),
                (0.70, 0.18),
            )
        ]
    if request.planned_detail == "undershoot":
        standard = [
            dataclasses.replace(
                request,
                flight_time=request.flight_time * flight_scale,
                target_offset=(-0.05, 0.0, 0.0),
                swing_gain=gain,
            )
            for flight_scale in (0.81, 0.90, 1.0)
            for gain in (0.77, 0.80, 0.84, 0.88)
        ]
        # At very low gravity, even the standard reduced-gain strike can carry
        # the ball through a high basket.  These slower, shorter-flight
        # fallbacks preserve a genuine active return (vx < 0) while making the
        # ball land measurably in front of the opening.
        low_gravity_fallbacks = [
            dataclasses.replace(
                request,
                flight_time=request.flight_time * flight_scale,
                target_offset=(target_x, 0.0, 0.0),
                swing_gain=gain,
            )
            for target_x in (0.0, 0.10, 0.20)
            for flight_scale in (0.55, 0.65, 0.75)
            for gain in (0.50, 0.60, 0.68, 0.74)
        ]
        high_gravity_fallbacks = []
        if request.gravity >= 10.0:
            # A low high-gravity interception can ride the paddle-clearance
            # boundary.  A long, shallow glancing swing preserves a genuinely
            # active negative-x return while allowing the ball to fall short.
            high_gravity_fallbacks = [
                dataclasses.replace(
                    request,
                    target_offset=(target_x, 0.0, 0.10),
                    flight_time=request.flight_time * flight_scale,
                    swing_gain=gain,
                    normal_correction=(correction_x, 0.0, -1.0),
                    swing_lead_frames=6,
                    minimum_swing_x=-0.005,
                )
                for target_x in (1.0, 0.8, 1.2)
                for flight_scale in (0.75, 0.65, 0.85)
                for correction_x in (6.0, 5.0, 7.0)
                for gain in (0.05, 0.08, 0.12)
            ]
        return standard + low_gravity_fallbacks + high_gravity_fallbacks
    if request.planned_detail == "overshoot":
        standard = [
            dataclasses.replace(request, swing_gain=gain) for gain in (1.25, 1.30, 1.35, 1.45)
        ]
        # The full paddle-floor constraint can remove some vertical windup
        # from low interception poses.  Aim progressively beyond the far side
        # of the basket as a deterministic fallback so overshoots remain true
        # downward opening-plane crossings rather than arbitrary long returns.
        distance_fallbacks = [
            dataclasses.replace(
                request,
                target_offset=(target_x, 0.0, 0.0),
                flight_time=request.flight_time * flight_scale,
                swing_gain=gain,
            )
            for target_x in (-0.15, -0.30, -0.45)
            for flight_scale in (0.80, 0.90)
            for gain in (1.30, 1.50, 1.70)
        ]
        high_gravity_fallbacks = [
            dataclasses.replace(
                request,
                flight_time=request.flight_time * flight_scale,
                swing_gain=gain,
                normal_correction=(0.0, 0.0, correction_z),
            )
            for flight_scale in (0.80, 0.90)
            for correction_z in (-0.30, -0.50)
            for gain in (1.00, 1.10, 1.20, 1.30)
        ]
        wide_high_gravity_fallbacks = []
        if request.gravity >= 13.0:
            wide_high_gravity_fallbacks = [
                dataclasses.replace(
                    request,
                    hit_frame=21,
                    initial_vz=4.50,
                    initial_z=BALL_RADIUS + 0.003,
                    ball_y=0.0,
                    flight_time=flight_time,
                    target_offset=(target_x, 0.0, 0.0),
                    impact_offset=(0.0, 0.0),
                    swing_gain=1.20,
                    normal_correction=(0.0, 0.0, correction_z),
                    roll=0.0,
                    swing_lead_frames=3,
                    follow_through_frames=2,
                )
                for flight_time, correction_z, target_x in (
                    (0.25, 1.50, -0.20),
                    (0.28, 1.50, -0.25),
                    (0.30, 2.00, -0.30),
                    (0.20, 2.50, -0.30),
                )
            ]
        return standard + distance_fallbacks + wide_high_gravity_fallbacks + high_gravity_fallbacks
    if request.planned_detail == "left_right_miss":
        sign = 1.0 if request.target_offset[1] >= 0.0 else -1.0
        standard = [
            dataclasses.replace(request, target_offset=(0.0, sign * offset, 0.0), swing_gain=gain)
            for offset in (0.24, 0.30, 0.36)
            for gain in (0.85, 1.0, 1.15)
        ]
        canonical_high_gravity_fallbacks = []
        if request.gravity >= 12.0:
            # The floor-clearance constraint changes the low-bounce contact
            # phase at the upper test gravities.  Use the same two physically
            # validated high-contact returns as the success controller, then
            # add a controlled lateral aim error.  The ball still has to
            # cross the rim plane from above, outside the usable opening.
            canonical_high_gravity_fallbacks = [
                dataclasses.replace(
                    request,
                    hit_frame=hit_frame,
                    initial_vz=initial_vz,
                    initial_z=BALL_RADIUS + 0.003,
                    ball_y=0.0,
                    flight_time=flight_time,
                    target_offset=(target_x, sign * target_y, 0.0),
                    impact_offset=(0.0, 0.0),
                    swing_gain=gain,
                    normal_correction=(0.0, 0.0, correction_z),
                    roll=0.0,
                    follow_through_frames=follow_through_frames,
                )
                for hit_frame, initial_vz, flight_time, target_x, target_y, gain, correction_z, follow_through_frames in (
                    (24, 4.50, 0.352, 0.00, 0.38, 1.50, 0.50, 3),
                    (21, 4.50, 0.300, 0.10, 0.28, 1.00, 0.50, 2),
                )
            ]
            canonical_high_gravity_fallbacks.extend(
                dataclasses.replace(
                    request,
                    hit_frame=21,
                    initial_vz=4.50,
                    initial_z=BALL_RADIUS + 0.003,
                    ball_y=0.0,
                    flight_time=0.25,
                    target_offset=(0.20, sign * target_y, 0.0),
                    impact_offset=(0.0, 0.0),
                    swing_gain=1.20,
                    normal_correction=(0.0, 0.0, 1.0),
                    roll=0.0,
                    swing_lead_frames=3,
                    follow_through_frames=2,
                )
                for target_y in (0.28, 0.36, 0.45)
            )
            canonical_high_gravity_fallbacks.extend(
                dataclasses.replace(
                    request,
                    hit_frame=21,
                    initial_vz=4.50,
                    initial_z=BALL_RADIUS + 0.003,
                    ball_y=0.0,
                    flight_time=flight_time,
                    target_offset=(target_x, sign * target_y, 0.0),
                    impact_offset=(0.0, 0.0),
                    swing_gain=gain,
                    normal_correction=(0.0, 0.0, correction_z),
                    roll=0.0,
                    swing_lead_frames=3,
                    follow_through_frames=2,
                )
                for flight_time, gain, correction_z, target_x, target_y in (
                    (0.24, 1.00, 1.00, 0.00, 0.45),
                    (0.24, 1.00, 1.00, 0.00, 0.50),
                    (0.25, 1.20, 1.00, 0.10, 0.36),
                    (0.25, 1.20, 1.00, 0.20, 0.55),
                    (0.25, 1.20, 1.50, 0.20, 0.28),
                    (0.25, 1.20, 1.50, 0.20, 0.45),
                    (0.25, 1.20, 1.00, 0.00, 0.75),
                )
            )
        canonical_mid_gravity_fallbacks = []
        if 6.0 <= request.gravity <= 8.0:
            gravity_scale = math.sqrt(request.gravity / 7.141964318346963)
            canonical_mid_gravity_fallbacks = [
                dataclasses.replace(
                    request,
                    hit_frame=28,
                    initial_vz=3.30 * gravity_scale,
                    initial_z=BALL_RADIUS + 0.043,
                    ball_y=0.0,
                    flight_time=0.642 / gravity_scale,
                    target_offset=(target_x, sign * target_y, 0.0),
                    impact_offset=(0.0, 0.0),
                    swing_gain=1.47,
                    normal_correction=(-0.25, 0.0, 0.0),
                    roll=0.0,
                )
                for target_x, target_y in (
                    (-0.20, 0.50),
                    (0.00, 0.50),
                    (0.30, 0.60),
                )
            ]
        canonical_low_gravity_fallbacks = []
        if 2.5 <= request.gravity <= 3.0:
            # Around g=2.7 the generic lateral request can lose too much
            # forward speed and fall short before reaching rim height.  This
            # higher, longer-flight return crosses the opening plane from
            # above while remaining laterally outside it.
            canonical_low_gravity_fallbacks = [
                dataclasses.replace(
                    request,
                    hit_frame=28,
                    initial_vz=2.02,
                    initial_z=BALL_RADIUS + 0.044,
                    ball_y=0.0,
                    flight_time=1.107,
                    target_offset=(-0.10, sign * 0.38, 0.0),
                    impact_offset=(0.0, 0.0),
                    swing_gain=1.30,
                    normal_correction=(0.0, 0.0, 0.0),
                    roll=0.0,
                )
            ]
        high_gravity_fallbacks = [
            dataclasses.replace(
                request,
                target_offset=(target_x, sign * offset, 0.0),
                flight_time=request.flight_time * flight_scale,
                swing_gain=gain,
                normal_correction=(0.0, 0.0, correction_z),
            )
            for flight_scale in (0.70, 0.80, 0.90)
            for correction_z in (-0.50, -0.30)
            for target_x in (-0.10, 0.0)
            for offset in (0.36, 0.42, 0.48)
            for gain in (0.90, 1.0, 1.10)
        ]
        low_gravity_fallbacks = [
            dataclasses.replace(
                request,
                target_offset=(target_x, sign * offset, 0.0),
                flight_time=request.flight_time * flight_scale,
                swing_gain=gain,
            )
            for target_x in (0.20, 0.40, 0.60)
            for offset in (0.24, 0.30, 0.36)
            for flight_scale in (0.55, 0.70)
            for gain in (0.55, 0.70)
        ]
        return (
            standard
            + canonical_high_gravity_fallbacks
            + canonical_mid_gravity_fallbacks
            + canonical_low_gravity_fallbacks
            + high_gravity_fallbacks
            + low_gravity_fallbacks
        )
    if request.planned_detail == "rim_hit":
        standard = [
            dataclasses.replace(request, target_offset=(offset, 0.0, 0.0), swing_gain=gain)
            for offset in (0.08, 0.12, 0.16, 0.20, -0.12, -0.16)
            for gain in (0.82, 0.95, 1.08)
        ]
        # Shorter gravity-aware flight requests move the physical impact arc
        # toward the near rim when the floor-clearance lift removes part of a
        # low strike's vertical windup.  Keep these after the original grid so
        # common episodes retain their established controller distribution.
        timed_fallbacks = [
            dataclasses.replace(
                request,
                target_offset=(offset, 0.0, 0.0),
                flight_time=request.flight_time * flight_scale,
                swing_gain=gain,
            )
            for offset in (0.08, 0.12, 0.16, 0.20)
            for flight_scale in (0.75, 0.85, 0.95)
            for gain in (1.35, 1.20, 1.50)
        ]
        wide_high_gravity_fallbacks = []
        if request.gravity >= 13.0:
            wide_high_gravity_fallbacks = [
                dataclasses.replace(
                    request,
                    hit_frame=21,
                    initial_vz=4.50,
                    initial_z=BALL_RADIUS + 0.003,
                    ball_y=0.0,
                    flight_time=0.20,
                    target_offset=(target_x, target_y, 0.0),
                    impact_offset=(0.0, 0.0),
                    swing_gain=1.20,
                    normal_correction=(0.0, 0.0, correction_z),
                    roll=0.0,
                    swing_lead_frames=3,
                    follow_through_frames=2,
                )
                for target_x, target_y, correction_z in (
                    (0.05, 0.00, 0.00),
                    (0.08, 0.00, 0.50),
                    (0.05, 0.04, 0.00),
                    (0.05, -0.04, 0.00),
                )
            ]
        return standard + timed_fallbacks + wide_high_gravity_fallbacks
    return [request]


def outcome_matches(metrics: dict, request: EpisodeRequest) -> bool:
    incoming = metrics.get("incoming_ball_velocity")
    outgoing = metrics.get("outgoing_ball_velocity")
    paddle = metrics.get("paddle_linear_velocity_at_impact")
    active_return = bool(
        metrics.get("valid_paddle_blade_contact")
        and incoming is not None
        and outgoing is not None
        and paddle is not None
        and incoming[0] > 0.0
        and outgoing[0] < 0.0
        and paddle[0] < 0.0
    )
    clearance_ok = bool(
        metrics.get("minimum_paddle_floor_clearance_m", float("-inf"))
        >= PADDLE_FLOOR_CLEARANCE_M - 1.0e-8
    )
    if not clearance_ok:
        return False
    if request.planned_success:
        return bool(metrics["success"] and active_return and metrics["basket_entry"])
    if metrics["success"]:
        return False
    detail = request.planned_detail
    if detail == "no_paddle_contact":
        return not metrics["paddle_contact"]
    if detail == "invalid_robot_contact":
        return metrics["failure_reason"] == "invalid_robot_contact"
    if detail == "rim_hit":
        return active_return and metrics["rim_hit"]
    if detail in {"undershoot", "overshoot", "left_right_miss"}:
        return active_return and metrics["failure_reason"] == detail
    return not metrics["success"]


def prepare_matching_episode(
    model: mujoco.MjModel,
    request: EpisodeRequest,
    *,
    require_ik: bool = True,
) -> tuple[PreparedEpisode, dict]:
    last_metrics: dict | None = None
    for candidate in candidate_requests(request):
        prepared = build_prepared(model, candidate, solve_arm=False)
        metrics, _, _, _ = simulate(model, prepared, renderer=None)
        last_metrics = metrics
        if not outcome_matches(metrics, candidate):
            continue
        if require_ik:
            attach_arm_trajectory(model, prepared)
            assert prepared.ik_position_errors is not None and prepared.ik_angle_errors is not None
            if float(np.max(prepared.ik_position_errors)) > 0.005:
                continue
            if float(np.max(prepared.ik_angle_errors)) > 5.0:
                continue
            metrics, _, _, _ = simulate(model, prepared, renderer=None)
            if not outcome_matches(metrics, candidate):
                continue
        return prepared, metrics
    summary = (
        "none"
        if last_metrics is None
        else json.dumps(
            {
                key: last_metrics.get(key)
                for key in (
                    "paddle_contact",
                    "valid_paddle_blade_contact",
                    "basket_entry",
                    "failure_reason",
                    "minimum_distance_to_basket_opening",
                    "opening_cross_position",
                )
            },
            sort_keys=True,
        )
    )
    raise RuntimeError(f"unable to realize {request.name}: last={summary}")
