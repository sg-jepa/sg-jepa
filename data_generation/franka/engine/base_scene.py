#!/usr/bin/env python3
"""Shared Franka scene construction and inverse-kinematics utilities.

The paddle is a collision-bearing mocap body.  A collision-disabled Franka
Panda follows the rigid handle mount with seven-joint damped-least-squares IK.
This deliberately matches the kinematically decoupled architecture of the
existing arm_paddle_ball dataset: contact remains stable and deterministic,
while the rendered robot visibly carries the blade.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

import mujoco
import numpy as np

FPS = 16
FRAMES = 64
WIDTH = 256
HEIGHT = 256
GRAVITY = 9.8
STEP_RATE = 512
STEPS_PER_FRAME = STEP_RATE // FPS
DT = 1.0 / STEP_RATE

BALL_RADIUS = 0.075
BALL_MASS = 0.10
PADDLE_HALF = np.array([0.018, 0.16, 0.20], dtype=np.float64)
HANDLE_LENGTH = 0.24
ACTION_SCALE = np.array([0.04, 0.04, 0.04], dtype=np.float64)
MAX_TILT = math.radians(30.0)
BASE_POSITION = np.array([0.86, 0.38, 0.0], dtype=np.float64)
HOME_QPOS = np.array([0.0, -0.55, 0.0, -2.15, 0.0, 1.65, 0.78], dtype=np.float64)
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
ACTUATOR_NAMES = [f"actuator{i}" for i in range(1, 8)]

ROOT = Path(__file__).resolve().parent
PANDA_DIR = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda"
PANDA_XML = PANDA_DIR / "panda.xml"
ASSET_DIR = PANDA_DIR / "assets"
PANDA_UPSTREAM = json.loads((PANDA_DIR / "UPSTREAM.json").read_text(encoding="utf-8"))

STATE_SCHEMA = [
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "qx",
    "qy",
    "qz",
    "qw",
    "wx",
    "wy",
    "wz",
    "anchor_x",
    "anchor_y",
    "anchor_z",
]
ACTION_SCHEMA = ["g", "dx", "dy", "dz", "phi", "theta"]
PADDLE_STATE_SCHEMA = [
    "paddle_x",
    "paddle_y",
    "paddle_z",
    "paddle_vx",
    "paddle_vy",
    "paddle_vz",
    "paddle_target_x",
    "paddle_target_y",
    "paddle_target_z",
]
CONTACT_SCHEMA = ["ball_paddle_contact", "ball_floor_contact", "ball_wall_contact"]
ARM_STATE_SCHEMA = (
    [f"joint{i}" for i in range(1, 8)]
    + [f"joint{i}_vel" for i in range(1, 8)]
    + [f"target_joint{i}" for i in range(1, 8)]
)


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))


def _joint_qpos_adr(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[_joint_id(model, name)])


def _joint_dof_adr(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_dofadr[_joint_id(model, name)])


def _elastic_solref(restitution: float, stiffness: float = 60000.0) -> str:
    ratio = -math.log(max(min(restitution, 1.0), 1.0e-6)) / math.pi
    zeta = ratio / math.sqrt(1.0 + ratio * ratio)
    damping = 2.0 * zeta * math.sqrt(stiffness)
    return f"{-stiffness:.10g} {-damping:.10g}"


def _normal_rotation(phi: float, theta: float) -> np.ndarray:
    """Return blade local-to-world rotation with local +x as its face normal."""
    theta = float(np.clip(theta, 0.0, MAX_TILT))
    normal = np.array(
        [-math.cos(theta), math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi)],
        dtype=np.float64,
    )
    normal /= np.linalg.norm(normal)
    local_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    local_z -= normal * float(np.dot(local_z, normal))
    local_z /= np.linalg.norm(local_z)
    local_y = np.cross(local_z, normal)
    local_y /= np.linalg.norm(local_y)
    return np.column_stack((normal, local_y, local_z))


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(-1))
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def _quat_nlerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q1 = q1.copy()
    if float(np.dot(q0, q1)) < 0.0:
        q1 *= -1.0
    q = (1.0 - alpha) * q0 + alpha * q1
    return q / np.linalg.norm(q)


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise RuntimeError(f"Franka body {name!r} not found")


def build_model(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    gravity: float = GRAVITY,
) -> mujoco.MjModel:
    root = ET.parse(PANDA_XML).getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str(ASSET_DIR))
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")

    option = root.find("option")
    assert option is not None
    option.set("timestep", f"{DT:.12g}")
    option.set("gravity", f"0 0 {-float(gravity):.10g}")
    option.set("integrator", "implicitfast")
    option.set("solver", "Newton")
    option.set("iterations", "80")
    option.set("cone", "elliptic")

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(WIDTH))
    global_visual.set("offheight", str(HEIGHT))
    quality = visual.find("quality")
    if quality is None:
        quality = ET.SubElement(visual, "quality")
    quality.set("shadowsize", "2048")

    asset = root.find("asset")
    assert asset is not None
    ET.SubElement(
        asset,
        "texture",
        name="scene_sky",
        type="skybox",
        builtin="gradient",
        width="512",
        height="3072",
        rgb1="0.30 0.35 0.38",
        rgb2="0.30 0.35 0.38",
    )
    ET.SubElement(
        asset,
        "material",
        name="ball_mat",
        rgba="0.94 0.045 0.035 1",
        specular="0.12",
        shininess="0.18",
    )
    ET.SubElement(
        asset,
        "material",
        name="floor_plain",
        rgba="0.56 0.59 0.58 1",
        specular="0.025",
        reflectance="0.02",
    )
    ET.SubElement(asset, "material", name="wall_main", rgba="0.30 0.35 0.38 1", specular="0.015")
    ET.SubElement(asset, "material", name="paddle_blue", rgba="0.06 0.58 0.96 1", specular="0.08")
    ET.SubElement(
        asset, "material", name="paddle_target", rgba="0.025 0.04 0.055 1", specular="0.05"
    )
    ET.SubElement(asset, "material", name="handle_wood", rgba="0.48 0.25 0.09 1", specular="0.04")

    worldbody = root.find("worldbody")
    assert worldbody is not None
    link0 = _find_body(root, "link0")
    link0.set("pos", _fmt(BASE_POSITION))
    # The arm is visually/kinematically coupled but collision-disabled.  This
    # guarantees that only the blue blade can create a successful contact.
    for geom in link0.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    hand = _find_body(root, "hand")
    ET.SubElement(
        hand,
        "site",
        name="paddle_mount_site",
        pos="0 0 0.1034",
        size="0.007",
        rgba="0 0 0 0",
        group="5",
    )

    for light in list(worldbody.findall("light")):
        worldbody.remove(light)
    ET.SubElement(
        worldbody,
        "light",
        name="key",
        pos="-1.2 -2.0 3.2",
        dir="0.25 0.35 -1",
        diffuse="0.88 0.88 0.88",
        castshadow="true",
    )
    ET.SubElement(
        worldbody,
        "light",
        name="fill",
        pos="1.7 -1.2 2.2",
        dir="-0.35 0.25 -1",
        diffuse="0.34 0.38 0.42",
        castshadow="false",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="observation",
        pos="0.15 -2.15 1.02",
        xyaxes="1 0 0 0 0.232 0.973",
        fovy="43",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="floor_geom",
        type="plane",
        pos="0 0 0",
        size="4 4 0.05",
        material="floor_plain",
        friction="0.06 0.004 0.0005",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="back_wall_visual",
        type="box",
        pos="0 0.76 0.84",
        size="4 0.025 0.84",
        material="wall_main",
        contype="0",
        conaffinity="0",
    )

    ball = ET.SubElement(worldbody, "body", name="ball", pos=_fmt(initial_position))
    ET.SubElement(ball, "freejoint", name="ball_free")
    ET.SubElement(
        ball,
        "geom",
        name="ball_geom",
        type="sphere",
        size=f"{BALL_RADIUS}",
        mass=f"{BALL_MASS}",
        material="ball_mat",
        friction="0.05 0.002 0.0005",
        solimp="0.99 0.99 0.001",
    )

    paddle = ET.SubElement(worldbody, "body", name="paddle", mocap="true", pos="0.75 0 0.5")
    ET.SubElement(
        paddle,
        "geom",
        name="paddle_blade",
        type="box",
        size=_fmt(PADDLE_HALF),
        material="paddle_blue",
        friction="0.25 0.005 0.0005",
        solimp="0.99 0.99 0.001",
    )
    ET.SubElement(
        paddle,
        "geom",
        name="paddle_target_visual",
        type="cylinder",
        size="0.062 0.0025",
        pos="0.0207 0 0",
        quat="0.70710678 0 0.70710678 0",
        material="paddle_target",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        paddle,
        "geom",
        name="paddle_handle",
        type="capsule",
        fromto=f"{-PADDLE_HALF[0]} 0 0 {-HANDLE_LENGTH} 0 0",
        size="0.018",
        material="handle_wood",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        paddle,
        "geom",
        name="paddle_adapter",
        type="cylinder",
        size="0.031 0.018",
        pos=f"{-HANDLE_LENGTH} 0 0",
        quat="0.70710678 0 0.70710678 0",
        rgba="0.22 0.24 0.27 1",
        contype="0",
        conaffinity="0",
    )

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    ET.SubElement(
        contact,
        "pair",
        geom1="ball_geom",
        geom2="floor_geom",
        solref=_elastic_solref(0.92),
        solimp="0.99 0.99 0.001",
        friction="0.05 0.002 0.0005",
    )
    ET.SubElement(
        contact,
        "pair",
        geom1="ball_geom",
        geom2="paddle_blade",
        solref=_elastic_solref(0.92),
        solimp="0.99 0.99 0.001",
        friction="0.25 0.005 0.0005",
    )

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.body_gravcomp[:] = 1.0
    return model


def _joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos = []
    dof = []
    actuator = []
    for joint_name, actuator_name in zip(JOINT_NAMES, ACTUATOR_NAMES, strict=False):
        qpos.append(_joint_qpos_adr(model, joint_name))
        dof.append(_joint_dof_adr(model, joint_name))
        actuator.append(int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)))
    return np.asarray(qpos), np.asarray(dof), np.asarray(actuator)


def _site_pose(data: mujoco.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    return data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy()


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    seed: np.ndarray,
    *,
    iterations: int = 180,
) -> tuple[np.ndarray, float, float]:
    qpos_adrs, dof_adrs, _ = _joint_addresses(model)
    site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "paddle_mount_site"))
    q = np.asarray(seed, dtype=np.float64).copy()
    joint_ids = [_joint_id(model, name) for name in JOINT_NAMES]
    lower = model.jnt_range[joint_ids, 0]
    upper = model.jnt_range[joint_ids, 1]
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    for _ in range(iterations):
        data.qpos[qpos_adrs] = q
        mujoco.mj_forward(model, data)
        current_position, current_rotation = _site_pose(data, site_id)
        position_error = target_position - current_position
        rotation_error = 0.5 * sum(
            (np.cross(current_rotation[:, axis], target_rotation[:, axis]) for axis in range(3)),
            start=np.zeros(3, dtype=np.float64),
        )
        if np.linalg.norm(position_error) < 3.0e-5 and np.linalg.norm(rotation_error) < 5.0e-4:
            break
        jacp.fill(0.0)
        jacr.fill(0.0)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        j = np.vstack((jacp[:, dof_adrs], 0.45 * jacr[:, dof_adrs]))
        error = np.concatenate((position_error, 0.45 * rotation_error))
        damping = 0.025
        dq = j.T @ np.linalg.solve(j @ j.T + (damping * damping) * np.eye(6), error)
        # Redundancy is resolved continuously toward a fixed, safe home branch.
        null_projector = np.eye(7) - j.T @ np.linalg.solve(
            j @ j.T + (damping * damping) * np.eye(6), j
        )
        dq += null_projector @ (0.015 * (HOME_QPOS - q))
        max_step = 0.12
        scale = min(1.0, max_step / max(float(np.max(np.abs(dq))), 1.0e-12))
        q = np.clip(q + scale * dq, lower + 2.0e-3, upper - 2.0e-3)
    data.qpos[qpos_adrs] = q
    mujoco.mj_forward(model, data)
    final_position, final_rotation = _site_pose(data, site_id)
    position_error = float(np.linalg.norm(target_position - final_position))
    trace = float(np.trace(final_rotation.T @ target_rotation))
    angle_error = math.degrees(math.acos(float(np.clip(0.5 * (trace - 1.0), -1.0, 1.0))))
    return q, position_error, angle_error


def _hand_target(
    paddle_position: np.ndarray, paddle_rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    handle_axis = paddle_rotation @ np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    position = paddle_position + HANDLE_LENGTH * handle_axis
    tool_x = paddle_rotation[:, 2].copy()
    tool_z = handle_axis / np.linalg.norm(handle_axis)
    tool_x -= tool_z * float(np.dot(tool_x, tool_z))
    tool_x /= np.linalg.norm(tool_x)
    tool_y = np.cross(tool_z, tool_x)
    tool_y /= np.linalg.norm(tool_y)
    rotation = np.column_stack((tool_x, tool_y, tool_z))
    return position, rotation


def _set_arm_state(
    model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, qvel: np.ndarray | None = None
) -> None:
    qpos_adrs, dof_adrs, actuator_adrs = _joint_addresses(model)
    data.qpos[qpos_adrs] = q
    data.qvel[dof_adrs] = 0.0 if qvel is None else qvel
    data.ctrl[actuator_adrs] = q
    for finger_name in ("finger_joint1", "finger_joint2"):
        data.qpos[_joint_qpos_adr(model, finger_name)] = 0.016


def _ball_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    body = data.body("ball")
    ball_qpos_adr = _joint_qpos_adr(model, "ball_free")
    ball_dof_adr = _joint_dof_adr(model, "ball_free")
    qpos = data.qpos[ball_qpos_adr : ball_qpos_adr + 7]
    qvel = data.qvel[ball_dof_adr : ball_dof_adr + 6]
    # MuJoCo free-joint quaternion is wxyz; public state remains xyzw.
    return np.asarray(
        [
            *body.xpos,
            *qvel[:3],
            qpos[4],
            qpos[5],
            qpos[6],
            qpos[3],
            *qvel[3:6],
            0.0,
            0.0,
            0.45,
        ],
        dtype=np.float32,
    )


def _contact_flags(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, bool, bool]:
    ball_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"))
    paddle_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_blade"))
    floor_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor_geom"))
    paddle = False
    floor = False
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        paddle |= pair == {ball_id, paddle_id}
        floor |= pair == {ball_id, floor_id}
    return paddle, floor, False


def solve_arm_trajectory(
    model: mujoco.MjModel,
    commands: np.ndarray,
    rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    targets = np.zeros((FRAMES, 7), dtype=np.float64)
    position_errors = np.zeros(FRAMES, dtype=np.float64)
    angle_errors = np.zeros(FRAMES, dtype=np.float64)
    seed = HOME_QPOS.copy()
    for frame in range(FRAMES):
        target_position, target_rotation = _hand_target(commands[frame], rotations[frame])
        seed, position_error, angle_error = solve_ik(
            model, data, target_position, target_rotation, seed
        )
        targets[frame] = seed
        position_errors[frame] = position_error
        angle_errors[frame] = angle_error
    return targets, position_errors, angle_errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()
