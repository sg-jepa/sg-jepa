import copy
import json
import math
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import robot_assets

UNIFORM_RESTITUTION = 1.0
UNIFORM_LATERAL_FRICTION = 0.0
UNIFORM_ROLLING_FRICTION = 0.0
UNIFORM_SPINNING_FRICTION = 0.0
UNIFORM_LINEAR_DAMPING = 0.0
UNIFORM_ANGULAR_DAMPING = 0.0
UNITREE_Z1_DIR = Path(__file__).resolve().parent / "third_party" / "mujoco_menagerie" / "unitree_z1"
UR5E_PREFIX = "ur5e_"
ROBOTIQ_PREFIX = "rq_"
UR5E_ROBOTIQ_MODEL = "ur5e_robotiq_2f85"
UR5E_JOINT_NAMES = [
    "ur5e_shoulder_pan_joint",
    "ur5e_shoulder_lift_joint",
    "ur5e_elbow_joint",
    "ur5e_wrist_1_joint",
    "ur5e_wrist_2_joint",
    "ur5e_wrist_3_joint",
]
ROBOTIQ_JOINT_NAMES = [
    "rq_right_driver_joint",
    "rq_right_coupler_joint",
    "rq_right_spring_link_joint",
    "rq_right_follower_joint",
    "rq_left_driver_joint",
    "rq_left_coupler_joint",
    "rq_left_spring_link_joint",
    "rq_left_follower_joint",
]
ROBOTIQ_DRIVER_JOINT_NAMES = ["rq_left_driver_joint", "rq_right_driver_joint"]
ROBOTIQ_ACTUATOR_NAME = "rq_fingers_actuator"
ARM_GRIPPER_TOOL_SITE = "arm_gripper_ur5e_tool_site"
ARM_GRIPPER_BASE_SITE = "arm_gripper_base_site"
ARM_GRIPPER_CENTER_SITE = "arm_gripper_center_site"
ARM_GRIPPER_LEFT_PAD_SITE = "arm_gripper_left_pad_site"
ARM_GRIPPER_RIGHT_PAD_SITE = "arm_gripper_right_pad_site"
ARM_GRIPPER_GRASP_SITE = "arm_gripper_grasp_center_site"
ARM_GRIPPER_LEFT_COLLISION_PAD_SITE = "arm_gripper_left_collision_pad_site"
ARM_GRIPPER_RIGHT_COLLISION_PAD_SITE = "arm_gripper_right_collision_pad_site"
ARM_GRIPPER_PALM_SITE = "arm_gripper_palm_site"
ARM_GRIPPER_HIDDEN_SITE_RGBA = "0 0 0 0"
ARM_GRIPPER_PAD_GEOMS = [
    "rq_left_pad1",
    "rq_left_pad2",
    "rq_right_pad1",
    "rq_right_pad2",
    "arm_gripper_left_collision_pad_geom",
    "arm_gripper_right_collision_pad_geom",
]
ARM_GRIPPER_PALM_GEOMS = ["arm_gripper_palm_collision_geom"]
ARM_CATCHER_EE_SITE = "arm_catcher_ee_site"
ARM_CATCHER_MOUNT_BODY = "arm_catcher_mount"
ARM_CATCHER_TOOL_TIP_LOCAL_X = 0.051
ARM_CATCHER_OPEN_AXIS_WORLD = (0.0, 1.0, 0.0)
ARM_CATCHER_TIP_NORMAL_WORLD = (0.0, 1.0, 0.0)
ARM_CATCHER_WORLD_QUATERNION_XYZW = (0.0, 0.0, 0.0, 1.0)
# How far behind the flange the net's back face sits, along the net normal.
# ARM_CATCHER_TOOL_TIP_LOCAL_X puts the tip at the end of link06, so a zero
# offset leaves the cage entirely clear of the wrist and the wrist shows
# through the net's open back. Seating the cage over the wrist instead is what
# the 5267eab render did. The mount origin is what every capture bound is
# measured against, so moving it carries the capture volume along with the
# cage; the arm extends further along the normal to compensate.
ARM_CATCHER_MOUNT_BACK_OFFSET = 0.0


def srgb_channel_to_linear(value):
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def hex_to_linear_rgba(hex_rgb, alpha=1.0):
    hex_rgb = hex_rgb.lstrip("#")
    channels = [int(hex_rgb[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return [srgb_channel_to_linear(channel) for channel in channels] + [alpha]


def material_from_hex(hex_rgb, *, alpha=1.0, roughness=0.8, specular=0.15):
    return {
        "base_color": hex_to_linear_rgba(hex_rgb, alpha=alpha),
        "roughness": roughness,
        "specular": specular,
    }


def dark_material(alpha=1.0):
    return material_from_hex("#202329", alpha=alpha, roughness=0.95, specular=0.0)


def wall_material(alpha=0.28):
    return material_from_hex("#59636F", alpha=alpha, roughness=0.85, specular=0.05)


def accent_material(color):
    return {"base_color": list(color), "roughness": 0.35, "specular": 0.2}


def checker_material(
    color_a,
    color_b,
    *,
    base_color=None,
    roughness=0.42,
    specular=0.18,
    emission=0.0,
    longitude_segments=12,
    latitude_segments=6,
    width=1024,
    height=512,
):
    return {
        "base_color": list(base_color or color_a),
        "roughness": roughness,
        "specular": specular,
        "emission": emission,
        "texture": {
            "kind": "latlong_checker",
            "rgb1": list(color_a[:3]),
            "rgb2": list(color_b[:3]),
            "rgba": [1.0, 1.0, 1.0, 1.0],
            "longitude_segments": int(longitude_segments),
            "latitude_segments": int(latitude_segments),
            "width": int(width),
            "height": int(height),
        },
    }


def make_camera(location, target, fovy_degrees):
    location = tuple(float(value) for value in location)
    target = tuple(float(value) for value in target)
    forward = normalize_vector(
        tuple(target_i - location_i for target_i, location_i in zip(target, location, strict=False))
    )
    world_up = (0.0, 0.0, 1.0)
    right = normalize_vector(cross(forward, world_up))
    camera_z_axis = tuple(-value for value in forward)
    camera_y_axis = normalize_vector(cross(camera_z_axis, right))
    return {
        "type": "perspective",
        "location": list(location),
        "target": list(target),
        "optical_axis_world": list(forward),
        "image_right_world": list(right),
        "image_up_world": list(camera_y_axis),
        "xyaxes": [*right, *camera_y_axis],
        "fovy": float(fovy_degrees),
        "resolution": [256, 256],
    }


def normalize_vector(vector):
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 1e-12:
        raise ValueError(f"Cannot normalize near-zero vector: {vector!r}")
    return tuple(component / length for component in vector)


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def wrap_angle_pi(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def rotate_vector_by_quaternion(vector, quaternion_xyzw):
    x, y, z, w = normalize_quaternion_xyzw(quaternion_xyzw)
    q_vec = (x, y, z)
    v = tuple(float(value) for value in vector)
    t = tuple(2.0 * value for value in cross(q_vec, v))
    return tuple(v[index] + w * t[index] + cross(q_vec, t)[index] for index in range(3))


def paddle_tilt_to_roll_pitch(phi, theta, max_theta):
    theta = clip_value(float(theta), 0.0, float(max_theta))
    phi = wrap_angle_pi(phi)
    normal = (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )
    roll = -math.asin(clip_value(normal[1], -1.0, 1.0))
    pitch = math.atan2(normal[0], max(normal[2], 1.0e-12))
    return (
        clip_value(roll, -float(max_theta), float(max_theta)),
        clip_value(pitch, -float(max_theta), float(max_theta)),
    )


def paddle_normal_to_phi_theta(normal, fallback_phi=0.0):
    nx, ny, nz = (float(value) for value in normal)
    horizontal = math.sqrt(nx * nx + ny * ny)
    phi = wrap_angle_pi(fallback_phi if horizontal <= 1.0e-9 else math.atan2(ny, nx))
    theta = math.atan2(horizontal, nz)
    return phi, max(0.0, theta)


def euler_xyz_to_quaternion_xyzw(rotation_euler):
    roll, pitch, yaw = rotation_euler
    cr = math.cos(0.5 * roll)
    sr = math.sin(0.5 * roll)
    cp = math.cos(0.5 * pitch)
    sp = math.sin(0.5 * pitch)
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return normalize_quaternion_xyzw((x, y, z, w))


def normalize_quaternion_xyzw(quaternion):
    length = math.sqrt(sum(component * component for component in quaternion))
    if length <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(component / length for component in quaternion)


def make_box(
    name,
    half_extents,
    position,
    *,
    mass,
    rotation_euler=(0.0, 0.0, 0.0),
    material=None,
    restitution=UNIFORM_RESTITUTION,
    lateral_friction=UNIFORM_LATERAL_FRICTION,
    rolling_friction=UNIFORM_ROLLING_FRICTION,
    spinning_friction=UNIFORM_SPINNING_FRICTION,
    linear_damping=UNIFORM_LINEAR_DAMPING,
    angular_damping=UNIFORM_ANGULAR_DAMPING,
    initial_velocity=(0.0, 0.0, 0.0),
    initial_angular_velocity=(0.0, 0.0, 0.0),
    contact_solref=None,
):
    return {
        "name": name,
        "kind": "box",
        "half_extents": list(half_extents),
        "position": list(position),
        "rotation_euler": list(rotation_euler),
        "mass": mass,
        "material": material or wall_material(),
        "dynamics": {
            "restitution": restitution,
            "lateral_friction": lateral_friction,
            "rolling_friction": rolling_friction,
            "spinning_friction": spinning_friction,
            "linear_damping": linear_damping,
            "angular_damping": angular_damping,
            "contact_solref": contact_solref,
        },
        "initial_velocity": list(initial_velocity),
        "initial_angular_velocity": list(initial_angular_velocity),
        "lock_rotation": False,
    }


def make_cube(
    name,
    half_extent,
    position,
    *,
    mass,
    rotation_euler=(0.0, 0.0, 0.0),
    material=None,
    restitution=UNIFORM_RESTITUTION,
    lateral_friction=UNIFORM_LATERAL_FRICTION,
    rolling_friction=UNIFORM_ROLLING_FRICTION,
    spinning_friction=UNIFORM_SPINNING_FRICTION,
    linear_damping=UNIFORM_LINEAR_DAMPING,
    angular_damping=UNIFORM_ANGULAR_DAMPING,
    initial_velocity=(0.0, 0.0, 0.0),
    initial_angular_velocity=(0.0, 0.0, 0.0),
):
    spec = make_box(
        name,
        half_extents=(half_extent, half_extent, half_extent),
        position=position,
        mass=mass,
        rotation_euler=rotation_euler,
        material=material or accent_material((0.62, 0.12, 0.10, 1.0)),
        restitution=restitution,
        lateral_friction=lateral_friction,
        rolling_friction=rolling_friction,
        spinning_friction=spinning_friction,
        linear_damping=linear_damping,
        angular_damping=angular_damping,
        initial_velocity=initial_velocity,
        initial_angular_velocity=initial_angular_velocity,
    )
    spec["kind"] = "cube"
    return spec


def make_sphere(
    name,
    radius,
    position,
    *,
    mass,
    material=None,
    restitution=UNIFORM_RESTITUTION,
    lateral_friction=UNIFORM_LATERAL_FRICTION,
    rolling_friction=UNIFORM_ROLLING_FRICTION,
    spinning_friction=UNIFORM_SPINNING_FRICTION,
    linear_damping=UNIFORM_LINEAR_DAMPING,
    angular_damping=UNIFORM_ANGULAR_DAMPING,
    initial_velocity=(0.0, 0.0, 0.0),
    initial_angular_velocity=(0.0, 0.0, 0.0),
    lock_rotation=True,
    contact_solref=None,
):
    return {
        "name": name,
        "kind": "ball",
        "radius": radius,
        "position": list(position),
        "rotation_euler": [0.0, 0.0, 0.0],
        "mass": mass,
        "material": material or accent_material((0.94, 0.45, 0.14, 1.0)),
        "dynamics": {
            "restitution": restitution,
            "lateral_friction": lateral_friction,
            "rolling_friction": rolling_friction,
            "spinning_friction": spinning_friction,
            "linear_damping": linear_damping,
            "angular_damping": angular_damping,
            "contact_solref": contact_solref,
        },
        "initial_velocity": list(initial_velocity),
        "initial_angular_velocity": list(initial_angular_velocity),
        "lock_rotation": lock_rotation,
    }


def frame_payload(frame_idx, fps, position, quaternion, linear_velocity, angular_velocity):
    return {
        "frame": frame_idx + 1,
        "time_seconds": frame_idx / fps,
        "position": [float(value) for value in position],
        "quaternion_xyzw": [float(value) for value in quaternion],
        "linear_velocity": [float(value) for value in linear_velocity],
        "angular_velocity": [float(value) for value in angular_velocity],
    }


def serialize_object(spec):
    payload = {
        "name": spec["name"],
        "kind": spec["kind"],
        "material": spec["material"],
        "position": spec["position"],
    }
    if spec["kind"] in {"box", "cube"}:
        payload["half_extents"] = spec["half_extents"]
        payload["rotation_euler"] = spec.get("rotation_euler", [0.0, 0.0, 0.0])
    else:
        payload["radius"] = spec["radius"]
    if "render_visible" in spec:
        payload["render_visible"] = bool(spec["render_visible"])
    if "render_only" in spec:
        payload["render_only"] = bool(spec["render_only"])
    if spec.get("natural_material"):
        payload["natural_material"] = spec["natural_material"]
    return payload


def make_dynamic_payload(spec):
    payload = {
        "name": spec["name"],
        "kind": spec["kind"],
        "material": spec["material"],
        "frames": [],
    }
    if spec["kind"] in {"box", "cube"}:
        payload["half_extents"] = spec["half_extents"]
    else:
        payload["radius"] = spec["radius"]
    return payload


def format_scalar(value):
    return f"{float(value):.10g}"


def format_vec(values):
    return " ".join(format_scalar(value) for value in values)


def xyzw_to_wxyz(quaternion_xyzw):
    x, y, z, w = quaternion_xyzw
    return (w, x, y, z)


def wxyz_to_xyzw(quaternion_wxyz):
    w, x, y, z = quaternion_wxyz
    return (x, y, z, w)


def mat_to_quaternion_xyzw(matrix_values):
    m = [float(value) for value in matrix_values]
    trace = m[0] + m[4] + m[8]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m[7] - m[5]) / scale
        y = (m[2] - m[6]) / scale
        z = (m[3] - m[1]) / scale
    elif m[0] > m[4] and m[0] > m[8]:
        scale = math.sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0
        w = (m[7] - m[5]) / scale
        x = 0.25 * scale
        y = (m[1] + m[3]) / scale
        z = (m[2] + m[6]) / scale
    elif m[4] > m[8]:
        scale = math.sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0
        w = (m[2] - m[6]) / scale
        x = (m[1] + m[3]) / scale
        y = 0.25 * scale
        z = (m[5] + m[7]) / scale
    else:
        scale = math.sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0
        w = (m[3] - m[1]) / scale
        x = (m[2] + m[6]) / scale
        y = (m[5] + m[7]) / scale
        z = 0.25 * scale
    return normalize_quaternion_xyzw((x, y, z, w))


def body_state(data, body_name):
    body = data.body(body_name)
    position = body.xpos.copy()
    quaternion = normalize_quaternion_xyzw(wxyz_to_xyzw(body.xquat.copy()))
    angular_velocity = body.cvel[:3].copy()
    linear_velocity = body.cvel[3:].copy()
    return position, quaternion, linear_velocity, angular_velocity


def geom_pose(data, geom_name):
    geom = data.geom(geom_name)
    return geom.xpos.copy(), mat_to_quaternion_xyzw(geom.xmat.copy())


def add_root_elements(root, gravity, timestep, offwidth=256, offheight=256):
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(
        root,
        "option",
        timestep=format_scalar(timestep),
        gravity=format_vec(gravity),
        integrator="implicitfast",
        solver="Newton",
        cone="elliptic",
        iterations="100",
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(offwidth), offheight=str(offheight))


def solref_for_restitution(restitution):
    if restitution >= 0.999:
        return "-1000 0"
    return None


def elastic_contact_solref(restitution, stiffness):
    """Direct-form solref (-stiffness, -damping) for a bouncy contact.

    The mass-normalized damping is chosen so the contact's damping ratio
    reproduces the requested restitution: e = exp(-pi * zeta / sqrt(1 - zeta^2)).
    """
    restitution = min(max(float(restitution), 1.0e-6), 1.0)
    stiffness = abs(float(stiffness))
    if restitution >= 0.999:
        return f"{format_scalar(-stiffness)} 0"
    ratio = -math.log(restitution) / math.pi
    zeta = ratio / math.sqrt(1.0 + ratio * ratio)
    damping = 2.0 * zeta * math.sqrt(stiffness)
    return f"{format_scalar(-stiffness)} {format_scalar(-damping)}"


def geom_attrs_from_spec(spec, *, dynamic_mass=None, collision=True, name=None, include_euler=True):
    material = spec["material"]
    rgba = material.get("base_color", [0.8, 0.8, 0.8, 1.0])
    dynamics = spec.get("dynamics", {})
    attrs = {
        "name": name or spec["name"],
        "rgba": format_vec(rgba),
    }
    if spec["kind"] in {"box", "cube"}:
        attrs["type"] = "box"
        attrs["size"] = format_vec(spec["half_extents"])
        if include_euler:
            attrs["euler"] = format_vec(spec.get("rotation_euler", (0.0, 0.0, 0.0)))
    else:
        attrs["type"] = "sphere"
        attrs["size"] = format_scalar(spec["radius"])
    if dynamic_mass is not None:
        attrs["mass"] = format_scalar(dynamic_mass)
    if not collision:
        attrs["contype"] = "0"
        attrs["conaffinity"] = "0"
    else:
        attrs["condim"] = "6"
        attrs["friction"] = format_vec(
            [
                dynamics.get("lateral_friction", UNIFORM_LATERAL_FRICTION),
                dynamics.get("spinning_friction", UNIFORM_SPINNING_FRICTION),
                dynamics.get("rolling_friction", UNIFORM_ROLLING_FRICTION),
            ]
        )
        solref = dynamics.get("contact_solref") or solref_for_restitution(
            dynamics.get("restitution", UNIFORM_RESTITUTION)
        )
        if solref is not None:
            attrs["solref"] = solref
    return attrs


def add_static_geom(worldbody, spec):
    attrs = geom_attrs_from_spec(spec)
    attrs["pos"] = format_vec(spec["position"])
    ET.SubElement(worldbody, "geom", attrs)


def add_moving_object_static_contacts(root, static_specs):
    contact = get_or_create_child(root, "contact")
    for spec in static_specs:
        if spec.get("render_only", False) or not spec.get("collision", True):
            continue
        dynamics = spec.get("dynamics", {})
        pair_attrs = {
            "geom1": "moving_object",
            "geom2": spec["name"],
            "condim": "6",
            "friction": format_vec(
                [
                    dynamics.get("lateral_friction", UNIFORM_LATERAL_FRICTION),
                    dynamics.get("spinning_friction", UNIFORM_SPINNING_FRICTION),
                    dynamics.get("rolling_friction", UNIFORM_ROLLING_FRICTION),
                ]
            ),
        }
        solref = dynamics.get("contact_solref") or solref_for_restitution(
            dynamics.get("restitution", UNIFORM_RESTITUTION)
        )
        if solref is not None:
            pair_attrs["solref"] = solref
        ET.SubElement(contact, "pair", pair_attrs)


def add_dynamic_body(worldbody, spec):
    body = ET.SubElement(worldbody, "body", name=spec["name"], pos="0 0 0")
    slide_damping = spec["dynamics"].get("linear_damping", UNIFORM_LINEAR_DAMPING)
    if spec.get("lock_rotation", spec["kind"] == "ball"):
        ET.SubElement(
            body,
            "joint",
            name=f"{spec['name']}_slide_x",
            type="slide",
            axis="1 0 0",
            damping=format_scalar(slide_damping),
        )
        ET.SubElement(
            body,
            "joint",
            name=f"{spec['name']}_slide_y",
            type="slide",
            axis="0 1 0",
            damping=format_scalar(slide_damping),
        )
        ET.SubElement(
            body,
            "joint",
            name=f"{spec['name']}_slide_z",
            type="slide",
            axis="0 0 1",
            damping=format_scalar(slide_damping),
        )
    else:
        ET.SubElement(
            body,
            "freejoint",
            name=f"{spec['name']}_free",
        )
    ET.SubElement(
        body,
        "geom",
        geom_attrs_from_spec(spec, dynamic_mass=spec["mass"], include_euler=False),
    )
    return body


def add_catcher_body(root, worldbody, catcher_spec):
    workspace_min = catcher_spec["workspace_min"]
    workspace_max = catcher_spec["workspace_max"]
    fixed_y = catcher_spec["fixed_y"]
    body = ET.SubElement(
        worldbody,
        "body",
        name=catcher_spec["name"],
        pos=format_vec((0.0, fixed_y, 0.0)),
    )
    joint_attrs = [
        ("x", "1 0 0", workspace_min[0], workspace_max[0]),
        ("z", "0 0 1", workspace_min[2], workspace_max[2]),
    ]
    for axis_name, axis, joint_min, joint_max in joint_attrs:
        ET.SubElement(
            body,
            "joint",
            name=f"{catcher_spec['name']}_slide_{axis_name}",
            type="slide",
            axis=axis,
            limited="true",
            range=f"{format_scalar(joint_min)} {format_scalar(joint_max)}",
            damping="32",
            armature="0.05",
        )

    for part in catcher_spec["parts"]:
        part_material = part.get("material", catcher_spec["material"])
        attrs = {
            "name": part["name"],
            "type": "box",
            "pos": format_vec(part["offset"]),
            "size": format_vec(part["half_extents"]),
            "rgba": format_vec(part_material.get("base_color", [0.16, 0.22, 0.32, 1.0])),
        }
        if part.get("collision", True):
            attrs.update(
                {
                    "mass": "1.0",
                    "condim": "6",
                    "friction": "0.65 0.04 0.04",
                }
            )
        else:
            attrs.update(
                {
                    "mass": "0.02",
                    "contype": "0",
                    "conaffinity": "0",
                }
            )
        ET.SubElement(
            body,
            "geom",
            attrs,
        )

    contact = ET.SubElement(root, "contact")
    for part in catcher_spec["parts"]:
        if not part.get("collision", True):
            continue
        ET.SubElement(
            contact,
            "pair",
            geom1="moving_object",
            geom2=part["name"],
            condim="6",
            friction="0.65 0.04 0.04",
            solref="0.035 1",
        )

    actuator = ET.SubElement(root, "actuator")
    for axis_name, joint_min, joint_max in (
        ("x", workspace_min[0], workspace_max[0]),
        ("z", workspace_min[2], workspace_max[2]),
    ):
        ET.SubElement(
            actuator,
            "position",
            name=f"{catcher_spec['name']}_act_{axis_name}",
            joint=f"{catcher_spec['name']}_slide_{axis_name}",
            kp="420",
            ctrlrange=f"{format_scalar(joint_min)} {format_scalar(joint_max)}",
            ctrllimited="true",
            forcerange="-160 160",
            forcelimited="true",
        )
    return body


def add_paddle_body(root, worldbody, paddle_spec):
    workspace_min = paddle_spec["workspace_min"]
    workspace_max = paddle_spec["workspace_max"]
    tilt_enabled = bool(paddle_spec.get("tilt_enabled", False))
    max_tilt_theta = float(paddle_spec.get("max_tilt_theta", 0.0))
    servo = paddle_spec.get("servo") or {}
    slide_damping = float(servo.get("slide_damping", 28.0))
    slide_armature = float(servo.get("slide_armature", 0.05))
    position_kp = float(servo.get("position_kp", 520.0))
    position_forcerange = abs(float(servo.get("position_forcerange", 240.0)))
    tilt_damping = float(servo.get("tilt_damping", 4.0))
    tilt_armature = float(servo.get("tilt_armature", 0.015))
    tilt_kp = float(servo.get("tilt_kp", 180.0))
    tilt_forcerange = abs(float(servo.get("tilt_forcerange", 90.0)))
    body = ET.SubElement(worldbody, "body", name=paddle_spec["name"], pos="0 0 0")
    joint_attrs = [
        ("x", "1 0 0", workspace_min[0], workspace_max[0]),
        ("y", "0 1 0", workspace_min[1], workspace_max[1]),
        ("z", "0 0 1", workspace_min[2], workspace_max[2]),
    ]
    for axis_name, axis, joint_min, joint_max in joint_attrs:
        ET.SubElement(
            body,
            "joint",
            name=f"{paddle_spec['name']}_slide_{axis_name}",
            type="slide",
            axis=axis,
            limited="true",
            range=f"{format_scalar(joint_min)} {format_scalar(joint_max)}",
            damping=format_scalar(slide_damping),
            armature=format_scalar(slide_armature),
        )
    if tilt_enabled:
        for axis_name, axis in (("roll", "1 0 0"), ("pitch", "0 1 0")):
            ET.SubElement(
                body,
                "joint",
                name=f"{paddle_spec['name']}_hinge_{axis_name}",
                type="hinge",
                axis=axis,
                limited="true",
                range=f"{format_scalar(-max_tilt_theta)} {format_scalar(max_tilt_theta)}",
                damping=format_scalar(tilt_damping),
                armature=format_scalar(tilt_armature),
            )

    friction = format_vec((paddle_spec.get("lateral_friction", 0.25), 0.0, 0.0))
    paddle_solref = paddle_spec.get("contact_solref") or solref_for_restitution(
        paddle_spec.get("restitution", UNIFORM_RESTITUTION)
    )
    collision_part_mass = float(paddle_spec.get("collision_part_mass", 0.45))
    for part in paddle_spec["parts"]:
        if part.get("render_only", False):
            continue
        part_material = part.get("material", paddle_spec["material"])
        attrs = {
            "name": part["name"],
            "type": "box",
            "pos": format_vec(part["offset"]),
            "size": format_vec(part["half_extents"]),
            "rgba": format_vec(part_material.get("base_color", [0.16, 0.22, 0.32, 1.0])),
        }
        if part.get("collision", True):
            attrs.update(
                {
                    "mass": format_scalar(collision_part_mass),
                    "condim": "6",
                    "friction": friction,
                }
            )
            if paddle_solref is not None:
                attrs["solref"] = paddle_solref
        else:
            attrs.update(
                {
                    "mass": "0.02",
                    "contype": "0",
                    "conaffinity": "0",
                }
            )
        ET.SubElement(body, "geom", attrs)

    contact = get_or_create_child(root, "contact")
    for part in paddle_spec["parts"]:
        if part.get("render_only", False) or not part.get("collision", True):
            continue
        pair_attrs = {
            "geom1": "moving_object",
            "geom2": part["name"],
            "condim": "6",
            "friction": friction,
        }
        if paddle_solref is not None:
            pair_attrs["solref"] = paddle_solref
        ET.SubElement(contact, "pair", pair_attrs)

    actuator = ET.SubElement(root, "actuator")
    for axis_name, joint_min, joint_max in (
        ("x", workspace_min[0], workspace_max[0]),
        ("y", workspace_min[1], workspace_max[1]),
        ("z", workspace_min[2], workspace_max[2]),
    ):
        ET.SubElement(
            actuator,
            "position",
            name=f"{paddle_spec['name']}_act_{axis_name}",
            joint=f"{paddle_spec['name']}_slide_{axis_name}",
            kp=format_scalar(position_kp),
            ctrlrange=f"{format_scalar(joint_min)} {format_scalar(joint_max)}",
            ctrllimited="true",
            forcerange=f"{format_scalar(-position_forcerange)} {format_scalar(position_forcerange)}",
            forcelimited="true",
        )
    if tilt_enabled:
        for axis_name in ("roll", "pitch"):
            ET.SubElement(
                actuator,
                "position",
                name=f"{paddle_spec['name']}_act_{axis_name}",
                joint=f"{paddle_spec['name']}_hinge_{axis_name}",
                kp=format_scalar(tilt_kp),
                ctrlrange=f"{format_scalar(-max_tilt_theta)} {format_scalar(max_tilt_theta)}",
                ctrllimited="true",
                forcerange=f"{format_scalar(-tilt_forcerange)} {format_scalar(tilt_forcerange)}",
                forcelimited="true",
            )
    return body


def get_or_create_child(parent, tag):
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def insert_before_worldbody(root, element):
    children = list(root)
    for index, child in enumerate(children):
        if child.tag == "worldbody":
            root.insert(index, element)
            return element
    root.append(element)
    return element


def get_or_create_root_child_before_worldbody(root, tag):
    child = root.find(tag)
    if child is not None:
        return child
    return insert_before_worldbody(root, ET.Element(tag))


def load_unitree_z1_root():
    return ET.parse(UNITREE_Z1_DIR / "z1.xml").getroot()


def parse_vec(text):
    return [float(value) for value in text.split()]


def scaled_vec_text(text, scale):
    return format_vec([value * scale for value in parse_vec(text)])


def scale_unitree_z1_tree(element, scale):
    if abs(scale - 1.0) < 1e-9:
        return
    for attr_name in ("pos", "size"):
        if element.get(attr_name):
            element.set(attr_name, scaled_vec_text(element.get(attr_name), scale))
    if element.tag == "inertial" and element.get("diaginertia"):
        element.set(
            "diaginertia",
            format_vec([value * scale * scale for value in parse_vec(element.get("diaginertia"))]),
        )
    for child in element:
        scale_unitree_z1_tree(child, scale)


def append_unitree_z1_assets(root, z1_root, scale=1.0):
    asset = get_or_create_root_child_before_worldbody(root, "asset")
    z1_asset = z1_root.find("asset")
    if z1_asset is None:
        return
    for child in z1_asset:
        child_copy = copy.deepcopy(child)
        mesh_file = child_copy.get("file")
        if mesh_file:
            child_copy.set("file", str((UNITREE_Z1_DIR / "assets" / mesh_file).resolve()))
            if abs(scale - 1.0) >= 1e-9:
                child_copy.set("scale", format_vec((scale, scale, scale)))
        asset.append(child_copy)


def append_unitree_z1_defaults(root, z1_root):
    for default in z1_root.findall("default"):
        insert_before_worldbody(root, copy.deepcopy(default))


def find_body_recursive(body, name):
    if body.get("name") == name:
        return body
    for child in body.findall("body"):
        found = find_body_recursive(child, name)
        if found is not None:
            return found
    return None


def disable_unitree_z1_link_collisions(body):
    for geom in body.findall(".//geom"):
        if geom.get("class") == "collision" or geom.get("group") == "3":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def get_asset_before_worldbody(root):
    return get_or_create_root_child_before_worldbody(root, "asset")


def enable_autolimits(root):
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("autolimits", "true")


def prefixed_ur5e_bundle():
    ur5e_root = robot_assets.load_xml_root(robot_assets.UR5E_DIR / "ur5e.xml")
    mesh_names, material_names = robot_assets.collect_asset_names(ur5e_root)
    return ur5e_root, mesh_names, material_names


def prefixed_robotiq_bundle():
    robotiq_root = robot_assets.load_xml_root(robot_assets.ROBOTIQ_2F85_DIR / "2f85.xml")
    mesh_names, material_names = robot_assets.collect_asset_names(robotiq_root)
    return robotiq_root, mesh_names, material_names


def append_prefixed_menagerie_model(
    root,
    source_root,
    source_dir,
    prefix,
    mesh_names,
    material_names,
    *,
    model_scale=1.0,
):
    append_scaled_prefixed_defaults(
        root,
        source_root,
        prefix,
        mesh_names,
        material_names,
        model_scale,
    )
    append_scaled_prefixed_assets(
        root,
        source_root,
        source_dir / "assets",
        prefix,
        mesh_names,
        material_names,
        model_scale,
    )


def scale_text_values(text, scale):
    return format_vec([float(value) * scale for value in text.split()])


def scale_general_actuator_strength(element, scale):
    if abs(scale - 1.0) < 1e-9:
        return
    strength_scale = scale**4
    for item in robot_assets.walk_xml(element):
        if item.tag != "general":
            continue
        for attr_name in ("gainprm", "biasprm", "forcerange"):
            if item.get(attr_name):
                item.set(attr_name, scale_text_values(item.get(attr_name), strength_scale))


def scale_xml_spatial_attributes(element, scale):
    if abs(scale - 1.0) < 1e-9:
        return
    for attr_name in ("pos", "size", "fromto"):
        if element.get(attr_name):
            element.set(attr_name, scale_text_values(element.get(attr_name), scale))
    if element.tag == "mesh" and element.get("scale"):
        element.set("scale", scale_text_values(element.get("scale"), scale))
    if element.tag == "inertial":
        if element.get("mass"):
            element.set("mass", format_scalar(float(element.get("mass")) * scale**3))
        if element.get("diaginertia"):
            element.set("diaginertia", scale_text_values(element.get("diaginertia"), scale**5))
    elif element.tag == "geom" and element.get("mass"):
        element.set("mass", format_scalar(float(element.get("mass")) * scale**3))
    for child in element:
        scale_xml_spatial_attributes(child, scale)


def append_scaled_prefixed_defaults(
    root, source_root, prefix, mesh_names, material_names, model_scale
):
    for default in source_root.findall("default"):
        prefixed = robot_assets.prefix_menagerie_tree(
            default,
            prefix,
            mesh_names,
            material_names,
        )
        scale_xml_spatial_attributes(prefixed, model_scale)
        scale_general_actuator_strength(prefixed, model_scale)
        insert_before_worldbody(root, prefixed)


def append_scaled_prefixed_assets(
    root, source_root, asset_dir, prefix, mesh_names, material_names, model_scale
):
    asset = get_asset_before_worldbody(root)
    source_asset = source_root.find("asset")
    if source_asset is None:
        return
    for child in source_asset:
        prefixed = robot_assets.prefixed_asset_child(
            child,
            asset_dir,
            prefix,
            mesh_names,
            material_names,
        )
        if child.tag == "mesh" and abs(model_scale - 1.0) >= 1e-9:
            if child.get("scale") or not child.get("class"):
                existing_scale = prefixed.get("scale", "1 1 1")
                prefixed.set("scale", scale_text_values(existing_scale, model_scale))
        asset.append(prefixed)


def disable_fragile_robotiq_mesh_collisions(gripper_body):
    for geom in gripper_body.findall(".//geom"):
        geom_class = geom.get("class", "")
        geom_name = geom.get("name", "")
        if geom_name in {"rq_left_pad1", "rq_left_pad2", "rq_right_pad1", "rq_right_pad2"}:
            geom.set("condim", "6")
            geom.set("friction", "1.25 0.08 0.08")
            geom.set("solref", "0.012 1.2")
            geom.set("solimp", "0.90 0.99 0.002")
            geom.set("priority", "2")
            continue
        if "collision" in geom_class or geom.get("group") == "3":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def disable_ur5e_link_collisions(ur5e_body):
    for geom in ur5e_body.findall(".//geom"):
        geom_class = geom.get("class", "")
        if "collision" in geom_class or geom.get("group") == "3":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("rgba", "0 0 0 0")


def add_arm_gripper_sites_and_collision(robotiq_body):
    robotiq_body.set("name", "rq_base_mount")
    ET.SubElement(
        robotiq_body,
        "site",
        name=ARM_GRIPPER_BASE_SITE,
        pos="0 0 0",
        size="0.012",
        rgba=ARM_GRIPPER_HIDDEN_SITE_RGBA,
    )

    base_body = robot_assets.find_body_recursive(robotiq_body, "rq_base")
    left_pad_body = robot_assets.find_body_recursive(robotiq_body, "rq_left_pad")
    right_pad_body = robot_assets.find_body_recursive(robotiq_body, "rq_right_pad")
    if base_body is None or left_pad_body is None or right_pad_body is None:
        raise ValueError("Could not find Robotiq base/pad bodies after prefixing.")

    ET.SubElement(
        base_body,
        "site",
        name=ARM_GRIPPER_CENTER_SITE,
        pos="0 0 0.145",
        type="sphere",
        size="0.010",
        rgba=ARM_GRIPPER_HIDDEN_SITE_RGBA,
    )
    ET.SubElement(
        base_body,
        "site",
        name=ARM_GRIPPER_GRASP_SITE,
        pos="0 0 0.110",
        type="sphere",
        size="0.012",
        rgba=ARM_GRIPPER_HIDDEN_SITE_RGBA,
    )
    ET.SubElement(
        base_body,
        "site",
        name=ARM_GRIPPER_PALM_SITE,
        pos="0 0 0.078",
        type="sphere",
        size="0.010",
        rgba=ARM_GRIPPER_HIDDEN_SITE_RGBA,
    )
    ET.SubElement(
        base_body,
        "geom",
        name="arm_gripper_palm_collision_geom",
        type="box",
        pos="0 0 0.078",
        size="0.030 0.020 0.018",
        rgba="0.10 0.55 0.80 0.0",
        mass="0",
        condim="6",
        friction="1.30 0.12 0.12",
        solref="0.024 1.7",
        solimp="0.92 0.995 0.001",
        priority="2",
    )

    for body, side in (
        (left_pad_body, "left"),
        (right_pad_body, "right"),
    ):
        ET.SubElement(
            body,
            "site",
            name=f"arm_gripper_{side}_pad_site",
            pos="0 -0.0026 0.01875",
            type="sphere",
            size="0.010",
            rgba=ARM_GRIPPER_HIDDEN_SITE_RGBA,
        )
        ET.SubElement(
            body,
            "site",
            name=f"arm_gripper_{side}_collision_pad_site",
            pos="0 -0.0026 0.01875",
            type="sphere",
            size="0.009",
            rgba=ARM_GRIPPER_HIDDEN_SITE_RGBA,
        )
        ET.SubElement(
            body,
            "geom",
            name=f"arm_gripper_{side}_collision_pad_geom",
            type="box",
            pos="0 -0.0026 0.01875",
            size="0.030 0.016 0.036",
            rgba="0.10 0.55 0.80 0.0",
            mass="0",
            condim="6",
            friction="2.10 0.18 0.18",
            solref="0.026 1.8",
            solimp="0.92 0.995 0.001",
            priority="3",
        )


def add_arm_gripper_contact_pairs(root):
    contact = get_or_create_child(root, "contact")
    for geom_name in [*ARM_GRIPPER_PAD_GEOMS, *ARM_GRIPPER_PALM_GEOMS]:
        ET.SubElement(
            contact,
            "pair",
            geom1="moving_object",
            geom2=geom_name,
            condim="6",
            friction="2.00 0.18 0.18",
            solref="0.030 1.9",
            solimp="0.92 0.995 0.001",
        )


def add_ur5e_robotiq_arm(root, worldbody, arm_spec, *, include_contacts):
    enable_autolimits(root)
    model_scale = float(arm_spec.get("model_scale", 1.0))
    ur5e_root, ur5e_mesh_names, ur5e_material_names = prefixed_ur5e_bundle()
    robotiq_root, robotiq_mesh_names, robotiq_material_names = prefixed_robotiq_bundle()
    append_prefixed_menagerie_model(
        root,
        ur5e_root,
        robot_assets.UR5E_DIR,
        UR5E_PREFIX,
        ur5e_mesh_names,
        ur5e_material_names,
        model_scale=model_scale,
    )
    append_prefixed_menagerie_model(
        root,
        robotiq_root,
        robot_assets.ROBOTIQ_2F85_DIR,
        ROBOTIQ_PREFIX,
        robotiq_mesh_names,
        robotiq_material_names,
        model_scale=model_scale,
    )

    ur5e_body = robot_assets.prefix_menagerie_tree(
        ur5e_root.find("worldbody/body"),
        UR5E_PREFIX,
        ur5e_mesh_names,
        ur5e_material_names,
    )
    scale_xml_spatial_attributes(ur5e_body, model_scale)
    ur5e_body.set("name", "arm_gripper_ur5e_base")
    ur5e_body.set("pos", format_vec(arm_spec["base_position"]))
    ur5e_body.attrib.pop("quat", None)
    ur5e_body.set("euler", format_vec(arm_spec.get("base_euler", (0.0, 0.0, 0.0))))
    disable_ur5e_link_collisions(ur5e_body)

    wrist_body = robot_assets.find_body_recursive(ur5e_body, "ur5e_wrist_3_link")
    if wrist_body is None:
        raise ValueError("Could not find UR5e wrist_3_link body.")
    attachment_site = robot_assets.find_direct_site(wrist_body, "ur5e_attachment_site")
    if attachment_site is None:
        raise ValueError("Could not find UR5e attachment_site.")
    ET.SubElement(
        wrist_body,
        "site",
        name=ARM_GRIPPER_TOOL_SITE,
        pos=attachment_site.get("pos", "0 0 0"),
        quat=attachment_site.get("quat", "1 0 0 0"),
        size=format_scalar(0.012 * model_scale),
        rgba=ARM_GRIPPER_HIDDEN_SITE_RGBA,
    )
    mount_body = ET.SubElement(
        wrist_body,
        "body",
        name="arm_gripper_mount",
        pos=attachment_site.get("pos", "0 0 0"),
        quat=attachment_site.get("quat", "1 0 0 0"),
    )

    robotiq_body = robot_assets.prefix_menagerie_tree(
        robotiq_root.find("worldbody/body"),
        ROBOTIQ_PREFIX,
        robotiq_mesh_names,
        robotiq_material_names,
    )
    add_arm_gripper_sites_and_collision(robotiq_body)
    scale_xml_spatial_attributes(robotiq_body, model_scale)
    disable_fragile_robotiq_mesh_collisions(robotiq_body)
    mount_body.append(robotiq_body)
    worldbody.append(ur5e_body)

    actuator = get_or_create_child(root, "actuator")
    for child in ur5e_root.find("actuator"):
        prefixed = robot_assets.prefix_menagerie_tree(
            child,
            UR5E_PREFIX,
            ur5e_mesh_names,
            ur5e_material_names,
        )
        scale_general_actuator_strength(prefixed, model_scale)
        actuator.append(prefixed)
    for child in robotiq_root.find("actuator"):
        prefixed = robot_assets.prefix_menagerie_tree(
            child,
            ROBOTIQ_PREFIX,
            robotiq_mesh_names,
            robotiq_material_names,
        )
        scale_general_actuator_strength(prefixed, model_scale)
        actuator.append(prefixed)
    for tag in ("contact", "tendon", "equality"):
        element = robotiq_root.find(tag)
        if element is None:
            continue
        if tag == "contact":
            target = get_or_create_child(root, "contact")
            for child in element:
                target.append(
                    robot_assets.prefix_menagerie_tree(
                        child,
                        ROBOTIQ_PREFIX,
                        robotiq_mesh_names,
                        robotiq_material_names,
                    )
                )
        else:
            root.append(
                robot_assets.prefix_menagerie_tree(
                    element,
                    ROBOTIQ_PREFIX,
                    robotiq_mesh_names,
                    robotiq_material_names,
                )
            )
    if include_contacts:
        add_arm_gripper_contact_pairs(root)


def add_unitree_z1_arm(root, worldbody, catcher_spec, arm_spec, *, include_net):
    z1_root = load_unitree_z1_root()
    model_scale = float(arm_spec.get("model_scale", 1.0))
    append_unitree_z1_defaults(root, z1_root)
    append_unitree_z1_assets(root, z1_root, model_scale)
    z1_body = copy.deepcopy(z1_root.find("worldbody/body"))
    scale_unitree_z1_tree(z1_body, model_scale)
    disable_unitree_z1_link_collisions(z1_body)
    z1_body.set("name", "unitree_z1_base")
    z1_body.set("pos", format_vec(arm_spec["base_position"]))
    z1_body.set("euler", format_vec(arm_spec.get("base_euler", (0.0, 0.0, 0.0))))
    link06 = find_body_recursive(z1_body, "link06")
    if link06 is None:
        raise ValueError("Could not find Unitree Z1 link06 body.")
    tip_local_x = float(arm_spec.get("tool_tip_local_x", ARM_CATCHER_TOOL_TIP_LOCAL_X))
    tip_local_pos = format_vec((tip_local_x * model_scale, 0.0, 0.0))
    ET.SubElement(
        link06,
        "site",
        name=ARM_CATCHER_EE_SITE,
        pos=tip_local_pos,
        size=format_scalar(0.015 * model_scale),
        rgba="0 0 0 0",
    )
    if include_net:
        mount_body = ET.SubElement(
            worldbody,
            "body",
            name=ARM_CATCHER_MOUNT_BODY,
            mocap="true",
            pos=format_vec(catcher_spec.get("initial_position", (0.0, 0.0, 0.0))),
            quat="1 0 0 0",
        )
        for part in catcher_spec["parts"]:
            part_material = part.get("material", catcher_spec["material"])
            attrs = {
                "name": part["name"],
                "type": "box",
                "pos": format_vec(part["offset"]),
                "size": format_vec(part["half_extents"]),
                "rgba": format_vec(part_material.get("base_color", [0.16, 0.22, 0.32, 1.0])),
            }
            if part.get("collision", True):
                attrs.update(
                    {
                        "condim": "6",
                        "friction": "0.65 0.04 0.04",
                        "solref": "0.035 1",
                    }
                )
            else:
                attrs.update(
                    {
                        "contype": "0",
                        "conaffinity": "0",
                    }
                )
            ET.SubElement(mount_body, "geom", attrs)
    worldbody.append(z1_body)

    z1_actuator = z1_root.find("actuator")
    if z1_actuator is not None:
        actuator = get_or_create_child(root, "actuator")
        for child in z1_actuator:
            if child.get("joint") == "jointGripper":
                continue
            actuator.append(copy.deepcopy(child))

    if include_net:
        contact = get_or_create_child(root, "contact")
        for part in catcher_spec["parts"]:
            if not part.get("collision", True):
                continue
            ET.SubElement(
                contact,
                "pair",
                geom1="moving_object",
                geom2=part["name"],
                condim="6",
                friction="0.65 0.04 0.04",
                solref="0.035 1",
            )


def add_arm_catcher_body(root, worldbody, catcher_spec, arm_spec):
    base_position = arm_spec["base_position"]
    shoulder_height = float(arm_spec["shoulder_height"])
    upper_length = float(arm_spec["upper_link_length"])
    lower_length = float(arm_spec["lower_link_length"])
    link_half_t = float(arm_spec["link_half_thickness"])
    arm_rgba = arm_spec["material"].get("base_color", [0.18, 0.21, 0.26, 1.0])
    joint_rgba = arm_spec["joint_material"].get("base_color", [0.34, 0.38, 0.45, 1.0])
    joint_limits = arm_spec["joint_limits"]

    base = ET.SubElement(
        worldbody,
        "body",
        name="arm_base",
        pos=format_vec(base_position),
    )
    ET.SubElement(
        base,
        "joint",
        name="shoulder_pan",
        type="hinge",
        axis="0 0 1",
        limited="true",
        range=f"{format_scalar(joint_limits[0][0])} {format_scalar(joint_limits[0][1])}",
        damping="12",
        armature="0.08",
    )
    ET.SubElement(
        base,
        "geom",
        name="arm_base_pedestal",
        type="box",
        pos=format_vec((0.0, 0.0, 0.5 * shoulder_height)),
        size=format_vec((0.18, 0.18, 0.5 * shoulder_height)),
        rgba=format_vec(joint_rgba),
        mass="1.0",
        contype="0",
        conaffinity="0",
    )

    shoulder = ET.SubElement(
        base,
        "body",
        name="arm_upper_link",
        pos=format_vec((0.0, 0.0, shoulder_height)),
    )
    ET.SubElement(
        shoulder,
        "joint",
        name="shoulder_lift",
        type="hinge",
        axis="1 0 0",
        limited="true",
        range=f"{format_scalar(joint_limits[1][0])} {format_scalar(joint_limits[1][1])}",
        damping="16",
        armature="0.08",
    )
    ET.SubElement(
        shoulder,
        "geom",
        name="arm_shoulder_block",
        type="box",
        pos="0 0 0",
        size=format_vec((0.16, 0.16, 0.13)),
        rgba=format_vec(joint_rgba),
        mass="0.6",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        shoulder,
        "geom",
        name="arm_upper_link_geom",
        type="box",
        pos=format_vec((0.0, 0.5 * upper_length, 0.0)),
        size=format_vec((link_half_t, 0.5 * upper_length, link_half_t)),
        rgba=format_vec(arm_rgba),
        mass="0.8",
        contype="0",
        conaffinity="0",
    )

    elbow = ET.SubElement(
        shoulder,
        "body",
        name="arm_lower_link",
        pos=format_vec((0.0, upper_length, 0.0)),
    )
    ET.SubElement(
        elbow,
        "joint",
        name="elbow_flex",
        type="hinge",
        axis="1 0 0",
        limited="true",
        range=f"{format_scalar(joint_limits[2][0])} {format_scalar(joint_limits[2][1])}",
        damping="14",
        armature="0.08",
    )
    ET.SubElement(
        elbow,
        "geom",
        name="arm_elbow_block",
        type="box",
        pos="0 0 0",
        size=format_vec((0.14, 0.14, 0.12)),
        rgba=format_vec(joint_rgba),
        mass="0.5",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        elbow,
        "geom",
        name="arm_lower_link_geom",
        type="box",
        pos=format_vec((0.0, 0.5 * lower_length, 0.0)),
        size=format_vec((link_half_t, 0.5 * lower_length, link_half_t)),
        rgba=format_vec(arm_rgba),
        mass="0.7",
        contype="0",
        conaffinity="0",
    )

    wrist_1 = ET.SubElement(
        elbow,
        "body",
        name="arm_wrist_1",
        pos=format_vec((0.0, lower_length, 0.0)),
    )
    ET.SubElement(
        wrist_1,
        "joint",
        name="wrist_1_pitch",
        type="hinge",
        axis="1 0 0",
        limited="true",
        range=f"{format_scalar(joint_limits[3][0])} {format_scalar(joint_limits[3][1])}",
        damping="10",
        armature="0.04",
    )
    ET.SubElement(
        wrist_1,
        "geom",
        name="arm_wrist_1_block",
        type="box",
        pos="0 0 0",
        size=format_vec((0.12, 0.12, 0.10)),
        rgba=format_vec(joint_rgba),
        mass="0.28",
        contype="0",
        conaffinity="0",
    )
    wrist_2 = ET.SubElement(
        wrist_1,
        "body",
        name="arm_wrist_2",
        pos="0 0 0",
    )
    ET.SubElement(
        wrist_2,
        "joint",
        name="wrist_2_roll",
        type="hinge",
        axis="0 1 0",
        limited="true",
        range=f"{format_scalar(joint_limits[4][0])} {format_scalar(joint_limits[4][1])}",
        damping="8",
        armature="0.03",
    )
    ET.SubElement(
        wrist_2,
        "geom",
        name="arm_wrist_2_block",
        type="box",
        pos="0 0 0",
        size=format_vec((0.11, 0.11, 0.09)),
        rgba=format_vec(joint_rgba),
        mass="0.24",
        contype="0",
        conaffinity="0",
    )
    wrist_3 = ET.SubElement(
        wrist_2,
        "body",
        name="arm_wrist_3",
        pos="0 0 0",
    )
    ET.SubElement(
        wrist_3,
        "joint",
        name="wrist_3_yaw",
        type="hinge",
        axis="0 0 1",
        limited="true",
        range=f"{format_scalar(joint_limits[5][0])} {format_scalar(joint_limits[5][1])}",
        damping="8",
        armature="0.03",
    )
    ET.SubElement(
        wrist_3,
        "geom",
        name="arm_wrist_3_block",
        type="box",
        pos="0 0 0",
        size=format_vec((0.10, 0.10, 0.08)),
        rgba=format_vec(joint_rgba),
        mass="0.22",
        contype="0",
        conaffinity="0",
    )
    ee = ET.SubElement(
        wrist_3,
        "body",
        name="arm_catcher_ee",
        pos="0 0 0",
    )
    ET.SubElement(
        ee,
        "geom",
        name="arm_wrist_block",
        type="box",
        pos="0 0 0",
        size=format_vec((0.13, 0.13, 0.13)),
        rgba=format_vec(joint_rgba),
        mass="0.45",
        contype="0",
        conaffinity="0",
    )

    for part in catcher_spec["parts"]:
        part_material = part.get("material", catcher_spec["material"])
        attrs = {
            "name": part["name"],
            "type": "box",
            "pos": format_vec(part["offset"]),
            "size": format_vec(part["half_extents"]),
            "rgba": format_vec(part_material.get("base_color", [0.16, 0.22, 0.32, 1.0])),
        }
        if part.get("collision", True):
            attrs.update(
                {
                    "mass": "0.35",
                    "condim": "6",
                    "friction": "0.65 0.04 0.04",
                }
            )
        else:
            attrs.update(
                {
                    "mass": "0.02",
                    "contype": "0",
                    "conaffinity": "0",
                }
            )
        ET.SubElement(ee, "geom", attrs)

    contact = ET.SubElement(root, "contact")
    for part in catcher_spec["parts"]:
        if not part.get("collision", True):
            continue
        ET.SubElement(
            contact,
            "pair",
            geom1="moving_object",
            geom2=part["name"],
            condim="6",
            friction="0.65 0.04 0.04",
            solref="0.035 1",
        )

    actuator = ET.SubElement(root, "actuator")
    for index, joint_name in enumerate(arm_spec["joint_names"]):
        joint_min, joint_max = joint_limits[index]
        ET.SubElement(
            actuator,
            "position",
            name=f"{joint_name}_act",
            joint=joint_name,
            kp="360",
            ctrlrange=f"{format_scalar(joint_min)} {format_scalar(joint_max)}",
            ctrllimited="true",
            forcerange="-180 180",
            forcelimited="true",
        )
    return base


def build_model(scene, *, step_rate):
    root = ET.Element("mujoco", model=f"{scene['metadata']['scenario']}_3d_sim")
    add_root_elements(root, scene["metadata"]["gravity"], timestep=1.0 / step_rate)
    worldbody = ET.SubElement(root, "worldbody")
    for spec in scene["static_objects"]:
        if spec.get("render_only", False):
            continue
        add_static_geom(worldbody, spec)
    dynamic_spec = scene["dynamic_objects"][0]
    add_dynamic_body(worldbody, dynamic_spec)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    return model, data


def build_paddle_model(scene, *, step_rate):
    root = ET.Element("mujoco", model=f"{scene['metadata']['scenario']}_3d_sim")
    add_root_elements(root, scene["metadata"]["gravity"], timestep=1.0 / step_rate)
    worldbody = ET.SubElement(root, "worldbody")
    for spec in scene["static_objects"]:
        if spec.get("render_only", False):
            continue
        add_static_geom(worldbody, spec)
    add_dynamic_body(worldbody, scene["dynamic_objects"][0])
    add_paddle_body(root, worldbody, scene["paddle"])
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    return model, data


def build_catcher_model(scene, *, step_rate):
    root = ET.Element("mujoco", model=f"{scene['metadata']['scenario']}_3d_sim")
    add_root_elements(root, scene["metadata"]["gravity"], timestep=1.0 / step_rate)
    worldbody = ET.SubElement(root, "worldbody")
    for spec in scene["static_objects"]:
        if spec.get("render_only", False):
            continue
        add_static_geom(worldbody, spec)
    add_dynamic_body(worldbody, scene["dynamic_objects"][0])
    add_catcher_body(root, worldbody, scene["catcher"])
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    return model, data


def build_arm_catcher_model(scene, *, step_rate):
    root = ET.Element("mujoco", model=f"{scene['metadata']['scenario']}_3d_sim")
    add_root_elements(root, scene["metadata"]["gravity"], timestep=1.0 / step_rate)
    worldbody = ET.SubElement(root, "worldbody")
    for spec in scene["static_objects"]:
        if spec.get("render_only", False):
            continue
        add_static_geom(worldbody, spec)
    add_dynamic_body(worldbody, scene["dynamic_objects"][0])
    add_moving_object_static_contacts(root, scene["static_objects"])
    add_unitree_z1_arm(root, worldbody, scene["catcher"], scene["arm"], include_net=True)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    return model, data


def build_arm_gripper_model(scene, *, step_rate):
    root = ET.Element("mujoco", model=f"{scene['metadata']['scenario']}_3d_sim")
    add_root_elements(root, scene["metadata"]["gravity"], timestep=1.0 / step_rate)
    worldbody = ET.SubElement(root, "worldbody")
    for spec in scene["static_objects"]:
        if spec.get("render_only", False):
            continue
        add_static_geom(worldbody, spec)
    add_dynamic_body(worldbody, scene["dynamic_objects"][0])
    add_ur5e_robotiq_arm(root, worldbody, scene["arm"], include_contacts=True)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    return model, data


def build_arm_paddle_model(scene, *, step_rate):
    root = ET.Element("mujoco", model=f"{scene['metadata']['scenario']}_3d_sim")
    add_root_elements(root, scene["metadata"]["gravity"], timestep=1.0 / step_rate)
    worldbody = ET.SubElement(root, "worldbody")
    for spec in scene["static_objects"]:
        if spec.get("render_only", False):
            continue
        add_static_geom(worldbody, spec)
    add_dynamic_body(worldbody, scene["dynamic_objects"][0])
    add_paddle_body(root, worldbody, scene["paddle"])
    add_unitree_z1_arm(root, worldbody, None, scene["arm"], include_net=False)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    return model, data


def initialize_dynamic_state(model, data, spec):
    if spec.get("lock_rotation", spec["kind"] == "ball"):
        for axis_name, position_value, velocity_value in zip(
            ("x", "y", "z"), spec["position"], spec["initial_velocity"], strict=False
        ):
            joint = model.joint(f"{spec['name']}_slide_{axis_name}")
            data.qpos[int(joint.qposadr[0])] = position_value
            data.qvel[int(joint.dofadr[0])] = velocity_value
    else:
        joint = model.joint(f"{spec['name']}_free")
        qpos_adr = int(joint.qposadr[0])
        dof_adr = int(joint.dofadr[0])
        initial_quaternion_xyzw = spec.get("initial_quaternion")
        quaternion_xyzw = (
            [float(value) for value in initial_quaternion_xyzw]
            if initial_quaternion_xyzw is not None
            else euler_xyz_to_quaternion_xyzw(spec.get("rotation_euler", (0.0, 0.0, 0.0)))
        )
        data.qpos[qpos_adr : qpos_adr + 3] = spec["position"]
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = xyzw_to_wxyz(quaternion_xyzw)
        data.qvel[dof_adr : dof_adr + 3] = spec["initial_velocity"]
        data.qvel[dof_adr + 3 : dof_adr + 6] = spec["initial_angular_velocity"]
    mujoco.mj_forward(model, data)


def initialize_catcher_state(model, data, catcher_spec):
    for axis_name, position_value in (
        ("x", catcher_spec["initial_position"][0]),
        ("z", catcher_spec["initial_position"][2]),
    ):
        joint = model.joint(f"{catcher_spec['name']}_slide_{axis_name}")
        data.qpos[int(joint.qposadr[0])] = position_value
        data.qvel[int(joint.dofadr[0])] = 0.0
    set_catcher_ctrl(data, catcher_spec["initial_target_position"])
    mujoco.mj_forward(model, data)


def set_catcher_ctrl(data, target_position):
    data.ctrl[0] = target_position[0]
    data.ctrl[1] = target_position[2]


def catcher_joint_state(model, data, catcher_spec):
    positions = [0.0, float(catcher_spec["fixed_y"]), 0.0]
    velocities = [0.0, 0.0, 0.0]
    for axis_name, vector_index in (("x", 0), ("z", 2)):
        joint = model.joint(f"{catcher_spec['name']}_slide_{axis_name}")
        positions[vector_index] = float(data.qpos[int(joint.qposadr[0])])
        velocities[vector_index] = float(data.qvel[int(joint.dofadr[0])])
    return positions, velocities


def initialize_paddle_state(model, data, paddle_spec):
    for axis_name, position_value in zip(
        ("x", "y", "z"), paddle_spec["initial_position"], strict=False
    ):
        joint = model.joint(f"{paddle_spec['name']}_slide_{axis_name}")
        data.qpos[int(joint.qposadr[0])] = position_value
        data.qvel[int(joint.dofadr[0])] = 0.0
    if paddle_spec.get("tilt_enabled", False):
        roll, pitch = paddle_tilt_to_roll_pitch(
            paddle_spec.get("initial_target_phi", 0.0),
            paddle_spec.get("initial_target_theta", 0.0),
            paddle_spec["max_tilt_theta"],
        )
        for axis_name, value in (("roll", roll), ("pitch", pitch)):
            joint = model.joint(f"{paddle_spec['name']}_hinge_{axis_name}")
            data.qpos[int(joint.qposadr[0])] = value
            data.qvel[int(joint.dofadr[0])] = 0.0
    set_paddle_ctrl(
        data,
        paddle_spec["initial_target_position"],
        paddle_spec,
        paddle_spec.get("initial_target_phi", 0.0),
        paddle_spec.get("initial_target_theta", 0.0),
    )
    mujoco.mj_forward(model, data)


def set_paddle_ctrl(data, target_position, paddle_spec=None, target_phi=0.0, target_theta=0.0):
    data.ctrl[0] = target_position[0]
    data.ctrl[1] = target_position[1]
    data.ctrl[2] = target_position[2]
    if paddle_spec is not None and paddle_spec.get("tilt_enabled", False):
        roll, pitch = paddle_tilt_to_roll_pitch(
            target_phi,
            target_theta,
            paddle_spec["max_tilt_theta"],
        )
        data.ctrl[3] = roll
        data.ctrl[4] = pitch


def paddle_joint_state(model, data, paddle_spec):
    positions = [0.0, 0.0, 0.0]
    velocities = [0.0, 0.0, 0.0]
    for axis_name, vector_index in (("x", 0), ("y", 1), ("z", 2)):
        joint = model.joint(f"{paddle_spec['name']}_slide_{axis_name}")
        positions[vector_index] = float(data.qpos[int(joint.qposadr[0])])
        velocities[vector_index] = float(data.qvel[int(joint.dofadr[0])])
    return positions, velocities


def paddle_orientation_state(model, data, paddle_spec, *, fallback_phi=0.0):
    if not paddle_spec.get("tilt_enabled", False):
        return {
            "orientation_phi": 0.0,
            "orientation_theta": 0.0,
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "angular_velocity": [0.0, 0.0, 0.0],
        }
    body = data.body(paddle_spec["name"])
    quaternion = normalize_quaternion_xyzw(wxyz_to_xyzw(body.xquat.copy()))
    matrix = [float(value) for value in body.xmat.copy()]
    normal = (matrix[2], matrix[5], matrix[8])
    phi, theta = paddle_normal_to_phi_theta(normal, fallback_phi=fallback_phi)
    angular_velocity = [float(value) for value in body.cvel[:3].copy()]
    return {
        "orientation_phi": float(phi),
        "orientation_theta": float(theta),
        "quaternion_xyzw": [float(value) for value in quaternion],
        "angular_velocity": angular_velocity,
    }


def set_locked_ball_state(
    model, data, ball_spec, position, velocity, *, angular_velocity=(0.0, 0.0, 0.0)
):
    """Force the ball's translational state.

    A rotation-locked ball rides on three slide joints; a free-rotating one on a
    freejoint, where the orientation is left untouched and the spin is reset by
    default (a ball held in a net or gripper is held still). Pass
    ``angular_velocity=None`` to leave the spin alone -- for corrections applied
    to a ball that is still in flight.
    """
    if ball_spec.get("lock_rotation", ball_spec["kind"] == "ball"):
        for axis_name, position_value, velocity_value in zip(
            ("x", "y", "z"), position, velocity, strict=False
        ):
            joint = model.joint(f"{ball_spec['name']}_slide_{axis_name}")
            data.qpos[int(joint.qposadr[0])] = position_value
            data.qvel[int(joint.dofadr[0])] = velocity_value
    else:
        joint = model.joint(f"{ball_spec['name']}_free")
        qpos_adr = int(joint.qposadr[0])
        dof_adr = int(joint.dofadr[0])
        data.qpos[qpos_adr : qpos_adr + 3] = position
        data.qvel[dof_adr : dof_adr + 3] = velocity
        if angular_velocity is not None:
            data.qvel[dof_adr + 3 : dof_adr + 6] = angular_velocity
    mujoco.mj_forward(model, data)


def enforce_ball_ground_bounce(model, data, ball_spec, radius, *, ground_z=0.0):
    position, _quaternion, linear_velocity, _angular_velocity = body_state(data, ball_spec["name"])
    min_center_z = float(ground_z) + float(radius)
    if float(position[2]) >= min_center_z:
        return False
    restitution = float(ball_spec.get("dynamics", {}).get("restitution", UNIFORM_RESTITUTION))
    restitution = min(max(restitution, 0.0), 1.0)
    corrected_velocity = [
        float(linear_velocity[0]),
        float(linear_velocity[1]),
        abs(float(linear_velocity[2])) * restitution,
    ]
    set_locked_ball_state(
        model,
        data,
        ball_spec,
        [float(position[0]), float(position[1]), min_center_z],
        corrected_velocity,
        angular_velocity=None,
    )
    return True


def clip_value(value, lower, upper):
    return min(max(value, lower), upper)


def clip_vector(values, lower, upper):
    return [
        clip_value(float(value), float(lower_value), float(upper_value))
        for value, lower_value, upper_value in zip(values, lower, upper, strict=False)
    ]


def arm_ik_for_target(arm_spec, target_position):
    base_x, base_y, base_z = [float(value) for value in arm_spec["base_position"]]
    shoulder_z = base_z + float(arm_spec["shoulder_height"])
    upper_length = float(arm_spec["upper_link_length"])
    lower_length = float(arm_spec["lower_link_length"])
    joint_limits = arm_spec["joint_limits"]

    dx = float(target_position[0]) - base_x
    dy = float(target_position[1]) - base_y
    dz = float(target_position[2]) - shoulder_z
    yaw = math.atan2(-dx, dy)
    radial = math.sqrt(dx * dx + dy * dy)
    radial = max(radial, 1e-4)

    distance = math.sqrt(radial * radial + dz * dz)
    min_reach = abs(upper_length - lower_length) + 0.04
    max_reach = upper_length + lower_length - 0.04
    if distance < min_reach:
        scale = min_reach / max(distance, 1e-4)
        radial *= scale
        dz *= scale
        distance = min_reach
    elif distance > max_reach:
        scale = max_reach / max(distance, 1e-4)
        radial *= scale
        dz *= scale
        distance = max_reach

    cos_elbow = (
        distance * distance - upper_length * upper_length - lower_length * lower_length
    ) / (2.0 * upper_length * lower_length)
    cos_elbow = clip_value(cos_elbow, -1.0, 1.0)
    elbow = math.acos(cos_elbow)
    shoulder = math.atan2(dz, radial) - math.atan2(
        lower_length * math.sin(elbow),
        upper_length + lower_length * math.cos(elbow),
    )

    wrist_1 = -(shoulder + elbow)
    wrist_2 = 0.0
    wrist_3 = -yaw
    targets = [yaw, shoulder, elbow, wrist_1, wrist_2, wrist_3]
    return [
        clip_value(targets[index], joint_limits[index][0], joint_limits[index][1])
        for index in range(len(targets))
    ]


def unitree_z1_joint_addresses(model, arm_spec):
    qpos_adrs = []
    dof_adrs = []
    for joint_name in arm_spec["joint_names"]:
        joint = model.joint(joint_name)
        qpos_adrs.append(int(joint.qposadr[0]))
        dof_adrs.append(int(joint.dofadr[0]))
    return qpos_adrs, dof_adrs


def solve_unitree_z1_ik(
    model, data, arm_spec, target_position, seed_qpos, *, allow_fallback_seeds=True
):
    qpos_adrs, dof_adrs = unitree_z1_joint_addresses(model, arm_spec)
    joint_limits = arm_spec["joint_limits"]
    home_qpos = np.asarray(arm_spec.get("home_qpos", seed_qpos), dtype=np.float64)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ARM_CATCHER_EE_SITE)
    if site_id < 0:
        raise ValueError(f"Missing {ARM_CATCHER_EE_SITE} in Unitree Z1 model.")
    target = np.asarray(target_position, dtype=np.float64)
    desired_normal = np.asarray(
        arm_spec.get("tip_normal_world", ARM_CATCHER_TIP_NORMAL_WORLD),
        dtype=np.float64,
    )
    desired_normal /= max(float(np.linalg.norm(desired_normal)), 1e-12)
    position_weight = 2.0
    normal_weight = 1.4
    # Bias the redundant DOF toward home so tracking does not wedge joints
    # against their limits (a corner the damped-least-squares step cannot
    # escape on its own, which previously required branch-flipping fallback
    # seeds and caused visible arm jumps).
    nullspace_gain = float(arm_spec.get("ik_nullspace_gain", 0.004))
    # Joints pinned to fixed values (name -> radians). For the arm catcher,
    # joint6 spins the flange about the constrained net normal without moving
    # the tip, i.e. it spans the task's whole self-motion; pinning it makes
    # the joint configuration a locally unique function of the EE position.
    locked_joints = {
        arm_spec["joint_names"].index(name): float(value)
        for name, value in (arm_spec.get("ik_locked_joints") or {}).items()
    }

    def apply_locked_joints(q):
        for joint_index, value in locked_joints.items():
            q[joint_index] = value

    def solve_from_seed(seed):
        q = np.asarray(seed, dtype=np.float64).copy()
        apply_locked_joints(q)
        best_q = q.copy()
        best_score = float("inf")
        float("inf")
        float("inf")
        iterations_since_best = 0
        for _ in range(220):
            for value, qpos_adr in zip(q, qpos_adrs, strict=False):
                data.qpos[qpos_adr] = float(value)
            mujoco.mj_forward(model, data)
            current = data.site_xpos[site_id].copy()
            rotation = data.site_xmat[site_id].reshape(3, 3).copy()
            current_normal = rotation[:, 0]
            position_error = target - current
            normal_cross_error = np.cross(current_normal, desired_normal)
            position_norm = float(np.linalg.norm(position_error))
            normal_vector_error = float(np.linalg.norm(current_normal - desired_normal))
            score = position_norm + 0.35 * normal_vector_error
            if score < best_score - 1.0e-7:
                best_score = score
                best_q = q.copy()
                iterations_since_best = 0
            else:
                iterations_since_best += 1
                if iterations_since_best >= 20:
                    break
            if position_norm < 1.0e-5 and normal_vector_error < 1.0e-5:
                best_q = q.copy()
                break

            jacp = np.zeros((3, model.nv), dtype=np.float64)
            jacr = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            jac = np.vstack(
                (
                    position_weight * jacp[:, dof_adrs],
                    normal_weight * jacr[:, dof_adrs],
                )
            )
            error = np.concatenate(
                (
                    position_weight * position_error,
                    normal_weight * normal_cross_error,
                )
            )
            lhs = jac @ jac.T + 2.5e-3 * np.eye(jac.shape[0])
            step = jac.T @ np.linalg.solve(lhs, 0.78 * error)
            # Project the home bias onto the task null space so it steers the
            # redundant DOF without fighting task convergence (an unprojected
            # bias leaves a millimetre-scale equilibrium offset that shows up
            # as stick-slip when tracking small per-frame steps).
            bias = nullspace_gain * (home_qpos - q)
            step += bias - jac.T @ np.linalg.solve(lhs, jac @ bias)
            step = np.clip(step, -0.10, 0.10)
            q += step
            for index, (joint_min, joint_max) in enumerate(joint_limits):
                q[index] = clip_value(q[index], joint_min, joint_max)
            apply_locked_joints(q)
        q = best_q.copy()
        best_refine_error = float("inf")
        refine_iterations_since_best = 0
        for _ in range(80):
            for value, qpos_adr in zip(q, qpos_adrs, strict=False):
                data.qpos[qpos_adr] = float(value)
            mujoco.mj_forward(model, data)
            rotation = data.site_xmat[site_id].reshape(3, 3).copy()
            current_normal = rotation[:, 0]
            normal_cross_error = np.cross(current_normal, desired_normal)
            normal_vector_error = float(np.linalg.norm(current_normal - desired_normal))
            if normal_vector_error < 1.0e-7:
                break
            if normal_vector_error < best_refine_error - 1.0e-9:
                best_refine_error = normal_vector_error
                refine_iterations_since_best = 0
            else:
                refine_iterations_since_best += 1
                if refine_iterations_since_best >= 10:
                    break
            jacp = np.zeros((3, model.nv), dtype=np.float64)
            jacr = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            jac = jacr[:, dof_adrs]
            lhs = jac @ jac.T + 1.0e-3 * np.eye(3)
            step = jac.T @ np.linalg.solve(lhs, 0.82 * normal_cross_error)
            step = np.clip(step, -0.06, 0.06)
            q += step
            for index, (joint_min, joint_max) in enumerate(joint_limits):
                q[index] = clip_value(q[index], joint_min, joint_max)
            apply_locked_joints(q)
        for value, qpos_adr in zip(q, qpos_adrs, strict=False):
            data.qpos[qpos_adr] = float(value)
        mujoco.mj_forward(model, data)
        refined_position = data.site_xpos[site_id].copy()
        refined_normal = data.site_xmat[site_id].reshape(3, 3)[:, 0].copy()
        refined_position_error = float(np.linalg.norm(refined_position - target))
        refined_normal_error = float(np.linalg.norm(refined_normal - desired_normal))
        return q, refined_position_error, refined_normal_error

    def deterministic_fallback_seeds():
        seeds = [home_qpos]
        seed_value = int(abs(target[0] * 1009.0 + target[1] * 917.0 + target[2] * 853.0) * 1000.0)
        rng = random.Random(seed_value)
        for _ in range(8):
            seeds.append(
                np.asarray(
                    [
                        rng.uniform(float(joint_min), float(joint_max))
                        for joint_min, joint_max in joint_limits
                    ],
                    dtype=np.float64,
                )
            )
        return seeds

    candidates = [solve_from_seed(seed_qpos)]
    if allow_fallback_seeds and (candidates[0][1] > 0.04 or candidates[0][2] > 0.02):
        candidates.extend(solve_from_seed(seed) for seed in deterministic_fallback_seeds())
    q, _, _ = min(candidates, key=lambda item: item[1] + 0.35 * item[2])
    for value, qpos_adr in zip(q, qpos_adrs, strict=False):
        data.qpos[qpos_adr] = float(value)
    mujoco.mj_forward(model, data)
    return [float(value) for value in q]


def set_arm_ctrl(data, joint_targets):
    for index, target in enumerate(joint_targets):
        data.ctrl[index] = float(target)


def arm_joint_state(model, data, arm_spec):
    positions = []
    velocities = []
    for joint_name in arm_spec["joint_names"]:
        joint = model.joint(joint_name)
        positions.append(float(data.qpos[int(joint.qposadr[0])]))
        velocities.append(float(data.qvel[int(joint.dofadr[0])]))
    return positions, velocities


def set_arm_joint_state(model, data, arm_spec, joint_positions, joint_velocities=None):
    if joint_velocities is None:
        joint_velocities = [0.0] * len(arm_spec["joint_names"])
    for joint_name, joint_position, joint_velocity in zip(
        arm_spec["joint_names"],
        joint_positions,
        joint_velocities,
        strict=False,
    ):
        joint = model.joint(joint_name)
        data.qpos[int(joint.qposadr[0])] = float(joint_position)
        data.qvel[int(joint.dofadr[0])] = float(joint_velocity)
    mujoco.mj_forward(model, data)


def initialize_arm_catcher_state(model, data, catcher_spec, arm_spec):
    target_position = clip_vector(
        catcher_spec["initial_target_position"],
        catcher_spec["workspace_min"],
        catcher_spec["workspace_max"],
    )
    joint_targets = solve_unitree_z1_ik(
        model,
        data,
        arm_spec,
        # Targets are net-back-face positions; the IK drives the flange.
        arm_catcher_tip_target_for_mount(
            target_position,
            catcher_spec.get("mount_back_offset", ARM_CATCHER_MOUNT_BACK_OFFSET),
            arm_spec.get("tip_normal_world", ARM_CATCHER_TIP_NORMAL_WORLD),
        ),
        arm_spec.get("home_qpos", [0.0] * len(arm_spec["joint_names"])),
    )
    for joint_name, joint_target in zip(arm_spec["joint_names"], joint_targets, strict=False):
        joint = model.joint(joint_name)
        data.qpos[int(joint.qposadr[0])] = joint_target
        data.qvel[int(joint.dofadr[0])] = 0.0
    set_arm_ctrl(data, joint_targets)
    mujoco.mj_forward(model, data)
    return target_position, joint_targets


def actuator_id(model, name):
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator < 0:
        raise ValueError(f"Missing actuator {name!r}.")
    return int(actuator)


def site_id(model, name):
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site < 0:
        raise ValueError(f"Missing site {name!r}.")
    return int(site)


def set_named_arm_ctrl(model, data, arm_spec, joint_targets):
    actuator_names = arm_spec.get("actuator_names")
    if actuator_names:
        for actuator_name, target in zip(actuator_names, joint_targets, strict=False):
            data.ctrl[actuator_id(model, actuator_name)] = float(target)
        return
    set_arm_ctrl(data, joint_targets)


def set_gripper_ctrl(model, data, grip_command):
    target = 255.0 if float(grip_command) >= 0.5 else 0.0
    data.ctrl[actuator_id(model, ROBOTIQ_ACTUATOR_NAME)] = target
    return target


def ur5e_joint_addresses(model, arm_spec):
    qpos_adrs = []
    dof_adrs = []
    for joint_name in arm_spec["joint_names"]:
        joint = model.joint(joint_name)
        qpos_adrs.append(int(joint.qposadr[0]))
        dof_adrs.append(int(joint.dofadr[0]))
    return qpos_adrs, dof_adrs


def solve_site_position_ik(
    model,
    data,
    arm_spec,
    site_name,
    target_position,
    seed_qpos,
    *,
    iterations=72,
    tolerance=0.006,
):
    qpos_adrs, dof_adrs = ur5e_joint_addresses(model, arm_spec)
    joint_limits = arm_spec["joint_limits"]
    home_qpos = np.asarray(arm_spec.get("home_qpos", seed_qpos), dtype=np.float64)
    q = np.asarray(seed_qpos, dtype=np.float64).copy()
    target = np.asarray(target_position, dtype=np.float64)
    ik_site_id = site_id(model, site_name)

    for _ in range(iterations):
        for value, qpos_adr in zip(q, qpos_adrs, strict=False):
            data.qpos[qpos_adr] = float(value)
        mujoco.mj_forward(model, data)
        current = data.site_xpos[ik_site_id].copy()
        error = target - current
        if float(np.linalg.norm(error)) < tolerance:
            break
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, ik_site_id)
        jac = jacp[:, dof_adrs]
        damping = 4.0e-3
        lhs = jac @ jac.T + damping * np.eye(3)
        step = jac.T @ np.linalg.solve(lhs, 0.72 * error)
        step += 0.010 * (home_qpos - q)
        step = np.clip(step, -0.14, 0.14)
        q += step
        for index, (joint_min, joint_max) in enumerate(joint_limits):
            q[index] = clip_value(q[index], joint_min, joint_max)

    for value, qpos_adr in zip(q, qpos_adrs, strict=False):
        data.qpos[qpos_adr] = float(value)
    mujoco.mj_forward(model, data)
    return [float(value) for value in q]


def initialize_arm_gripper_state(model, data, catcher_spec, arm_spec, gripper_spec):
    target_position = clip_vector(
        catcher_spec["initial_target_position"],
        catcher_spec["workspace_min"],
        catcher_spec["workspace_max"],
    )
    home_qpos = arm_spec.get("home_qpos", [0.0] * len(arm_spec["joint_names"]))
    set_arm_joint_state(model, data, arm_spec, home_qpos, [0.0] * len(home_qpos))
    joint_targets = solve_site_position_ik(
        model,
        data,
        arm_spec,
        ARM_GRIPPER_GRASP_SITE,
        target_position,
        home_qpos,
    )
    set_arm_joint_state(model, data, arm_spec, joint_targets, [0.0] * len(joint_targets))
    set_named_arm_ctrl(model, data, arm_spec, joint_targets)
    set_gripper_ctrl(model, data, gripper_spec.get("open_command", 0.0))
    mujoco.mj_forward(model, data)
    return target_position, joint_targets


def settle_arm_gripper_open_state(
    model,
    data,
    dynamic_spec,
    arm_spec,
    gripper_spec,
    joint_targets,
    *,
    steps=120,
):
    open_command = gripper_spec.get("open_command", 0.0)
    for _ in range(steps):
        set_named_arm_ctrl(model, data, arm_spec, joint_targets)
        set_gripper_ctrl(model, data, open_command)
        mujoco.mj_step(model, data)

    data.qvel[:] = 0.0
    set_named_arm_ctrl(model, data, arm_spec, joint_targets)
    set_gripper_ctrl(model, data, open_command)
    initialize_dynamic_state(model, data, dynamic_spec)
    data.time = 0.0
    mujoco.mj_forward(model, data)


def site_pose(model, data, name):
    site = site_id(model, name)
    position = data.site_xpos[site].copy()
    quaternion = mat_to_quaternion_xyzw(data.site_xmat[site].copy())
    return position, quaternion


def site_linear_velocity(model, data, name):
    site = site_id(model, name)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site)
    return jacp @ data.qvel


def site_local_position(model, data, site_name, world_position):
    site = site_id(model, site_name)
    rel = np.asarray(world_position, dtype=np.float64) - data.site_xpos[site]
    rotation = data.site_xmat[site].reshape(3, 3)
    return rotation.T @ rel


def mocap_id_for_body(model, body_name):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body < 0:
        raise ValueError(f"Missing mocap body {body_name!r}.")
    mocap_id = int(model.body_mocapid[body])
    if mocap_id < 0:
        raise ValueError(f"Body {body_name!r} is not a mocap body.")
    return mocap_id


def set_mocap_body_pose(
    model, data, body_name, position, quaternion_xyzw=ARM_CATCHER_WORLD_QUATERNION_XYZW
):
    mocap_id = mocap_id_for_body(model, body_name)
    data.mocap_pos[mocap_id] = np.asarray(position, dtype=np.float64)
    data.mocap_quat[mocap_id] = xyzw_to_wxyz(normalize_quaternion_xyzw(quaternion_xyzw))


def arm_catcher_mount_offset_vector(back_offset, tip_normal=ARM_CATCHER_TIP_NORMAL_WORLD):
    """Displacement from the flange to the net's back face, in world units."""
    normal = np.asarray(tip_normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    return -float(back_offset) * normal


def arm_catcher_tip_target_for_mount(
    mount_target, back_offset, tip_normal=ARM_CATCHER_TIP_NORMAL_WORLD
):
    """Flange position that puts the net's back face at mount_target."""
    return np.asarray(mount_target, dtype=np.float64) - arm_catcher_mount_offset_vector(
        back_offset, tip_normal
    )


def sync_arm_catcher_mount_to_tip(
    model,
    data,
    back_offset=ARM_CATCHER_MOUNT_BACK_OFFSET,
    tip_normal=ARM_CATCHER_TIP_NORMAL_WORLD,
):
    tip_position, _ = site_pose(model, data, ARM_CATCHER_EE_SITE)
    mount_target = np.asarray(tip_position, dtype=np.float64)
    mount_target = mount_target + arm_catcher_mount_offset_vector(back_offset, tip_normal)
    set_mocap_body_pose(
        model,
        data,
        ARM_CATCHER_MOUNT_BODY,
        mount_target,
        ARM_CATCHER_WORLD_QUATERNION_XYZW,
    )
    mujoco.mj_forward(model, data)
    tip_velocity = site_linear_velocity(model, data, ARM_CATCHER_EE_SITE)
    mount_position = data.body(ARM_CATCHER_MOUNT_BODY).xpos.copy()
    return mount_position, tip_velocity


def sync_arm_catcher_mount_for_scene(model, data, catcher_spec, arm_spec):
    """sync_arm_catcher_mount_to_tip with the offset a scene's specs imply."""
    return sync_arm_catcher_mount_to_tip(
        model,
        data,
        back_offset=catcher_spec.get("mount_back_offset", ARM_CATCHER_MOUNT_BACK_OFFSET),
        tip_normal=arm_spec.get("tip_normal_world", ARM_CATCHER_TIP_NORMAL_WORLD),
    )


def catcher_origin_is_back_face(catcher_spec):
    return catcher_spec.get("mount_origin") == "back_face_center"


def catcher_depth_bounds(catcher_spec):
    depth = float(catcher_spec["net_depth"])
    if catcher_origin_is_back_face(catcher_spec):
        return 0.0, depth
    half_depth = 0.5 * depth
    return -half_depth, half_depth


def catcher_front_y_offset(catcher_spec):
    return catcher_depth_bounds(catcher_spec)[1]


def catcher_latch_y_bounds(catcher_spec):
    if "latch_y_min" in catcher_spec and "latch_y_max" in catcher_spec:
        return float(catcher_spec["latch_y_min"]), float(catcher_spec["latch_y_max"])
    depth = float(catcher_spec["net_depth"])
    if catcher_origin_is_back_face(catcher_spec):
        return 0.08 * depth, 0.75 * depth
    half_depth = 0.5 * depth
    return -0.50 * half_depth, 0.25 * half_depth


def catcher_capture_center_limits(catcher_spec):
    return (
        float(
            catcher_spec.get(
                "capture_center_half_width", 0.45 * float(catcher_spec["net_half_width"])
            )
        ),
        float(
            catcher_spec.get(
                "capture_center_half_height", 0.45 * float(catcher_spec["net_half_height"])
            )
        ),
        float(
            catcher_spec.get(
                "capture_center_radius",
                0.48
                * min(
                    float(catcher_spec["net_half_width"]), float(catcher_spec["net_half_height"])
                ),
            )
        ),
    )


def catcher_inside_capture_center(rel_pos, catcher_spec):
    center_half_width, center_half_height, center_radius = catcher_capture_center_limits(
        catcher_spec
    )
    center_distance = math.hypot(float(rel_pos[0]), float(rel_pos[2]))
    return (
        abs(float(rel_pos[0])) <= center_half_width
        and abs(float(rel_pos[2])) <= center_half_height
        and center_distance <= center_radius
    )


def catcher_reached_front_plane(ball_position, catcher_position, catcher_spec, radius=0.0):
    rel_y = float(ball_position[1] - catcher_position[1])
    radius_margin = max(0.08, 0.55 * float(radius))
    return rel_y <= catcher_front_y_offset(catcher_spec) + radius_margin


def validate_arm_catcher_mount(
    model,
    data,
    catcher_spec,
    *,
    position_tolerance=1e-6,
    axis_tolerance=1e-6,
    back_offset=None,
    tip_normal=ARM_CATCHER_TIP_NORMAL_WORLD,
):
    if back_offset is None:
        back_offset = catcher_spec.get("mount_back_offset", ARM_CATCHER_MOUNT_BACK_OFFSET)
    tip_site = site_id(model, ARM_CATCHER_EE_SITE)
    tip_position = data.site_xpos[tip_site].copy()
    # The mount is seated back from the flange by design, so the expected mount
    # position is the flange displaced along the net normal.
    expected_mount = tip_position + arm_catcher_mount_offset_vector(back_offset, tip_normal)
    tip_rotation = data.site_xmat[tip_site].reshape(3, 3).copy()
    mount_body = data.body(ARM_CATCHER_MOUNT_BODY)
    mount_position = mount_body.xpos.copy()
    rotation = mount_body.xmat.reshape(3, 3).copy()
    open_axis = np.asarray(
        catcher_spec.get("open_axis_world", ARM_CATCHER_OPEN_AXIS_WORLD), dtype=np.float64
    )
    open_axis /= max(float(np.linalg.norm(open_axis)), 1e-12)
    flange_circle_normal = tip_rotation[:, 0]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ARM_CATCHER_MOUNT_BODY)
    parent_id = int(model.body_parentid[body_id])
    parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) or "world"
    geom_body_ok = True
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if geom_name.startswith("arm_catcher_") and int(model.geom_bodyid[geom_id]) != body_id:
            geom_body_ok = False
            break
    check = {
        "tip_mount_position_error": float(np.linalg.norm(expected_mount - mount_position)),
        "mount_back_offset": float(back_offset),
        "flange_circle_normal_world": [float(value) for value in flange_circle_normal],
        "flange_circle_normal_error": float(np.linalg.norm(flange_circle_normal - open_axis)),
        "flange_circle_plane_to_catcher_plane_distance": float(
            abs(tip_position[1] - mount_position[1])
        ),
        "x_axis_world_error": float(np.linalg.norm(rotation[:, 0] - np.asarray((1.0, 0.0, 0.0)))),
        "z_axis_world_error": float(np.linalg.norm(rotation[:, 2] - np.asarray((0.0, 0.0, 1.0)))),
        "open_axis_world_error": float(np.linalg.norm(rotation[:, 1] - open_axis)),
        "mount_body_parent": parent_name,
        "mount_body_parent_is_world": bool(parent_id == 0),
        "catcher_geoms_on_mount_body": bool(geom_body_ok),
        "mount_origin": catcher_spec.get("mount_origin"),
        "open_axis_world": [float(value) for value in open_axis],
    }
    if (
        check["tip_mount_position_error"] > position_tolerance
        or check["x_axis_world_error"] > axis_tolerance
        or check["z_axis_world_error"] > axis_tolerance
        or check["open_axis_world_error"] > axis_tolerance
        or not check["mount_body_parent_is_world"]
        or not check["catcher_geoms_on_mount_body"]
    ):
        raise ValueError(f"Invalid arm catcher mount: {check}")
    return check


def gripper_joint_state(model, data, gripper_spec):
    positions = []
    velocities = []
    for joint_name in gripper_spec["joint_names"]:
        joint = model.joint(joint_name)
        positions.append(float(data.qpos[int(joint.qposadr[0])]))
        velocities.append(float(data.qvel[int(joint.dofadr[0])]))
    return positions, velocities


def driver_joint_state(model, data):
    positions = []
    velocities = []
    for joint_name in ROBOTIQ_DRIVER_JOINT_NAMES:
        joint = model.joint(joint_name)
        positions.append(float(data.qpos[int(joint.qposadr[0])]))
        velocities.append(float(data.qvel[int(joint.dofadr[0])]))
    return positions, velocities


def gripper_opening_width(model, data):
    left = data.site_xpos[site_id(model, ARM_GRIPPER_LEFT_PAD_SITE)]
    right = data.site_xpos[site_id(model, ARM_GRIPPER_RIGHT_PAD_SITE)]
    return float(np.linalg.norm(left - right))


def arm_gripper_site_positions(model, data):
    names = [
        ARM_GRIPPER_LEFT_PAD_SITE,
        ARM_GRIPPER_RIGHT_PAD_SITE,
        ARM_GRIPPER_LEFT_COLLISION_PAD_SITE,
        ARM_GRIPPER_RIGHT_COLLISION_PAD_SITE,
        ARM_GRIPPER_GRASP_SITE,
    ]
    return {
        name: [float(value) for value in data.site_xpos[site_id(model, name)]] for name in names
    }


def vector_norm(values):
    return math.sqrt(sum(float(value) * float(value) for value in values))


def smoothstep01(value):
    value = clip_value(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def predict_ball_at_y(position, velocity, *, gravity, radius, catch_y, horizon=2.25, dt=0.015):
    pos = [float(value) for value in position]
    vel = [float(value) for value in velocity]
    previous_pos = list(pos)
    elapsed = 0.0
    while elapsed < horizon:
        previous_pos[:] = pos
        vel[2] -= gravity * dt
        pos[0] += vel[0] * dt
        pos[1] += vel[1] * dt
        pos[2] += vel[2] * dt
        if pos[2] < radius:
            pos[2] = radius
            if vel[2] < 0.0:
                vel[2] = -vel[2]
        elapsed += dt
        if previous_pos[1] >= catch_y >= pos[1] or previous_pos[1] <= catch_y <= pos[1]:
            denom = pos[1] - previous_pos[1]
            alpha = 0.0 if abs(denom) < 1e-8 else (catch_y - previous_pos[1]) / denom
            alpha = clip_value(alpha, 0.0, 1.0)
            return [
                previous_pos[index] + alpha * (pos[index] - previous_pos[index])
                for index in range(3)
            ], elapsed
    return pos, elapsed


def catcher_policy_config(
    policy_type,
    rng,
    *,
    planned_outcome=None,
    miss_bias=None,
    control_axes=None,
    policy_noise_scale=1.0,
    policy_bias_scale=1.0,
):
    control_axes = list(control_axes or ["x", "z"])
    uses_depth = "y" in control_axes
    policy_noise_scale = float(policy_noise_scale)
    policy_bias_scale = float(policy_bias_scale)

    def bias_values(x_value, z_value, y_value=0.0):
        if uses_depth:
            return [float(x_value), float(y_value), float(z_value)]
        return [float(x_value), float(z_value)]

    def delta_values(x_value, z_value, y_value=0.0):
        if uses_depth:
            return [float(x_value), float(y_value), float(z_value)]
        return [float(x_value), float(z_value)]

    def scaled_bias(values):
        return [float(value) * policy_bias_scale for value in values]

    if planned_outcome == "capture":
        return {
            "delay_frames": 0,
            "noise_std": 0.012 * policy_noise_scale,
            "bias": scaled_bias(
                bias_values(
                    rng.uniform(-0.025, 0.025),
                    rng.uniform(-0.020, 0.020),
                    rng.uniform(-0.045, 0.045),
                )
            ),
            "max_delta": delta_values(0.42, 0.34, 0.16),
        }
    if planned_outcome == "miss":
        bias = miss_bias if miss_bias is not None else scaled_bias(bias_values(1.65, 0.0, 0.0))
        if uses_depth and len(bias) < 3:
            bias = [float(bias[0]), 0.25, float(bias[1])]
        return {
            "delay_frames": rng.randint(3, 7),
            "noise_std": 0.035 * policy_noise_scale,
            "bias": [float(value) for value in bias],
            "max_delta": delta_values(0.24, 0.20, 0.12),
        }
    if policy_type == "expert":
        return {
            "delay_frames": 0,
            "noise_std": 0.035 * policy_noise_scale,
            "bias": scaled_bias(
                bias_values(
                    rng.uniform(-0.08, 0.08),
                    rng.uniform(-0.04, 0.04),
                    rng.uniform(-0.08, 0.08),
                )
            ),
            "max_delta": delta_values(0.32, 0.25, 0.14),
        }
    if policy_type == "noisy_delayed":
        return {
            "delay_frames": rng.randint(4, 10),
            "noise_std": 0.12 * policy_noise_scale,
            "bias": scaled_bias(
                bias_values(
                    rng.uniform(-0.45, 0.45),
                    rng.uniform(-0.22, 0.22),
                    rng.uniform(-0.18, 0.18),
                )
            ),
            "max_delta": delta_values(0.22, 0.18, 0.10),
        }
    return {
        "delay_frames": 0,
        "noise_std": 0.0,
        "bias": bias_values(0.0, 0.0, 0.0),
        "max_delta": delta_values(0.18, 0.14, 0.08),
    }


def compute_catcher_action(
    *,
    catcher_spec,
    policy_type,
    policy_config,
    rng,
    target_position,
    observation_history,
    gravity,
    radius,
):
    workspace_min = catcher_spec["workspace_min"]
    workspace_max = catcher_spec["workspace_max"]
    control_axes = list(catcher_spec.get("control_axes", ["x", "z"]))
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    max_delta = policy_config["max_delta"]
    if policy_type == "random":
        return [
            rng.uniform(-max_delta[index], max_delta[index]) for index in range(len(control_axes))
        ]

    delay = min(policy_config["delay_frames"], len(observation_history) - 1)
    observed_position, observed_velocity = observation_history[-1 - delay]
    catch_y = (
        float(target_position[1]) if "y" in control_axes else float(catcher_spec["fixed_y"])
    ) + catcher_front_y_offset(catcher_spec)
    predicted_position, _ = predict_ball_at_y(
        observed_position,
        observed_velocity,
        gravity=gravity,
        radius=radius,
        catch_y=catch_y,
    )
    bias = policy_config["bias"]
    bias_x = float(bias[0])
    if "y" in control_axes:
        bias_y = float(bias[1])
        bias_z = float(bias[2])
    else:
        bias_y = 0.0
        bias_z = float(bias[1])
    noise_std = float(policy_config["noise_std"])
    desired = [
        predicted_position[0] + bias_x,
        catcher_spec["fixed_y"] + bias_y,
        predicted_position[2] + bias_z,
    ]
    desired = [
        desired[index]
        + (
            rng.gauss(0.0, 0.5 * noise_std)
            if index == 1 and "y" in control_axes
            else rng.gauss(0.0, noise_std)
            if index in (0, 2)
            else 0.0
        )
        for index in range(3)
    ]
    desired = clip_vector(desired, workspace_min, workspace_max)
    return [
        clip_value(
            desired[axis_to_index[axis]] - target_position[axis_to_index[axis]],
            -max_delta[index],
            max_delta[index],
        )
        for index, axis in enumerate(control_axes)
    ]


def arm_catcher_planner_axis_limits(catcher_spec, key, control_axes, default_xyz):
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    values = catcher_spec.get(key, default_xyz)
    return [abs(float(values[axis_to_index[axis]])) for axis in control_axes]


def arm_catcher_planner_speed_limits(catcher_spec, control_axes):
    per_frame_max = catcher_spec.get("per_frame_max_delta_xyz", [0.20, 0.14, 0.20])
    default_speed = [0.9 * abs(float(value)) for value in per_frame_max]
    return arm_catcher_planner_axis_limits(
        catcher_spec,
        "planner_max_speed_xyz",
        control_axes,
        default_speed,
    )


def arm_catcher_planner_accel_limits(catcher_spec, control_axes):
    per_frame_max = catcher_spec.get("per_frame_max_delta_xyz", [0.20, 0.14, 0.20])
    default_accel = [0.22 * abs(float(value)) for value in per_frame_max]
    return arm_catcher_planner_axis_limits(
        catcher_spec,
        "planner_max_accel_xyz",
        control_axes,
        default_accel,
    )


def limit_physical_step_change(previous_delta, desired_delta, max_change):
    previous = np.asarray(previous_delta, dtype=np.float64)
    desired = np.asarray(desired_delta, dtype=np.float64)
    limits = np.asarray(max_change, dtype=np.float64)
    return [float(value) for value in previous + np.clip(desired - previous, -limits, limits)]


def predict_ball_at_y_analytic(position, velocity, *, gravity, radius, catch_y, horizon=4.5):
    """Exact ballistic crossing of the catch plane for an undamped elastic ball.

    The y motion is uniform and the z motion is a piecewise parabola with
    perfectly elastic ground bounces at z = radius, so the crossing time and
    point have closed forms — no integration error, unlike predict_ball_at_y.
    Returns (crossing_position, time_to_crossing).
    """
    x0, y0, z0 = (float(value) for value in position)
    vx, vy, vz = (float(value) for value in velocity)
    if abs(vy) < 1.0e-9:
        return [x0, y0, z0], float(horizon)
    t_cross = (catch_y - y0) / vy
    if t_cross <= 0.0:
        return [x0, catch_y, max(z0, radius)], 0.0
    t_cross = min(t_cross, float(horizon))

    t = 0.0
    z = z0
    v = vz
    if z <= radius:
        z = radius
        v = abs(v)
    for _ in range(200):
        remaining = t_cross - t
        if remaining <= 0.0:
            break
        if gravity <= 1.0e-12:
            landing = z + v * remaining
            if v >= 0.0 or landing >= radius:
                z = landing
                break
            time_to_ground = (radius - z) / v
            z = radius
            v = -v
            t += time_to_ground
            continue
        time_to_ground = (v + math.sqrt(max(v * v + 2.0 * gravity * (z - radius), 0.0))) / gravity
        if time_to_ground <= 1.0e-9 or time_to_ground >= remaining:
            z = z + v * remaining - 0.5 * gravity * remaining * remaining
            break
        z = radius
        v = gravity * time_to_ground - v
        t += time_to_ground
    return [x0 + vx * t_cross, catch_y, max(z, radius)], t_cross


def compute_arm_catcher_interception_step(
    *,
    catcher_spec,
    policy_type,
    policy_config,
    rng,
    target_position,
    observation_history,
    gravity,
    radius,
    fps,
    wiggle_offset=None,
):
    """Per-frame EE step that reaches the predicted catch point in time.

    Unlike the greedy proportional policy, this uses the ballistic
    time-of-arrival at the catch plane to pace the end effector so it arrives
    a few frames before the ball and settles there, instead of chasing the
    ball with a lagging clipped step. An optional smooth wiggle offset is
    blended into the desired point and faded to zero as the ball approaches,
    so exploration never costs the interception.
    """
    control_axes = list(catcher_spec.get("control_axes", ["x", "z"]))
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    max_speed = arm_catcher_planner_speed_limits(catcher_spec, control_axes)
    if policy_type == "random":
        return [rng.uniform(-limit, limit) for limit in max_speed]

    dt = 1.0 / float(fps)
    delay = min(int(policy_config["delay_frames"]), len(observation_history) - 1)
    observed_position, observed_velocity = observation_history[-1 - delay]
    catch_y = (
        float(target_position[1]) if "y" in control_axes else float(catcher_spec["fixed_y"])
    ) + catcher_front_y_offset(catcher_spec)
    predicted_position, time_to_plane = predict_ball_at_y_analytic(
        observed_position,
        observed_velocity,
        gravity=gravity,
        radius=radius,
        catch_y=catch_y,
        horizon=4.5,
    )
    bias = policy_config["bias"]
    bias_x = float(bias[0])
    if "y" in control_axes:
        bias_y = float(bias[1])
        bias_z = float(bias[2])
    else:
        bias_y = 0.0
        bias_z = float(bias[1])
    noise_std = float(policy_config["noise_std"])
    desired = [
        predicted_position[0] + bias_x + rng.gauss(0.0, noise_std),
        float(catcher_spec["fixed_y"])
        + bias_y
        + (rng.gauss(0.0, 0.5 * noise_std) if "y" in control_axes else 0.0),
        predicted_position[2] + bias_z + rng.gauss(0.0, noise_std),
    ]
    time_from_now = max(time_to_plane - delay * dt, 0.0)
    desired = clip_vector(desired, catcher_spec["workspace_min"], catcher_spec["workspace_max"])
    planner_min_z = catcher_spec.get("planner_min_z")
    if planner_min_z is not None:
        desired[2] = max(desired[2], float(planner_min_z))
    margin_frames = float(catcher_spec.get("planner_arrival_margin_frames", 3.0))
    # Floor well above 1: once the arrival deadline has passed, a floor of 1
    # would pass the per-frame gauss noise in `desired` straight into the
    # step at full amplitude; spreading it over 8 frames keeps the hold
    # steady while prediction drift is still tracked.
    remaining_frames = max(time_from_now / dt - margin_frames, 8.0)
    step = [
        (desired[axis_to_index[axis]] - float(target_position[axis_to_index[axis]]))
        / remaining_frames
        for axis in control_axes
    ]
    if wiggle_offset is not None:
        # Step-level injection: a position-level wiggle would be divided by
        # remaining_frames and vanish from the recorded actions. The fade
        # zeroes it well before interception; the caller's accel limiter and
        # the speed clip below keep the perturbed step smooth and bounded.
        fade_start = float(catcher_spec.get("planner_wiggle_fade_seconds", 0.55))
        # Fade relative to the planner's arrival deadline, not the catch:
        # the wiggle should be gone by the time the arm settles at the
        # interception point, so the hold phase is visually still.
        arrival_seconds = margin_frames * dt
        wiggle_gain = smoothstep01((time_from_now - arrival_seconds - fade_start) / 0.40)
        for index, axis in enumerate(control_axes):
            step[index] += wiggle_gain * float(wiggle_offset[axis_to_index[axis]])
    return [
        clip_value(value, -max_speed[index], max_speed[index]) for index, value in enumerate(step)
    ]


def contact_flags(model, data):
    flags = {
        "ball_catcher_contact": False,
        "ball_floor_contact": False,
        "ball_wall_contact": False,
    }
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom_names = []
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            geom_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        if "moving_object" not in geom_names:
            continue
        other_name = geom_names[1] if geom_names[0] == "moving_object" else geom_names[0]
        if other_name.startswith("catcher_") or other_name.startswith("arm_catcher_"):
            flags["ball_catcher_contact"] = True
        elif other_name == "approach_ball_ground":
            flags["ball_floor_contact"] = True
        elif other_name.startswith("approach_ball_") and other_name.endswith("_wall"):
            flags["ball_wall_contact"] = True
    return flags


def paddle_contact_flags(model, data):
    flags = {
        "ball_paddle_contact": False,
        "ball_floor_contact": False,
        "ball_wall_contact": False,
    }
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom_names = []
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            geom_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        if "moving_object" not in geom_names:
            continue
        other_name = geom_names[1] if geom_names[0] == "moving_object" else geom_names[0]
        if other_name.startswith("paddle_"):
            flags["ball_paddle_contact"] = True
        elif other_name == "approach_ball_ground":
            flags["ball_floor_contact"] = True
        elif other_name.startswith("approach_ball_") and other_name.endswith("_wall"):
            flags["ball_wall_contact"] = True
    return flags


def arm_gripper_contact_flags(model, data):
    flags = {
        "ball_gripper_left_pad_contact": False,
        "ball_gripper_right_pad_contact": False,
        "ball_gripper_palm_contact": False,
        "ball_floor_contact": False,
        "ball_wall_contact": False,
    }
    left_geoms = {"rq_left_pad1", "rq_left_pad2", "arm_gripper_left_collision_pad_geom"}
    right_geoms = {"rq_right_pad1", "rq_right_pad2", "arm_gripper_right_collision_pad_geom"}
    palm_geoms = set(ARM_GRIPPER_PALM_GEOMS)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom_names = []
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            geom_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        if "moving_object" not in geom_names:
            continue
        other_name = geom_names[1] if geom_names[0] == "moving_object" else geom_names[0]
        if other_name in left_geoms:
            flags["ball_gripper_left_pad_contact"] = True
        elif other_name in right_geoms:
            flags["ball_gripper_right_pad_contact"] = True
        elif other_name in palm_geoms:
            flags["ball_gripper_palm_contact"] = True
        elif other_name == "approach_ball_ground":
            flags["ball_floor_contact"] = True
        elif other_name.startswith("approach_ball_") and other_name.endswith("_wall"):
            flags["ball_wall_contact"] = True
    return flags


def arm_gripper_policy_config(policy_type, rng, *, planned_outcome=None, miss_bias=None):
    if planned_outcome == "capture":
        return {
            "delay_frames": 0,
            "noise_std": 0.0025,
            "bias": [
                rng.uniform(-0.004, 0.004),
                rng.uniform(-0.006, 0.006),
                rng.uniform(-0.006, 0.006),
            ],
            "max_delta": [0.085, 0.045, 0.065],
        }
    if planned_outcome == "miss":
        bias = miss_bias if miss_bias is not None else [0.16, 0.04, 0.06]
        return {
            "delay_frames": rng.randint(4, 8),
            "noise_std": 0.030,
            "bias": [float(value) for value in bias],
            "max_delta": [0.050, 0.030, 0.042],
        }
    if policy_type == "noisy_delayed":
        return {
            "delay_frames": rng.randint(3, 7),
            "noise_std": 0.024,
            "bias": [
                rng.uniform(-0.075, 0.075),
                rng.uniform(-0.030, 0.030),
                rng.uniform(-0.045, 0.045),
            ],
            "max_delta": [0.055, 0.032, 0.045],
        }
    if policy_type == "random":
        return {
            "delay_frames": 0,
            "noise_std": 0.0,
            "bias": [0.0, 0.0, 0.0],
            "max_delta": [0.035, 0.025, 0.032],
        }
    return {
        "delay_frames": 1,
        "noise_std": 0.012,
        "bias": [0.0, 0.0, 0.0],
        "max_delta": [0.070, 0.040, 0.055],
    }


def compute_arm_gripper_action(
    *,
    catcher_spec,
    policy_type,
    policy_config,
    rng,
    target_position,
    observation_history,
    gravity,
    radius,
    grasp_rotation=None,
):
    action_scale = catcher_spec["action_scale_xyz"]
    if policy_type == "random":
        normalized = [rng.uniform(-1.0, 1.0) for _ in range(3)]
        physical = [
            clip_value(
                normalized[index] * action_scale[index],
                -policy_config["max_delta"][index],
                policy_config["max_delta"][index],
            )
            for index in range(3)
        ]
        return normalized, physical

    delay = min(policy_config["delay_frames"], len(observation_history) - 1)
    observed_position, observed_velocity = observation_history[-1 - delay]
    predicted_position, _ = predict_ball_at_y(
        observed_position,
        observed_velocity,
        gravity=gravity,
        radius=radius,
        catch_y=target_position[1],
        horizon=5.5,
        dt=0.010,
    )
    bias = policy_config["bias"]
    desired = [
        predicted_position[0] + bias[0],
        target_position[1] + bias[1],
        predicted_position[2] + bias[2],
    ]
    if grasp_rotation is not None:
        desired_ball_local = np.asarray(
            catcher_spec.get("desired_ball_local_position", [0.0, 0.0, -0.020]),
            dtype=np.float64,
        )
        desired = [
            float(value)
            for value in (
                np.asarray(desired, dtype=np.float64)
                - np.asarray(grasp_rotation, dtype=np.float64) @ desired_ball_local
            )
        ]
    noise_std = float(policy_config["noise_std"])
    desired = [
        desired[index] + rng.gauss(0.0, noise_std if index != 1 else 0.45 * noise_std)
        for index in range(3)
    ]
    desired = clip_vector(desired, catcher_spec["workspace_min"], catcher_spec["workspace_max"])
    physical = [
        clip_value(
            desired[index] - target_position[index],
            -policy_config["max_delta"][index],
            policy_config["max_delta"][index],
        )
        for index in range(3)
    ]
    normalized = [
        clip_value(physical[index] / action_scale[index], -1.0, 1.0) for index in range(3)
    ]
    return normalized, [normalized[index] * action_scale[index] for index in range(3)]


def arm_gripper_grip_command(current_grip_command, mouth_metrics, *, captured, missed):
    if captured:
        return 1.0
    if float(current_grip_command) >= 0.5:
        return 1.0
    if missed:
        return 0.0
    return 1.0 if mouth_metrics["preclose_ready"] else 0.0


def arm_gripper_mouth_metrics(model, data, ball_position, ball_velocity, radius):
    ball_local = site_local_position(model, data, ARM_GRIPPER_GRASP_SITE, ball_position)
    grasp_site = site_id(model, ARM_GRIPPER_GRASP_SITE)
    grasp_rotation = data.site_xmat[grasp_site].reshape(3, 3)
    ball_velocity_local = grasp_rotation.T @ np.asarray(ball_velocity, dtype=np.float64)
    left_pad = site_local_position(
        model,
        data,
        ARM_GRIPPER_GRASP_SITE,
        data.site_xpos[site_id(model, ARM_GRIPPER_LEFT_PAD_SITE)],
    )
    right_pad = site_local_position(
        model,
        data,
        ARM_GRIPPER_GRASP_SITE,
        data.site_xpos[site_id(model, ARM_GRIPPER_RIGHT_PAD_SITE)],
    )
    left_collision_pad = site_local_position(
        model,
        data,
        ARM_GRIPPER_GRASP_SITE,
        data.site_xpos[site_id(model, ARM_GRIPPER_LEFT_COLLISION_PAD_SITE)],
    )
    right_collision_pad = site_local_position(
        model,
        data,
        ARM_GRIPPER_GRASP_SITE,
        data.site_xpos[site_id(model, ARM_GRIPPER_RIGHT_COLLISION_PAD_SITE)],
    )
    palm = site_local_position(
        model,
        data,
        ARM_GRIPPER_GRASP_SITE,
        data.site_xpos[site_id(model, ARM_GRIPPER_PALM_SITE)],
    )
    pad_mid = 0.25 * (left_pad + right_pad + left_collision_pad + right_collision_pad)
    opening = gripper_opening_width(model, data)
    pad_z = max(
        float(left_pad[2]),
        float(right_pad[2]),
        float(left_collision_pad[2]),
        float(right_collision_pad[2]),
    )
    palm_z = float(palm[2])
    lower_z = palm_z - max(0.055, 0.35 * radius)
    upper_z = pad_z + max(0.180, 1.10 * radius)
    lateral_y = abs(float(ball_local[1] - pad_mid[1]))
    lateral_x = abs(float(ball_local[0] - pad_mid[0]))
    inside_x_bound = max(0.145, min(0.320, radius + 0.160))
    preclose_x_bound = inside_x_bound + max(0.090, 0.70 * radius)
    inside_y_bound = 0.5 * opening + max(0.085, 1.10 * radius)
    preclose_y_bound = 0.5 * opening + max(0.135, 1.30 * radius)
    inside_y = lateral_y <= inside_y_bound
    preclose_y = lateral_y <= preclose_y_bound
    inside_x = lateral_x <= inside_x_bound
    preclose_x = lateral_x <= preclose_x_bound
    inside_z = lower_z <= float(ball_local[2]) <= upper_z
    inside_mouth = bool(inside_x and inside_y and inside_z)
    preclose_z = (
        lower_z - max(0.090, 0.55 * radius)
        <= float(ball_local[2])
        <= upper_z + max(0.200, 1.20 * radius)
    )
    approaching_mouth = float(ball_velocity_local[2]) > -0.080
    preclose_ready = bool(preclose_x and preclose_y and preclose_z and approaching_mouth)
    return {
        "ball_local_position": [float(value) for value in ball_local],
        "ball_local_velocity": [float(value) for value in ball_velocity_local],
        "pad_mid_local_position": [float(value) for value in pad_mid],
        "palm_local_position": [float(value) for value in palm],
        "mouth_lower_z": float(lower_z),
        "mouth_upper_z": float(upper_z),
        "mouth_lateral_x_bound": float(inside_x_bound),
        "preclose_lateral_x_bound": float(preclose_x_bound),
        "mouth_lateral_y_bound": float(inside_y_bound),
        "preclose_lateral_y_bound": float(preclose_y_bound),
        "inside_gripper_mouth": inside_mouth,
        "preclose_ready": preclose_ready,
    }


def arm_gripper_grasp_candidate(
    model,
    data,
    ball_position,
    ball_velocity,
    flags,
    radius,
    grip_command,
    mouth_metrics,
):
    opening = gripper_opening_width(model, data)
    pad_contact = flags["ball_gripper_left_pad_contact"] or flags["ball_gripper_right_pad_contact"]
    environment_contact = flags["ball_floor_contact"] or flags["ball_wall_contact"]
    closed = float(grip_command) >= 0.5 and opening <= max(0.320, min(0.620, 3.60 * radius))
    controlled_speed = vector_norm(ball_velocity) <= 5.5
    return bool(
        mouth_metrics["inside_gripper_mouth"]
        and closed
        and controlled_speed
        and pad_contact
        and not environment_contact
    )


def arm_gripper_miss_reason(ball_position, grasp_position, flags, radius, reached_grasp_region):
    rel_y = float(ball_position[1] - grasp_position[1])
    passed_gripper = rel_y < -max(0.20, 3.0 * radius)
    if passed_gripper:
        return "passed_grasp_region"
    if reached_grasp_region and flags["ball_wall_contact"]:
        return "post_interaction_wall_contact"
    return None


def caught_frame(
    ball_position,
    ball_velocity,
    catcher_position,
    catcher_velocity,
    flags,
    catcher_spec,
    radius=0.0,
):
    rel_pos = [ball_position[index] - catcher_position[index] for index in range(3)]
    back_y, front_y = catcher_depth_bounds(catcher_spec)
    if "capture_front_margin" in catcher_spec:
        front_margin = front_y + max(0.0, float(catcher_spec["capture_front_margin"]))
    else:
        front_margin = front_y + float(radius) + 0.12
    if "capture_back_margin" in catcher_spec:
        back_margin = back_y - max(0.0, float(catcher_spec["capture_back_margin"]))
    else:
        back_margin = back_y - max(0.06, 0.15 * float(radius))
    inside_depth = back_margin <= rel_pos[1] <= front_margin
    return bool(inside_depth and catcher_inside_capture_center(rel_pos, catcher_spec))


def miss_reason(ball_position, catcher_position, flags, catcher_spec, radius, reached_catch_plane):
    rel_pos = [ball_position[index] - catcher_position[index] for index in range(3)]
    half_width = catcher_spec["net_half_width"]
    half_height = catcher_spec["net_half_height"]
    back_y, front_y = catcher_depth_bounds(catcher_spec)
    entered_net_depth = rel_pos[1] <= front_y
    entered_strict_center_check_depth = rel_pos[1] <= front_y - min(0.05, 0.22 * (front_y - back_y))
    passed_back = rel_pos[1] <= back_y - max(0.08, 0.25 * radius)
    outside_opening = (
        abs(rel_pos[0]) > half_width + 0.20 * radius
        or abs(rel_pos[2]) > half_height + 0.20 * radius
    )
    outside_capture_center = not catcher_inside_capture_center(rel_pos, catcher_spec)

    if passed_back:
        return "passed_catch_plane"
    if entered_strict_center_check_depth and outside_capture_center:
        return "outside_center_capture_window"
    if entered_net_depth and outside_opening:
        return "outside_net_opening"
    if reached_catch_plane and flags["ball_floor_contact"]:
        return "post_interaction_floor_contact"
    if reached_catch_plane and flags["ball_wall_contact"]:
        return "post_interaction_wall_contact"
    return None


def add_vectors(a, b):
    return [float(a[index]) + float(b[index]) for index in range(3)]


def clamp_position_above_ground(position, radius, *, ground_z=0.0):
    clamped = [float(value) for value in position]
    clamped[2] = max(clamped[2], float(ground_z) + float(radius))
    return clamped


def apply_control_delta(target_position, physical_delta, control_axes, catcher_spec):
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    updated = [float(value) for value in target_position]
    per_frame_max_delta = catcher_spec.get("per_frame_max_delta_xyz")
    for axis, delta in zip(control_axes, physical_delta, strict=False):
        axis_index = axis_to_index[axis]
        physical_step = float(delta)
        if per_frame_max_delta is not None:
            limit = abs(float(per_frame_max_delta[axis_index]))
            physical_step = clip_value(physical_step, -limit, limit)
        updated[axis_index] += physical_step
    if "y" not in control_axes:
        updated[1] = float(catcher_spec["fixed_y"])
    return clip_vector(updated, catcher_spec["workspace_min"], catcher_spec["workspace_max"])


def normalized_delta_from_physical_delta(physical_delta, control_axes, catcher_spec):
    action_scale = catcher_spec.get("action_scale_xyz", catcher_spec.get("action_scale_xz"))
    return [
        clip_value(float(delta) / float(action_scale[index]), -1.0, 1.0)
        for index, delta in enumerate(physical_delta)
    ]


def action_smoothness_summary(actions):
    if not actions:
        return {
            "max_normalized_action_step": 0.0,
            "rms_normalized_action_step": 0.0,
            "max_normalized_action_second_difference": 0.0,
            "rms_normalized_action_second_difference": 0.0,
            "max_normalized_command_abs": 0.0,
        }
    action_values = np.asarray(actions, dtype=np.float64)
    command_values = (
        action_values[:, 1:]
        if action_values.ndim == 2 and action_values.shape[1] > 1
        else np.zeros((0, 0))
    )
    if command_values.size == 0:
        return {
            "max_normalized_action_step": 0.0,
            "rms_normalized_action_step": 0.0,
            "max_normalized_action_second_difference": 0.0,
            "rms_normalized_action_second_difference": 0.0,
            "max_normalized_command_abs": 0.0,
        }
    frame_deltas = np.diff(command_values, axis=0)
    frame_second_deltas = np.diff(command_values, n=2, axis=0)
    return {
        "max_normalized_action_step": float(np.max(np.abs(frame_deltas)))
        if frame_deltas.size
        else 0.0,
        "rms_normalized_action_step": float(np.sqrt(np.mean(frame_deltas * frame_deltas)))
        if frame_deltas.size
        else 0.0,
        "max_normalized_action_second_difference": (
            float(np.max(np.abs(frame_second_deltas))) if frame_second_deltas.size else 0.0
        ),
        "rms_normalized_action_second_difference": (
            float(np.sqrt(np.mean(frame_second_deltas * frame_second_deltas)))
            if frame_second_deltas.size
            else 0.0
        ),
        "max_normalized_command_abs": float(np.max(np.abs(command_values))),
    }


def has_consecutive_true(values, run_length):
    streak = 0
    for value in values:
        streak = streak + 1 if value else 0
        if streak >= run_length:
            return True
    return False


def make_catcher_render_payloads(catcher_spec, catcher_frames, fps):
    payloads = []
    for part in catcher_spec["parts"]:
        material = part.get("material", catcher_spec["material"])
        payload = {
            "name": part["name"],
            "kind": "box",
            "material": material,
            "half_extents": part["half_extents"],
            "frames": [],
        }
        offset = part["offset"]
        for frame_idx, catcher_frame in enumerate(catcher_frames):
            position = [catcher_frame["position"][index] + offset[index] for index in range(3)]
            payload["frames"].append(
                frame_payload(
                    frame_idx,
                    fps,
                    position,
                    (0.0, 0.0, 0.0, 1.0),
                    catcher_frame["linear_velocity"],
                    (0.0, 0.0, 0.0),
                )
            )
        payloads.append(payload)
    return payloads


def paddle_policy_config(rng):
    return {
        "policy_type": "expert",
        "noise_std_xy": 0.003,
        "noise_std_z": 0.0,
        "lead_time_seconds": 0.045,
        "reset_z": 0.42,
        "low_gravity_reset_z": 0.42,
        "strike_z": 0.66,
        "low_gravity_strike_z": 0.51,
        "mid_gravity_strike_boost": 0.04,
        "mid_gravity_strike_center": 1.0,
        "mid_gravity_strike_sigma": 0.30,
        "mid_low_gravity_strike_boost": 0.025,
        "mid_low_gravity_strike_center": 1.8,
        "mid_low_gravity_strike_sigma": 0.45,
        "min_strike_z": 0.42,
        "max_strike_z": 0.70,
        "target_apex_z": 1.10,
        "gravity_pulse_min": 0.5,
        "gravity_pulse_full": 3.0,
        "strike_band": 0.025,
        "high_ball_z": 1.20,
        "high_ball_vz_threshold": -0.05,
        "effective_restitution": 0.92,
        "height_gain": 0.06,
        "apex_deadband": 0.02,
        "hold_frames_after_contact": 2,
        "low_gravity_hold_frames_after_contact": 2,
        "low_gravity_hold_threshold": 0.75,
        "failed_bounce_hold_frames": 3,
        "min_bounce_velocity_after_contact": 0.05,
        "strike_min_descend_speed": 0.18,
        "contact_release_clearance_z": 0.035,
        "contact_release_drop_z": 0.08,
        "post_contact_reset_frames": 8,
        "xy_track_lead_seconds": 0.10,
        "xy_track_max_lead_seconds": 0.18,
        "xy_track_center_bias": 0.06,
        "xy_track_max_center_offset": 0.035,
        "tilt_deadband_velocity": 0.025,
        "tilt_centering_time_seconds": 0.85,
        "tilt_max_centering_speed": 0.70,
        "tilt_velocity_gain": 1.0,
        "tilt_impulse_gain": 1.10,
        "tilt_min_incoming_speed": 0.75,
        "tilt_phi_noise_std": 0.0,
        "tilt_theta_noise_std": 0.0,
        "noise_seed_offset": rng.randint(0, 2**31 - 1),
    }


def update_adaptive_strike_trim(policy_config, *, gravity, achieved_apex):
    """Per-bounce integral trim that drives the flight apex to the target.

    The strike stroke needed for a given apex depends on gravity, servo lag,
    and contact timing; a fixed stroke therefore equilibrates at a
    gravity-dependent apex. After each counted paddle contact the measured
    apex of the completed flight updates a persistent strike offset. The gain
    scales with sqrt(g) because apex sensitivity to strike speed scales like
    v_out/g ~ 1/sqrt(g).
    """
    gain = float(policy_config.get("adaptive_apex_trim_gain", 0.0))
    if gain <= 0.0 or achieved_apex is None or float(gravity) <= 1.0e-9:
        return
    if float(gravity) <= float(policy_config.get("velocity_strike_max_gravity", 0.0)):
        # Velocity-matched ramp strikes handle these gravities feedforward;
        # the stroke trim does not apply.
        return
    error = float(policy_config["target_apex_z"]) - float(achieved_apex)
    limit = abs(float(policy_config.get("adaptive_strike_trim_limit", 0.05)))
    trim = float(policy_config.get("adaptive_strike_trim", 0.0))
    trim += gain * math.sqrt(float(gravity) / 9.8) * error
    policy_config["adaptive_strike_trim"] = clip_value(trim, -limit, limit)


def estimate_ball_contact_time(position, velocity, *, gravity, contact_z):
    height = float(position[2]) - float(contact_z)
    if height <= 0.0:
        return 0.0
    vz = float(velocity[2])
    if float(gravity) <= 1.0e-9:
        if vz >= -1.0e-9:
            return None
        return height / (-vz)
    discriminant = vz * vz + 2.0 * float(gravity) * height
    if discriminant < 0.0:
        return None
    time_value = (vz + math.sqrt(discriminant)) / float(gravity)
    if time_value < 0.0:
        return None
    return time_value


def compute_paddle_action(
    *,
    paddle_spec,
    policy_config,
    rng,
    target_position,
    ball_position,
    ball_velocity,
    gravity,
    radius,
    force_reset=False,
    force_strike=False,
    force_miss=False,
    force_launch=False,
    force_hold=False,
    failure_position_bias=(0.0, 0.0),
    failure_tilt_bias=(0.0, 0.0),
    failure_hold_position=None,
):
    workspace_min = paddle_spec["workspace_min"]
    workspace_max = paddle_spec["workspace_max"]
    action_scale = paddle_spec["action_scale_xyz"]
    base_half_z = float(paddle_spec["base_half_extents"][2])
    center_x, center_y = paddle_spec.get("tilt_center_xy", [0.0, -3.0])
    tilt_phi = 0.0
    tilt_theta = 0.0
    gravity_pulse_fraction = clip_value(
        (float(gravity) - float(policy_config["gravity_pulse_min"]))
        / (float(policy_config["gravity_pulse_full"]) - float(policy_config["gravity_pulse_min"])),
        0.0,
        1.0,
    )
    reset_z = float(policy_config["low_gravity_reset_z"]) + gravity_pulse_fraction * (
        float(policy_config["reset_z"]) - float(policy_config["low_gravity_reset_z"])
    )
    desired = [
        float(ball_position[0]) + rng.gauss(0.0, policy_config["noise_std_xy"]),
        float(ball_position[1]) + rng.gauss(0.0, policy_config["noise_std_xy"]),
        reset_z,
    ]
    nominal_strike_z = float(policy_config["low_gravity_strike_z"]) + gravity_pulse_fraction * (
        float(policy_config["strike_z"]) - float(policy_config["low_gravity_strike_z"])
    )
    mid_gravity_strike_sigma = max(1.0e-6, float(policy_config["mid_gravity_strike_sigma"]))
    mid_gravity_offset = (
        float(gravity) - float(policy_config["mid_gravity_strike_center"])
    ) / mid_gravity_strike_sigma
    nominal_strike_z += float(policy_config["mid_gravity_strike_boost"]) * math.exp(
        -0.5 * mid_gravity_offset * mid_gravity_offset
    )
    mid_low_gravity_strike_sigma = max(
        1.0e-6,
        float(policy_config["mid_low_gravity_strike_sigma"]),
    )
    mid_low_gravity_offset = (
        float(gravity) - float(policy_config["mid_low_gravity_strike_center"])
    ) / mid_low_gravity_strike_sigma
    nominal_strike_z += float(policy_config["mid_low_gravity_strike_boost"]) * math.exp(
        -0.5 * mid_low_gravity_offset * mid_low_gravity_offset
    )
    nominal_strike_z += float(policy_config.get("adaptive_strike_trim", 0.0))
    min_strike_z = max(
        float(policy_config["min_strike_z"]),
        nominal_strike_z - float(policy_config["strike_band"]),
    )
    max_strike_z = min(
        float(policy_config["max_strike_z"]),
        nominal_strike_z + float(policy_config["strike_band"]),
    )
    target_apex_z = float(policy_config["target_apex_z"])
    nominal_contact_z = nominal_strike_z + base_half_z + float(radius)
    incoming_speed = math.sqrt(
        max(
            0.0,
            float(ball_velocity[2]) ** 2
            + 2.0 * float(gravity) * max(float(ball_position[2]) - nominal_contact_z, 0.0),
        )
    )
    if float(gravity) > 1.0e-9:
        passive_apex_z = nominal_contact_z + (
            float(policy_config["effective_restitution"]) * incoming_speed
        ) ** 2 / (2.0 * float(gravity))
        apex_error = target_apex_z - passive_apex_z
        if abs(apex_error) < float(policy_config["apex_deadband"]):
            apex_error = 0.0
    else:
        apex_error = 0.0
    height_gain = float(policy_config["height_gain"])
    height_gain_per_gravity = float(policy_config.get("height_gain_per_gravity", 0.0))
    if height_gain_per_gravity > 0.0:
        # Apex sensitivity to strike speed scales like 1/g, so a uniform
        # feedback loop gain needs the height gain to grow with gravity.
        height_gain = min(height_gain_per_gravity * float(gravity), 0.6)
    strike_z = nominal_strike_z + height_gain * apex_error
    high_ball_vz_threshold = float(policy_config.get("high_ball_vz_threshold", -0.05))
    if (
        float(ball_position[2]) >= float(policy_config["high_ball_z"])
        and float(ball_velocity[2]) > high_ball_vz_threshold
    ):
        strike_z = min_strike_z
    strike_z = clip_value(
        strike_z + rng.gauss(0.0, policy_config["noise_std_z"]),
        min_strike_z,
        max_strike_z,
    )

    current_contact_z = strike_z + base_half_z + float(radius)
    contact_time = estimate_ball_contact_time(
        ball_position,
        ball_velocity,
        gravity=gravity,
        contact_z=current_contact_z,
    )
    min_descend_speed = float(policy_config["strike_min_descend_speed"])
    near_contact_surface = bool(paddle_spec.get("tilt_enabled", False)) and float(
        ball_position[2]
    ) <= current_contact_z + float(policy_config["contact_release_clearance_z"])
    gentle_near_contact = near_contact_surface and float(ball_velocity[2]) > -min_descend_speed
    descending_to_strike = float(ball_velocity[2]) <= -min_descend_speed
    # At low gravity the rebound apex is hypersensitive to the strike stroke
    # (~1/g), so position-based strikes hunt on frame-timing jitter. Instead
    # ride the paddle target on a constant-velocity ramp matched to the
    # rebound speed the target apex needs; a ramp is timing-insensitive.
    use_velocity_strike = (
        0.0 < float(gravity) <= float(policy_config.get("velocity_strike_max_gravity", 0.0))
    )
    # force_launch: a rest-start ball sits on the paddle at zero speed, so the
    # normal descend-triggered strike never fires. Drive the paddle up on the
    # same velocity-matched ramp (at any gravity) until the ball is airborne,
    # then hand off to the normal regulator, which pumps it to the target apex.
    if use_velocity_strike or force_launch:
        in_ramp_window = (
            force_strike
            or force_launch
            or (
                not force_reset
                and descending_to_strike
                and contact_time is not None
                and contact_time <= float(policy_config.get("velocity_strike_lead_seconds", 0.30))
            )
        )
        if in_ramp_window:
            strike_restitution = float(
                policy_config.get(
                    "strike_effective_restitution",
                    policy_config["effective_restitution"],
                )
            )
            control_dt = float(policy_config.get("control_dt", 0.0625))
            rebound_speed_needed = math.sqrt(
                max(0.0, 2.0 * float(gravity) * (target_apex_z - current_contact_z))
            )
            paddle_speed_needed = (rebound_speed_needed - strike_restitution * incoming_speed) / (
                1.0 + strike_restitution
            )
            speed_limit = float(action_scale[2]) / max(control_dt, 1.0e-6)
            paddle_speed_needed = clip_value(paddle_speed_needed, -speed_limit, speed_limit)
            desired[2] = clip_value(
                float(target_position[2]) + paddle_speed_needed * control_dt,
                float(workspace_min[2]),
                float(policy_config["max_strike_z"]),
            )
        elif gentle_near_contact:
            desired[2] = max(
                float(workspace_min[2]),
                reset_z - float(policy_config["contact_release_drop_z"]),
            )
    elif force_strike:
        desired[2] = strike_z
    elif gentle_near_contact:
        desired[2] = max(
            float(workspace_min[2]), reset_z - float(policy_config["contact_release_drop_z"])
        )
    elif (
        not force_reset
        and descending_to_strike
        and (contact_time is None or contact_time <= float(policy_config["lead_time_seconds"]))
    ):
        desired[2] = strike_z

    xy_lead_time = float(policy_config["xy_track_lead_seconds"])
    if contact_time is not None:
        xy_lead_time = min(xy_lead_time, max(0.0, float(contact_time)))
    xy_lead_time = clip_value(
        xy_lead_time,
        0.0,
        float(policy_config["xy_track_max_lead_seconds"]),
    )
    predicted_x = float(ball_position[0]) + xy_lead_time * float(ball_velocity[0])
    predicted_y = float(ball_position[1]) + xy_lead_time * float(ball_velocity[1])
    center_error_x = float(center_x) - predicted_x
    center_error_y = float(center_y) - predicted_y
    center_bias_x = float(policy_config["xy_track_center_bias"]) * center_error_x
    center_bias_y = float(policy_config["xy_track_center_bias"]) * center_error_y
    center_bias_norm = math.sqrt(center_bias_x * center_bias_x + center_bias_y * center_bias_y)
    max_center_offset = float(policy_config["xy_track_max_center_offset"])
    if center_bias_norm > max_center_offset > 0.0:
        center_bias_scale = max_center_offset / center_bias_norm
        center_bias_x *= center_bias_scale
        center_bias_y *= center_bias_scale
    desired[0] = predicted_x + center_bias_x + rng.gauss(0.0, policy_config["noise_std_xy"])
    desired[1] = predicted_y + center_bias_y + rng.gauss(0.0, policy_config["noise_std_xy"])

    if paddle_spec.get("tilt_enabled", False):
        centering_time = max(1.0e-6, float(policy_config["tilt_centering_time_seconds"]))
        target_velocity_x = center_error_x / centering_time
        target_velocity_y = center_error_y / centering_time
        target_speed = math.sqrt(
            target_velocity_x * target_velocity_x + target_velocity_y * target_velocity_y
        )
        max_centering_speed = float(policy_config["tilt_max_centering_speed"])
        if target_speed > max_centering_speed > 0.0:
            target_velocity_scale = max_centering_speed / target_speed
            target_velocity_x *= target_velocity_scale
            target_velocity_y *= target_velocity_scale

        velocity_gain = float(policy_config["tilt_velocity_gain"])
        correction_x = target_velocity_x - velocity_gain * float(ball_velocity[0])
        correction_y = target_velocity_y - velocity_gain * float(ball_velocity[1])
        correction_speed = math.sqrt(correction_x * correction_x + correction_y * correction_y)
        deadband = float(policy_config["tilt_deadband_velocity"])
        if correction_speed > deadband:
            max_tilt = min(
                float(paddle_spec.get("policy_max_tilt_theta", paddle_spec["max_tilt_theta"])),
                float(paddle_spec["max_tilt_theta"]),
            )
            effective_restitution = float(policy_config["effective_restitution"])
            impulse_scale = (
                float(policy_config["tilt_impulse_gain"])
                * (1.0 + effective_restitution)
                * max(incoming_speed, float(policy_config["tilt_min_incoming_speed"]))
            )
            raw_theta = math.atan(
                max(0.0, correction_speed - deadband) / max(impulse_scale, 1.0e-6)
            )
            tilt_phi = wrap_angle_pi(
                math.atan2(correction_y, correction_x)
                + rng.gauss(0.0, float(policy_config["tilt_phi_noise_std"]))
            )
            tilt_theta = clip_value(
                max_tilt * math.tanh(raw_theta / max(max_tilt, 1.0e-6))
                + rng.gauss(0.0, float(policy_config["tilt_theta_noise_std"])),
                0.0,
                max_tilt,
            )
    if force_miss:
        # Planned-failure "loss of control": the paddle keeps roughly tracking
        # the ball but its hit position drifts off (persistent wrong position),
        # so it slides off the ball and the ball drops to the floor. It also
        # stops pumping (holds at the reset height, passive bounces only) and
        # holds a *fixed* xy (its pose when the flail engaged) plus the drifting
        # offset -- it does NOT chase the ball, so once the ball drifts off the
        # blade there are no further hits and its flight stays bounded (chasing
        # a fast-drifting ball repeatedly edge-hits it and launches it to
        # infinity). The mishandled ball is allowed to leave the tight frame.
        base_x, base_y = (
            failure_hold_position
            if failure_hold_position is not None
            else (float(ball_position[0]), float(ball_position[1]))
        )
        desired[0] = float(base_x) + float(failure_position_bias[0])
        desired[1] = float(base_y) + float(failure_position_bias[1])
        desired[2] = reset_z
        # Flat paddle: the normal centering tilt sees the drifting ball off
        # centre and tilts to recover it, flinging the ball sideways out of view.
        # The failure runs level (or with only the wrong-tilt bias, when set).
        bias_theta = float(failure_tilt_bias[1])
        tilt_phi = wrap_angle_pi(float(failure_tilt_bias[0])) if bias_theta > 0.0 else 0.0
        tilt_theta = clip_value(bias_theta, 0.0, float(paddle_spec["max_tilt_theta"]))
    if force_reset:
        tilt_theta = 0.0
    if force_hold:
        # Hold the rested ball visibly still on the paddle: keep the paddle
        # where it is (zero delta) and level before the launch begins.
        desired = list(target_position)
        tilt_phi = 0.0
        tilt_theta = 0.0

    desired = clip_vector(desired, workspace_min, workspace_max)
    physical_delta = [
        clip_value(
            desired[index] - float(target_position[index]),
            -action_scale[index],
            action_scale[index],
        )
        for index in range(3)
    ]
    normalized_delta = [
        clip_value(physical_delta[index] / action_scale[index], -1.0, 1.0) for index in range(3)
    ]
    return (
        normalized_delta,
        [normalized_delta[index] * action_scale[index] for index in range(3)],
        tilt_phi,
        tilt_theta,
    )


def make_paddle_render_payloads(paddle_spec, paddle_frames, fps):
    payloads = []
    for part in paddle_spec["parts"]:
        material = part.get("material", paddle_spec["material"])
        payload = {
            "name": part["name"],
            "kind": "box",
            "material": material,
            "half_extents": part["half_extents"],
            "frames": [],
        }
        offset = part["offset"]
        part_rotation = part.get("rotation_euler")
        part_quaternion = (
            euler_xyz_to_quaternion_xyzw(part_rotation) if part_rotation is not None else None
        )
        for frame_idx, paddle_frame in enumerate(paddle_frames):
            quaternion = paddle_frame.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])
            rotated_offset = rotate_vector_by_quaternion(offset, quaternion)
            position = [
                paddle_frame["position"][index] + rotated_offset[index] for index in range(3)
            ]
            frame_quaternion = (
                normalize_quaternion_xyzw(quaternion_multiply_xyzw(quaternion, part_quaternion))
                if part_quaternion is not None
                else quaternion
            )
            payload["frames"].append(
                frame_payload(
                    frame_idx,
                    fps,
                    position,
                    frame_quaternion,
                    paddle_frame["linear_velocity"],
                    paddle_frame.get("angular_velocity", [0.0, 0.0, 0.0]),
                )
            )
        payloads.append(payload)
    return payloads


def make_arm_link_render_payloads(arm_spec):
    shoulder_height = float(arm_spec["shoulder_height"])
    upper_length = float(arm_spec["upper_link_length"])
    lower_length = float(arm_spec["lower_link_length"])
    link_half_t = float(arm_spec["link_half_thickness"])
    arm_material = arm_spec["material"]
    joint_material = arm_spec["joint_material"]
    specs = [
        ("arm_base_pedestal", (0.18, 0.18, 0.5 * shoulder_height), joint_material),
        ("arm_shoulder_block", (0.16, 0.16, 0.13), joint_material),
        ("arm_upper_link_geom", (link_half_t, 0.5 * upper_length, link_half_t), arm_material),
        ("arm_elbow_block", (0.14, 0.14, 0.12), joint_material),
        ("arm_lower_link_geom", (link_half_t, 0.5 * lower_length, link_half_t), arm_material),
        ("arm_wrist_1_block", (0.12, 0.12, 0.10), joint_material),
        ("arm_wrist_2_block", (0.11, 0.11, 0.09), joint_material),
        ("arm_wrist_3_block", (0.10, 0.10, 0.08), joint_material),
        ("arm_wrist_block", (0.13, 0.13, 0.13), joint_material),
    ]
    return [
        {
            "name": name,
            "kind": "box",
            "material": material,
            "half_extents": list(half_extents),
            "frames": [],
        }
        for name, half_extents, material in specs
    ]


def append_arm_link_render_frames(arm_payloads, data, frame_idx, fps):
    for payload in arm_payloads:
        position, quaternion = geom_pose(data, payload["name"])
        payload["frames"].append(
            frame_payload(
                frame_idx,
                fps,
                position,
                quaternion,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            )
        )


def simulate_paddle_ball_scene(
    scene, output_json, *, fps, frames_per_episode, step_rate, steps_per_frame
):
    model, data = build_paddle_model(scene, step_rate=step_rate)
    dynamic_spec = scene["dynamic_objects"][0]
    paddle_spec = scene["paddle"]
    initialize_dynamic_state(model, data, dynamic_spec)
    initialize_paddle_state(model, data, paddle_spec)

    gravity = abs(float(scene["metadata"]["gravity"][2]))
    radius = float(dynamic_spec["radius"])
    policy_type = paddle_spec["policy_type"]
    policy_rng = random.Random(int(scene["metadata"]["random_seed"]) + 6262)
    policy_config = paddle_policy_config(policy_rng)
    policy_overrides = paddle_spec.get("policy_config_overrides")
    if policy_overrides:
        policy_config.update(policy_overrides)
    target_position = [float(value) for value in paddle_spec["initial_target_position"]]
    target_phi = float(paddle_spec.get("initial_target_phi", 0.0))
    target_theta = float(paddle_spec.get("initial_target_theta", 0.0))
    action_schema = scene["metadata"].get("action_schema", ["g", "dx", "dy", "dz"])
    tilt_enabled = bool(
        paddle_spec.get("tilt_enabled", False)
        and "phi" in action_schema
        and "theta" in action_schema
    )
    dynamic_payload = make_dynamic_payload(dynamic_spec)
    paddle_frames = []
    contacts = []
    actions = []
    zero_delta = [0.0, 0.0, 0.0]
    last_action = (
        [gravity, *zero_delta, target_phi, target_theta] if tilt_enabled else [gravity, *zero_delta]
    )
    paddle_contact_count = 0
    previous_interval_paddle_contact = False
    last_paddle_contact_frame = -1000
    flight_apex_z = None
    failure_miss_frame = paddle_spec.get("failure_miss_frame")
    # The sideways miss slide must start right after a contact (ball
    # ascending); sliding while the ball descends whacks it laterally with
    # the paddle edge and ricochets it around the arena. Engaged at the first
    # counted contact past the scheduled frame, with a late fallback in case
    # contacts stop.
    miss_engaged_frame = None
    hold_strike_until_frame = -1
    reset_until_frame = -1
    floor_failure = False
    wall_contact_count = 0

    def update_environment_failures(flags):
        nonlocal floor_failure
        if flags["ball_floor_contact"]:
            floor_failure = True

    for frame_idx in range(frames_per_episode):
        paddle_position, paddle_velocity = paddle_joint_state(model, data, paddle_spec)
        orientation_state = paddle_orientation_state(
            model,
            data,
            paddle_spec,
            fallback_phi=target_phi,
        )
        ball_position, quaternion, ball_velocity, angular_velocity = body_state(
            data, dynamic_spec["name"]
        )
        if flight_apex_z is not None:
            flight_apex_z = max(flight_apex_z, float(ball_position[2]))
        frame_flags = paddle_contact_flags(model, data)
        update_environment_failures(frame_flags)

        dynamic_payload["frames"].append(
            frame_payload(
                frame_idx, fps, ball_position, quaternion, ball_velocity, angular_velocity
            )
        )
        paddle_frames.append(
            {
                "frame": frame_idx + 1,
                "time_seconds": frame_idx / fps,
                "position": [float(value) for value in paddle_position],
                "linear_velocity": [float(value) for value in paddle_velocity],
                "target_position": [float(value) for value in target_position],
                "orientation_phi": orientation_state["orientation_phi"],
                "orientation_theta": orientation_state["orientation_theta"],
                "target_phi": float(target_phi),
                "target_theta": float(target_theta),
                "quaternion_xyzw": orientation_state["quaternion_xyzw"],
                "angular_velocity": orientation_state["angular_velocity"],
            }
        )
        interval_flags = {name: bool(value) for name, value in frame_flags.items()}

        if frame_idx < frames_per_episode - 1:
            normalized_delta, physical_delta, action_phi, action_theta = compute_paddle_action(
                paddle_spec=paddle_spec,
                policy_config=policy_config,
                rng=policy_rng,
                target_position=target_position,
                ball_position=ball_position,
                ball_velocity=ball_velocity,
                gravity=gravity,
                radius=radius,
                force_reset=frame_idx < reset_until_frame,
                force_strike=frame_idx < hold_strike_until_frame,
            )
            target_position = clip_vector(
                [target_position[index] + physical_delta[index] for index in range(3)],
                paddle_spec["workspace_min"],
                paddle_spec["workspace_max"],
            )
            if tilt_enabled:
                target_phi = wrap_angle_pi(action_phi)
                target_theta = clip_value(action_theta, 0.0, float(paddle_spec["max_tilt_theta"]))
                action = [gravity, *normalized_delta, target_phi, target_theta]
            else:
                target_phi = 0.0
                target_theta = 0.0
                action = [gravity, *normalized_delta]
            actions.append([float(value) for value in action])
            last_action = action
            for _ in range(steps_per_frame):
                set_paddle_ctrl(data, target_position, paddle_spec, target_phi, target_theta)
                mujoco.mj_step(model, data)
                substep_flags = paddle_contact_flags(model, data)
                update_environment_failures(substep_flags)
                for name, value in substep_flags.items():
                    interval_flags[name] = bool(interval_flags[name] or value)
        else:
            actions.append([float(value) for value in last_action])

        interval_paddle_contact = bool(interval_flags["ball_paddle_contact"])
        if (
            interval_paddle_contact
            and not previous_interval_paddle_contact
            and frame_idx - last_paddle_contact_frame >= 6
        ):
            paddle_contact_count += 1
            last_paddle_contact_frame = frame_idx
            if (
                failure_miss_frame is not None
                and miss_engaged_frame is None
                and frame_idx >= int(failure_miss_frame)
            ):
                miss_engaged_frame = frame_idx + 1
            hold_frames_after_contact = int(policy_config["hold_frames_after_contact"])
            if gravity <= float(policy_config["low_gravity_hold_threshold"]):
                hold_frames_after_contact = int(
                    policy_config["low_gravity_hold_frames_after_contact"]
                )
            post_contact_velocity = body_state(data, dynamic_spec["name"])[2]
            bounce_confirmed = float(post_contact_velocity[2]) > float(
                policy_config["min_bounce_velocity_after_contact"]
            )
            update_adaptive_strike_trim(
                policy_config,
                gravity=gravity,
                achieved_apex=flight_apex_z,
            )
            flight_apex_z = float(ball_position[2])
            if not bounce_confirmed:
                hold_frames_after_contact = int(policy_config["failed_bounce_hold_frames"])
            hold_strike_until_frame = max(
                hold_strike_until_frame,
                frame_idx + hold_frames_after_contact,
            )
            if bounce_confirmed:
                reset_until_frame = max(
                    reset_until_frame,
                    frame_idx + int(policy_config["post_contact_reset_frames"]),
                )
        previous_interval_paddle_contact = interval_paddle_contact
        if (
            failure_miss_frame is not None
            and miss_engaged_frame is None
            and frame_idx >= int(failure_miss_frame)
            and float(ball_velocity[2]) > 0.0
        ):
            miss_engaged_frame = frame_idx + 1
        if (
            failure_miss_frame is not None
            and miss_engaged_frame is None
            and frame_idx >= int(failure_miss_frame) + 32
        ):
            miss_engaged_frame = frame_idx + 1
        if interval_flags["ball_wall_contact"]:
            wall_contact_count += 1
        contacts.append(interval_flags)

    success = paddle_contact_count >= 2 and not floor_failure
    if success:
        outcome = "success"
        terminal_reason = "two_or_more_paddle_contacts_without_floor_contact"
    elif floor_failure:
        outcome = "floor_failure"
        terminal_reason = "ball_floor_contact"
    else:
        outcome = "insufficient_paddle_contacts"
        terminal_reason = "fewer_than_two_paddle_contacts"

    paddle_payload = {
        "name": paddle_spec["name"],
        "frames": paddle_frames,
        "initial_position": paddle_spec["initial_position"],
        "initial_target_position": paddle_spec["initial_target_position"],
        "workspace_min": paddle_spec["workspace_min"],
        "workspace_max": paddle_spec["workspace_max"],
        "action_scale_xyz": paddle_spec["action_scale_xyz"],
        "tilt_enabled": tilt_enabled,
        "initial_target_phi": paddle_spec.get("initial_target_phi", 0.0),
        "initial_target_theta": paddle_spec.get("initial_target_theta", 0.0),
        "max_tilt_theta": paddle_spec.get("max_tilt_theta", 0.0),
        "policy_max_tilt_theta": paddle_spec.get("policy_max_tilt_theta", 0.0),
        "tilt_center_xy": paddle_spec.get("tilt_center_xy", [0.0, -3.0]),
        "base_half_extents": paddle_spec["base_half_extents"],
        "rim_height": paddle_spec["rim_height"],
        "rim_half_thickness": paddle_spec["rim_half_thickness"],
        "policy_type": policy_type,
        "policy_config": policy_config,
        "outcome": outcome,
        "paddle_contact_count": paddle_contact_count,
        "floor_failure": floor_failure,
        "wall_contact_count": wall_contact_count,
        "terminal_reason": terminal_reason,
    }
    metadata = dict(scene["metadata"])
    metadata.update(
        {
            "policy_type": policy_type,
            "policy_config": policy_config,
            "outcome": outcome,
            "success": success,
            "paddle_contact_count": paddle_contact_count,
            "floor_failure": floor_failure,
            "wall_contact_count": wall_contact_count,
            "terminal_reason": terminal_reason,
            "action_dim": len(metadata["action_schema"]),
        }
    )
    result = {
        "metadata": metadata,
        "camera": scene["camera"],
        "static_objects": [serialize_object(spec) for spec in scene["static_objects"]],
        "dynamic_objects": [
            dynamic_payload,
            *make_paddle_render_payloads(paddle_spec, paddle_frames, fps),
        ],
        "actions": actions,
        "paddle": paddle_payload,
        "contacts": contacts,
        "success": success,
        "outcome": outcome,
    }
    output_json.write_text(json.dumps(result, separators=(",", ":")))


def paddle_command_quaternion(phi, theta, max_theta):
    roll, pitch = paddle_tilt_to_roll_pitch(phi, theta, max_theta)
    return euler_xyz_to_quaternion_xyzw((roll, pitch, 0.0))


def quaternion_multiply_xyzw(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def arm_paddle_mount_pose(paddle_position, paddle_quaternion, paddle_spec):
    handle = paddle_spec["handle"]
    direction_local = handle["direction_local"]
    grip_local = [
        float(handle["root_offset"][index])
        + float(direction_local[index]) * float(handle["grip_distance"])
        for index in range(3)
    ]
    grip_world = rotate_vector_by_quaternion(grip_local, paddle_quaternion)
    mount = [float(paddle_position[index]) + grip_world[index] for index in range(3)]
    grip_axis = rotate_vector_by_quaternion(
        tuple(-float(value) for value in direction_local),
        paddle_quaternion,
    )
    return mount, [float(value) for value in grip_axis]


def solve_arm_paddle_tip_ik(model, data, arm_spec, mount_position, tip_normal, seed_qpos):
    return solve_unitree_z1_ik(
        model,
        data,
        {**arm_spec, "tip_normal_world": [float(value) for value in tip_normal]},
        mount_position,
        seed_qpos,
    )


def arm_paddle_tip_tracking_error(model, data, mount_position, tip_normal):
    site = site_id(model, ARM_CATCHER_EE_SITE)
    tip_position = data.site_xpos[site].copy()
    tip_x_axis = data.site_xmat[site].reshape(3, 3)[:, 0].copy()
    position_error = float(
        np.linalg.norm(tip_position - np.asarray(mount_position, dtype=np.float64))
    )
    normal_error = float(np.linalg.norm(tip_x_axis - np.asarray(tip_normal, dtype=np.float64)))
    return position_error, normal_error


def simulate_arm_paddle_ball_scene(
    scene, output_json, *, fps, frames_per_episode, step_rate, steps_per_frame
):
    model, data = build_arm_paddle_model(scene, step_rate=step_rate)
    dynamic_spec = scene["dynamic_objects"][0]
    paddle_spec = scene["paddle"]
    arm_spec = scene["arm"]
    initialize_dynamic_state(model, data, dynamic_spec)
    initialize_paddle_state(model, data, paddle_spec)

    gravity = abs(float(scene["metadata"]["gravity"][2]))
    radius = float(dynamic_spec["radius"])
    policy_type = paddle_spec["policy_type"]
    policy_rng = random.Random(int(scene["metadata"]["random_seed"]) + 6262)
    policy_config = paddle_policy_config(policy_rng)
    policy_overrides = paddle_spec.get("policy_config_overrides")
    if policy_overrides:
        policy_config.update(policy_overrides)
    target_position = [float(value) for value in paddle_spec["initial_target_position"]]
    target_phi = float(paddle_spec.get("initial_target_phi", 0.0))
    target_theta = float(paddle_spec.get("initial_target_theta", 0.0))
    action_schema = scene["metadata"].get("action_schema", ["g", "dx", "dy", "dz"])
    tilt_enabled = bool(
        paddle_spec.get("tilt_enabled", False)
        and "phi" in action_schema
        and "theta" in action_schema
    )
    max_tilt_theta = float(paddle_spec.get("max_tilt_theta", 0.0))
    dynamic_payload = make_dynamic_payload(dynamic_spec)
    paddle_frames = []
    arm_frames = []
    contacts = []
    actions = []
    zero_delta = [0.0, 0.0, 0.0]
    last_action = (
        [gravity, *zero_delta, target_phi, target_theta] if tilt_enabled else [gravity, *zero_delta]
    )
    paddle_contact_count = 0
    previous_interval_paddle_contact = False
    last_paddle_contact_frame = -1000
    flight_apex_z = None
    failure_miss_frame = paddle_spec.get("failure_miss_frame")
    # The sideways miss slide must start right after a contact (ball
    # ascending); sliding while the ball descends whacks it laterally with
    # the paddle edge and ricochets it around the arena. Engaged at the first
    # counted contact past the scheduled frame, with a late fallback in case
    # contacts stop.
    miss_engaged_frame = None
    hold_strike_until_frame = -1
    reset_until_frame = -1
    floor_failure = False
    wall_contact_count = 0

    # Command-channel perturbation (robustness augmentation): small per-frame
    # Gaussian jitter plus a few sudden single-frame command errors, injected on
    # the recorded paddle command. Because the command drives the sim and is
    # recorded verbatim, the policy's own later commands (also recorded) capture
    # the recovery, so state->action stays dynamically consistent.
    action_scale_xyz = paddle_spec["action_scale_xyz"]
    command_noise = paddle_spec.get("command_noise")
    command_noise_rng = random.Random(int(scene["metadata"]["random_seed"]) + 9595)
    sudden_error_offsets = {}
    if command_noise is not None:
        sudden_low, sudden_high = command_noise.get("sudden_error_count_range", (0, 0))
        frac_lo, frac_hi = command_noise.get("sudden_error_frame_fraction_range", (0.2, 0.85))
        sudden_mag = float(command_noise.get("sudden_error_magnitude", 0.0))
        sudden_z_mag = float(command_noise.get("sudden_error_z_magnitude", 0.0))
        for _ in range(command_noise_rng.randint(int(sudden_low), int(sudden_high))):
            error_frame = command_noise_rng.randint(
                int(frac_lo * frames_per_episode),
                int(frac_hi * frames_per_episode),
            )
            error_angle = command_noise_rng.uniform(-math.pi, math.pi)
            sudden_error_offsets[error_frame] = (
                sudden_mag * math.cos(error_angle),
                sudden_mag * math.sin(error_angle),
                command_noise_rng.uniform(-sudden_z_mag, sudden_z_mag),
            )

    # Planned-failure biases (applied once the flail engages): a wrong hit
    # position that ramps in (the paddle drifts off the ball so it reliably
    # misses and the ball drops) and a wrong tilt (mild erratic mis-hits on the
    # early bounces). The tilt is gravity-scaled ~1/sqrt(g) so the lateral kick
    # stays gentle at high gravity instead of slamming the fast ball out.
    fail_pos_dir = (0.0, 0.0)
    fail_pos_max = 0.0
    fail_ramp_frames = 1
    fail_tilt_phi = 0.0
    fail_tilt_theta = 0.0
    if command_noise is not None and failure_miss_frame is not None:
        fail_bias_rng = random.Random(int(scene["metadata"]["random_seed"]) + 4242)
        pos_angle = fail_bias_rng.uniform(-math.pi, math.pi)
        fail_pos_dir = (math.cos(pos_angle), math.sin(pos_angle))
        fail_pos_max = float(command_noise.get("fail_pos_bias", 0.0))
        fail_ramp_frames = max(1, int(command_noise.get("fail_pos_ramp_frames", 1)))
        fail_tilt_phi = fail_bias_rng.uniform(-math.pi, math.pi)
        fail_tilt_theta = float(command_noise.get("fail_tilt_bias_theta", 0.0)) * min(
            2.0, math.sqrt(9.8 / max(gravity, 1.0))
        )
    miss_hold_position = None

    # Rest-start launch: the ball begins at rest on the paddle, so the policy
    # drives the paddle up on the velocity-matched ramp until the ball has
    # clearly left the blade, then hands off to the normal regulator.
    rest_start_episode = bool(scene["metadata"].get("rest_start"))
    rest_launch_done = not rest_start_episode
    rest_initial_z = float(dynamic_spec["position"][2])
    # ~0.15 s stationary hold then a brief lift (at 32 fps): long stationary
    # runs would let the world model overfit the "ball at rest" encoding.
    rest_launch_clear_rise = 0.06
    rest_hold_frames = 5
    rest_launch_timeout_frames = rest_hold_frames + 12

    zero_arm_joint_velocities = [0.0] * len(arm_spec["joint_names"])
    target_mount, target_grip_axis = arm_paddle_mount_pose(
        target_position,
        paddle_command_quaternion(target_phi, target_theta, max_tilt_theta),
        paddle_spec,
    )
    joint_targets = solve_arm_paddle_tip_ik(
        model,
        data,
        arm_spec,
        target_mount,
        target_grip_axis,
        arm_spec.get("home_qpos", [0.0] * len(arm_spec["joint_names"])),
    )
    arm_joint_positions = list(joint_targets)
    set_arm_joint_state(model, data, arm_spec, arm_joint_positions, zero_arm_joint_velocities)
    set_named_arm_ctrl(model, data, arm_spec, joint_targets)
    mount_tracking_summary = {
        "checks": 0,
        "max_tip_position_error": 0.0,
        "mean_tip_position_error": 0.0,
        "max_tip_normal_error": 0.0,
        "mean_tip_normal_error": 0.0,
    }

    def update_environment_failures(flags):
        nonlocal floor_failure
        if flags["ball_floor_contact"]:
            floor_failure = True

    for frame_idx in range(frames_per_episode):
        paddle_position, paddle_velocity = paddle_joint_state(model, data, paddle_spec)
        orientation_state = paddle_orientation_state(
            model,
            data,
            paddle_spec,
            fallback_phi=target_phi,
        )
        ball_position, quaternion, ball_velocity, angular_velocity = body_state(
            data, dynamic_spec["name"]
        )
        if flight_apex_z is not None:
            flight_apex_z = max(flight_apex_z, float(ball_position[2]))
        if (
            not rest_launch_done
            and frame_idx >= rest_hold_frames
            and (
                float(ball_position[2]) > rest_initial_z + rest_launch_clear_rise
                or frame_idx >= rest_launch_timeout_frames
            )
        ):
            rest_launch_done = True
        force_hold = rest_start_episode and not rest_launch_done and frame_idx < rest_hold_frames
        force_launch = rest_start_episode and not rest_launch_done and not force_hold
        frame_flags = paddle_contact_flags(model, data)
        update_environment_failures(frame_flags)

        mount_position, grip_axis = arm_paddle_mount_pose(
            paddle_position,
            orientation_state["quaternion_xyzw"],
            paddle_spec,
        )
        previous_arm_joint_positions = list(arm_joint_positions)
        arm_joint_positions = solve_arm_paddle_tip_ik(
            model,
            data,
            arm_spec,
            mount_position,
            grip_axis,
            previous_arm_joint_positions,
        )
        set_arm_joint_state(model, data, arm_spec, arm_joint_positions, zero_arm_joint_velocities)
        arm_joint_velocities = (
            [
                (current - previous) * fps
                for current, previous in zip(
                    arm_joint_positions, previous_arm_joint_positions, strict=False
                )
            ]
            if frame_idx > 0
            else list(zero_arm_joint_velocities)
        )
        tip_position_error, tip_normal_error = arm_paddle_tip_tracking_error(
            model,
            data,
            mount_position,
            grip_axis,
        )
        checks = mount_tracking_summary["checks"]
        mount_tracking_summary["checks"] = checks + 1
        mount_tracking_summary["max_tip_position_error"] = max(
            mount_tracking_summary["max_tip_position_error"], tip_position_error
        )
        mount_tracking_summary["mean_tip_position_error"] = (
            mount_tracking_summary["mean_tip_position_error"] * checks + tip_position_error
        ) / (checks + 1)
        mount_tracking_summary["max_tip_normal_error"] = max(
            mount_tracking_summary["max_tip_normal_error"], tip_normal_error
        )
        mount_tracking_summary["mean_tip_normal_error"] = (
            mount_tracking_summary["mean_tip_normal_error"] * checks + tip_normal_error
        ) / (checks + 1)

        dynamic_payload["frames"].append(
            frame_payload(
                frame_idx, fps, ball_position, quaternion, ball_velocity, angular_velocity
            )
        )
        paddle_frames.append(
            {
                "frame": frame_idx + 1,
                "time_seconds": frame_idx / fps,
                "position": [float(value) for value in paddle_position],
                "linear_velocity": [float(value) for value in paddle_velocity],
                "target_position": [float(value) for value in target_position],
                "orientation_phi": orientation_state["orientation_phi"],
                "orientation_theta": orientation_state["orientation_theta"],
                "target_phi": float(target_phi),
                "target_theta": float(target_theta),
                "quaternion_xyzw": orientation_state["quaternion_xyzw"],
                "angular_velocity": orientation_state["angular_velocity"],
            }
        )
        arm_frames.append(
            {
                "frame": frame_idx + 1,
                "time_seconds": frame_idx / fps,
                "joint_position": [float(value) for value in arm_joint_positions],
                "joint_velocity": [float(value) for value in arm_joint_velocities],
                "joint_target": [float(value) for value in joint_targets],
                "tip_position_error": tip_position_error,
                "tip_normal_error": tip_normal_error,
            }
        )
        interval_flags = {name: bool(value) for name, value in frame_flags.items()}

        if frame_idx < frames_per_episode - 1:
            miss_active = miss_engaged_frame is not None and frame_idx >= miss_engaged_frame
            if miss_active:
                if miss_hold_position is None:
                    miss_hold_position = (
                        float(target_position[0]),
                        float(target_position[1]),
                    )
                ramp = clip_value((frame_idx - miss_engaged_frame) / fail_ramp_frames, 0.0, 1.0)
                frame_position_bias = (
                    fail_pos_dir[0] * fail_pos_max * ramp,
                    fail_pos_dir[1] * fail_pos_max * ramp,
                )
                frame_tilt_bias = (fail_tilt_phi, fail_tilt_theta)
            else:
                frame_position_bias = (0.0, 0.0)
                frame_tilt_bias = (0.0, 0.0)
            normalized_delta, physical_delta, action_phi, action_theta = compute_paddle_action(
                paddle_spec=paddle_spec,
                policy_config=policy_config,
                rng=policy_rng,
                target_position=target_position,
                ball_position=ball_position,
                ball_velocity=ball_velocity,
                gravity=gravity,
                radius=radius,
                force_reset=frame_idx < reset_until_frame,
                force_strike=frame_idx < hold_strike_until_frame,
                force_miss=miss_active,
                force_launch=force_launch,
                force_hold=force_hold,
                failure_position_bias=frame_position_bias,
                failure_tilt_bias=frame_tilt_bias,
                failure_hold_position=miss_hold_position,
            )
            # The rest-start hold/launch stays un-perturbed. Everywhere else the
            # paddle command carries noise: the normal robustness jitter during
            # play, and a milder jitter (no sudden errors) during a planned
            # failure, on top of the persistent wrong-position/wrong-tilt biases.
            is_failure_episode = failure_miss_frame is not None
            apply_noise = command_noise is not None and not force_hold and not force_launch
            if apply_noise:
                if miss_active:
                    std_xy = command_noise["fail_jitter_delta_std_xy"]
                    std_z = command_noise["fail_jitter_delta_std_z"]
                    tilt_phi_std = command_noise["fail_jitter_tilt_std"]
                    tilt_theta_std = command_noise["fail_jitter_tilt_std"]
                    apply_sudden = False
                else:
                    std_xy = command_noise["delta_std_xy"]
                    std_z = command_noise["delta_std_z"]
                    tilt_phi_std = command_noise["tilt_phi_std"]
                    tilt_theta_std = command_noise["tilt_theta_std"]
                    apply_sudden = not is_failure_episode
                noise_dx = command_noise_rng.gauss(0.0, std_xy)
                noise_dy = command_noise_rng.gauss(0.0, std_xy)
                noise_dz = command_noise_rng.gauss(0.0, std_z)
                if apply_sudden and frame_idx in sudden_error_offsets:
                    error_dx, error_dy, error_dz = sudden_error_offsets[frame_idx]
                    noise_dx += error_dx
                    noise_dy += error_dy
                    noise_dz += error_dz
                normalized_delta = [
                    clip_value(normalized_delta[0] + noise_dx, -1.0, 1.0),
                    clip_value(normalized_delta[1] + noise_dy, -1.0, 1.0),
                    clip_value(normalized_delta[2] + noise_dz, -1.0, 1.0),
                ]
                physical_delta = [
                    normalized_delta[index] * action_scale_xyz[index] for index in range(3)
                ]
                if tilt_enabled:
                    action_phi += command_noise_rng.gauss(0.0, tilt_phi_std)
                    action_theta += command_noise_rng.gauss(0.0, tilt_theta_std)
            target_position = clip_vector(
                [target_position[index] + physical_delta[index] for index in range(3)],
                paddle_spec["workspace_min"],
                paddle_spec["workspace_max"],
            )
            if tilt_enabled:
                target_phi = wrap_angle_pi(action_phi)
                target_theta = clip_value(action_theta, 0.0, max_tilt_theta)
                action = [gravity, *normalized_delta, target_phi, target_theta]
            else:
                target_phi = 0.0
                target_theta = 0.0
                action = [gravity, *normalized_delta]
            actions.append([float(value) for value in action])
            last_action = action
            target_mount, target_grip_axis = arm_paddle_mount_pose(
                target_position,
                paddle_command_quaternion(target_phi, target_theta, max_tilt_theta),
                paddle_spec,
            )
            joint_targets = solve_arm_paddle_tip_ik(
                model,
                data,
                arm_spec,
                target_mount,
                target_grip_axis,
                joint_targets,
            )
            set_arm_joint_state(
                model, data, arm_spec, arm_joint_positions, zero_arm_joint_velocities
            )
            set_named_arm_ctrl(model, data, arm_spec, joint_targets)
            for _ in range(steps_per_frame):
                set_paddle_ctrl(data, target_position, paddle_spec, target_phi, target_theta)
                mujoco.mj_step(model, data)
                substep_flags = paddle_contact_flags(model, data)
                update_environment_failures(substep_flags)
                for name, value in substep_flags.items():
                    interval_flags[name] = bool(interval_flags[name] or value)
        else:
            actions.append([float(value) for value in last_action])

        interval_paddle_contact = bool(interval_flags["ball_paddle_contact"])
        if (
            interval_paddle_contact
            and not previous_interval_paddle_contact
            and frame_idx - last_paddle_contact_frame >= 6
        ):
            paddle_contact_count += 1
            last_paddle_contact_frame = frame_idx
            if (
                failure_miss_frame is not None
                and miss_engaged_frame is None
                and frame_idx >= int(failure_miss_frame)
            ):
                miss_engaged_frame = frame_idx + 1
            hold_frames_after_contact = int(policy_config["hold_frames_after_contact"])
            if gravity <= float(policy_config["low_gravity_hold_threshold"]):
                hold_frames_after_contact = int(
                    policy_config["low_gravity_hold_frames_after_contact"]
                )
            post_contact_velocity = body_state(data, dynamic_spec["name"])[2]
            bounce_confirmed = float(post_contact_velocity[2]) > float(
                policy_config["min_bounce_velocity_after_contact"]
            )
            update_adaptive_strike_trim(
                policy_config,
                gravity=gravity,
                achieved_apex=flight_apex_z,
            )
            flight_apex_z = float(ball_position[2])
            if not bounce_confirmed:
                hold_frames_after_contact = int(policy_config["failed_bounce_hold_frames"])
            hold_strike_until_frame = max(
                hold_strike_until_frame,
                frame_idx + hold_frames_after_contact,
            )
            if bounce_confirmed:
                reset_until_frame = max(
                    reset_until_frame,
                    frame_idx + int(policy_config["post_contact_reset_frames"]),
                )
        previous_interval_paddle_contact = interval_paddle_contact
        if (
            failure_miss_frame is not None
            and miss_engaged_frame is None
            and frame_idx >= int(failure_miss_frame)
            and float(ball_velocity[2]) > 0.0
        ):
            miss_engaged_frame = frame_idx + 1
        if (
            failure_miss_frame is not None
            and miss_engaged_frame is None
            and frame_idx >= int(failure_miss_frame) + 32
        ):
            miss_engaged_frame = frame_idx + 1
        if interval_flags["ball_wall_contact"]:
            wall_contact_count += 1
        contacts.append(interval_flags)

    success = paddle_contact_count >= 2 and not floor_failure
    if success:
        outcome = "success"
        terminal_reason = "two_or_more_paddle_contacts_without_floor_contact"
    elif floor_failure:
        outcome = "floor_failure"
        terminal_reason = "ball_floor_contact"
    else:
        outcome = "insufficient_paddle_contacts"
        terminal_reason = "fewer_than_two_paddle_contacts"

    paddle_payload = {
        "name": paddle_spec["name"],
        "frames": paddle_frames,
        "initial_position": paddle_spec["initial_position"],
        "initial_target_position": paddle_spec["initial_target_position"],
        "workspace_min": paddle_spec["workspace_min"],
        "workspace_max": paddle_spec["workspace_max"],
        "action_scale_xyz": paddle_spec["action_scale_xyz"],
        "tilt_enabled": tilt_enabled,
        "initial_target_phi": paddle_spec.get("initial_target_phi", 0.0),
        "initial_target_theta": paddle_spec.get("initial_target_theta", 0.0),
        "max_tilt_theta": paddle_spec.get("max_tilt_theta", 0.0),
        "policy_max_tilt_theta": paddle_spec.get("policy_max_tilt_theta", 0.0),
        "tilt_center_xy": paddle_spec.get("tilt_center_xy", [0.0, -3.0]),
        "base_half_extents": paddle_spec["base_half_extents"],
        "rim_height": paddle_spec["rim_height"],
        "rim_half_thickness": paddle_spec["rim_half_thickness"],
        "policy_type": policy_type,
        "policy_config": policy_config,
        "outcome": outcome,
        "paddle_contact_count": paddle_contact_count,
        "floor_failure": floor_failure,
        "wall_contact_count": wall_contact_count,
        "terminal_reason": terminal_reason,
        "mount": "paddle_handle_socketed_into_unitree_z1_end_effector_flange",
        "handle": paddle_spec.get("handle"),
    }
    arm_payload = {
        "name": arm_spec["name"],
        "robot_model": arm_spec.get("robot_model"),
        "model_scale": arm_spec.get("model_scale", 1.0),
        "frames": arm_frames,
        "base_position": arm_spec["base_position"],
        "base_euler": arm_spec.get("base_euler", [0.0, 0.0, 0.0]),
        "joint_names": arm_spec["joint_names"],
        "actuator_names": arm_spec.get("actuator_names"),
        "joint_limits": arm_spec["joint_limits"],
        "home_qpos": arm_spec.get("home_qpos"),
        "tool_tip_local_x": arm_spec.get("tool_tip_local_x", ARM_CATCHER_TOOL_TIP_LOCAL_X),
        "mount_tracking": mount_tracking_summary,
        "paddle_handle": paddle_spec.get("handle"),
        "ik_controller": "unitree_z1_tip_site_position_and_paddle_handle_axis_damped_least_squares",
    }
    metadata = dict(scene["metadata"])
    metadata.update(
        {
            "policy_type": policy_type,
            "policy_config": policy_config,
            "outcome": outcome,
            "success": success,
            "paddle_contact_count": paddle_contact_count,
            "floor_failure": floor_failure,
            "wall_contact_count": wall_contact_count,
            "terminal_reason": terminal_reason,
            "action_dim": len(metadata["action_schema"]),
            "arm_paddle_mount_tracking": mount_tracking_summary,
            "arm_state_schema": [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint1_vel",
                "joint2_vel",
                "joint3_vel",
                "joint4_vel",
                "joint5_vel",
                "joint6_vel",
                "target_joint1",
                "target_joint2",
                "target_joint3",
                "target_joint4",
                "target_joint5",
                "target_joint6",
            ],
        }
    )
    result = {
        "metadata": metadata,
        "camera": scene["camera"],
        "static_objects": [serialize_object(spec) for spec in scene["static_objects"]],
        "dynamic_objects": [
            dynamic_payload,
            *make_paddle_render_payloads(paddle_spec, paddle_frames, fps),
        ],
        "actions": actions,
        "paddle": paddle_payload,
        "arm": arm_payload,
        "contacts": contacts,
        "success": success,
        "outcome": outcome,
    }
    output_json.write_text(json.dumps(result, separators=(",", ":")))


def simulate_catcher_ball_scene(
    scene, output_json, *, fps, frames_per_episode, step_rate, steps_per_frame
):
    model, data = build_catcher_model(scene, step_rate=step_rate)
    dynamic_spec = scene["dynamic_objects"][0]
    catcher_spec = scene["catcher"]
    initialize_dynamic_state(model, data, dynamic_spec)
    initialize_catcher_state(model, data, catcher_spec)

    gravity = abs(float(scene["metadata"]["gravity"][2]))
    radius = float(dynamic_spec["radius"])
    policy_type = catcher_spec["policy_type"]
    control_axes = list(catcher_spec.get("control_axes", ["x", "z"]))
    planned_outcome = catcher_spec.get(
        "planned_outcome",
        scene["metadata"].get("planned_outcome", "unplanned"),
    )
    policy_rng = random.Random(int(scene["metadata"]["random_seed"]) + 4242)
    policy_config = catcher_policy_config(
        policy_type,
        policy_rng,
        planned_outcome=planned_outcome,
        miss_bias=catcher_spec.get("miss_bias"),
        control_axes=control_axes,
        policy_noise_scale=catcher_spec.get("policy_noise_scale", 1.0),
        policy_bias_scale=catcher_spec.get("policy_bias_scale", 1.0),
    )
    target_position = [float(value) for value in catcher_spec["initial_target_position"]]
    dynamic_payload = make_dynamic_payload(dynamic_spec)
    catcher_frames = []
    contacts = []
    caught_flags = []
    captured_flags = []
    missed_flags = []
    truncated_flags = []
    actions = []
    observation_history = []
    zero_delta = [0.0] * len(control_axes)
    last_action = [gravity, *zero_delta]
    captured = False
    missed = False
    reached_catch_plane = False
    capture_frame = None
    miss_frame = None
    terminal_reason = None
    latch_offset = None
    capture_window = 3

    for frame_idx in range(frames_per_episode):
        catcher_position, catcher_velocity = catcher_joint_state(model, data, catcher_spec)
        if captured:
            set_locked_ball_state(
                model,
                data,
                dynamic_spec,
                add_vectors(catcher_position, latch_offset),
                catcher_velocity,
            )
        ball_position, quaternion, ball_velocity, angular_velocity = body_state(
            data, dynamic_spec["name"]
        )
        quaternion = (0.0, 0.0, 0.0, 1.0)
        angular_velocity = (0.0, 0.0, 0.0)
        flags = contact_flags(model, data)
        observation_history.append((list(ball_position), list(ball_velocity)))
        if (
            catcher_reached_front_plane(ball_position, catcher_position, catcher_spec, radius)
            or flags["ball_catcher_contact"]
        ):
            reached_catch_plane = True
        candidate_caught = (
            caught_frame(
                ball_position,
                ball_velocity,
                catcher_position,
                catcher_velocity,
                flags,
                catcher_spec,
                radius,
            )
            if not missed
            else False
        )
        if (
            not captured
            and not missed
            and has_consecutive_true(
                [*caught_flags, candidate_caught][-capture_window:], capture_window
            )
        ):
            rel_pos = [ball_position[index] - catcher_position[index] for index in range(3)]
            half_width = catcher_spec["net_half_width"]
            half_height = catcher_spec["net_half_height"]
            latch_y_min, latch_y_max = catcher_latch_y_bounds(catcher_spec)
            latch_offset = [
                clip_value(rel_pos[0], -0.45 * half_width, 0.45 * half_width),
                clip_value(rel_pos[1], latch_y_min, latch_y_max),
                clip_value(rel_pos[2], -0.45 * half_height, 0.45 * half_height),
            ]
            captured = True
            capture_frame = frame_idx
            set_locked_ball_state(
                model,
                data,
                dynamic_spec,
                add_vectors(catcher_position, latch_offset),
                catcher_velocity,
            )
            ball_position, quaternion, ball_velocity, angular_velocity = body_state(
                data, dynamic_spec["name"]
            )
            quaternion = (0.0, 0.0, 0.0, 1.0)
            angular_velocity = (0.0, 0.0, 0.0)
            flags = contact_flags(model, data)
            candidate_caught = True

        if not captured and not missed:
            reason = miss_reason(
                ball_position,
                catcher_position,
                flags,
                catcher_spec,
                radius,
                reached_catch_plane,
            )
            if reason is not None:
                missed = True
                miss_frame = frame_idx
                terminal_reason = reason
                candidate_caught = False

        dynamic_payload["frames"].append(
            frame_payload(
                frame_idx, fps, ball_position, quaternion, ball_velocity, angular_velocity
            )
        )
        catcher_frames.append(
            {
                "frame": frame_idx + 1,
                "time_seconds": frame_idx / fps,
                "position": [float(value) for value in catcher_position],
                "linear_velocity": [float(value) for value in catcher_velocity],
                "target_position": [float(value) for value in target_position],
            }
        )
        contacts.append({name: bool(value) for name, value in flags.items()})
        caught_flags.append(bool(candidate_caught))
        captured_flags.append(bool(captured))
        missed_flags.append(bool(missed))
        truncated_flags.append(False)

        if frame_idx < frames_per_episode - 1:
            if captured or missed:
                delta = [0.0, 0.0]
            else:
                delta = compute_catcher_action(
                    catcher_spec=catcher_spec,
                    policy_type=policy_type,
                    policy_config=policy_config,
                    rng=policy_rng,
                    target_position=target_position,
                    observation_history=observation_history,
                    gravity=gravity,
                    radius=radius,
                )
            target_position = clip_vector(
                [
                    target_position[0] + delta[0],
                    catcher_spec["fixed_y"],
                    target_position[2] + delta[1],
                ],
                catcher_spec["workspace_min"],
                catcher_spec["workspace_max"],
            )
            action = [gravity, *delta]
            actions.append([float(value) for value in action])
            last_action = action
            for _ in range(steps_per_frame):
                set_catcher_ctrl(data, target_position)
                mujoco.mj_step(model, data)
                if captured:
                    catcher_step_position, catcher_step_velocity = catcher_joint_state(
                        model, data, catcher_spec
                    )
                    set_locked_ball_state(
                        model,
                        data,
                        dynamic_spec,
                        add_vectors(catcher_step_position, latch_offset),
                        catcher_step_velocity,
                    )
        else:
            actions.append([float(value) for value in last_action])

    success = any(captured_flags)
    truncated_before_interaction = False
    outcome = "capture" if success else None
    if success:
        terminal_reason = terminal_reason or "captured"
    elif missed:
        outcome = "miss"
        if miss_frame is None:
            miss_frame = frames_per_episode - 1
        terminal_reason = terminal_reason or "missed"
    elif not reached_catch_plane:
        outcome = "truncated"
        truncated_before_interaction = True
        terminal_reason = "max_length_before_interaction"
    else:
        outcome = "miss"
        missed = True
        miss_frame = frames_per_episode - 1
        missed_flags[-1] = True
        terminal_reason = "max_length_after_interaction"

    if missed and miss_frame is not None:
        missed_flags = [frame_idx >= miss_frame for frame_idx in range(frames_per_episode)]
    truncated_flags = [truncated_before_interaction for _ in range(frames_per_episode)]

    catcher_payload = {
        "name": catcher_spec["name"],
        "frames": catcher_frames,
        "initial_position": catcher_spec["initial_position"],
        "initial_target_position": catcher_spec["initial_target_position"],
        "workspace_min": catcher_spec["workspace_min"],
        "workspace_max": catcher_spec["workspace_max"],
        "target_y": catcher_spec["target_y"],
        "fixed_y": catcher_spec["fixed_y"],
        "net_half_width": catcher_spec["net_half_width"],
        "net_half_height": catcher_spec["net_half_height"],
        "net_depth": catcher_spec["net_depth"],
        "policy_type": policy_type,
        "policy_config": policy_config,
        "planned_outcome": planned_outcome,
        "outcome": outcome,
        "capture_window_frames": capture_window,
        "capture_frame": capture_frame,
        "miss_frame": miss_frame,
        "truncated_before_interaction": truncated_before_interaction,
        "terminal_reason": terminal_reason,
        "latch_offset": latch_offset,
    }
    metadata = dict(scene["metadata"])
    metadata.update(
        {
            "policy_type": policy_type,
            "policy_config": policy_config,
            "planned_outcome": planned_outcome,
            "outcome": outcome,
            "success": success,
            "capture_window_frames": capture_window,
            "capture_frame": capture_frame,
            "miss_frame": miss_frame,
            "truncated_before_interaction": truncated_before_interaction,
            "terminal_reason": terminal_reason,
            "latch_after_capture": True,
            "action_dim": len(metadata["action_schema"]),
        }
    )
    result = {
        "metadata": metadata,
        "camera": scene["camera"],
        "static_objects": [serialize_object(spec) for spec in scene["static_objects"]],
        "dynamic_objects": [
            dynamic_payload,
            *make_catcher_render_payloads(catcher_spec, catcher_frames, fps),
        ],
        "actions": actions,
        "catcher": catcher_payload,
        "contacts": contacts,
        "caught_frames": caught_flags,
        "captured": captured_flags,
        "missed": missed_flags,
        "truncated": truncated_flags,
        "success": success,
        "outcome": outcome,
    }
    output_json.write_text(json.dumps(result, separators=(",", ":")))


def simulate_arm_catcher_ball_scene(
    scene, output_json, *, fps, frames_per_episode, step_rate, steps_per_frame
):
    model, data = build_arm_catcher_model(scene, step_rate=step_rate)
    dynamic_spec = scene["dynamic_objects"][0]
    catcher_spec = scene["catcher"]
    arm_spec = scene["arm"]
    # Every mount position below is the net's back face, which is seated behind
    # the flange the IK actually drives. These two wrappers are the only places
    # that conversion happens.
    mount_back_offset = float(catcher_spec.get("mount_back_offset", ARM_CATCHER_MOUNT_BACK_OFFSET))
    mount_tip_normal = arm_spec.get("tip_normal_world", ARM_CATCHER_TIP_NORMAL_WORLD)

    def sync_mount():
        return sync_arm_catcher_mount_to_tip(
            model,
            data,
            back_offset=mount_back_offset,
            tip_normal=mount_tip_normal,
        )

    def tip_target_for(mount_target):
        return arm_catcher_tip_target_for_mount(mount_target, mount_back_offset, mount_tip_normal)

    initialize_dynamic_state(model, data, dynamic_spec)
    target_position, joint_targets = initialize_arm_catcher_state(
        model, data, catcher_spec, arm_spec
    )

    gravity = abs(float(scene["metadata"]["gravity"][2]))
    radius = float(dynamic_spec["radius"])
    policy_type = catcher_spec["policy_type"]
    control_axes = list(catcher_spec.get("control_axes", ["x", "z"]))
    planned_outcome = catcher_spec.get(
        "planned_outcome",
        scene["metadata"].get("planned_outcome", "unplanned"),
    )
    policy_rng = random.Random(int(scene["metadata"]["random_seed"]) + 8484)
    policy_config = catcher_policy_config(
        policy_type,
        policy_rng,
        planned_outcome=planned_outcome,
        miss_bias=catcher_spec.get("miss_bias"),
        control_axes=control_axes,
        policy_noise_scale=catcher_spec.get("policy_noise_scale", 1.0),
        policy_bias_scale=catcher_spec.get("policy_bias_scale", 1.0),
    )
    dynamic_payload = make_dynamic_payload(dynamic_spec)
    catcher_frames = []
    arm_frames = []
    contacts = []
    caught_flags = []
    captured_flags = []
    missed_flags = []
    truncated_flags = []
    actions = []
    observation_history = []
    zero_delta = [0.0] * len(control_axes)
    smoothed_physical_delta = list(zero_delta)
    wiggle_amplitude = catcher_spec.get("planner_wiggle_amplitude_xyz")
    wiggle_tau = float(catcher_spec.get("planner_wiggle_tau_seconds", 0.35))
    wiggle_state = [0.0, 0.0, 0.0]
    last_action = [gravity, *zero_delta]
    captured = False
    missed = False
    reached_catch_plane = False
    capture_frame = None
    miss_frame = None
    terminal_reason = None
    latch_offset = None
    latch_world_jump = None
    capture_window = max(1, int(catcher_spec.get("capture_window_frames", 2)))
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    planner_max_accel = arm_catcher_planner_accel_limits(catcher_spec, control_axes)
    # The commanded target may run ahead of the achieved EE (the IK has a
    # few-mm residual floor, so it needs an offset larger than that to act),
    # but a leash keeps an IK stall from turning into unbounded target drift.
    planner_target_leash = [
        3.0 * value for value in arm_catcher_planner_speed_limits(catcher_spec, control_axes)
    ]
    max_joint_step = abs(float(catcher_spec.get("planner_max_joint_step", 0.30)))
    latch_half_width = float(
        catcher_spec.get("latch_half_width", 0.45 * float(catcher_spec["net_half_width"]))
    )
    latch_half_height = float(
        catcher_spec.get("latch_half_height", 0.45 * float(catcher_spec["net_half_height"]))
    )
    latch_y_min, latch_y_max = catcher_latch_y_bounds(catcher_spec)
    max_latch_jump = catcher_spec.get("capture_max_latch_jump")

    def compute_latch_candidate(ball_position_value, ee_position_value):
        rel_pos = [
            float(ball_position_value[index]) - float(ee_position_value[index])
            for index in range(3)
        ]
        offset = [
            clip_value(rel_pos[0], -latch_half_width, latch_half_width),
            clip_value(rel_pos[1], latch_y_min, latch_y_max),
            clip_value(rel_pos[2], -latch_half_height, latch_half_height),
        ]
        locked = clamp_position_above_ground(
            add_vectors(ee_position_value, offset),
            radius,
            ground_z=ground_z,
        )
        jump = float(
            np.linalg.norm(
                np.asarray(locked, dtype=np.float64)
                - np.asarray([float(value) for value in ball_position_value], dtype=np.float64)
            )
        )
        return offset, locked, jump

    zero_arm_joint_velocities = [0.0] * len(arm_spec["joint_names"])
    ground_z = float(catcher_spec.get("ground_z", 0.0))
    mount_validation_summary = {
        "checks": 0,
        "max_tip_mount_position_error": 0.0,
        "max_flange_circle_normal_error": 0.0,
        "max_flange_circle_plane_to_catcher_plane_distance": 0.0,
        "max_x_axis_world_error": 0.0,
        "max_z_axis_world_error": 0.0,
        "max_open_axis_world_error": 0.0,
        "mount_body_parent": None,
        "mount_body_parent_is_world": None,
        "catcher_geoms_on_mount_body": None,
        "mount_origin": catcher_spec.get("mount_origin"),
        "open_axis_world": list(catcher_spec.get("open_axis_world", ARM_CATCHER_OPEN_AXIS_WORLD)),
        "flange_circle_normal_start": None,
        "flange_circle_normal_final": None,
    }

    def record_mount_validation():
        check = validate_arm_catcher_mount(model, data, catcher_spec)
        mount_validation_summary["checks"] += 1
        mount_validation_summary["max_tip_mount_position_error"] = max(
            mount_validation_summary["max_tip_mount_position_error"],
            check["tip_mount_position_error"],
        )
        mount_validation_summary["max_flange_circle_normal_error"] = max(
            mount_validation_summary["max_flange_circle_normal_error"],
            check["flange_circle_normal_error"],
        )
        mount_validation_summary["max_flange_circle_plane_to_catcher_plane_distance"] = max(
            mount_validation_summary["max_flange_circle_plane_to_catcher_plane_distance"],
            check["flange_circle_plane_to_catcher_plane_distance"],
        )
        mount_validation_summary["max_x_axis_world_error"] = max(
            mount_validation_summary["max_x_axis_world_error"],
            check["x_axis_world_error"],
        )
        mount_validation_summary["max_z_axis_world_error"] = max(
            mount_validation_summary["max_z_axis_world_error"],
            check["z_axis_world_error"],
        )
        mount_validation_summary["max_open_axis_world_error"] = max(
            mount_validation_summary["max_open_axis_world_error"],
            check["open_axis_world_error"],
        )
        mount_validation_summary["mount_body_parent"] = check["mount_body_parent"]
        mount_validation_summary["mount_body_parent_is_world"] = check["mount_body_parent_is_world"]
        mount_validation_summary["catcher_geoms_on_mount_body"] = check[
            "catcher_geoms_on_mount_body"
        ]
        if mount_validation_summary["flange_circle_normal_start"] is None:
            mount_validation_summary["flange_circle_normal_start"] = list(
                check["flange_circle_normal_world"]
            )
        mount_validation_summary["flange_circle_normal_final"] = list(
            check["flange_circle_normal_world"]
        )

    def limit_joint_targets_by_mount_delta(
        current_joint_positions, proposed_joint_targets, current_mount_position
    ):
        per_frame_max_delta = catcher_spec.get("per_frame_max_delta_xyz")
        if per_frame_max_delta is None:
            return [float(value) for value in proposed_joint_targets]

        limits = np.asarray(per_frame_max_delta, dtype=np.float64)
        current_mount = np.asarray(current_mount_position, dtype=np.float64)
        start_joints = np.asarray(current_joint_positions, dtype=np.float64)
        end_joints = np.asarray(proposed_joint_targets, dtype=np.float64)

        saved_qpos = data.qpos.copy()
        saved_qvel = data.qvel.copy()
        saved_mocap_pos = data.mocap_pos.copy()
        saved_mocap_quat = data.mocap_quat.copy()

        def restore_state():
            data.qpos[:] = saved_qpos
            data.qvel[:] = saved_qvel
            data.mocap_pos[:] = saved_mocap_pos
            data.mocap_quat[:] = saved_mocap_quat
            mujoco.mj_forward(model, data)

        def mount_position_for(beta):
            candidate_joints = (1.0 - beta) * start_joints + beta * end_joints
            set_arm_joint_state(
                model,
                data,
                arm_spec,
                candidate_joints,
                zero_arm_joint_velocities,
            )
            candidate_mount, _ = sync_mount()
            return candidate_joints, np.asarray(candidate_mount, dtype=np.float64)

        def within_limits(candidate_mount):
            return bool(np.all(np.abs(candidate_mount - current_mount) <= limits + 1.0e-9))

        full_joints, full_mount = mount_position_for(1.0)
        if within_limits(full_mount):
            restore_state()
            return [float(value) for value in full_joints]

        best_joints = start_joints
        low = 0.0
        high = 1.0
        for _ in range(12):
            mid = 0.5 * (low + high)
            candidate_joints, candidate_mount = mount_position_for(mid)
            if within_limits(candidate_mount):
                best_joints = candidate_joints
                low = mid
            else:
                high = mid

        restore_state()
        return [float(value) for value in best_joints]

    for frame_idx in range(frames_per_episode):
        ee_position, ee_velocity = sync_mount()
        if captured and latch_offset is not None:
            set_locked_ball_state(
                model,
                data,
                dynamic_spec,
                clamp_position_above_ground(
                    add_vectors(ee_position, latch_offset),
                    radius,
                    ground_z=ground_z,
                ),
                (0.0, 0.0, 0.0),
            )
            ee_position, ee_velocity = sync_mount()
        record_mount_validation()
        ball_position, quaternion, ball_velocity, angular_velocity = body_state(
            data, dynamic_spec["name"]
        )
        flags = contact_flags(model, data)
        if (
            not captured
            and float(ball_position[2]) < ground_z + radius
            and enforce_ball_ground_bounce(
                model,
                data,
                dynamic_spec,
                radius,
                ground_z=ground_z,
            )
        ):
            ball_position, quaternion, ball_velocity, angular_velocity = body_state(
                data,
                dynamic_spec["name"],
            )
            flags = contact_flags(model, data)
            flags["ball_floor_contact"] = True
        observation_history.append((list(ball_position), list(ball_velocity)))
        if (
            catcher_reached_front_plane(ball_position, ee_position, catcher_spec, radius)
            or flags["ball_catcher_contact"]
        ):
            reached_catch_plane = True
        candidate_caught = (
            caught_frame(
                ball_position,
                ball_velocity,
                ee_position,
                ee_velocity,
                flags,
                catcher_spec,
                radius,
            )
            if not missed
            else False
        )
        if (
            not captured
            and not missed
            and has_consecutive_true(
                [*caught_flags, candidate_caught][-capture_window:], capture_window
            )
        ):
            candidate_latch_offset, candidate_locked_position, candidate_latch_world_jump = (
                compute_latch_candidate(ball_position, ee_position)
            )
            if max_latch_jump is not None and candidate_latch_world_jump > float(max_latch_jump):
                candidate_caught = False
            else:
                latch_offset = candidate_latch_offset
                latch_world_jump = candidate_latch_world_jump
                captured = True
                capture_frame = frame_idx
                joint_targets, _ = arm_joint_state(model, data, arm_spec)
                set_arm_ctrl(data, joint_targets)
                target_position = clip_vector(
                    list(ee_position),
                    catcher_spec["workspace_min"],
                    catcher_spec["workspace_max"],
                )
                set_locked_ball_state(
                    model,
                    data,
                    dynamic_spec,
                    clamp_position_above_ground(
                        add_vectors(ee_position, latch_offset),
                        radius,
                        ground_z=ground_z,
                    ),
                    (0.0, 0.0, 0.0),
                )
                ball_position, quaternion, ball_velocity, angular_velocity = body_state(
                    data, dynamic_spec["name"]
                )
                flags = contact_flags(model, data)
                candidate_caught = True

        if not captured and not missed:
            reason = miss_reason(
                ball_position,
                ee_position,
                flags,
                catcher_spec,
                radius,
                reached_catch_plane,
            )
            if reason is not None:
                missed = True
                miss_frame = frame_idx
                terminal_reason = reason
                candidate_caught = False

        joint_positions, joint_velocities = arm_joint_state(model, data, arm_spec)
        dynamic_payload["frames"].append(
            frame_payload(
                frame_idx, fps, ball_position, quaternion, ball_velocity, angular_velocity
            )
        )
        catcher_frames.append(
            {
                "frame": frame_idx + 1,
                "time_seconds": frame_idx / fps,
                "position": [float(value) for value in ee_position],
                "linear_velocity": [float(value) for value in ee_velocity],
                "target_position": [float(value) for value in target_position],
            }
        )
        arm_frames.append(
            {
                "frame": frame_idx + 1,
                "time_seconds": frame_idx / fps,
                "joint_position": [float(value) for value in joint_positions],
                "joint_velocity": [float(value) for value in joint_velocities],
                "joint_target": [float(value) for value in joint_targets],
            }
        )
        contacts.append({name: bool(value) for name, value in flags.items()})
        caught_flags.append(bool(candidate_caught))
        captured_flags.append(bool(captured))
        missed_flags.append(bool(missed))
        truncated_flags.append(False)

        if frame_idx < frames_per_episode - 1:
            if captured or missed:
                desired_step = list(zero_delta)
            else:
                wiggle_offset = None
                if wiggle_amplitude is not None and policy_type != "random":
                    ou_alpha = math.exp(-1.0 / (float(fps) * max(wiggle_tau, 1.0e-6)))
                    ou_scale = math.sqrt(max(1.0 - ou_alpha * ou_alpha, 0.0))
                    wiggle_state = [
                        ou_alpha * wiggle_state[index]
                        + ou_scale
                        * abs(float(wiggle_amplitude[index]))
                        * policy_rng.gauss(0.0, 1.0)
                        for index in range(3)
                    ]
                    wiggle_offset = wiggle_state
                desired_step = compute_arm_catcher_interception_step(
                    catcher_spec=catcher_spec,
                    policy_type=policy_type,
                    policy_config=policy_config,
                    rng=policy_rng,
                    target_position=target_position,
                    observation_history=observation_history,
                    gravity=gravity,
                    radius=radius,
                    fps=fps,
                    wiggle_offset=wiggle_offset,
                )
            physical_delta = limit_physical_step_change(
                smoothed_physical_delta,
                desired_step,
                planner_max_accel,
            )
            smoothed_physical_delta = list(physical_delta)
            target_position = apply_control_delta(
                list(target_position),
                physical_delta,
                control_axes,
                catcher_spec,
            )
            for index, axis in enumerate(control_axes):
                axis_index = axis_to_index[axis]
                target_position[axis_index] = clip_value(
                    target_position[axis_index],
                    float(ee_position[axis_index]) - planner_target_leash[index],
                    float(ee_position[axis_index]) + planner_target_leash[index],
                )
            current_joint_positions, _ = arm_joint_state(model, data, arm_spec)
            if vector_norm(physical_delta) > 1.0e-9:
                saved_qpos = data.qpos.copy()
                saved_qvel = data.qvel.copy()
                joint_targets = solve_unitree_z1_ik(
                    model,
                    data,
                    arm_spec,
                    tip_target_for(target_position),
                    current_joint_positions,
                    allow_fallback_seeds=False,
                )
                data.qpos[:] = saved_qpos
                data.qvel[:] = saved_qvel
                mujoco.mj_forward(model, data)
                joint_targets = [
                    float(current)
                    + clip_value(float(target) - float(current), -max_joint_step, max_joint_step)
                    for current, target in zip(current_joint_positions, joint_targets, strict=False)
                ]
                joint_targets = limit_joint_targets_by_mount_delta(
                    current_joint_positions,
                    joint_targets,
                    ee_position,
                )
            else:
                joint_targets = [float(value) for value in current_joint_positions]
            start_joint_positions = np.asarray(current_joint_positions, dtype=np.float64)
            end_joint_positions = np.asarray(joint_targets, dtype=np.float64)
            for step_index in range(steps_per_frame):
                beta = smoothstep01((step_index + 1) / max(steps_per_frame, 1))
                step_joint_positions = (
                    1.0 - beta
                ) * start_joint_positions + beta * end_joint_positions
                set_arm_ctrl(data, joint_targets)
                set_arm_joint_state(
                    model,
                    data,
                    arm_spec,
                    step_joint_positions,
                    zero_arm_joint_velocities,
                )
                ee_step_position, _ = sync_mount()
                if captured:
                    set_locked_ball_state(
                        model,
                        data,
                        dynamic_spec,
                        clamp_position_above_ground(
                            add_vectors(ee_step_position, latch_offset),
                            radius,
                            ground_z=ground_z,
                        ),
                        (0.0, 0.0, 0.0),
                    )
                else:
                    mujoco.mj_step(model, data)
                    enforce_ball_ground_bounce(
                        model,
                        data,
                        dynamic_spec,
                        radius,
                        ground_z=ground_z,
                    )
                    ee_step_position, ee_step_velocity = sync_mount()
                    # Substep-resolution capture check: at frame rate a
                    # fast-falling ball can cross the whole capture window
                    # between samples, which would make steep approaches
                    # uncatchable regardless of arm placement.
                    if not missed:
                        step_ball_position, _, step_ball_velocity, _ = body_state(
                            data,
                            dynamic_spec["name"],
                        )
                        step_flags = contact_flags(model, data)
                        if caught_frame(
                            step_ball_position,
                            step_ball_velocity,
                            ee_step_position,
                            ee_step_velocity,
                            step_flags,
                            catcher_spec,
                            radius,
                        ):
                            step_latch_offset, step_locked_position, step_latch_jump = (
                                compute_latch_candidate(step_ball_position, ee_step_position)
                            )
                            if max_latch_jump is None or step_latch_jump <= float(max_latch_jump):
                                latch_offset = step_latch_offset
                                latch_world_jump = step_latch_jump
                                captured = True
                                capture_frame = min(frame_idx + 1, frames_per_episode - 1)
                                joint_targets = list(joint_targets)
                                set_locked_ball_state(
                                    model,
                                    data,
                                    dynamic_spec,
                                    step_locked_position,
                                    (0.0, 0.0, 0.0),
                                )
            set_arm_joint_state(
                model,
                data,
                arm_spec,
                joint_targets,
                zero_arm_joint_velocities,
            )
            ee_after_position, _ = sync_mount()
            if captured:
                set_locked_ball_state(
                    model,
                    data,
                    dynamic_spec,
                    clamp_position_above_ground(
                        add_vectors(ee_after_position, latch_offset),
                        radius,
                        ground_z=ground_z,
                    ),
                    (0.0, 0.0, 0.0),
                )
            achieved_physical_delta = [
                float(ee_after_position[axis_to_index[axis]])
                - float(ee_position[axis_to_index[axis]])
                for axis in control_axes
            ]
            normalized_delta = normalized_delta_from_physical_delta(
                achieved_physical_delta,
                control_axes,
                catcher_spec,
            )
            action = [gravity, *normalized_delta]
            actions.append([float(value) for value in action])
            last_action = action
        else:
            actions.append([float(value) for value in last_action])

    success = any(captured_flags)
    truncated_before_interaction = False
    outcome = "capture" if success else None
    if success:
        terminal_reason = terminal_reason or "captured"
    elif missed:
        outcome = "miss"
        if miss_frame is None:
            miss_frame = frames_per_episode - 1
        terminal_reason = terminal_reason or "missed"
    elif not reached_catch_plane:
        outcome = "truncated"
        truncated_before_interaction = True
        terminal_reason = "max_length_before_interaction"
    else:
        outcome = "miss"
        missed = True
        miss_frame = frames_per_episode - 1
        missed_flags[-1] = True
        terminal_reason = "max_length_after_interaction"

    if missed and miss_frame is not None:
        missed_flags = [frame_idx >= miss_frame for frame_idx in range(frames_per_episode)]
    truncated_flags = [truncated_before_interaction for _ in range(frames_per_episode)]

    catcher_payload = {
        "name": catcher_spec["name"],
        "frames": catcher_frames,
        "initial_position": catcher_spec["initial_position"],
        "initial_target_position": catcher_spec["initial_target_position"],
        "workspace_min": catcher_spec["workspace_min"],
        "workspace_max": catcher_spec["workspace_max"],
        "target_y": catcher_spec["target_y"],
        "fixed_y": catcher_spec["fixed_y"],
        "net_half_width": catcher_spec["net_half_width"],
        "net_half_height": catcher_spec["net_half_height"],
        "net_depth": catcher_spec["net_depth"],
        "capture_center_half_width": catcher_spec.get("capture_center_half_width"),
        "capture_center_half_height": catcher_spec.get("capture_center_half_height"),
        "capture_center_radius": catcher_spec.get("capture_center_radius"),
        "capture_front_margin": catcher_spec.get("capture_front_margin"),
        "capture_back_margin": catcher_spec.get("capture_back_margin"),
        "capture_max_latch_jump": catcher_spec.get("capture_max_latch_jump"),
        "latch_half_width": catcher_spec.get("latch_half_width"),
        "latch_half_height": catcher_spec.get("latch_half_height"),
        "latch_y_min": catcher_spec.get("latch_y_min"),
        "latch_y_max": catcher_spec.get("latch_y_max"),
        "mount_origin": catcher_spec.get("mount_origin"),
        "back_face_mount_offset": catcher_spec.get("back_face_mount_offset"),
        "open_axis_world": catcher_spec.get("open_axis_world"),
        "x_axis_world": catcher_spec.get("x_axis_world"),
        "z_axis_world": catcher_spec.get("z_axis_world"),
        "control_axes": control_axes,
        "action_scale_xyz": catcher_spec.get("action_scale_xyz"),
        "per_frame_max_delta_xyz": catcher_spec.get("per_frame_max_delta_xyz"),
        "policy_type": policy_type,
        "policy_config": policy_config,
        "planned_outcome": planned_outcome,
        "outcome": outcome,
        "capture_window_frames": capture_window,
        "capture_frame": capture_frame,
        "miss_frame": miss_frame,
        "truncated_before_interaction": truncated_before_interaction,
        "terminal_reason": terminal_reason,
        "latch_offset": latch_offset,
        "latch_world_jump": latch_world_jump,
        "freeze_arm_after_capture": False,
        "mount_validation": mount_validation_summary,
        "action_smoothness": action_smoothness_summary(actions),
    }
    if catcher_frames:
        start_mount_position = catcher_frames[0]["position"]
        final_mount_position = catcher_frames[-1]["position"]
        reference_frame = capture_frame if capture_frame is not None else frames_per_episode - 1
        reference_mount_position = catcher_frames[reference_frame]["position"]
        mount_positions = [
            np.asarray(frame["position"], dtype=np.float64) for frame in catcher_frames
        ]
        frame_deltas = [
            float(np.linalg.norm(mount_positions[index] - mount_positions[index - 1]))
            for index in range(1, len(mount_positions))
        ]
        frame_accels = [
            float(
                np.linalg.norm(
                    mount_positions[index + 1]
                    - 2.0 * mount_positions[index]
                    + mount_positions[index - 1]
                )
            )
            for index in range(1, len(mount_positions) - 1)
        ]
        mount_validation_summary["start_mount_position"] = list(start_mount_position)
        mount_validation_summary["reference_mount_position"] = list(reference_mount_position)
        mount_validation_summary["final_mount_position"] = list(final_mount_position)
        mount_validation_summary["start_to_reference_mount_distance"] = float(
            np.linalg.norm(
                np.asarray(start_mount_position, dtype=np.float64)
                - np.asarray(reference_mount_position, dtype=np.float64)
            )
        )
        mount_validation_summary["max_frame_mount_delta"] = (
            max(frame_deltas) if frame_deltas else 0.0
        )
        mount_validation_summary["max_frame_mount_acceleration_delta"] = (
            max(frame_accels) if frame_accels else 0.0
        )
    arm_payload = {
        "name": arm_spec["name"],
        "robot_model": arm_spec.get("robot_model"),
        "model_scale": arm_spec.get("model_scale", 1.0),
        "frames": arm_frames,
        "base_position": arm_spec["base_position"],
        "base_euler": arm_spec.get("base_euler", [0.0, 0.0, 0.0]),
        "shoulder_height": arm_spec["shoulder_height"],
        "upper_link_length": arm_spec["upper_link_length"],
        "lower_link_length": arm_spec["lower_link_length"],
        "joint_names": arm_spec["joint_names"],
        "joint_limits": arm_spec["joint_limits"],
        "tip_normal_world": list(arm_spec.get("tip_normal_world", ARM_CATCHER_TIP_NORMAL_WORLD)),
        "ik_controller": "unitree_z1_tip_site_position_and_world_y_normal_damped_least_squares",
    }
    metadata = dict(scene["metadata"])
    metadata.update(
        {
            "policy_type": policy_type,
            "policy_config": policy_config,
            "planned_outcome": planned_outcome,
            "outcome": outcome,
            "success": success,
            "capture_window_frames": capture_window,
            "capture_frame": capture_frame,
            "miss_frame": miss_frame,
            "truncated_before_interaction": truncated_before_interaction,
            "terminal_reason": terminal_reason,
            "latch_after_capture": True,
            "latch_world_jump": latch_world_jump,
            "capture_front_margin": catcher_spec.get("capture_front_margin"),
            "capture_back_margin": catcher_spec.get("capture_back_margin"),
            "capture_max_latch_jump": catcher_spec.get("capture_max_latch_jump"),
            "latch_half_width": catcher_spec.get("latch_half_width"),
            "latch_half_height": catcher_spec.get("latch_half_height"),
            "latch_y_min": catcher_spec.get("latch_y_min"),
            "latch_y_max": catcher_spec.get("latch_y_max"),
            "per_frame_max_delta_xyz": catcher_spec.get("per_frame_max_delta_xyz"),
            "freeze_arm_after_capture": False,
            "arm_catcher_mount_validation": mount_validation_summary,
            "arm_catcher_action_smoothness": action_smoothness_summary(actions),
            "action_dim": len(metadata["action_schema"]),
            "arm_state_schema": [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint1_vel",
                "joint2_vel",
                "joint3_vel",
                "joint4_vel",
                "joint5_vel",
                "joint6_vel",
                "target_joint1",
                "target_joint2",
                "target_joint3",
                "target_joint4",
                "target_joint5",
                "target_joint6",
            ],
        }
    )
    result = {
        "metadata": metadata,
        "camera": scene["camera"],
        "static_objects": [serialize_object(spec) for spec in scene["static_objects"]],
        "dynamic_objects": [
            dynamic_payload,
            *make_catcher_render_payloads(catcher_spec, catcher_frames, fps),
        ],
        "actions": actions,
        "catcher": catcher_payload,
        "arm": arm_payload,
        "contacts": contacts,
        "caught_frames": caught_flags,
        "captured": captured_flags,
        "missed": missed_flags,
        "truncated": truncated_flags,
        "success": success,
        "outcome": outcome,
    }
    output_json.write_text(json.dumps(result, separators=(",", ":")))


def simulate_arm_gripper_ball_scene(
    scene, output_json, *, fps, frames_per_episode, step_rate, steps_per_frame
):
    model, data = build_arm_gripper_model(scene, step_rate=step_rate)
    dynamic_spec = scene["dynamic_objects"][0]
    catcher_spec = scene["catcher"]
    arm_spec = scene["arm"]
    gripper_spec = scene["gripper"]
    initialize_dynamic_state(model, data, dynamic_spec)
    target_position, joint_targets = initialize_arm_gripper_state(
        model,
        data,
        catcher_spec,
        arm_spec,
        gripper_spec,
    )
    settle_arm_gripper_open_state(
        model,
        data,
        dynamic_spec,
        arm_spec,
        gripper_spec,
        joint_targets,
    )

    gravity = abs(float(scene["metadata"]["gravity"][2]))
    radius = float(dynamic_spec["radius"])
    policy_type = catcher_spec["policy_type"]
    planned_outcome = catcher_spec.get(
        "planned_outcome",
        scene["metadata"].get("planned_outcome", "unplanned"),
    )
    scene["metadata"].get("ball_initial_state", {}).get("launch_plan", {})
    policy_rng = random.Random(int(scene["metadata"]["random_seed"]) + 12848)
    policy_config = arm_gripper_policy_config(
        policy_type,
        policy_rng,
        planned_outcome=planned_outcome,
        miss_bias=catcher_spec.get("miss_bias"),
    )

    dynamic_payload = make_dynamic_payload(dynamic_spec)
    catcher_frames = []
    arm_frames = []
    gripper_frames = []
    end_effector_frames = []
    contacts = []
    grasp_candidate_flags = []
    grasped_flags = []
    captured_flags = []
    missed_flags = []
    truncated_flags = []
    actions = []
    observation_history = []
    zero_delta = [0.0, 0.0, 0.0]
    current_grip_command = gripper_spec.get("open_command", 0.0)
    current_gripper_ctrl = set_gripper_ctrl(model, data, current_grip_command)
    last_action = [gravity, *zero_delta, current_grip_command]
    captured = False
    missed = False
    reached_grasp_region = False
    capture_frame = None
    close_start_frame = None
    miss_frame = None
    terminal_reason = None
    latch_offset_local = None
    latch_world_position = None
    latch_world_jump = None
    latch_substep_index = None
    latch_contact_flags = None
    latch_mouth_metrics = None
    capture_window = 1
    stable_grasp_steps_required = max(2, steps_per_frame // 8)
    stable_grasp_steps = 0

    def locked_ball_position():
        grasp_site = site_id(model, ARM_GRIPPER_GRASP_SITE)
        grasp_position = data.site_xpos[grasp_site].copy()
        grasp_rotation = data.site_xmat[grasp_site].reshape(3, 3)
        return grasp_position + grasp_rotation @ np.asarray(latch_offset_local, dtype=np.float64)

    def lock_ball_to_gripper():
        grasp_velocity = site_linear_velocity(model, data, ARM_GRIPPER_GRASP_SITE)
        set_locked_ball_state(
            model,
            data,
            dynamic_spec,
            locked_ball_position(),
            grasp_velocity,
        )

    def latch_current_ball(
        frame_index, substep_index=None, contact_flags=None, mouth_metrics_at_latch=None
    ):
        nonlocal captured
        nonlocal capture_frame
        nonlocal current_grip_command
        nonlocal current_gripper_ctrl
        nonlocal latch_offset_local
        nonlocal latch_world_position
        nonlocal latch_world_jump
        nonlocal latch_substep_index
        nonlocal latch_contact_flags
        nonlocal latch_mouth_metrics

        ball_position, _, _, _ = body_state(data, dynamic_spec["name"])
        local_offset = site_local_position(model, data, ARM_GRIPPER_GRASP_SITE, ball_position)
        latch_offset_local = [float(value) for value in local_offset]
        latch_world_position = [float(value) for value in ball_position]
        latch_contact_flags = (
            None
            if contact_flags is None
            else {name: bool(value) for name, value in contact_flags.items()}
        )
        latch_mouth_metrics = (
            None if mouth_metrics_at_latch is None else copy.deepcopy(mouth_metrics_at_latch)
        )
        captured = True
        capture_frame = int(frame_index)
        current_grip_command = 1.0
        current_gripper_ctrl = set_gripper_ctrl(model, data, current_grip_command)
        locked_position = locked_ball_position()
        latch_world_jump = float(
            np.linalg.norm(locked_position - np.asarray(ball_position, dtype=np.float64))
        )
        latch_substep_index = None if substep_index is None else int(substep_index)
        lock_ball_to_gripper()

    for frame_idx in range(frames_per_episode):
        if captured:
            lock_ball_to_gripper()
        mujoco.mj_forward(model, data)

        grasp_position, grasp_quaternion = site_pose(model, data, ARM_GRIPPER_GRASP_SITE)
        grasp_velocity = site_linear_velocity(model, data, ARM_GRIPPER_GRASP_SITE)
        ball_position, quaternion, ball_velocity, angular_velocity = body_state(
            data, dynamic_spec["name"]
        )
        quaternion = (0.0, 0.0, 0.0, 1.0)
        angular_velocity = (0.0, 0.0, 0.0)
        flags = arm_gripper_contact_flags(model, data)
        mouth_metrics = arm_gripper_mouth_metrics(
            model,
            data,
            ball_position,
            ball_velocity,
            radius,
        )
        observation_history.append((list(ball_position), list(ball_velocity)))

        grasp_region_depth = max(0.12, 4.5 * radius)
        if float(ball_position[1]) <= float(grasp_position[1]) + grasp_region_depth:
            reached_grasp_region = True
        candidate_grasped = (
            arm_gripper_grasp_candidate(
                model,
                data,
                ball_position,
                ball_velocity,
                flags,
                radius,
                current_grip_command,
                mouth_metrics,
            )
            if not missed
            else False
        )

        if not captured and not missed:
            if candidate_grasped:
                stable_grasp_steps += 1
            else:
                stable_grasp_steps = 0
        if not captured and not missed and stable_grasp_steps >= stable_grasp_steps_required:
            latch_current_ball(frame_idx, contact_flags=flags, mouth_metrics_at_latch=mouth_metrics)
            ball_position, quaternion, ball_velocity, angular_velocity = body_state(
                data, dynamic_spec["name"]
            )
            quaternion = (0.0, 0.0, 0.0, 1.0)
            angular_velocity = (0.0, 0.0, 0.0)
            flags = arm_gripper_contact_flags(model, data)
            mouth_metrics = arm_gripper_mouth_metrics(
                model,
                data,
                ball_position,
                ball_velocity,
                radius,
            )
            candidate_grasped = True

        if not captured and not missed:
            reason = arm_gripper_miss_reason(
                ball_position,
                grasp_position,
                flags,
                radius,
                reached_grasp_region,
            )
            if reason is not None:
                missed = True
                miss_frame = frame_idx
                terminal_reason = reason
                candidate_grasped = False

        joint_positions, joint_velocities = arm_joint_state(model, data, arm_spec)
        gripper_joint_positions, gripper_joint_velocities = gripper_joint_state(
            model, data, gripper_spec
        )
        driver_positions, driver_velocities = driver_joint_state(model, data)
        site_positions = arm_gripper_site_positions(model, data)
        opening_width = gripper_opening_width(model, data)

        dynamic_payload["frames"].append(
            frame_payload(
                frame_idx, fps, ball_position, quaternion, ball_velocity, angular_velocity
            )
        )
        catcher_frames.append(
            {
                "frame": frame_idx + 1,
                "time_seconds": frame_idx / fps,
                "position": [float(value) for value in grasp_position],
                "linear_velocity": [float(value) for value in grasp_velocity],
                "target_position": [float(value) for value in target_position],
            }
        )
        arm_frames.append(
            {
                "frame": frame_idx + 1,
                "time_seconds": frame_idx / fps,
                "joint_position": [float(value) for value in joint_positions],
                "joint_velocity": [float(value) for value in joint_velocities],
                "joint_target": [float(value) for value in joint_targets],
            }
        )
        gripper_frames.append(
            {
                "frame": frame_idx + 1,
                "time_seconds": frame_idx / fps,
                "grip_command": float(current_grip_command),
                "actuator_target": float(current_gripper_ctrl),
                "opening_width": float(opening_width),
                "joint_position": [float(value) for value in gripper_joint_positions],
                "joint_velocity": [float(value) for value in gripper_joint_velocities],
                "driver_joint_position": [float(value) for value in driver_positions],
                "driver_joint_velocity": [float(value) for value in driver_velocities],
                "left_pad_position": site_positions[ARM_GRIPPER_LEFT_PAD_SITE],
                "right_pad_position": site_positions[ARM_GRIPPER_RIGHT_PAD_SITE],
                "left_collision_pad_position": site_positions[ARM_GRIPPER_LEFT_COLLISION_PAD_SITE],
                "right_collision_pad_position": site_positions[
                    ARM_GRIPPER_RIGHT_COLLISION_PAD_SITE
                ],
                "grasp_center_position": site_positions[ARM_GRIPPER_GRASP_SITE],
                "ball_local_position": mouth_metrics["ball_local_position"],
                "ball_local_velocity": mouth_metrics["ball_local_velocity"],
                "pad_mid_local_position": mouth_metrics["pad_mid_local_position"],
                "palm_local_position": mouth_metrics["palm_local_position"],
                "mouth_lower_z": mouth_metrics["mouth_lower_z"],
                "mouth_upper_z": mouth_metrics["mouth_upper_z"],
                "mouth_lateral_x_bound": mouth_metrics["mouth_lateral_x_bound"],
                "preclose_lateral_x_bound": mouth_metrics["preclose_lateral_x_bound"],
                "mouth_lateral_y_bound": mouth_metrics["mouth_lateral_y_bound"],
                "preclose_lateral_y_bound": mouth_metrics["preclose_lateral_y_bound"],
                "inside_gripper_mouth": bool(mouth_metrics["inside_gripper_mouth"]),
                "preclose_ready": bool(mouth_metrics["preclose_ready"]),
            }
        )
        end_effector_frames.append(
            {
                "frame": frame_idx + 1,
                "time_seconds": frame_idx / fps,
                "position": [float(value) for value in grasp_position],
                "quaternion_xyzw": [float(value) for value in grasp_quaternion],
                "linear_velocity": [float(value) for value in grasp_velocity],
                "target_position": [float(value) for value in target_position],
            }
        )
        contacts.append({name: bool(value) for name, value in flags.items()})
        grasp_candidate_flags.append(bool(candidate_grasped))
        grasped_flags.append(bool(captured))
        captured_flags.append(bool(captured))
        missed_flags.append(bool(missed))
        truncated_flags.append(False)

        if frame_idx < frames_per_episode - 1:
            if captured:
                physical_delta = [
                    0.0,
                    -0.006,
                    0.010,
                ]
                physical_delta = [
                    clip_value(
                        physical_delta[index],
                        -catcher_spec["action_scale_xyz"][index],
                        catcher_spec["action_scale_xyz"][index],
                    )
                    for index in range(3)
                ]
                normalized_delta = [
                    clip_value(
                        physical_delta[index] / catcher_spec["action_scale_xyz"][index],
                        -1.0,
                        1.0,
                    )
                    for index in range(3)
                ]
                next_grip_command = 1.0
            elif missed:
                normalized_delta = list(zero_delta)
                physical_delta = list(zero_delta)
                next_grip_command = 1.0 if float(current_grip_command) >= 0.5 else 0.0
            else:
                grasp_rotation = (
                    data.site_xmat[site_id(model, ARM_GRIPPER_GRASP_SITE)].reshape(3, 3).copy()
                )
                normalized_delta, physical_delta = compute_arm_gripper_action(
                    catcher_spec=catcher_spec,
                    policy_type=policy_type,
                    policy_config=policy_config,
                    rng=policy_rng,
                    target_position=target_position,
                    observation_history=observation_history,
                    gravity=gravity,
                    radius=radius,
                    grasp_rotation=grasp_rotation,
                )
                next_grip_command = arm_gripper_grip_command(
                    current_grip_command,
                    mouth_metrics,
                    captured=captured,
                    missed=missed,
                )
                if (
                    close_start_frame is None
                    and float(current_grip_command) < 0.5
                    and float(next_grip_command) >= 0.5
                ):
                    close_start_frame = frame_idx + 1

            target_position = clip_vector(
                [target_position[index] + physical_delta[index] for index in range(3)],
                catcher_spec["workspace_min"],
                catcher_spec["workspace_max"],
            )
            current_joint_positions, _ = arm_joint_state(model, data, arm_spec)
            saved_qpos = data.qpos.copy()
            saved_qvel = data.qvel.copy()
            joint_targets = solve_site_position_ik(
                model,
                data,
                arm_spec,
                ARM_GRIPPER_GRASP_SITE,
                target_position,
                current_joint_positions,
            )
            data.qpos[:] = saved_qpos
            data.qvel[:] = saved_qvel
            mujoco.mj_forward(model, data)
            action = [gravity, *normalized_delta, next_grip_command]
            actions.append([float(value) for value in action])
            last_action = action
            current_grip_command = next_grip_command
            current_gripper_ctrl = set_gripper_ctrl(model, data, current_grip_command)
            for substep_index in range(steps_per_frame):
                set_named_arm_ctrl(model, data, arm_spec, joint_targets)
                set_gripper_ctrl(model, data, current_grip_command)
                mujoco.mj_step(model, data)
                if captured:
                    lock_ball_to_gripper()
                elif not missed:
                    sub_ball_position, _, sub_ball_velocity, _ = body_state(
                        data, dynamic_spec["name"]
                    )
                    sub_flags = arm_gripper_contact_flags(model, data)
                    sub_mouth_metrics = arm_gripper_mouth_metrics(
                        model,
                        data,
                        sub_ball_position,
                        sub_ball_velocity,
                        radius,
                    )
                    if arm_gripper_grasp_candidate(
                        model,
                        data,
                        sub_ball_position,
                        sub_ball_velocity,
                        sub_flags,
                        radius,
                        current_grip_command,
                        sub_mouth_metrics,
                    ):
                        stable_grasp_steps += 1
                    else:
                        stable_grasp_steps = 0
                    if stable_grasp_steps >= stable_grasp_steps_required:
                        latch_current_ball(
                            frame_idx + 1,
                            substep_index,
                            contact_flags=sub_flags,
                            mouth_metrics_at_latch=sub_mouth_metrics,
                        )
        else:
            actions.append([float(value) for value in last_action])

    success = any(captured_flags)
    truncated_before_interaction = False
    outcome = "capture" if success else None
    if success:
        terminal_reason = terminal_reason or "grasped"
    elif missed:
        outcome = "miss"
        if miss_frame is None:
            miss_frame = frames_per_episode - 1
        terminal_reason = terminal_reason or "missed"
    elif not reached_grasp_region:
        outcome = "truncated"
        truncated_before_interaction = True
        terminal_reason = "max_length_before_interaction"
    else:
        outcome = "miss"
        missed = True
        miss_frame = frames_per_episode - 1
        missed_flags[-1] = True
        terminal_reason = "max_length_after_interaction"

    if missed and miss_frame is not None:
        missed_flags = [frame_idx >= miss_frame for frame_idx in range(frames_per_episode)]
    truncated_flags = [truncated_before_interaction for _ in range(frames_per_episode)]

    catcher_payload = {
        "name": catcher_spec["name"],
        "frames": catcher_frames,
        "initial_position": catcher_spec["initial_position"],
        "initial_target_position": catcher_spec["initial_target_position"],
        "workspace_min": catcher_spec["workspace_min"],
        "workspace_max": catcher_spec["workspace_max"],
        "target_y": catcher_spec["target_y"],
        "fixed_y": catcher_spec["fixed_y"],
        "control_axes": ["x", "y", "z"],
        "action_scale_xyz": catcher_spec["action_scale_xyz"],
        "policy_type": policy_type,
        "policy_config": policy_config,
        "planned_outcome": planned_outcome,
        "outcome": outcome,
        "capture_window_frames": capture_window,
        "capture_frame": capture_frame,
        "close_start_frame": close_start_frame,
        "miss_frame": miss_frame,
        "truncated_before_interaction": truncated_before_interaction,
        "terminal_reason": terminal_reason,
        "latch_offset_local": latch_offset_local,
        "latch_world_position": latch_world_position,
        "latch_world_jump": latch_world_jump,
        "latch_substep_index": latch_substep_index,
        "latch_contact_flags": latch_contact_flags,
        "latch_mouth_metrics": latch_mouth_metrics,
        "mouth_test": "gripper_local_pad_palm_bounds",
        "stable_grasp_steps_required": stable_grasp_steps_required,
    }
    arm_payload = {
        "name": arm_spec["name"],
        "robot_model": arm_spec.get("robot_model"),
        "frames": arm_frames,
        "base_position": arm_spec["base_position"],
        "base_euler": arm_spec.get("base_euler", [0.0, 0.0, 0.0]),
        "joint_names": arm_spec["joint_names"],
        "joint_limits": arm_spec["joint_limits"],
        "actuator_names": arm_spec["actuator_names"],
        "home_qpos": arm_spec.get("home_qpos"),
        "model_scale": arm_spec.get("model_scale", 1.0),
        "render_model_scale": arm_spec.get("render_model_scale", arm_spec.get("model_scale", 1.0)),
        "ik_controller": "damped_least_squares_site_position",
        "ik_site": ARM_GRIPPER_GRASP_SITE,
    }
    gripper_payload = {
        "name": gripper_spec["name"],
        "robot_model": gripper_spec.get("robot_model"),
        "frames": gripper_frames,
        "joint_names": gripper_spec["joint_names"],
        "driver_joint_names": list(ROBOTIQ_DRIVER_JOINT_NAMES),
        "actuator_name": ROBOTIQ_ACTUATOR_NAME,
        "open_command": gripper_spec.get("open_command", 0.0),
        "close_command": gripper_spec.get("close_command", 1.0),
        "open_actuator_target": 0.0,
        "closed_actuator_target": 255.0,
        "site_names": {
            "tool": ARM_GRIPPER_TOOL_SITE,
            "base": ARM_GRIPPER_BASE_SITE,
            "center": ARM_GRIPPER_CENTER_SITE,
            "left_pad": ARM_GRIPPER_LEFT_PAD_SITE,
            "right_pad": ARM_GRIPPER_RIGHT_PAD_SITE,
            "left_collision_pad": ARM_GRIPPER_LEFT_COLLISION_PAD_SITE,
            "right_collision_pad": ARM_GRIPPER_RIGHT_COLLISION_PAD_SITE,
            "grasp_center": ARM_GRIPPER_GRASP_SITE,
            "palm": ARM_GRIPPER_PALM_SITE,
        },
    }
    metadata = dict(scene["metadata"])
    metadata.update(
        {
            "policy_type": policy_type,
            "policy_config": policy_config,
            "planned_outcome": planned_outcome,
            "outcome": outcome,
            "success": success,
            "capture_window_frames": capture_window,
            "capture_window_sim_steps": stable_grasp_steps_required,
            "capture_frame": capture_frame,
            "close_start_frame": close_start_frame,
            "miss_frame": miss_frame,
            "truncated_before_interaction": truncated_before_interaction,
            "terminal_reason": terminal_reason,
            "latch_after_capture": True,
            "latch_world_position": latch_world_position,
            "latch_world_jump": latch_world_jump,
            "latch_substep_index": latch_substep_index,
            "latch_contact_flags": latch_contact_flags,
            "latch_mouth_metrics": latch_mouth_metrics,
            "action_dim": len(metadata["action_schema"]),
            "grasped": success,
            "gripper_type": "robotiq_2f85",
            "arm_robot_model": "universal_robots_ur5e",
            "replay_contract": "render restores saved UR5e, Robotiq, end-effector, and ball state; no IK or latch recomputation",
        }
    )
    result = {
        "metadata": metadata,
        "camera": scene["camera"],
        "static_objects": [serialize_object(spec) for spec in scene["static_objects"]],
        "dynamic_objects": [dynamic_payload],
        "actions": actions,
        "catcher": catcher_payload,
        "arm": arm_payload,
        "gripper": gripper_payload,
        "end_effector": {
            "name": ARM_GRIPPER_GRASP_SITE,
            "frames": end_effector_frames,
            "target_site": ARM_GRIPPER_GRASP_SITE,
        },
        "contacts": contacts,
        "caught_frames": grasp_candidate_flags,
        "grasped": grasped_flags,
        "captured": captured_flags,
        "missed": missed_flags,
        "truncated": truncated_flags,
        "success": success,
        "outcome": outcome,
    }
    output_json.write_text(json.dumps(result, separators=(",", ":")))


def simulate_scene(scene, output_json, *, fps, frames_per_episode, step_rate, steps_per_frame):
    if scene["metadata"].get("scenario") in {"paddle_ball", "paddle_ball_zonly"}:
        simulate_paddle_ball_scene(
            scene,
            output_json,
            fps=fps,
            frames_per_episode=frames_per_episode,
            step_rate=step_rate,
            steps_per_frame=steps_per_frame,
        )
        return
    if scene["metadata"].get("scenario") == "arm_paddle_ball":
        simulate_arm_paddle_ball_scene(
            scene,
            output_json,
            fps=fps,
            frames_per_episode=frames_per_episode,
            step_rate=step_rate,
            steps_per_frame=steps_per_frame,
        )
        return
    if scene["metadata"].get("scenario") == "catcher_ball":
        simulate_catcher_ball_scene(
            scene,
            output_json,
            fps=fps,
            frames_per_episode=frames_per_episode,
            step_rate=step_rate,
            steps_per_frame=steps_per_frame,
        )
        return
    if scene["metadata"].get("scenario") == "arm_catcher_ball":
        simulate_arm_catcher_ball_scene(
            scene,
            output_json,
            fps=fps,
            frames_per_episode=frames_per_episode,
            step_rate=step_rate,
            steps_per_frame=steps_per_frame,
        )
        return
    if scene["metadata"].get("scenario") == "arm_gripper_ball":
        simulate_arm_gripper_ball_scene(
            scene,
            output_json,
            fps=fps,
            frames_per_episode=frames_per_episode,
            step_rate=step_rate,
            steps_per_frame=steps_per_frame,
        )
        return

    model, data = build_model(scene, step_rate=step_rate)
    dynamic_spec = scene["dynamic_objects"][0]
    initialize_dynamic_state(model, data, dynamic_spec)
    dynamic_payload = make_dynamic_payload(dynamic_spec)

    for frame_idx in range(frames_per_episode):
        if frame_idx > 0:
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
        position, quaternion, linear_velocity, angular_velocity = body_state(
            data, dynamic_spec["name"]
        )
        if dynamic_spec["kind"] == "ball":
            quaternion = (0.0, 0.0, 0.0, 1.0)
            angular_velocity = (0.0, 0.0, 0.0)
        dynamic_payload["frames"].append(
            frame_payload(frame_idx, fps, position, quaternion, linear_velocity, angular_velocity)
        )

    result = {
        "metadata": scene["metadata"],
        "camera": scene["camera"],
        "static_objects": [serialize_object(spec) for spec in scene["static_objects"]],
        "dynamic_objects": [dynamic_payload],
    }
    output_json.write_text(json.dumps(result, separators=(",", ":")))
