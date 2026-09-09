import json
import math
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import sim_support as demo_generate
from common import (
    BASE_SEED,
    CANONICAL_EPISODES_PER_TASK,
    DURATION_SECONDS,
    DYNAMIC_SHAPE_MODE,
    ELASTIC_RESTITUTION,
    EPISODES_PER_TASK,
    FPS,
    FRAMES_PER_EPISODE,
    GRAVITY_BATCH_EPISODES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RENDER_START_METHOD,
    SELECTED_SPLIT,
    SHAPE_ID_MAP,
    SIM_WORKERS,
    SPLIT_TO_ID,
    STATE_SCHEMA,
    STEP_RATE,
    STEPS_PER_FRAME,
    TASK_PARAM_MEANINGS,
    TASKS,
    TEST_EPISODES_PER_BATCH,
    TEST_GRAVITIES,
    TRAIN_EPISODES_PER_BATCH,
    TRAIN_G_MIN,
    TRAIN_GAUSSIAN_MEAN,
    TRAIN_GAUSSIAN_STD,
    ensure_clean_dir,
    ensure_dir,
    episode_dir,
    task_action_schema,
    task_has_catcher_fields,
    task_has_paddle_fields,
    task_output_dir,
)

_FIXED_CUBE_HALF_EXTENT_OVERRIDE = os.environ.get(
    "DEMO_MUJOCO_FIXED_CUBE_HALF_EXTENT",
    os.environ.get("DEMO_MUJOCO_FIXED_BOX_HALF_EXTENT", ""),
).strip()
if _FIXED_CUBE_HALF_EXTENT_OVERRIDE:
    FIXED_CUBE_HALF_EXTENT = float(_FIXED_CUBE_HALF_EXTENT_OVERRIDE)
    if FIXED_CUBE_HALF_EXTENT <= 0.0:
        raise ValueError(
            f"DEMO_MUJOCO_FIXED_CUBE_HALF_EXTENT must be positive; got {FIXED_CUBE_HALF_EXTENT!r}"
        )
else:
    FIXED_CUBE_HALF_EXTENT = None

_FIXED_BALL_RADIUS_OVERRIDE = os.environ.get(
    "DEMO_MUJOCO_FIXED_BALL_RADIUS",
    os.environ.get("DEMO_MUJOCO_FIXED_CIRCLE_RADIUS", ""),
).strip()
if _FIXED_BALL_RADIUS_OVERRIDE:
    FIXED_BALL_RADIUS = float(_FIXED_BALL_RADIUS_OVERRIDE)
    if FIXED_BALL_RADIUS <= 0.0:
        raise ValueError(
            f"DEMO_MUJOCO_FIXED_BALL_RADIUS must be positive; got {FIXED_BALL_RADIUS!r}"
        )
else:
    FIXED_BALL_RADIUS = None

ARM_GRIPPER_DEFAULT_BALL_SCALE = 2.5
ARM_GRIPPER_DEFAULT_MODEL_SCALE = 5.2
ARM_GRIPPER_DEFAULT_RENDER_MODEL_SCALE = 5.2
_ARM_GRIPPER_DEBUG_SCALE_OVERRIDE = os.environ.get(
    "DEMO_MUJOCO_ARM_GRIPPER_DEBUG_SCALE", ""
).strip()
ARM_GRIPPER_DEBUG_SCALE = (
    float(_ARM_GRIPPER_DEBUG_SCALE_OVERRIDE) if _ARM_GRIPPER_DEBUG_SCALE_OVERRIDE else 1.0
)
_ARM_GRIPPER_BALL_SCALE_DEFAULT = (
    ARM_GRIPPER_DEBUG_SCALE if _ARM_GRIPPER_DEBUG_SCALE_OVERRIDE else ARM_GRIPPER_DEFAULT_BALL_SCALE
)
_ARM_GRIPPER_MODEL_SCALE_DEFAULT = (
    ARM_GRIPPER_DEBUG_SCALE
    if _ARM_GRIPPER_DEBUG_SCALE_OVERRIDE
    else ARM_GRIPPER_DEFAULT_MODEL_SCALE
)
_ARM_GRIPPER_RENDER_MODEL_SCALE_DEFAULT = (
    ARM_GRIPPER_DEBUG_SCALE
    if _ARM_GRIPPER_DEBUG_SCALE_OVERRIDE
    else ARM_GRIPPER_DEFAULT_RENDER_MODEL_SCALE
)
ARM_GRIPPER_BALL_SCALE = float(
    os.environ.get("DEMO_MUJOCO_ARM_GRIPPER_BALL_SCALE", str(_ARM_GRIPPER_BALL_SCALE_DEFAULT))
)
ARM_GRIPPER_MODEL_SCALE = float(
    os.environ.get("DEMO_MUJOCO_ARM_GRIPPER_MODEL_SCALE", str(_ARM_GRIPPER_MODEL_SCALE_DEFAULT))
)
ARM_GRIPPER_RENDER_MODEL_SCALE = float(
    os.environ.get(
        "DEMO_MUJOCO_ARM_GRIPPER_RENDER_MODEL_SCALE",
        str(_ARM_GRIPPER_RENDER_MODEL_SCALE_DEFAULT),
    )
)
ARM_GRIPPER_CAPTURE_RETRY_LIMIT = int(
    os.environ.get("DEMO_MUJOCO_ARM_GRIPPER_CAPTURE_RETRY_LIMIT", "240")
)
ARM_CATCHER_CAPTURE_RETRY_LIMIT = int(
    os.environ.get("DEMO_MUJOCO_ARM_CATCHER_CAPTURE_RETRY_LIMIT", "96")
)
ARM_GRIPPER_CAPTURE_MAX_CLOSE_LEAD_FRAMES = int(
    os.environ.get("DEMO_MUJOCO_ARM_GRIPPER_CAPTURE_MAX_CLOSE_LEAD_FRAMES", "4")
)
for _scale_name, _scale_value in (
    ("DEMO_MUJOCO_ARM_GRIPPER_DEBUG_SCALE", ARM_GRIPPER_DEBUG_SCALE),
    ("DEMO_MUJOCO_ARM_GRIPPER_BALL_SCALE", ARM_GRIPPER_BALL_SCALE),
    ("DEMO_MUJOCO_ARM_GRIPPER_MODEL_SCALE", ARM_GRIPPER_MODEL_SCALE),
    ("DEMO_MUJOCO_ARM_GRIPPER_RENDER_MODEL_SCALE", ARM_GRIPPER_RENDER_MODEL_SCALE),
):
    if _scale_value <= 0.0:
        raise ValueError(f"{_scale_name} must be positive; got {_scale_value!r}")
if ARM_GRIPPER_CAPTURE_RETRY_LIMIT <= 0:
    raise ValueError(
        "DEMO_MUJOCO_ARM_GRIPPER_CAPTURE_RETRY_LIMIT must be positive; "
        f"got {ARM_GRIPPER_CAPTURE_RETRY_LIMIT!r}"
    )
if ARM_CATCHER_CAPTURE_RETRY_LIMIT <= 0:
    raise ValueError(
        "DEMO_MUJOCO_ARM_CATCHER_CAPTURE_RETRY_LIMIT must be positive; "
        f"got {ARM_CATCHER_CAPTURE_RETRY_LIMIT!r}"
    )
if ARM_GRIPPER_CAPTURE_MAX_CLOSE_LEAD_FRAMES < 0:
    raise ValueError(
        "DEMO_MUJOCO_ARM_GRIPPER_CAPTURE_MAX_CLOSE_LEAD_FRAMES must be nonnegative; "
        f"got {ARM_GRIPPER_CAPTURE_MAX_CLOSE_LEAD_FRAMES!r}"
    )

_DYNAMIC_OBJECT_RGBA_OVERRIDE = os.environ.get("DEMO_MUJOCO_DYNAMIC_OBJECT_RGBA", "").strip()
if _DYNAMIC_OBJECT_RGBA_OVERRIDE:
    _color_components = [
        float(value.strip()) for value in _DYNAMIC_OBJECT_RGBA_OVERRIDE.split(",") if value.strip()
    ]
    if len(_color_components) == 3:
        _color_components.append(1.0)
    if len(_color_components) != 4:
        raise ValueError(
            "DEMO_MUJOCO_DYNAMIC_OBJECT_RGBA must have 3 or 4 comma-separated values; "
            f"got {_DYNAMIC_OBJECT_RGBA_OVERRIDE!r}"
        )
    DYNAMIC_OBJECT_COLOR = tuple(_color_components)
else:
    DYNAMIC_OBJECT_COLOR = (0.95, 0.08, 0.04, 1.0)


def make_camera(location, target, fovy_degrees):
    camera = demo_generate.make_camera(location, target, fovy_degrees)
    camera["resolution"] = [IMAGE_WIDTH, IMAGE_HEIGHT]
    return camera


def projected_screen_x(position, camera):
    location = camera["location"]
    right = camera["image_right_world"]
    forward = camera["optical_axis_world"]
    relative = [float(position[index]) - float(location[index]) for index in range(3)]
    depth = sum(relative[index] * float(forward[index]) for index in range(3))
    if depth <= 1.0e-9:
        return float("inf")
    tangent = math.tan(math.radians(float(camera["fovy"])) * 0.5)
    return sum(relative[index] * float(right[index]) for index in range(3)) / (depth * tangent)


def widen_range(values, scale, min_value, max_value):
    center = 0.5 * (values[0] + values[1])
    half_width = 0.5 * (values[1] - values[0]) * scale
    return (
        max(min_value, center - half_width),
        min(max_value, center + half_width),
    )


def scale_linear(value, scale):
    return float(value) * scale


def scale_vector(values, scale):
    return tuple(scale_linear(value, scale) for value in values)


def scale_range(values, scale):
    return tuple(scale_linear(value, scale) for value in values)


def flattenable_grid_depth(half_thickness, depth_scale):
    return max(1.0e-6, float(half_thickness) * float(depth_scale))


def offset_range(values, offset):
    return tuple(float(value) + float(offset) for value in values)


def deterministic_fraction(seed):
    return random.Random(int(seed)).random()


def float_env(name, default):
    return float(os.environ.get(name, str(default)).strip())


def positive_float_env(name, default):
    value = float_env(name, default)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive; got {value!r}")
    return value


def nonnegative_float_env(name, default):
    value = float_env(name, default)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative; got {value!r}")
    return value


def fraction_env(name, default):
    value = float_env(name, default)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]; got {value!r}")
    return value


BALL_RADIUS_RANGE = (0.66, 0.90)
CUBE_HALF_EXTENT_RANGE = (0.58, 0.78)
OPEN_BOX_CENTER = (0.0, 0.0, 4.6)
OPEN_BOX_INNER_HALF_EXTENTS = (3.5, 3.5, 3.5)
OPEN_BOX_WALL_THICKNESS = 0.40
OPEN_BOX_OBJECT_EDGE_MARGIN = 0.25
OPEN_BOX_CAMERA_LOCATION = (0.0, -11.0, 16.65)
OPEN_BOX_CAMERA_TARGET = (0.0, 1.1, 4.55)
OPEN_BOX_CAMERA_FOVY_DEGREES = 32.0
OPEN_BOX_GRID_SPACING = 0.875
OPEN_BOX_GRID_HALF_THICKNESS = 0.012
OPEN_BOX_WALL_GRID_HALF_THICKNESS = 0.05
OPEN_BOX_GRID_SURFACE_OFFSET = 0.024
OPEN_BOX_INITIAL_X_SPEED_RANGE = (0.45, 1.10)
OPEN_BOX_INITIAL_Y_SPEED_INWARD_RANGE = (0.65, 1.15)
OPEN_BOX_INITIAL_Y_SPEED_CAMERA_RANGE = (0.25, 0.55)
OPEN_BOX_INITIAL_VERTICAL_SPEED_RANGE = (-0.25, 0.70)
APPROACH_BALL_GROUND_CENTER = (0.0, 0.0, -0.05)
APPROACH_BALL_GROUND_HALF_EXTENTS = (7.0, 9.0, 0.05)
APPROACH_BALL_GROUND_THICKNESS = 0.10
APPROACH_BALL_SIDE_WALL_THICKNESS = 0.25
APPROACH_BALL_SIDE_WALL_HEIGHT = 3.50
APPROACH_BALL_CAMERA_LOCATION = (0.0, -10.0, 7.10)
APPROACH_BALL_CAMERA_TARGET = (0.0, -1.0, 0.80)
APPROACH_BALL_CAMERA_FOVY_DEGREES = 54.0
APPROACH_BALL_CAMERA_ELEVATION_DEGREES = 35.0
APPROACH_BALL_GRID_SPACING = 1.0
APPROACH_BALL_GRID_HALF_THICKNESS = 0.015
APPROACH_BALL_GRID_SURFACE_OFFSET = 0.018
APPROACH_BALL_RADIUS_RANGE = (0.32, 0.45)
APPROACH_BALL_OBJECT_COLOR = (0.95, 0.08, 0.04, 1.0)
APPROACH_BALL_START_X_RANGE = (-1.2, 1.2)
APPROACH_BALL_START_Y_RANGE = (7.2, 8.7)
APPROACH_BALL_HEIGHT_ABOVE_RADIUS_RANGE = (1.6, 2.8)
APPROACH_BALL_NEAR_TARGET_X_RANGE = (-2.2, 2.2)
APPROACH_BALL_NEAR_TARGET_Y_RANGE = (-3.8, -3.0)
APPROACH_BALL_NEAR_TARGET_TIME_RANGE = (3.75, 4.20)
APPROACH_BALL_FIRST_BOUNCE_TIME_RANGE = (1.0, 1.4)
# approach_ball at real physical scale (2026-07): the constants above remain
# legacy reference values (arm_catcher/arm_paddle/catcher_ball still derive
# their scenes from them); the approach_ball task itself now runs at the same
# 1/7.5 physical rescale as arm_catcher_ball, with bounce timing re-tuned for
# the real-gravity sweep (train N(9.8, 2.0), test 0..20 plus planets).
APPROACH_BALL_REFERENCE_SCALE = 7.5
APPROACH_BALL_SCENE_SCALE = 1.0 / APPROACH_BALL_REFERENCE_SCALE
APPROACH_BALL_REAL_GROUND_CENTER = scale_vector(
    APPROACH_BALL_GROUND_CENTER, APPROACH_BALL_SCENE_SCALE
)
APPROACH_BALL_REAL_GROUND_HALF_EXTENTS = scale_vector(
    APPROACH_BALL_GROUND_HALF_EXTENTS, APPROACH_BALL_SCENE_SCALE
)
APPROACH_BALL_REAL_SIDE_WALL_THICKNESS = scale_linear(
    APPROACH_BALL_SIDE_WALL_THICKNESS, APPROACH_BALL_SCENE_SCALE
)
APPROACH_BALL_REAL_SIDE_WALL_HEIGHT = scale_linear(
    APPROACH_BALL_SIDE_WALL_HEIGHT, APPROACH_BALL_SCENE_SCALE
)
# Closer, lower camera than the pure 1/7.5 rescale of the legacy pose
# ((0, -1.333, 0.947) at 35 deg elevation), chosen against the trajectory
# envelope (spawn apex at NDC y <= 0.9, nearest floor bounce inside the
# bottom edge) so the ball reads larger without clipping. First candidate
# ((0, -0.92, 0.61) at 30 deg) made the final near-camera ball too large;
# this pose sits halfway back toward the rescaled legacy pose: ball ~1.3x
# the rescaled-legacy size near the camera, ~1.1x at spawn.
APPROACH_BALL_REAL_CAMERA_LOCATION = (0.0, -1.03, 0.78)
APPROACH_BALL_REAL_CAMERA_TARGET = (0.0, 0.03, 0.105)
APPROACH_BALL_REAL_CAMERA_ELEVATION_DEGREES = 32.5
APPROACH_BALL_REAL_GRID_SPACING = scale_linear(
    APPROACH_BALL_GRID_SPACING, APPROACH_BALL_SCENE_SCALE
)
APPROACH_BALL_REAL_GRID_HALF_THICKNESS = scale_linear(
    APPROACH_BALL_GRID_HALF_THICKNESS, APPROACH_BALL_SCENE_SCALE
)
APPROACH_BALL_REAL_GRID_SURFACE_OFFSET = scale_linear(
    APPROACH_BALL_GRID_SURFACE_OFFSET, APPROACH_BALL_SCENE_SCALE
)
APPROACH_BALL_GRID_DEPTH_SCALE = nonnegative_float_env(
    "DEMO_MUJOCO_APPROACH_BALL_GRID_DEPTH_SCALE",
    0.0,
)
APPROACH_BALL_REAL_GRID_DEPTH_HALF_THICKNESS = flattenable_grid_depth(
    APPROACH_BALL_REAL_GRID_HALF_THICKNESS,
    APPROACH_BALL_GRID_DEPTH_SCALE,
)
# Fixed 6 cm ball (top of the legacy sampled range scaled by 1/7.5, close to
# the arm_catcher_ball fixed radius of 0.50 * scale ~= 0.067 m).
APPROACH_BALL_REAL_BALL_RADIUS = 0.06
APPROACH_BALL_REAL_START_X_RANGE = scale_range(
    APPROACH_BALL_START_X_RANGE, APPROACH_BALL_SCENE_SCALE
)
APPROACH_BALL_REAL_START_Y_RANGE = scale_range(
    APPROACH_BALL_START_Y_RANGE, APPROACH_BALL_SCENE_SCALE
)
# Minimum gap between the spawned ball surface and the back wall's inner
# face (at ground_half_y). The raw start-y range reaches half_y - 0.04, so a
# ball radius above 4 cm can spawn embedded in the wall, and the stiff
# elastic wall contact then ejects it toward the camera at several m/s.
APPROACH_BALL_REAL_BACK_WALL_CLEARANCE = 0.02
APPROACH_BALL_REAL_HEIGHT_ABOVE_RADIUS_RANGE = scale_range(
    APPROACH_BALL_HEIGHT_ABOVE_RADIUS_RANGE, APPROACH_BALL_SCENE_SCALE
)
APPROACH_BALL_REAL_NEAR_TARGET_X_RANGE = scale_range(
    APPROACH_BALL_NEAR_TARGET_X_RANGE, APPROACH_BALL_SCENE_SCALE
)
APPROACH_BALL_REAL_NEAR_TARGET_Y_RANGE = scale_range(
    APPROACH_BALL_NEAR_TARGET_Y_RANGE, APPROACH_BALL_SCENE_SCALE
)
APPROACH_BALL_REAL_VX_NOISE_RANGE = scale_range((-0.08, 0.08), APPROACH_BALL_SCENE_SCALE)
APPROACH_BALL_REAL_VY_NOISE_RANGE = scale_range((-0.12, 0.12), APPROACH_BALL_SCENE_SCALE)
# First-bounce time at the reference gravity (9.8): slightly above the
# free-fall time from the sampled drop heights (~0.24 s), the same small
# initial toss the legacy range gave at g~4. Scaled by sqrt(9.8/g) at
# sampling time (as in arm_catcher) so post-bounce arc heights (~0.5*g*t^2)
# stay comparable across the 0..20 gravity sweep.
APPROACH_BALL_REAL_FIRST_BOUNCE_TIME_RANGE = (0.26, 0.38)
# Hard ceiling on the ball-center height of any pre- or post-bounce apex,
# just below the side-wall top (0.4667): with restitution 1.0 an apex at
# this height at the back of the arena projects to NDC y ~0.91 under the
# current camera, so the whole trajectory stays in frame at every gravity.
APPROACH_BALL_REAL_MAX_BOUNCE_APEX_Z = 0.46
# Real-scale contacts: the default elastic solref (stiffness 1000) lets the
# ~5 cm ball tunnel through the 1.3 cm floor above ~2 m/s impact speed --
# high-gravity bounces reach ~4.5 m/s -- so the ground/wall pair contacts
# reuse the stiffness that fixed the same failure in arm_paddle_ball.
APPROACH_BALL_CONTACT_STIFFNESS = 6.0e4
APPROACH_BALL_GROUND_CONTACT_SOLREF = demo_generate.elastic_contact_solref(
    ELASTIC_RESTITUTION, APPROACH_BALL_CONTACT_STIFFNESS
)


def approach_ball_first_bounce_time_range(gravity):
    factor = math.sqrt(9.8 / max(float(gravity), 2.0))
    low = min(max(APPROACH_BALL_REAL_FIRST_BOUNCE_TIME_RANGE[0] * factor, 0.20), 1.00)
    high = min(max(APPROACH_BALL_REAL_FIRST_BOUNCE_TIME_RANGE[1] * factor, low + 0.05), 1.10)
    return low, high


PADDLE_BALL_RADIUS = 0.20
PADDLE_BALL_MASS = 0.10
PADDLE_BALL_RESTITUTION = 0.92
PADDLE_BALL_FRICTION = 0.05
PADDLE_BALL_ZONLY_FRICTION = 0.0
PADDLE_RESTITUTION = 0.92
PADDLE_FRICTION = 0.25
PADDLE_ZONLY_FRICTION = 0.0
PADDLE_GROUND_HALF_EXTENTS = (2.6, 4.1, 0.05)
PADDLE_WORKSPACE_MIN = (-1.15, -3.35, 0.35)
PADDLE_WORKSPACE_MAX = (1.15, -2.65, 0.92)
PADDLE_ACTION_SCALE_XYZ = (0.22, 0.18, 0.16)
PADDLE_MAX_TILT_THETA = math.radians(30.0)
PADDLE_POLICY_MAX_TILT_THETA = math.radians(24.0)
PADDLE_BASE_HALF_EXTENTS = (0.3375, 0.3375, 0.01575)
PADDLE_RIM_HEIGHT = 0.0
PADDLE_RIM_HALF_THICKNESS = 0.0
PADDLE_INITIAL_POSITION = (0.0, -3.0, 0.48)
PADDLE_BALL_INITIAL_X_RANGE = (-0.22, 0.22)
PADDLE_BALL_INITIAL_Y_RANGE = (-3.08, -2.92)
PADDLE_BALL_INITIAL_Z_RANGE = (1.10, 1.18)
PADDLE_BALL_INITIAL_VX_RANGE = (-0.18, 0.18)
PADDLE_BALL_INITIAL_VY_RANGE = (-0.16, 0.16)
PADDLE_BALL_INITIAL_VZ_RANGE = (-0.42, -0.12)
PADDLE_BALL_ZONLY_INITIAL_VX_RANGE = (0.0, 0.0)
PADDLE_BALL_ZONLY_INITIAL_VY_RANGE = (0.0, 0.0)
PADDLE_BALL_ZONLY_INITIAL_VZ_RANGE = (0.0, 0.0)
PADDLE_BALL_INITIAL_WX_RANGE = (0.0, 0.0)
PADDLE_BALL_INITIAL_WY_RANGE = (0.0, 0.0)
PADDLE_BALL_INITIAL_WZ_RANGE = (0.0, 0.0)
PADDLE_BALL_MATERIAL = demo_generate.checker_material(
    (0.95, 0.05, 0.04, 1.0),
    (1.0, 1.0, 1.0, 1.0),
    base_color=APPROACH_BALL_OBJECT_COLOR,
    roughness=0.82,
    specular=0.04,
    emission=0.08,
    longitude_segments=4,
    latitude_segments=2,
)
PADDLE_MATERIAL = demo_generate.material_from_hex("#86C8E7", roughness=0.76, specular=0.10)
PADDLE_BALL_CAMERA_LOCATION = (0.0, -5.65, 3.25)
PADDLE_BALL_CAMERA_TARGET = (0.0, -3.0, 0.78)
PADDLE_BALL_CAMERA_FOVY_DEGREES = 42.0
ARM_PADDLE_ROBOT_MODEL = "unitree_z1"
ARM_PADDLE_REFERENCE_SCALE = 5.0
ARM_PADDLE_SCENE_SCALE = 1.0 / ARM_PADDLE_REFERENCE_SCALE
# Vertical bounce heights and speeds rescale by the gravity ratio (real 9.8
# vs legacy 4.0) rather than the geometric scale, so a bounce still spans
# roughly the same number of frames per period as the legacy tuning.
ARM_PADDLE_GRAVITY_RATIO = 9.8 / 4.0
ARM_PADDLE_MODEL_SCALE = 1.0
ARM_PADDLE_BASE_POSITION = scale_vector((1.0, -1.5, 0.0), ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_BASE_EULER = (0.0, 0.0, -0.5 * math.pi)
# User-selected sweep candidate "05_zoom_0.47": the previous location scaled
# to 0.47x its offset from the target. Target z sits 2 cm below the sweep
# value so planned-failure floor bounces stay inside the bottom frame edge.
ARM_PADDLE_CAMERA_LOCATION = (0.0735, -1.161, 0.6813)
ARM_PADDLE_CAMERA_TARGET = (0.05, -0.55, 0.29)
ARM_PADDLE_CAMERA_FOVY_DEGREES = PADDLE_BALL_CAMERA_FOVY_DEGREES
ARM_PADDLE_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
ARM_PADDLE_ACTUATOR_NAMES = ["motor1", "motor2", "motor3", "motor4", "motor5", "motor6"]
ARM_PADDLE_HOME_QPOS = [-0.87, 1.69, -0.81, 0.12, 0.57, 0.0]
ARM_PADDLE_TOOL_TIP_LOCAL_X = 0.051
ARM_PADDLE_HANDLE_ANGLE = math.radians(45.0)
ARM_PADDLE_HANDLE_ROOT_OFFSET = scale_vector((0.0, 0.3375, 0.0), ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_HANDLE_GRIP_DISTANCE = scale_linear(0.44, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_HANDLE_EMBED_DEPTH = scale_linear(0.10, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_HANDLE_HALF_WIDTH = scale_linear(0.048, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_HANDLE_MATERIAL = demo_generate.material_from_hex(
    "#B08048", roughness=0.85, specular=0.06
)
ARM_PADDLE_GROUND_HALF_EXTENTS = scale_vector(PADDLE_GROUND_HALF_EXTENTS, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_GROUND_CENTER = scale_vector(APPROACH_BALL_GROUND_CENTER, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_SIDE_WALL_HEIGHT = scale_linear(APPROACH_BALL_SIDE_WALL_HEIGHT, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_SIDE_WALL_THICKNESS = scale_linear(
    APPROACH_BALL_SIDE_WALL_THICKNESS, ARM_PADDLE_SCENE_SCALE
)
ARM_PADDLE_GRID_SPACING = scale_linear(APPROACH_BALL_GRID_SPACING, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_GRID_HALF_THICKNESS = scale_linear(
    APPROACH_BALL_GRID_HALF_THICKNESS, ARM_PADDLE_SCENE_SCALE
)
ARM_PADDLE_GRID_SURFACE_OFFSET = scale_linear(
    APPROACH_BALL_GRID_SURFACE_OFFSET, ARM_PADDLE_SCENE_SCALE
)
ARM_PADDLE_GRID_DEPTH_SCALE = nonnegative_float_env(
    "DEMO_MUJOCO_ARM_PADDLE_GRID_DEPTH_SCALE",
    0.0,
)
ARM_PADDLE_GRID_DEPTH_HALF_THICKNESS = flattenable_grid_depth(
    ARM_PADDLE_GRID_HALF_THICKNESS,
    ARM_PADDLE_GRID_DEPTH_SCALE,
)
ARM_PADDLE_WORKSPACE_MIN = scale_vector(PADDLE_WORKSPACE_MIN, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_WORKSPACE_MAX = scale_vector(PADDLE_WORKSPACE_MAX, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_ACTION_SCALE_XYZ = scale_vector(PADDLE_ACTION_SCALE_XYZ, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_PADDLE_HALF_EXTENTS = scale_vector(PADDLE_BASE_HALF_EXTENTS, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_INITIAL_POSITION = scale_vector(PADDLE_INITIAL_POSITION, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_TILT_CENTER_XY = scale_vector((0.0, -3.0), ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_BALL_RADIUS = scale_linear(PADDLE_BALL_RADIUS, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_BALL_MASS = PADDLE_BALL_MASS
# Enlarged initial ball horizontal spread (~2.5x the scaled paddle_ball range of
# +-0.044 x / +-0.016 y about the centre -0.6) for more state diversity; kept
# inside the workspace (x +-0.23, y [-0.67,-0.53]) with margin so the policy can
# still reach and centre it.
ARM_PADDLE_BALL_INITIAL_X_RANGE = (-0.11, 0.11)
ARM_PADDLE_BALL_INITIAL_Y_RANGE = (-0.645, -0.555)
# Legacy strike-height contact plane: strike_z + paddle_half_z + ball_radius.
_ARM_PADDLE_LEGACY_STRIKE_CONTACT_Z = 0.66 + PADDLE_BASE_HALF_EXTENTS[2] + PADDLE_BALL_RADIUS
_ARM_PADDLE_STRIKE_CONTACT_Z = (
    scale_linear(0.66, ARM_PADDLE_SCENE_SCALE)
    + ARM_PADDLE_PADDLE_HALF_EXTENTS[2]
    + ARM_PADDLE_BALL_RADIUS
)
# Extra shrink applied to the bounce envelope on top of the gravity-ratio
# mapping (user-tuned: halved twice, then raised 1.25x => 0.3125 of the pure
# gravity-ratio value).
ARM_PADDLE_BOUNCE_HEIGHT_SCALE = 0.3125
_ARM_PADDLE_VERTICAL_RATIO = ARM_PADDLE_GRAVITY_RATIO * ARM_PADDLE_BOUNCE_HEIGHT_SCALE
# Vertical speeds follow v ~ sqrt(g * h): gravity ratio for g, vertical ratio for h.
_ARM_PADDLE_VERTICAL_SPEED_RATIO = math.sqrt(ARM_PADDLE_GRAVITY_RATIO * _ARM_PADDLE_VERTICAL_RATIO)
ARM_PADDLE_TARGET_APEX_Z = _ARM_PADDLE_STRIKE_CONTACT_Z + _ARM_PADDLE_VERTICAL_RATIO * (
    1.10 - _ARM_PADDLE_LEGACY_STRIKE_CONTACT_Z
)
ARM_PADDLE_BALL_INITIAL_Z_RANGE = tuple(
    _ARM_PADDLE_STRIKE_CONTACT_Z
    + _ARM_PADDLE_VERTICAL_RATIO * (value - _ARM_PADDLE_LEGACY_STRIKE_CONTACT_Z)
    for value in PADDLE_BALL_INITIAL_Z_RANGE
)
ARM_PADDLE_BALL_INITIAL_VX_RANGE = scale_range(PADDLE_BALL_INITIAL_VX_RANGE, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_BALL_INITIAL_VY_RANGE = scale_range(PADDLE_BALL_INITIAL_VY_RANGE, ARM_PADDLE_SCENE_SCALE)
ARM_PADDLE_BALL_INITIAL_VZ_RANGE = scale_range(
    PADDLE_BALL_INITIAL_VZ_RANGE, _ARM_PADDLE_VERTICAL_SPEED_RATIO
)
# Real-scale contacts: the default soft elastic solref (stiffness 1000)
# lets a 4 cm ball at several m/s tunnel through the 2 cm floor, and the
# near-inelastic default ball-paddle contact cannot sustain bouncing, so the
# paddle contact carries a true restitution-0.92 solref instead.
ARM_PADDLE_CONTACT_STIFFNESS = 6.0e4
ARM_PADDLE_GROUND_CONTACT_SOLREF = demo_generate.elastic_contact_solref(
    1.0, ARM_PADDLE_CONTACT_STIFFNESS
)
# The floor/walls get a natural ball bounce (moderate restitution) plus modest
# friction so a ball that a planned failure knocks off the paddle bounces a few
# times and rolls -- rather than skating across the default frictionless elastic
# floor to infinity, or (with too much friction) dead-stopping on impact. The
# friction bleeds the lateral speed so it comes to rest within the (very wide)
# floor. arm_paddle successes never touch the floor or walls, so this only shapes
# how a failed ball behaves once it leaves the paddle.
ARM_PADDLE_GROUND_LANDING_SOLREF = demo_generate.elastic_contact_solref(
    0.72, ARM_PADDLE_CONTACT_STIFFNESS
)
ARM_PADDLE_GROUND_LANDING_FRICTION = 0.25
ARM_PADDLE_PADDLE_CONTACT_SOLREF = demo_generate.elastic_contact_solref(
    PADDLE_RESTITUTION, ARM_PADDLE_CONTACT_STIFFNESS
)
# Heavier paddle: with a 0.1 kg ball, a 0.45 kg paddle recoils on impact and
# drops the effective ball-paddle restitution to ~0.63, which the strike
# cannot make up at g=20. 2.0 kg raises the two-body effective restitution to
# ~0.83 so the shared apex target stays reachable across the gravity range.
ARM_PADDLE_PADDLE_COLLISION_MASS = 2.0
# Paddle servo: keep the legacy tracking pole kp/damping ~= 18.6/s -- strike
# energy regulation relies on servo lag making contact speed proportional to
# the commanded stroke -- and scale kp/damping/forcerange with the paddle
# mass so the pole and damping ratio are unchanged; gravity sag m*g/kp stays
# below ~8 mm at the g=20 test extreme.
ARM_PADDLE_PADDLE_SERVO = {
    "slide_damping": 280.0,
    "slide_armature": 0.05,
    "position_kp": 5200.0,
    "position_forcerange": 1200.0,
    "tilt_damping": 4.0,
    "tilt_armature": 0.015,
    "tilt_kp": 180.0,
    "tilt_forcerange": 90.0,
}
ARM_PADDLE_POLICY_CONFIG_OVERRIDES = {
    "noise_std_xy": 0.003 * ARM_PADDLE_SCENE_SCALE,
    # Shallower reset at high gravity: the bounce period at g~20 (~5 frames)
    # is shorter than a full swing cycle of the lagged servo, so a deep reset
    # leaves the paddle still descending at the next contact and eats the
    # rebound energy no matter how high the strike is trimmed.
    "reset_z": 0.52 * ARM_PADDLE_SCENE_SCALE,
    "low_gravity_reset_z": 0.42 * ARM_PADDLE_SCENE_SCALE,
    "strike_z": 0.66 * ARM_PADDLE_SCENE_SCALE,
    "low_gravity_strike_z": 0.45 * ARM_PADDLE_SCENE_SCALE,
    # The legacy mid-gravity strike boosts compensated for the near-inelastic
    # default contact; with a truly elastic paddle they only inflate apexes.
    "mid_gravity_strike_boost": 0.0,
    "mid_gravity_strike_center": 1.0 * ARM_PADDLE_GRAVITY_RATIO,
    "mid_gravity_strike_sigma": 0.30 * ARM_PADDLE_GRAVITY_RATIO,
    "mid_low_gravity_strike_boost": 0.0,
    "mid_low_gravity_strike_center": 1.8 * ARM_PADDLE_GRAVITY_RATIO,
    "mid_low_gravity_strike_sigma": 0.45 * ARM_PADDLE_GRAVITY_RATIO,
    "min_strike_z": 0.42 * ARM_PADDLE_SCENE_SCALE,
    # Near the top of the workspace: at g=20 the adaptive trim needs the full
    # stroke authority to reach the shared target apex.
    "max_strike_z": 0.875 * ARM_PADDLE_SCENE_SCALE,
    "target_apex_z": ARM_PADDLE_TARGET_APEX_Z,
    "gravity_pulse_min": 0.5 * ARM_PADDLE_GRAVITY_RATIO,
    "gravity_pulse_full": 3.0 * ARM_PADDLE_GRAVITY_RATIO,
    # Wide strike band plus a strong apex gain: with a truly elastic paddle
    # the apex feedback is the main energy regulator at real scale. The band
    # must reach down to the reset height so the strike can back off to a
    # zero stroke when the ball is too energetic.
    "strike_band": 0.050,
    "high_ball_z": _ARM_PADDLE_STRIKE_CONTACT_Z
    + _ARM_PADDLE_VERTICAL_RATIO * (1.20 - _ARM_PADDLE_LEGACY_STRIKE_CONTACT_Z),
    "high_ball_vz_threshold": -0.05 * _ARM_PADDLE_VERTICAL_SPEED_RATIO,
    # Gravity-scaled apex gain: ~0.25 at train gravity (enough that ~0.2 m of
    # apex excess retracts the whole strike stroke) without making the
    # 1/g-sensitive low-gravity loop oscillate.
    "height_gain": 0.08,
    "height_gain_per_gravity": 0.025,
    # Per-bounce integral trim on the strike height: drives the rebound apex
    # to target_apex_z at every gravity (a fixed stroke equilibrates at a
    # gravity-dependent apex).
    # sqrt(g)-scaled per-bounce gain gives a gravity-independent loop gain of
    # ~4.5x this base; 0.1 keeps it comfortably below the oscillation point.
    "adaptive_apex_trim_gain": 0.1,
    "adaptive_strike_trim_limit": 0.06,
    "adaptive_strike_trim": 0.0,
    # Feedforward initial trim fitted to the converged per-gravity values
    # (trim* ~ -0.012*ln(g)); low-gravity episodes have too few bounces for
    # the integral alone to converge within an episode.
    "adaptive_trim_init_log_coeff": -0.012,
    "adaptive_trim_init_ref_g": 1.0,
    # Below this gravity the strike is a velocity-matched target ramp instead
    # of a position stroke: the ~1/g apex sensitivity makes stroke control
    # hunt on frame-timing jitter at low g.
    "velocity_strike_max_gravity": 1.5,
    "velocity_strike_lead_seconds": 0.30,
    # Effective restitution of the ball on the 2 kg servo-held paddle,
    # calibrated from measured low-gravity rebounds (two-body estimate 0.83
    # plus the servo holding the paddle against recoil).
    "strike_effective_restitution": 0.87,
    "control_dt": 1.0 / FPS,
    "apex_deadband": 0.02 * _ARM_PADDLE_VERTICAL_RATIO,
    "low_gravity_hold_threshold": 0.75 * ARM_PADDLE_GRAVITY_RATIO,
    "min_bounce_velocity_after_contact": 0.05 * _ARM_PADDLE_VERTICAL_SPEED_RATIO,
    "strike_min_descend_speed": 0.18 * _ARM_PADDLE_VERTICAL_SPEED_RATIO,
    "contact_release_clearance_z": 0.035 * ARM_PADDLE_SCENE_SCALE,
    "contact_release_drop_z": 0.08 * ARM_PADDLE_SCENE_SCALE,
    # High-g bounces span only ~3-4 frames with the quarter-height apex; the
    # post-contact hold+reset windows must stay shorter than that or they
    # swallow the next strike.
    "hold_frames_after_contact": 1,
    "post_contact_reset_frames": 1,
    "xy_track_max_center_offset": 0.035 * ARM_PADDLE_SCENE_SCALE,
    "tilt_deadband_velocity": 0.025 * ARM_PADDLE_SCENE_SCALE,
    "tilt_max_centering_speed": 0.70 * ARM_PADDLE_SCENE_SCALE,
    "tilt_min_incoming_speed": 0.75 * _ARM_PADDLE_VERTICAL_SPEED_RATIO,
}
CATCHER_BALL_FIXED_Y = -3.25
CATCHER_BALL_WORKSPACE_MIN = (-2.8, CATCHER_BALL_FIXED_Y, 0.35)
CATCHER_BALL_WORKSPACE_MAX = (2.8, CATCHER_BALL_FIXED_Y, 2.80)
CATCHER_BALL_INITIAL_X_RANGE = (-0.65, 0.65)
CATCHER_BALL_INITIAL_Z_RANGE = (0.45, 1.05)
CATCHER_BALL_NET_HALF_WIDTH = 0.82
CATCHER_BALL_NET_HALF_HEIGHT = 0.62
CATCHER_BALL_NET_DEPTH = 0.64
CATCHER_BALL_RIM_HALF_THICKNESS = 0.045
CATCHER_BALL_GRID_HALF_THICKNESS = 0.018
CATCHER_BALL_COLLISION_WALL_THICKNESS = 0.055
CATCHER_BALL_MATERIAL = demo_generate.material_from_hex("#293A52", roughness=0.90, specular=0.04)
CATCHER_BALL_NET_MATERIAL = demo_generate.material_from_hex(
    "#37506F", roughness=0.92, specular=0.03
)
CATCHER_BALL_COLLISION_MATERIAL = demo_generate.material_from_hex(
    "#293A52", alpha=0.18, roughness=0.95, specular=0.0
)
CATCHER_BALL_CATCH_FRONT_Y = CATCHER_BALL_FIXED_Y + 0.5 * CATCHER_BALL_NET_DEPTH
CATCHER_BALL_CAPTURE_CROSS_FRAME_RANGE = (42, 48)
CATCHER_BALL_TRUNCATED_CROSS_FRAME_RANGE = (74, 88)
CATCHER_BALL_RADIUS = 0.40
CATCHER_BALL_TARGET_X_LIMIT = APPROACH_BALL_GROUND_HALF_EXTENTS[0] - CATCHER_BALL_RADIUS
CATCHER_BALL_TARGET_X_RANGE = (
    -min(6.50, CATCHER_BALL_TARGET_X_LIMIT),
    min(6.50, CATCHER_BALL_TARGET_X_LIMIT),
)
CATCHER_BALL_TRUNCATED_TARGET_X_RANGE = (
    -min(8.32, CATCHER_BALL_TARGET_X_LIMIT),
    min(8.32, CATCHER_BALL_TARGET_X_LIMIT),
)
CATCHER_BALL_TARGET_Z_LIMITS = (CATCHER_BALL_RADIUS, APPROACH_BALL_SIDE_WALL_HEIGHT)
CATCHER_BALL_TARGET_Z_RANGE = widen_range(
    (
        CATCHER_BALL_WORKSPACE_MIN[2] + 1.35,
        CATCHER_BALL_WORKSPACE_MAX[2] - 0.35,
    ),
    1.5,
    *CATCHER_BALL_TARGET_Z_LIMITS,
)
CATCHER_BALL_TRUNCATED_TARGET_Z_RANGE = widen_range(
    (
        CATCHER_BALL_WORKSPACE_MIN[2] + 0.22,
        CATCHER_BALL_WORKSPACE_MAX[2] - 0.28,
    ),
    1.5,
    *CATCHER_BALL_TARGET_Z_LIMITS,
)
CATCHER_BALL_OUTCOME_CYCLE = (
    "capture",
    "miss",
    "capture",
    "capture",
    "capture",
    "capture",
    "miss",
    "capture",
    "capture",
    "capture",
    "capture",
    "capture",
    "miss",
    "capture",
    "capture",
    "capture",
    "capture",
    "capture",
    "truncated",
    "capture",
)
ARM_CATCHER_BALL_OUTCOME_CYCLE = (
    "capture",
    "miss",
    "capture",
    "capture",
    "capture",
)
ARM_CATCHER_REFERENCE_SCALE = 7.5
ARM_CATCHER_SCENE_SCALE = 1.0 / ARM_CATCHER_REFERENCE_SCALE
ARM_CATCHER_START_Y_MULTIPLIER = positive_float_env(
    "DEMO_MUJOCO_ARM_CATCHER_START_Y_MULTIPLIER",
    2.0,
)
ARM_CATCHER_BALL_SPEED_SCALE = positive_float_env(
    "DEMO_MUJOCO_ARM_CATCHER_BALL_SPEED_SCALE",
    0.25,
)
_ARM_CATCHER_CROSS_FRAME_MULTIPLIER_OVERRIDE = os.environ.get(
    "DEMO_MUJOCO_ARM_CATCHER_CROSS_FRAME_MULTIPLIER",
    "",
).strip()
if _ARM_CATCHER_CROSS_FRAME_MULTIPLIER_OVERRIDE:
    ARM_CATCHER_CROSS_FRAME_MULTIPLIER = positive_float_env(
        "DEMO_MUJOCO_ARM_CATCHER_CROSS_FRAME_MULTIPLIER",
        0.4 / ARM_CATCHER_BALL_SPEED_SCALE,
    )
    ARM_CATCHER_CAPTURE_CROSS_FRAME_RANGE = (
        16.0 * ARM_CATCHER_CROSS_FRAME_MULTIPLIER,
        22.0 * ARM_CATCHER_CROSS_FRAME_MULTIPLIER,
    )
    ARM_CATCHER_CAPTURE_CROSS_TIME_RANGE = tuple(
        frame / FPS for frame in ARM_CATCHER_CAPTURE_CROSS_FRAME_RANGE
    )
else:
    ARM_CATCHER_CAPTURE_CROSS_TIME_RANGE = (
        positive_float_env("DEMO_MUJOCO_ARM_CATCHER_CAPTURE_TIME_MIN", 3.0),
        positive_float_env("DEMO_MUJOCO_ARM_CATCHER_CAPTURE_TIME_MAX", 3.6),
    )
    if ARM_CATCHER_CAPTURE_CROSS_TIME_RANGE[0] >= ARM_CATCHER_CAPTURE_CROSS_TIME_RANGE[1]:
        raise ValueError(
            "DEMO_MUJOCO_ARM_CATCHER_CAPTURE_TIME_MIN must be smaller than "
            "DEMO_MUJOCO_ARM_CATCHER_CAPTURE_TIME_MAX; got "
            f"{ARM_CATCHER_CAPTURE_CROSS_TIME_RANGE!r}"
        )
    ARM_CATCHER_CAPTURE_CROSS_FRAME_RANGE = tuple(
        time_seconds * FPS for time_seconds in ARM_CATCHER_CAPTURE_CROSS_TIME_RANGE
    )
    ARM_CATCHER_CROSS_FRAME_MULTIPLIER = sum(ARM_CATCHER_CAPTURE_CROSS_FRAME_RANGE) / (16.0 + 22.0)
ARM_CATCHER_Y_ACTION_MULTIPLIER = positive_float_env(
    "DEMO_MUJOCO_ARM_CATCHER_Y_ACTION_MULTIPLIER",
    2.0,
)
# Arena trimmed back to the 5267eab proportions. Doubling it for the longer
# throw pushed the back wall to 14.0 and the room read as a corridor, but the
# launch only reaches y=9.6, so hugging it at 10.6 keeps the whole throw and
# restores the legacy box. Recentring puts the near floor edge back at -7.1,
# where 5267eab had it (just off the bottom of frame from this camera).
ARM_CATCHER_REFERENCE_GROUND_CENTER_Y = 1.75
ARM_CATCHER_REFERENCE_GROUND_HALF_EXTENTS = (
    5.2,
    8.85,
    0.05,
)
# The 5267eab camera, unchanged. Kept in legacy reference units, then scaled to
# scene units below with ARM_CATCHER_SCENE_SCALE. With the arena trimmed above,
# this lands the arm base within 0.01 ndc of where the legacy render put it.
ARM_CATCHER_REFERENCE_CAMERA_LOCATION = (
    0.50,
    -9.05,
    6.58,
)
ARM_CATCHER_REFERENCE_CAMERA_TARGET = (
    0.90,
    -2.50,
    1.35,
)
ARM_CATCHER_GROUND_CENTER = scale_vector(
    (
        APPROACH_BALL_GROUND_CENTER[0],
        ARM_CATCHER_REFERENCE_GROUND_CENTER_Y,
        APPROACH_BALL_GROUND_CENTER[2],
    ),
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_GROUND_HALF_EXTENTS = scale_vector(
    ARM_CATCHER_REFERENCE_GROUND_HALF_EXTENTS,
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_SIDE_WALL_HEIGHT = scale_linear(
    APPROACH_BALL_SIDE_WALL_HEIGHT,
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_SIDE_WALL_THICKNESS = scale_linear(
    APPROACH_BALL_SIDE_WALL_THICKNESS,
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_GRID_SPACING = scale_linear(APPROACH_BALL_GRID_SPACING, ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_GRID_SURFACE_OFFSET = scale_linear(
    APPROACH_BALL_GRID_SURFACE_OFFSET,
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_GROUND_GRID_HALF_THICKNESS = scale_linear(
    APPROACH_BALL_GRID_HALF_THICKNESS,
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_GRID_DEPTH_SCALE = nonnegative_float_env(
    "DEMO_MUJOCO_ARM_CATCHER_GRID_DEPTH_SCALE",
    os.environ.get("DEMO_MUJOCO_ARM_CATCHER_WALL_GRID_DEPTH_SCALE", "1.0"),
)
# Keep the grid-line width unchanged, and let arm_catcher_ball flatten the
# out-of-plane grid depth when asked. The flattening was added to hide thick
# striping under the unscaled light rig; with the rig scaled to the scene
# (render_light_scale below) the full-depth grid renders as it did at 5267eab,
# so this defaults back to 1.0.
ARM_CATCHER_GRID_DEPTH_HALF_THICKNESS = flattenable_grid_depth(
    ARM_CATCHER_GROUND_GRID_HALF_THICKNESS,
    ARM_CATCHER_GRID_DEPTH_SCALE,
)
ARM_CATCHER_CAMERA_LOCATION = scale_vector(
    ARM_CATCHER_REFERENCE_CAMERA_LOCATION,
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_CAMERA_TARGET = scale_vector(
    ARM_CATCHER_REFERENCE_CAMERA_TARGET,
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_CAMERA_FOVY_DEGREES = 54.0
# Base mounted 1.35 raw units from the net (mount_back_y, below) and shifted
# +0.6 raw units in x. With the net normal locked to +y, keeping the base near
# the catch plane keeps the shoulder and elbow outside the wrist's forward
# swing. The capture x-range below stays within the measured reach envelope.
#
# The whole apparatus (this base position and mount_back_y together) was
# later shifted +0.6 raw in y as a rigid unit -- base y and mount_back_y move
# by the same amount, so the 1.35 raw gap (and every workspace/reach number
# tuned against it) is unchanged; only where the catch happens in the scene
# moved.
ARM_CATCHER_BASE_POSITION = scale_vector((3.6, -3.45, 0.05), ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_MODEL_SCALE = 1.0
ARM_CATCHER_SHOULDER_HEIGHT = scale_linear(0.0, ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_UPPER_LINK_LENGTH = 0.43
ARM_CATCHER_LOWER_LINK_LENGTH = 0.32
ARM_CATCHER_LINK_HALF_THICKNESS = scale_linear(0.075, ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_ACTION_SCALE_XYZ = scale_vector(
    (0.42, 0.20 * ARM_CATCHER_Y_ACTION_MULTIPLIER, 0.34),
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_NET_HALF_WIDTH = scale_linear(0.478125, ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_NET_HALF_HEIGHT = scale_linear(0.46125, ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_NET_DEPTH = scale_linear(0.225, ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_RIM_HALF_THICKNESS = scale_linear(0.016875, ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_GRID_HALF_THICKNESS = scale_linear(0.0061875, ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_COLLISION_WALL_THICKNESS = scale_linear(0.021375, ARM_CATCHER_SCENE_SCALE)
# The tool tip sits at the flange, 0.3825 legacy units out from link06, which
# is exactly where link06's mesh ends -- so a cage bolted there stands entirely
# clear of the wrist and the wrist shows through its open back. 5267eab mounted
# the net 0.13 out instead, seating the cage over the wrist. Seat it back by the
# difference; the mount origin carries the capture bounds with it, and the arm
# reaches 0.2525 further along the net normal to put the net in the same place.
ARM_CATCHER_MOUNT_BACK_OFFSET = scale_linear(
    float_env("DEMO_MUJOCO_ARM_CATCHER_MOUNT_BACK_OFFSET", 0.3825 - 0.13),
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_MOUNT_BACK_Y = scale_linear(-2.10, ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_CATCH_FRONT_Y = ARM_CATCHER_MOUNT_BACK_Y + ARM_CATCHER_NET_DEPTH
ARM_CATCHER_WORKSPACE_MIN = (
    scale_linear(0.0, ARM_CATCHER_SCENE_SCALE),
    ARM_CATCHER_MOUNT_BACK_Y
    - scale_linear(0.20 * ARM_CATCHER_Y_ACTION_MULTIPLIER, ARM_CATCHER_SCENE_SCALE),
    scale_linear(0.85, ARM_CATCHER_SCENE_SCALE),
)
# With the null-space home bias, the end effector tracks to sub-2 mm error up
# to z~4.1 reference units across the capture x-range. The x ceiling is 3.75:
# beyond x~3.9 the tip-normal IK crosses a branch fold and wedges joint 5 at
# its limit, while planned captures remain at x <= 3.45.
ARM_CATCHER_WORKSPACE_MAX = (
    scale_linear(3.75, ARM_CATCHER_SCENE_SCALE),
    ARM_CATCHER_MOUNT_BACK_Y
    + scale_linear(0.35 * ARM_CATCHER_Y_ACTION_MULTIPLIER, ARM_CATCHER_SCENE_SCALE),
    scale_linear(4.00, ARM_CATCHER_SCENE_SCALE),
)
ARM_CATCHER_INITIAL_X_RANGE = scale_range((0.90, 1.35), ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_INITIAL_Y_RANGE = (
    ARM_CATCHER_MOUNT_BACK_Y - scale_linear(0.05, ARM_CATCHER_SCENE_SCALE),
    ARM_CATCHER_MOUNT_BACK_Y + scale_linear(0.05, ARM_CATCHER_SCENE_SCALE),
)
ARM_CATCHER_INITIAL_Z_RANGE = scale_range((1.50, 2.25), ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_START_X_RANGE = scale_range((-1.60, 4.20), ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_START_Y_RANGE = scale_range(
    (
        3.8 * ARM_CATCHER_START_Y_MULTIPLIER,
        4.8 * ARM_CATCHER_START_Y_MULTIPLIER,
    ),
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_FINAL_TARGET_X_SHIFT_FRACTION = float_env(
    "DEMO_MUJOCO_ARM_CATCHER_FINAL_TARGET_X_SHIFT_FRACTION",
    0.0,
)
ARM_CATCHER_REFERENCE_OPEN_BOX_WIDTH_X = 2.0 * ARM_CATCHER_REFERENCE_GROUND_HALF_EXTENTS[0]
ARM_CATCHER_FINAL_TARGET_X_SHIFT = scale_linear(
    ARM_CATCHER_FINAL_TARGET_X_SHIFT_FRACTION * ARM_CATCHER_REFERENCE_OPEN_BOX_WIDTH_X,
    ARM_CATCHER_SCENE_SCALE,
)
# The 0.85-2.25 raw-unit range leaves margin at the IK-feasible lower edge and
# stops short of the base's own x-position (3.6). This avoids the high-z
# corner where reaching vertically beside the base straightens the shoulder.
ARM_CATCHER_CAPTURE_TARGET_X_RANGE = offset_range(
    scale_range((0.85, 2.25), ARM_CATCHER_SCENE_SCALE),
    ARM_CATCHER_FINAL_TARGET_X_SHIFT,
)
ARM_CATCHER_MISS_TARGET_X_LEFT_RANGE = offset_range(
    scale_range((-2.20, -1.70), ARM_CATCHER_SCENE_SCALE),
    ARM_CATCHER_FINAL_TARGET_X_SHIFT,
)
ARM_CATCHER_MISS_TARGET_X_RIGHT_RANGE = offset_range(
    scale_range((2.70, 3.20), ARM_CATCHER_SCENE_SCALE),
    ARM_CATCHER_FINAL_TARGET_X_SHIFT,
)
ARM_CATCHER_TRUNCATED_TARGET_X_RANGE = offset_range(
    scale_range((1.50, 1.50), ARM_CATCHER_SCENE_SCALE),
    ARM_CATCHER_FINAL_TARGET_X_SHIFT,
)
ARM_CATCHER_TARGET_Z_RANGE = scale_range((1.30, 3.23), ARM_CATCHER_SCENE_SCALE)
# Capture crossings sample a hint uniformly between the x-dependent reach
# floor (arm_catcher_reachable_mount_z) and this ceiling, so the arm catches
# anywhere in its vertical reach instead of only near the old low band.
ARM_CATCHER_CAPTURE_MAX_TARGET_Z_HINT = scale_linear(2.85, ARM_CATCHER_SCENE_SCALE)
# Lowest crossing height a planned capture may target: slightly below the
# measured EE reach floor (see arm_catcher_reachable_mount_z), close enough
# for the latch window to cover the difference.
ARM_CATCHER_CAPTURE_MIN_TARGET_Z = scale_linear(1.86, ARM_CATCHER_SCENE_SCALE)
# Lowest EE height the interception planner may command; keeps IK away from
# the joint-limit corner below the reachable band.
ARM_CATCHER_PLANNER_MIN_Z = scale_linear(1.90, ARM_CATCHER_SCENE_SCALE)
# Max vertical ball speed at the catch plane for planned captures. Substep
# detection could register much faster crossings, but the simulated bounce
# phase drifts a few tens of ms from the ideal elastic model over an episode,
# so a fast (steep) crossing moves vertically by more than the capture window
# between prediction and arrival. Keeping crossings within ~0.03 of an arc
# apex (|vz| <= sqrt(2*g*0.03)) makes the crossing height drift-insensitive
# while still allowing the apex itself to sit anywhere in the raised reach.
ARM_CATCHER_CAPTURE_MAX_CROSS_VZ = positive_float_env(
    "DEMO_MUJOCO_ARM_CATCHER_CAPTURE_MAX_CROSS_VZ",
    scale_linear(7.0, ARM_CATCHER_SCENE_SCALE),
)
# 1.1x the long-standing 0.50 reference radius. Everything keyed off this
# constant (capture centre tolerances, front margin, latch bounds) scales with it.
ARM_CATCHER_BALL_RADIUS = 1.1 * scale_linear(0.50, ARM_CATCHER_SCENE_SCALE)
# Lowered after the late-catch retune so the average trajectory height is
# roughly 0.85x the previous 128-frame dataset while staying above the IK
# reach floor used by planned captures.
ARM_CATCHER_START_HEIGHT_ABOVE_RADIUS_RANGE = scale_range((0.85, 2.72), ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_CAPTURE_CENTER_HALF_WIDTH = 0.25 * ARM_CATCHER_BALL_RADIUS
ARM_CATCHER_CAPTURE_CENTER_HALF_HEIGHT = 0.25 * ARM_CATCHER_BALL_RADIUS
ARM_CATCHER_CAPTURE_CENTER_RADIUS = 0.25 * ARM_CATCHER_BALL_RADIUS
ARM_CATCHER_CAPTURE_FRONT_MARGIN = positive_float_env(
    "DEMO_MUJOCO_ARM_CATCHER_CAPTURE_FRONT_MARGIN",
    max(
        0.55 * ARM_CATCHER_BALL_RADIUS,
        ARM_CATCHER_BALL_RADIUS
        + 2.0 * ARM_CATCHER_COLLISION_WALL_THICKNESS
        - ARM_CATCHER_NET_DEPTH,
    ),
)
ARM_CATCHER_CAPTURE_BACK_MARGIN = positive_float_env(
    "DEMO_MUJOCO_ARM_CATCHER_CAPTURE_BACK_MARGIN",
    0.15 * ARM_CATCHER_BALL_RADIUS,
)
ARM_CATCHER_CAPTURE_MAX_LATCH_JUMP = positive_float_env(
    "DEMO_MUJOCO_ARM_CATCHER_CAPTURE_MAX_LATCH_JUMP",
    0.20 * ARM_CATCHER_BALL_RADIUS,
)
ARM_CATCHER_LATCH_HALF_WIDTH = ARM_CATCHER_CAPTURE_CENTER_HALF_WIDTH
ARM_CATCHER_LATCH_HALF_HEIGHT = ARM_CATCHER_CAPTURE_CENTER_HALF_HEIGHT
ARM_CATCHER_LATCH_Y_MIN = -ARM_CATCHER_CAPTURE_BACK_MARGIN
ARM_CATCHER_LATCH_Y_MAX = ARM_CATCHER_NET_DEPTH + ARM_CATCHER_CAPTURE_FRONT_MARGIN
# First-bounce time at the reference gravity (9.8); scaled by sqrt(9.8/g) at
# sampling time so post-bounce arc heights (~0.5*g*t^2) stay comparable
# across the 0..20 gravity sweep instead of exploding at high g.
ARM_CATCHER_FIRST_BOUNCE_TIME_RANGE = (0.46, 0.90)


def arm_catcher_first_bounce_time_range(gravity):
    factor = math.sqrt(9.8 / max(float(gravity), 2.0))
    low = min(max(ARM_CATCHER_FIRST_BOUNCE_TIME_RANGE[0] * factor, 0.25), 1.05)
    high = min(max(ARM_CATCHER_FIRST_BOUNCE_TIME_RANGE[1] * factor, low + 0.05), 1.10)
    return low, high


ARM_CATCHER_MAX_TRAJECTORY_Z = scale_linear(4.68, ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_DIRECT_CAPTURE_START_HEIGHT_RANGE = scale_range((1.02, 1.70), ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_TARGET_Y_JITTER = scale_linear(0.08, ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_CAPTURE_TARGET_Y_JITTER = scale_linear(0.025, ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_TARGET_Z_HINT_JITTER = scale_linear(0.035, ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_CAPTURE_VX_NOISE_RANGE = scale_range((-0.08, 0.08), ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_MISS_VX_NOISE_RANGE = scale_range((-1.20, 1.20), ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_CAPTURE_VY_NOISE_RANGE = scale_range((-0.08, 0.08), ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_MISS_VY_NOISE_RANGE = scale_range((-1.20, 1.20), ARM_CATCHER_SCENE_SCALE)
ARM_CATCHER_TARGET_SMOOTHING_ALPHA = 0.42
ARM_CATCHER_SMOOTH_MAX_DELTA_XYZ = scale_vector(
    (0.085, 0.040 * ARM_CATCHER_Y_ACTION_MULTIPLIER, 0.105),
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_ACTION_SMOOTHING_ALPHA = 0.24
ARM_CATCHER_ACTION_MAX_DELTA_CHANGE_XYZ = scale_vector(
    (0.018, 0.009 * ARM_CATCHER_Y_ACTION_MULTIPLIER, 0.024),
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_PER_FRAME_MAX_DELTA_XYZ = scale_vector(
    (0.12, 0.05 * ARM_CATCHER_Y_ACTION_MULTIPLIER, 0.12),
    ARM_CATCHER_SCENE_SCALE,
)
# The planner spreads the remaining travel over (time_to_crossing - margin),
# so the mount settles at the interception point ~margin frames before the
# ball arrives and holds there; 34 frames at 32 fps puts arrival around
# 2.0-2.5 s for the 3.0-3.6 s capture window.
ARM_CATCHER_PLANNER_ARRIVAL_MARGIN_FRAMES = positive_float_env(
    "DEMO_MUJOCO_ARM_CATCHER_PLANNER_ARRIVAL_MARGIN_FRAMES",
    34.0,
)
ARM_CATCHER_PLANNER_MAX_SPEED_XYZ = scale_vector(
    (0.18, 0.12 * ARM_CATCHER_Y_ACTION_MULTIPLIER, 0.18),
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_PLANNER_MAX_ACCEL_XYZ = scale_vector(
    (0.042, 0.028 * ARM_CATCHER_Y_ACTION_MULTIPLIER, 0.042),
    ARM_CATCHER_SCENE_SCALE,
)
# Smooth exploratory wiggle: an Ornstein-Uhlenbeck offset added to the
# planner's per-frame step, faded to zero well before interception so
# captures are unaffected. Amplitude is the stationary standard deviation in
# scene units per frame; it contributes amplitude*fps to the EE velocity, so
# keep it small at 32 fps. Tau sets how often the wiggle changes direction.
ARM_CATCHER_PLANNER_WIGGLE_AMPLITUDE_XYZ = scale_vector(
    (0.024, 0.006 * ARM_CATCHER_Y_ACTION_MULTIPLIER, 0.020),
    ARM_CATCHER_SCENE_SCALE,
)
ARM_CATCHER_PLANNER_WIGGLE_TAU_SECONDS = positive_float_env(
    "DEMO_MUJOCO_ARM_CATCHER_PLANNER_WIGGLE_TAU_SECONDS",
    0.90,
)
ARM_CATCHER_PLANNER_WIGGLE_FADE_SECONDS = positive_float_env(
    "DEMO_MUJOCO_ARM_CATCHER_PLANNER_WIGGLE_FADE_SECONDS",
    0.55,
)
ARM_CATCHER_PLANNER_MAX_JOINT_STEP = positive_float_env(
    "DEMO_MUJOCO_ARM_CATCHER_PLANNER_MAX_JOINT_STEP",
    0.30,
)
ARM_CATCHER_CAPTURE_MAX_FRAME_MOUNT_DELTA = float_env(
    "DEMO_MUJOCO_ARM_CATCHER_CAPTURE_MAX_FRAME_MOUNT_DELTA",
    0.060,
)
ARM_CATCHER_CAPTURE_MAX_FRAME_MOUNT_ACCELERATION_DELTA = float_env(
    "DEMO_MUJOCO_ARM_CATCHER_CAPTURE_MAX_FRAME_MOUNT_ACCELERATION_DELTA",
    0.090,
)
ARM_CATCHER_FIXED_INITIAL_MOUNT_POSITION = (
    min(
        max(scale_linear(2.40, ARM_CATCHER_SCENE_SCALE), ARM_CATCHER_WORKSPACE_MIN[0]),
        ARM_CATCHER_WORKSPACE_MAX[0],
    ),
    min(
        max(ARM_CATCHER_MOUNT_BACK_Y, ARM_CATCHER_WORKSPACE_MIN[1]),
        ARM_CATCHER_WORKSPACE_MAX[1],
    ),
    min(
        max(scale_linear(2.30, ARM_CATCHER_SCENE_SCALE), ARM_CATCHER_WORKSPACE_MIN[2]),
        ARM_CATCHER_WORKSPACE_MAX[2],
    ),
)
# The reach-constrained z band (see arm_catcher_reachable_mount_z) caps the
# vertical leg of the catch motion at ~0.05 scene units, so the old 0.60
# threshold rejected geometrically valid captures whose target x lay near the
# fixed initial mount x.
ARM_CATCHER_CAPTURE_MIN_MOUNT_TRAVEL = positive_float_env(
    "DEMO_MUJOCO_ARM_CATCHER_CAPTURE_MIN_MOUNT_TRAVEL",
    scale_linear(0.40, ARM_CATCHER_SCENE_SCALE),
)
ARM_CATCHER_SCREEN_RIGHT_CAPTURE_FRACTION = fraction_env(
    "DEMO_MUJOCO_ARM_CATCHER_SCREEN_RIGHT_CAPTURE_FRACTION",
    0.35,
)
# Panning the camera +0.9 in x (2026-07 sweep) shifted launch points left on
# screen by ~0.20 NDC, so the old 0.18 threshold would have made the
# screen-right requirement far stricter than intended; 0.0 preserves the
# original spatial semantics under the new framing.
ARM_CATCHER_SCREEN_RIGHT_MIN_X = float_env(
    "DEMO_MUJOCO_ARM_CATCHER_SCREEN_RIGHT_MIN_X",
    0.0,
)
ARM_CATCHER_LAUNCH_TARGET_Y_RADIUS_FACTOR = float_env(
    "DEMO_MUJOCO_ARM_CATCHER_LAUNCH_TARGET_Y_RADIUS_FACTOR",
    ARM_CATCHER_CAPTURE_FRONT_MARGIN / ARM_CATCHER_BALL_RADIUS,
)
ARM_CATCHER_LAUNCH_TARGET_Y_OFFSET = (
    ARM_CATCHER_LAUNCH_TARGET_Y_RADIUS_FACTOR * ARM_CATCHER_BALL_RADIUS
)
ARM_CATCHER_JOINT_LIMITS = [
    (-2.61799, 2.61799),
    (0.0, 2.96706),
    (-2.87979, 0.0),
    (-1.51844, 1.51844),
    (-1.34390, 1.34390),
    (-2.79253, 2.79253),
]
ARM_CATCHER_MATERIAL = demo_generate.material_from_hex("#2E3542", roughness=0.86, specular=0.06)
ARM_CATCHER_JOINT_MATERIAL = demo_generate.material_from_hex(
    "#566273", roughness=0.82, specular=0.08
)
# Back to the flat red ball 5267eab used. The red/cyan checker made the tumble
# legible, but it reads as a different object; the spin below is kept because it
# is part of the tuned bounce behaviour even though it is now invisible.
ARM_CATCHER_BALL_MATERIAL = demo_generate.accent_material(APPROACH_BALL_OBJECT_COLOR)
# Spin off the throw, in rad/s per axis. The ball is a sphere, so the spin has no
# effect on the free-flight path; it only shows the checker pattern tumbling and
# couples into the ground/net contacts. +-6 rad/s is roughly the rolling rate
# (v/r) of a ball at these launch speeds, so it reads as a hand toss.
ARM_CATCHER_BALL_INITIAL_SPIN_RANGE = (-6.0, 6.0)
# Floor and wall friction (previously frictionless). Matches
# ARM_PADDLE_GROUND_LANDING_FRICTION; build_approach_ball_ground pairs it with
# small rolling/spinning friction so the ball spins up on the bounce rather than
# sliding.
ARM_CATCHER_GROUND_LANDING_FRICTION = 0.25
UNIFORM_LATERAL_FRICTION = demo_generate.UNIFORM_LATERAL_FRICTION
UNIFORM_LINEAR_DAMPING = demo_generate.UNIFORM_LINEAR_DAMPING
UNIFORM_ANGULAR_DAMPING = demo_generate.UNIFORM_ANGULAR_DAMPING
# Keep arm_gripper_ball on the pre-sweep camera; only arm_catcher_ball moved.
ARM_GRIPPER_CAMERA_LOCATION = (0.35, -8.70, 8.20)
ARM_GRIPPER_CAMERA_TARGET = (0.80, -0.40, 1.15)
ARM_GRIPPER_CAMERA_FOVY_DEGREES = ARM_CATCHER_CAMERA_FOVY_DEGREES
ARM_GRIPPER_GROUND_HALF_EXTENTS = ARM_CATCHER_REFERENCE_GROUND_HALF_EXTENTS
ARM_GRIPPER_CATCH_FRONT_Y = -2.70 + 0.225
ARM_GRIPPER_BASE_POSITION = (2.90, -4.12, 0.05)
ARM_GRIPPER_BASE_EULER = (0.0, 0.0, math.pi)
ARM_GRIPPER_HOME_QPOS = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
ARM_GRIPPER_JOINT_NAMES = [
    "ur5e_shoulder_pan_joint",
    "ur5e_shoulder_lift_joint",
    "ur5e_elbow_joint",
    "ur5e_wrist_1_joint",
    "ur5e_wrist_2_joint",
    "ur5e_wrist_3_joint",
]
ARM_GRIPPER_ACTUATOR_NAMES = [
    "ur5e_shoulder_pan",
    "ur5e_shoulder_lift",
    "ur5e_elbow",
    "ur5e_wrist_1",
    "ur5e_wrist_2",
    "ur5e_wrist_3",
]
ARM_GRIPPER_JOINT_LIMITS = [
    (-6.28319, 6.28319),
    (-6.28319, 6.28319),
    (-3.1415, 3.1415),
    (-6.28319, 6.28319),
    (-6.28319, 6.28319),
    (-6.28319, 6.28319),
]
ARM_GRIPPER_ROBOTIQ_JOINT_NAMES = [
    "rq_right_driver_joint",
    "rq_right_coupler_joint",
    "rq_right_spring_link_joint",
    "rq_right_follower_joint",
    "rq_left_driver_joint",
    "rq_left_coupler_joint",
    "rq_left_spring_link_joint",
    "rq_left_follower_joint",
]
ARM_GRIPPER_WORKSPACE_MIN = (-1.15, ARM_GRIPPER_CATCH_FRONT_Y - 0.12, 0.45)
ARM_GRIPPER_WORKSPACE_MAX = (3.05, ARM_GRIPPER_CATCH_FRONT_Y + 0.08, 0.85)
ARM_GRIPPER_INITIAL_TARGET_X_RANGE = (1.15, 2.15)
ARM_GRIPPER_INITIAL_TARGET_Y_RANGE = (
    ARM_GRIPPER_CATCH_FRONT_Y - 0.04,
    ARM_GRIPPER_CATCH_FRONT_Y + 0.04,
)
ARM_GRIPPER_INITIAL_TARGET_Z_RANGE = (0.50, 0.68)
ARM_GRIPPER_ACTION_SCALE_XYZ = (0.085, 0.045, 0.065)
ARM_GRIPPER_CATCH_Y_RANGE = (ARM_GRIPPER_CATCH_FRONT_Y - 0.030, ARM_GRIPPER_CATCH_FRONT_Y + 0.030)
ARM_GRIPPER_CAPTURE_CROSS_FRAME_RANGE = (40.0, 48.0)
ARM_GRIPPER_TRUNCATED_CROSS_FRAME_RANGE = CATCHER_BALL_TRUNCATED_CROSS_FRAME_RANGE
ARM_GRIPPER_CAPTURE_TARGET_X_RANGE = (0.25, 1.30)
ARM_GRIPPER_MISS_TARGET_X_LEFT_RANGE = (-2.00, -1.45)
ARM_GRIPPER_MISS_TARGET_X_RIGHT_RANGE = (3.35, 3.85)
ARM_GRIPPER_TARGET_Z_RANGE = (0.58, 0.82)
ARM_GRIPPER_START_X_RANGE = (0.50, 0.50)
ARM_GRIPPER_START_Y_RANGE = (3.8, 4.8)
ARM_GRIPPER_START_HEIGHT_ABOVE_RADIUS_RANGE = (0.15, 1.90)
ARM_GRIPPER_FIRST_BOUNCE_TIME_RANGE = (0.65, 1.75)
ARM_GRIPPER_BALL_RADIUS = 0.035
ARM_GRIPPER_BALL_MASS_RANGE = (0.045, 0.090)


def choose_dynamic_shape(episode_index):
    if DYNAMIC_SHAPE_MODE == "ball_only":
        return "ball"
    if DYNAMIC_SHAPE_MODE == "cube_only":
        return "cube"
    return "ball" if episode_index % 2 == 0 else "cube"


def random_unit_vector(rng):
    z = rng.uniform(-1.0, 1.0)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    radius = math.sqrt(max(1.0 - z * z, 0.0))
    return (radius * math.cos(angle), radius * math.sin(angle), z)


def sample_dynamic_object(rng, episode_index, position, mass, initial_velocity):
    shape_name = choose_dynamic_shape(episode_index)
    if shape_name == "ball":
        radius = (
            FIXED_BALL_RADIUS if FIXED_BALL_RADIUS is not None else rng.uniform(*BALL_RADIUS_RANGE)
        )
        spec = demo_generate.make_sphere(
            "moving_object",
            radius=radius,
            position=position,
            mass=mass,
            material=demo_generate.accent_material(DYNAMIC_OBJECT_COLOR),
            restitution=ELASTIC_RESTITUTION,
            lateral_friction=UNIFORM_LATERAL_FRICTION,
            linear_damping=UNIFORM_LINEAR_DAMPING,
            angular_damping=UNIFORM_ANGULAR_DAMPING,
            initial_velocity=initial_velocity,
            initial_angular_velocity=(0.0, 0.0, 0.0),
            lock_rotation=True,
        )
        return spec, shape_name, (radius, radius, radius), radius

    half_extent = (
        FIXED_CUBE_HALF_EXTENT
        if FIXED_CUBE_HALF_EXTENT is not None
        else rng.uniform(*CUBE_HALF_EXTENT_RANGE)
    )
    angular_axis = random_unit_vector(rng)
    angular_speed = rng.uniform(1.25, 3.25)
    initial_angular_velocity = tuple(component * angular_speed for component in angular_axis)
    spec = demo_generate.make_cube(
        "moving_object",
        half_extent=half_extent,
        position=position,
        mass=mass,
        rotation_euler=(
            rng.uniform(-0.8, 0.8),
            rng.uniform(-0.8, 0.8),
            rng.uniform(-0.8, 0.8),
        ),
        material=demo_generate.accent_material(DYNAMIC_OBJECT_COLOR),
        restitution=ELASTIC_RESTITUTION,
        lateral_friction=UNIFORM_LATERAL_FRICTION,
        linear_damping=UNIFORM_LINEAR_DAMPING,
        angular_damping=UNIFORM_ANGULAR_DAMPING,
        initial_velocity=initial_velocity,
        initial_angular_velocity=initial_angular_velocity,
    )
    bounding_radius = math.sqrt(3.0) * half_extent
    return spec, shape_name, (half_extent, half_extent, half_extent), bounding_radius


def build_episode_assignments(task_name):
    assignments = []
    task_index = TASKS.index(task_name)
    full_batches, remainder = divmod(CANONICAL_EPISODES_PER_TASK, GRAVITY_BATCH_EPISODES)
    batch_count = full_batches + int(remainder > 0)

    for batch_index in range(batch_count):
        batch_rng = random.Random(BASE_SEED + 1000 * task_index + 7919 * batch_index)
        batch_assignments = []
        for _ in range(TRAIN_EPISODES_PER_BATCH):
            gravity = max(batch_rng.gauss(TRAIN_GAUSSIAN_MEAN, TRAIN_GAUSSIAN_STD), TRAIN_G_MIN)
            batch_assignments.append(("train", gravity))

        for test_index in range(TEST_EPISODES_PER_BATCH):
            gravity = TEST_GRAVITIES[(task_index + batch_index + test_index) % len(TEST_GRAVITIES)]
            batch_assignments.append(("test", gravity))

        batch_rng.shuffle(batch_assignments)
        assignments.extend(batch_assignments)

    return assignments[:CANONICAL_EPISODES_PER_TASK]


def build_common_metadata(
    task_name, description, gravity, episode_index, split_name, phys, task_params, state_anchor
):
    return {
        "scenario": task_name,
        "description": description,
        "fps": FPS,
        "num_frames": FRAMES_PER_EPISODE,
        "duration_seconds": DURATION_SECONDS,
        "step_rate": STEP_RATE,
        "gravity": [0.0, 0.0, -gravity],
        "episode_index": episode_index,
        "split_name": split_name,
        "split_id": SPLIT_TO_ID[split_name],
        "state_schema": STATE_SCHEMA,
        "action_schema": task_action_schema(task_name),
        "gravity_action_indices": [0],
        "catcher_action_indices": (
            list(range(1, len(task_action_schema(task_name))))
            if task_has_catcher_fields(task_name)
            else []
        ),
        "paddle_action_indices": (
            list(range(1, len(task_action_schema(task_name))))
            if task_has_paddle_fields(task_name)
            else []
        ),
        "action_alignment": (
            "action[t] is applied between frame[t] and frame[t+1]; "
            "the final action row is repeated and unused for transition."
        ),
        "task_param_meanings": TASK_PARAM_MEANINGS[task_name],
        "episode_phys": phys,
        "task_params": task_params,
        "state_anchor": state_anchor,
    }


def open_box_wall_material():
    return demo_generate.material_from_hex("#FBFCFF", roughness=0.97, specular=0.02)


def open_box_grid_material():
    return demo_generate.material_from_hex("#E1E5EC", roughness=0.98, specular=0.0)


def approach_ball_ground_material():
    return demo_generate.material_from_hex("#FBFCFF", roughness=0.97, specular=0.02)


def approach_ball_grid_material():
    return demo_generate.material_from_hex("#D5DDE8", roughness=0.98, specular=0.0)


def make_render_only_box(name, half_extents, position, material):
    spec = demo_generate.make_box(
        name,
        half_extents=half_extents,
        position=position,
        mass=0.0,
        material=material,
        restitution=ELASTIC_RESTITUTION,
        lateral_friction=UNIFORM_LATERAL_FRICTION,
    )
    spec["render_only"] = True
    return spec


def grid_offsets(half_extent, spacing):
    max_index = int(math.floor((half_extent - 0.05) / spacing))
    return [index * spacing for index in range(-max_index, max_index + 1)]


def build_open_box_grid():
    center_x, center_y, center_z = OPEN_BOX_CENTER
    half_x, half_y, half_z = OPEN_BOX_INNER_HALF_EXTENTS
    line_t = OPEN_BOX_GRID_HALF_THICKNESS
    wall_line_t = OPEN_BOX_WALL_GRID_HALF_THICKNESS
    surface_offset = OPEN_BOX_GRID_SURFACE_OFFSET
    grid_material = open_box_grid_material()
    grid_specs = []

    # Back wall interior face: x-z plane at max_y.
    back_y = center_y + half_y - surface_offset
    for offset_x in grid_offsets(half_x, OPEN_BOX_GRID_SPACING):
        grid_specs.append(
            make_render_only_box(
                f"open_box_back_grid_x_{len(grid_specs):03d}",
                (wall_line_t, line_t, half_z),
                (center_x + offset_x, back_y, center_z),
                grid_material,
            )
        )
    for offset_z in grid_offsets(half_z, OPEN_BOX_GRID_SPACING):
        grid_specs.append(
            make_render_only_box(
                f"open_box_back_grid_z_{len(grid_specs):03d}",
                (half_x, line_t, wall_line_t),
                (center_x, back_y, center_z + offset_z),
                grid_material,
            )
        )

    # Side wall interior faces: y-z planes at min_x and max_x.
    for side_name, side_x, x_depth in (
        ("min_x", center_x - half_x + surface_offset, line_t),
        ("max_x", center_x + half_x - surface_offset, line_t),
    ):
        for offset_y in grid_offsets(half_y, OPEN_BOX_GRID_SPACING):
            grid_specs.append(
                make_render_only_box(
                    f"open_box_{side_name}_grid_y_{len(grid_specs):03d}",
                    (x_depth, wall_line_t, half_z),
                    (side_x, center_y + offset_y, center_z),
                    grid_material,
                )
            )
        for offset_z in grid_offsets(half_z, OPEN_BOX_GRID_SPACING):
            grid_specs.append(
                make_render_only_box(
                    f"open_box_{side_name}_grid_z_{len(grid_specs):03d}",
                    (x_depth, half_y, wall_line_t),
                    (side_x, center_y, center_z + offset_z),
                    grid_material,
                )
            )

    # Floor interior face: x-y plane at min_z.
    floor_z = center_z - half_z + surface_offset
    for offset_x in grid_offsets(half_x, OPEN_BOX_GRID_SPACING):
        grid_specs.append(
            make_render_only_box(
                f"open_box_floor_grid_x_{len(grid_specs):03d}",
                (line_t, half_y, line_t),
                (center_x + offset_x, center_y, floor_z),
                grid_material,
            )
        )
    for offset_y in grid_offsets(half_y, OPEN_BOX_GRID_SPACING):
        grid_specs.append(
            make_render_only_box(
                f"open_box_floor_grid_y_{len(grid_specs):03d}",
                (half_x, line_t, line_t),
                (center_x, center_y + offset_y, floor_z),
                grid_material,
            )
        )

    return grid_specs


def build_open_box_walls():
    center_x, center_y, center_z = OPEN_BOX_CENTER
    half_x, half_y, half_z = OPEN_BOX_INNER_HALF_EXTENTS
    wall_half_t = 0.5 * OPEN_BOX_WALL_THICKNESS
    wall_material = open_box_wall_material()
    wall_kwargs = {
        "mass": 0.0,
        "restitution": ELASTIC_RESTITUTION,
        "lateral_friction": UNIFORM_LATERAL_FRICTION,
    }
    physical_walls = [
        demo_generate.make_box(
            "open_box_min_x_wall",
            half_extents=(
                wall_half_t,
                half_y + OPEN_BOX_WALL_THICKNESS,
                half_z + OPEN_BOX_WALL_THICKNESS,
            ),
            position=(center_x - half_x - wall_half_t, center_y, center_z),
            material=wall_material,
            **wall_kwargs,
        ),
        demo_generate.make_box(
            "open_box_max_x_wall",
            half_extents=(
                wall_half_t,
                half_y + OPEN_BOX_WALL_THICKNESS,
                half_z + OPEN_BOX_WALL_THICKNESS,
            ),
            position=(center_x + half_x + wall_half_t, center_y, center_z),
            material=wall_material,
            **wall_kwargs,
        ),
        demo_generate.make_box(
            "open_box_back_wall",
            half_extents=(
                half_x + OPEN_BOX_WALL_THICKNESS,
                wall_half_t,
                half_z + OPEN_BOX_WALL_THICKNESS,
            ),
            position=(center_x, center_y + half_y + wall_half_t, center_z),
            material=wall_material,
            **wall_kwargs,
        ),
        demo_generate.make_box(
            "open_box_floor",
            half_extents=(
                half_x + OPEN_BOX_WALL_THICKNESS,
                half_y + OPEN_BOX_WALL_THICKNESS,
                wall_half_t,
            ),
            position=(center_x, center_y, center_z - half_z - wall_half_t),
            material=wall_material,
            **wall_kwargs,
        ),
    ]
    return physical_walls + build_open_box_grid()


def build_approach_ball_grid(
    ground_half_extents=APPROACH_BALL_GROUND_HALF_EXTENTS,
    *,
    ground_center=APPROACH_BALL_GROUND_CENTER,
    side_wall_height=APPROACH_BALL_SIDE_WALL_HEIGHT,
    grid_spacing=APPROACH_BALL_GRID_SPACING,
    grid_half_thickness=APPROACH_BALL_GRID_HALF_THICKNESS,
    grid_surface_offset=APPROACH_BALL_GRID_SURFACE_OFFSET,
    ground_grid_depth_half_thickness=None,
    wall_grid_depth_half_thickness=None,
):
    center_x, center_y, center_z = ground_center
    half_x, half_y, half_z = ground_half_extents
    line_t = grid_half_thickness
    ground_depth_t = (
        line_t
        if ground_grid_depth_half_thickness is None
        else max(1.0e-6, float(ground_grid_depth_half_thickness))
    )
    wall_depth_t = (
        line_t
        if wall_grid_depth_half_thickness is None
        else max(1.0e-6, float(wall_grid_depth_half_thickness))
    )
    grid_z = center_z + half_z + grid_surface_offset
    wall_half_z = 0.5 * side_wall_height
    wall_center_z = center_z + half_z + wall_half_z
    grid_material = approach_ball_grid_material()
    grid_specs = []

    # Ground top face: x-y plane.
    for offset_x in grid_offsets(half_x, grid_spacing):
        grid_specs.append(
            make_render_only_box(
                f"approach_ball_grid_x_{len(grid_specs):03d}",
                (line_t, half_y, ground_depth_t),
                (center_x + offset_x, center_y, grid_z),
                grid_material,
            )
        )
    for offset_y in grid_offsets(half_y, grid_spacing):
        grid_specs.append(
            make_render_only_box(
                f"approach_ball_grid_y_{len(grid_specs):03d}",
                (half_x, line_t, ground_depth_t),
                (center_x, center_y + offset_y, grid_z),
                grid_material,
            )
        )

    # Left and right wall interior faces: y-z planes.
    for side_name, side_x in (
        ("left", center_x - half_x + grid_surface_offset),
        ("right", center_x + half_x - grid_surface_offset),
    ):
        for offset_y in grid_offsets(half_y, grid_spacing):
            grid_specs.append(
                make_render_only_box(
                    f"approach_ball_{side_name}_wall_grid_y_{len(grid_specs):03d}",
                    (wall_depth_t, line_t, wall_half_z),
                    (side_x, center_y + offset_y, wall_center_z),
                    grid_material,
                )
            )
        for offset_z in grid_offsets(wall_half_z, grid_spacing):
            grid_specs.append(
                make_render_only_box(
                    f"approach_ball_{side_name}_wall_grid_z_{len(grid_specs):03d}",
                    (wall_depth_t, half_y, line_t),
                    (side_x, center_y, wall_center_z + offset_z),
                    grid_material,
                )
            )

    # Back wall interior face: x-z plane at the far positive-y end.
    back_y = center_y + half_y - grid_surface_offset
    for offset_x in grid_offsets(half_x, grid_spacing):
        grid_specs.append(
            make_render_only_box(
                f"approach_ball_back_wall_grid_x_{len(grid_specs):03d}",
                (line_t, wall_depth_t, wall_half_z),
                (center_x + offset_x, back_y, wall_center_z),
                grid_material,
            )
        )
    for offset_z in grid_offsets(wall_half_z, grid_spacing):
        grid_specs.append(
            make_render_only_box(
                f"approach_ball_back_wall_grid_z_{len(grid_specs):03d}",
                (half_x, wall_depth_t, line_t),
                (center_x, back_y, wall_center_z + offset_z),
                grid_material,
            )
        )

    return grid_specs


def build_approach_ball_ground(
    ground_half_extents=APPROACH_BALL_GROUND_HALF_EXTENTS,
    *,
    ground_center=APPROACH_BALL_GROUND_CENTER,
    side_wall_height=APPROACH_BALL_SIDE_WALL_HEIGHT,
    side_wall_thickness=APPROACH_BALL_SIDE_WALL_THICKNESS,
    grid_spacing=APPROACH_BALL_GRID_SPACING,
    grid_half_thickness=APPROACH_BALL_GRID_HALF_THICKNESS,
    grid_surface_offset=APPROACH_BALL_GRID_SURFACE_OFFSET,
    contact_solref=None,
    wall_natural_material=None,
    ground_grid_depth_half_thickness=None,
    wall_grid_depth_half_thickness=None,
    ground_bottom_extension=0.0,
    ground_lateral_extension=0.0,
    landing_friction=None,
):
    center_x, center_y, center_z = ground_center
    half_x, half_y, half_z = ground_half_extents
    wall_half_t = 0.5 * side_wall_thickness
    wall_half_z = 0.5 * side_wall_height
    wall_center_z = center_z + half_z + wall_half_z
    wall_material = approach_ball_ground_material()
    # High friction stops a knocked-off failure ball skating across the floor
    # (the default floor is frictionless), applied to the floor and walls alike.
    surface_friction = (
        UNIFORM_LATERAL_FRICTION if landing_friction is None else float(landing_friction)
    )
    friction_kwargs = (
        {}
        if landing_friction is None
        else {
            "rolling_friction": 0.005,
            "spinning_friction": 0.005,
        }
    )
    # Extend the ground collision box downward (top face unchanged) and far out
    # laterally so a planned-failure ball that is knocked hard out of the arena
    # still lands on the floor (floor_failure) instead of skipping past a thin
    # floor between timesteps or free-falling forever past the arena footprint.
    # Wall placement below keeps using the original (un-extended) extents.
    ground_collision_half_z = half_z + 0.5 * ground_bottom_extension
    ground_collision_center_z = center_z - 0.5 * ground_bottom_extension
    ground = demo_generate.make_box(
        "approach_ball_ground",
        half_extents=(
            half_x + ground_lateral_extension,
            half_y + ground_lateral_extension,
            ground_collision_half_z,
        ),
        position=(center_x, center_y, ground_collision_center_z),
        mass=0.0,
        material=wall_material,
        restitution=ELASTIC_RESTITUTION,
        lateral_friction=surface_friction,
        contact_solref=contact_solref,
        **friction_kwargs,
    )
    wall_kwargs = {
        "mass": 0.0,
        "restitution": ELASTIC_RESTITUTION,
        "lateral_friction": surface_friction,
        "contact_solref": contact_solref,
        **friction_kwargs,
    }
    boundary_walls = [
        demo_generate.make_box(
            "approach_ball_left_wall",
            half_extents=(wall_half_t, half_y, wall_half_z),
            position=(center_x - half_x - wall_half_t, center_y, wall_center_z),
            material=wall_material,
            **wall_kwargs,
        ),
        demo_generate.make_box(
            "approach_ball_right_wall",
            half_extents=(wall_half_t, half_y, wall_half_z),
            position=(center_x + half_x + wall_half_t, center_y, wall_center_z),
            material=wall_material,
            **wall_kwargs,
        ),
        demo_generate.make_box(
            "approach_ball_back_wall",
            half_extents=(half_x + side_wall_thickness, wall_half_t, wall_half_z),
            position=(center_x, center_y + half_y + wall_half_t, wall_center_z),
            material=wall_material,
            **wall_kwargs,
        ),
    ]
    if wall_natural_material:
        for wall_spec in boundary_walls:
            wall_spec["natural_material"] = wall_natural_material
    return [ground, *boundary_walls] + build_approach_ball_grid(
        ground_half_extents,
        ground_center=ground_center,
        side_wall_height=side_wall_height,
        grid_spacing=grid_spacing,
        grid_half_thickness=grid_half_thickness,
        grid_surface_offset=grid_surface_offset,
        ground_grid_depth_half_thickness=ground_grid_depth_half_thickness,
        wall_grid_depth_half_thickness=wall_grid_depth_half_thickness,
    )


def sample_open_box_initial_velocity(rng, episode_index):
    vx = rng.choice([-1.0, 1.0]) * rng.uniform(*OPEN_BOX_INITIAL_X_SPEED_RANGE)
    if episode_index % 4 == 0:
        vy = -rng.uniform(*OPEN_BOX_INITIAL_Y_SPEED_CAMERA_RANGE)
    else:
        vy = rng.uniform(*OPEN_BOX_INITIAL_Y_SPEED_INWARD_RANGE)
    vz = rng.uniform(*OPEN_BOX_INITIAL_VERTICAL_SPEED_RANGE)
    return vx, vy, vz


def build_open_box_episode(rng, gravity, episode_index, split_name):
    mass = rng.uniform(0.5, 2.0)
    initial_velocity = sample_open_box_initial_velocity(rng, episode_index)
    dynamic_object, shape_name, size_xyz, bounding_radius = sample_dynamic_object(
        rng,
        episode_index,
        position=OPEN_BOX_CENTER,
        mass=mass,
        initial_velocity=initial_velocity,
    )

    center_x, center_y, center_z = OPEN_BOX_CENTER
    half_x, half_y, half_z = OPEN_BOX_INNER_HALF_EXTENTS
    margin = bounding_radius + OPEN_BOX_OBJECT_EDGE_MARGIN
    floor_z = center_z - half_z

    x_min = center_x - half_x + margin
    x_max = center_x + half_x - margin
    y_max = center_y + half_y - margin
    if initial_velocity[1] < 0.0:
        y_min = max(center_y + 0.45, center_y - half_y + margin + 1.15)
    else:
        y_min = center_y - half_y + margin + 0.70
    y_min = min(y_min, y_max - 0.10)
    z_min = floor_z + margin + 0.10
    z_max = min(center_z + 1.15, center_z + half_z - margin - 0.15)
    z_min = min(z_min, z_max - 0.10)

    dynamic_object["position"] = [
        rng.uniform(x_min, x_max),
        rng.uniform(y_min, y_max),
        rng.uniform(z_min, z_max),
    ]

    phys = [
        gravity,
        mass,
        size_xyz[0],
        size_xyz[1],
        size_xyz[2],
        SHAPE_ID_MAP[shape_name],
        OPEN_BOX_INNER_HALF_EXTENTS[0],
        OPEN_BOX_INNER_HALF_EXTENTS[1],
        OPEN_BOX_INNER_HALF_EXTENTS[2],
        OPEN_BOX_WALL_THICKNESS,
    ]
    metadata = build_common_metadata(
        "open_box",
        "A single 3D object bouncing inside a box with the camera-side wall and top wall open.",
        gravity,
        episode_index,
        split_name,
        phys,
        [*OPEN_BOX_INNER_HALF_EXTENTS, OPEN_BOX_WALL_THICKNESS],
        list(OPEN_BOX_CENTER),
    )
    return {
        "metadata": metadata,
        "camera": make_camera(
            OPEN_BOX_CAMERA_LOCATION,
            OPEN_BOX_CAMERA_TARGET,
            OPEN_BOX_CAMERA_FOVY_DEGREES,
        ),
        "static_objects": build_open_box_walls(),
        "dynamic_objects": [dynamic_object],
    }


def sample_approach_ball_initial_state(rng, gravity, radius):
    start_x = rng.uniform(*APPROACH_BALL_REAL_START_X_RANGE)
    start_y_high = min(
        APPROACH_BALL_REAL_START_Y_RANGE[1],
        APPROACH_BALL_REAL_GROUND_HALF_EXTENTS[1] - radius - APPROACH_BALL_REAL_BACK_WALL_CLEARANCE,
    )
    start_y = rng.uniform(APPROACH_BALL_REAL_START_Y_RANGE[0], start_y_high)
    start_z = radius + rng.uniform(*APPROACH_BALL_REAL_HEIGHT_ABOVE_RADIUS_RANGE)
    target_x = rng.uniform(*APPROACH_BALL_REAL_NEAR_TARGET_X_RANGE)
    target_y = rng.uniform(*APPROACH_BALL_REAL_NEAR_TARGET_Y_RANGE)
    target_time = rng.uniform(*APPROACH_BALL_NEAR_TARGET_TIME_RANGE)
    first_bounce_time = rng.uniform(*approach_ball_first_bounce_time_range(gravity))

    vx = (target_x - start_x) / target_time + rng.uniform(*APPROACH_BALL_REAL_VX_NOISE_RANGE)
    vy = (target_y - start_y) / target_time + rng.uniform(*APPROACH_BALL_REAL_VY_NOISE_RANGE)
    vz = (
        radius - start_z + 0.5 * gravity * first_bounce_time * first_bounce_time
    ) / first_bounce_time
    # An elastic rebound returns the ball center to start_z + vz^2/(2*g), so
    # at low gravity the bounce-time formula above demands a downward vz
    # whose rebound apex grows like 1/g and climbs out of the camera frame.
    # Clamping |vz| to the energy budget of the apex ceiling bounds every
    # pre- and post-bounce apex at any gravity; at g=0 the limit is zero and
    # the episode degenerates to a flat, bounce-free approach.
    vz_limit = math.sqrt(2.0 * gravity * max(APPROACH_BALL_REAL_MAX_BOUNCE_APEX_Z - start_z, 0.0))
    vz = min(max(vz, -vz_limit), vz_limit)
    return (start_x, start_y, start_z), (vx, vy, vz)


def choose_catcher_ball_planned_outcome(episode_index):
    return CATCHER_BALL_OUTCOME_CYCLE[episode_index % len(CATCHER_BALL_OUTCOME_CYCLE)]


def choose_arm_catcher_ball_planned_outcome(episode_index):
    return ARM_CATCHER_BALL_OUTCOME_CYCLE[episode_index % len(ARM_CATCHER_BALL_OUTCOME_CYCLE)]


def require_arm_catcher_screen_right_capture(episode_index):
    seed = BASE_SEED + 1009 * (int(episode_index) + 1) + 17041
    return deterministic_fraction(seed) < ARM_CATCHER_SCREEN_RIGHT_CAPTURE_FRACTION


def arm_catcher_capture_episode_ordinal(episode_index):
    if choose_arm_catcher_ball_planned_outcome(episode_index) != "capture":
        return None
    return sum(
        1
        for index in range(int(episode_index))
        if choose_arm_catcher_ball_planned_outcome(index) == "capture"
    )


def arm_catcher_capture_episode_count(episode_count):
    return sum(
        1
        for index in range(int(episode_count))
        if choose_arm_catcher_ball_planned_outcome(index) == "capture"
    )


def sample_arm_catcher_capture_target_x(rng, episode_index):
    low, high = ARM_CATCHER_CAPTURE_TARGET_X_RANGE
    capture_count = arm_catcher_capture_episode_count(EPISODES_PER_TASK)
    ordinal = (
        arm_catcher_capture_episode_ordinal(episode_index) if episode_index is not None else None
    )
    if ordinal is None or capture_count <= 0:
        target_x = rng.uniform(low, high)
        return target_x, {
            "mode": "uniform_random",
            "range": [low, high],
        }

    stride = 7
    while math.gcd(stride, capture_count) != 1:
        stride += 2
    bucket = (ordinal * stride + capture_count // 3) % capture_count
    jitter = rng.random()
    unit = (bucket + jitter) / capture_count
    target_x = low + unit * (high - low)
    return target_x, {
        "mode": "stratified_uniform_capture_episode",
        "range": [low, high],
        "ordinal": ordinal,
        "count": capture_count,
        "bucket": bucket,
        "stride": stride,
        "jitter": jitter,
        "unit": unit,
    }


def sample_arm_catcher_capture_target_z_hint(rng, episode_index, target_x):
    """Stratified-uniform crossing-height hint across capture episodes.

    Mirrors sample_arm_catcher_capture_target_x with a different stride and
    offset so height and lateral strata stay decorrelated; without
    stratification the realized catch heights skew high because low crossings
    are harder for the launch sampler to satisfy and get re-rolled.
    """
    low = arm_catcher_reachable_mount_z(target_x)
    high = ARM_CATCHER_CAPTURE_MAX_TARGET_Z_HINT
    capture_count = arm_catcher_capture_episode_count(EPISODES_PER_TASK)
    ordinal = (
        arm_catcher_capture_episode_ordinal(episode_index) if episode_index is not None else None
    )
    if ordinal is None or capture_count <= 0:
        hint = rng.uniform(low, high)
        return hint, {
            "mode": "uniform_random",
            "range": [low, high],
        }

    stride = 11
    while math.gcd(stride, capture_count) != 1:
        stride += 2
    bucket = (ordinal * stride + (2 * capture_count) // 5) % capture_count
    jitter = rng.random()
    unit = (bucket + jitter) / capture_count
    hint = low + unit * (high - low)
    return hint, {
        "mode": "stratified_uniform_capture_episode",
        "range": [low, high],
        "ordinal": ordinal,
        "count": capture_count,
        "bucket": bucket,
        "stride": stride,
        "jitter": jitter,
        "unit": unit,
    }


def choose_catcher_ball_policy(planned_outcome):
    if planned_outcome == "miss":
        return "noisy_delayed"
    return "expert"


def sample_catcher_miss_bias(rng):
    return [
        rng.choice([-1.0, 1.0]) * rng.uniform(1.45, 1.95),
        rng.uniform(-0.20, 0.20),
    ]


def vertical_position_after_bounces(start_z, vz, gravity, radius, target_time):
    z = float(start_z)
    velocity = float(vz)
    step_count = max(1, int(math.ceil(target_time / 0.005)))
    dt = target_time / step_count
    for _ in range(step_count):
        velocity -= gravity * dt
        z += velocity * dt
        if z < radius:
            z = radius
            if velocity < 0.0:
                velocity = -velocity
    return z


def vertical_state_and_peak_after_bounces(start_z, vz, gravity, radius, target_time):
    z = float(start_z)
    velocity = float(vz)
    peak_z = z
    step_count = max(1, int(math.ceil(target_time / 0.005)))
    dt = target_time / step_count
    for _ in range(step_count):
        velocity -= gravity * dt
        z += velocity * dt
        if z < radius:
            z = radius
            if velocity < 0.0:
                velocity = -velocity
        peak_z = max(peak_z, z)
    return z, velocity, peak_z


def vertical_position_and_peak_after_bounces(start_z, vz, gravity, radius, target_time):
    z, _velocity, peak_z = vertical_state_and_peak_after_bounces(
        start_z, vz, gravity, radius, target_time
    )
    return z, peak_z


def direct_projectile_peak_z(start_z, vz, gravity, target_time):
    if gravity <= 1.0e-9 or vz <= 0.0:
        return max(float(start_z), float(start_z) + float(vz) * float(target_time))
    apex_time = min(max(float(vz) / float(gravity), 0.0), float(target_time))
    return float(start_z) + float(vz) * apex_time - 0.5 * float(gravity) * apex_time * apex_time


def sample_catcher_vertical_velocity(rng, gravity, radius, target_time, planned_outcome):
    if planned_outcome == "truncated":
        min_z, max_z = CATCHER_BALL_TRUNCATED_TARGET_Z_RANGE
    else:
        min_z, max_z = CATCHER_BALL_TARGET_Z_RANGE
    bounce_range = (0.95, 1.40) if planned_outcome != "truncated" else (1.15, 1.75)
    height_range = (1.15, 2.35) if planned_outcome != "truncated" else (1.35, 2.65)

    for _ in range(96):
        start_z = radius + rng.uniform(*height_range)
        first_bounce_time = rng.uniform(*bounce_range)
        vz = (
            radius - start_z + 0.5 * gravity * first_bounce_time * first_bounce_time
        ) / first_bounce_time
        target_z = vertical_position_after_bounces(
            start_z,
            vz,
            gravity,
            radius,
            target_time,
        )
        if min_z <= target_z <= max_z:
            return start_z, vz, target_z, first_bounce_time

    first_bounce_time = sum(bounce_range) * 0.5
    start_z = radius + sum(height_range) * 0.5
    vz = (
        radius - start_z + 0.5 * gravity * first_bounce_time * first_bounce_time
    ) / first_bounce_time
    target_z = vertical_position_after_bounces(start_z, vz, gravity, radius, target_time)
    return start_z, vz, target_z, first_bounce_time


def sample_catcher_ball_initial_state(rng, gravity, radius, planned_outcome):
    start_x = rng.uniform(-1.05, 1.05)
    start_y = rng.uniform(7.15, 8.45)
    if planned_outcome == "truncated":
        target_frame = rng.uniform(*CATCHER_BALL_TRUNCATED_CROSS_FRAME_RANGE)
        target_x = rng.uniform(*CATCHER_BALL_TRUNCATED_TARGET_X_RANGE)
    else:
        target_frame = rng.uniform(*CATCHER_BALL_CAPTURE_CROSS_FRAME_RANGE)
        target_x = rng.uniform(*CATCHER_BALL_TARGET_X_RANGE)
    target_time = target_frame / FPS
    target_y = CATCHER_BALL_CATCH_FRONT_Y + rng.uniform(-0.035, 0.035)
    start_z, vz, target_z, first_bounce_time = sample_catcher_vertical_velocity(
        rng,
        gravity,
        radius,
        target_time,
        planned_outcome,
    )
    vx = (target_x - start_x) / target_time + rng.uniform(-0.035, 0.035)
    vy = (target_y - start_y) / target_time + rng.uniform(-0.025, 0.025)
    launch_metadata = {
        "planned_cross_frame": target_frame,
        "planned_cross_time": target_time,
        "capture_cross_time_range": list(ARM_CATCHER_CAPTURE_CROSS_TIME_RANGE),
        "capture_cross_frame_range": list(ARM_CATCHER_CAPTURE_CROSS_FRAME_RANGE),
        "planned_cross_y": target_y,
        "planned_cross_x": target_x,
        "estimated_cross_z": target_z,
        "first_bounce_time": first_bounce_time,
        "target_x_range": (
            list(CATCHER_BALL_TRUNCATED_TARGET_X_RANGE)
            if planned_outcome == "truncated"
            else list(CATCHER_BALL_TARGET_X_RANGE)
        ),
        "target_z_range": (
            list(CATCHER_BALL_TRUNCATED_TARGET_Z_RANGE)
            if planned_outcome == "truncated"
            else list(CATCHER_BALL_TARGET_Z_RANGE)
        ),
    }
    return (start_x, start_y, start_z), (vx, vy, vz), launch_metadata


def arm_catcher_reachable_mount_z(target_x):
    # With the net normal held along +y, the end effector cannot go below
    # ~1.85 reference units anywhere in the capture x-range; lower targets
    # wedge joint 4 at its limit. Keep the mount-z hint in the measured band.
    target_x = min(
        max(float(target_x), ARM_CATCHER_CAPTURE_TARGET_X_RANGE[0]),
        ARM_CATCHER_CAPTURE_TARGET_X_RANGE[1],
    )
    first_break_x = scale_linear(1.30, ARM_CATCHER_SCENE_SCALE)
    second_break_x = scale_linear(2.75, ARM_CATCHER_SCENE_SCALE)
    high_z = scale_linear(2.12, ARM_CATCHER_SCENE_SCALE)
    low_z = scale_linear(1.92, ARM_CATCHER_SCENE_SCALE)
    slope = 0.138
    if target_x < first_break_x:
        center_z = high_z
    elif target_x < second_break_x:
        center_z = high_z - slope * (target_x - first_break_x)
    else:
        center_z = low_z
    return min(max(center_z, ARM_CATCHER_TARGET_Z_RANGE[0]), ARM_CATCHER_TARGET_Z_RANGE[1])


def sample_arm_catcher_vertical_velocity(
    rng,
    gravity,
    radius,
    target_time,
    planned_outcome,
    target_z_hint=None,
):
    min_z, max_z = ARM_CATCHER_TARGET_Z_RANGE
    if planned_outcome == "capture":
        min_z = max(min_z, ARM_CATCHER_CAPTURE_MIN_TARGET_Z)
    if gravity <= 1.0e-9 and planned_outcome in {"capture", "miss"}:
        target_z = float(target_z_hint) if target_z_hint is not None else rng.uniform(min_z, max_z)
        target_z = min(max(target_z, min_z), max_z)
        start_z = radius + rng.uniform(*ARM_CATCHER_DIRECT_CAPTURE_START_HEIGHT_RANGE)
        vz = (target_z - start_z + 0.5 * gravity * target_time * target_time) / target_time
        peak_z = direct_projectile_peak_z(start_z, vz, gravity, target_time)
        return start_z, vz, target_z, None, peak_z

    bounce_low, bounce_high = arm_catcher_first_bounce_time_range(gravity)
    if planned_outcome != "truncated":
        bounce_high = min(bounce_high, max(bounce_low, target_time - 0.12))

    height_low, height_high = ARM_CATCHER_START_HEIGHT_ABOVE_RADIUS_RANGE
    if planned_outcome == "capture" and target_z_hint is not None:
        # Elastic arcs never peak below the start height and the catch
        # crossing sits near an apex, so a start above the hint makes the
        # hint unreachable; clamping keeps low-crossing hints feasible.
        height_high = min(
            height_high,
            max(height_low + 1.0e-3, float(target_z_hint) + 0.05 - radius),
        )
        # A long first-bounce time from a low start implies a hard upward
        # launch (high apex); allow bounce times down to the free-drop time
        # of the hint height so low arcs exist in the sample space.
        drop_time = math.sqrt(2.0 * max(float(target_z_hint) - radius, 0.01) / max(gravity, 1.0e-6))
        bounce_low = min(bounce_low, max(0.9 * drop_time, 0.12))
        bounce_high = max(bounce_high, bounce_low + 0.05)
    for _ in range(512):
        start_z = radius + rng.uniform(height_low, height_high)
        first_bounce_time = rng.uniform(bounce_low, bounce_high)
        vz = (
            radius - start_z + 0.5 * gravity * first_bounce_time * first_bounce_time
        ) / first_bounce_time
        target_z, target_vz, peak_z = vertical_state_and_peak_after_bounces(
            start_z,
            vz,
            gravity,
            radius,
            target_time,
        )
        close_to_hint = target_z_hint is None or abs(
            target_z - float(target_z_hint)
        ) <= scale_linear(0.28, ARM_CATCHER_SCENE_SCALE)
        # A planned capture must cross the catch plane near a bounce apex:
        # at 16 fps the caught check samples once per frame, and a fast-falling
        # ball transits the capture window between samples, making the catch
        # undetectable regardless of arm placement.
        slow_enough_to_catch = (
            planned_outcome != "capture" or abs(target_vz) <= ARM_CATCHER_CAPTURE_MAX_CROSS_VZ
        )
        if (
            min_z <= target_z <= max_z
            and peak_z <= ARM_CATCHER_MAX_TRAJECTORY_Z
            and close_to_hint
            and slow_enough_to_catch
        ):
            return start_z, vz, target_z, first_bounce_time, peak_z

    first_bounce_time = 0.5 * (bounce_low + bounce_high)
    start_z = radius + ARM_CATCHER_START_HEIGHT_ABOVE_RADIUS_RANGE[0]
    vz = (
        radius - start_z + 0.5 * gravity * first_bounce_time * first_bounce_time
    ) / first_bounce_time
    target_z, peak_z = vertical_position_and_peak_after_bounces(
        start_z, vz, gravity, radius, target_time
    )
    return start_z, vz, target_z, first_bounce_time, peak_z


_ARM_CATCHER_LAUNCH_CAMERA = None


def arm_catcher_launch_camera():
    global _ARM_CATCHER_LAUNCH_CAMERA
    if _ARM_CATCHER_LAUNCH_CAMERA is None:
        _ARM_CATCHER_LAUNCH_CAMERA = make_camera(
            ARM_CATCHER_CAMERA_LOCATION,
            ARM_CATCHER_CAMERA_TARGET,
            ARM_CATCHER_CAMERA_FOVY_DEGREES,
        )
    return _ARM_CATCHER_LAUNCH_CAMERA


def arm_catcher_launch_prepass_extent(position, velocity, gravity, radius, horizon_time):
    """Max projected ball extent (NDC units, 1.0 = frame edge) before the catch.

    Mirrors the dataset's camera_prepass_ball_in_frame verification so launch
    sampling can reject trajectories that would leave the frame.
    """
    camera = arm_catcher_launch_camera()
    location = camera["location"]
    right = camera["image_right_world"]
    up = camera["image_up_world"]
    forward = camera["optical_axis_world"]
    tangent = math.tan(math.radians(float(camera["fovy"])) * 0.5)
    x, y, z = (float(value) for value in position)
    vx, vy, vz = (float(value) for value in velocity)
    dt = 0.005
    steps = max(1, int(math.ceil(horizon_time / dt)))
    dt = horizon_time / steps
    frame_interval = max(1, int(round(1.0 / (FPS * dt))))
    worst = 0.0
    for step in range(steps + 1):
        if step % frame_interval == 0 or step == steps:
            rel = (x - location[0], y - location[1], z - location[2])
            depth = sum(rel[index] * forward[index] for index in range(3))
            if depth <= 1.0e-9:
                return float("inf")
            px = sum(rel[index] * right[index] for index in range(3)) / (depth * tangent)
            py = sum(rel[index] * up[index] for index in range(3)) / (depth * tangent)
            margin = radius / (depth * tangent)
            worst = max(worst, abs(px) + margin, abs(py) + margin)
        vz -= gravity * dt
        x += vx * dt
        y += vy * dt
        z += vz * dt
        if z < radius:
            z = radius
            if vz < 0.0:
                vz = -vz
    return worst


# Do not lower this for any outcome subset: tried splitting a stricter
# margin onto miss/truncated launches only (they don't carry the capture
# side's screen-right constraint) to fight analytic-vs-real bounce drift,
# but for several miss gravities (-9 to -20) *no* candidate within 1536
# resamples scores below ~0.9 analytically, so any tighter bound just
# exhausts the loop and returns a far worse fallback (seen up to 3.3) --
# 2026-07 arm_catcher_ball reposition round. 0.92 remains the best tested
# balance; the one remaining marginal camera_prepass_ball_in_frame case
# (~1.05 real vs 1.0 limit, on a single "miss" episode) is a narrow,
# pre-existing edge case, not something this margin can fix.
ARM_CATCHER_LAUNCH_MAX_PREPASS_EXTENT = float_env(
    "DEMO_MUJOCO_ARM_CATCHER_LAUNCH_MAX_PREPASS_EXTENT",
    0.92,
)


def sample_arm_catcher_ball_initial_state(
    rng,
    gravity,
    radius,
    planned_outcome,
    episode_index=None,
):
    prepass_limit = ARM_CATCHER_LAUNCH_MAX_PREPASS_EXTENT
    result = None
    for _ in range(1536):
        result = _sample_arm_catcher_ball_launch_once(
            rng,
            gravity,
            radius,
            planned_outcome,
            episode_index=episode_index,
        )
        position, velocity, launch_metadata = result
        horizon_time = min(
            float(launch_metadata["planned_cross_time"]),
            (FRAMES_PER_EPISODE - 1) / FPS,
        )
        extent = arm_catcher_launch_prepass_extent(
            position,
            velocity,
            gravity,
            radius,
            horizon_time,
        )
        launch_metadata["prepass_screen_extent"] = extent
        if extent > prepass_limit:
            continue
        if planned_outcome == "capture":
            # Capture episodes also require a screen-right launch; checking
            # here keeps the retry loop from burning full attempts on starts
            # the acceptance test would reject anyway.
            screen_x = projected_screen_x(position, arm_catcher_launch_camera())
            if screen_x < ARM_CATCHER_SCREEN_RIGHT_MIN_X:
                continue
        return result
    return result


def _sample_arm_catcher_ball_launch_once(
    rng,
    gravity,
    radius,
    planned_outcome,
    episode_index=None,
):
    start_x = rng.uniform(*ARM_CATCHER_START_X_RANGE)
    start_y = rng.uniform(*ARM_CATCHER_START_Y_RANGE)
    target_x_sampling = None
    target_z_hint_sampling = None
    if planned_outcome == "truncated":
        target_frame = rng.uniform(*CATCHER_BALL_TRUNCATED_CROSS_FRAME_RANGE)
        target_x = rng.uniform(*ARM_CATCHER_TRUNCATED_TARGET_X_RANGE)
    elif planned_outcome == "miss":
        target_frame = rng.uniform(*ARM_CATCHER_CAPTURE_CROSS_FRAME_RANGE)
        if rng.random() < 0.5:
            target_x = rng.uniform(*ARM_CATCHER_MISS_TARGET_X_LEFT_RANGE)
        else:
            target_x = rng.uniform(*ARM_CATCHER_MISS_TARGET_X_RIGHT_RANGE)
    else:
        target_frame = rng.uniform(*ARM_CATCHER_CAPTURE_CROSS_FRAME_RANGE)
        target_x, target_x_sampling = sample_arm_catcher_capture_target_x(
            rng,
            episode_index,
        )
    target_time = target_frame / FPS
    target_y_jitter = (
        ARM_CATCHER_CAPTURE_TARGET_Y_JITTER
        if planned_outcome == "capture"
        else ARM_CATCHER_TARGET_Y_JITTER
    )
    target_y = (
        ARM_CATCHER_CATCH_FRONT_Y
        + ARM_CATCHER_LAUNCH_TARGET_Y_OFFSET
        + rng.uniform(
            -target_y_jitter,
            target_y_jitter,
        )
    )

    min_z, max_z = ARM_CATCHER_TARGET_Z_RANGE
    target_z_hint = None
    if planned_outcome == "capture":
        target_z_hint, target_z_hint_sampling = sample_arm_catcher_capture_target_z_hint(
            rng,
            episode_index,
            target_x,
        )
        target_z_hint += rng.uniform(
            -ARM_CATCHER_TARGET_Z_HINT_JITTER,
            ARM_CATCHER_TARGET_Z_HINT_JITTER,
        )
    start_z, vz, target_z, first_bounce_time, peak_z = sample_arm_catcher_vertical_velocity(
        rng,
        gravity,
        radius,
        target_time,
        planned_outcome,
        target_z_hint=target_z_hint,
    )

    vx_noise_range = (
        ARM_CATCHER_CAPTURE_VX_NOISE_RANGE
        if planned_outcome == "capture"
        else ARM_CATCHER_MISS_VX_NOISE_RANGE
    )
    vy_noise_range = (
        ARM_CATCHER_CAPTURE_VY_NOISE_RANGE
        if planned_outcome == "capture"
        else ARM_CATCHER_MISS_VY_NOISE_RANGE
    )
    vx = (target_x - start_x) / target_time + rng.uniform(*vx_noise_range)
    vy = (target_y - start_y) / target_time + rng.uniform(*vy_noise_range)
    launch_metadata = {
        "planned_cross_frame": target_frame,
        "planned_cross_time": target_time,
        "planned_cross_y": target_y,
        "planned_cross_x": target_x,
        "estimated_cross_z": target_z,
        "estimated_peak_z": peak_z,
        "first_bounce_time": first_bounce_time,
        "direct_projectile_to_reachable_arm_mount": planned_outcome == "capture",
        "direct_projectile_launch": first_bounce_time is None,
        "ground_bounce_launch": first_bounce_time is not None,
        "low_arc_truncated_launch": planned_outcome == "truncated",
        "arm_reachable_launch": True,
        "planned_outcome": planned_outcome,
        "ball_speed_scale": ARM_CATCHER_BALL_SPEED_SCALE,
        "start_y_multiplier": ARM_CATCHER_START_Y_MULTIPLIER,
        "cross_frame_multiplier": ARM_CATCHER_CROSS_FRAME_MULTIPLIER,
        "max_trajectory_z": ARM_CATCHER_MAX_TRAJECTORY_Z,
        "vx_noise_range": list(vx_noise_range),
        "vy_noise_range": list(vy_noise_range),
        "target_x_range": (
            list(ARM_CATCHER_CAPTURE_TARGET_X_RANGE)
            if planned_outcome == "capture"
            else [
                list(ARM_CATCHER_MISS_TARGET_X_LEFT_RANGE),
                list(ARM_CATCHER_MISS_TARGET_X_RIGHT_RANGE),
            ]
            if planned_outcome == "miss"
            else list(ARM_CATCHER_TRUNCATED_TARGET_X_RANGE)
        ),
        "target_x_sampling": target_x_sampling,
        "target_z_hint_sampling": target_z_hint_sampling,
        "target_z_range": [min_z, max_z],
    }
    return (start_x, start_y, start_z), (vx, vy, vz), launch_metadata


def sample_arm_gripper_ball_initial_state(rng, gravity, radius, planned_outcome):
    start_x = rng.uniform(*ARM_GRIPPER_START_X_RANGE)
    start_y = rng.uniform(*ARM_GRIPPER_START_Y_RANGE)
    if planned_outcome == "truncated":
        target_frame = rng.uniform(*ARM_GRIPPER_TRUNCATED_CROSS_FRAME_RANGE)
        target_x = rng.uniform(*ARM_GRIPPER_CAPTURE_TARGET_X_RANGE)
    elif planned_outcome == "miss":
        target_frame = rng.uniform(*ARM_GRIPPER_CAPTURE_CROSS_FRAME_RANGE)
        if rng.random() < 0.5:
            target_x = rng.uniform(*ARM_GRIPPER_MISS_TARGET_X_LEFT_RANGE)
        else:
            target_x = rng.uniform(*ARM_GRIPPER_MISS_TARGET_X_RIGHT_RANGE)
    else:
        target_frame = rng.uniform(*ARM_GRIPPER_CAPTURE_CROSS_FRAME_RANGE)
        target_x = rng.uniform(*ARM_GRIPPER_CAPTURE_TARGET_X_RANGE)

    target_time = target_frame / FPS
    target_y = rng.uniform(*ARM_GRIPPER_CATCH_Y_RANGE)

    min_z, max_z = ARM_GRIPPER_TARGET_Z_RANGE
    for _ in range(128):
        start_z = radius + rng.uniform(*ARM_GRIPPER_START_HEIGHT_ABOVE_RADIUS_RANGE)
        first_bounce_time = rng.uniform(*ARM_GRIPPER_FIRST_BOUNCE_TIME_RANGE)
        vz = (
            radius - start_z + 0.5 * gravity * first_bounce_time * first_bounce_time
        ) / first_bounce_time
        target_z = vertical_position_after_bounces(
            start_z,
            vz,
            gravity,
            radius,
            target_time,
        )
        if min_z <= target_z <= max_z:
            break
    else:
        first_bounce_time = 1.15
        start_z = radius + 1.55
        vz = (
            radius - start_z + 0.5 * gravity * first_bounce_time * first_bounce_time
        ) / first_bounce_time
        target_z = vertical_position_after_bounces(start_z, vz, gravity, radius, target_time)

    vx = (target_x - start_x) / target_time + rng.uniform(-0.025, 0.025)
    vy = (target_y - start_y) / target_time + rng.uniform(-0.025, 0.025)
    launch_metadata = {
        "planned_cross_frame": target_frame,
        "planned_cross_time": target_time,
        "planned_cross_y": target_y,
        "planned_cross_x": target_x,
        "estimated_cross_z": target_z,
        "first_bounce_time": first_bounce_time,
        "arm_catcher_style_launch": True,
        "hybrid_gripper_scale_ball": True,
        "target_x_range": (
            list(ARM_GRIPPER_CAPTURE_TARGET_X_RANGE)
            if planned_outcome == "capture"
            else [
                list(ARM_GRIPPER_MISS_TARGET_X_LEFT_RANGE),
                list(ARM_GRIPPER_MISS_TARGET_X_RIGHT_RANGE),
            ]
            if planned_outcome == "miss"
            else list(ARM_GRIPPER_CAPTURE_TARGET_X_RANGE)
        ),
        "target_z_range": [min_z, max_z],
    }
    return (start_x, start_y, start_z), (vx, vy, vz), launch_metadata


def build_approach_ball_episode(rng, gravity, episode_index, split_name):
    mass = rng.uniform(0.5, 2.0)
    radius = FIXED_BALL_RADIUS if FIXED_BALL_RADIUS is not None else APPROACH_BALL_REAL_BALL_RADIUS
    initial_position, initial_velocity = sample_approach_ball_initial_state(
        rng,
        gravity,
        radius,
    )
    dynamic_object = demo_generate.make_sphere(
        "moving_object",
        radius=radius,
        position=initial_position,
        mass=mass,
        material=demo_generate.accent_material(APPROACH_BALL_OBJECT_COLOR),
        restitution=ELASTIC_RESTITUTION,
        lateral_friction=UNIFORM_LATERAL_FRICTION,
        linear_damping=UNIFORM_LINEAR_DAMPING,
        angular_damping=UNIFORM_ANGULAR_DAMPING,
        initial_velocity=initial_velocity,
        initial_angular_velocity=(0.0, 0.0, 0.0),
        lock_rotation=True,
    )

    phys = [
        gravity,
        mass,
        radius,
        radius,
        radius,
        SHAPE_ID_MAP["ball"],
        APPROACH_BALL_REAL_GROUND_HALF_EXTENTS[0],
        APPROACH_BALL_REAL_GROUND_HALF_EXTENTS[1],
        APPROACH_BALL_REAL_SIDE_WALL_HEIGHT,
        APPROACH_BALL_REAL_SIDE_WALL_THICKNESS,
    ]
    metadata = build_common_metadata(
        "approach_ball",
        "A single red ball thrown from far down a gridded ground plane toward the camera.",
        gravity,
        episode_index,
        split_name,
        phys,
        [
            APPROACH_BALL_REAL_GROUND_HALF_EXTENTS[0],
            APPROACH_BALL_REAL_GROUND_HALF_EXTENTS[1],
            APPROACH_BALL_REAL_SIDE_WALL_HEIGHT,
            APPROACH_BALL_REAL_CAMERA_ELEVATION_DEGREES,
        ],
        list(APPROACH_BALL_REAL_CAMERA_TARGET),
    )
    metadata["scene_scale"] = APPROACH_BALL_SCENE_SCALE
    return {
        "metadata": metadata,
        "camera": make_camera(
            APPROACH_BALL_REAL_CAMERA_LOCATION,
            APPROACH_BALL_REAL_CAMERA_TARGET,
            APPROACH_BALL_CAMERA_FOVY_DEGREES,
        ),
        "static_objects": build_approach_ball_ground(
            APPROACH_BALL_REAL_GROUND_HALF_EXTENTS,
            ground_center=APPROACH_BALL_REAL_GROUND_CENTER,
            side_wall_height=APPROACH_BALL_REAL_SIDE_WALL_HEIGHT,
            side_wall_thickness=APPROACH_BALL_REAL_SIDE_WALL_THICKNESS,
            grid_spacing=APPROACH_BALL_REAL_GRID_SPACING,
            grid_half_thickness=APPROACH_BALL_REAL_GRID_HALF_THICKNESS,
            grid_surface_offset=APPROACH_BALL_REAL_GRID_SURFACE_OFFSET,
            contact_solref=APPROACH_BALL_GROUND_CONTACT_SOLREF,
            ground_grid_depth_half_thickness=APPROACH_BALL_REAL_GRID_DEPTH_HALF_THICKNESS,
            wall_grid_depth_half_thickness=APPROACH_BALL_REAL_GRID_DEPTH_HALF_THICKNESS,
        ),
        "dynamic_objects": [dynamic_object],
    }


def make_paddle_parts(base_half_extents=PADDLE_BASE_HALF_EXTENTS):
    return [
        {
            "name": "paddle_base",
            "half_extents": list(base_half_extents),
            "offset": [0.0, 0.0, 0.0],
            "material": PADDLE_MATERIAL,
            "collision": True,
        },
    ]


PADDLE_BALL_TASK_PROFILE = {
    "ball_radius": PADDLE_BALL_RADIUS,
    "ball_mass": PADDLE_BALL_MASS,
    "ground_half_extents": PADDLE_GROUND_HALF_EXTENTS,
    "ground_center": APPROACH_BALL_GROUND_CENTER,
    "side_wall_height": APPROACH_BALL_SIDE_WALL_HEIGHT,
    "side_wall_thickness": APPROACH_BALL_SIDE_WALL_THICKNESS,
    "grid_spacing": APPROACH_BALL_GRID_SPACING,
    "grid_half_thickness": APPROACH_BALL_GRID_HALF_THICKNESS,
    "grid_surface_offset": APPROACH_BALL_GRID_SURFACE_OFFSET,
    "grid_depth_half_thickness": None,
    "workspace_min": PADDLE_WORKSPACE_MIN,
    "workspace_max": PADDLE_WORKSPACE_MAX,
    "action_scale_xyz": PADDLE_ACTION_SCALE_XYZ,
    "base_half_extents": PADDLE_BASE_HALF_EXTENTS,
    "initial_position": PADDLE_INITIAL_POSITION,
    "tilt_center_xy": (0.0, -3.0),
    "initial_x_range": PADDLE_BALL_INITIAL_X_RANGE,
    "initial_y_range": PADDLE_BALL_INITIAL_Y_RANGE,
    "initial_z_range": PADDLE_BALL_INITIAL_Z_RANGE,
    "initial_vx_range": PADDLE_BALL_INITIAL_VX_RANGE,
    "initial_vy_range": PADDLE_BALL_INITIAL_VY_RANGE,
    "initial_vz_range": PADDLE_BALL_INITIAL_VZ_RANGE,
    "camera_location": PADDLE_BALL_CAMERA_LOCATION,
    "camera_target": PADDLE_BALL_CAMERA_TARGET,
    "camera_fovy": PADDLE_BALL_CAMERA_FOVY_DEGREES,
    "ground_contact_solref": None,
    "ball_contact_solref": None,
    "paddle_contact_solref": None,
    "paddle_collision_mass": None,
    "wall_natural_material": None,
    "servo": None,
    "policy_config_overrides": None,
}

# Command-channel perturbations for robustness augmentation (recorded on the
# paddle command, applied to every arm_paddle_ball episode). Small per-frame
# Gaussian jitter keeps the trajectory off the perfect limit cycle; the sudden
# single-frame errors displace the paddle target briefly so the recorded policy
# corrections that follow demonstrate recovery. Magnitudes are in normalized
# action units (xyz delta) and radians (tilt).
ARM_PADDLE_COMMAND_NOISE = {
    "delta_std_xy": 0.09,
    "delta_std_z": 0.05,
    "tilt_phi_std": 0.04,
    "tilt_theta_std": 0.03,
    "sudden_error_count_range": (2, 3),
    "sudden_error_frame_fraction_range": (0.15, 0.85),
    "sudden_error_magnitude": 0.5,
    "sudden_error_z_magnitude": 0.18,
    # Planned-failure "loss of control": the paddle keeps bouncing the (low-apex,
    # slow) ball but its hit position drifts off over `fail_pos_ramp_frames` so
    # it slides off the ball and the ball drops freely to the floor, with mild
    # position + tilt jitter for erratic mis-hits. No persistent tilt bias and no
    # vertical jitter -- a tilt/downward jerk on a fast high-gravity ball slams
    # it through the floor or far out of view.
    "fail_pos_bias": 0.14,
    "fail_pos_ramp_frames": 12,
    "fail_tilt_bias_theta": 0.0,
    "fail_jitter_delta_std_xy": 0.05,
    "fail_jitter_delta_std_z": 0.0,
    "fail_jitter_tilt_std": 0.0,
}

ARM_PADDLE_TASK_PROFILE = {
    "ball_radius": ARM_PADDLE_BALL_RADIUS,
    "ball_mass": ARM_PADDLE_BALL_MASS,
    "ground_half_extents": ARM_PADDLE_GROUND_HALF_EXTENTS,
    "ground_center": ARM_PADDLE_GROUND_CENTER,
    "side_wall_height": ARM_PADDLE_SIDE_WALL_HEIGHT,
    "side_wall_thickness": ARM_PADDLE_SIDE_WALL_THICKNESS,
    "grid_spacing": ARM_PADDLE_GRID_SPACING,
    "grid_half_thickness": ARM_PADDLE_GRID_HALF_THICKNESS,
    "grid_surface_offset": ARM_PADDLE_GRID_SURFACE_OFFSET,
    "grid_depth_half_thickness": ARM_PADDLE_GRID_DEPTH_HALF_THICKNESS,
    "workspace_min": ARM_PADDLE_WORKSPACE_MIN,
    "workspace_max": ARM_PADDLE_WORKSPACE_MAX,
    "action_scale_xyz": ARM_PADDLE_ACTION_SCALE_XYZ,
    "base_half_extents": ARM_PADDLE_PADDLE_HALF_EXTENTS,
    "initial_position": ARM_PADDLE_INITIAL_POSITION,
    "tilt_center_xy": ARM_PADDLE_TILT_CENTER_XY,
    "initial_x_range": ARM_PADDLE_BALL_INITIAL_X_RANGE,
    "initial_y_range": ARM_PADDLE_BALL_INITIAL_Y_RANGE,
    "initial_z_range": ARM_PADDLE_BALL_INITIAL_Z_RANGE,
    "initial_vx_range": ARM_PADDLE_BALL_INITIAL_VX_RANGE,
    "initial_vy_range": ARM_PADDLE_BALL_INITIAL_VY_RANGE,
    "initial_vz_range": ARM_PADDLE_BALL_INITIAL_VZ_RANGE,
    "camera_location": ARM_PADDLE_CAMERA_LOCATION,
    "camera_target": ARM_PADDLE_CAMERA_TARGET,
    "camera_fovy": ARM_PADDLE_CAMERA_FOVY_DEGREES,
    "ground_contact_solref": ARM_PADDLE_GROUND_LANDING_SOLREF,
    "ground_landing_friction": ARM_PADDLE_GROUND_LANDING_FRICTION,
    "ball_contact_solref": ARM_PADDLE_GROUND_CONTACT_SOLREF,
    "paddle_contact_solref": ARM_PADDLE_PADDLE_CONTACT_SOLREF,
    "paddle_collision_mass": ARM_PADDLE_PADDLE_COLLISION_MASS,
    # The close-up camera sees the wall inner faces head-on; the emissive
    # variant lifts them to the pixel brightness arm_catcher walls render at.
    "wall_natural_material": "natural_wall_lit_mat",
    "servo": ARM_PADDLE_PADDLE_SERVO,
    "policy_config_overrides": ARM_PADDLE_POLICY_CONFIG_OVERRIDES,
    "command_noise": ARM_PADDLE_COMMAND_NOISE,
    # A thick, wide ground collision box: a hard-mishandled failure ball can be
    # knocked out of the arena fast enough to skip a thin floor or fly past its
    # footprint; the extensions keep it landing on the floor (no tunnel, no
    # free-fall) even when it leaves the frame.
    "ground_bottom_extension": 2.0,
    "ground_lateral_extension": 80.0,
}


def build_paddle_ball_episode(
    rng,
    gravity,
    episode_index,
    split_name,
    task_name="paddle_ball",
    profile=None,
):
    profile = profile if profile is not None else PADDLE_BALL_TASK_PROFILE
    is_zonly = task_name == "paddle_ball_zonly"
    radius = profile["ball_radius"]
    mass = profile["ball_mass"]
    ball_friction = PADDLE_BALL_ZONLY_FRICTION if is_zonly else PADDLE_BALL_FRICTION
    paddle_friction = PADDLE_ZONLY_FRICTION if is_zonly else PADDLE_FRICTION
    lock_ball_rotation = bool(is_zonly)
    tilt_enabled = not is_zonly
    initial_position = (
        rng.uniform(*profile["initial_x_range"]),
        rng.uniform(*profile["initial_y_range"]),
        rng.uniform(*profile["initial_z_range"]),
    )
    initial_velocity_ranges = (
        (
            PADDLE_BALL_ZONLY_INITIAL_VX_RANGE,
            PADDLE_BALL_ZONLY_INITIAL_VY_RANGE,
            PADDLE_BALL_ZONLY_INITIAL_VZ_RANGE,
        )
        if is_zonly
        else (
            profile["initial_vx_range"],
            profile["initial_vy_range"],
            profile["initial_vz_range"],
        )
    )
    initial_velocity = tuple(
        rng.uniform(*velocity_range) for velocity_range in initial_velocity_ranges
    )
    initial_angular_velocity = (
        rng.uniform(*PADDLE_BALL_INITIAL_WX_RANGE),
        rng.uniform(*PADDLE_BALL_INITIAL_WY_RANGE),
        rng.uniform(*PADDLE_BALL_INITIAL_WZ_RANGE),
    )
    dynamic_object = demo_generate.make_sphere(
        "moving_object",
        radius=radius,
        position=initial_position,
        mass=mass,
        material=PADDLE_BALL_MATERIAL,
        restitution=PADDLE_BALL_RESTITUTION,
        lateral_friction=ball_friction,
        linear_damping=UNIFORM_LINEAR_DAMPING,
        angular_damping=UNIFORM_ANGULAR_DAMPING,
        initial_velocity=initial_velocity,
        initial_angular_velocity=initial_angular_velocity,
        lock_rotation=lock_ball_rotation,
        contact_solref=profile["ball_contact_solref"],
    )

    phys = [
        gravity,
        mass,
        radius,
        radius,
        radius,
        SHAPE_ID_MAP["ball"],
        profile["ground_half_extents"][0],
        profile["ground_half_extents"][1],
        profile["side_wall_height"],
        profile["side_wall_thickness"],
    ]
    metadata = build_common_metadata(
        task_name,
        "A red ball bouncing on an XYZ-controlled paddle.",
        gravity,
        episode_index,
        split_name,
        phys,
        [
            profile["ground_half_extents"][0],
            profile["ground_half_extents"][1],
            profile["side_wall_height"],
            APPROACH_BALL_CAMERA_ELEVATION_DEGREES,
        ],
        list(profile["camera_target"]),
    )
    metadata.update(
        {
            "policy_type": "expert",
            "success": False,
            "random_seed": BASE_SEED + 10000 * TASKS.index(task_name) + episode_index,
            "physics": {
                "gravity": gravity,
                "ball_radius": radius,
                "ball_mass": mass,
                "ball_restitution": PADDLE_BALL_RESTITUTION,
                "ball_friction": ball_friction,
                "paddle_restitution": PADDLE_RESTITUTION,
                "paddle_friction": paddle_friction,
                "ball_rotation_locked": lock_ball_rotation,
            },
            "fixed_physics": {
                "ball_radius": radius,
                "ball_mass": mass,
                "ball_restitution": PADDLE_BALL_RESTITUTION,
                "ball_friction": ball_friction,
                "paddle_restitution": PADDLE_RESTITUTION,
                "paddle_friction": paddle_friction,
                "ball_rotation_locked": lock_ball_rotation,
            },
            "ball_initial_state": {
                "position": list(initial_position),
                "linear_velocity": list(initial_velocity),
                "angular_velocity": list(initial_angular_velocity),
                "radius": radius,
                "mass": mass,
            },
            "paddle_initial_state": {
                "position": list(profile["initial_position"]),
                "target_position": list(profile["initial_position"]),
                "orientation_phi": 0.0,
                "orientation_theta": 0.0,
                "target_phi": 0.0,
                "target_theta": 0.0,
                "workspace_min": list(profile["workspace_min"]),
                "workspace_max": list(profile["workspace_max"]),
            },
            "paddle_type": (
                "xyz_tilt_controlled_flat_paddle" if tilt_enabled else "xyz_controlled_flat_paddle"
            ),
            "ball_rotation_locked": lock_ball_rotation,
            "latch_after_capture": False,
            "action_space": (
                "normalized_paddle_delta_xyz_plus_absolute_tilt"
                if tilt_enabled
                else "normalized_paddle_delta_xyz"
            ),
            "normalized_action_bounds": [-1.0, 1.0],
            "action_scale_xyz": list(profile["action_scale_xyz"]),
            "tilt_enabled": tilt_enabled,
            "tilt_angle_units": "radians",
            "tilt_phi_range": [-math.pi, math.pi],
            "tilt_theta_range": [0.0, PADDLE_MAX_TILT_THETA],
            "paddle_max_tilt_theta": PADDLE_MAX_TILT_THETA,
            "paddle_policy_max_tilt_theta": PADDLE_POLICY_MAX_TILT_THETA,
            "camera_parameters": {
                "location": list(profile["camera_location"]),
                "target": list(profile["camera_target"]),
                "fovy": profile["camera_fovy"],
            },
        }
    )
    paddle = {
        "name": "paddle",
        "initial_position": list(profile["initial_position"]),
        "initial_target_position": list(profile["initial_position"]),
        "initial_target_phi": 0.0,
        "initial_target_theta": 0.0,
        "workspace_min": list(profile["workspace_min"]),
        "workspace_max": list(profile["workspace_max"]),
        "action_scale_xyz": list(profile["action_scale_xyz"]),
        "tilt_enabled": tilt_enabled,
        "max_tilt_theta": PADDLE_MAX_TILT_THETA,
        "policy_max_tilt_theta": PADDLE_POLICY_MAX_TILT_THETA,
        "tilt_center_xy": list(profile["tilt_center_xy"]),
        "base_half_extents": list(profile["base_half_extents"]),
        "rim_height": PADDLE_RIM_HEIGHT,
        "rim_half_thickness": PADDLE_RIM_HALF_THICKNESS,
        "material": PADDLE_MATERIAL,
        "restitution": PADDLE_RESTITUTION,
        "lateral_friction": paddle_friction,
        "policy_type": "expert",
        "parts": make_paddle_parts(profile["base_half_extents"]),
    }
    if profile["paddle_contact_solref"] is not None:
        paddle["contact_solref"] = profile["paddle_contact_solref"]
    if profile["paddle_collision_mass"] is not None:
        paddle["collision_part_mass"] = profile["paddle_collision_mass"]
    if profile["servo"] is not None:
        paddle["servo"] = dict(profile["servo"])
    if profile["policy_config_overrides"] is not None:
        overrides = dict(profile["policy_config_overrides"])
        init_coeff = overrides.pop("adaptive_trim_init_log_coeff", None)
        init_ref_g = overrides.pop("adaptive_trim_init_ref_g", 1.0)
        if init_coeff is not None and gravity > 1.0e-9:
            limit = abs(overrides.get("adaptive_strike_trim_limit", 0.05))
            initial_trim = float(init_coeff) * math.log(gravity / float(init_ref_g))
            overrides["adaptive_strike_trim"] = min(max(initial_trim, -limit), limit)
        paddle["policy_config_overrides"] = overrides
    if profile.get("command_noise") is not None:
        paddle["command_noise"] = dict(profile["command_noise"])
    return {
        "metadata": metadata,
        "camera": make_camera(
            profile["camera_location"],
            profile["camera_target"],
            profile["camera_fovy"],
        ),
        "static_objects": build_approach_ball_ground(
            profile["ground_half_extents"],
            ground_center=profile["ground_center"],
            side_wall_height=profile["side_wall_height"],
            side_wall_thickness=profile["side_wall_thickness"],
            grid_spacing=profile["grid_spacing"],
            grid_half_thickness=profile["grid_half_thickness"],
            grid_surface_offset=profile["grid_surface_offset"],
            contact_solref=profile["ground_contact_solref"],
            wall_natural_material=profile["wall_natural_material"],
            ground_grid_depth_half_thickness=profile.get("grid_depth_half_thickness"),
            wall_grid_depth_half_thickness=profile.get("grid_depth_half_thickness"),
            ground_bottom_extension=profile.get("ground_bottom_extension", 0.0),
            ground_lateral_extension=profile.get("ground_lateral_extension", 0.0),
            landing_friction=profile.get("ground_landing_friction"),
        ),
        "dynamic_objects": [dynamic_object],
        "paddle": paddle,
    }


def build_paddle_ball_zonly_episode(rng, gravity, episode_index, split_name):
    return build_paddle_ball_episode(
        rng,
        gravity,
        episode_index,
        split_name,
        task_name="paddle_ball_zonly",
    )


def make_arm_paddle_handle():
    direction_local = (
        0.0,
        math.cos(ARM_PADDLE_HANDLE_ANGLE),
        math.sin(ARM_PADDLE_HANDLE_ANGLE),
    )
    total_length = ARM_PADDLE_HANDLE_GRIP_DISTANCE + ARM_PADDLE_HANDLE_EMBED_DEPTH
    part = {
        "name": "paddle_handle",
        "half_extents": [
            ARM_PADDLE_HANDLE_HALF_WIDTH,
            0.5 * total_length,
            ARM_PADDLE_HANDLE_HALF_WIDTH,
        ],
        "offset": [
            ARM_PADDLE_HANDLE_ROOT_OFFSET[index] + direction_local[index] * 0.5 * total_length
            for index in range(3)
        ],
        "rotation_euler": [ARM_PADDLE_HANDLE_ANGLE, 0.0, 0.0],
        "material": ARM_PADDLE_HANDLE_MATERIAL,
        "collision": False,
        "render_only": True,
    }
    handle_spec = {
        "root_offset": list(ARM_PADDLE_HANDLE_ROOT_OFFSET),
        "direction_local": list(direction_local),
        "grip_distance": ARM_PADDLE_HANDLE_GRIP_DISTANCE,
        "embed_depth": ARM_PADDLE_HANDLE_EMBED_DEPTH,
        "half_width": ARM_PADDLE_HANDLE_HALF_WIDTH,
        "angle_radians": ARM_PADDLE_HANDLE_ANGLE,
    }
    return part, handle_spec


ARM_PADDLE_PLANNED_FAILURE_FRACTION = 0.1
ARM_PADDLE_FAILURE_MISS_FRAME_FRACTION_RANGE = (0.08, 0.16)
ARM_PADDLE_FAILURE_MISS_OFFSET_X = 0.24

# Rest-start episodes: the ball begins stationary sitting near the paddle centre
# with the paddle (and hence the ball) at a randomised workspace position; the
# policy then initiates and sustains the bounce while re-centring it. Ranges are
# in real-scale metres about the paddle centre ARM_PADDLE_TILT_CENTER_XY.
ARM_PADDLE_REST_START_FRACTION = 0.5
# Below this gravity the policy cannot lift a *resting* ball to the target apex
# within the episode (the ball barely falls, so the velocity-matched ramp strike
# has nothing to work with), so rest-start is confined to higher gravities.
ARM_PADDLE_REST_START_MIN_GRAVITY = 1.5
# Lateral spread kept small: the close-up camera frames the ball tightly, and
# the re-centring swing from a laterally offset start would otherwise carry the
# ball toward the arena walls.
ARM_PADDLE_REST_START_PADDLE_X_RANGE = (-0.08, 0.08)
ARM_PADDLE_REST_START_PADDLE_Y_OFFSET_RANGE = (-0.035, 0.035)
ARM_PADDLE_REST_START_BALL_OFFSET_RANGE = (-0.02, 0.02)
# Rest-start bounces are deliberately gentle and low (apex just above the paddle,
# vs the ~0.35 of the flight episodes). A high launch has long airtime, and
# re-centring the ball across it overshoots into the walls and ricochets; low
# bounces keep the ball near the paddle and in frame, and read naturally as the
# arm dribbling the rested ball back to centre.
ARM_PADDLE_REST_START_TARGET_APEX_Z = 0.25
ARM_PADDLE_REST_START_BALL_GAP = 0.001

_ARM_PADDLE_PLANNED_FAILURE_INDICES = None
_ARM_PADDLE_REST_START_INDICES = None


def random_unit_quaternion(rng):
    """Uniform random orientation (Shoemake) as an xyzw quaternion."""
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()
    sqrt1_minus_u1 = math.sqrt(1.0 - u1)
    sqrt_u1 = math.sqrt(u1)
    two_pi = 2.0 * math.pi
    return [
        sqrt1_minus_u1 * math.sin(two_pi * u2),
        sqrt1_minus_u1 * math.cos(two_pi * u2),
        sqrt_u1 * math.sin(two_pi * u3),
        sqrt_u1 * math.cos(two_pi * u3),
    ]


def arm_paddle_rest_start_indices():
    """Deterministic ~50% set of rest-start episodes, evenly spaced.

    Disjoint from the planned-failure set: a rest-start ball reaches its first
    paddle contact late (it falls from rest), which desynchronises the deliberate
    miss slide and ricochets the ball off the walls. Rest-start episodes are
    therefore all planned successes, and planned failures stay bounce-start.
    """
    global _ARM_PADDLE_REST_START_INDICES
    if _ARM_PADDLE_REST_START_INDICES is None:
        assignments = build_episode_assignments("arm_paddle_ball")
        total = len(assignments)
        target = round(ARM_PADDLE_REST_START_FRACTION * total)
        failures = arm_paddle_planned_failure_indices()
        eligible = [
            index
            for index, (_, gravity) in enumerate(assignments)
            if gravity > ARM_PADDLE_REST_START_MIN_GRAVITY and index not in failures
        ]
        indices = set()
        if target > 0 and eligible:
            target = min(target, len(eligible))
            stride = len(eligible) / target
            for slot in range(target):
                indices.add(eligible[min(int((slot + 0.5) * stride), len(eligible) - 1)])
        _ARM_PADDLE_REST_START_INDICES = frozenset(indices)
    return _ARM_PADDLE_REST_START_INDICES


def arm_paddle_planned_failure_indices():
    """Deterministic planned-failure episode set (~20% of the task).

    Zero-gravity episodes are counted first (the bounce task is physically
    impossible there), then evenly spaced episodes fill the remaining quota.
    """
    global _ARM_PADDLE_PLANNED_FAILURE_INDICES
    if _ARM_PADDLE_PLANNED_FAILURE_INDICES is None:
        assignments = build_episode_assignments("arm_paddle_ball")
        total = len(assignments)
        target = max(1, round(ARM_PADDLE_PLANNED_FAILURE_FRACTION * total))
        failures = {
            index for index, (_, gravity) in enumerate(assignments) if abs(gravity) <= 1.0e-9
        }
        remaining = [index for index in range(total) if index not in failures]
        needed = target - len(failures)
        if needed > 0 and remaining:
            stride = len(remaining) / needed
            for slot in range(needed):
                failures.add(remaining[min(int((slot + 0.5) * stride), len(remaining) - 1)])
        _ARM_PADDLE_PLANNED_FAILURE_INDICES = frozenset(failures)
    return _ARM_PADDLE_PLANNED_FAILURE_INDICES


def build_arm_paddle_ball_episode(rng, gravity, episode_index, split_name):
    scene = build_paddle_ball_episode(
        rng,
        gravity,
        episode_index,
        split_name,
        task_name="arm_paddle_ball",
        profile=ARM_PADDLE_TASK_PROFILE,
    )
    planned_outcome = (
        "failure" if episode_index in arm_paddle_planned_failure_indices() else "success"
    )
    failure_miss_frame = None
    failure_miss_offset_x = None
    if planned_outcome == "failure" and gravity > 1.0e-9:
        # Separate RNG stream so planned-success episodes are unaffected.
        failure_rng = random.Random(
            BASE_SEED + 10000 * TASKS.index("arm_paddle_ball") + 5551 * (episode_index + 1)
        )
        failure_miss_frame = failure_rng.randint(
            int(ARM_PADDLE_FAILURE_MISS_FRAME_FRACTION_RANGE[0] * FRAMES_PER_EPISODE),
            int(ARM_PADDLE_FAILURE_MISS_FRAME_FRACTION_RANGE[1] * FRAMES_PER_EPISODE),
        )
        failure_miss_offset_x = failure_rng.choice([-1.0, 1.0]) * ARM_PADDLE_FAILURE_MISS_OFFSET_X
        scene["paddle"]["failure_miss_frame"] = failure_miss_frame
        scene["paddle"]["failure_miss_offset_x"] = failure_miss_offset_x
    scene["metadata"].update(
        {
            "planned_outcome": planned_outcome,
            "failure_miss_frame": failure_miss_frame,
            "failure_miss_offset_x": failure_miss_offset_x,
        }
    )

    episode_seed = int(scene["metadata"]["random_seed"])
    # Random initial ball orientation (all episodes). The ball is a free body
    # (a sphere), so orientation has no dynamics effect -- only the rendered
    # checker pattern rotates -- but it diversifies the observed state.
    orientation_rng = random.Random(episode_seed + 7373)
    ball_quaternion = random_unit_quaternion(orientation_rng)
    scene["dynamic_objects"][0]["initial_quaternion"] = ball_quaternion
    scene["metadata"]["ball_initial_state"]["quaternion_xyzw"] = ball_quaternion

    # Rest-start episodes: the ball begins stationary near the paddle centre
    # with the paddle at a randomised position; the policy gets it bouncing and
    # re-centres it.
    rest_start = episode_index in arm_paddle_rest_start_indices()
    if rest_start:
        rest_rng = random.Random(episode_seed + 8484)
        paddle_x = rest_rng.uniform(*ARM_PADDLE_REST_START_PADDLE_X_RANGE)
        paddle_y = ARM_PADDLE_TILT_CENTER_XY[1] + rest_rng.uniform(
            *ARM_PADDLE_REST_START_PADDLE_Y_OFFSET_RANGE
        )
        paddle_z = ARM_PADDLE_INITIAL_POSITION[2]
        paddle_position = [paddle_x, paddle_y, paddle_z]
        # The ball rests stationary directly on the paddle surface (its lowest
        # point touching the blade), near the centre.
        ball_z = (
            paddle_z
            + ARM_PADDLE_PADDLE_HALF_EXTENTS[2]
            + ARM_PADDLE_BALL_RADIUS
            + ARM_PADDLE_REST_START_BALL_GAP
        )
        ball_position = [
            paddle_x + rest_rng.uniform(*ARM_PADDLE_REST_START_BALL_OFFSET_RANGE),
            paddle_y + rest_rng.uniform(*ARM_PADDLE_REST_START_BALL_OFFSET_RANGE),
            ball_z,
        ]
        scene["paddle"]["initial_position"] = list(paddle_position)
        scene["paddle"]["initial_target_position"] = list(paddle_position)
        scene["dynamic_objects"][0]["position"] = list(ball_position)
        scene["dynamic_objects"][0]["initial_velocity"] = [0.0, 0.0, 0.0]
        scene["metadata"]["ball_initial_state"]["position"] = list(ball_position)
        scene["metadata"]["ball_initial_state"]["linear_velocity"] = [0.0, 0.0, 0.0]
        scene["metadata"]["paddle_initial_state"]["position"] = list(paddle_position)
        scene["metadata"]["paddle_initial_state"]["target_position"] = list(paddle_position)
        # Gentle low bounces for the rested ball (see constant note).
        rest_overrides = dict(scene["paddle"].get("policy_config_overrides") or {})
        rest_overrides["target_apex_z"] = ARM_PADDLE_REST_START_TARGET_APEX_Z
        scene["paddle"]["policy_config_overrides"] = rest_overrides
    scene["metadata"]["initial_condition"] = "rest_start" if rest_start else "bounce_start"
    scene["metadata"]["rest_start"] = bool(rest_start)
    scene["metadata"]["command_noise"] = dict(ARM_PADDLE_COMMAND_NOISE)
    if planned_outcome == "failure" and not rest_start:
        # Regulate the (soon-to-fail) ball to a low apex so it is slow: a mild
        # wrong tilt then only nudges it gently instead of slamming the fast
        # high-gravity ball through the floor or far out of view.
        fail_overrides = dict(scene["paddle"].get("policy_config_overrides") or {})
        fail_overrides["target_apex_z"] = ARM_PADDLE_REST_START_TARGET_APEX_Z
        scene["paddle"]["policy_config_overrides"] = fail_overrides

    handle_part, handle_spec = make_arm_paddle_handle()
    scene["paddle"]["parts"].append(handle_part)
    scene["paddle"]["handle"] = handle_spec
    scene["camera"] = make_camera(
        ARM_PADDLE_CAMERA_LOCATION,
        ARM_PADDLE_CAMERA_TARGET,
        ARM_PADDLE_CAMERA_FOVY_DEGREES,
    )
    arm_spec = {
        "name": "arm",
        "robot_model": ARM_PADDLE_ROBOT_MODEL,
        "base_position": list(ARM_PADDLE_BASE_POSITION),
        "base_euler": list(ARM_PADDLE_BASE_EULER),
        "model_scale": ARM_PADDLE_MODEL_SCALE,
        "joint_names": list(ARM_PADDLE_JOINT_NAMES),
        "actuator_names": list(ARM_PADDLE_ACTUATOR_NAMES),
        "joint_limits": [list(limits) for limits in ARM_CATCHER_JOINT_LIMITS],
        "home_qpos": list(ARM_PADDLE_HOME_QPOS),
        "tool_tip_local_x": ARM_PADDLE_TOOL_TIP_LOCAL_X,
        "tip_normal_world": [
            0.0,
            -math.cos(ARM_PADDLE_HANDLE_ANGLE),
            -math.sin(ARM_PADDLE_HANDLE_ANGLE),
        ],
        "material": ARM_CATCHER_MATERIAL,
        "joint_material": ARM_CATCHER_JOINT_MATERIAL,
        "ik_nullspace_gain": 0.012,
        # joint6 spins the flange about the tool x-axis, which the IK already
        # constrains to the handle axis; it spans the task's entire 1-DOF
        # self-motion without moving the tip, so pinning it makes the joint
        # configuration a locally unique function of the grip pose. The arm is
        # kinematic-only (no ball/paddle coupling), so paddle physics and the
        # policy are unaffected.
        "ik_locked_joints": {"joint6": 0.0},
        "source": "google-deepmind/mujoco_menagerie/unitree_z1",
    }
    scene["arm"] = arm_spec
    scene["metadata"].update(
        {
            "description": (
                "A red ball bouncing on a flat paddle held by its rear handle in the "
                "end-effector of a Unitree Z1 robot arm."
            ),
            "paddle_type": "unitree_z1_held_xyz_tilt_controlled_flat_paddle_with_handle",
            "paddle_mount": "paddle_handle_socketed_into_unitree_z1_end_effector_flange",
            "paddle_handle": dict(handle_spec),
            "state_anchor": list(ARM_PADDLE_CAMERA_TARGET),
            "camera_parameters": {
                "location": list(ARM_PADDLE_CAMERA_LOCATION),
                "target": list(ARM_PADDLE_CAMERA_TARGET),
                "fovy": ARM_PADDLE_CAMERA_FOVY_DEGREES,
            },
            "arm_control": (
                "actions are high-level paddle target deltas and tilt; the Unitree Z1 "
                "holds the paddle by its rear handle (grip point at the handle end, "
                "flange axis along the handle) and tracks the grip pose with "
                "damped-least-squares IK joint position targets"
            ),
            "arm_initial_state": {
                "base_position": list(ARM_PADDLE_BASE_POSITION),
                "base_euler": list(ARM_PADDLE_BASE_EULER),
                "joint_names": list(ARM_PADDLE_JOINT_NAMES),
                "joint_limits": [list(limits) for limits in ARM_CATCHER_JOINT_LIMITS],
                "home_qpos": list(ARM_PADDLE_HOME_QPOS),
                "tool_tip_local_x": ARM_PADDLE_TOOL_TIP_LOCAL_X,
                "robot_model": ARM_PADDLE_ROBOT_MODEL,
                "model_scale": ARM_PADDLE_MODEL_SCALE,
                "scale_note": "real_scale",
                "source": arm_spec["source"],
            },
        }
    )
    return scene


def build_catcher_ball_episode(rng, gravity, episode_index, split_name):
    mass = rng.uniform(0.5, 2.0)
    radius = FIXED_BALL_RADIUS if FIXED_BALL_RADIUS is not None else CATCHER_BALL_RADIUS
    planned_outcome = choose_catcher_ball_planned_outcome(episode_index)
    policy_type = choose_catcher_ball_policy(planned_outcome)
    miss_bias = sample_catcher_miss_bias(rng) if planned_outcome == "miss" else [0.0, 0.0]
    initial_position, initial_velocity, launch_metadata = sample_catcher_ball_initial_state(
        rng,
        gravity,
        radius,
        planned_outcome,
    )
    dynamic_object = demo_generate.make_sphere(
        "moving_object",
        radius=radius,
        position=initial_position,
        mass=mass,
        material=demo_generate.accent_material(APPROACH_BALL_OBJECT_COLOR),
        restitution=ELASTIC_RESTITUTION,
        lateral_friction=UNIFORM_LATERAL_FRICTION,
        linear_damping=UNIFORM_LINEAR_DAMPING,
        angular_damping=UNIFORM_ANGULAR_DAMPING,
        initial_velocity=initial_velocity,
        initial_angular_velocity=(0.0, 0.0, 0.0),
        lock_rotation=True,
    )

    catcher_initial_position = [
        rng.uniform(*CATCHER_BALL_INITIAL_X_RANGE),
        CATCHER_BALL_FIXED_Y,
        rng.uniform(*CATCHER_BALL_INITIAL_Z_RANGE),
    ]

    phys = [
        gravity,
        mass,
        radius,
        radius,
        radius,
        SHAPE_ID_MAP["ball"],
        APPROACH_BALL_GROUND_HALF_EXTENTS[0],
        APPROACH_BALL_GROUND_HALF_EXTENTS[1],
        APPROACH_BALL_SIDE_WALL_HEIGHT,
        APPROACH_BALL_SIDE_WALL_THICKNESS,
    ]
    metadata = build_common_metadata(
        "catcher_ball",
        "A red ball thrown toward a near-camera front-facing net catcher.",
        gravity,
        episode_index,
        split_name,
        phys,
        [
            APPROACH_BALL_GROUND_HALF_EXTENTS[0],
            APPROACH_BALL_GROUND_HALF_EXTENTS[1],
            APPROACH_BALL_SIDE_WALL_HEIGHT,
            APPROACH_BALL_CAMERA_ELEVATION_DEGREES,
        ],
        list(APPROACH_BALL_CAMERA_TARGET),
    )
    metadata.update(
        {
            "policy_type": policy_type,
            "planned_outcome": planned_outcome,
            "outcome": None,
            "capture_frame": None,
            "miss_frame": None,
            "truncated_before_interaction": False,
            "terminal_reason": None,
            "random_seed": BASE_SEED + 10000 * TASKS.index("catcher_ball") + episode_index,
            "ball_initial_state": {
                "position": list(initial_position),
                "linear_velocity": list(initial_velocity),
                "radius": radius,
                "fixed_radius_default": CATCHER_BALL_RADIUS,
                "mass": mass,
                "launch_plan": launch_metadata,
            },
            "catcher_initial_state": {
                "position": list(catcher_initial_position),
                "target_position": list(catcher_initial_position),
                "workspace_min": list(CATCHER_BALL_WORKSPACE_MIN),
                "workspace_max": list(CATCHER_BALL_WORKSPACE_MAX),
                "planned_miss_bias_xz": list(miss_bias),
            },
            "catcher_type": "front_facing_net",
            "fixed_catch_depth_y": CATCHER_BALL_FIXED_Y,
            "catch_front_y": CATCHER_BALL_CATCH_FRONT_Y,
            "latch_after_capture": True,
            "camera_parameters": {
                "location": list(APPROACH_BALL_CAMERA_LOCATION),
                "target": list(APPROACH_BALL_CAMERA_TARGET),
                "fovy": APPROACH_BALL_CAMERA_FOVY_DEGREES,
            },
        }
    )

    catcher = {
        "name": "catcher",
        "initial_position": list(catcher_initial_position),
        "initial_target_position": list(catcher_initial_position),
        "workspace_min": list(CATCHER_BALL_WORKSPACE_MIN),
        "workspace_max": list(CATCHER_BALL_WORKSPACE_MAX),
        "fixed_y": CATCHER_BALL_FIXED_Y,
        "target_y": CATCHER_BALL_FIXED_Y,
        "net_half_width": CATCHER_BALL_NET_HALF_WIDTH,
        "net_half_height": CATCHER_BALL_NET_HALF_HEIGHT,
        "net_depth": CATCHER_BALL_NET_DEPTH,
        "material": CATCHER_BALL_MATERIAL,
        "policy_type": policy_type,
        "planned_outcome": planned_outcome,
        "miss_bias": list(miss_bias),
        "parts": [
            {
                "name": "catcher_rim_left",
                "half_extents": [
                    CATCHER_BALL_RIM_HALF_THICKNESS,
                    CATCHER_BALL_RIM_HALF_THICKNESS,
                    CATCHER_BALL_NET_HALF_HEIGHT + CATCHER_BALL_RIM_HALF_THICKNESS,
                ],
                "offset": [
                    -CATCHER_BALL_NET_HALF_WIDTH,
                    0.5 * CATCHER_BALL_NET_DEPTH,
                    0.0,
                ],
                "material": CATCHER_BALL_MATERIAL,
                "collision": False,
            },
            {
                "name": "catcher_rim_right",
                "half_extents": [
                    CATCHER_BALL_RIM_HALF_THICKNESS,
                    CATCHER_BALL_RIM_HALF_THICKNESS,
                    CATCHER_BALL_NET_HALF_HEIGHT + CATCHER_BALL_RIM_HALF_THICKNESS,
                ],
                "offset": [
                    CATCHER_BALL_NET_HALF_WIDTH,
                    0.5 * CATCHER_BALL_NET_DEPTH,
                    0.0,
                ],
                "material": CATCHER_BALL_MATERIAL,
                "collision": False,
            },
            {
                "name": "catcher_rim_bottom",
                "half_extents": [
                    CATCHER_BALL_NET_HALF_WIDTH + CATCHER_BALL_RIM_HALF_THICKNESS,
                    CATCHER_BALL_RIM_HALF_THICKNESS,
                    CATCHER_BALL_RIM_HALF_THICKNESS,
                ],
                "offset": [
                    0.0,
                    0.5 * CATCHER_BALL_NET_DEPTH,
                    -CATCHER_BALL_NET_HALF_HEIGHT,
                ],
                "material": CATCHER_BALL_MATERIAL,
                "collision": False,
            },
            {
                "name": "catcher_rim_top",
                "half_extents": [
                    CATCHER_BALL_NET_HALF_WIDTH + CATCHER_BALL_RIM_HALF_THICKNESS,
                    CATCHER_BALL_RIM_HALF_THICKNESS,
                    CATCHER_BALL_RIM_HALF_THICKNESS,
                ],
                "offset": [
                    0.0,
                    0.5 * CATCHER_BALL_NET_DEPTH,
                    CATCHER_BALL_NET_HALF_HEIGHT,
                ],
                "material": CATCHER_BALL_MATERIAL,
                "collision": False,
            },
            {
                "name": "catcher_collision_back",
                "half_extents": [
                    CATCHER_BALL_NET_HALF_WIDTH,
                    CATCHER_BALL_COLLISION_WALL_THICKNESS,
                    CATCHER_BALL_NET_HALF_HEIGHT,
                ],
                "offset": [0.0, -0.5 * CATCHER_BALL_NET_DEPTH, 0.0],
                "material": CATCHER_BALL_COLLISION_MATERIAL,
                "collision": True,
            },
            {
                "name": "catcher_collision_left",
                "half_extents": [
                    CATCHER_BALL_COLLISION_WALL_THICKNESS,
                    0.5 * CATCHER_BALL_NET_DEPTH,
                    CATCHER_BALL_NET_HALF_HEIGHT,
                ],
                "offset": [-CATCHER_BALL_NET_HALF_WIDTH, 0.0, 0.0],
                "material": CATCHER_BALL_COLLISION_MATERIAL,
                "collision": True,
            },
            {
                "name": "catcher_collision_right",
                "half_extents": [
                    CATCHER_BALL_COLLISION_WALL_THICKNESS,
                    0.5 * CATCHER_BALL_NET_DEPTH,
                    CATCHER_BALL_NET_HALF_HEIGHT,
                ],
                "offset": [CATCHER_BALL_NET_HALF_WIDTH, 0.0, 0.0],
                "material": CATCHER_BALL_COLLISION_MATERIAL,
                "collision": True,
            },
            {
                "name": "catcher_collision_bottom",
                "half_extents": [
                    CATCHER_BALL_NET_HALF_WIDTH,
                    0.5 * CATCHER_BALL_NET_DEPTH,
                    CATCHER_BALL_COLLISION_WALL_THICKNESS,
                ],
                "offset": [0.0, 0.0, -CATCHER_BALL_NET_HALF_HEIGHT],
                "material": CATCHER_BALL_COLLISION_MATERIAL,
                "collision": True,
            },
        ],
    }
    for grid_index, offset_x in enumerate(
        (-0.5 * CATCHER_BALL_NET_HALF_WIDTH, 0.0, 0.5 * CATCHER_BALL_NET_HALF_WIDTH)
    ):
        catcher["parts"].append(
            {
                "name": f"catcher_net_grid_vertical_{grid_index}",
                "half_extents": [
                    CATCHER_BALL_GRID_HALF_THICKNESS,
                    CATCHER_BALL_GRID_HALF_THICKNESS,
                    CATCHER_BALL_NET_HALF_HEIGHT,
                ],
                "offset": [offset_x, -0.5 * CATCHER_BALL_NET_DEPTH, 0.0],
                "material": CATCHER_BALL_NET_MATERIAL,
                "collision": False,
            }
        )
    for grid_index, offset_z in enumerate(
        (-0.5 * CATCHER_BALL_NET_HALF_HEIGHT, 0.0, 0.5 * CATCHER_BALL_NET_HALF_HEIGHT)
    ):
        catcher["parts"].append(
            {
                "name": f"catcher_net_grid_horizontal_{grid_index}",
                "half_extents": [
                    CATCHER_BALL_NET_HALF_WIDTH,
                    CATCHER_BALL_GRID_HALF_THICKNESS,
                    CATCHER_BALL_GRID_HALF_THICKNESS,
                ],
                "offset": [0.0, -0.5 * CATCHER_BALL_NET_DEPTH, offset_z],
                "material": CATCHER_BALL_NET_MATERIAL,
                "collision": False,
            }
        )
    return {
        "metadata": metadata,
        "camera": make_camera(
            APPROACH_BALL_CAMERA_LOCATION,
            APPROACH_BALL_CAMERA_TARGET,
            APPROACH_BALL_CAMERA_FOVY_DEGREES,
        ),
        "static_objects": build_approach_ball_ground(),
        "dynamic_objects": [dynamic_object],
        "catcher": catcher,
    }


def make_arm_catcher_parts():
    parts = [
        {
            "name": "arm_catcher_rim_left",
            "half_extents": [
                ARM_CATCHER_RIM_HALF_THICKNESS,
                ARM_CATCHER_RIM_HALF_THICKNESS,
                ARM_CATCHER_NET_HALF_HEIGHT + ARM_CATCHER_RIM_HALF_THICKNESS,
            ],
            "offset": [
                -ARM_CATCHER_NET_HALF_WIDTH,
                ARM_CATCHER_NET_DEPTH,
                0.0,
            ],
            "material": CATCHER_BALL_MATERIAL,
            "collision": False,
        },
        {
            "name": "arm_catcher_rim_right",
            "half_extents": [
                ARM_CATCHER_RIM_HALF_THICKNESS,
                ARM_CATCHER_RIM_HALF_THICKNESS,
                ARM_CATCHER_NET_HALF_HEIGHT + ARM_CATCHER_RIM_HALF_THICKNESS,
            ],
            "offset": [
                ARM_CATCHER_NET_HALF_WIDTH,
                ARM_CATCHER_NET_DEPTH,
                0.0,
            ],
            "material": CATCHER_BALL_MATERIAL,
            "collision": False,
        },
        {
            "name": "arm_catcher_rim_bottom",
            "half_extents": [
                ARM_CATCHER_NET_HALF_WIDTH + ARM_CATCHER_RIM_HALF_THICKNESS,
                ARM_CATCHER_RIM_HALF_THICKNESS,
                ARM_CATCHER_RIM_HALF_THICKNESS,
            ],
            "offset": [
                0.0,
                ARM_CATCHER_NET_DEPTH,
                -ARM_CATCHER_NET_HALF_HEIGHT,
            ],
            "material": CATCHER_BALL_MATERIAL,
            "collision": False,
        },
        {
            "name": "arm_catcher_rim_top",
            "half_extents": [
                ARM_CATCHER_NET_HALF_WIDTH + ARM_CATCHER_RIM_HALF_THICKNESS,
                ARM_CATCHER_RIM_HALF_THICKNESS,
                ARM_CATCHER_RIM_HALF_THICKNESS,
            ],
            "offset": [
                0.0,
                ARM_CATCHER_NET_DEPTH,
                ARM_CATCHER_NET_HALF_HEIGHT,
            ],
            "material": CATCHER_BALL_MATERIAL,
            "collision": False,
        },
        {
            "name": "arm_catcher_collision_back",
            "half_extents": [
                ARM_CATCHER_NET_HALF_WIDTH,
                ARM_CATCHER_COLLISION_WALL_THICKNESS,
                ARM_CATCHER_NET_HALF_HEIGHT,
            ],
            "offset": [0.0, ARM_CATCHER_COLLISION_WALL_THICKNESS, 0.0],
            "material": CATCHER_BALL_COLLISION_MATERIAL,
            "collision": True,
        },
        {
            "name": "arm_catcher_collision_left",
            "half_extents": [
                ARM_CATCHER_COLLISION_WALL_THICKNESS,
                0.5 * ARM_CATCHER_NET_DEPTH,
                ARM_CATCHER_NET_HALF_HEIGHT,
            ],
            "offset": [-ARM_CATCHER_NET_HALF_WIDTH, 0.5 * ARM_CATCHER_NET_DEPTH, 0.0],
            "material": CATCHER_BALL_COLLISION_MATERIAL,
            "collision": True,
        },
        {
            "name": "arm_catcher_collision_right",
            "half_extents": [
                ARM_CATCHER_COLLISION_WALL_THICKNESS,
                0.5 * ARM_CATCHER_NET_DEPTH,
                ARM_CATCHER_NET_HALF_HEIGHT,
            ],
            "offset": [ARM_CATCHER_NET_HALF_WIDTH, 0.5 * ARM_CATCHER_NET_DEPTH, 0.0],
            "material": CATCHER_BALL_COLLISION_MATERIAL,
            "collision": True,
        },
        {
            "name": "arm_catcher_collision_bottom",
            "half_extents": [
                ARM_CATCHER_NET_HALF_WIDTH,
                0.5 * ARM_CATCHER_NET_DEPTH,
                ARM_CATCHER_COLLISION_WALL_THICKNESS,
            ],
            "offset": [0.0, 0.5 * ARM_CATCHER_NET_DEPTH, -ARM_CATCHER_NET_HALF_HEIGHT],
            "material": CATCHER_BALL_COLLISION_MATERIAL,
            "collision": True,
        },
    ]
    for grid_index, offset_x in enumerate(
        scale_vector((-0.24, 0.0, 0.24), ARM_CATCHER_SCENE_SCALE)
    ):
        parts.append(
            {
                "name": f"arm_catcher_net_grid_vertical_{grid_index}",
                "half_extents": [
                    ARM_CATCHER_GRID_HALF_THICKNESS,
                    ARM_CATCHER_GRID_HALF_THICKNESS,
                    ARM_CATCHER_NET_HALF_HEIGHT,
                ],
                "offset": [offset_x, ARM_CATCHER_GRID_HALF_THICKNESS, 0.0],
                "material": CATCHER_BALL_NET_MATERIAL,
                "collision": False,
            }
        )
    for grid_index, offset_z in enumerate(
        scale_vector((-0.20, 0.0, 0.20), ARM_CATCHER_SCENE_SCALE)
    ):
        parts.append(
            {
                "name": f"arm_catcher_net_grid_horizontal_{grid_index}",
                "half_extents": [
                    ARM_CATCHER_NET_HALF_WIDTH,
                    ARM_CATCHER_GRID_HALF_THICKNESS,
                    ARM_CATCHER_GRID_HALF_THICKNESS,
                ],
                "offset": [0.0, ARM_CATCHER_GRID_HALF_THICKNESS, offset_z],
                "material": CATCHER_BALL_NET_MATERIAL,
                "collision": False,
            }
        )
    return parts


def sample_arm_catcher_initial_mount_position(rng, launch_metadata):
    return list(ARM_CATCHER_FIXED_INITIAL_MOUNT_POSITION)


def build_arm_catcher_ball_episode(rng, gravity, episode_index, split_name):
    mass = rng.uniform(0.5, 2.0)
    radius = FIXED_BALL_RADIUS if FIXED_BALL_RADIUS is not None else ARM_CATCHER_BALL_RADIUS
    planned_outcome = choose_arm_catcher_ball_planned_outcome(episode_index)
    policy_type = choose_catcher_ball_policy(planned_outcome)
    miss_bias = (
        list(scale_vector((-1.35, 0.25, -0.28), ARM_CATCHER_SCENE_SCALE))
        if planned_outcome == "miss"
        else [0.0, 0.0, 0.0]
    )
    initial_position, initial_velocity, launch_metadata = sample_arm_catcher_ball_initial_state(
        rng,
        gravity,
        radius,
        planned_outcome,
        episode_index=episode_index,
    )
    # Drawn after the launch sampling so the launch state is unchanged, and from
    # the episode rng so each capture retry attempt gets its own spin.
    initial_angular_velocity = tuple(
        rng.uniform(*ARM_CATCHER_BALL_INITIAL_SPIN_RANGE) for _ in range(3)
    )
    dynamic_object = demo_generate.make_sphere(
        "moving_object",
        radius=radius,
        position=initial_position,
        mass=mass,
        material=ARM_CATCHER_BALL_MATERIAL,
        restitution=ELASTIC_RESTITUTION,
        lateral_friction=UNIFORM_LATERAL_FRICTION,
        linear_damping=UNIFORM_LINEAR_DAMPING,
        angular_damping=UNIFORM_ANGULAR_DAMPING,
        initial_velocity=initial_velocity,
        initial_angular_velocity=initial_angular_velocity,
        # Free-rotating like arm_paddle_ball, so the checker pattern tumbles
        # instead of sliding rigidly.
        lock_rotation=False,
    )

    catcher_initial_position = sample_arm_catcher_initial_mount_position(rng, launch_metadata)
    camera = make_camera(
        ARM_CATCHER_CAMERA_LOCATION,
        ARM_CATCHER_CAMERA_TARGET,
        ARM_CATCHER_CAMERA_FOVY_DEGREES,
    )
    screen_initial_x = projected_screen_x(initial_position, camera)
    screen_cross_x = projected_screen_x(
        (
            launch_metadata["planned_cross_x"],
            launch_metadata["planned_cross_y"],
            launch_metadata["estimated_cross_z"],
        ),
        camera,
    )
    screen_right_capture_required = (
        planned_outcome == "capture" and require_arm_catcher_screen_right_capture(episode_index)
    )

    phys = [
        gravity,
        mass,
        radius,
        radius,
        radius,
        SHAPE_ID_MAP["ball"],
        ARM_CATCHER_GROUND_HALF_EXTENTS[0],
        ARM_CATCHER_GROUND_HALF_EXTENTS[1],
        ARM_CATCHER_SIDE_WALL_HEIGHT,
        ARM_CATCHER_SIDE_WALL_THICKNESS,
    ]
    metadata = build_common_metadata(
        "arm_catcher_ball",
        "A red ball thrown toward a right-mounted MuJoCo robot arm with a front-facing net end-effector.",
        gravity,
        episode_index,
        split_name,
        phys,
        [
            ARM_CATCHER_GROUND_HALF_EXTENTS[0],
            ARM_CATCHER_GROUND_HALF_EXTENTS[1],
            ARM_CATCHER_SIDE_WALL_HEIGHT,
            APPROACH_BALL_CAMERA_ELEVATION_DEGREES,
        ],
        list(ARM_CATCHER_CAMERA_TARGET),
    )
    arm_spec = {
        "name": "arm",
        "base_position": list(ARM_CATCHER_BASE_POSITION),
        "base_euler": [0.0, 0.0, math.pi],
        "shoulder_height": ARM_CATCHER_SHOULDER_HEIGHT,
        "upper_link_length": ARM_CATCHER_UPPER_LINK_LENGTH,
        "lower_link_length": ARM_CATCHER_LOWER_LINK_LENGTH,
        "link_half_thickness": ARM_CATCHER_LINK_HALF_THICKNESS,
        "robot_model": "unitree_z1",
        "model_scale": ARM_CATCHER_MODEL_SCALE,
        "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        "joint_limits": [list(limits) for limits in ARM_CATCHER_JOINT_LIMITS],
        "material": ARM_CATCHER_MATERIAL,
        "joint_material": ARM_CATCHER_JOINT_MATERIAL,
        "tool_tip_local_x": 0.051,
        "tip_normal_world": [0.0, 1.0, 0.0],
        "home_qpos": [0.0, 0.785, -0.261, -0.523, 0.0, 0.0],
        "ik_nullspace_gain": 0.012,
        "ik_locked_joints": {"joint6": 0.0},
    }
    metadata.update(
        {
            "policy_type": policy_type,
            "planned_outcome": planned_outcome,
            "outcome": None,
            "capture_frame": None,
            "miss_frame": None,
            "truncated_before_interaction": False,
            "terminal_reason": None,
            "random_seed": BASE_SEED + 10000 * TASKS.index("arm_catcher_ball") + episode_index,
            "ball_initial_state": {
                "position": list(initial_position),
                "linear_velocity": list(initial_velocity),
                "radius": radius,
                "mass": mass,
                "fixed_radius_default": ARM_CATCHER_BALL_RADIUS,
                "launch_plan": launch_metadata,
            },
            "catcher_initial_state": {
                "position": list(catcher_initial_position),
                "target_position": list(catcher_initial_position),
                "workspace_min": list(ARM_CATCHER_WORKSPACE_MIN),
                "workspace_max": list(ARM_CATCHER_WORKSPACE_MAX),
                "planned_miss_bias_xyz": list(miss_bias),
            },
            "catcher_type": "right_mounted_mujoco_robot_arm_front_facing_net",
            "scene_scale": ARM_CATCHER_SCENE_SCALE,
            "ball_speed_scale": ARM_CATCHER_BALL_SPEED_SCALE,
            "start_y_multiplier": ARM_CATCHER_START_Y_MULTIPLIER,
            "cross_frame_multiplier": ARM_CATCHER_CROSS_FRAME_MULTIPLIER,
            "capture_cross_time_range": list(ARM_CATCHER_CAPTURE_CROSS_TIME_RANGE),
            "capture_cross_frame_range": list(ARM_CATCHER_CAPTURE_CROSS_FRAME_RANGE),
            "y_action_multiplier": ARM_CATCHER_Y_ACTION_MULTIPLIER,
            "target_smoothing_alpha": ARM_CATCHER_TARGET_SMOOTHING_ALPHA,
            "smooth_max_delta_xyz": list(ARM_CATCHER_SMOOTH_MAX_DELTA_XYZ),
            "action_smoothing_alpha": ARM_CATCHER_ACTION_SMOOTHING_ALPHA,
            "action_max_delta_change_xyz": list(ARM_CATCHER_ACTION_MAX_DELTA_CHANGE_XYZ),
            "per_frame_max_delta_xyz": list(ARM_CATCHER_PER_FRAME_MAX_DELTA_XYZ),
            "planner_type": "time_aware_ballistic_interception",
            "planner_arrival_margin_frames": ARM_CATCHER_PLANNER_ARRIVAL_MARGIN_FRAMES,
            "planner_max_speed_xyz": list(ARM_CATCHER_PLANNER_MAX_SPEED_XYZ),
            "planner_max_accel_xyz": list(ARM_CATCHER_PLANNER_MAX_ACCEL_XYZ),
            "planner_max_joint_step": ARM_CATCHER_PLANNER_MAX_JOINT_STEP,
            "planner_min_z": ARM_CATCHER_PLANNER_MIN_Z,
            "capture_min_target_z": ARM_CATCHER_CAPTURE_MIN_TARGET_Z,
            "capture_max_target_z_hint": ARM_CATCHER_CAPTURE_MAX_TARGET_Z_HINT,
            "planner_wiggle_amplitude_xyz": list(ARM_CATCHER_PLANNER_WIGGLE_AMPLITUDE_XYZ),
            "planner_wiggle_tau_seconds": ARM_CATCHER_PLANNER_WIGGLE_TAU_SECONDS,
            "planner_wiggle_fade_seconds": ARM_CATCHER_PLANNER_WIGGLE_FADE_SECONDS,
            "capture_max_frame_mount_delta": ARM_CATCHER_CAPTURE_MAX_FRAME_MOUNT_DELTA,
            "capture_max_frame_mount_acceleration_delta": (
                ARM_CATCHER_CAPTURE_MAX_FRAME_MOUNT_ACCELERATION_DELTA
            ),
            "screen_initial_x": screen_initial_x,
            "screen_cross_x": screen_cross_x,
            "screen_right_capture_required": screen_right_capture_required,
            "screen_right_capture_fraction": ARM_CATCHER_SCREEN_RIGHT_CAPTURE_FRACTION,
            "screen_right_capture_min_x": ARM_CATCHER_SCREEN_RIGHT_MIN_X,
            "policy_noise_scale": ARM_CATCHER_SCENE_SCALE,
            "policy_bias_scale": ARM_CATCHER_SCENE_SCALE,
            "max_trajectory_z": ARM_CATCHER_MAX_TRAJECTORY_Z,
            "nominal_catch_depth_y": ARM_CATCHER_CATCH_FRONT_Y,
            "mount_back_face_y": ARM_CATCHER_MOUNT_BACK_Y,
            "catch_front_y": ARM_CATCHER_CATCH_FRONT_Y,
            "final_target_x_shift": ARM_CATCHER_FINAL_TARGET_X_SHIFT,
            "final_target_x_shift_fraction_of_workspace": ARM_CATCHER_FINAL_TARGET_X_SHIFT
            / (ARM_CATCHER_WORKSPACE_MAX[0] - ARM_CATCHER_WORKSPACE_MIN[0]),
            "final_target_x_shift_fraction_of_open_box_width": ARM_CATCHER_FINAL_TARGET_X_SHIFT_FRACTION,
            "capture_target_x_range": list(ARM_CATCHER_CAPTURE_TARGET_X_RANGE),
            "capture_target_x_distribution": "stratified_uniform_capture_episode",
            "fixed_initial_mount_position": list(ARM_CATCHER_FIXED_INITIAL_MOUNT_POSITION),
            "capture_min_mount_travel": ARM_CATCHER_CAPTURE_MIN_MOUNT_TRAVEL,
            "capture_center_window": {
                "half_width": ARM_CATCHER_CAPTURE_CENTER_HALF_WIDTH,
                "half_height": ARM_CATCHER_CAPTURE_CENTER_HALF_HEIGHT,
                "radius": ARM_CATCHER_CAPTURE_CENTER_RADIUS,
            },
            "capture_depth_window": {
                "front_margin": ARM_CATCHER_CAPTURE_FRONT_MARGIN,
                "back_margin": ARM_CATCHER_CAPTURE_BACK_MARGIN,
                "max_latch_jump": ARM_CATCHER_CAPTURE_MAX_LATCH_JUMP,
            },
            "latch_window": {
                "half_width": ARM_CATCHER_LATCH_HALF_WIDTH,
                "half_height": ARM_CATCHER_LATCH_HALF_HEIGHT,
                "y_min": ARM_CATCHER_LATCH_Y_MIN,
                "y_max": ARM_CATCHER_LATCH_Y_MAX,
            },
            "truncated_target_x_range": list(ARM_CATCHER_TRUNCATED_TARGET_X_RANGE),
            "miss_target_x_ranges": [
                list(ARM_CATCHER_MISS_TARGET_X_LEFT_RANGE),
                list(ARM_CATCHER_MISS_TARGET_X_RIGHT_RANGE),
            ],
            "latch_after_capture": True,
            "capture_window_frames": 1,
            "action_space": "normalized_end_effector_delta_xyz",
            "normalized_action_bounds": [-1.0, 1.0],
            "action_scale_xyz": list(ARM_CATCHER_ACTION_SCALE_XYZ),
            "arm_initial_state": {
                "base_position": list(ARM_CATCHER_BASE_POSITION),
                "joint_names": list(arm_spec["joint_names"]),
                "joint_limits": [list(limits) for limits in ARM_CATCHER_JOINT_LIMITS],
                "upper_link_length": ARM_CATCHER_UPPER_LINK_LENGTH,
                "lower_link_length": ARM_CATCHER_LOWER_LINK_LENGTH,
                "mount_side": "right",
                "tool_tip_local_x": 0.051,
                "tip_normal_world": [0.0, 1.0, 0.0],
                "robot_model": "unitree_z1",
                "model_scale": ARM_CATCHER_MODEL_SCALE,
                "scale_note": "physical_unitree_z1_scale_1_with_scene_scale",
                "source": "google-deepmind/mujoco_menagerie/unitree_z1",
            },
            "camera_parameters": {
                "location": list(ARM_CATCHER_CAMERA_LOCATION),
                "target": list(ARM_CATCHER_CAMERA_TARGET),
                "fovy": ARM_CATCHER_CAMERA_FOVY_DEGREES,
            },
            # Scale the natural light rig with the scene so the walls keep the
            # diffuse term they had before the 1/7.5 rescale. Opt-in per task:
            # the rig is shared, and the other natural-env tasks are still lit
            # by the unscaled one.
            "render_light_scale": ARM_CATCHER_SCENE_SCALE,
        }
    )

    catcher = {
        "name": "arm_catcher",
        "initial_position": list(catcher_initial_position),
        "initial_target_position": list(catcher_initial_position),
        "workspace_min": list(ARM_CATCHER_WORKSPACE_MIN),
        "workspace_max": list(ARM_CATCHER_WORKSPACE_MAX),
        "fixed_y": ARM_CATCHER_MOUNT_BACK_Y,
        "target_y": ARM_CATCHER_MOUNT_BACK_Y,
        "ground_z": ARM_CATCHER_GROUND_CENTER[2] + ARM_CATCHER_GROUND_HALF_EXTENTS[2],
        "net_half_width": ARM_CATCHER_NET_HALF_WIDTH,
        "net_half_height": ARM_CATCHER_NET_HALF_HEIGHT,
        "net_depth": ARM_CATCHER_NET_DEPTH,
        "capture_center_half_width": ARM_CATCHER_CAPTURE_CENTER_HALF_WIDTH,
        "capture_center_half_height": ARM_CATCHER_CAPTURE_CENTER_HALF_HEIGHT,
        "capture_center_radius": ARM_CATCHER_CAPTURE_CENTER_RADIUS,
        "mount_origin": "back_face_center",
        "back_face_mount_offset": [0.0, 0.0, 0.0],
        "mount_back_offset": ARM_CATCHER_MOUNT_BACK_OFFSET,
        "open_axis_world": [0.0, 1.0, 0.0],
        "x_axis_world": [1.0, 0.0, 0.0],
        "z_axis_world": [0.0, 0.0, 1.0],
        "control_axes": ["x", "y", "z"],
        "action_scale_xyz": list(ARM_CATCHER_ACTION_SCALE_XYZ),
        "target_smoothing_alpha": ARM_CATCHER_TARGET_SMOOTHING_ALPHA,
        "smooth_max_delta_xyz": list(ARM_CATCHER_SMOOTH_MAX_DELTA_XYZ),
        "action_smoothing_alpha": ARM_CATCHER_ACTION_SMOOTHING_ALPHA,
        "action_max_delta_change_xyz": list(ARM_CATCHER_ACTION_MAX_DELTA_CHANGE_XYZ),
        "per_frame_max_delta_xyz": list(ARM_CATCHER_PER_FRAME_MAX_DELTA_XYZ),
        "planner_arrival_margin_frames": ARM_CATCHER_PLANNER_ARRIVAL_MARGIN_FRAMES,
        "planner_max_speed_xyz": list(ARM_CATCHER_PLANNER_MAX_SPEED_XYZ),
        "planner_max_accel_xyz": list(ARM_CATCHER_PLANNER_MAX_ACCEL_XYZ),
        "planner_max_joint_step": ARM_CATCHER_PLANNER_MAX_JOINT_STEP,
        "planner_min_z": ARM_CATCHER_PLANNER_MIN_Z,
        "planner_wiggle_amplitude_xyz": list(ARM_CATCHER_PLANNER_WIGGLE_AMPLITUDE_XYZ),
        "planner_wiggle_tau_seconds": ARM_CATCHER_PLANNER_WIGGLE_TAU_SECONDS,
        "planner_wiggle_fade_seconds": ARM_CATCHER_PLANNER_WIGGLE_FADE_SECONDS,
        "capture_front_margin": ARM_CATCHER_CAPTURE_FRONT_MARGIN,
        "capture_back_margin": ARM_CATCHER_CAPTURE_BACK_MARGIN,
        "capture_max_latch_jump": ARM_CATCHER_CAPTURE_MAX_LATCH_JUMP,
        "latch_half_width": ARM_CATCHER_LATCH_HALF_WIDTH,
        "latch_half_height": ARM_CATCHER_LATCH_HALF_HEIGHT,
        "latch_y_min": ARM_CATCHER_LATCH_Y_MIN,
        "latch_y_max": ARM_CATCHER_LATCH_Y_MAX,
        "material": CATCHER_BALL_MATERIAL,
        "policy_type": policy_type,
        "policy_noise_scale": ARM_CATCHER_SCENE_SCALE,
        "policy_bias_scale": ARM_CATCHER_SCENE_SCALE,
        "planned_outcome": planned_outcome,
        "capture_window_frames": 1,
        "miss_bias": list(miss_bias),
        "parts": make_arm_catcher_parts(),
    }
    return {
        "metadata": metadata,
        "camera": camera,
        "static_objects": build_approach_ball_ground(
            ARM_CATCHER_GROUND_HALF_EXTENTS,
            ground_center=ARM_CATCHER_GROUND_CENTER,
            side_wall_height=ARM_CATCHER_SIDE_WALL_HEIGHT,
            side_wall_thickness=ARM_CATCHER_SIDE_WALL_THICKNESS,
            grid_spacing=ARM_CATCHER_GRID_SPACING,
            grid_half_thickness=ARM_CATCHER_GROUND_GRID_HALF_THICKNESS,
            grid_surface_offset=ARM_CATCHER_GRID_SURFACE_OFFSET,
            wall_natural_material="natural_wall_floor_mat",
            ground_grid_depth_half_thickness=ARM_CATCHER_GRID_DEPTH_HALF_THICKNESS,
            wall_grid_depth_half_thickness=ARM_CATCHER_GRID_DEPTH_HALF_THICKNESS,
            landing_friction=ARM_CATCHER_GROUND_LANDING_FRICTION,
        ),
        "dynamic_objects": [dynamic_object],
        "catcher": catcher,
        "arm": arm_spec,
    }


def build_arm_gripper_ball_episode(rng, gravity, episode_index, split_name):
    mass = rng.uniform(*ARM_GRIPPER_BALL_MASS_RANGE)
    unscaled_radius = (
        FIXED_BALL_RADIUS if FIXED_BALL_RADIUS is not None else ARM_GRIPPER_BALL_RADIUS
    )
    radius = unscaled_radius * ARM_GRIPPER_BALL_SCALE
    planned_outcome = choose_catcher_ball_planned_outcome(episode_index)
    policy_type = choose_catcher_ball_policy(planned_outcome)
    if planned_outcome == "miss":
        miss_bias = [
            rng.choice([-1.0, 1.0]) * rng.uniform(0.12, 0.20),
            rng.uniform(-0.020, 0.030),
            rng.choice([-1.0, 1.0]) * rng.uniform(0.030, 0.070),
        ]
    else:
        miss_bias = [0.0, 0.0, 0.0]
    initial_position, initial_velocity, launch_metadata = sample_arm_gripper_ball_initial_state(
        rng,
        gravity,
        radius,
        planned_outcome,
    )
    dynamic_object = demo_generate.make_sphere(
        "moving_object",
        radius=radius,
        position=initial_position,
        mass=mass,
        material=demo_generate.accent_material(APPROACH_BALL_OBJECT_COLOR),
        restitution=ELASTIC_RESTITUTION,
        lateral_friction=UNIFORM_LATERAL_FRICTION,
        linear_damping=UNIFORM_LINEAR_DAMPING,
        angular_damping=UNIFORM_ANGULAR_DAMPING,
        initial_velocity=initial_velocity,
        initial_angular_velocity=(0.0, 0.0, 0.0),
        lock_rotation=True,
    )

    if planned_outcome == "capture":
        gripper_initial_target = [
            min(
                max(launch_metadata["planned_cross_x"], ARM_GRIPPER_INITIAL_TARGET_X_RANGE[0]),
                ARM_GRIPPER_INITIAL_TARGET_X_RANGE[1],
            ),
            launch_metadata["planned_cross_y"],
            min(
                max(
                    launch_metadata["estimated_cross_z"] - 0.02,
                    ARM_GRIPPER_INITIAL_TARGET_Z_RANGE[0],
                ),
                ARM_GRIPPER_INITIAL_TARGET_Z_RANGE[1],
            ),
        ]
    else:
        gripper_initial_target = [
            rng.uniform(*ARM_GRIPPER_INITIAL_TARGET_X_RANGE),
            rng.uniform(*ARM_GRIPPER_INITIAL_TARGET_Y_RANGE),
            rng.uniform(*ARM_GRIPPER_INITIAL_TARGET_Z_RANGE),
        ]
    phys = [
        gravity,
        mass,
        radius,
        radius,
        radius,
        SHAPE_ID_MAP["ball"],
        APPROACH_BALL_GROUND_HALF_EXTENTS[0],
        APPROACH_BALL_GROUND_HALF_EXTENTS[1],
        APPROACH_BALL_SIDE_WALL_HEIGHT,
        APPROACH_BALL_SIDE_WALL_THICKNESS,
    ]
    metadata = build_common_metadata(
        "arm_gripper_ball",
        "A red ball thrown toward a standard UR5e arm with a Robotiq 2F-85 gripper.",
        gravity,
        episode_index,
        split_name,
        phys,
        [
            APPROACH_BALL_GROUND_HALF_EXTENTS[0],
            APPROACH_BALL_GROUND_HALF_EXTENTS[1],
            APPROACH_BALL_SIDE_WALL_HEIGHT,
            APPROACH_BALL_CAMERA_ELEVATION_DEGREES,
        ],
        list(ARM_GRIPPER_CAMERA_TARGET),
    )
    arm_spec = {
        "name": "arm",
        "robot_model": "ur5e_robotiq_2f85",
        "base_position": list(ARM_GRIPPER_BASE_POSITION),
        "base_euler": list(ARM_GRIPPER_BASE_EULER),
        "joint_names": list(ARM_GRIPPER_JOINT_NAMES),
        "actuator_names": list(ARM_GRIPPER_ACTUATOR_NAMES),
        "joint_limits": [list(limits) for limits in ARM_GRIPPER_JOINT_LIMITS],
        "home_qpos": list(ARM_GRIPPER_HOME_QPOS),
        "model_scale": ARM_GRIPPER_MODEL_SCALE,
        "render_model_scale": ARM_GRIPPER_RENDER_MODEL_SCALE,
        "source": "google-deepmind/mujoco_menagerie/universal_robots_ur5e",
    }
    gripper_spec = {
        "name": "robotiq_2f85",
        "robot_model": "robotiq_2f85",
        "joint_names": list(ARM_GRIPPER_ROBOTIQ_JOINT_NAMES),
        "actuator_name": "rq_fingers_actuator",
        "open_command": 0.0,
        "close_command": 1.0,
        "source": "google-deepmind/mujoco_menagerie/robotiq_2f85",
    }
    metadata.update(
        {
            "policy_type": policy_type,
            "planned_outcome": planned_outcome,
            "outcome": None,
            "capture_frame": None,
            "miss_frame": None,
            "truncated_before_interaction": False,
            "terminal_reason": None,
            "random_seed": BASE_SEED + 10000 * TASKS.index("arm_gripper_ball") + episode_index,
            "ball_initial_state": {
                "position": list(initial_position),
                "linear_velocity": list(initial_velocity),
                "radius": radius,
                "mass": mass,
                "fixed_radius_default": ARM_GRIPPER_BALL_RADIUS,
                "unscaled_radius": unscaled_radius,
                "debug_scale": ARM_GRIPPER_BALL_SCALE,
                "legacy_debug_scale": ARM_GRIPPER_DEBUG_SCALE,
                "scale_note": "robotiq_grasp_scale_ball_with_arm_catcher_style_launch",
                "launch_plan": launch_metadata,
            },
            "catcher_initial_state": {
                "position": list(gripper_initial_target),
                "target_position": list(gripper_initial_target),
                "workspace_min": list(ARM_GRIPPER_WORKSPACE_MIN),
                "workspace_max": list(ARM_GRIPPER_WORKSPACE_MAX),
                "planned_miss_bias_xyz": list(miss_bias),
            },
            "catcher_type": "standard_ur5e_with_robotiq_2f85_gripper",
            "environment_reference_task": "arm_catcher_ball",
            "ball_scale_policy": "robotiq_scale_ball_with_arm_catcher_environment",
            "ball_dynamics_reference": "arm_catcher_ball",
            "outcome_ratio_target": {"capture": 0.80, "miss": 0.15, "truncated": 0.05},
            "capture_tuning_note": "capture episodes use a slightly slower gripper-scale crossing window and a tighter visible catch band for physical Robotiq grasps",
            "physical_model_scale": ARM_GRIPPER_MODEL_SCALE,
            "debug_render_model_scale": ARM_GRIPPER_RENDER_MODEL_SCALE,
            "rendered_ground_half_extents": list(ARM_GRIPPER_GROUND_HALF_EXTENTS),
            "nominal_catch_depth_y": sum(ARM_GRIPPER_CATCH_Y_RANGE) * 0.5,
            "catch_front_y": ARM_GRIPPER_CATCH_FRONT_Y,
            "latch_after_capture": True,
            "action_space": "normalized_end_effector_delta_xyz_plus_binary_grip",
            "normalized_action_bounds": {"delta_xyz": [-1.0, 1.0], "grip": [0.0, 1.0]},
            "action_scale_xyz": list(ARM_GRIPPER_ACTION_SCALE_XYZ),
            "grip_action_mapping": {"open": 0.0, "close": 1.0, "actuator_targets": [0.0, 255.0]},
            "arm_initial_state": {
                "base_position": list(ARM_GRIPPER_BASE_POSITION),
                "base_euler": list(ARM_GRIPPER_BASE_EULER),
                "joint_names": list(ARM_GRIPPER_JOINT_NAMES),
                "joint_limits": [list(limits) for limits in ARM_GRIPPER_JOINT_LIMITS],
                "home_qpos": list(ARM_GRIPPER_HOME_QPOS),
                "model_scale": ARM_GRIPPER_MODEL_SCALE,
                "render_model_scale": ARM_GRIPPER_RENDER_MODEL_SCALE,
                "robot_model": "ur5e",
                "source": arm_spec["source"],
            },
            "gripper_initial_state": {
                "robot_model": "robotiq_2f85",
                "open_command": 0.0,
                "close_command": 1.0,
                "joint_names": list(ARM_GRIPPER_ROBOTIQ_JOINT_NAMES),
                "source": gripper_spec["source"],
            },
            "camera_parameters": {
                "location": list(ARM_GRIPPER_CAMERA_LOCATION),
                "target": list(ARM_GRIPPER_CAMERA_TARGET),
                "fovy": ARM_GRIPPER_CAMERA_FOVY_DEGREES,
            },
        }
    )

    catcher = {
        "name": "arm_gripper",
        "initial_position": list(gripper_initial_target),
        "initial_target_position": list(gripper_initial_target),
        "workspace_min": list(ARM_GRIPPER_WORKSPACE_MIN),
        "workspace_max": list(ARM_GRIPPER_WORKSPACE_MAX),
        "fixed_y": gripper_initial_target[1],
        "target_y": gripper_initial_target[1],
        "control_axes": ["x", "y", "z"],
        "action_scale_xyz": list(ARM_GRIPPER_ACTION_SCALE_XYZ),
        "policy_type": policy_type,
        "planned_outcome": planned_outcome,
        "miss_bias": list(miss_bias),
        "desired_ball_local_position": [0.04, -0.055, -0.02],
    }
    return {
        "metadata": metadata,
        "camera": make_camera(
            ARM_GRIPPER_CAMERA_LOCATION,
            ARM_GRIPPER_CAMERA_TARGET,
            ARM_GRIPPER_CAMERA_FOVY_DEGREES,
        ),
        "static_objects": build_approach_ball_ground(ARM_GRIPPER_GROUND_HALF_EXTENTS),
        "dynamic_objects": [dynamic_object],
        "catcher": catcher,
        "arm": arm_spec,
        "gripper": gripper_spec,
    }


def build_episode_scene(task_name, rng, gravity, episode_index, split_name):
    builders = {
        "open_box": build_open_box_episode,
        "approach_ball": build_approach_ball_episode,
        "catcher_ball": build_catcher_ball_episode,
        "paddle_ball": build_paddle_ball_episode,
        "paddle_ball_zonly": build_paddle_ball_zonly_episode,
        "arm_catcher_ball": build_arm_catcher_ball_episode,
        "arm_gripper_ball": build_arm_gripper_ball_episode,
        "arm_paddle_ball": build_arm_paddle_ball_episode,
    }
    return builders[task_name](rng, gravity, episode_index, split_name)


def acceptable_arm_gripper_capture(result):
    if result.get("outcome") != "capture":
        return False
    metadata = result.get("metadata", {})
    capture_frame = metadata.get("capture_frame")
    close_start_frame = metadata.get("close_start_frame")
    if capture_frame is None or close_start_frame is None:
        return False
    if capture_frame < close_start_frame:
        return False
    if capture_frame - close_start_frame > ARM_GRIPPER_CAPTURE_MAX_CLOSE_LEAD_FRAMES:
        return False
    latch_jump = float(metadata.get("latch_world_jump") or 0.0)
    if latch_jump > 0.08:
        return False
    contact_flags = metadata.get("latch_contact_flags") or {}
    mouth_metrics = metadata.get("latch_mouth_metrics") or {}
    pad_contact = contact_flags.get("ball_gripper_left_pad_contact") or contact_flags.get(
        "ball_gripper_right_pad_contact"
    )
    environment_contact = contact_flags.get("ball_floor_contact") or contact_flags.get(
        "ball_wall_contact"
    )
    return bool(
        mouth_metrics.get("inside_gripper_mouth") and pad_contact and not environment_contact
    )


def acceptable_arm_catcher_capture(result):
    metadata = result.get("metadata", {})
    if (
        metadata.get("planned_outcome") != "capture"
        or result.get("outcome") != "capture"
        or not result.get("success")
        or metadata.get("capture_frame") is None
    ):
        return False
    capture_time = float(metadata["capture_frame"]) / FPS
    capture_time_min, capture_time_max = (
        metadata.get("capture_cross_time_range") or ARM_CATCHER_CAPTURE_CROSS_TIME_RANGE
    )
    if not float(capture_time_min) <= capture_time <= float(capture_time_max):
        return False
    mount_validation = metadata.get("arm_catcher_mount_validation", {})
    min_mount_travel = float(
        metadata.get("capture_min_mount_travel", ARM_CATCHER_CAPTURE_MIN_MOUNT_TRAVEL)
    )
    mount_travel = float(mount_validation.get("start_to_reference_mount_distance") or 0.0)
    if mount_travel < min_mount_travel:
        return False
    max_mount_delta = float(mount_validation.get("max_frame_mount_delta") or 0.0)
    max_allowed_mount_delta = float(
        metadata.get(
            "capture_max_frame_mount_delta",
            ARM_CATCHER_CAPTURE_MAX_FRAME_MOUNT_DELTA,
        )
    )
    if max_mount_delta > max_allowed_mount_delta:
        return False
    max_mount_acceleration_delta = float(
        mount_validation.get("max_frame_mount_acceleration_delta") or 0.0
    )
    max_allowed_mount_acceleration_delta = float(
        metadata.get(
            "capture_max_frame_mount_acceleration_delta",
            ARM_CATCHER_CAPTURE_MAX_FRAME_MOUNT_ACCELERATION_DELTA,
        )
    )
    if max_mount_acceleration_delta > max_allowed_mount_acceleration_delta:
        return False
    latch_jump = metadata.get("latch_world_jump")
    max_latch_jump = float(
        metadata.get("capture_max_latch_jump", ARM_CATCHER_CAPTURE_MAX_LATCH_JUMP)
    )
    if latch_jump is None or float(latch_jump) > max_latch_jump:
        return False
    if metadata.get("screen_right_capture_required"):
        screen_initial_x = metadata.get("screen_initial_x")
        if screen_initial_x is None:
            screen_initial_x = projected_screen_x(
                metadata["ball_initial_state"]["position"],
                result["camera"],
            )
        min_screen_x = float(
            metadata.get("screen_right_capture_min_x", ARM_CATCHER_SCREEN_RIGHT_MIN_X)
        )
        if float(screen_initial_x) < min_screen_x:
            return False
    return True


def episode_jobs():
    jobs = []
    for task_name in TASKS:
        assignments = build_episode_assignments(task_name)
        selected = [
            (source_index, split_name, gravity)
            for source_index, (split_name, gravity) in enumerate(assignments)
            if SELECTED_SPLIT == "all" or split_name == SELECTED_SPLIT
        ]
        if len(selected) != EPISODES_PER_TASK:
            raise ValueError(
                f"selected {len(selected)} {SELECTED_SPLIT} episodes, expected {EPISODES_PER_TASK}"
            )
        for episode_index, (source_index, split_name, gravity) in enumerate(selected):
            jobs.append((task_name, episode_index, source_index, split_name, gravity))
    return jobs


def retry_source_indices(task_name, task_index, episode_index, max_attempts):
    if max_attempts <= 1:
        return [episode_index]
    if task_name != "arm_catcher_ball":
        source_indices = [episode_index]
        source_indices.extend(
            source_index for source_index in range(max_attempts) if source_index != episode_index
        )
        return source_indices[:max_attempts]

    rng = random.Random(BASE_SEED + 10000 * task_index + 1009 * (episode_index + 1))
    source_indices = [episode_index]
    seen = {episode_index}
    candidate_span = max(max_attempts * 64, EPISODES_PER_TASK * 64, 4096)
    while len(source_indices) < max_attempts:
        candidate = rng.randrange(candidate_span)
        if candidate in seen:
            continue
        seen.add(candidate)
        source_indices.append(candidate)
    return source_indices


def arm_catcher_capture_prescreen_reason(scene):
    """Reject geometrically doomed capture attempts without simulating.

    Both predicates below are pure functions of the sampled launch and the
    fixed camera/mount, so a failing attempt can be skipped for the cost of
    scene construction instead of a full episode simulation.
    """
    metadata = scene["metadata"]
    if metadata.get("screen_right_capture_required"):
        min_screen_x = float(
            metadata.get("screen_right_capture_min_x", ARM_CATCHER_SCREEN_RIGHT_MIN_X)
        )
        if float(metadata["screen_initial_x"]) < min_screen_x:
            return "screen_initial_x_below_min"
    ball_state = metadata["ball_initial_state"]
    start_mount = metadata["catcher_initial_state"]["position"]
    # Propagate the actual (noise-inclusive) launch to the catch plane; the
    # planned cross point ignores the sampled velocity noise, which shifts
    # the realized crossing by up to ~0.05 scene units.
    crossing, _time_to_plane = demo_generate.predict_ball_at_y_analytic(
        ball_state["position"],
        ball_state["linear_velocity"],
        gravity=abs(float(metadata["gravity"][2])),
        radius=float(ball_state["radius"]),
        catch_y=ARM_CATCHER_CATCH_FRONT_Y,
    )
    predicted_mount_z = min(
        max(float(crossing[2]), ARM_CATCHER_PLANNER_MIN_Z),
        ARM_CATCHER_WORKSPACE_MAX[2],
    )
    predicted_travel = math.hypot(
        float(crossing[0]) - float(start_mount[0]),
        predicted_mount_z - float(start_mount[2]),
    )
    if predicted_travel < 0.85 * ARM_CATCHER_CAPTURE_MIN_MOUNT_TRAVEL:
        return "predicted_mount_travel_below_min"
    return None


def generate_episode_job(job):
    task_name, output_episode_index, episode_index, split_name, gravity = job
    episode_root = episode_dir(task_name, output_episode_index)
    ensure_dir(episode_root)
    output_json = episode_root / "scene.json"
    task_index = TASKS.index(task_name)
    planned_outcome = (
        choose_catcher_ball_planned_outcome(episode_index)
        if task_name == "arm_gripper_ball"
        else choose_arm_catcher_ball_planned_outcome(episode_index)
        if task_name == "arm_catcher_ball"
        else None
    )
    if task_name == "arm_gripper_ball" and planned_outcome == "capture":
        max_attempts = ARM_GRIPPER_CAPTURE_RETRY_LIMIT
    elif task_name == "arm_catcher_ball" and planned_outcome == "capture":
        max_attempts = ARM_CATCHER_CAPTURE_RETRY_LIMIT
    else:
        max_attempts = 1
    source_indices = retry_source_indices(
        task_name,
        task_index,
        episode_index,
        max_attempts,
    )

    capture_accepted = planned_outcome != "capture"
    for attempt_index, source_index in enumerate(source_indices):
        source_seed = BASE_SEED + 10000 * task_index + source_index
        rng = random.Random(source_seed)
        scene = build_episode_scene(task_name, rng, gravity, episode_index, split_name)
        scene["metadata"]["source_episode_index"] = episode_index
        scene["metadata"]["output_episode_index"] = output_episode_index
        if task_name in {"arm_gripper_ball", "arm_catcher_ball"}:
            scene["metadata"]["episode_random_seed"] = (
                BASE_SEED + 10000 * task_index + episode_index
            )
            scene["metadata"]["random_seed"] = source_seed
            scene["metadata"]["random_seed_source_episode_index"] = source_index
            scene["metadata"]["capture_retry_attempt"] = attempt_index
            scene["metadata"]["capture_retry_limit"] = max_attempts
            scene["metadata"]["capture_retry_enabled"] = planned_outcome == "capture"
        if (
            task_name == "arm_catcher_ball"
            and planned_outcome == "capture"
            and attempt_index < len(source_indices) - 1
            and arm_catcher_capture_prescreen_reason(scene) is not None
        ):
            continue
        demo_generate.simulate_scene(
            scene,
            output_json,
            fps=FPS,
            frames_per_episode=FRAMES_PER_EPISODE,
            step_rate=STEP_RATE,
            steps_per_frame=STEPS_PER_FRAME,
        )
        if planned_outcome != "capture":
            break
        result = json.loads(output_json.read_text())
        if task_name == "arm_gripper_ball" and acceptable_arm_gripper_capture(result):
            capture_accepted = True
            break
        if task_name == "arm_catcher_ball" and acceptable_arm_catcher_capture(result):
            capture_accepted = True
            break
    if task_name in {"arm_gripper_ball", "arm_catcher_ball"} and planned_outcome == "capture":
        result = json.loads(output_json.read_text())
        result["metadata"]["capture_retry_exhausted"] = not capture_accepted
        result["metadata"]["capture_retry_accepted"] = capture_accepted
        output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return output_json


def main():
    for task_name in TASKS:
        ensure_clean_dir(task_output_dir(task_name))
    jobs = episode_jobs()
    ctx = mp.get_context(RENDER_START_METHOD)
    with ProcessPoolExecutor(max_workers=SIM_WORKERS, mp_context=ctx) as executor:
        futures = {executor.submit(generate_episode_job, job): job for job in jobs}
        for future in as_completed(futures):
            output_json = future.result()
            print(f"Wrote {output_json}", flush=True)


if __name__ == "__main__":
    main()
