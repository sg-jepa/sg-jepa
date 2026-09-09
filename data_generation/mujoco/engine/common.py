import math
import os
import shutil
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")


ROOT_DIR = Path(__file__).resolve().parent
WORK_DIR = Path(os.environ.get("DEMO_MUJOCO_WORK_DIR", ROOT_DIR)).expanduser()
if not WORK_DIR.is_absolute():
    WORK_DIR = (ROOT_DIR / WORK_DIR).resolve()

DATA_ROOT = WORK_DIR / "data"
OUTPUT_ROOT = WORK_DIR / "outputs"
STAGING_ROOT = WORK_DIR / "_staging"
SHARD_ROOT = STAGING_ROOT / "shards"

ALL_TASKS = [
    "open_box",
    "approach_ball",
    "catcher_ball",
    "paddle_ball",
    "paddle_ball_zonly",
    "arm_catcher_ball",
    "arm_gripper_ball",
    "arm_paddle_ball",
]
_TASKS_OVERRIDE = os.environ.get("DEMO_MUJOCO_TASKS", "").strip()
if _TASKS_OVERRIDE:
    _requested_tasks = []
    for _task_name in _TASKS_OVERRIDE.split(","):
        _task_name = _task_name.strip()
        if _task_name and _task_name not in _requested_tasks:
            _requested_tasks.append(_task_name)
    _invalid_tasks = [name for name in _requested_tasks if name not in ALL_TASKS]
    if _invalid_tasks:
        raise ValueError(
            f"Unknown task names in DEMO_MUJOCO_TASKS: {_invalid_tasks}. "
            f"Valid tasks are: {ALL_TASKS}"
        )
    TASKS = _requested_tasks
else:
    TASKS = list(ALL_TASKS)

ALL_OUTPUT_VERSIONS = ["v0"]
_VERSIONS_OVERRIDE = os.environ.get("DEMO_MUJOCO_OUTPUT_VERSIONS", "").strip()
if _VERSIONS_OVERRIDE:
    _requested_versions = []
    for _version_name in _VERSIONS_OVERRIDE.split(","):
        _version_name = _version_name.strip()
        if _version_name and _version_name not in _requested_versions:
            _requested_versions.append(_version_name)
    _invalid_versions = [name for name in _requested_versions if name not in ALL_OUTPUT_VERSIONS]
    if _invalid_versions:
        raise ValueError(
            f"Unknown versions in DEMO_MUJOCO_OUTPUT_VERSIONS: {_invalid_versions}. "
            f"Valid versions are: {ALL_OUTPUT_VERSIONS}"
        )
    OUTPUT_VERSIONS = _requested_versions
else:
    OUTPUT_VERSIONS = list(ALL_OUTPUT_VERSIONS)

_SHAPE_MODE = os.environ.get("DEMO_MUJOCO_DYNAMIC_SHAPE_MODE", "mixed").strip().lower()
if _SHAPE_MODE in {"mixed", "ball_only", "circle_only", "sphere_only"}:
    DYNAMIC_SHAPE_MODE = "mixed" if _SHAPE_MODE == "mixed" else "ball_only"
elif _SHAPE_MODE in {"cube_only", "box_only", "square_only"}:
    DYNAMIC_SHAPE_MODE = "cube_only"
else:
    raise ValueError(
        "Unknown DEMO_MUJOCO_DYNAMIC_SHAPE_MODE="
        f"{_SHAPE_MODE!r}. Valid values are: mixed, ball_only, circle_only, "
        "sphere_only, cube_only, box_only, square_only."
    )

EPISODES_PER_TASK = int(os.environ.get("DEMO_MUJOCO_EPISODES_PER_TASK", "30"))
CANONICAL_EPISODES_PER_TASK = int(
    os.environ.get("DEMO_MUJOCO_CANONICAL_EPISODES_PER_TASK", str(EPISODES_PER_TASK))
)
SELECTED_SPLIT = os.environ.get("DEMO_MUJOCO_SELECTED_SPLIT", "all").strip().lower()
_DEFAULT_FRAMES_PER_EPISODE = "128"
FRAMES_PER_EPISODE = int(
    os.environ.get("DEMO_MUJOCO_FRAMES_PER_EPISODE", _DEFAULT_FRAMES_PER_EPISODE)
)
FPS = int(os.environ.get("DEMO_MUJOCO_FPS", "32"))
STEPS_PER_FRAME = int(os.environ.get("DEMO_MUJOCO_STEPS_PER_FRAME", "20"))
STEP_RATE = int(os.environ.get("DEMO_MUJOCO_STEP_RATE", str(FPS * STEPS_PER_FRAME)))
if EPISODES_PER_TASK <= 0 or CANONICAL_EPISODES_PER_TASK <= 0 or FRAMES_PER_EPISODE <= 0:
    raise ValueError(
        f"DEMO_MUJOCO_EPISODES_PER_TASK and DEMO_MUJOCO_FRAMES_PER_EPISODE "
        f"must be positive; got episodes={EPISODES_PER_TASK}, frames={FRAMES_PER_EPISODE}"
    )
if SELECTED_SPLIT not in {"train", "test", "all"}:
    raise ValueError("DEMO_MUJOCO_SELECTED_SPLIT must be train, test, or all")
if FPS <= 0 or STEPS_PER_FRAME <= 0 or STEP_RATE <= 0:
    raise ValueError(
        f"DEMO_MUJOCO_FPS, DEMO_MUJOCO_STEPS_PER_FRAME, and DEMO_MUJOCO_STEP_RATE "
        f"must be positive; got fps={FPS}, steps_per_frame={STEPS_PER_FRAME}, step_rate={STEP_RATE}"
    )
DURATION_SECONDS = FRAMES_PER_EPISODE / FPS

IMAGE_WIDTH = int(os.environ.get("DEMO_MUJOCO_IMAGE_WIDTH", "256"))
IMAGE_HEIGHT = int(os.environ.get("DEMO_MUJOCO_IMAGE_HEIGHT", "256"))
if IMAGE_WIDTH <= 0 or IMAGE_HEIGHT <= 0:
    raise ValueError(
        f"DEMO_MUJOCO_IMAGE_WIDTH and DEMO_MUJOCO_IMAGE_HEIGHT must be positive; "
        f"got width={IMAGE_WIDTH}, height={IMAGE_HEIGHT}"
    )

CPU_COUNT = os.cpu_count() or 1
SIM_WORKERS = min(8, CPU_COUNT)
RENDER_WORKERS = min(4, CPU_COUNT)
PACK_WORKERS = min(len(TASKS), SIM_WORKERS)
RENDER_START_METHOD = "spawn"

RENDER_SHARD_EPISODES = int(os.environ.get("DEMO_MUJOCO_SHARD_EPISODES", min(8, EPISODES_PER_TASK)))
if RENDER_SHARD_EPISODES <= 0:
    raise ValueError(f"DEMO_MUJOCO_SHARD_EPISODES must be positive; got {RENDER_SHARD_EPISODES}")

BASE_SEED = int(os.environ.get("DEMO_MUJOCO_BASE_SEED", "20260525"))
GRAVITY_BATCH_EPISODES = int(
    os.environ.get("DEMO_MUJOCO_GRAVITY_BATCH_EPISODES", str(EPISODES_PER_TASK))
)
_DEFAULT_TRAIN_EPISODES_PER_BATCH = min(GRAVITY_BATCH_EPISODES, 5)
_DEFAULT_TEST_EPISODES_PER_BATCH = GRAVITY_BATCH_EPISODES - _DEFAULT_TRAIN_EPISODES_PER_BATCH
TRAIN_EPISODES_PER_BATCH = int(
    os.environ.get(
        "DEMO_MUJOCO_TRAIN_EPISODES_PER_BATCH",
        str(_DEFAULT_TRAIN_EPISODES_PER_BATCH),
    )
)
TEST_EPISODES_PER_BATCH = int(
    os.environ.get(
        "DEMO_MUJOCO_TEST_EPISODES_PER_BATCH",
        str(_DEFAULT_TEST_EPISODES_PER_BATCH),
    )
)
if GRAVITY_BATCH_EPISODES <= 0 or TRAIN_EPISODES_PER_BATCH < 0 or TEST_EPISODES_PER_BATCH < 0:
    raise ValueError(
        "DEMO_MUJOCO_GRAVITY_BATCH_EPISODES must be positive, and train/test "
        "episode counts must be non-negative; "
        f"got gravity_batch_episodes={GRAVITY_BATCH_EPISODES}, "
        f"train_episodes_per_batch={TRAIN_EPISODES_PER_BATCH}, "
        f"test_episodes_per_batch={TEST_EPISODES_PER_BATCH}"
    )
if TRAIN_EPISODES_PER_BATCH + TEST_EPISODES_PER_BATCH != GRAVITY_BATCH_EPISODES:
    raise ValueError(
        "DEMO_MUJOCO_TRAIN_EPISODES_PER_BATCH + DEMO_MUJOCO_TEST_EPISODES_PER_BATCH "
        "must equal DEMO_MUJOCO_GRAVITY_BATCH_EPISODES; "
        f"got {TRAIN_EPISODES_PER_BATCH} + {TEST_EPISODES_PER_BATCH} != "
        f"{GRAVITY_BATCH_EPISODES}"
    )

ELASTIC_RESTITUTION = 1.0

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
ACTION_SCHEMA = ["g"]
CATCHER_BALL_ACTION_SCHEMA = ["g", "catcher_dx", "catcher_dz"]
ARM_CATCHER_BALL_ACTION_SCHEMA = ["g", "dx", "dy", "dz"]
ARM_GRIPPER_BALL_ACTION_SCHEMA = ["g", "dx", "dy", "dz", "grip"]
PADDLE_BALL_ACTION_SCHEMA = ["g", "dx", "dy", "dz", "phi", "theta"]
PADDLE_BALL_ZONLY_ACTION_SCHEMA = ["g", "dx", "dy", "dz"]
PADDLE_BALL_TASKS = {"paddle_ball", "paddle_ball_zonly", "arm_paddle_ball"}
CATCHER_STATE_SCHEMA = [
    "catcher_x",
    "catcher_y",
    "catcher_z",
    "catcher_vx",
    "catcher_vy",
    "catcher_vz",
    "catcher_target_x",
    "catcher_target_y",
    "catcher_target_z",
]
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
CONTACT_SCHEMA = ["ball_catcher_contact", "ball_floor_contact", "ball_wall_contact"]
PADDLE_BALL_CONTACT_SCHEMA = ["ball_paddle_contact", "ball_floor_contact", "ball_wall_contact"]
ARM_GRIPPER_CONTACT_SCHEMA = [
    "ball_gripper_left_pad_contact",
    "ball_gripper_right_pad_contact",
    "ball_gripper_palm_contact",
    "ball_floor_contact",
    "ball_wall_contact",
]
CATCHER_POLICY_ID_MAP = {"expert": 0, "noisy_delayed": 1, "random": 2}
PADDLE_POLICY_ID_MAP = {"expert": 0}
ARM_STATE_SCHEMA = [
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
]
GRIPPER_STATE_SCHEMA = [
    "grip_command",
    "actuator_target",
    "opening_width",
    "left_driver_joint",
    "right_driver_joint",
    "left_driver_joint_vel",
    "right_driver_joint_vel",
    "left_pad_x",
    "left_pad_y",
    "left_pad_z",
    "right_pad_x",
    "right_pad_y",
    "right_pad_z",
    "left_collision_pad_x",
    "left_collision_pad_y",
    "left_collision_pad_z",
    "right_collision_pad_x",
    "right_collision_pad_y",
    "right_collision_pad_z",
    "grasp_center_x",
    "grasp_center_y",
    "grasp_center_z",
]
END_EFFECTOR_STATE_SCHEMA = [
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_qx",
    "ee_qy",
    "ee_qz",
    "ee_qw",
    "ee_vx",
    "ee_vy",
    "ee_vz",
    "target_x",
    "target_y",
    "target_z",
]
PHYS_SCHEMA = [
    "g",
    "mass",
    "size_x",
    "size_y",
    "size_z",
    "shape_id",
    "box_half_x",
    "box_half_y",
    "box_half_z",
    "wall_thickness",
]
SHAPE_ID_MAP = {"ball": 0, "cube": 1}
SPLIT_TO_ID = {"train": 0, "test": 1}
TRAIN_GAUSSIAN_MEAN = float(os.environ.get("DEMO_MUJOCO_TRAIN_GAUSSIAN_MEAN", "9.8"))
TRAIN_GAUSSIAN_STD = float(os.environ.get("DEMO_MUJOCO_TRAIN_GAUSSIAN_STD", "2.0"))
TRAIN_G_MIN = float(os.environ.get("DEMO_MUJOCO_TRAIN_G_MIN", "0.0"))
_TEST_GRAVITIES_OVERRIDE = os.environ.get("DEMO_MUJOCO_TEST_GRAVITIES", "").strip()
if _TEST_GRAVITIES_OVERRIDE:
    TEST_GRAVITIES = [
        float(value.strip()) for value in _TEST_GRAVITIES_OVERRIDE.split(",") if value.strip()
    ]
    if not TEST_GRAVITIES and TEST_EPISODES_PER_BATCH:
        raise ValueError("DEMO_MUJOCO_TEST_GRAVITIES must contain at least one value.")
else:
    TEST_GRAVITIES = [
        *[float(value) for value in range(21)],
        8.87,
        3.721,
        1.625,
        0.62,
    ]

TASK_PARAM_MEANINGS = {
    "open_box": ["inner_half_x", "inner_half_y", "inner_half_z", "wall_thickness"],
    "approach_ball": ["ground_half_x", "ground_half_y", "side_wall_height", "camera_elevation_deg"],
    "catcher_ball": ["ground_half_x", "ground_half_y", "side_wall_height", "camera_elevation_deg"],
    "paddle_ball": ["ground_half_x", "ground_half_y", "side_wall_height", "camera_elevation_deg"],
    "paddle_ball_zonly": [
        "ground_half_x",
        "ground_half_y",
        "side_wall_height",
        "camera_elevation_deg",
    ],
    "arm_catcher_ball": [
        "ground_half_x",
        "ground_half_y",
        "side_wall_height",
        "camera_elevation_deg",
    ],
    "arm_gripper_ball": [
        "ground_half_x",
        "ground_half_y",
        "side_wall_height",
        "camera_elevation_deg",
    ],
    "arm_paddle_ball": [
        "ground_half_x",
        "ground_half_y",
        "side_wall_height",
        "camera_elevation_deg",
    ],
}


def task_has_catcher_fields(task_name):
    return task_name in {"catcher_ball", "arm_catcher_ball", "arm_gripper_ball"}


def task_has_paddle_fields(task_name):
    return task_name in PADDLE_BALL_TASKS


def task_has_arm_fields(task_name):
    return task_name in {"arm_catcher_ball", "arm_gripper_ball", "arm_paddle_ball"}


def task_has_gripper_fields(task_name):
    return task_name == "arm_gripper_ball"


def task_action_schema(task_name):
    if task_name == "catcher_ball":
        return list(CATCHER_BALL_ACTION_SCHEMA)
    if task_name == "arm_catcher_ball":
        return list(ARM_CATCHER_BALL_ACTION_SCHEMA)
    if task_name == "arm_gripper_ball":
        return list(ARM_GRIPPER_BALL_ACTION_SCHEMA)
    if task_name in {"paddle_ball", "arm_paddle_ball"}:
        return list(PADDLE_BALL_ACTION_SCHEMA)
    if task_name == "paddle_ball_zonly":
        return list(PADDLE_BALL_ZONLY_ACTION_SCHEMA)
    return list(ACTION_SCHEMA)


def task_contact_schema(task_name):
    if task_name == "arm_gripper_ball":
        return list(ARM_GRIPPER_CONTACT_SCHEMA)
    if task_name in PADDLE_BALL_TASKS:
        return list(PADDLE_BALL_CONTACT_SCHEMA)
    return list(CONTACT_SCHEMA)


def ensure_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def episode_dir(task_name, episode_index):
    return OUTPUT_ROOT / task_name / f"episode_{episode_index:03d}"


def scene_json_path(task_name, episode_index):
    return episode_dir(task_name, episode_index) / "scene.json"


def task_output_dir(task_name):
    return DATA_ROOT / task_name


def task_version_dir(task_name, version):
    return task_output_dir(task_name) / version


def task_version_shard_dir(task_name, version):
    return SHARD_ROOT / task_name / version


def shard_path(task_name, version, shard_index):
    return task_version_shard_dir(task_name, version) / f"shard_{shard_index:03d}.h5"


def shard_count():
    return math.ceil(EPISODES_PER_TASK / RENDER_SHARD_EPISODES)


def shard_ranges():
    ranges = []
    for start in range(0, EPISODES_PER_TASK, RENDER_SHARD_EPISODES):
        stop = min(start + RENDER_SHARD_EPISODES, EPISODES_PER_TASK)
        shard_index = start // RENDER_SHARD_EPISODES
        ranges.append((shard_index, start, stop))
    return ranges
