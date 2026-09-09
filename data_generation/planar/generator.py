"""Square and Right Triangle MuJoCo generator.

Physics constants, RNG ordering, action/state fields, and the paper-scale split
policy follow generator revision ``6473211``. Bounded mode changes only the
episode/frame/image counts so the actual simulator can be tested cheaply.
"""

from __future__ import annotations

import json
import math
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from data_generation.common.lance import EpisodeRecord, write_lance_episodes

BASE_SEED = 20260525
FPS = 16
STEPS_PER_FRAME = 20
SIMULATION_HZ = FPS * STEPS_PER_FRAME
TEST_GRAVITIES = tuple(float(value) for value in np.arange(-2.0, 10.01, 0.5))
ACTION_NAMES = ("impulse_x", "impulse_z", "g")
STATE_NAMES = ("x", "z", "vx", "vz", "theta", "omega", "anchor_x", "anchor_z")
SPLIT_TO_ID = {"train": 0, "test": 1}

BOX_CENTER_X = 0.0
BOX_CENTER_Z = 4.6
BOX_INNER_HALF_WIDTH = 4.8
BOX_INNER_HALF_HEIGHT = 4.8
BOX_WALL_THICKNESS = 0.40
BOX_OBJECT_EDGE_MARGIN = 0.22
BOX_CAMERA_ORTHO_SCALE = 10.4
BOX_INITIAL_SPEED_MIN = 0.35
BOX_INITIAL_SPEED_MAX = 1.10


def _format(values: Any) -> str:
    if np.isscalar(values):
        return f"{float(values):.10g}"
    return " ".join(f"{float(value):.10g}" for value in values)


def _triangle_mesh(hypotenuse: float, half_depth: float) -> tuple[list[float], list[int]]:
    leg = hypotenuse / math.sqrt(2.0)
    vertices_xz = (
        (-leg / 3.0, -leg / 3.0),
        (2.0 * leg / 3.0, -leg / 3.0),
        (-leg / 3.0, 2.0 * leg / 3.0),
    )
    vertices = [
        coordinate
        for y in (-half_depth, half_depth)
        for x, z in vertices_xz
        for coordinate in (x, y, z)
    ]
    faces = [0, 1, 2, 3, 5, 4]
    for index in range(3):
        following = (index + 1) % 3
        faces.extend((index, following, 3 + following, index, 3 + following, 3 + index))
    return vertices, faces


def _assignments(
    episodes: int,
    seed: int,
    *,
    paper_scale: bool,
    split: str,
) -> list[tuple[int, str, float]]:
    if paper_scale:
        expected = {"all": 48_000, "train": 8_000, "test": 40_000}[split]
        if episodes != expected:
            raise ValueError(f"paper-scale {split} generation requires episodes={expected}")
        assignments: list[tuple[int, str, float]] = []
        for batch in range(48_000 // 30):
            rng = random.Random(seed + 7919 * batch)
            group = [("train", max(rng.gauss(4.0, 0.5), 0.1)) for _ in range(5)] + [
                ("test", gravity) for gravity in TEST_GRAVITIES
            ]
            rng.shuffle(group)
            assignments.extend((30 * batch + offset, *item) for offset, item in enumerate(group))
        if split != "all":
            assignments = [item for item in assignments if item[1] == split]
        return assignments
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if split == "train":
        rng = random.Random(seed)
        return [(index, "train", max(rng.gauss(4.0, 0.5), 0.1)) for index in range(episodes)]
    if split == "test":
        return [
            (
                index,
                "test",
                TEST_GRAVITIES[round(index * (len(TEST_GRAVITIES) - 1) / max(1, episodes - 1))],
            )
            for index in range(episodes)
        ]
    if episodes < 2:
        raise ValueError("split=all requires at least two episodes")
    train_count = max(2, episodes // 2)
    train_count = min(train_count, episodes - 1)
    test_count = episodes - train_count
    rng = random.Random(seed)
    train = [("train", max(rng.gauss(4.0, 0.5), 0.1)) for _ in range(train_count)]
    test = [
        ("test", TEST_GRAVITIES[round(index * (len(TEST_GRAVITIES) - 1) / max(1, test_count - 1))])
        for index in range(test_count)
    ]
    return [(index, *item) for index, item in enumerate(train + test)]


def _episode_parameters(
    task: str, episode_index: int, gravity: float, seed: int
) -> dict[str, float]:
    rng = random.Random(seed + episode_index)
    mass = rng.uniform(0.5, 2.0)
    rotation = rng.uniform(-0.6, 0.6)
    heading = rng.uniform(0.0, 2.0 * math.pi)
    speed = rng.uniform(BOX_INITIAL_SPEED_MIN, BOX_INITIAL_SPEED_MAX)
    # Preserve random.choice even for the one-item production shape list.
    rng.choice([task])
    if task == "square":
        size_x = size_z = 1.0
        shape_id = 1.0
    elif task == "right_triangle":
        leg = 2.0 / math.sqrt(2.0)
        size_x = size_z = math.sqrt(5.0) * leg / 3.0
        shape_id = 4.0
    else:
        raise ValueError("planar generator supports square and right_triangle")
    left = BOX_CENTER_X - BOX_INNER_HALF_WIDTH + size_x + BOX_OBJECT_EDGE_MARGIN
    right = BOX_CENTER_X + BOX_INNER_HALF_WIDTH - size_x - BOX_OBJECT_EDGE_MARGIN
    bottom = BOX_CENTER_Z - BOX_INNER_HALF_HEIGHT + size_z + BOX_OBJECT_EDGE_MARGIN
    top = BOX_CENTER_Z + BOX_INNER_HALF_HEIGHT - size_z - BOX_OBJECT_EDGE_MARGIN
    bottom = max(bottom, BOX_CENTER_Z + 0.12 * BOX_INNER_HALF_HEIGHT)
    return {
        "mass": mass,
        "rotation": rotation,
        "vx": speed * math.cos(heading),
        "vz": speed * math.sin(heading),
        "x": rng.uniform(left, right),
        "z": rng.uniform(bottom, top),
        "gravity": gravity,
        "size_x": size_x,
        "size_z": size_z,
        "shape_id": shape_id,
    }


def _model_xml(task: str, parameters: dict[str, float], image_size: int) -> str:
    root = ET.Element("mujoco", model=f"{task}_paper_generator")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(
        root,
        "option",
        timestep=_format(1.0 / SIMULATION_HZ),
        gravity=_format((0.0, 0.0, -parameters["gravity"])),
        integrator="implicitfast",
        solver="Newton",
        cone="elliptic",
        iterations="100",
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(image_size), offheight=str(image_size))
    ET.SubElement(
        visual,
        "headlight",
        ambient="1 1 1",
        diffuse="0 0 0",
        specular="0 0 0",
    )
    if task == "right_triangle":
        asset = ET.SubElement(root, "asset")
        vertices, faces = _triangle_mesh(2.0, 0.35)
        ET.SubElement(
            asset,
            "mesh",
            name="moving_object_right_triangle_mesh",
            vertex=_format(vertices),
            face=" ".join(str(value) for value in faces),
        )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "camera",
        name="render_camera",
        pos=_format((BOX_CENTER_X, -8.0, BOX_CENTER_Z)),
        xyaxes="1 0 0 0 0 1",
        projection="orthographic",
        fovy=_format(BOX_CAMERA_ORTHO_SCALE),
        resolution=f"{image_size} {image_size}",
    )
    ET.SubElement(
        world,
        "geom",
        name="background_panel",
        type="box",
        pos=_format((BOX_CENTER_X, 16.0, BOX_CENTER_Z)),
        size=_format((7.02, 0.02, 7.02)),
        rgba="1 1 1 1",
        contype="0",
        conaffinity="0",
    )
    dark = _format((0.005605, 0.005605, 0.005605, 1.0))
    half_wall = BOX_WALL_THICKNESS / 2.0
    walls = (
        ("left_wall", (half_wall, 0.55, 5.2), (-5.0, 0.0, 4.6)),
        ("right_wall", (half_wall, 0.55, 5.2), (5.0, 0.0, 4.6)),
        ("bottom_wall", (5.2, 0.55, half_wall), (0.0, 0.0, -0.4)),
        ("top_wall", (5.2, 0.55, half_wall), (0.0, 0.0, 9.6)),
    )
    for name, size, position in walls:
        ET.SubElement(
            world,
            "geom",
            name=name,
            type="box",
            size=_format(size),
            pos=_format(position),
            rgba=dark,
            condim="6",
            friction="0 0 0",
            solref="-1000 0",
        )
    body = ET.SubElement(world, "body", name="moving_object", pos="0 0 0")
    ET.SubElement(body, "joint", name="moving_object_slide_x", type="slide", axis="1 0 0")
    ET.SubElement(body, "joint", name="moving_object_slide_z", type="slide", axis="0 0 1")
    ET.SubElement(body, "joint", name="moving_object_hinge_y", type="hinge", axis="0 1 0")
    attributes = {
        "name": "moving_object",
        "mass": _format(parameters["mass"]),
        "rgba": _format((0.55, 0.12, 0.12, 1.0)),
        "condim": "6",
        "friction": "0 0 0",
        "solref": "-1000 0",
    }
    if task == "square":
        attributes.update(type="box", size="1 0.35 1")
    else:
        attributes.update(type="mesh", mesh="moving_object_right_triangle_mesh")
    ET.SubElement(body, "geom", attributes)
    return ET.tostring(root, encoding="unicode")


def _render_episode(
    task: str,
    parameters: dict[str, float],
    *,
    frames: int,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise ImportError("install sg-jepa[data] to generate MuJoCo datasets") from exc

    model = mujoco.MjModel.from_xml_string(_model_xml(task, parameters, image_size))
    data = mujoco.MjData(model)
    slide_x = model.joint("moving_object_slide_x")
    slide_z = model.joint("moving_object_slide_z")
    hinge = model.joint("moving_object_hinge_y")
    data.qpos[int(slide_x.qposadr[0])] = parameters["x"]
    data.qpos[int(slide_z.qposadr[0])] = parameters["z"]
    data.qpos[int(hinge.qposadr[0])] = parameters["rotation"]
    data.qvel[int(slide_x.dofadr[0])] = parameters["vx"]
    data.qvel[int(slide_z.dofadr[0])] = parameters["vz"]
    mujoco.mj_forward(model, data)
    pixels = np.empty((frames, image_size, image_size, 3), dtype=np.uint8)
    state = np.empty((frames, len(STATE_NAMES)), dtype=np.float32)
    action = np.zeros((frames, len(ACTION_NAMES)), dtype=np.float32)
    action[:, 2] = parameters["gravity"]
    action[0, :2] = parameters["mass"] * np.asarray(
        (parameters["vx"], parameters["vz"]), dtype=np.float32
    )
    renderer = mujoco.Renderer(model, height=image_size, width=image_size)
    try:
        for frame in range(frames):
            if frame:
                for _ in range(STEPS_PER_FRAME):
                    mujoco.mj_step(model, data)
            body = data.body("moving_object")
            quaternion = body.xquat
            theta = math.atan2(
                2.0 * (quaternion[0] * quaternion[2] - quaternion[3] * quaternion[1]),
                1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2),
            )
            state[frame] = (
                body.xpos[0],
                body.xpos[2],
                body.cvel[3],
                body.cvel[5],
                theta,
                body.cvel[1],
                BOX_CENTER_X,
                BOX_CENTER_Z,
            )
            renderer.update_scene(data, camera="render_camera")
            renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = 0
            pixels[frame] = renderer.render()
    finally:
        renderer.close()
    return pixels, action, state


def generate_planar_dataset(
    task: str,
    output_dir: str | Path,
    *,
    episodes: int = 4,
    frames: int = 8,
    image_size: int = 64,
    seed: int = BASE_SEED,
    split: str = "all",
    paper_scale: bool = False,
) -> dict[str, Any]:
    """Generate a one-row-per-frame Lance dataset with the MuJoCo simulator."""

    if task not in {"square", "right_triangle"}:
        raise ValueError("planar generator supports square and right_triangle")
    if split not in {"train", "test", "all"}:
        raise ValueError("split must be train, test, or all")
    if frames < 2 or image_size < 32:
        raise ValueError("frames must be >=2 and image_size must be >=32")
    if paper_scale and (frames, image_size) != (64, 128):
        raise ValueError("paper-scale generation requires frames=64 and image_size=128")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    assignments = _assignments(episodes, seed, paper_scale=paper_scale, split=split)

    def records():
        for local_index, (source_index, split_name, gravity) in enumerate(assignments):
            parameters = _episode_parameters(task, source_index, gravity, seed)
            pixels, action, state = _render_episode(
                task,
                parameters,
                frames=frames,
                image_size=image_size,
            )
            physics = np.asarray(
                (
                    gravity,
                    parameters["mass"],
                    parameters["size_x"],
                    parameters["size_z"],
                    parameters["shape_id"],
                    BOX_INNER_HALF_WIDTH,
                    BOX_INNER_HALF_HEIGHT,
                    BOX_WALL_THICKNESS,
                ),
                dtype=np.float32,
            )
            yield EpisodeRecord(
                local_index,
                split_name,
                pixels,
                state,
                action,
                gravity,
                physics,
                source_episode_index=source_index,
            )

    dataset_path = write_lance_episodes(
        records(),
        output_dir / "data.lance",
        task=task,
        fps=FPS,
        image_size=image_size,
        frames_per_episode=frames,
        action_names=ACTION_NAMES,
        state_names=STATE_NAMES,
        physics_names=(
            "g",
            "mass",
            "size_x",
            "size_z",
            "shape_id",
            "box_half_width",
            "box_half_height",
            "wall_thickness",
        ),
        extra_metadata={
            "generator_revision": "64732110c7f776f34d66d02522bcd29672b302d6",
            "paper_scale": paper_scale,
            "seed": seed,
        },
    )
    manifest = {
        "schema_version": 1,
        "task": task,
        "generator": "mujoco",
        "generator_revision": "64732110c7f776f34d66d02522bcd29672b302d6",
        "paper_scale": paper_scale,
        "split": split,
        "seed": seed,
        "episodes": episodes,
        "frames_per_episode": frames,
        "image_size": image_size,
        "fps": FPS,
        "train_episodes": sum(item[1] == "train" for item in assignments),
        "test_episodes": sum(item[1] == "test" for item in assignments),
        "action_names": list(ACTION_NAMES),
        "state_names": list(STATE_NAMES),
        "dataset": dataset_path.name,
        "qualitative_reproduction": True,
    }
    (output_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


__all__ = ["generate_planar_dataset"]
