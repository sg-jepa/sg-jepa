import copy
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import sim_support as sim_model
from PIL import Image

_WORKER_RENDERER = None
UNITREE_Z1_DIR = Path(__file__).resolve().parent / "third_party" / "mujoco_menagerie" / "unitree_z1"
GENERATED_ASSET_DIR = Path(__file__).resolve().parent / "_generated_assets"


def render_env_mode():
    value = os.environ.get("DEMO_MUJOCO_RENDER_ENV", "natural").strip().lower()
    if value not in {"custom", "default", "natural"}:
        raise ValueError(
            f"DEMO_MUJOCO_RENDER_ENV must be 'custom', 'default', or 'natural'; got {value!r}"
        )
    return value


def render_supersample_scale():
    if render_env_mode() == "default":
        return 1
    raw_value = os.environ.get("DEMO_MUJOCO_RENDER_SUPERSAMPLE", "2").strip()
    try:
        scale = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"DEMO_MUJOCO_RENDER_SUPERSAMPLE must be an integer; got {raw_value!r}"
        ) from exc
    if scale <= 0:
        raise ValueError(f"DEMO_MUJOCO_RENDER_SUPERSAMPLE must be positive; got {scale!r}")
    return min(scale, 4)


def downsample_frame(frame, output_width, output_height, scale):
    if scale == 1:
        return frame
    return (
        frame.reshape(output_height, scale, output_width, scale, 3)
        .mean(axis=(1, 3))
        .round()
        .clip(0, 255)
        .astype(np.uint8)
    )


def open_box_shadow_lift():
    if render_env_mode() in {"default", "natural"}:
        return 0.0
    raw_value = os.environ.get("DEMO_MUJOCO_OPEN_BOX_SHADOW_LIFT", "0.24").strip()
    try:
        lift = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"DEMO_MUJOCO_OPEN_BOX_SHADOW_LIFT must be a float in [0, 1); got {raw_value!r}"
        ) from exc
    if lift < 0.0 or lift >= 1.0:
        raise ValueError(f"DEMO_MUJOCO_OPEN_BOX_SHADOW_LIFT must be in [0, 1); got {lift!r}")
    return lift


def arm_catcher_wall_floor_emission():
    # Zero now that the light rig scales with the scene (see add_natural_lights)
    # and the walls get a real diffuse term again; this was 0.34 to compensate
    # for the unscaled rig.
    raw_value = os.environ.get("DEMO_MUJOCO_ARM_CATCHER_WALL_EMISSION", "0.0").strip()
    try:
        emission = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"DEMO_MUJOCO_ARM_CATCHER_WALL_EMISSION must be a float in [0, 1]; got {raw_value!r}"
        ) from exc
    if not 0.0 <= emission <= 1.0:
        raise ValueError(
            f"DEMO_MUJOCO_ARM_CATCHER_WALL_EMISSION must be in [0, 1]; got {emission!r}"
        )
    return emission


def apply_open_box_tone_map(frame, lift):
    if lift <= 0.0:
        return frame
    values = frame.astype(np.float32) / 255.0
    values = lift + (1.0 - lift) * values
    return np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)


def render_soften_strength():
    if render_env_mode() in {"default", "natural"}:
        return 0.0
    raw_value = os.environ.get("DEMO_MUJOCO_RENDER_SOFTEN", "0.14").strip()
    try:
        strength = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"DEMO_MUJOCO_RENDER_SOFTEN must be a float in [0, 1); got {raw_value!r}"
        ) from exc
    if strength < 0.0 or strength >= 1.0:
        raise ValueError(f"DEMO_MUJOCO_RENDER_SOFTEN must be in [0, 1); got {strength!r}")
    return strength


def soften_frame(frame, strength):
    if strength <= 0.0:
        return frame
    values = frame.astype(np.float32)
    padded = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blurred = (
        4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    ) / 8.0
    mixed = (1.0 - strength) * values + strength * blurred
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


@dataclass
class OffscreenRenderer:
    width: int
    height: int
    render_scale: int = 1
    max_geom: int = 2048
    _gl_context: object | None = None

    @property
    def render_width(self):
        return self.width * self.render_scale

    @property
    def render_height(self):
        return self.height * self.render_scale

    def ensure_context(self):
        if self._gl_context is None:
            self._gl_context = mujoco.GLContext(self.render_width, self.render_height)
        self._gl_context.make_current()

    def free(self):
        if self._gl_context is not None:
            self._gl_context.free()
            self._gl_context = None

    def render_episode_frames(self, payload, *, version, task_name, episode_index):
        self.ensure_context()
        model = build_render_model(
            payload,
            self.render_width,
            self.render_height,
            version=version,
            task_name=task_name,
            episode_index=episode_index,
        )
        data = mujoco.MjData(model)
        scene = mujoco.MjvScene(model, maxgeom=self.max_geom)
        option = mujoco.MjvOption()
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "render_camera")
        )
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
        viewport = mujoco.MjrRect(0, 0, self.render_width, self.render_height)
        pixels = np.empty((self.render_height, self.render_width, 3), dtype=np.uint8)
        pose_offsets = freejoint_qpos_offsets(model, payload["dynamic_objects"])
        arm_qpos_offsets = arm_joint_qpos_offsets(model, payload)
        gripper_qpos_offsets = gripper_joint_qpos_offsets(model, payload)
        frame_count = int(payload["metadata"]["num_frames"])
        frame_stack = np.empty((frame_count, self.height, self.width, 3), dtype=np.uint8)
        use_default_render_env = render_env_mode() == "default"
        soften_strength = render_soften_strength()
        shadow_lift = (
            open_box_shadow_lift() if payload["metadata"].get("scenario") == "open_box" else 0.0
        )

        try:
            for local_frame_index in range(frame_count):
                if arm_qpos_offsets is not None:
                    set_arm_joint_positions(
                        data,
                        arm_qpos_offsets,
                        payload["arm"]["frames"][local_frame_index]["joint_position"],
                    )
                if gripper_qpos_offsets is not None:
                    set_arm_joint_positions(
                        data,
                        gripper_qpos_offsets,
                        payload["gripper"]["frames"][local_frame_index]["joint_position"],
                    )
                for dynamic_spec in payload["dynamic_objects"]:
                    frame = dynamic_spec["frames"][local_frame_index]
                    qpos_adr = pose_offsets[dynamic_spec["name"]]
                    set_freejoint_pose(data, qpos_adr, frame["position"], frame["quaternion_xyzw"])

                mujoco.mj_forward(model, data)
                mujoco.mjv_updateScene(
                    model,
                    data,
                    option,
                    None,
                    camera,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    scene,
                )
                if not use_default_render_env:
                    scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = 1
                mujoco.mjr_render(viewport, scene, context)
                mujoco.mjr_readPixels(pixels, None, viewport, context)
                frame = np.flipud(pixels)
                frame_stack[local_frame_index] = downsample_frame(
                    frame,
                    self.width,
                    self.height,
                    self.render_scale,
                )
                frame_stack[local_frame_index] = soften_frame(
                    frame_stack[local_frame_index],
                    soften_strength,
                )
                frame_stack[local_frame_index] = apply_open_box_tone_map(
                    frame_stack[local_frame_index],
                    shadow_lift,
                )
            return frame_stack
        finally:
            context.free()
            del context
            del scene
            del data
            del model
            del option
            del camera


def require_mujoco():
    return mujoco


def init_worker_renderer(width, height):
    global _WORKER_RENDERER
    if _WORKER_RENDERER is None:
        _WORKER_RENDERER = OffscreenRenderer(
            width=width,
            height=height,
            render_scale=render_supersample_scale(),
        )
        _WORKER_RENDERER.ensure_context()
    return _WORKER_RENDERER


def get_worker_renderer(width, height):
    renderer = init_worker_renderer(width, height)
    renderer.ensure_context()
    return renderer


def close_worker_renderer():
    global _WORKER_RENDERER
    if _WORKER_RENDERER is not None:
        _WORKER_RENDERER.free()
        _WORKER_RENDERER = None


def assert_egl_available(width, height):
    renderer = OffscreenRenderer(
        width=width,
        height=height,
        render_scale=render_supersample_scale(),
    )
    try:
        renderer.ensure_context()
    except Exception as exc:
        raise RuntimeError(
            "Failed to create a MuJoCo EGL offscreen context. "
            "Set `MUJOCO_GL=egl` and ensure EGL-capable GPU drivers are available."
        ) from exc
    finally:
        renderer.free()


def xyzw_to_wxyz(quaternion_xyzw):
    x, y, z, w = quaternion_xyzw
    return (w, x, y, z)


def format_scalar(value):
    return f"{float(value):.10g}"


def format_vec(values):
    return " ".join(format_scalar(value) for value in values)


def material_rgba(material_spec):
    base_color = material_spec.get("base_color", [0.8, 0.8, 0.8, 1.0])
    if len(base_color) == 3:
        base_color = [*base_color, 1.0]
    return base_color


def material_texture_spec(spec):
    material = spec.get("material", {})
    texture = material.get("texture")
    if not isinstance(texture, dict):
        return None
    return texture


def textured_material_name(spec):
    texture = material_texture_spec(spec)
    if texture is None:
        return None
    return texture.get("material_name", f"{spec['name']}_textured_mat")


def texture_kind(texture):
    return texture.get("kind", "checker")


def stable_asset_token(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def texture_rgb(texture, key, fallback):
    values = texture.get(key, fallback)
    return [float(value) for value in values[:3]]


def uint8_rgb(values):
    return [int(round(255.0 * min(1.0, max(0.0, float(value))))) for value in values[:3]]


def latlong_checker_texture_payload(texture):
    return {
        "kind": "latlong_checker",
        "rgb1": texture_rgb(texture, "rgb1", [0.95, 0.05, 0.04]),
        "rgb2": texture_rgb(texture, "rgb2", [1.0, 1.0, 1.0]),
        "longitude_segments": int(texture.get("longitude_segments", 16)),
        "latitude_segments": int(texture.get("latitude_segments", 8)),
        "width": int(texture.get("width", 1024)),
        "height": int(texture.get("height", 512)),
    }


def ensure_latlong_checker_texture(texture):
    payload = latlong_checker_texture_payload(texture)
    GENERATED_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_ASSET_DIR / f"latlong_checker_{stable_asset_token(payload)}.png"
    if path.exists():
        return path

    width = max(16, int(payload["width"]))
    height = max(8, int(payload["height"]))
    longitude_segments = max(1, int(payload["longitude_segments"]))
    latitude_segments = max(1, int(payload["latitude_segments"]))
    x_indices = np.floor(np.arange(width, dtype=np.float64) * longitude_segments / width).astype(
        np.int32
    )
    y_indices = np.floor(np.arange(height, dtype=np.float64) * latitude_segments / height).astype(
        np.int32
    )
    checker = (x_indices[None, :] + y_indices[:, None]) % 2
    colors = np.asarray(
        [
            uint8_rgb(payload["rgb1"]),
            uint8_rgb(payload["rgb2"]),
        ],
        dtype=np.uint8,
    )
    image = colors[checker]
    tmp_path = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    Image.fromarray(image, mode="RGB").save(tmp_path)
    tmp_path.replace(path)
    return path


def latlong_sphere_mesh_payload(spec):
    texture = material_texture_spec(spec) or {}
    latitude_segments = max(1, int(texture.get("latitude_segments", 8)))
    longitude_segments = max(1, int(texture.get("longitude_segments", 16)))
    return {
        "kind": "latlong_uv_sphere",
        "radius": round(float(spec["radius"]), 8),
        "latitude_steps": max(48, latitude_segments * 8),
        "longitude_steps": max(96, longitude_segments * 8),
    }


def latlong_sphere_mesh_name(spec):
    return f"{spec['name']}_latlong_uv_sphere_mesh"


def uses_latlong_sphere_mesh(spec):
    texture = material_texture_spec(spec)
    return (
        spec.get("kind") == "ball"
        and texture is not None
        and texture_kind(texture) == "latlong_checker"
    )


def ensure_latlong_sphere_mesh(spec):
    payload = latlong_sphere_mesh_payload(spec)
    GENERATED_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_ASSET_DIR / f"latlong_sphere_{stable_asset_token(payload)}.obj"
    if path.exists():
        return path

    radius = float(payload["radius"])
    latitude_steps = int(payload["latitude_steps"])
    longitude_steps = int(payload["longitude_steps"])
    vertices = []
    texcoords = []
    for lat_index in range(latitude_steps + 1):
        theta = math.pi * lat_index / latitude_steps
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        v_coord = 1.0 - lat_index / latitude_steps
        for lon_index in range(longitude_steps + 1):
            phi = 2.0 * math.pi * lon_index / longitude_steps
            vertices.append(
                (
                    radius * sin_theta * math.cos(phi),
                    radius * sin_theta * math.sin(phi),
                    radius * cos_theta,
                )
            )
            texcoords.append((lon_index / longitude_steps, v_coord))

    def vertex_index(lat_index, lon_index):
        return lat_index * (longitude_steps + 1) + lon_index + 1

    tmp_path = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated lat/long UV sphere for paddle_ball rendering.\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.10g} {y:.10g} {z:.10g}\n")
        for u, v_coord in texcoords:
            handle.write(f"vt {u:.10g} {v_coord:.10g}\n")
        for lat_index in range(latitude_steps):
            for lon_index in range(longitude_steps):
                a = vertex_index(lat_index, lon_index)
                b = vertex_index(lat_index + 1, lon_index)
                c = vertex_index(lat_index + 1, lon_index + 1)
                d = vertex_index(lat_index, lon_index + 1)
                if lat_index == 0:
                    handle.write(f"f {b}/{b} {c}/{c} {a}/{a}\n")
                elif lat_index == latitude_steps - 1:
                    handle.write(f"f {a}/{a} {b}/{b} {d}/{d}\n")
                else:
                    handle.write(f"f {a}/{a} {b}/{b} {d}/{d}\n")
                    handle.write(f"f {b}/{b} {c}/{c} {d}/{d}\n")
    tmp_path.replace(path)
    return path


def natural_material_name(spec):
    if material_texture_spec(spec) is not None:
        return None
    if render_env_mode() != "natural" or spec.get("render_only"):
        return None
    override = spec.get("natural_material")
    if override:
        return override
    if spec["name"].startswith("open_box_") or spec["name"].startswith("approach_ball_"):
        return "natural_wall_mat"
    if spec["kind"] in {"ball", "cube"}:
        rgba = material_rgba(spec["material"])
        if rgba[0] > 0.8 and rgba[1] <= 0.15 and rgba[2] <= 0.12:
            return "natural_bright_object_mat"
        return "natural_object_mat"
    return None


def geom_attrs_from_spec(spec, *, mass=None, collisions=False, name=None):
    attrs = {
        "name": name or spec["name"],
    }
    material_name = textured_material_name(spec) or natural_material_name(spec)
    if material_name is None:
        attrs["rgba"] = format_vec(material_rgba(spec["material"]))
    else:
        attrs["material"] = material_name
    if spec["kind"] in {"box", "cube"}:
        attrs["type"] = "box"
        attrs["size"] = format_vec(spec["half_extents"])
    elif uses_latlong_sphere_mesh(spec):
        attrs["type"] = "mesh"
        attrs["mesh"] = latlong_sphere_mesh_name(spec)
    else:
        attrs["type"] = "sphere"
        attrs["size"] = format_scalar(spec["radius"])
    if mass is not None:
        attrs["mass"] = format_scalar(mass)
    if not collisions:
        attrs["contype"] = "0"
        attrs["conaffinity"] = "0"
    return attrs


def add_static_geom(worldbody, spec):
    if not spec.get("render_visible", True):
        return
    attrs = geom_attrs_from_spec(spec, collisions=False)
    attrs["pos"] = format_vec(spec["position"])
    if spec["kind"] in {"box", "cube"}:
        attrs["euler"] = format_vec(spec.get("rotation_euler", (0.0, 0.0, 0.0)))
    if spec.get("render_only"):
        attrs.pop("contype", None)
        attrs.pop("conaffinity", None)
        ET.SubElement(worldbody, "site", attrs)
    else:
        ET.SubElement(worldbody, "geom", attrs)


def add_dynamic_body(worldbody, spec):
    body = ET.SubElement(worldbody, "body", name=spec["name"], pos="0 0 0")
    ET.SubElement(body, "freejoint", name=f"{spec['name']}_free")
    ET.SubElement(body, "geom", geom_attrs_from_spec(spec, mass=1.0, collisions=False))
    return body


def add_camera(worldbody, camera_spec, width, height):
    projection = camera_spec.get("type", "perspective")
    fovy = camera_spec.get("fovy", camera_spec.get("ortho_scale", 45.0))
    attrs = {
        "name": "render_camera",
        "pos": format_vec(camera_spec["location"]),
        "projection": projection,
        "fovy": format_scalar(fovy),
        "resolution": f"{width} {height}",
    }
    if "xyaxes" in camera_spec:
        attrs["xyaxes"] = format_vec(camera_spec["xyaxes"])
    else:
        attrs["xyaxes"] = "1 0 0 0 0 1"
    ET.SubElement(worldbody, "camera", attrs)


def add_lights(worldbody, *, scenario):
    if scenario == "open_box":
        light_pos = "0 0 16"
        light_dir = "0 0 -1"
        key_ambient = "0.76 0.76 0.76"
        key_diffuse = "0.28 0.28 0.28"
    else:
        light_pos = "0 -5 12"
        light_dir = "0 0 -1"
        key_ambient = "0.18 0.18 0.18"
        key_diffuse = "0.8 0.8 0.8"
    ET.SubElement(
        worldbody,
        "light",
        name="key_light",
        pos=light_pos,
        dir=light_dir,
        diffuse=key_diffuse,
        ambient=key_ambient,
        specular="0.02 0.02 0.02",
        cutoff="89",
        exponent="1",
        castshadow="true",
    )


def add_natural_lights(worldbody, light_scale=1.0):
    """Place the natural key/fill rig, optionally rescaled with the scene.

    The positions below are in absolute world units and were picked for the
    original 1.0-scale scenes. A task rendered at 1/7.5 real scale puts a
    half-metre room under a light 13 m up, which degenerates into a
    straight-down directional light and leaves the vertical walls with no
    diffuse term at all. Passing that task's scene scale keeps the rig in the
    same relationship to the room, which is what the pre-rescale renders had.
    """
    light_scale = float(light_scale)

    def scaled(position):
        return format_vec(tuple(value * light_scale for value in position))

    ET.SubElement(
        worldbody,
        "light",
        name="natural_key_light",
        pos=scaled((0.0, 0.0, 13.2)),
        dir="0.0 0.0 -1.0",
        diffuse="0.70 0.72 0.76",
        ambient="0.055 0.058 0.065",
        specular="0.05 0.05 0.05",
        cutoff="72",
        exponent="2",
        castshadow="true",
    )
    ET.SubElement(
        worldbody,
        "light",
        name="natural_fill_light",
        pos=scaled((4.5, -5.8, 8.0)),
        dir="-0.45 0.45 -0.77",
        diffuse="0.13 0.15 0.18",
        ambient="0.025 0.025 0.03",
        specular="0 0 0",
        cutoff="80",
        exponent="1",
        castshadow="false",
    )


def add_backdrop(worldbody):
    ET.SubElement(
        worldbody,
        "geom",
        name="distant_backdrop",
        type="box",
        pos="0 9.6 4.8",
        size="10 0.02 8",
        rgba="1 1 1 1",
        contype="0",
        conaffinity="0",
    )


def get_or_create_asset(root):
    for child in root:
        if child.tag == "asset":
            return child
    return ET.SubElement(root, "asset")


def add_skybox_asset(root):
    asset = get_or_create_asset(root)
    ET.SubElement(
        asset,
        "texture",
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1="1 1 1",
        rgb2="1 1 1",
        width="64",
        height="64",
    )


def add_natural_assets(root):
    asset = get_or_create_asset(root)
    ET.SubElement(
        asset,
        "texture",
        name="natural_skybox",
        type="skybox",
        builtin="gradient",
        rgb1="0.86 0.88 0.90",
        rgb2="0.70 0.72 0.74",
        width="128",
        height="128",
    )
    ET.SubElement(
        asset,
        "material",
        name="natural_wall_mat",
        rgba="0.975 0.985 1.0 1",
        specular="0.025",
        shininess="0.08",
        reflectance="0",
    )
    # Same wall material with a small emissive term: vertical wall faces get
    # almost no direct light from the fixed overhead key light, and close-up
    # cameras (arm_paddle_ball) see those faces head-on; the emission lifts
    # them to the same pixel brightness the arm_catcher_ball walls render at.
    ET.SubElement(
        asset,
        "material",
        name="natural_wall_lit_mat",
        rgba="0.975 0.985 1.0 1",
        emission="0.13",
        specular="0.025",
        shininess="0.08",
        reflectance="0",
    )
    ET.SubElement(
        asset,
        "material",
        name="natural_wall_floor_mat",
        rgba="0.975 0.985 1.0 1",
        emission=format_scalar(arm_catcher_wall_floor_emission()),
        specular="0.015",
        shininess="0.04",
        reflectance="0",
    )
    ET.SubElement(
        asset,
        "material",
        name="natural_bright_object_mat",
        rgba="0.95 0.08 0.04 1",
        specular="0.12",
        shininess="0.18",
        reflectance="0",
    )
    ET.SubElement(
        asset,
        "material",
        name="natural_object_mat",
        rgba="0.62 0.12 0.10 1",
        specular="0.12",
        shininess="0.18",
        reflectance="0",
    )


def add_textured_material_assets(root, specs):
    textured_specs = [spec for spec in specs if material_texture_spec(spec) is not None]
    if not textured_specs:
        return
    asset = get_or_create_asset(root)
    existing_names = {child.get("name") for child in asset if child.get("name")}
    for spec in textured_specs:
        texture = material_texture_spec(spec)
        material = spec["material"]
        material_name = textured_material_name(spec)
        texture_name = texture.get("texture_name", f"{spec['name']}_checker_tex")
        if texture_name not in existing_names:
            kind = texture_kind(texture)
            if kind == "latlong_checker":
                ET.SubElement(
                    asset,
                    "texture",
                    name=texture_name,
                    type=texture.get("type", "2d"),
                    file=str(ensure_latlong_checker_texture(texture)),
                )
            elif kind == "checker":
                ET.SubElement(
                    asset,
                    "texture",
                    name=texture_name,
                    type=texture.get("type", "2d"),
                    builtin="checker",
                    rgb1=format_vec(texture.get("rgb1", [0.95, 0.05, 0.04])),
                    rgb2=format_vec(texture.get("rgb2", [1.0, 1.0, 1.0])),
                    width=str(int(texture.get("width", 256))),
                    height=str(int(texture.get("height", 256))),
                )
            else:
                raise ValueError(f"Unsupported material texture kind: {texture.get('kind')!r}")
            existing_names.add(texture_name)
        if material_name not in existing_names:
            texrepeat = texture.get("texrepeat", [1.0, 1.0])
            ET.SubElement(
                asset,
                "material",
                name=material_name,
                texture=texture_name,
                texrepeat=format_vec(texrepeat),
                rgba=format_vec(texture.get("rgba", [1.0, 1.0, 1.0, 1.0])),
                specular=format_scalar(material.get("specular", 0.18)),
                emission=format_scalar(material.get("emission", 0.0)),
                shininess=format_scalar(1.0 - float(material.get("roughness", 0.42))),
                reflectance="0",
            )
            existing_names.add(material_name)
        if uses_latlong_sphere_mesh(spec):
            mesh_name = latlong_sphere_mesh_name(spec)
            if mesh_name not in existing_names:
                ET.SubElement(
                    asset,
                    "mesh",
                    name=mesh_name,
                    file=str(ensure_latlong_sphere_mesh(spec)),
                )
                existing_names.add(mesh_name)


def build_render_model(payload, width, height, *, version, task_name, episode_index):
    del version, task_name, episode_index
    scenario = payload["metadata"]["scenario"]
    render_mode = render_env_mode()
    use_default_render_env = render_mode == "default"
    use_natural_render_env = render_mode == "natural"
    root = ET.Element("mujoco", model=f"{scenario}_3d_render")
    ET.SubElement(root, "compiler", angle="radian")
    if not use_default_render_env:
        ET.SubElement(root, "option", gravity="0 0 0")
        if use_natural_render_env:
            add_natural_assets(root)
        else:
            add_skybox_asset(root)
        visual = ET.SubElement(root, "visual")
        ET.SubElement(
            root, "statistic", center=format_vec(payload["metadata"]["state_anchor"]), extent="8"
        )
        ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
        ET.SubElement(
            visual,
            "quality",
            offsamples="8" if use_natural_render_env else "4",
            shadowsize="8192" if use_natural_render_env else "1024",
            numslices="64" if use_natural_render_env else "36",
            numstacks="32" if use_natural_render_env else "18",
            numquads="8" if use_natural_render_env else "4",
        )
        ET.SubElement(
            visual,
            "headlight",
            ambient="0.045 0.045 0.04"
            if use_natural_render_env
            else "0 0 0"
            if scenario == "open_box"
            else "0.28 0.28 0.28",
            diffuse="0.035 0.035 0.03"
            if use_natural_render_env
            else "0 0 0"
            if scenario == "open_box"
            else "0.45 0.45 0.45",
            specular="0 0 0"
            if use_natural_render_env or scenario == "open_box"
            else "0.08 0.08 0.08",
        )
    add_textured_material_assets(
        root,
        [
            *payload.get("static_objects", []),
            *payload.get("dynamic_objects", []),
        ],
    )
    worldbody = ET.SubElement(root, "worldbody")
    add_camera(worldbody, payload["camera"], width, height)
    if not use_default_render_env:
        if use_natural_render_env:
            add_natural_lights(
                worldbody,
                float(payload["metadata"].get("render_light_scale", 1.0)),
            )
        else:
            add_lights(worldbody, scenario=scenario)
            add_backdrop(worldbody)
    for spec in payload["static_objects"]:
        add_static_geom(worldbody, spec)
    if payload.get("arm", {}).get("robot_model") == "unitree_z1":
        add_unitree_z1_render_arm(root, worldbody, payload["arm"])
    elif payload.get("arm", {}).get("robot_model") == "ur5e_robotiq_2f85":
        arm_spec = copy.deepcopy(payload["arm"])
        render_model_scale = float(
            arm_spec.get("render_model_scale", arm_spec.get("model_scale", 1.0))
        )
        arm_spec["model_scale"] = render_model_scale
        sim_model.add_ur5e_robotiq_arm(root, worldbody, arm_spec, include_contacts=False)
    for spec in payload["dynamic_objects"]:
        add_dynamic_body(worldbody, spec)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def freejoint_qpos_offsets(model, dynamic_specs):
    offsets = {}
    for spec in dynamic_specs:
        joint = model.joint(f"{spec['name']}_free")
        offsets[spec["name"]] = int(joint.qposadr[0])
    return offsets


def set_freejoint_pose(data, qpos_adr, position, quaternion_xyzw):
    data.qpos[qpos_adr : qpos_adr + 3] = position
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = xyzw_to_wxyz(quaternion_xyzw)


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


def style_unitree_z1_render_tree(element):
    if element.tag == "geom":
        geom_class = element.get("class", "")
        if geom_class == "visual":
            element.set("rgba", "0.46 0.50 0.56 1")
            element.set("group", "0")
        elif "collision" in geom_class or element.get("group") == "3":
            element.set("rgba", "0 0 0 0")
    for child in element:
        style_unitree_z1_render_tree(child)


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


def add_unitree_z1_render_arm(root, worldbody, arm_spec):
    z1_root = load_unitree_z1_root()
    model_scale = float(arm_spec.get("model_scale", 1.0))
    append_unitree_z1_defaults(root, z1_root)
    append_unitree_z1_assets(root, z1_root, model_scale)
    z1_body = copy.deepcopy(z1_root.find("worldbody/body"))
    scale_unitree_z1_tree(z1_body, model_scale)
    style_unitree_z1_render_tree(z1_body)
    z1_body.set("name", "unitree_z1_base")
    z1_body.set("pos", format_vec(arm_spec["base_position"]))
    z1_body.set("euler", format_vec(arm_spec.get("base_euler", (0.0, 0.0, 0.0))))
    worldbody.append(z1_body)


def arm_joint_qpos_offsets(model, payload):
    arm_spec = payload.get("arm")
    if not arm_spec or arm_spec.get("robot_model") not in {"unitree_z1", "ur5e_robotiq_2f85"}:
        return None
    return [int(model.joint(joint_name).qposadr[0]) for joint_name in arm_spec["joint_names"]]


def gripper_joint_qpos_offsets(model, payload):
    gripper_spec = payload.get("gripper")
    if not gripper_spec:
        return None
    return [int(model.joint(joint_name).qposadr[0]) for joint_name in gripper_spec["joint_names"]]


def set_arm_joint_positions(data, qpos_offsets, joint_positions):
    for qpos_adr, value in zip(qpos_offsets, joint_positions, strict=False):
        data.qpos[qpos_adr] = float(value)
