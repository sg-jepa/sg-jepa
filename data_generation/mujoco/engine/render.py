import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import render_support as demo_render
from common import (
    ARM_STATE_SCHEMA,
    CATCHER_POLICY_ID_MAP,
    CATCHER_STATE_SCHEMA,
    END_EFFECTOR_STATE_SCHEMA,
    FPS,
    FRAMES_PER_EPISODE,
    GRIPPER_STATE_SCHEMA,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OUTPUT_VERSIONS,
    PADDLE_POLICY_ID_MAP,
    PADDLE_STATE_SCHEMA,
    PHYS_SCHEMA,
    RENDER_SHARD_EPISODES,
    RENDER_START_METHOD,
    RENDER_WORKERS,
    SHAPE_ID_MAP,
    SPLIT_TO_ID,
    STATE_SCHEMA,
    TASK_PARAM_MEANINGS,
    TASKS,
    ensure_clean_dir,
    ensure_dir,
    scene_json_path,
    shard_path,
    shard_ranges,
    task_action_schema,
    task_contact_schema,
    task_has_arm_fields,
    task_has_catcher_fields,
    task_has_gripper_fields,
    task_has_paddle_fields,
    task_version_shard_dir,
)


def build_state_vector(scene_payload, frame_payload):
    x, y, z = frame_payload["position"]
    vx, vy, vz = frame_payload["linear_velocity"]
    qx, qy, qz, qw = frame_payload["quaternion_xyzw"]
    wx, wy, wz = frame_payload["angular_velocity"]
    anchor_x, anchor_y, anchor_z = scene_payload["metadata"]["state_anchor"]
    return np.array(
        [
            x,
            y,
            z,
            vx,
            vy,
            vz,
            qx,
            qy,
            qz,
            qw,
            wx,
            wy,
            wz,
            anchor_x,
            anchor_y,
            anchor_z,
        ],
        dtype=np.float32,
    )


def build_action_vector(scene_payload):
    gravity = float(scene_payload["metadata"]["episode_phys"][0])
    return np.array([gravity], dtype=np.float32)


def build_action_array(scene_payload, action_schema):
    if "actions" in scene_payload:
        actions = np.asarray(scene_payload["actions"], dtype=np.float32).copy()
        if actions.shape != (FRAMES_PER_EPISODE, len(action_schema)):
            raise ValueError(
                f"Expected actions shape {(FRAMES_PER_EPISODE, len(action_schema))}; "
                f"got {actions.shape}"
            )
        actions[:, action_schema.index("g")] = float(scene_payload["metadata"]["episode_phys"][0])
        return actions

    gravity = float(scene_payload["metadata"]["episode_phys"][0])
    action = np.zeros((FRAMES_PER_EPISODE, len(action_schema)), dtype=np.float32)
    action[:, action_schema.index("g")] = gravity
    return action


def build_catcher_state_array(scene_payload):
    frames = scene_payload["catcher"]["frames"]
    if len(frames) != FRAMES_PER_EPISODE:
        raise ValueError(f"Expected {FRAMES_PER_EPISODE} catcher frames; got {len(frames)}")
    state = np.empty((FRAMES_PER_EPISODE, len(CATCHER_STATE_SCHEMA)), dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        state[frame_index] = np.asarray(
            [
                *frame["position"],
                *frame["linear_velocity"],
                *frame["target_position"],
            ],
            dtype=np.float32,
        )
    return state


def build_paddle_state_array(scene_payload):
    frames = scene_payload["paddle"]["frames"]
    if len(frames) != FRAMES_PER_EPISODE:
        raise ValueError(f"Expected {FRAMES_PER_EPISODE} paddle frames; got {len(frames)}")
    state = np.empty((FRAMES_PER_EPISODE, len(PADDLE_STATE_SCHEMA)), dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        state[frame_index] = np.asarray(
            [
                *frame["position"],
                *frame["linear_velocity"],
                *frame["target_position"],
            ],
            dtype=np.float32,
        )
    return state


def build_arm_state_array(scene_payload):
    frames = scene_payload["arm"]["frames"]
    if len(frames) != FRAMES_PER_EPISODE:
        raise ValueError(f"Expected {FRAMES_PER_EPISODE} arm frames; got {len(frames)}")
    state = np.empty((FRAMES_PER_EPISODE, len(ARM_STATE_SCHEMA)), dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        state[frame_index] = np.asarray(
            [
                *frame["joint_position"],
                *frame["joint_velocity"],
                *frame["joint_target"],
            ],
            dtype=np.float32,
        )
    return state


def build_contact_array(scene_payload, contact_schema):
    contacts = scene_payload["contacts"]
    if len(contacts) != FRAMES_PER_EPISODE:
        raise ValueError(f"Expected {FRAMES_PER_EPISODE} contact frames; got {len(contacts)}")
    values = np.zeros((FRAMES_PER_EPISODE, len(contact_schema)), dtype=np.uint8)
    for frame_index, frame_contacts in enumerate(contacts):
        values[frame_index] = [int(bool(frame_contacts[name])) for name in contact_schema]
    return values


def build_gripper_state_array(scene_payload):
    frames = scene_payload["gripper"]["frames"]
    if len(frames) != FRAMES_PER_EPISODE:
        raise ValueError(f"Expected {FRAMES_PER_EPISODE} gripper frames; got {len(frames)}")
    state = np.empty((FRAMES_PER_EPISODE, len(GRIPPER_STATE_SCHEMA)), dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        state[frame_index] = np.asarray(
            [
                frame["grip_command"],
                frame["actuator_target"],
                frame["opening_width"],
                *frame["driver_joint_position"],
                *frame["driver_joint_velocity"],
                *frame["left_pad_position"],
                *frame["right_pad_position"],
                *frame["left_collision_pad_position"],
                *frame["right_collision_pad_position"],
                *frame["grasp_center_position"],
            ],
            dtype=np.float32,
        )
    return state


def build_end_effector_state_array(scene_payload):
    frames = scene_payload["end_effector"]["frames"]
    if len(frames) != FRAMES_PER_EPISODE:
        raise ValueError(f"Expected {FRAMES_PER_EPISODE} end-effector frames; got {len(frames)}")
    state = np.empty((FRAMES_PER_EPISODE, len(END_EFFECTOR_STATE_SCHEMA)), dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        state[frame_index] = np.asarray(
            [
                *frame["position"],
                *frame["quaternion_xyzw"],
                *frame["linear_velocity"],
                *frame["target_position"],
            ],
            dtype=np.float32,
        )
    return state


def build_captured_array(scene_payload):
    captured = scene_payload.get("captured")
    if captured is None:
        return np.zeros((FRAMES_PER_EPISODE,), dtype=np.uint8)
    if len(captured) != FRAMES_PER_EPISODE:
        raise ValueError(f"Expected {FRAMES_PER_EPISODE} captured frames; got {len(captured)}")
    return np.asarray([int(bool(value)) for value in captured], dtype=np.uint8)


def build_bool_label_array(scene_payload, name):
    values = scene_payload.get(name)
    if values is None:
        return np.zeros((FRAMES_PER_EPISODE,), dtype=np.uint8)
    if len(values) != FRAMES_PER_EPISODE:
        raise ValueError(f"Expected {FRAMES_PER_EPISODE} {name} frames; got {len(values)}")
    return np.asarray([int(bool(value)) for value in values], dtype=np.uint8)


def build_episode_metadata(scene_payload):
    metadata = dict(scene_payload["metadata"])
    metadata["success"] = bool(scene_payload.get("success", metadata.get("success", False)))
    if "catcher" in scene_payload:
        metadata["catcher"] = {
            key: value for key, value in scene_payload["catcher"].items() if key != "frames"
        }
    if "paddle" in scene_payload:
        metadata["paddle"] = {
            key: value for key, value in scene_payload["paddle"].items() if key != "frames"
        }
    if "arm" in scene_payload:
        metadata["arm"] = {
            key: value for key, value in scene_payload["arm"].items() if key != "frames"
        }
    if "gripper" in scene_payload:
        metadata["gripper"] = {
            key: value for key, value in scene_payload["gripper"].items() if key != "frames"
        }
    if "end_effector" in scene_payload:
        metadata["end_effector"] = {
            key: value for key, value in scene_payload["end_effector"].items() if key != "frames"
        }
    return json.dumps(metadata, separators=(",", ":"))


def frame_stack_is_zero_filled(frame_stack):
    frame_means = frame_stack.mean(axis=(1, 2, 3))
    return bool(frame_means[1:].max() < 1e-6)


def render_episode_pixels(scene_payload, *, task_name, episode_index, version, max_attempts=4):
    last_frame_means = None
    for attempt in range(1, max_attempts + 1):
        try:
            renderer = demo_render.get_worker_renderer(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
            frame_stack = renderer.render_episode_frames(
                scene_payload,
                version=version,
                task_name=task_name,
                episode_index=episode_index,
            )
        except Exception:
            demo_render.close_worker_renderer()
            if attempt == max_attempts:
                raise
            continue
        if not frame_stack_is_zero_filled(frame_stack):
            return frame_stack, attempt - 1
        last_frame_means = frame_stack.mean(axis=(1, 2, 3))
        demo_render.close_worker_renderer()
    raise RuntimeError(
        "MuJoCo offscreen render produced zero-filled frames after retries "
        f"for {task_name}/{version}/episode_{episode_index:03d}; "
        f"last_frame_means={last_frame_means.tolist()}"
    )


def render_shard_job(job):
    task_name, version, shard_index, start_episode, stop_episode = job
    action_schema = task_action_schema(task_name)
    contact_schema = task_contact_schema(task_name)
    has_catcher_fields = task_has_catcher_fields(task_name)
    has_paddle_fields = task_has_paddle_fields(task_name)
    has_arm_fields = task_has_arm_fields(task_name)
    has_gripper_fields = task_has_gripper_fields(task_name)
    shard_output_dir = task_version_shard_dir(task_name, version)
    ensure_dir(shard_output_dir)
    output_path = shard_path(task_name, version, shard_index)
    temp_path = output_path.with_suffix(".h5.tmp")
    if temp_path.exists():
        temp_path.unlink()
    if output_path.exists():
        output_path.unlink()

    episode_count = stop_episode - start_episode
    total_frames = episode_count * FRAMES_PER_EPISODE
    start_time = time.perf_counter()
    retry_count = 0

    try:
        with h5py.File(temp_path, "w") as handle:
            pixels = handle.create_dataset(
                "pixels",
                shape=(total_frames, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                dtype="u1",
                chunks=(FRAMES_PER_EPISODE, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            )
            state = handle.create_dataset(
                "state",
                shape=(total_frames, len(STATE_SCHEMA)),
                dtype="f4",
                chunks=(FRAMES_PER_EPISODE, len(STATE_SCHEMA)),
            )
            actions = handle.create_dataset(
                "actions",
                shape=(total_frames, len(action_schema)),
                dtype="f4",
                chunks=(FRAMES_PER_EPISODE, len(action_schema)),
            )
            reward = handle.create_dataset(
                "reward",
                shape=(total_frames,),
                dtype="f4",
                chunks=(FRAMES_PER_EPISODE,),
            )
            phys = handle.create_dataset(
                "phys",
                shape=(episode_count, len(PHYS_SCHEMA)),
                dtype="f4",
            )
            split_id = handle.create_dataset("split_id", shape=(episode_count,), dtype="i1")
            if has_catcher_fields:
                catcher_state = handle.create_dataset(
                    "catcher_state",
                    shape=(total_frames, len(CATCHER_STATE_SCHEMA)),
                    dtype="f4",
                    chunks=(FRAMES_PER_EPISODE, len(CATCHER_STATE_SCHEMA)),
                )
                contact = handle.create_dataset(
                    "contact",
                    shape=(total_frames, len(contact_schema)),
                    dtype="u1",
                    chunks=(FRAMES_PER_EPISODE, len(contact_schema)),
                )
                captured = handle.create_dataset(
                    "captured",
                    shape=(total_frames,),
                    dtype="u1",
                    chunks=(FRAMES_PER_EPISODE,),
                )
                missed = handle.create_dataset(
                    "missed",
                    shape=(total_frames,),
                    dtype="u1",
                    chunks=(FRAMES_PER_EPISODE,),
                )
                truncated = handle.create_dataset(
                    "truncated",
                    shape=(total_frames,),
                    dtype="u1",
                    chunks=(FRAMES_PER_EPISODE,),
                )
                success = handle.create_dataset("success", shape=(episode_count,), dtype="u1")
                policy_type_id = handle.create_dataset(
                    "policy_type_id", shape=(episode_count,), dtype="i1"
                )
                episode_metadata = handle.create_dataset(
                    "episode_metadata",
                    shape=(episode_count,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            if has_paddle_fields:
                paddle_state = handle.create_dataset(
                    "paddle_state",
                    shape=(total_frames, len(PADDLE_STATE_SCHEMA)),
                    dtype="f4",
                    chunks=(FRAMES_PER_EPISODE, len(PADDLE_STATE_SCHEMA)),
                )
                contact = handle.create_dataset(
                    "contact",
                    shape=(total_frames, len(contact_schema)),
                    dtype="u1",
                    chunks=(FRAMES_PER_EPISODE, len(contact_schema)),
                )
                success = handle.create_dataset("success", shape=(episode_count,), dtype="u1")
                policy_type_id = handle.create_dataset(
                    "policy_type_id", shape=(episode_count,), dtype="i1"
                )
                episode_metadata = handle.create_dataset(
                    "episode_metadata",
                    shape=(episode_count,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            if has_arm_fields:
                arm_state = handle.create_dataset(
                    "arm_state",
                    shape=(total_frames, len(ARM_STATE_SCHEMA)),
                    dtype="f4",
                    chunks=(FRAMES_PER_EPISODE, len(ARM_STATE_SCHEMA)),
                )
            if has_gripper_fields:
                gripper_state = handle.create_dataset(
                    "gripper_state",
                    shape=(total_frames, len(GRIPPER_STATE_SCHEMA)),
                    dtype="f4",
                    chunks=(FRAMES_PER_EPISODE, len(GRIPPER_STATE_SCHEMA)),
                )
                end_effector_state = handle.create_dataset(
                    "end_effector_state",
                    shape=(total_frames, len(END_EFFECTOR_STATE_SCHEMA)),
                    dtype="f4",
                    chunks=(FRAMES_PER_EPISODE, len(END_EFFECTOR_STATE_SCHEMA)),
                )
                grasped = handle.create_dataset(
                    "grasped",
                    shape=(total_frames,),
                    dtype="u1",
                    chunks=(FRAMES_PER_EPISODE,),
                )
            handle.create_dataset(
                "ep_len",
                data=np.full((episode_count,), FRAMES_PER_EPISODE, dtype=np.int32),
            )
            handle.create_dataset(
                "ep_offset",
                data=np.arange(0, total_frames, FRAMES_PER_EPISODE, dtype=np.int64),
            )
            handle.create_dataset(
                "episode_index",
                data=np.arange(start_episode, stop_episode, dtype=np.int32),
            )
            source_episode_index = handle.create_dataset(
                "source_episode_index", shape=(episode_count,), dtype="i4"
            )

            actions[...] = 0.0
            reward[...] = 0.0

            handle.attrs["task_name"] = task_name
            handle.attrs["version"] = version
            handle.attrs["fps"] = FPS
            handle.attrs["frames_per_episode"] = FRAMES_PER_EPISODE
            handle.attrs["state_schema"] = json.dumps(STATE_SCHEMA)
            handle.attrs["action_schema"] = json.dumps(action_schema)
            handle.attrs["phys_schema"] = json.dumps(PHYS_SCHEMA)
            handle.attrs["shape_id_map"] = json.dumps(SHAPE_ID_MAP)
            handle.attrs["split_to_id"] = json.dumps(SPLIT_TO_ID)
            handle.attrs["task_param_meanings"] = json.dumps(TASK_PARAM_MEANINGS[task_name])
            handle.attrs["has_catcher_fields"] = int(has_catcher_fields)
            handle.attrs["has_paddle_fields"] = int(has_paddle_fields)
            handle.attrs["has_arm_fields"] = int(has_arm_fields)
            if has_catcher_fields:
                handle.attrs["catcher_state_schema"] = json.dumps(CATCHER_STATE_SCHEMA)
                handle.attrs["contact_schema"] = json.dumps(contact_schema)
                handle.attrs["policy_id_map"] = json.dumps(CATCHER_POLICY_ID_MAP)
                handle.attrs["captured_schema"] = json.dumps(["captured"])
                handle.attrs["missed_schema"] = json.dumps(["missed"])
                handle.attrs["truncated_schema"] = json.dumps(["truncated"])
            if has_paddle_fields:
                handle.attrs["paddle_state_schema"] = json.dumps(PADDLE_STATE_SCHEMA)
                handle.attrs["contact_schema"] = json.dumps(contact_schema)
                handle.attrs["policy_id_map"] = json.dumps(PADDLE_POLICY_ID_MAP)
            if has_arm_fields:
                handle.attrs["arm_state_schema"] = json.dumps(ARM_STATE_SCHEMA)
            handle.attrs["has_gripper_fields"] = int(has_gripper_fields)
            if has_gripper_fields:
                handle.attrs["gripper_state_schema"] = json.dumps(GRIPPER_STATE_SCHEMA)
                handle.attrs["end_effector_state_schema"] = json.dumps(END_EFFECTOR_STATE_SCHEMA)
                handle.attrs["grasped_schema"] = json.dumps(["grasped"])
            handle.attrs["shard_index"] = shard_index
            handle.attrs["start_episode"] = start_episode
            handle.attrs["stop_episode"] = stop_episode

            frame_offset = 0
            for local_episode_index, episode_index in enumerate(range(start_episode, stop_episode)):
                scene_payload = json.loads(scene_json_path(task_name, episode_index).read_text())
                source_episode_index[local_episode_index] = int(
                    scene_payload["metadata"].get("source_episode_index", episode_index)
                )
                episode_pixels, episode_retries = render_episode_pixels(
                    scene_payload,
                    task_name=task_name,
                    episode_index=episode_index,
                    version=version,
                )
                retry_count += episode_retries
                episode_state = np.empty((FRAMES_PER_EPISODE, len(STATE_SCHEMA)), dtype=np.float32)
                for local_frame_index, dynamic_frame in enumerate(
                    scene_payload["dynamic_objects"][0]["frames"]
                ):
                    episode_state[local_frame_index] = build_state_vector(
                        scene_payload, dynamic_frame
                    )

                pixels[frame_offset : frame_offset + FRAMES_PER_EPISODE] = episode_pixels
                state[frame_offset : frame_offset + FRAMES_PER_EPISODE] = episode_state
                actions[frame_offset : frame_offset + FRAMES_PER_EPISODE] = build_action_array(
                    scene_payload,
                    action_schema,
                )
                phys[local_episode_index] = np.asarray(
                    scene_payload["metadata"]["episode_phys"], dtype=np.float32
                )
                split_id[local_episode_index] = scene_payload["metadata"]["split_id"]
                if has_catcher_fields:
                    catcher_state[frame_offset : frame_offset + FRAMES_PER_EPISODE] = (
                        build_catcher_state_array(scene_payload)
                    )
                    contact[frame_offset : frame_offset + FRAMES_PER_EPISODE] = build_contact_array(
                        scene_payload,
                        contact_schema,
                    )
                    captured[frame_offset : frame_offset + FRAMES_PER_EPISODE] = (
                        build_captured_array(scene_payload)
                    )
                    missed[frame_offset : frame_offset + FRAMES_PER_EPISODE] = (
                        build_bool_label_array(
                            scene_payload,
                            "missed",
                        )
                    )
                    truncated[frame_offset : frame_offset + FRAMES_PER_EPISODE] = (
                        build_bool_label_array(
                            scene_payload,
                            "truncated",
                        )
                    )
                    success[local_episode_index] = int(bool(scene_payload["success"]))
                    policy_type = scene_payload["metadata"]["policy_type"]
                    policy_type_id[local_episode_index] = CATCHER_POLICY_ID_MAP[policy_type]
                    episode_metadata[local_episode_index] = build_episode_metadata(scene_payload)
                if has_paddle_fields:
                    paddle_state[frame_offset : frame_offset + FRAMES_PER_EPISODE] = (
                        build_paddle_state_array(scene_payload)
                    )
                    contact[frame_offset : frame_offset + FRAMES_PER_EPISODE] = build_contact_array(
                        scene_payload,
                        contact_schema,
                    )
                    success[local_episode_index] = int(bool(scene_payload["success"]))
                    policy_type = scene_payload["metadata"]["policy_type"]
                    policy_type_id[local_episode_index] = PADDLE_POLICY_ID_MAP[policy_type]
                    episode_metadata[local_episode_index] = build_episode_metadata(scene_payload)
                if has_arm_fields:
                    arm_state[frame_offset : frame_offset + FRAMES_PER_EPISODE] = (
                        build_arm_state_array(scene_payload)
                    )
                if has_gripper_fields:
                    gripper_state[frame_offset : frame_offset + FRAMES_PER_EPISODE] = (
                        build_gripper_state_array(scene_payload)
                    )
                    end_effector_state[frame_offset : frame_offset + FRAMES_PER_EPISODE] = (
                        build_end_effector_state_array(scene_payload)
                    )
                    grasped[frame_offset : frame_offset + FRAMES_PER_EPISODE] = (
                        build_bool_label_array(
                            scene_payload,
                            "grasped",
                        )
                    )
                frame_offset += FRAMES_PER_EPISODE

            handle.attrs["render_retries"] = retry_count
    finally:
        demo_render.close_worker_renderer()

    temp_path.replace(output_path)
    elapsed = time.perf_counter() - start_time
    return task_name, version, shard_index, episode_count, retry_count, elapsed, output_path


def shard_jobs():
    jobs = []
    for task_name in TASKS:
        for version in OUTPUT_VERSIONS:
            for shard_index, start_episode, stop_episode in shard_ranges():
                jobs.append((task_name, version, shard_index, start_episode, stop_episode))
    return jobs


def resolved_render_workers():
    override = os.environ.get("DEMO_MUJOCO_RENDER_WORKERS")
    if override is not None:
        return max(1, int(override))
    if os.environ.get("MUJOCO_GL", "").lower() == "egl":
        return 1
    return RENDER_WORKERS


def main():
    demo_render.assert_egl_available(IMAGE_WIDTH, IMAGE_HEIGHT)
    for task_name in TASKS:
        for version in OUTPUT_VERSIONS:
            ensure_clean_dir(task_version_shard_dir(task_name, version))

    jobs = shard_jobs()
    total_start = time.perf_counter()
    worker_count = min(resolved_render_workers(), len(jobs))
    ctx = mp.get_context(RENDER_START_METHOD)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=ctx,
        max_tasks_per_child=1,
    ) as executor:
        futures = {executor.submit(render_shard_job, job): job for job in jobs}
        for future in as_completed(futures):
            task_name, version, shard_index, episode_count, retry_count, elapsed, output_path = (
                future.result()
            )
            print(
                f"Wrote {output_path} "
                f"(episodes={episode_count}, retries={retry_count}, elapsed_seconds={elapsed:.2f})",
                flush=True,
            )

    total_elapsed = time.perf_counter() - total_start
    print(
        f"Rendered {len(jobs)} shards "
        f"(episodes_per_shard<={RENDER_SHARD_EPISODES}) with {worker_count} workers "
        f"in {total_elapsed:.2f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
