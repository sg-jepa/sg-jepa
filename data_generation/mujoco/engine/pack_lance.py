import json
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO

import h5py
import lance
import numpy as np
import pyarrow as pa
from common import (
    EPISODES_PER_TASK,
    FPS,
    FRAMES_PER_EPISODE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OUTPUT_VERSIONS,
    PACK_WORKERS,
    PHYS_SCHEMA,
    RENDER_START_METHOD,
    SHARD_ROOT,
    STATE_SCHEMA,
    TASKS,
    ensure_dir,
    shard_path,
    shard_ranges,
    task_version_dir,
    task_version_shard_dir,
)
from PIL import Image


def remove_path(path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def encode_metadata_value(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value).encode("utf-8")


def encode_jpeg_rgb(frame_pixels, *, quality=95):
    image = Image.fromarray(frame_pixels, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def decode_json_attr(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def build_lance_schema(task_name, version, shard_attrs):
    action_schema = decode_json_attr(shard_attrs["action_schema"])
    has_catcher_fields = bool(int(shard_attrs.get("has_catcher_fields", 0)))
    has_paddle_fields = bool(int(shard_attrs.get("has_paddle_fields", 0)))
    has_arm_fields = bool(int(shard_attrs.get("has_arm_fields", 0)))
    has_gripper_fields = bool(int(shard_attrs.get("has_gripper_fields", 0)))
    metadata = {
        b"task_name": task_name.encode("utf-8"),
        b"version": version.encode("utf-8"),
        b"fps": str(FPS).encode("utf-8"),
        b"frames_per_episode": str(FRAMES_PER_EPISODE).encode("utf-8"),
        b"pixels_shape": json.dumps([IMAGE_HEIGHT, IMAGE_WIDTH, 3]).encode("utf-8"),
        b"pixels_dtype": b"uint8",
        b"pixels_encoding": b"jpeg_rgb",
        b"state_schema": encode_metadata_value(shard_attrs["state_schema"]),
        b"action_schema": encode_metadata_value(shard_attrs["action_schema"]),
        b"phys_schema": encode_metadata_value(shard_attrs["phys_schema"]),
        b"shape_id_map": encode_metadata_value(shard_attrs["shape_id_map"]),
        b"split_to_id": encode_metadata_value(shard_attrs["split_to_id"]),
        b"task_param_meanings": encode_metadata_value(shard_attrs["task_param_meanings"]),
    }
    fields = [
        pa.field("episode_idx", pa.int32()),
        pa.field("step_idx", pa.int32()),
        pa.field("split_id", pa.int8()),
        pa.field("pixels", pa.binary()),
        pa.field("state", pa.list_(pa.float32(), len(STATE_SCHEMA))),
        pa.field("action", pa.list_(pa.float32(), len(action_schema))),
        pa.field("reward", pa.float32()),
        pa.field("phys", pa.list_(pa.float32(), len(PHYS_SCHEMA))),
        pa.field("gravity", pa.float32()),
        pa.field("source_episode_index", pa.int32()),
        pa.field("task", pa.string()),
        pa.field("episode_metadata", pa.string()),
    ]
    if has_catcher_fields:
        catcher_state_schema = decode_json_attr(shard_attrs["catcher_state_schema"])
        contact_schema = decode_json_attr(shard_attrs["contact_schema"])
        metadata[b"catcher_state_schema"] = encode_metadata_value(
            shard_attrs["catcher_state_schema"]
        )
        metadata[b"contact_schema"] = encode_metadata_value(shard_attrs["contact_schema"])
        metadata[b"policy_id_map"] = encode_metadata_value(shard_attrs["policy_id_map"])
        if "captured_schema" in shard_attrs:
            metadata[b"captured_schema"] = encode_metadata_value(shard_attrs["captured_schema"])
        if "missed_schema" in shard_attrs:
            metadata[b"missed_schema"] = encode_metadata_value(shard_attrs["missed_schema"])
        if "truncated_schema" in shard_attrs:
            metadata[b"truncated_schema"] = encode_metadata_value(shard_attrs["truncated_schema"])
        fields.extend(
            [
                pa.field("catcher_state", pa.list_(pa.float32(), len(catcher_state_schema))),
                pa.field("contact", pa.list_(pa.uint8(), len(contact_schema))),
                pa.field("captured", pa.bool_()),
                pa.field("missed", pa.bool_()),
                pa.field("truncated", pa.bool_()),
                pa.field("success", pa.bool_()),
                pa.field("policy_type_id", pa.int8()),
            ]
        )
    if has_paddle_fields:
        paddle_state_schema = decode_json_attr(shard_attrs["paddle_state_schema"])
        contact_schema = decode_json_attr(shard_attrs["contact_schema"])
        metadata[b"paddle_state_schema"] = encode_metadata_value(shard_attrs["paddle_state_schema"])
        metadata[b"contact_schema"] = encode_metadata_value(shard_attrs["contact_schema"])
        metadata[b"policy_id_map"] = encode_metadata_value(shard_attrs["policy_id_map"])
        fields.extend(
            [
                pa.field("paddle_state", pa.list_(pa.float32(), len(paddle_state_schema))),
                pa.field("contact", pa.list_(pa.uint8(), len(contact_schema))),
                pa.field("success", pa.bool_()),
                pa.field("policy_type_id", pa.int8()),
            ]
        )
    if has_arm_fields:
        arm_state_schema = decode_json_attr(shard_attrs["arm_state_schema"])
        metadata[b"arm_state_schema"] = encode_metadata_value(shard_attrs["arm_state_schema"])
        fields.append(pa.field("arm_state", pa.list_(pa.float32(), len(arm_state_schema))))
    if has_gripper_fields:
        gripper_state_schema = decode_json_attr(shard_attrs["gripper_state_schema"])
        end_effector_state_schema = decode_json_attr(shard_attrs["end_effector_state_schema"])
        metadata[b"gripper_state_schema"] = encode_metadata_value(
            shard_attrs["gripper_state_schema"]
        )
        metadata[b"end_effector_state_schema"] = encode_metadata_value(
            shard_attrs["end_effector_state_schema"]
        )
        metadata[b"grasped_schema"] = encode_metadata_value(shard_attrs["grasped_schema"])
        fields.extend(
            [
                pa.field("gripper_state", pa.list_(pa.float32(), len(gripper_state_schema))),
                pa.field(
                    "end_effector_state", pa.list_(pa.float32(), len(end_effector_state_schema))
                ),
                pa.field("grasped", pa.bool_()),
            ]
        )
    return pa.schema(fields, metadata=metadata)


def fixed_size_list_array(values, list_size):
    flat_values = np.asarray(values, dtype=np.float32).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(flat_values, type=pa.float32()),
        list_size,
    )


def fixed_size_uint8_list_array(values, list_size):
    flat_values = np.asarray(values, dtype=np.uint8).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(flat_values, type=pa.uint8()),
        list_size,
    )


def h5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def make_record_batch(
    *,
    schema,
    episode_index,
    source_episode_index,
    split_id_value,
    episode_pixels,
    episode_state,
    episode_actions,
    episode_reward,
    episode_phys,
    action_schema,
    task_name,
    catcher_state=None,
    contact=None,
    captured=None,
    missed=None,
    truncated=None,
    paddle_state=None,
    success=None,
    policy_type_id=None,
    episode_metadata=None,
    arm_state=None,
    gripper_state=None,
    end_effector_state=None,
    grasped=None,
):
    frame_count = episode_pixels.shape[0]
    gravity_value = float(episode_actions[0, action_schema.index("g")])
    if episode_metadata is None:
        episode_metadata = json.dumps(
            {
                "task": task_name,
                "episode_index": int(episode_index),
                "source_episode_index": int(source_episode_index),
                "split": "train" if int(split_id_value) == 0 else "test",
                "gravity": gravity_value,
            },
            sort_keys=True,
        )
    arrays = [
        pa.array([episode_index] * frame_count, type=pa.int32()),
        pa.array(np.arange(frame_count, dtype=np.int32), type=pa.int32()),
        pa.array([int(split_id_value)] * frame_count, type=pa.int8()),
        pa.array([encode_jpeg_rgb(frame) for frame in episode_pixels], type=pa.binary()),
        fixed_size_list_array(episode_state, len(STATE_SCHEMA)),
        fixed_size_list_array(episode_actions, len(action_schema)),
        pa.array(episode_reward.tolist(), type=pa.float32()),
        fixed_size_list_array(
            np.repeat(episode_phys[np.newaxis, :], frame_count, axis=0),
            len(PHYS_SCHEMA),
        ),
        pa.array([gravity_value] * frame_count, type=pa.float32()),
        pa.array([source_episode_index] * frame_count, type=pa.int32()),
        pa.array([task_name] * frame_count, type=pa.string()),
        pa.array([episode_metadata] * frame_count, type=pa.string()),
    ]
    if catcher_state is not None:
        arrays.extend(
            [
                fixed_size_list_array(catcher_state, catcher_state.shape[1]),
                fixed_size_uint8_list_array(contact, contact.shape[1]),
                pa.array([bool(value) for value in captured], type=pa.bool_()),
                pa.array([bool(value) for value in missed], type=pa.bool_()),
                pa.array([bool(value) for value in truncated], type=pa.bool_()),
                pa.array([bool(success)] * frame_count, type=pa.bool_()),
                pa.array([int(policy_type_id)] * frame_count, type=pa.int8()),
            ]
        )
    if paddle_state is not None:
        arrays.extend(
            [
                fixed_size_list_array(paddle_state, paddle_state.shape[1]),
                fixed_size_uint8_list_array(contact, contact.shape[1]),
                pa.array([bool(success)] * frame_count, type=pa.bool_()),
                pa.array([int(policy_type_id)] * frame_count, type=pa.int8()),
            ]
        )
    if arm_state is not None:
        arrays.append(fixed_size_list_array(arm_state, arm_state.shape[1]))
    if gripper_state is not None:
        arrays.extend(
            [
                fixed_size_list_array(gripper_state, gripper_state.shape[1]),
                fixed_size_list_array(end_effector_state, end_effector_state.shape[1]),
                pa.array([bool(value) for value in grasped], type=pa.bool_()),
            ]
        )
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def merge_task_version(task_name, version):
    version_dir = task_version_dir(task_name, version)
    ensure_dir(version_dir)
    data_path = version_dir / "data.lance"
    remove_path(data_path)

    shard_paths = [
        shard_path(task_name, version, shard_index) for shard_index, _, _ in shard_ranges()
    ]
    for path in shard_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing shard file: {path}")

    with h5py.File(shard_paths[0], "r") as first_shard:
        shard_attrs = dict(first_shard.attrs)

    schema = build_lance_schema(task_name, version, shard_attrs)
    action_schema = decode_json_attr(shard_attrs["action_schema"])
    has_catcher_fields = bool(int(shard_attrs.get("has_catcher_fields", 0)))
    has_paddle_fields = bool(int(shard_attrs.get("has_paddle_fields", 0)))
    has_arm_fields = bool(int(shard_attrs.get("has_arm_fields", 0)))
    has_gripper_fields = bool(int(shard_attrs.get("has_gripper_fields", 0)))
    seen_episode_indices = set()

    def record_batches():
        for shard_file in shard_paths:
            with h5py.File(shard_file, "r") as shard:
                episode_indices = shard["episode_index"][...]
                shard_source_episode_indices = shard["source_episode_index"][...]
                shard_pixels = shard["pixels"][...]
                shard_state = shard["state"][...]
                shard_actions = shard["actions"][...]
                shard_reward = shard["reward"][...]
                shard_phys = shard["phys"][...]
                shard_split_id = shard["split_id"][...]
                if has_catcher_fields:
                    shard_catcher_state = shard["catcher_state"][...]
                    shard_contact = shard["contact"][...]
                    shard_captured = shard["captured"][...]
                    shard_missed = shard["missed"][...]
                    shard_truncated = shard["truncated"][...]
                    shard_success = shard["success"][...]
                    shard_policy_type_id = shard["policy_type_id"][...]
                    shard_episode_metadata = shard["episode_metadata"][...]
                if has_paddle_fields:
                    shard_paddle_state = shard["paddle_state"][...]
                    shard_contact = shard["contact"][...]
                    shard_success = shard["success"][...]
                    shard_policy_type_id = shard["policy_type_id"][...]
                    shard_episode_metadata = shard["episode_metadata"][...]
                if has_arm_fields:
                    shard_arm_state = shard["arm_state"][...]
                if has_gripper_fields:
                    shard_gripper_state = shard["gripper_state"][...]
                    shard_end_effector_state = shard["end_effector_state"][...]
                    shard_grasped = shard["grasped"][...]

                for local_episode_index, episode_index in enumerate(episode_indices.tolist()):
                    if episode_index in seen_episode_indices:
                        raise ValueError(f"Duplicate episode_index={episode_index} in {shard_file}")
                    seen_episode_indices.add(episode_index)

                    src_start = local_episode_index * FRAMES_PER_EPISODE
                    src_stop = src_start + FRAMES_PER_EPISODE

                    episode_pixels = shard_pixels[src_start:src_stop]
                    episode_state = shard_state[src_start:src_stop]
                    episode_actions = shard_actions[src_start:src_stop]
                    episode_reward = shard_reward[src_start:src_stop]
                    episode_phys = shard_phys[local_episode_index]
                    episode_split_id = shard_split_id[local_episode_index]
                    episode_source_index = shard_source_episode_indices[local_episode_index]
                    extra_kwargs = {}
                    if has_catcher_fields:
                        extra_kwargs = {
                            "catcher_state": shard_catcher_state[src_start:src_stop],
                            "contact": shard_contact[src_start:src_stop],
                            "captured": shard_captured[src_start:src_stop],
                            "missed": shard_missed[src_start:src_stop],
                            "truncated": shard_truncated[src_start:src_stop],
                            "success": bool(shard_success[local_episode_index]),
                            "policy_type_id": int(shard_policy_type_id[local_episode_index]),
                            "episode_metadata": h5_string(
                                shard_episode_metadata[local_episode_index]
                            ),
                        }
                    if has_paddle_fields:
                        extra_kwargs = {
                            "paddle_state": shard_paddle_state[src_start:src_stop],
                            "contact": shard_contact[src_start:src_stop],
                            "success": bool(shard_success[local_episode_index]),
                            "policy_type_id": int(shard_policy_type_id[local_episode_index]),
                            "episode_metadata": h5_string(
                                shard_episode_metadata[local_episode_index]
                            ),
                        }
                    if has_arm_fields:
                        extra_kwargs["arm_state"] = shard_arm_state[src_start:src_stop]
                    if has_gripper_fields:
                        extra_kwargs["gripper_state"] = shard_gripper_state[src_start:src_stop]
                        extra_kwargs["end_effector_state"] = shard_end_effector_state[
                            src_start:src_stop
                        ]
                        extra_kwargs["grasped"] = shard_grasped[src_start:src_stop]

                    yield make_record_batch(
                        schema=schema,
                        episode_index=episode_index,
                        source_episode_index=episode_source_index,
                        split_id_value=episode_split_id,
                        episode_pixels=episode_pixels,
                        episode_state=episode_state,
                        episode_actions=episode_actions,
                        episode_reward=episode_reward,
                        episode_phys=episode_phys,
                        action_schema=action_schema,
                        task_name=task_name,
                        **extra_kwargs,
                    )

    dataset = lance.write_dataset(
        record_batches(),
        str(data_path),
        schema=schema,
        mode="overwrite",
    )
    expected_episode_indices = set(range(EPISODES_PER_TASK))
    if seen_episode_indices != expected_episode_indices:
        raise ValueError(
            f"Incomplete shard coverage for {task_name}/{version}: "
            f"expected={sorted(expected_episode_indices)} "
            f"got={sorted(seen_episode_indices)}"
        )
    expected_rows = EPISODES_PER_TASK * FRAMES_PER_EPISODE
    if dataset.count_rows() != expected_rows:
        raise ValueError(
            f"Unexpected Lance row count for {task_name}/{version}: "
            f"expected={expected_rows} got={dataset.count_rows()}"
        )

    shutil.rmtree(task_version_shard_dir(task_name, version))
    return data_path


def main():
    jobs = [(task_name, version) for task_name in TASKS for version in OUTPUT_VERSIONS]
    max_workers = min(PACK_WORKERS, len(jobs))
    ctx = mp.get_context(RENDER_START_METHOD)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(merge_task_version, task_name, version): (task_name, version)
            for task_name, version in jobs
        }
        for future in as_completed(futures):
            data_path = future.result()
            print(f"Wrote {data_path}", flush=True)

    if SHARD_ROOT.exists() and not any(SHARD_ROOT.iterdir()):
        SHARD_ROOT.rmdir()


if __name__ == "__main__":
    main()
