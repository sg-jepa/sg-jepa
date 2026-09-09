from types import SimpleNamespace

import numpy as np

from sg_jepa.envs.catcher import ArmCatcherBallEnv, _metadata_gravity
from sg_jepa.evaluation.frozen_catcher import action_smoothness, summarize_records


def test_catcher_metadata_preserves_signed_gravity() -> None:
    assert _metadata_gravity({"gravity": [0.0, 0.0, 1.0]}) == -1.0
    assert _metadata_gravity({"gravity": [0.0, 0.0, -9.0], "episode_phys": [-0.5]}) == -0.5


def test_catcher_rebuild_uses_recorded_physical_metadata() -> None:
    generated_scene = {
        "metadata": {"scenario": "arm_catcher_ball"},
        "camera": {"fovy": 10.0},
        "dynamic_objects": [
            {
                "position": [9.0, 9.0, 9.0],
                "initial_angular_velocity": [0.4, 0.5, 0.6],
                "lock_rotation": False,
            }
        ],
        "catcher": {"net_depth": 9.0},
        "arm": {"base_position": [9.0, 9.0, 9.0]},
    }
    runtime = SimpleNamespace(
        generate=SimpleNamespace(build_episode_scene=lambda *_args: generated_scene)
    )
    environment = object.__new__(ArmCatcherBallEnv)
    environment.runtime = runtime
    environment.generate = runtime.generate
    metadata = {
        "scenario": "arm_catcher_ball",
        "episode_index": 17,
        "split_name": "test",
        "random_seed": 123,
        "gravity": [0.0, 0.0, 0.5],
        "episode_phys": [-0.5],
        "camera_parameters": {"fovy": 54.0},
        "ball_initial_state": {
            "position": [1.0, 2.0, 3.0],
            "mass": 0.1,
            "radius": 0.2,
            "linear_velocity": [4.0, 5.0, 6.0],
        },
        "catcher": {"net_depth": 0.03},
        "arm": {"base_position": [0.1, 0.2, 0.3]},
    }

    rebuilt = environment._rebuild_generator_scene(metadata)

    assert rebuilt["metadata"] == metadata
    assert rebuilt["camera"]["fovy"] == 54.0
    assert rebuilt["dynamic_objects"][0]["position"] == [1.0, 2.0, 3.0]
    assert rebuilt["dynamic_objects"][0]["initial_velocity"] == [4.0, 5.0, 6.0]
    assert rebuilt["dynamic_objects"][0]["initial_angular_velocity"] == [0.4, 0.5, 0.6]
    assert rebuilt["dynamic_objects"][0]["lock_rotation"] is False
    assert rebuilt["catcher"]["net_depth"] == 0.03
    assert rebuilt["arm"]["base_position"] == [0.1, 0.2, 0.3]


def test_catcher_uses_production_mount_coordinate_helpers() -> None:
    calls = []
    runtime = SimpleNamespace(
        sim=SimpleNamespace(
            sync_arm_catcher_mount_for_scene=lambda *args: (
                calls.append(("sync", args)) or np.asarray([1.0, 2.0, 3.0]),
                np.asarray([0.1, 0.2, 0.3]),
            ),
            arm_catcher_tip_target_for_mount=lambda *args: (
                calls.append(("target", args)) or np.asarray([4.0, 5.0, 6.0])
            ),
        )
    )
    environment = object.__new__(ArmCatcherBallEnv)
    environment.sim = runtime.sim
    environment.model = object()
    environment.data = object()
    environment.catcher_spec = {"mount_back_offset": 0.125}
    environment.arm_spec = {"tip_normal_world": [0.0, -1.0, 0.0]}

    position, quaternion, velocity, angular_velocity = environment._end_effector_state()
    tip_target = environment._tip_target_for_mount([3.0, 2.0, 1.0])

    assert position.tolist() == [1.0, 2.0, 3.0]
    assert quaternion.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert velocity.tolist() == [0.1, 0.2, 0.3]
    assert angular_velocity.tolist() == [0.0, 0.0, 0.0]
    assert tip_target.tolist() == [4.0, 5.0, 6.0]
    assert calls[0][0] == "sync"
    assert calls[1][1][1:] == (0.125, [0.0, -1.0, 0.0])


def test_catcher_metrics_are_deterministic() -> None:
    actions = np.asarray(
        [[4.0, 0.0, 0.0, 0.0], [4.0, 0.1, 0.0, 0.0], [4.0, 0.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    smoothness = action_smoothness(actions)
    assert np.isclose(smoothness["action_step_rms"], 0.1)
    assert np.isclose(smoothness["action_second_difference_max"], 0.0, atol=1.0e-7)
    records = [
        {
            "captured": True,
            "missed": False,
            "terminal_reason": "captured",
            "time_to_capture_seconds": 1.0,
            **smoothness,
        },
        {
            "captured": False,
            "missed": True,
            "terminal_reason": "passed_catch_plane",
            "time_to_capture_seconds": None,
            **smoothness,
        },
    ]
    summary = summarize_records(records)
    assert summary["capture_rate"] == 0.5
    assert summary["terminal_reasons"] == {"captured": 1, "passed_catch_plane": 1}
