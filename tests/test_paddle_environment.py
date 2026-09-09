from types import SimpleNamespace

from sg_jepa.envs.paddle import ArmPaddleBallEnv


def test_paddle_rebuild_uses_recorded_initial_condition() -> None:
    generated_scene = {
        "metadata": {
            "scenario": "arm_paddle_ball",
            "episode_index": 17,
            "split_name": "test",
            "random_seed": 123,
            "gravity": [0.0, 0.0, -4.0],
        },
        "dynamic_objects": [
            {
                "position": [9.0, 9.0, 9.0],
                "initial_velocity": [9.0, 9.0, 9.0],
                "initial_angular_velocity": [9.0, 9.0, 9.0],
            }
        ],
        "paddle": {
            "initial_position": [9.0, 9.0, 9.0],
            "initial_target_position": [9.0, 9.0, 9.0],
            "initial_target_phi": 9.0,
            "initial_target_theta": 9.0,
        },
    }
    runtime = SimpleNamespace(
        generate=SimpleNamespace(build_episode_scene=lambda *_args: generated_scene)
    )
    environment = object.__new__(ArmPaddleBallEnv)
    environment.runtime = runtime
    environment.generate = runtime.generate
    metadata = {
        "scenario": "arm_paddle_ball",
        "episode_index": 17,
        "split_name": "test",
        "random_seed": 123,
        "gravity": [0.0, 0.0, -4.0],
        "ball_initial_state": {
            "position": [1.0, 2.0, 3.0],
            "linear_velocity": [4.0, 5.0, 6.0],
            "angular_velocity": [7.0, 8.0, 9.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "paddle_initial_state": {
            "position": [0.1, 0.2, 0.3],
            "target_position": [0.4, 0.5, 0.6],
            "target_phi": 0.7,
            "target_theta": 0.8,
        },
    }

    rebuilt = environment._rebuild_scene(metadata)

    assert rebuilt["metadata"] == metadata
    assert rebuilt["dynamic_objects"][0]["position"] == [1.0, 2.0, 3.0]
    assert rebuilt["dynamic_objects"][0]["initial_velocity"] == [4.0, 5.0, 6.0]
    assert rebuilt["dynamic_objects"][0]["initial_angular_velocity"] == [7.0, 8.0, 9.0]
    assert rebuilt["dynamic_objects"][0]["initial_quaternion"] == [0.0, 0.0, 0.0, 1.0]
    assert rebuilt["paddle"]["initial_position"] == [0.1, 0.2, 0.3]
    assert rebuilt["paddle"]["initial_target_position"] == [0.4, 0.5, 0.6]
    assert rebuilt["paddle"]["initial_target_phi"] == 0.7
    assert rebuilt["paddle"]["initial_target_theta"] == 0.8
