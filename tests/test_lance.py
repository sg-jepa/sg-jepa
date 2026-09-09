import json
import os
from pathlib import Path

import numpy as np
import pytest

from data_generation.common.lance import EpisodeRecord, write_lance_episodes
from sg_jepa.data import (
    TrajectoryWindowDataset,
    open_trajectory_store,
    validate_trajectory_dataset,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SGJEPA_TEST_LANCE") != "1",
    reason="set SGJEPA_TEST_LANCE=1 in an environment with the Lance extra",
)


def test_common_lance_schema_loads_into_training_windows(tmp_path: Path) -> None:
    pytest.importorskip("lance")
    pixels = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    state = np.arange(16, dtype=np.float32).reshape(4, 4)
    records = [
        EpisodeRecord(
            episode_index=index,
            split=split,
            pixels=pixels + index,
            state=state,
            action=np.full((4, 3), index + 1, dtype=np.float32),
            gravity=float(index + 1),
            physics=np.asarray([index + 1], dtype=np.float32),
            source_episode_index=100 + index,
        )
        for index, split in enumerate(("train", "train", "test"))
    ]
    dataset = tmp_path / "data.lance"
    write_lance_episodes(
        records,
        dataset,
        task="square",
        fps=16,
        image_size=32,
        frames_per_episode=4,
        action_names=("impulse_x", "impulse_z", "g"),
        state_names=("x", "z", "vx", "vz"),
        physics_names=("g",),
    )
    (tmp_path / "generation_manifest.json").write_text(json.dumps({"dataset": "data.lance"}) + "\n")
    report = validate_trajectory_dataset(tmp_path, expected_task="square")
    assert report["episodes"] == 3
    assert report["frames_min"] == report["frames_max"] == 4
    store = open_trajectory_store(tmp_path)
    assert [episode.source_id for episode in store.episodes] == [100, 101, 102]
    windows = TrajectoryWindowDataset(dataset, num_steps=3)
    assert len(windows) == 4
    assert windows[0]["pixels"].shape == (3, 3, 32, 32)
