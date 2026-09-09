from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from sg_jepa import training as training_module
from sg_jepa.config import (
    ExperimentConfig,
    ObjectiveConfig,
    TrainConfig,
    WorldModelConfig,
)
from sg_jepa.data import ActionStatistics, split_development_episodes


def test_paper_episode_split_is_deterministic_and_leakage_free() -> None:
    store = SimpleNamespace(
        split_id=np.asarray([0] * 8_000 + [1] * 200, dtype=np.int8),
        episodes=[SimpleNamespace(source_id=index + 10_000) for index in range(8_200)],
    )
    first = split_development_episodes(store, train_fraction=0.9, seed=42)
    second = split_development_episodes(store, train_fraction=0.9, seed=42)
    assert first == second
    assert len(first.train_indices) == 7_200
    assert len(first.val_indices) == 800
    assert set(first.train_indices).isdisjoint(first.val_indices)
    assert set(first.train_episode_ids).isdisjoint(first.val_episode_ids)
    assert not (set(first.train_indices) | set(first.val_indices)) & set(range(8_000, 8_200))


class _ToyDataset(Dataset):
    action_dim = 1
    action_statistics = ActionStatistics((0.0,), (1.0,))

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int):
        return {
            "pixels": torch.tensor([self.values[index]], dtype=torch.float32),
            "action": torch.zeros(1),
        }


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1))


def test_native_training_accumulates_validates_and_resumes(monkeypatch, tmp_path) -> None:
    train = _ToyDataset([1.0, 2.0, 3.0, 4.0, 5.0])
    validation = _ToyDataset([1.5, 2.5])
    split = {
        "train_episode_indices": [0, 1, 2, 3],
        "val_episode_indices": [4],
        "train_window_count": len(train),
        "val_window_count": len(validation),
    }
    monkeypatch.setattr(
        training_module,
        "partition_window_datasets",
        lambda *_args, **_kwargs: (train, validation, split),
    )
    monkeypatch.setattr(training_module, "build_world_model", lambda _config: _ToyModel())

    def objective(model, batch, _config, _sigreg):
        loss = (model.weight * batch["pixels"].mean()).square().mean()
        return {"loss": loss, "prediction_loss": loss, "sigreg_loss": loss * 0.0}

    monkeypatch.setattr(training_module, "compute_objective", objective)
    config = ExperimentConfig(
        name="toy",
        model=WorldModelConfig(
            image_size=8,
            patch_size=8,
            embed_dim=8,
            action_dim=1,
            history_size=1,
            predictor_kind="gru",
        ),
        objective=ObjectiveConfig(
            prediction_weight=1.0,
            rollout_weight=0.0,
            rollout_horizon=1,
            sigreg_weight=0.0,
            sigreg_num_proj=1,
        ),
        train=TrainConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1.0e-3,
            muon_learning_rate=1.0e-3,
            gradient_accumulation_steps=2,
            precision="fp32",
        ),
    )
    output = tmp_path / "run"
    interrupted = training_module.run_training(
        config,
        tmp_path,
        output,
        device="cpu",
        max_steps=1,
    )
    assert interrupted["steps_after"] == 1
    assert interrupted["completed_epochs"] == 0
    assert interrupted["best_checkpoint"] is None

    resumed = training_module.run_training(
        replace(config),
        tmp_path,
        output,
        device="cpu",
        resume=output / "checkpoint.pt",
        max_steps=1,
    )
    assert resumed["steps_before"] == 1
    assert resumed["steps_after"] == 2
    assert resumed["completed_epochs"] == 1
    assert resumed["best_checkpoint"] is not None
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=False)
    assert checkpoint["cursor"] == {
        "next_epoch": 1,
        "next_microbatch": 0,
        "completed_epochs": 1,
    }
    assert (output / "checkpoint_epoch_001.pt").is_file()
