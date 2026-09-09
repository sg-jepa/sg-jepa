from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from sg_jepa.checkpoints import load_state_dict, strict_load
from sg_jepa.train_utils import rng_state


class _UntrustedObject:
    pass


def test_project_training_checkpoint_rng_state_loads_safely(tmp_path: Path) -> None:
    source = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "training.pt"
    torch.save({"model": source.state_dict(), "rng_state": rng_state()}, checkpoint)

    restored = load_state_dict(checkpoint)
    target = torch.nn.Linear(3, 2)
    strict_load(target, checkpoint)

    assert set(restored) == set(source.state_dict())
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[name], rtol=0, atol=0)


def test_weights_only_checkpoint_loader_still_rejects_arbitrary_globals(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "untrusted.pt"
    torch.save(
        {"model": {"weight": torch.ones(1)}, "object": _UntrustedObject()},
        checkpoint,
    )

    with pytest.raises(pickle.UnpicklingError):
        load_state_dict(checkpoint)
