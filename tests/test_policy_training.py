from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from sg_jepa.control.actions import (
    ActionNormalizer,
    controls_to_raw_actions,
    raw_actions_to_controls,
)
from sg_jepa.control.checkpoint import load_policy_checkpoint
from sg_jepa.control.sampling import (
    PAPER_PADDLE_SAMPLER_KIND,
    DeterministicFrankaStrikeBatchSampler,
    DeterministicPaperPaddleBatchSampler,
    load_paddle_quality_weights,
)
from sg_jepa.control.training import (
    DeterministicStepBatchSampler,
    _CachedEpisode,
    _lr_lambda,
    run_policy_training,
)


def test_catcher_action_contract_round_trip_and_clip() -> None:
    raw = torch.tensor([[4.0, -2.0, 0.25, 3.0]])
    controls = raw_actions_to_controls(raw)
    assert controls.shape == (1, 3)
    restored = controls_to_raw_actions(controls, 4.0, xyz_limit=1.0)
    torch.testing.assert_close(restored, torch.tensor([[4.0, -1.0, 0.25, 1.0]]))
    normalizer = ActionNormalizer.fit(torch.tensor([[-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]]))
    torch.testing.assert_close(
        normalizer.denormalize(normalizer.normalize(controls, clip=True)),
        torch.tensor([[-1.0, 0.25, 3.0]]),
    )
    assert normalizer.to_dict()["control_schema"] == ["dx", "dy", "dz"]


def test_paddle_tilt_vector_codec_matches_source_runtime() -> None:
    raw = torch.tensor(
        [
            [4.0, 0.25, -0.50, 0.75, 0.0, 0.2],
            [4.0, -0.25, 0.50, -0.75, -torch.pi / 2.0, 0.3],
        ]
    )
    controls = raw_actions_to_controls(raw, control_codec="paddle_tilt_vector")
    expected_tilt = (
        torch.stack(
            (
                torch.sin(raw[:, 5]) * torch.cos(raw[:, 4]),
                torch.sin(raw[:, 5]) * torch.sin(raw[:, 4]),
            ),
            dim=-1,
        )
        / 0.5
    )
    torch.testing.assert_close(controls[:, :3], raw[:, 1:4])
    torch.testing.assert_close(controls[:, 3:5], expected_tilt)

    restored = controls_to_raw_actions(
        controls,
        4.0,
        control_codec="paddle_tilt_vector",
        max_tilt_theta=torch.pi / 6.0,
    )
    torch.testing.assert_close(restored, raw)


def test_warmup_cosine_and_step_sampler_resume() -> None:
    assert _lr_lambda(0, warmup_steps=2, total_steps=10) == 0.5
    assert _lr_lambda(1, warmup_steps=2, total_steps=10) == 1.0
    assert _lr_lambda(2, warmup_steps=2, total_steps=10) == 1.0
    assert _lr_lambda(10, warmup_steps=2, total_steps=10) == 0.0
    full = list(DeterministicStepBatchSampler(11, 3, seed=5, start_step=0, total_steps=8))
    resumed = list(DeterministicStepBatchSampler(11, 3, seed=5, start_step=3, total_steps=8))
    assert resumed == full[3:]
    assert all(len(batch) == 3 for batch in full)


def test_paper_paddle_sampler_artifact_and_resume_contract() -> None:
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "data_generation/manifests/paddle_fps16_v2_quality_weights.npz"
    )
    artifact = load_paddle_quality_weights(
        artifact_path,
        expected_sha256="92418b6c91131f35b138d9371624ff7d5291198d01bd44b7bdceeeabf7f269e6",
    )
    assert artifact.weights.shape == (7_098, 48)
    weights = np.linspace(0.25, 4.0, 257)
    full = list(
        DeterministicPaperPaddleBatchSampler(
            257,
            quality_weights=weights,
            seed=42,
            start_step=0,
            total_steps=1_030,
        )
    )
    resumed = list(
        DeterministicPaperPaddleBatchSampler(
            257,
            quality_weights=weights,
            seed=42,
            start_step=1_019,
            total_steps=1_030,
        )
    )
    assert resumed == full[1_019:]
    assert all(len(batch) == 256 for batch in full)
    contract = DeterministicPaperPaddleBatchSampler(
        257,
        quality_weights=weights,
        seed=42,
        start_step=0,
        total_steps=1,
    ).contract()
    assert contract == {
        "kind": PAPER_PADDLE_SAMPLER_KIND,
        "batch_size": 256,
        "legacy_replay_count": 205,
        "soft_quality_count": 51,
        "seed": 42,
        "chunk_steps": 1_024,
    }


def test_franka_strike_sampler_population_and_resume_contract() -> None:
    episodes = [
        SimpleNamespace(strike_frame=12, controls=torch.empty(64, 5)),
        SimpleNamespace(strike_frame=20, controls=torch.empty(64, 5)),
    ]
    dataset = SimpleNamespace(
        episodes=episodes,
        config=SimpleNamespace(action_horizon=16),
        windows=[(episode, start) for episode in range(2) for start in range(48)],
    )
    full = list(
        DeterministicFrankaStrikeBatchSampler(
            dataset,
            8,
            seed=42,
            start_step=0,
            total_steps=7,
        )
    )
    resumed = list(
        DeterministicFrankaStrikeBatchSampler(
            dataset,
            8,
            seed=42,
            start_step=3,
            total_steps=7,
        )
    )
    assert resumed == full[3:]
    sampler = DeterministicFrankaStrikeBatchSampler(
        dataset,
        8,
        seed=42,
        start_step=0,
        total_steps=1,
    )
    assert sampler.contract() == {
        "kind": "deterministic_strike_balanced_v1",
        "batch_size": 8,
        "uniform_per_batch": 4,
        "approach_per_batch": 4,
        "approach_horizon": 16,
        "uniform_population": 32,
        "approach_population": 28,
        "seed": 42,
    }


def _tiny_config(path: Path) -> None:
    payload = {
        "policy": {
            "observation_horizon": 2,
            "action_horizon": 4,
            "execution_horizon": 2,
            "embedding_dim": 4,
            "action_dim": 3,
            "diffusion_steps": 4,
            "diffusion_step_embed_dim": 8,
            "down_dims": [8, 16],
            "kernel_size": 3,
            "n_groups": 4,
            "clip_sample": True,
            "feature_mode": "projected_cls",
            "condition_spec": "projected_cls_gravity",
            "history_contract": "repeat_frame_zero_v1",
            "observation_adapter": "latent_gru",
            "gru_hidden_dim": 8,
            "gru_num_layers": 1,
            "gru_dropout": 0.0,
        },
        "training": {
            "steps": 2,
            "batch_size": 2,
            "learning_rate": 1.0e-3,
            "betas": [0.9, 0.99],
            "weight_decay": 0.0,
            "warmup_steps": 1,
            "validate_every": 1,
            "checkpoint_every": 1,
            "max_val_batches": 1,
            "ema_inv_gamma": 1.0,
            "ema_power": 0.75,
            "ema_max_decay": 0.999,
            "top_k": 1,
            "include_validation_in_gradient": False,
            "seed": 11,
            "num_workers": 0,
            "precision": "fp32",
        },
        "evaluation": {
            "sampler": "ddpm",
            "inference_steps": 4,
            "eta": 0.0,
            "rollout_seeds": 1,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_policy_resume_matches_uninterrupted_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "policy.yaml"
    _tiny_config(config_path)
    store = SimpleNamespace(
        has_success=True,
        success=np.asarray([True, True, True, True]),
        split_id=np.asarray([0, 0, 0, 0], dtype=np.int8),
        episodes=[SimpleNamespace(source_id=value) for value in range(4)],
        lengths=np.asarray([9, 9, 9, 9]),
        action_dim=4,
    )

    def fake_encoder(*_args, **_kwargs):
        return object(), {"artifact": "fixture", "weights_sha256": "fixture"}

    def fake_cache(
        _store,
        indices,
        _encoder,
        *,
        action_dim,
        control_codec,
        action_reconstruction,
        require_strike_frame,
        feature_cache_dtype,
        frame_batch_size,
    ):
        del frame_batch_size
        assert action_dim == 3
        assert control_codec == "direct"
        assert action_reconstruction == "stored"
        assert require_strike_frame is False
        assert feature_cache_dtype == "fp16"
        result = {}
        for index in indices:
            generator = torch.Generator().manual_seed(100 + int(index))
            result[int(index)] = _CachedEpisode(
                features=torch.randn(9, 4, generator=generator),
                controls=torch.randn(9, 3, generator=generator).clamp(-1.0, 1.0),
                gravity=3.0 + float(index),
                source_id=int(index),
            )
        return result

    monkeypatch.setattr("sg_jepa.control.training._encoder", fake_encoder)
    monkeypatch.setattr("sg_jepa.control.training.open_trajectory_store", lambda _path: store)
    monkeypatch.setattr("sg_jepa.control.training._cache_episodes", fake_cache)
    dataset_path = tmp_path / "fixture.lance"
    full_output = tmp_path / "full"
    resumed_output = tmp_path / "resumed"
    full_report = run_policy_training(config_path, dataset_path, full_output, device="cpu")
    first_report = run_policy_training(
        config_path, dataset_path, resumed_output, device="cpu", max_steps=1
    )
    resumed_report = run_policy_training(
        config_path,
        dataset_path,
        resumed_output,
        device="cpu",
        max_steps=1,
        resume=resumed_output / "checkpoints/last_policy.pt",
    )
    assert full_report["steps_after"] == resumed_report["steps_after"] == 2
    assert first_report["interrupted_for_smoke"] is True
    assert resumed_report["best_policy_checkpoint"] is not None
    assert (
        len(json.loads((resumed_output / "checkpoints/top_k.json").read_text())["checkpoints"]) == 1
    )
    full = load_policy_checkpoint(full_output / "checkpoints/last_policy.pt")
    resumed = load_policy_checkpoint(resumed_output / "checkpoints/last_policy.pt")
    for name, value in full["policy_state_dict"].items():
        torch.testing.assert_close(value, resumed["policy_state_dict"][name], rtol=0, atol=0)
    for name, value in full["ema"]["shadow"].items():
        torch.testing.assert_close(value, resumed["ema"]["shadow"][name], rtol=0, atol=0)
