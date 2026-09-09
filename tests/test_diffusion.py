import numpy as np
import torch
from torch import nn

from sg_jepa.control import inference
from sg_jepa.control.actions import (
    FIXED_PADDLE_GRAVITY,
    ActionNormalizer,
    FeatureNormalizer,
    controls_to_raw_actions,
    full_range_translation_delta,
    raw_actions_to_controls,
)
from sg_jepa.control.diffusion import GaussianDiffusion1D


class ZeroNoise(nn.Module):
    def forward(self, sample, timestep, embeddings, gravity):
        return torch.zeros_like(sample)


def test_ddim_honors_step_count_and_is_deterministic() -> None:
    diffusion = GaussianDiffusion1D(8)
    embeddings = torch.zeros(2, 4, 3)
    gravity = torch.zeros(2, 1)
    noise = torch.randn(2, 8, 5, generator=torch.Generator().manual_seed(2))
    kwargs = dict(
        model=ZeroNoise(),
        embeddings=embeddings,
        gravity=gravity,
        action_horizon=8,
        action_dim=5,
        sampler="ddim",
        num_inference_steps=4,
        eta=0.0,
        initial_noise=noise,
    )
    first = diffusion.sample(**kwargs)
    second = diffusion.sample(**kwargs)
    assert torch.equal(first, second)
    assert diffusion.last_sample_timesteps == (7, 5, 2, 0)


def test_ddpm_executes_the_complete_schedule() -> None:
    diffusion = GaussianDiffusion1D(4)
    embeddings = torch.zeros(1, 2, 3)
    gravity = torch.zeros(1, 1)
    result = diffusion.sample(
        ZeroNoise(),
        embeddings,
        gravity,
        action_horizon=4,
        action_dim=5,
        sampler="ddpm",
        num_inference_steps=4,
        generator=torch.Generator().manual_seed(5),
    )
    assert result.shape == (1, 4, 5)
    assert diffusion.last_sample_timesteps == (3, 2, 1, 0)


def test_franka_action_round_trip() -> None:
    controls = torch.randn(2, 3, 5)
    raw = controls_to_raw_actions(controls, torch.tensor((4.0, 5.0)))
    assert raw.shape == (2, 3, 6)
    assert torch.equal(raw_actions_to_controls(raw), controls)


def test_feature_normalizer_round_trips_checkpoint_payload() -> None:
    payload = {
        "embedding_mean": [1.0, -2.0],
        "embedding_std": [2.0, 4.0],
        "gravity_mean": 9.0,
        "gravity_std": 2.0,
        "epsilon": 1.0e-6,
        "fit_population": "successful_world_model_training_partition_only",
    }
    normalizer = FeatureNormalizer.from_dict(payload)
    values = torch.tensor([[3.0, 2.0]])
    assert torch.equal(normalizer.normalize_embeddings(values), torch.ones_like(values))
    assert normalizer.normalize_gravity(11.0) == 1.0


def test_paddle_feature_normalizer_uses_fixed_physical_gravity_range() -> None:
    normalizer = FeatureNormalizer.fit(
        torch.tensor([[1.0, 2.0], [3.0, 6.0]]),
        np.asarray([3.8, 4.2]),
        gravity_kind=FIXED_PADDLE_GRAVITY,
    )
    assert normalizer.gravity_mean == normalizer.gravity_std == 10.0
    assert normalizer.normalize_gravity(0.0) == -1.0
    assert normalizer.normalize_gravity(20.0) == 1.0
    assert FeatureNormalizer.from_dict(normalizer.to_dict()) == normalizer


def test_action_normalizer_fit_and_round_trip() -> None:
    controls = torch.tensor([[-2.0, 0.0, 1.0, 3.0, 4.0], [2.0, 2.0, 5.0, 7.0, 8.0]])
    normalizer = ActionNormalizer.fit(controls)
    restored = normalizer.denormalize(normalizer.normalize(controls))
    torch.testing.assert_close(restored, controls)


def test_full_range_action_contract_preserves_uncapped_delta() -> None:
    action = np.asarray([2.5, -1.5, 0.25], dtype=np.float32)
    scale = np.asarray([0.04, 0.04, 0.04], dtype=np.float64)
    delta = full_range_translation_delta(action, scale)
    np.testing.assert_allclose(delta, action.astype(np.float64) * scale, rtol=1.0e-7)
    assert np.max(np.abs(action)) > 1.0


def test_converted_encoder_retains_source_identity(tmp_path) -> None:
    checkpoint = tmp_path / "encoder.safetensors"
    checkpoint.write_bytes(b"converted weights")
    checkpoint.with_suffix(".json").write_text(
        '{"source": {"sha256": "historical-source-sha256"}}\n'
    )
    assert inference._checkpoint_source_identity(checkpoint) == "historical-source-sha256"
