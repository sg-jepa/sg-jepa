import torch
from torch import nn

from sg_jepa.losses import autoregressive_rollout_loss


class AdditiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_encoder = nn.Identity()

    def predict(self, embedding: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return embedding + condition


def test_discounted_rollout_is_normalized_weighted_mean() -> None:
    model = AdditiveModel()
    embedding = torch.zeros(1, 5, 2)
    action = torch.ones(1, 5, 2)
    loss, terms = autoregressive_rollout_loss(
        model=model,
        emb=embedding,
        batch={"action": action},
        history_size=2,
        horizon=3,
        gamma=0.5,
    )
    per_horizon = torch.tensor((1.0, 4.0, 9.0))
    weights = torch.tensor((1.0, 0.5, 0.25))
    expected = (per_horizon * weights).sum() / weights.sum()
    assert torch.allclose(loss, expected)
    assert list(terms) == ["rollout_h1_loss", "rollout_h2_loss", "rollout_h3_loss"]


def test_rollout_rejects_short_sequences() -> None:
    with torch.no_grad():
        try:
            autoregressive_rollout_loss(
                model=AdditiveModel(),
                emb=torch.zeros(1, 3, 2),
                batch={"action": torch.ones(1, 3, 2)},
                history_size=2,
                horizon=2,
            )
        except ValueError as error:
            assert "cover rollout horizon" in str(error) or "sequence length" in str(error)
        else:  # pragma: no cover
            raise AssertionError("short rollout unexpectedly accepted")
