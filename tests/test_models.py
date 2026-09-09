import pytest
import torch

from sg_jepa.models.lewm import (
    ActionConditionedGRUPredictor,
    ActionConditionedSSMPredictor,
    Predictor,
)


@pytest.mark.parametrize("kind", ["transformer", "gru", "ssm"])
def test_predictor_is_causal(kind: str) -> None:
    torch.manual_seed(3)
    common = dict(
        num_frames=5,
        input_dim=8,
        hidden_dim=8,
        output_dim=8,
        depth=2,
        heads=2,
        mlp_dim=16,
        dim_head=4,
        dropout=0.0,
        emb_dropout=0.0,
        cond_dim=8,
    )
    if kind == "transformer":
        predictor = Predictor(**common)
    elif kind == "gru":
        predictor = ActionConditionedGRUPredictor(
            **common, conditioning="concat", residual_init=0.1
        )
    else:
        predictor = ActionConditionedSSMPredictor(
            **common,
            conditioning="concat",
            residual_init=0.1,
            ssm_state_dim=4,
            ssm_conv_kernel=3,
            ssm_expand=2,
        )
    predictor.eval()
    embedding = torch.randn(2, 5, 8)
    condition = torch.randn(2, 5, 8)
    changed_embedding = embedding.clone()
    changed_condition = condition.clone()
    changed_embedding[:, -1] += 100
    changed_condition[:, -1] -= 100
    first = predictor(embedding, condition)
    changed = predictor(changed_embedding, changed_condition)
    assert torch.equal(first[:, :-1], changed[:, :-1])


def test_gru_checkpoint_parameter_names_are_stable() -> None:
    predictor = ActionConditionedGRUPredictor(
        num_frames=4,
        input_dim=8,
        hidden_dim=8,
        output_dim=8,
        depth=2,
        heads=2,
        mlp_dim=16,
        dim_head=4,
        dropout=0.0,
        emb_dropout=0.0,
        cond_dim=8,
        conditioning="concat",
        residual_init=0.1,
    )
    keys = set(predictor.state_dict())
    assert "pos_embedding" in keys
    assert "layers.0.gru.weight_ih_l0" in keys
    assert "layers.1.residual_scale" in keys
