import torch
from torch import nn

from sg_jepa.optim import MuonAdamW, split_named_parameters


def test_muon_routes_only_matrices() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
    muon, adamw = split_named_parameters(model.named_parameters())
    assert {name for name, _ in muon} == {"0.weight"}
    assert {name for name, _ in adamw} == {"0.bias", "1.weight", "1.bias"}
    optimizer = MuonAdamW(
        [
            {"params": [value for _, value in muon], "use_muon": True},
            {"params": [value for _, value in adamw], "use_muon": False},
        ]
    )
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
