import torch

from sg_jepa.losses import SIGReg


def test_sigreg_is_finite_and_differentiable() -> None:
    torch.manual_seed(7)
    embeddings = torch.randn(3, 12, 8, requires_grad=True)
    loss = SIGReg(knots=7, num_proj=16)(embeddings)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
