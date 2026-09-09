"""Device-safe sliced Epps-Pulley regularization used by the paper runs."""

from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        if knots < 2 or num_proj <= 0:
            raise ValueError("knots must be >= 2 and num_proj must be positive")
        self.num_proj = int(num_proj)
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt_step = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt_step, dtype=torch.float32)
        weights[[0, -1]] = dt_step
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        basis = torch.randn(
            projections.size(-1),
            self.num_proj,
            device=projections.device,
            dtype=projections.dtype,
        )
        basis = basis.div_(basis.norm(p=2, dim=0).clamp_min(1e-12))
        x_t = (projections @ basis).unsqueeze(-1) * self.t.to(projections)
        error = (x_t.cos().mean(-3) - self.phi.to(projections)).square()
        error = error + x_t.sin().mean(-3).square()
        statistic = (error @ self.weights.to(projections)) * projections.size(-2)
        return statistic.mean()
