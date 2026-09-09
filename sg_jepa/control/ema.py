"""Small state-dict EMA with Diffusion Policy's power-law warmup."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn


class EMAModel:
    def __init__(
        self,
        model: nn.Module,
        *,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        max_decay: float = 0.9999,
    ) -> None:
        if inv_gamma <= 0 or power <= 0 or not 0 <= max_decay < 1:
            raise ValueError("invalid EMA schedule")
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.max_decay = float(max_decay)
        self.num_updates = 0
        self.shadow: OrderedDict[str, torch.Tensor] = OrderedDict(
            (name, value.detach().clone()) for name, value in model.state_dict().items()
        )

    @property
    def decay(self) -> float:
        if self.num_updates <= 0:
            return 0.0
        value = 1.0 - (1.0 + self.num_updates / self.inv_gamma) ** (-self.power)
        return min(self.max_decay, value)

    @torch.no_grad()
    def update(self, model: nn.Module) -> float:
        self.num_updates += 1
        decay = self.decay
        state = model.state_dict()
        if state.keys() != self.shadow.keys():
            raise ValueError("model state structure changed after EMA initialization")
        for name, value in state.items():
            shadow = self.shadow[name]
            if torch.is_floating_point(value):
                shadow.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                shadow.copy_(value.detach())
        return decay

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict[str, Any]:
        return {
            "shadow": self.shadow,
            "num_updates": self.num_updates,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "max_decay": self.max_decay,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.num_updates = int(state["num_updates"])
        self.inv_gamma = float(state["inv_gamma"])
        self.power = float(state["power"])
        self.max_decay = float(state["max_decay"])
        if self.shadow.keys() != state["shadow"].keys():
            raise ValueError("EMA checkpoint structure does not match model")
        for name, value in state["shadow"].items():
            self.shadow[name].copy_(value)
