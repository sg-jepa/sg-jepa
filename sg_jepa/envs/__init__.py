"""Lazy access to optional MuJoCo control environments."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ArmCatcherBallEnv",
    "ArmPaddleBallEnv",
    "BasketRuntime",
    "BasketSimulator",
    "CatcherRuntime",
    "PaddleRuntime",
]


def __getattr__(name: str) -> Any:
    if name in {"BasketRuntime", "BasketSimulator"}:
        return getattr(import_module(".franka", __name__), name)
    if name in {"ArmCatcherBallEnv", "CatcherRuntime"}:
        return getattr(import_module(".catcher", __name__), name)
    if name in {"ArmPaddleBallEnv", "PaddleRuntime"}:
        return getattr(import_module(".paddle", __name__), name)
    raise AttributeError(name)
