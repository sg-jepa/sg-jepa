"""Muon + AdamW optimizer utilities for Semigroup-JEPA training."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

MUON_LR_ADJUSTMENTS = ("original", "match_rms_adamw")


def zeropower_via_newtonschulz5(
    update: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate the orthogonal factor of a 2D update."""

    if update.ndim != 2:
        raise ValueError(f"Muon expects 2D updates, got shape {tuple(update.shape)}")

    a, b, c = (3.4445, -4.7750, 2.0315)
    work_dtype = torch.bfloat16 if update.device.type == "cuda" else torch.float32
    x = update.to(dtype=work_dtype)
    x = x / (x.norm() + eps)

    transposed = False
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True

    for _ in range(int(steps)):
        xx_t = x @ x.T
        bx = xx_t @ x
        x = a * x + b * bx + c * (xx_t @ bx)

    if transposed:
        x = x.T
    return x


def _validate_muon_lr_adjustment(value: str) -> str:
    if value not in MUON_LR_ADJUSTMENTS:
        allowed = ", ".join(MUON_LR_ADJUSTMENTS)
        raise ValueError(f"Unsupported Muon LR adjustment {value!r}; choose {allowed}")
    return value


def muon_update_scale(
    update: torch.Tensor,
    *,
    adjustment: str = "original",
    rms_match_scale: float = 0.2,
) -> float:
    """Return the per-matrix Muon LR multiplier.

    ``original`` follows Keller Jordan's implementation:
    sqrt(max(1, rows / cols)).

    ``match_rms_adamw`` follows Liu et al. 2025:
    0.2 * sqrt(max(rows, cols)), with the 0.2 exposed as rms_match_scale.
    """

    if update.ndim != 2:
        raise ValueError(f"Muon update scale expects 2D tensor, got {update.ndim}D")
    adjustment = _validate_muon_lr_adjustment(adjustment)
    rows, cols = update.shape
    if adjustment == "match_rms_adamw":
        return float(rms_match_scale) * math.sqrt(max(rows, cols))
    return math.sqrt(max(1.0, rows / cols))


class MuonAdamW(torch.optim.Optimizer):
    """Route 2D tensors to Muon and all other tensors to AdamW."""

    def __init__(self, param_groups: list[dict[str, Any]]) -> None:
        if not param_groups:
            raise ValueError("MuonAdamW requires at least one parameter group")
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each parameter group must define use_muon")
            if group["use_muon"]:
                group.setdefault("lr", 5e-5)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("momentum", 0.95)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
                group.setdefault("lr_adjustment", "match_rms_adamw")
                group.setdefault("rms_match_scale", 0.2)
                _validate_muon_lr_adjustment(str(group["lr_adjustment"]))
            else:
                group.setdefault("lr", 5e-5)
                group.setdefault("weight_decay", 1e-3)
                group.setdefault("betas", (0.9, 0.999))
                group.setdefault("eps", 1e-8)
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        momentum = float(group["momentum"])
        ns_steps = int(group["ns_steps"])
        nesterov = bool(group["nesterov"])
        lr_adjustment = str(group["lr_adjustment"])
        rms_match_scale = float(group["rms_match_scale"])

        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("MuonAdamW does not support sparse gradients")
            if grad.ndim != 2:
                raise RuntimeError(f"Muon group received non-2D gradient shape {tuple(grad.shape)}")

            state = self.state[param]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(param)

            if weight_decay:
                param.mul_(1.0 - lr * weight_decay)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)
            update = grad.add(buf, alpha=momentum) if nesterov else buf
            update = zeropower_via_newtonschulz5(update, steps=ns_steps)
            update = update * muon_update_scale(
                update,
                adjustment=lr_adjustment,
                rms_match_scale=rms_match_scale,
            )
            param.add_(update.to(dtype=param.dtype), alpha=-lr)

    def _step_adamw_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])

        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("MuonAdamW does not support sparse gradients")

            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)

            state["step"] += 1
            step = int(state["step"])

            if weight_decay:
                param.mul_(1.0 - lr * weight_decay)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-(lr / bias_correction1))


def split_named_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    """Split trainable parameters by the requested matrix rule."""

    muon_params: list[tuple[str, torch.nn.Parameter]] = []
    adamw_params: list[tuple[str, torch.nn.Parameter]] = []
    seen: set[int] = set()
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        if param.ndim == 2:
            muon_params.append((name, param))
        else:
            adamw_params.append((name, param))
    return muon_params, adamw_params


def _param_groups_from_named(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    adamw_lr: float,
    weight_decay: float,
    muon_lr: float,
    muon_momentum: float,
    muon_ns_steps: int,
    muon_lr_adjustment: str,
    muon_rms_match_scale: float,
) -> list[dict[str, Any]]:
    muon_named, adamw_named = split_named_parameters(named_parameters)
    param_groups: list[dict[str, Any]] = []
    if muon_named:
        param_groups.append(
            {
                "params": [param for _, param in muon_named],
                "use_muon": True,
                "lr": float(muon_lr),
                "weight_decay": float(weight_decay),
                "momentum": float(muon_momentum),
                "nesterov": True,
                "ns_steps": int(muon_ns_steps),
                "lr_adjustment": _validate_muon_lr_adjustment(muon_lr_adjustment),
                "rms_match_scale": float(muon_rms_match_scale),
            }
        )
    if adamw_named:
        param_groups.append(
            {
                "params": [param for _, param in adamw_named],
                "use_muon": False,
                "lr": float(adamw_lr),
                "weight_decay": float(weight_decay),
                "betas": (0.9, 0.999),
                "eps": 1e-8,
            }
        )
    return param_groups


def build_muon_adamw_optimizer(
    module: torch.nn.Module,
    cfg: Any,
    *,
    adamw_lr: float | None = None,
    weight_decay: float | None = None,
) -> MuonAdamW:
    """Build MuonAdamW for a named module."""

    return MuonAdamW(
        _param_groups_from_named(
            module.named_parameters(),
            adamw_lr=float(cfg.optimizer.lr if adamw_lr is None else adamw_lr),
            weight_decay=float(
                cfg.optimizer.weight_decay if weight_decay is None else weight_decay
            ),
            muon_lr=float(cfg.optimizer.muon_lr),
            muon_momentum=float(cfg.optimizer.muon_momentum),
            muon_ns_steps=int(cfg.optimizer.muon_ns_steps),
            muon_lr_adjustment=str(getattr(cfg.optimizer, "muon_lr_adjustment", "match_rms_adamw")),
            muon_rms_match_scale=float(getattr(cfg.optimizer, "muon_rms_match_scale", 0.2)),
        )
    )


def build_adamw_optimizer(
    module: torch.nn.Module,
    cfg: Any,
    *,
    lr: float | None = None,
    weight_decay: float | None = None,
) -> torch.optim.AdamW:
    """Build a plain AdamW optimizer over all trainable module parameters."""

    return torch.optim.AdamW(
        (param for param in module.parameters() if param.requires_grad),
        lr=float(cfg.optimizer.lr if lr is None else lr),
        weight_decay=float(cfg.optimizer.weight_decay if weight_decay is None else weight_decay),
        betas=(0.9, 0.999),
        eps=1e-8,
    )


@dataclass(frozen=True)
class MuonAdamWFactory:
    """Picklable optimizer factory accepted by stable_pretraining.create_optimizer."""

    adamw_lr: float = 5e-5
    weight_decay: float = 1e-3
    muon_lr: float = 5e-5
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_lr_adjustment: str = "match_rms_adamw"
    muon_rms_match_scale: float = 0.2

    def __call__(self, params) -> MuonAdamW:
        named = [
            (f"param_{idx}", param)
            for idx, param in enumerate(params)
            if getattr(param, "requires_grad", False)
        ]
        return MuonAdamW(
            _param_groups_from_named(
                named,
                adamw_lr=self.adamw_lr,
                weight_decay=self.weight_decay,
                muon_lr=self.muon_lr,
                muon_momentum=self.muon_momentum,
                muon_ns_steps=self.muon_ns_steps,
                muon_lr_adjustment=self.muon_lr_adjustment,
                muon_rms_match_scale=self.muon_rms_match_scale,
            )
        )


def make_muon_adamw_factory(
    cfg: Any,
    *,
    adamw_lr: float | None = None,
    weight_decay: float | None = None,
    muon_lr: float | None = None,
    muon_lr_adjustment: str | None = None,
    muon_rms_match_scale: float | None = None,
) -> MuonAdamWFactory:
    return MuonAdamWFactory(
        adamw_lr=float(cfg.optimizer.lr if adamw_lr is None else adamw_lr),
        weight_decay=float(cfg.optimizer.weight_decay if weight_decay is None else weight_decay),
        muon_lr=float(cfg.optimizer.muon_lr if muon_lr is None else muon_lr),
        muon_momentum=float(cfg.optimizer.muon_momentum),
        muon_ns_steps=int(cfg.optimizer.muon_ns_steps),
        muon_lr_adjustment=str(
            getattr(cfg.optimizer, "muon_lr_adjustment", "match_rms_adamw")
            if muon_lr_adjustment is None
            else muon_lr_adjustment
        ),
        muon_rms_match_scale=float(
            getattr(cfg.optimizer, "muon_rms_match_scale", 0.2)
            if muon_rms_match_scale is None
            else muon_rms_match_scale
        ),
    )


def _entries(named_params: list[tuple[str, torch.nn.Parameter]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(param.shape),
            "ndim": int(param.ndim),
            "numel": int(param.numel()),
        }
        for name, param in named_params
    ]


def optimizer_param_group_report(
    modules: dict[str, torch.nn.Module],
    cfg: Any | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable report of Muon vs AdamW routing."""

    report: dict[str, Any] = {
        "rule": ("trainable parameter.ndim == 2 -> muon; all other trainable parameters -> adamw"),
        "modules": {},
        "counts": {
            "muon_tensors": 0,
            "muon_numel": 0,
            "adamw_tensors": 0,
            "adamw_numel": 0,
        },
    }
    optimizer_type = "muon_adamw"
    if cfg is not None:
        optimizer_type = str(getattr(cfg.optimizer, "type", "muon_adamw")).lower()
        report["optimizer"] = {
            "type": optimizer_type,
            "world_model_adamw_lr": float(cfg.optimizer.lr),
            "muon_lr": float(cfg.optimizer.muon_lr),
            "muon_lr_adjustment": str(
                getattr(cfg.optimizer, "muon_lr_adjustment", "match_rms_adamw")
            ),
            "muon_rms_match_scale": float(getattr(cfg.optimizer, "muon_rms_match_scale", 0.2)),
            "weight_decay": float(cfg.optimizer.weight_decay),
            "muon_momentum": float(cfg.optimizer.muon_momentum),
            "muon_ns_steps": int(cfg.optimizer.muon_ns_steps),
            "lr_schedule": str(getattr(cfg.optimizer, "lr_schedule", "constant")),
            "lr_cooldown_start_fraction": float(
                getattr(cfg.optimizer, "lr_cooldown_start_fraction", 0.5)
            ),
            "lr_cycle_count": float(getattr(cfg.optimizer, "lr_cycle_count", 1.0)),
            "lr_min_factor": float(getattr(cfg.optimizer, "lr_min_factor", 0.1)),
        }
        if hasattr(cfg, "online_decoder") and bool(cfg.online_decoder.enabled):
            report["optimizer"]["online_decoder_adamw_lr"] = float(cfg.online_decoder.lr)
            report["optimizer"]["online_decoder_muon_lr"] = float(cfg.online_decoder.muon_lr)
            report["optimizer"]["online_decoder_weight_decay"] = float(
                cfg.online_decoder.weight_decay
            )

    for module_name, module in modules.items():
        if optimizer_type == "adamw":
            muon_named = []
            adamw_named = [
                (name, param) for name, param in module.named_parameters() if param.requires_grad
            ]
        else:
            muon_named, adamw_named = split_named_parameters(module.named_parameters())
        muon_entries = _entries(muon_named)
        adamw_entries = _entries(adamw_named)
        report["modules"][module_name] = {
            "muon": muon_entries,
            "adamw": adamw_entries,
            "counts": {
                "muon_tensors": len(muon_entries),
                "muon_numel": sum(item["numel"] for item in muon_entries),
                "adamw_tensors": len(adamw_entries),
                "adamw_numel": sum(item["numel"] for item in adamw_entries),
            },
        }
        for key, value in report["modules"][module_name]["counts"].items():
            report["counts"][key] += value
    return report


__all__ = [
    "MuonAdamW",
    "MuonAdamWFactory",
    "build_adamw_optimizer",
    "build_muon_adamw_optimizer",
    "make_muon_adamw_factory",
    "optimizer_param_group_report",
    "split_named_parameters",
    "zeropower_via_newtonschulz5",
]
