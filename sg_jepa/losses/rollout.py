"""Discounted autoregressive latent rollout loss for v10 LeWM training."""

from __future__ import annotations

from typing import Any

import torch


def _rollout_condition_embeddings(
    model,
    batch: dict[str, Any],
    action: torch.Tensor,
) -> torch.Tensor:
    """Return predictor conditions for every action step in ``action``."""

    if hasattr(model, "encode_action_condition"):
        if "gravity_condition" not in batch:
            raise KeyError("gravity-conditioned rollout loss requires batch['gravity_condition']")
        gravity_condition = batch["gravity_condition"]
        if (
            torch.is_tensor(gravity_condition)
            and gravity_condition.ndim >= 3
            and gravity_condition.size(1) >= action.size(1)
        ):
            gravity_condition = gravity_condition[:, : action.size(1)]
        encoded = model.encode_action_condition(action, gravity_condition)
        if not isinstance(encoded, tuple) or not encoded:
            raise TypeError(
                "encode_action_condition must return a non-empty tuple whose last "
                "item is the predictor condition embedding"
            )
        return encoded[-1]
    return model.action_encoder(action)


def autoregressive_rollout_loss(
    *,
    model,
    emb: torch.Tensor,
    batch: dict[str, Any],
    history_size: int,
    horizon: int,
    gamma: float = 0.95,
    target_detach: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute normalized discounted autoregressive latent MSE.

    ``emb`` is the encoded latent sequence with shape ``(B, T, D)``. The rollout
    starts from the first ``history_size`` real latents, predicts the next latent,
    appends that prediction, and repeats using the same sliding-window contract
    as the model inference ``rollout`` methods.
    """

    if "action" not in batch:
        raise KeyError("rollout loss requires batch['action']")
    if emb.ndim != 3:
        raise ValueError(f"emb must have shape (B,T,D), got {tuple(emb.shape)}")
    action = batch["action"]
    if action.ndim != 3:
        raise ValueError(f"batch['action'] must have shape (B,T,A), got {tuple(action.shape)}")

    history_size = int(history_size)
    horizon = int(horizon)
    if history_size <= 0:
        raise ValueError(f"history_size must be positive, got {history_size}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")

    batch_size, sequence_len, _dim = emb.shape
    if action.shape[0] != batch_size or action.shape[1] < history_size + horizon:
        raise ValueError(
            "action sequence must align with embeddings and cover rollout horizon; "
            f"emb={tuple(emb.shape)} action={tuple(action.shape)} "
            f"history={history_size} horizon={horizon}"
        )
    if sequence_len < history_size + horizon:
        raise ValueError(
            "Need encoded sequence length >= history_size + horizon, "
            f"got T={sequence_len}, history={history_size}, horizon={horizon}"
        )

    cond_emb = _rollout_condition_embeddings(model, batch, action[:, : history_size + horizon])
    if cond_emb.shape[:2] != (batch_size, history_size + horizon):
        raise ValueError(
            "rollout condition embeddings must align with action time axis; "
            f"got {tuple(cond_emb.shape)} for batch={batch_size}, "
            f"time={history_size + horizon}"
        )

    emb_roll = emb[:, :history_size]
    preds: list[torch.Tensor] = []
    for step in range(horizon):
        emb_trunc = emb_roll[:, -history_size:]
        cond_trunc = cond_emb[:, step : step + history_size]
        pred_next = model.predict(emb_trunc, cond_trunc)[:, -1:]
        preds.append(pred_next)
        emb_roll = torch.cat([emb_roll, pred_next], dim=1)

    pred = torch.cat(preds, dim=1)
    target = emb[:, history_size : history_size + horizon]
    if target_detach:
        target = target.detach()

    loss_per_horizon = (pred - target).pow(2).mean(dim=(0, 2))
    weights = float(gamma) ** torch.arange(
        horizon,
        device=emb.device,
        dtype=emb.dtype,
    )
    weights = weights / weights.mean()
    loss = (weights * loss_per_horizon).mean()

    logs = {f"rollout_h{idx + 1}_loss": value for idx, value in enumerate(loss_per_horizon)}
    return loss, logs


__all__ = ["autoregressive_rollout_loss"]
