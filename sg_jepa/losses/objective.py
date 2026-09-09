"""Original LeWM and Semigroup-JEPA training objectives."""

from __future__ import annotations

from typing import Any

import torch

from sg_jepa.config import ObjectiveConfig

from .rollout import autoregressive_rollout_loss
from .sigreg import SIGReg


def compute_objective(
    model,
    batch: dict[str, Any],
    config: ObjectiveConfig,
    sigreg: SIGReg,
) -> dict[str, torch.Tensor]:
    """Compute the exact K=0 or normalized discounted K-step objective."""

    output = model.encode(dict(batch))
    embeddings = output["emb"]
    history = int(model.predictor.pos_embedding.shape[1])
    if embeddings.shape[1] < history + max(1, config.rollout_horizon):
        raise ValueError("encoded sequence is too short for the configured objective")
    predicted = model.predict(
        embeddings[:, :history],
        output["act_emb"][:, :history],
    )
    target = embeddings[:, 1 : history + 1]
    pred_loss = (predicted - target).pow(2).mean()
    sigreg_loss = sigreg(embeddings.transpose(0, 1))
    total = config.prediction_weight * pred_loss + config.sigreg_weight * sigreg_loss
    losses: dict[str, torch.Tensor] = {
        "loss": total,
        "prediction_loss": pred_loss,
        "sigreg_loss": sigreg_loss,
    }
    if config.rollout_weight > 0:
        rollout_loss, rollout_terms = autoregressive_rollout_loss(
            model=model,
            emb=embeddings,
            batch=batch,
            history_size=history,
            horizon=config.rollout_horizon,
            gamma=config.rollout_gamma,
            target_detach=config.target_detach,
        )
        losses["rollout_loss"] = rollout_loss
        losses.update(rollout_terms)
        losses["loss"] = losses["loss"] + config.rollout_weight * rollout_loss
    return losses
