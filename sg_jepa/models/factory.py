"""Construct checkpoint-compatible main-text world models."""

from __future__ import annotations

from torch import nn

from sg_jepa.config import WorldModelConfig

from .lewm import (
    MLP,
    ActionConditionedGRUPredictor,
    ActionConditionedSSMPredictor,
    Embedder,
    LeWM,
    Predictor,
)

_VIT_SIZES = {
    "tiny": (192, 12, 3),
    "small": (384, 12, 6),
    "base": (768, 12, 12),
    "large": (1024, 24, 16),
}


def build_vit_encoder(config: WorldModelConfig) -> nn.Module:
    """Reproduce ``stable_pretraining.backbone.utils.vit_hf`` locally."""

    try:
        from transformers import ViTConfig, ViTModel
    except ImportError as exc:  # pragma: no cover - dependency error is explicit.
        raise ImportError("Install the core transformers dependency to build LeWM") from exc

    if config.encoder_scale not in _VIT_SIZES:
        raise ValueError(f"unsupported encoder_scale={config.encoder_scale!r}")
    hidden, layers, heads = _VIT_SIZES[config.encoder_scale]
    hidden = int(config.encoder_hidden_size or hidden)
    layers = int(config.encoder_layers or layers)
    heads = int(
        config.encoder_heads
        or (heads if config.encoder_hidden_size is None else max(1, hidden // 64))
    )
    if hidden % heads:
        raise ValueError("encoder hidden size must be divisible by encoder heads")
    encoder_config = ViTConfig(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=4 * hidden,
        image_size=config.image_size,
        patch_size=config.patch_size,
    )
    encoder = ViTModel(
        encoder_config,
        add_pooling_layer=False,
        use_mask_token=False,
    )
    encoder.config.interpolate_pos_encoding = True
    return encoder


def build_world_model(config: WorldModelConfig) -> LeWM:
    encoder = build_vit_encoder(config)
    predictor_kwargs = dict(
        num_frames=config.history_size,
        input_dim=config.embed_dim,
        hidden_dim=config.predictor_hidden_dim,
        output_dim=config.embed_dim,
        depth=config.predictor_depth,
        heads=config.predictor_heads,
        mlp_dim=config.predictor_mlp_dim,
        dim_head=config.predictor_dim_head,
        dropout=config.predictor_dropout,
        emb_dropout=config.predictor_emb_dropout,
        cond_dim=config.embed_dim,
    )
    if config.predictor_kind == "transformer":
        predictor = Predictor(**predictor_kwargs)
    elif config.predictor_kind == "gru":
        predictor = ActionConditionedGRUPredictor(
            **predictor_kwargs,
            conditioning=config.predictor_conditioning,
            residual_init=config.predictor_residual_init,
        )
    else:
        predictor = ActionConditionedSSMPredictor(
            **predictor_kwargs,
            conditioning=config.predictor_conditioning,
            residual_init=config.predictor_residual_init,
            ssm_state_dim=config.ssm_state_dim,
            ssm_conv_kernel=config.ssm_conv_kernel,
            ssm_expand=config.ssm_expand,
            ssm_dt_rank=config.ssm_dt_rank,
        )
    action_encoder = Embedder(input_dim=config.action_dim, emb_dim=config.embed_dim)
    projector = MLP(
        input_dim=int(encoder.config.hidden_size),
        output_dim=config.embed_dim,
        hidden_dim=config.projector_hidden_dim,
        norm_fn=nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=config.embed_dim,
        output_dim=config.embed_dim,
        hidden_dim=config.projector_hidden_dim,
        norm_fn=nn.BatchNorm1d,
    )
    return LeWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
