"""Checkpoint-compatible token predictor for the DINO-WM baseline.

This is the minimal predictor-only dependency closure from the paper's v11
implementation.  Images are encoded separately by a frozen DINOv2 ViT-S/14;
the model here consumes the resulting 64 patch tokens per frame.  Parameter
names intentionally match the paper checkpoints exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

AttentionBackend = Literal["sdpa", "reference"]


@dataclass(frozen=True)
class PredictorConfig:
    """Architecture recorded in every DINO-WM checkpoint."""

    history_size: int = 20
    num_patches: int = 64
    visual_dim: int = 384
    action_dim: int = 3
    proprio_dim: int = 1
    action_emb_dim: int = 10
    proprio_emb_dim: int = 10
    depth: int = 6
    heads: int = 16
    mlp_dim: int = 2048
    dim_head: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.0
    attention_backend: AttentionBackend = "sdpa"

    def __post_init__(self) -> None:
        integer_fields = (
            "history_size",
            "num_patches",
            "visual_dim",
            "action_dim",
            "proprio_dim",
            "action_emb_dim",
            "proprio_emb_dim",
            "depth",
            "heads",
            "mlp_dim",
            "dim_head",
        )
        for name in integer_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0 or not 0.0 <= self.emb_dropout < 1.0:
            raise ValueError("dropout values must be in [0, 1)")
        if self.attention_backend not in {"sdpa", "reference"}:
            raise ValueError("attention_backend must be 'sdpa' or 'reference'")

    @property
    def observation_dim(self) -> int:
        return self.visual_dim + self.proprio_emb_dim

    @property
    def model_dim(self) -> int:
        return self.observation_dim + self.action_emb_dim

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> PredictorConfig:
        return cls(**values)  # type: ignore[arg-type]


def frame_causal_mask(
    num_frames: int,
    num_patches: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Mask in which every patch in frame ``t`` sees frames ``<= t``."""

    if num_frames <= 0 or num_patches <= 0:
        raise ValueError("num_frames and num_patches must be positive")
    frames = torch.arange(num_frames, device=device).repeat_interleave(num_patches)
    return frames[:, None] >= frames[None, :]


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    dropout: float,
    training: bool,
) -> torch.Tensor:
    scores = query @ key.transpose(-2, -1) * float(query.shape[-1]) ** -0.5
    while mask.ndim < scores.ndim:
        mask = mask.unsqueeze(0)
    probabilities = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    probabilities = F.dropout(probabilities, p=dropout, training=training and dropout > 0.0)
    return probabilities @ value


class TemporalEmbedding(nn.Module):
    """The upstream 1x1 temporal convolution."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.patch_embed = nn.Conv1d(input_dim, output_dim, kernel_size=1, stride=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError(f"expected [B,T,{self.input_dim}], got {tuple(values.shape)}")
        values = values.to(dtype=self.patch_embed.weight.dtype)
        return self.patch_embed(values.transpose(1, 2)).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class BlockCausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int,
        dropout: float,
        backend: AttentionBackend,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.backend = backend
        inner_dim = heads * dim_head
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, values: torch.Tensor, *, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = values.shape
        query, key, value = (
            item.reshape(batch, sequence, self.heads, self.dim_head).transpose(1, 2)
            for item in self.to_qkv(self.norm(values)).chunk(3, dim=-1)
        )
        mask = attention_mask[:sequence, :sequence]
        dropout = self.dropout if self.training else 0.0
        if self.backend == "sdpa":
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=dropout,
                is_causal=False,
            )
        else:
            attended = _reference_attention(
                query,
                key,
                value,
                mask,
                dropout=dropout,
                training=self.training,
            )
        return self.to_out(attended.transpose(1, 2).reshape(batch, sequence, -1))


class DinoWMPredictor(nn.Module):
    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.history_size * config.num_patches, config.model_dim)
        )
        self.embedding_dropout = nn.Dropout(config.emb_dropout)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        BlockCausalSelfAttention(
                            config.model_dim,
                            heads=config.heads,
                            dim_head=config.dim_head,
                            dropout=config.dropout,
                            backend=config.attention_backend,
                        ),
                        FeedForward(config.model_dim, config.mlp_dim, config.dropout),
                    ]
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.model_dim)
        self.register_buffer(
            "attention_mask",
            frame_causal_mask(config.history_size, config.num_patches),
            persistent=False,
        )

    def forward(self, combined_tokens: torch.Tensor) -> torch.Tensor:
        if combined_tokens.ndim != 4:
            raise ValueError("combined tokens must have shape [B,T,P,D]")
        batch, frames, patches, channels = combined_tokens.shape
        if frames > self.config.history_size:
            raise ValueError("input is longer than configured history")
        if (patches, channels) != (self.config.num_patches, self.config.model_dim):
            raise ValueError("combined token shape differs from predictor config")
        sequence = frames * patches
        values = combined_tokens.reshape(batch, sequence, channels)
        values = self.embedding_dropout(values + self.pos_embedding[:, :sequence].to(values.dtype))
        for attention, feed_forward in self.layers:
            values = values + attention(values, attention_mask=self.attention_mask)
            values = values + feed_forward(values)
        return self.norm(values).reshape(batch, frames, patches, channels)


@dataclass
class TeacherForcedOutput:
    loss: torch.Tensor
    visual_loss: torch.Tensor
    proprio_loss: torch.Tensor
    predicted_visual_tokens: torch.Tensor
    target_visual_tokens: torch.Tensor


class DinoWorldModel(nn.Module):
    """DINO patch-token dynamics model used for the paper baseline."""

    def __init__(self, config: PredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or PredictorConfig()
        self.action_encoder = TemporalEmbedding(self.config.action_dim, self.config.action_emb_dim)
        self.proprio_encoder = TemporalEmbedding(
            self.config.proprio_dim, self.config.proprio_emb_dim
        )
        self.predictor = DinoWMPredictor(self.config)

    def _proprio(self, tokens: torch.Tensor, proprio: torch.Tensor | None) -> torch.Tensor:
        if tokens.ndim != 4 or tuple(tokens.shape[2:]) != (
            self.config.num_patches,
            self.config.visual_dim,
        ):
            raise ValueError("visual tokens differ from the predictor config")
        if proprio is None:
            return torch.zeros(
                tokens.shape[0],
                tokens.shape[1],
                self.config.proprio_dim,
                device=tokens.device,
            )
        return proprio

    def encode_inputs(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        proprio = self._proprio(tokens, proprio)
        if actions.shape[:2] != tokens.shape[:2] or actions.shape[-1] != self.config.action_dim:
            raise ValueError("actions must have shape [B,T,action_dim]")
        proprio_embedding = self.proprio_encoder(proprio).unsqueeze(2)
        action_embedding = self.action_encoder(actions).unsqueeze(2)
        patches = tokens.shape[2]
        return torch.cat(
            (
                tokens.to(action_embedding.dtype),
                proprio_embedding.expand(-1, -1, patches, -1),
                action_embedding.expand(-1, -1, patches, -1),
            ),
            dim=-1,
        )

    def split_embeddings(
        self, combined: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual_end = self.config.visual_dim
        proprio_end = visual_end + self.config.proprio_emb_dim
        return (
            combined[..., :visual_end],
            combined[..., visual_end:proprio_end],
            combined[..., proprio_end:],
        )

    def teacher_forced_loss(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        proprio: torch.Tensor | None = None,
    ) -> TeacherForcedOutput:
        """Compute the exact H-to-H shifted one-step embedding objective."""

        history = self.config.history_size
        if tokens.shape[1] != history + 1:
            raise ValueError(f"training windows must contain exactly H+1={history + 1} frames")
        proprio = self._proprio(tokens, proprio)
        prediction = self.predictor(
            self.encode_inputs(tokens[:, :history], actions[:, :history], proprio[:, :history])
        )
        predicted_visual, predicted_proprio, _ = self.split_embeddings(prediction)
        target_visual = tokens[:, 1 : history + 1]
        target_proprio = self.proprio_encoder(proprio[:, 1 : history + 1])
        target_proprio = target_proprio.unsqueeze(2).expand(-1, -1, self.config.num_patches, -1)
        visual_loss = (predicted_visual.float() - target_visual.float()).square().mean()
        proprio_loss = (predicted_proprio.float() - target_proprio.float()).square().mean()
        loss = (
            visual_loss * self.config.visual_dim + proprio_loss * self.config.proprio_emb_dim
        ) / self.config.observation_dim
        return TeacherForcedOutput(
            loss=loss,
            visual_loss=visual_loss,
            proprio_loss=proprio_loss,
            predicted_visual_tokens=predicted_visual,
            target_visual_tokens=target_visual,
        )

    @torch.no_grad()
    def rollout_visual_tokens(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        proprio: torch.Tensor | None = None,
        *,
        horizon: int | None = None,
    ) -> torch.Tensor:
        history = self.config.history_size
        horizon = tokens.shape[1] - history if horizon is None else int(horizon)
        if horizon < 0 or actions.shape[1] < history + horizon:
            raise ValueError("insufficient frames for requested rollout")
        proprio = self._proprio(tokens, proprio)
        if proprio.shape[1] < history:
            raise ValueError("insufficient proprio frames")
        combined = self.encode_inputs(
            tokens[:, :history], actions[:, :history], proprio[:, :history]
        )
        visual_sequence = [tokens[:, :history]]
        for step in range(horizon):
            prediction = self.predictor(combined[:, -history:])[:, -1:]
            visual, _proprio, _action = self.split_embeddings(prediction)
            visual_sequence.append(visual)
            next_action = self.action_encoder(
                actions[:, history + step : history + step + 1]
            ).unsqueeze(2)
            prediction = torch.cat(
                (
                    prediction[..., : self.config.observation_dim],
                    next_action.expand(-1, -1, self.config.num_patches, -1),
                ),
                dim=-1,
            )
            combined = torch.cat((combined, prediction), dim=1)
        return torch.cat(visual_sequence, dim=1)


def build_dino_world_model(config: PredictorConfig | dict[str, object]) -> DinoWorldModel:
    if isinstance(config, dict):
        config = PredictorConfig.from_dict(config)
    return DinoWorldModel(config)


__all__ = [
    "DinoWorldModel",
    "PredictorConfig",
    "TeacherForcedOutput",
    "build_dino_world_model",
    "frame_causal_mask",
]
