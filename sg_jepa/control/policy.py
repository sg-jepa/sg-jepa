"""Official-style conditional 1D U-Net for low-dimensional actions."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import PolicyConfig


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("sinusoidal embedding dimension must be even and >= 4")
        self.dim = int(dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time = time.reshape(-1).float()
        half = self.dim // 2
        frequency = math.log(10_000.0) / (half - 1)
        frequency = torch.exp(torch.arange(half, device=time.device, dtype=time.dtype) * -frequency)
        angles = time[:, None] * frequency[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int) -> None:
        super().__init__()
        if out_channels % groups:
            raise ValueError(f"out_channels={out_channels} is not divisible by groups={groups}")
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.Mish(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ConditionalResidualBlock1D(nn.Module):
    """Two convolutional blocks with scale-and-bias (FiLM) conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int,
        groups: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            (
                Conv1dBlock(in_channels, out_channels, kernel_size, groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, groups),
            )
        )
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, value: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](value)
        scale_bias = self.cond_encoder(cond).reshape(cond.shape[0], 2, out.shape[1], 1)
        scale, bias = scale_bias.unbind(dim=1)
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual(value)


class Downsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # Match the official Diffusion Policy Downsample1d exactly.
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class Upsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class ResidualCrossAttention1D(nn.Module):
    """Let action-horizon features query the complete observation memory."""

    def __init__(
        self,
        channels: int,
        memory_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("cross-attention channels must be divisible by heads")
        self.query_norm = nn.LayerNorm(int(channels))
        self.memory_norm = nn.LayerNorm(int(memory_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=int(channels),
            num_heads=int(heads),
            dropout=float(dropout),
            kdim=int(memory_dim),
            vdim=int(memory_dim),
            batch_first=True,
        )
        # The new branch starts as an exact residual no-op. This preserves the
        # latent-GRU optimization geometry at initialization while allowing the
        # output projection to learn on the first update.
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, value: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or memory.ndim != 3:
            raise ValueError("cross-attention expects [B,C,T] and [B,S,D]")
        if value.shape[0] != memory.shape[0]:
            raise ValueError("action and observation batches differ")
        query = value.transpose(1, 2)
        attended, _weights = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        return (query + attended).transpose(1, 2)


class ObservationConditionEncoder(nn.Module):
    """Encode frozen observation histories into one global U-Net condition."""

    def __init__(
        self,
        horizon: int,
        embedding_dim: int,
        history_contract: str,
        *,
        observation_adapter: str = "identity",
        adapter_hidden_dim: int = 64,
        adapter_output_dim: int = 16,
        gru_hidden_dim: int = 256,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.0,
        cross_attention_memory_dim: int = 256,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.embedding_dim = int(embedding_dim)
        self.history_contract = str(history_contract)
        self.observation_adapter = str(observation_adapter)
        self.temporal_gru: nn.GRU | None = None
        self.temporal_norm: nn.Module = nn.Identity()
        self.memory_projection: nn.Linear | None = None
        self.memory_position: nn.Parameter | None = None
        self.memory_output_norm: nn.Module = nn.Identity()
        if self.observation_adapter == "identity":
            self.frame_adapter: nn.Module = nn.Identity()
            self.condition_embedding_dim = self.embedding_dim
        elif self.observation_adapter == "frame_mlp":
            self.frame_adapter = nn.Sequential(
                nn.LayerNorm(self.embedding_dim),
                nn.Linear(self.embedding_dim, int(adapter_hidden_dim)),
                nn.Mish(),
                nn.Linear(int(adapter_hidden_dim), int(adapter_output_dim)),
            )
            self.condition_embedding_dim = int(adapter_output_dim)
        elif self.observation_adapter in {
            "latent_gru",
            "latent_gru_delta",
            "latent_gru_crossattn",
        }:
            self.frame_adapter = nn.Identity()
            input_width = self.embedding_dim * (
                2 if self.observation_adapter == "latent_gru_delta" else 1
            )
            self.temporal_gru = nn.GRU(
                input_size=input_width,
                hidden_size=int(gru_hidden_dim),
                num_layers=int(gru_num_layers),
                batch_first=True,
                dropout=float(gru_dropout),
                bidirectional=False,
            )
            self.temporal_norm = nn.LayerNorm(int(gru_hidden_dim))
            self.condition_embedding_dim = int(gru_hidden_dim)
            if self.observation_adapter == "latent_gru_crossattn":
                memory_dim = int(cross_attention_memory_dim)
                self.memory_projection = nn.Linear(self.embedding_dim, memory_dim)
                self.memory_position = nn.Parameter(torch.empty(1, self.horizon, memory_dim))
                nn.init.normal_(self.memory_position, mean=0.0, std=0.02)
                self.memory_output_norm = nn.LayerNorm(memory_dim)
        else:
            raise ValueError(f"unsupported observation_adapter={self.observation_adapter!r}")

    def temporal_inputs(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return the exact GRU inputs, exposing the delta contract for tests."""

        if self.observation_adapter in {"latent_gru", "latent_gru_crossattn"}:
            return embeddings
        if self.observation_adapter == "latent_gru_delta":
            deltas = torch.zeros_like(embeddings)
            deltas[:, 1:] = embeddings[:, 1:] - embeddings[:, :-1]
            return torch.cat((embeddings, deltas), dim=-1)
        raise RuntimeError("temporal_inputs is only defined for latent-GRU adapters")

    def attention_memory(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return temporally positioned frozen-LeWM tokens for cross-attention."""

        if self.observation_adapter != "latent_gru_crossattn":
            raise RuntimeError("attention_memory is only defined for latent_gru_crossattn")
        expected = (self.horizon, self.embedding_dim)
        if embeddings.ndim != 3 or tuple(embeddings.shape[1:]) != expected:
            raise ValueError(
                "embeddings must have shape "
                f"[B,{self.horizon},{self.embedding_dim}], got {tuple(embeddings.shape)}"
            )
        assert self.memory_projection is not None
        assert self.memory_position is not None
        memory = self.memory_projection(embeddings) + self.memory_position
        return self.memory_output_norm(memory)

    def forward(
        self,
        embeddings: torch.Tensor,
        gravity: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (self.horizon, self.embedding_dim)
        if embeddings.ndim != 3 or tuple(embeddings.shape[1:]) != expected:
            raise ValueError(
                "embeddings must have shape "
                f"[B,{self.horizon},{self.embedding_dim}], got {tuple(embeddings.shape)}"
            )
        if gravity.ndim == 1:
            gravity = gravity[:, None]
        if gravity.ndim != 2 or gravity.shape != (embeddings.shape[0], 1):
            raise ValueError(f"gravity must have shape [B,1], got {tuple(gravity.shape)}")
        if self.temporal_gru is not None:
            if history_mask is not None:
                raise ValueError("latent-GRU adapters use repeat-frame-zero histories")
            _outputs, hidden = self.temporal_gru(self.temporal_inputs(embeddings))
            encoded = self.temporal_norm(hidden[-1])
            return torch.cat((encoded, gravity.to(encoded.dtype)), dim=-1)

        adapted = self.frame_adapter(embeddings)
        expected_adapted = (
            embeddings.shape[0],
            self.horizon,
            self.condition_embedding_dim,
        )
        if tuple(adapted.shape) != expected_adapted:
            raise RuntimeError(
                "observation adapter returned shape "
                f"{tuple(adapted.shape)}, expected {expected_adapted}"
            )
        if self.history_contract == "masked_zero_flatten_v3":
            if history_mask is None:
                raise ValueError("masked_zero_flatten_v3 requires history_mask")
            if tuple(history_mask.shape) != (embeddings.shape[0], self.horizon):
                raise ValueError(
                    f"history_mask must have shape [B,{self.horizon}], "
                    f"got {tuple(history_mask.shape)}"
                )
            mask = history_mask.to(device=embeddings.device, dtype=torch.bool)
            transitions = mask[:, 1:].to(torch.int8) - mask[:, :-1].to(torch.int8)
            if bool((transitions < 0).any()):
                raise ValueError("history_mask valid entries must form a suffix")
            masked = adapted * mask[..., None].to(adapted.dtype)
            flattened = masked.flatten(start_dim=1)
            return torch.cat(
                (
                    flattened,
                    gravity.to(flattened.dtype),
                    mask.to(flattened.dtype),
                ),
                dim=-1,
            )
        if history_mask is not None:
            raise ValueError("legacy_repeat_masked_v1 does not accept history_mask")
        flattened = adapted.flatten(start_dim=1)
        return torch.cat((flattened, gravity.to(flattened.dtype)), dim=-1)


class ConditionalUnet1D(nn.Module):
    """Official-style low-dimensional Diffusion Policy U-Net.

    Public inputs are noisy action trajectories ``[B,16,5]``, integer DDPM
    timesteps, frozen LeWorldModel embeddings ``[B,20,256]``, and normalized
    scalar gravity.  All temporal convolutions operate on the action horizon.
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyConfig()
        cfg = self.config
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(cfg.diffusion_step_embed_dim),
            nn.Linear(cfg.diffusion_step_embed_dim, cfg.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(cfg.diffusion_step_embed_dim * 4, cfg.diffusion_step_embed_dim),
        )
        self.observation_encoder = ObservationConditionEncoder(
            cfg.observation_horizon,
            cfg.embedding_dim,
            "legacy_repeat_masked_v1",
            observation_adapter=cfg.observation_adapter,
            adapter_hidden_dim=cfg.adapter_hidden_dim,
            adapter_output_dim=cfg.adapter_output_dim,
            gru_hidden_dim=cfg.gru_hidden_dim,
            gru_num_layers=cfg.gru_num_layers,
            gru_dropout=cfg.gru_dropout,
            cross_attention_memory_dim=cfg.cross_attention_memory_dim,
        )
        # Every adapter yields one global observation condition. The legacy
        # variants flatten histories; latent-GRU variants use their final
        # causal hidden state. All append one normalized gravity scalar.
        observation_condition_dim = cfg.observation_condition_dim
        cond_dim = cfg.diffusion_step_embed_dim + observation_condition_dim

        all_dims = (cfg.action_dim,) + cfg.down_dims
        in_out = list(zip(all_dims[:-1], all_dims[1:], strict=True))
        self.use_cross_attention = cfg.observation_adapter == "latent_gru_crossattn"
        self.down_modules = nn.ModuleList()
        for level, (in_channels, out_channels) in enumerate(in_out):
            is_last = level == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    (
                        ConditionalResidualBlock1D(
                            in_channels,
                            out_channels,
                            cond_dim,
                            cfg.kernel_size,
                            cfg.n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            out_channels,
                            out_channels,
                            cond_dim,
                            cfg.kernel_size,
                            cfg.n_groups,
                        ),
                        nn.Identity() if is_last else Downsample1D(out_channels),
                    )
                )
            )

        self.down_cross_attention = nn.ModuleList()
        if self.use_cross_attention:
            for channels in cfg.down_dims[:-1]:
                self.down_cross_attention.append(
                    ResidualCrossAttention1D(
                        channels,
                        cfg.cross_attention_memory_dim,
                        cfg.cross_attention_heads,
                        cfg.cross_attention_dropout,
                    )
                )

        middle_channels = cfg.down_dims[-1]
        self.mid_modules = nn.ModuleList(
            (
                ConditionalResidualBlock1D(
                    middle_channels,
                    middle_channels,
                    cond_dim,
                    cfg.kernel_size,
                    cfg.n_groups,
                ),
                ConditionalResidualBlock1D(
                    middle_channels,
                    middle_channels,
                    cond_dim,
                    cfg.kernel_size,
                    cfg.n_groups,
                ),
            )
        )
        self.mid_cross_attention: ResidualCrossAttention1D | None = None
        if self.use_cross_attention:
            self.mid_cross_attention = ResidualCrossAttention1D(
                middle_channels,
                cfg.cross_attention_memory_dim,
                cfg.cross_attention_heads,
                cfg.cross_attention_dropout,
            )

        self.up_modules = nn.ModuleList()
        self.up_cross_attention = nn.ModuleList()
        for in_channels, out_channels in reversed(in_out[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    (
                        ConditionalResidualBlock1D(
                            out_channels * 2,
                            in_channels,
                            cond_dim,
                            cfg.kernel_size,
                            cfg.n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            in_channels,
                            in_channels,
                            cond_dim,
                            cfg.kernel_size,
                            cfg.n_groups,
                        ),
                        Upsample1D(in_channels),
                    )
                )
            )
            if self.use_cross_attention:
                self.up_cross_attention.append(
                    ResidualCrossAttention1D(
                        in_channels,
                        cfg.cross_attention_memory_dim,
                        cfg.cross_attention_heads,
                        cfg.cross_attention_dropout,
                    )
                )

        first_width = cfg.down_dims[0]
        self.final_conv = nn.Sequential(
            Conv1dBlock(first_width, first_width, cfg.kernel_size, cfg.n_groups),
            nn.Conv1d(first_width, cfg.action_dim, kernel_size=1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        embeddings: torch.Tensor,
        gravity: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        if sample.ndim != 3 or tuple(sample.shape[1:]) != (cfg.action_horizon, cfg.action_dim):
            raise ValueError(
                f"sample must have shape [B,{cfg.action_horizon},{cfg.action_dim}], "
                f"got {tuple(sample.shape)}"
            )
        if not torch.is_tensor(timestep):
            timestep = torch.as_tensor(timestep, device=sample.device, dtype=torch.long)
        timestep = timestep.to(device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(sample.shape[0])
        if timestep.ndim != 1 or timestep.shape[0] != sample.shape[0]:
            raise ValueError(f"timestep must be scalar or [B], got {tuple(timestep.shape)}")

        time_cond = self.time_encoder(timestep)
        obs_cond = self.observation_encoder(embeddings, gravity)
        observation_memory = (
            self.observation_encoder.attention_memory(embeddings)
            if self.use_cross_attention
            else None
        )
        cond = torch.cat((time_cond.to(obs_cond.dtype), obs_cond), dim=-1)
        value = sample.transpose(1, 2)
        skips: list[torch.Tensor] = []
        for level, (block1, block2, downsample) in enumerate(self.down_modules):
            value = block1(value, cond)
            value = block2(value, cond)
            if level < len(self.down_cross_attention):
                assert observation_memory is not None
                value = self.down_cross_attention[level](value, observation_memory)
            skips.append(value)
            value = downsample(value)
        for block in self.mid_modules:
            value = block(value, cond)
        if self.mid_cross_attention is not None:
            assert observation_memory is not None
            value = self.mid_cross_attention(value, observation_memory)
        for level, (block1, block2, upsample) in enumerate(self.up_modules):
            skip = skips.pop()
            if value.shape[-1] != skip.shape[-1]:
                raise RuntimeError(
                    f"U-Net skip temporal mismatch: {value.shape[-1]} != {skip.shape[-1]}"
                )
            value = torch.cat((value, skip), dim=1)
            value = block1(value, cond)
            value = block2(value, cond)
            if level < len(self.up_cross_attention):
                assert observation_memory is not None
                value = self.up_cross_attention[level](value, observation_memory)
            value = upsample(value)
        result = self.final_conv(value).transpose(1, 2)
        if result.shape != sample.shape:
            raise RuntimeError(
                f"U-Net changed sample shape {tuple(sample.shape)} -> {tuple(result.shape)}"
            )
        return result


def build_policy_model(config: PolicyConfig) -> nn.Module:
    """Instantiate the baseline conditional diffusion U-Net."""

    return ConditionalUnet1D(config)
