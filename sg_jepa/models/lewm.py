"""Checkpoint-compatible LeWM and Semigroup-JEPA building blocks.

The module preserves the parameter names and tensor contracts of the final v10
paper implementation while removing its experiment-version namespace.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        cond_dim=None,
    ):
        super().__init__()
        cond_dim = input_dim if cond_dim is None else cond_dim
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(cond_dim, hidden_dim) if cond_dim != hidden_dim else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        )

    def forward(self, x, c):
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        return self.output_proj(self.norm(x))


class Embedder(nn.Module):
    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.float().permute(0, 2, 1)
        x = self.patch_embed(x).permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        cond_dim=None,
    ):
        super().__init__()
        self.output_dim = output_dim or input_dim
        self.cond_dim = input_dim if cond_dim is None else cond_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            self.output_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            cond_dim=self.cond_dim,
        )

    def forward(self, x, c):
        t = x.size(1)
        if c.shape[:2] != x.shape[:2]:
            raise ValueError(
                "Transformer predictor expected x and c with matching batch/time axes; "
                f"got x={tuple(x.shape)} c={tuple(c.shape)}"
            )
        x = self.dropout(x + self.pos_embedding[:, :t])
        return self.transformer(x, c)


class ActionConditionedGRULayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim,
        cond_dim,
        mlp_dim,
        dropout=0.0,
        conditioning="film_concat",
        residual_init=0.1,
    ):
        super().__init__()
        if conditioning not in {"concat", "film_concat"}:
            raise ValueError(
                f"Unknown GRU conditioning mode {conditioning!r}; "
                "expected 'concat' or 'film_concat'."
            )
        self.conditioning = conditioning
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
        self.film = None
        if conditioning == "film_concat":
            self.film = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, 2 * hidden_dim, bias=True),
            )
            nn.init.constant_(self.film[-1].weight, 0)
            nn.init.constant_(self.film[-1].bias, 0)
        self.input_mlp = nn.Sequential(
            nn.Linear(hidden_dim + cond_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x, c):
        h = self.norm(x)
        if self.film is not None:
            shift, scale = self.film(c).chunk(2, dim=-1)
            h = modulate(h, shift, scale)
        h = self.input_mlp(torch.cat([h, c], dim=-1))
        h, _ = self.gru(h)
        return x + self.residual_scale * self.dropout(h)


class ActionConditionedGRUPredictor(nn.Module):
    """Drop-in action-conditioned recurrent predictor for LeWM."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        input_dim,
        hidden_dim,
        output_dim=None,
        mlp_dim=None,
        dropout=0.0,
        emb_dropout=0.0,
        cond_dim=None,
        conditioning="film_concat",
        residual_init=0.1,
        heads=None,
        dim_head=None,
    ):
        super().__init__()
        del heads, dim_head
        if depth <= 0:
            raise ValueError(f"GRU predictor depth must be positive, got {depth}")
        if conditioning not in {"concat", "film_concat"}:
            raise ValueError(
                f"Unknown GRU conditioning mode {conditioning!r}; "
                "expected 'concat' or 'film_concat'."
            )
        self.output_dim = output_dim or input_dim
        self.cond_dim = input_dim if cond_dim is None else cond_dim
        self.conditioning = conditioning
        mlp_dim = int(mlp_dim or hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(self.cond_dim, hidden_dim) if self.cond_dim != hidden_dim else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                ActionConditionedGRULayer(
                    hidden_dim=hidden_dim,
                    cond_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    conditioning=conditioning,
                    residual_init=residual_init,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = (
            nn.Linear(hidden_dim, self.output_dim)
            if hidden_dim != self.output_dim
            else nn.Identity()
        )

    def forward(self, x, c):
        t = x.size(1)
        if c.shape[:2] != x.shape[:2]:
            raise ValueError(
                "GRU predictor expected x and c with matching batch/time axes; "
                f"got x={tuple(x.shape)} c={tuple(c.shape)}"
            )
        x = self.dropout(x + self.pos_embedding[:, :t])
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for layer in self.layers:
            x = layer(x, c)
        return self.output_proj(self.norm(x))


class SelectiveSSMBlock(nn.Module):
    """Causal Mamba/S6-style selective state-space sequence block.

    This is a compact local adaptation of the pure-PyTorch VideoMamba S6
    reference in ``external/stable-pretraining``. It keeps the implementation
    dependency-free while preserving the key selective-SSM ingredients:
    input-dependent delta/B/C parameters, stable negative state matrix, causal
    depthwise convolution, and a strictly left-to-right scan.
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_state <= 0:
            raise ValueError(f"d_state must be positive, got {d_state}")
        if d_conv <= 0:
            raise ValueError(f"d_conv must be positive, got {d_conv}")
        if expand <= 0:
            raise ValueError(f"expand must be positive, got {expand}")
        if dt_rank is None:
            dt_rank = math.ceil(d_model / 16)
        if dt_rank <= 0:
            raise ValueError(f"dt_rank must be positive, got {dt_rank}")

        d_inner = int(expand) * int(d_model)
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.d_inner = int(d_inner)
        self.dt_rank = int(dt_rank)

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=0,
            bias=True,
        )
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -(self.dt_rank**-0.5), self.dt_rank**-0.5)

        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"SSM block expects x with shape (B,T,D), got {tuple(x.shape)}")
        batch_size, seq_len, _ = x.shape

        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)

        u_conv = u.transpose(1, 2)
        u_conv = F.pad(u_conv, (self.d_conv - 1, 0))
        u = self.conv1d(u_conv).transpose(1, 2)
        u = F.silu(u)

        x_dbl = self.x_proj(u)
        delta, b_param, c_param = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))

        a = -torch.exp(self.A_log.float())
        delta_a = torch.exp(delta.unsqueeze(-1) * a)
        delta_b_u = delta.unsqueeze(-1) * b_param.unsqueeze(2) * u.unsqueeze(-1)

        state = u.new_zeros(batch_size, self.d_inner, self.d_state)
        outputs = []
        for idx in range(seq_len):
            state = delta_a[:, idx] * state + delta_b_u[:, idx]
            outputs.append(torch.einsum("bdn,bn->bd", state, c_param[:, idx].to(state.dtype)))
        y = torch.stack(outputs, dim=1).to(x.dtype)

        y = y + u * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class ActionConditionedSSMLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim,
        cond_dim,
        mlp_dim,
        dropout=0.0,
        conditioning="film_concat",
        residual_init=0.1,
        ssm_state_dim=16,
        ssm_conv_kernel=4,
        ssm_expand=2,
        ssm_dt_rank=None,
    ):
        super().__init__()
        if conditioning not in {"concat", "film_concat"}:
            raise ValueError(
                f"Unknown SSM conditioning mode {conditioning!r}; "
                "expected 'concat' or 'film_concat'."
            )
        self.conditioning = conditioning
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
        self.film = None
        if conditioning == "film_concat":
            self.film = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, 2 * hidden_dim, bias=True),
            )
            nn.init.constant_(self.film[-1].weight, 0)
            nn.init.constant_(self.film[-1].bias, 0)
        self.input_mlp = nn.Sequential(
            nn.Linear(hidden_dim + cond_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.ssm = SelectiveSSMBlock(
            hidden_dim,
            d_state=int(ssm_state_dim),
            d_conv=int(ssm_conv_kernel),
            expand=int(ssm_expand),
            dt_rank=ssm_dt_rank,
        )

    def forward(self, x, c):
        h = self.norm(x)
        if self.film is not None:
            shift, scale = self.film(c).chunk(2, dim=-1)
            h = modulate(h, shift, scale)
        h = self.input_mlp(torch.cat([h, c], dim=-1))
        h = self.ssm(h)
        return x + self.residual_scale * self.dropout(h)


class ActionConditionedSSMPredictor(nn.Module):
    """Drop-in action-conditioned causal SSM predictor for LeWM.

    The predictor keeps the same ``forward(x, c)`` contract as ``Predictor``:
    ``x[:, t]`` and ``c[:, t]`` produce the prediction aligned with
    ``z_{t+1}`` in the existing training loop. Action is concatenated into every
    SSM layer; ``film_concat`` additionally mirrors the current
    Transformer's AdaLN-style action modulation.
    """

    def __init__(
        self,
        *,
        num_frames,
        depth,
        input_dim,
        hidden_dim,
        output_dim=None,
        mlp_dim=None,
        dropout=0.0,
        emb_dropout=0.0,
        cond_dim=None,
        conditioning="film_concat",
        residual_init=0.1,
        heads=None,
        dim_head=None,
        ssm_state_dim=16,
        ssm_conv_kernel=4,
        ssm_expand=2,
        ssm_dt_rank=None,
    ):
        super().__init__()
        del heads, dim_head
        if depth <= 0:
            raise ValueError(f"SSM predictor depth must be positive, got {depth}")
        if conditioning not in {"concat", "film_concat"}:
            raise ValueError(
                f"Unknown SSM conditioning mode {conditioning!r}; "
                "expected 'concat' or 'film_concat'."
            )
        self.output_dim = output_dim or input_dim
        self.cond_dim = input_dim if cond_dim is None else cond_dim
        self.conditioning = conditioning
        mlp_dim = int(mlp_dim or hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(self.cond_dim, hidden_dim) if self.cond_dim != hidden_dim else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                ActionConditionedSSMLayer(
                    hidden_dim=hidden_dim,
                    cond_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    conditioning=conditioning,
                    residual_init=residual_init,
                    ssm_state_dim=ssm_state_dim,
                    ssm_conv_kernel=ssm_conv_kernel,
                    ssm_expand=ssm_expand,
                    ssm_dt_rank=ssm_dt_rank,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = (
            nn.Linear(hidden_dim, self.output_dim)
            if hidden_dim != self.output_dim
            else nn.Identity()
        )

    def forward(self, x, c):
        t = x.size(1)
        if c.shape[:2] != x.shape[:2]:
            raise ValueError(
                "SSM predictor expected x and c with matching batch/time axes; "
                f"got x={tuple(x.shape)} c={tuple(c.shape)}"
            )
        x = self.dropout(x + self.pos_embedding[:, :t])
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for layer in self.layers:
            x = layer(x, c)
        return self.output_proj(self.norm(x))


class LeWM(nn.Module):
    def __init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        pixels = info["pixels"].to(next(self.encoder.parameters()).dtype)
        batch_size = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        cls_raw = output.last_hidden_state[:, 0]
        if getattr(self, "store_cls_raw", False):
            info["cls_raw"] = rearrange(cls_raw, "(b t) d -> b t d", b=batch_size)
        emb = self.projector(cls_raw)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=batch_size)
        if getattr(self, "store_patch_tokens", False):
            patch_tokens = output.last_hidden_state[:, 1:]
            info["patch_tokens"] = rearrange(patch_tokens, "(b t) p d -> b t p d", b=batch_size)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb, act_emb):
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    def rollout(self, info, action_sequence, history_size: int = 3):
        assert "pixels" in info, "pixels not in info_dict"
        history = info["pixels"].size(2)
        batch_size, num_samples, horizon = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [history, horizon - history], dim=2)
        info["action"] = act_0
        n_steps = horizon - history
        if "emb" not in info:
            init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            init = self.encode(init)
            info["emb"] = init["emb"].detach().unsqueeze(1).expand(batch_size, num_samples, -1, -1)

        emb_init = rearrange(info["emb"], "b s ... -> (b s) ...")
        act_flat = rearrange(act_0, "b s ... -> (b s) ...")
        act_future_flat = rearrange(act_future, "b s ... -> (b s) ...")
        all_act_emb = self.action_encoder(torch.cat([act_flat, act_future_flat], dim=1))

        emb_list = list(emb_init.unbind(dim=1))
        for t in range(n_steps + 1):
            lo = max(0, history + t - history_size)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)
            act_trunc = all_act_emb[:, lo : history + t]
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        emb = torch.stack(emb_list, dim=1)
        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=batch_size, s=num_samples)
        return info

    def criterion(self, info_dict: dict):
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"][..., -1:, :].expand_as(pred_emb)
        return F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert "goal" in info_dict, "goal not in info_dict"
        if "goal_emb" not in info_dict:
            goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            goal["pixels"] = goal["goal"]
            for key in info_dict:
                if key.startswith("goal_"):
                    goal[key[len("goal_") :]] = goal.pop(key)
            goal.pop("action")
            info_dict["goal_emb"] = self.encode(goal)["emb"]
        return self.criterion(self.rollout(info_dict, action_candidates))


__all__ = [
    "ActionConditionedGRULayer",
    "ActionConditionedGRUPredictor",
    "ActionConditionedSSMLayer",
    "ActionConditionedSSMPredictor",
    "Embedder",
    "LeWM",
    "MLP",
    "Predictor",
    "SelectiveSSMBlock",
]
