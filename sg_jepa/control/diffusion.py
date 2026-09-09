"""Cosine-schedule DDPM objective and sampler for action trajectories."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol-Dhariwal squared-cosine schedule, clipped like Diffusers DDPM."""

    if timesteps <= 1:
        raise ValueError("timesteps must be greater than one")
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1.0e-4, 0.999).float()


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    selected = values.gather(0, timesteps)
    return selected.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion1D(nn.Module):
    """100-step epsilon-prediction DDPM with a cosine variance schedule."""

    def __init__(self, timesteps: int = 100, *, clip_sample: bool = True) -> None:
        super().__init__()
        self.timesteps = int(timesteps)
        self.clip_sample = bool(clip_sample)
        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1.0e-20))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod),
        )

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if clean.shape != noise.shape:
            raise ValueError("clean and noise shapes differ")
        return (
            _extract(self.sqrt_alpha_cumprod, timesteps, clean.shape) * clean
            + _extract(self.sqrt_one_minus_alpha_cumprod, timesteps, clean.shape) * noise
        )

    def training_loss(
        self,
        model: nn.Module,
        clean_actions: torch.Tensor,
        embeddings: torch.Tensor,
        gravity: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        model_kwargs: dict[str, Any] | None = None,
        horizon_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = clean_actions.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.timesteps,
                (batch,),
                device=clean_actions.device,
                dtype=torch.long,
            )
        if noise is None:
            noise = torch.randn_like(clean_actions)
        noisy = self.add_noise(clean_actions, noise, timesteps)
        predicted_noise = model(
            noisy,
            timesteps,
            embeddings,
            gravity,
            **(model_kwargs or {}),
        )
        error = F.mse_loss(predicted_noise.float(), noise.float(), reduction="none")
        if horizon_weights is not None:
            if horizon_weights.ndim != 1 or horizon_weights.shape[0] != clean_actions.shape[1]:
                raise ValueError("horizon_weights must have shape [action_horizon]")
            error = error * horizon_weights.to(error)[None, :, None]
        return error.mean()

    def _step(
        self,
        model: nn.Module,
        sample: torch.Tensor,
        timestep: int,
        embeddings: torch.Tensor,
        gravity: torch.Tensor,
        *,
        generator: torch.Generator | None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        batch = sample.shape[0]
        t = torch.full((batch,), timestep, device=sample.device, dtype=torch.long)
        predicted_noise = model(
            sample,
            t,
            embeddings,
            gravity,
            **(model_kwargs or {}),
        )
        alpha_bar = _extract(self.alpha_cumprod, t, sample.shape)
        predicted_clean = (sample - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(
            alpha_bar
        )
        if self.clip_sample:
            predicted_clean = predicted_clean.clamp(-1.0, 1.0)
        mean = (
            _extract(self.posterior_mean_coef1, t, sample.shape) * predicted_clean
            + _extract(self.posterior_mean_coef2, t, sample.shape) * sample
        )
        if timestep == 0:
            return mean
        noise = torch.randn(
            sample.shape,
            dtype=sample.dtype,
            device=sample.device,
            generator=generator,
        )
        variance = _extract(self.posterior_variance, t, sample.shape)
        return mean + torch.sqrt(variance) * noise

    def _ddim_step(
        self,
        model: nn.Module,
        sample: torch.Tensor,
        timestep: int,
        previous_timestep: int,
        embeddings: torch.Tensor,
        gravity: torch.Tensor,
        *,
        eta: float,
        generator: torch.Generator | None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """One generalized DDIM step; ``eta=0`` is strictly deterministic."""

        batch = sample.shape[0]
        t = torch.full((batch,), timestep, device=sample.device, dtype=torch.long)
        predicted_noise = model(
            sample,
            t,
            embeddings,
            gravity,
            **(model_kwargs or {}),
        )
        alpha_t = self.alpha_cumprod[timestep].to(device=sample.device, dtype=sample.dtype)
        alpha_previous = (
            self.alpha_cumprod[previous_timestep].to(device=sample.device, dtype=sample.dtype)
            if previous_timestep >= 0
            else torch.ones((), device=sample.device, dtype=sample.dtype)
        )
        predicted_clean = (sample - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(
            alpha_t
        )
        if self.clip_sample:
            predicted_clean = predicted_clean.clamp(-1.0, 1.0)
        sigma = float(eta) * torch.sqrt(
            torch.clamp(
                (1.0 - alpha_previous) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_previous),
                min=0.0,
            )
        )
        direction = (
            torch.sqrt(torch.clamp(1.0 - alpha_previous - sigma.square(), min=0.0))
            * predicted_noise
        )
        previous = torch.sqrt(alpha_previous) * predicted_clean + direction
        if previous_timestep >= 0 and eta > 0.0:
            previous = previous + sigma * torch.randn(
                sample.shape,
                dtype=sample.dtype,
                device=sample.device,
                generator=generator,
            )
        return previous

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        embeddings: torch.Tensor,
        gravity: torch.Tensor,
        *,
        action_horizon: int,
        action_dim: int,
        generator: torch.Generator | None = None,
        callback: Callable[[int, torch.Tensor], None] | None = None,
        sampler: str = "ddpm",
        num_inference_steps: int | None = None,
        eta: float = 0.0,
        model_kwargs: dict[str, Any] | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sampler = str(sampler).lower()
        if sampler not in {"ddpm", "ddim"}:
            raise ValueError("sampler must be 'ddpm' or 'ddim'")
        inference_steps = (
            self.timesteps if num_inference_steps is None else int(num_inference_steps)
        )
        if not 1 <= inference_steps <= self.timesteps:
            raise ValueError("num_inference_steps must be in [1, diffusion timesteps]")
        if sampler == "ddpm" and inference_steps != self.timesteps:
            raise ValueError("reduced-step inference requires sampler='ddim'")
        if eta < 0:
            raise ValueError("DDIM eta cannot be negative")
        executed_timesteps: list[int] = []
        shape = (embeddings.shape[0], int(action_horizon), int(action_dim))
        if initial_noise is None:
            sample = torch.randn(
                shape,
                device=embeddings.device,
                dtype=embeddings.dtype,
                generator=generator,
            )
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(f"initial_noise must have shape {shape}")
            sample = initial_noise.to(device=embeddings.device, dtype=embeddings.dtype)
        if sampler == "ddpm":
            for timestep in reversed(range(self.timesteps)):
                sample = self._step(
                    model,
                    sample,
                    timestep,
                    embeddings,
                    gravity,
                    generator=generator,
                    model_kwargs=model_kwargs,
                )
                executed_timesteps.append(int(timestep))
                if callback is not None:
                    callback(timestep, sample)
        else:
            schedule = (
                torch.linspace(
                    self.timesteps - 1,
                    0,
                    inference_steps,
                    dtype=torch.float64,
                )
                .round()
                .to(torch.long)
            )
            # inference_steps <= timesteps guarantees strict descent, but fail
            # loudly if a future schedule change violates that invariant.
            if schedule.unique().numel() != inference_steps:
                raise RuntimeError("DDIM timestep schedule contains duplicates")
            steps = [int(value) for value in schedule.tolist()]
            for index, timestep in enumerate(steps):
                previous_timestep = steps[index + 1] if index + 1 < len(steps) else -1
                sample = self._ddim_step(
                    model,
                    sample,
                    timestep,
                    previous_timestep,
                    embeddings,
                    gravity,
                    eta=float(eta),
                    generator=generator,
                    model_kwargs=model_kwargs,
                )
                executed_timesteps.append(int(timestep))
                if callback is not None:
                    callback(timestep, sample)
        # Expose the executed schedule for callers and focused sampler tests.
        # This diagnostic state is deliberately absent from checkpoints.
        self.last_sample_timesteps = tuple(executed_timesteps)
        return sample
