"""
FLUX.2 latent upscaling for ComfyUI: ComfyUI latents (2x2 patchified, batch norm normalized) are unpatchified to the
32 channel grid the upscaler works on, padded to the window grid, upscaled by the residual flow sampler with
classifier free guidance on the high frequency feature, and patchified back.
"""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as functional

from stdev.flux_pretrain.model import (
    LATENT_CHANNELS,
    WINDOW_SIZE,
    FluxUpscalerTransformer,
    from_residual,
    shift_sigmas,
)

PATCH_SIZE = 2


@dataclass(frozen=True)
class UpscaleSettings:
    steps: int
    cfg: float
    hff: float
    negative_hff: float | None
    shift: float
    noise_scale: float
    seed: int


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents: [B, 4C, H, W], channel order (C, 2, 2)
    returns:
        latents: [B, C, 2H, 2W]
    """
    batch_size, channels, height, width = latents.shape
    latents = latents.reshape(batch_size, channels // 4, PATCH_SIZE, PATCH_SIZE, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, channels // 4, height * PATCH_SIZE, width * PATCH_SIZE)


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents: [B, C, H, W]
    returns:
        latents: [B, 4C, H/2, W/2], channel order (C, 2, 2)
    """
    batch_size, channels, height, width = latents.shape
    latents = latents.reshape(batch_size, channels, height // 2, PATCH_SIZE, width // 2, PATCH_SIZE)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch_size, channels * 4, height // 2, width // 2)


def pad_to_window_grid(latents: torch.Tensor) -> torch.Tensor:
    """
    Replicate padding (bottom, right) so that the upscaled latent grid is a multiple of the window size.

    args:
        latents: [B, C, H, W]
    returns:
        latents: [B, C, H_padded, W_padded]
    """
    multiple = WINDOW_SIZE // 2
    pad_height = -latents.shape[2] % multiple
    pad_width = -latents.shape[3] % multiple
    if pad_height == 0 and pad_width == 0:
        return latents
    return functional.pad(latents, (0, pad_width, 0, pad_height), mode="replicate")


@torch.no_grad()
def sample_residual(
    model: FluxUpscalerTransformer,
    latents_source: torch.Tensor,
    settings: UpscaleSettings,
    seed: int,
    on_step: Callable[[], None],
) -> torch.Tensor:
    """
    Euler sampling of the residual flow for one sample; guidance: negative + cfg * (positive - negative),
    the negative branch uses a fixed hff or the null embedding.

    args:
        latents_source: [1, 32, h, w], normalized
    returns:
        latents: [1, 32, 2h, 2w], normalized, dtype: torch.float32
    """
    device = latents_source.device
    dtype = next(model.parameters()).dtype
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(
        (1, LATENT_CHANNELS, 2 * latents_source.shape[2], 2 * latents_source.shape[3]),
        generator=generator,
    )
    samples = settings.noise_scale * noise.to(device)
    source = latents_source.to(dtype)
    sigmas = shift_sigmas(
        torch.linspace(1.0, 0.0, settings.steps + 1, device=device), settings.shift
    )
    positive_hff = torch.full((1,), settings.hff, device=device)
    positive_mask = torch.ones(1, dtype=torch.bool, device=device)
    use_guidance = settings.cfg != 1.0
    negative_hff = torch.full(
        (1,), 0.0 if settings.negative_hff is None else settings.negative_hff, device=device
    )
    negative_mask = torch.full(
        (1,), settings.negative_hff is not None, dtype=torch.bool, device=device
    )
    for current_sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        sigma_batch = current_sigma.expand(1)
        velocity = model(
            samples.to(dtype), sigma_batch, source, positive_hff, positive_mask
        ).float()
        if use_guidance:
            negative_velocity = model(
                samples.to(dtype), sigma_batch, source, negative_hff, negative_mask
            ).float()
            velocity = negative_velocity + settings.cfg * (velocity - negative_velocity)
        samples = samples + (next_sigma - current_sigma) * velocity
        on_step()
    return from_residual(samples, latents_source, model.config.residual_std)


@torch.no_grad()
def upscale_latents(
    model: FluxUpscalerTransformer,
    latents: torch.Tensor,
    settings: UpscaleSettings,
    device: torch.device,
    on_step: Callable[[], None],
) -> torch.Tensor:
    """
    args:
        latents: [B, 128, h, w], ComfyUI FLUX.2 latents
    returns:
        latents: [B, 128, 2h, 2w], ComfyUI FLUX.2 latents, dtype: torch.float32, on the CPU
    """
    results = []
    for index in range(latents.shape[0]):
        source = unpatchify_latents(
            latents[index : index + 1].to(device).float()
        )  # [1, 32, 2h, 2w]
        height, width = source.shape[2:]
        upscaled = sample_residual(
            model, pad_to_window_grid(source), settings, settings.seed + index, on_step
        )
        upscaled = upscaled[:, :, : 2 * height, : 2 * width]
        results.append(patchify_latents(upscaled).cpu())
    return torch.cat(results)
