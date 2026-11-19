"""
Wan 2.1 latent upscaling for ComfyUI: raw VAE latents are normalized, padded to the window grid, upscaled by
the residual flow sampler with classifier free guidance on the high frequency feature, and denormalized.
"""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as functional

from stdev.wan_pretrain.model import WanUpscalerTransformer, from_residual, shift_sigmas

LATENTS_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
LATENTS_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.916,
)
WINDOW_SIZE = 8


@dataclass(frozen=True)
class UpscaleSettings:
    spatial: bool
    temporal: bool
    steps: int
    cfg: float
    hff: float
    negative_hff: float | None
    shift: float
    noise_scale: float
    seed: int


def normalization_statistics(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    returns:
        mean, std: [1, 16, 1, 1, 1], dtype: torch.float32
    """
    mean = torch.tensor(LATENTS_MEAN, device=device)[None, :, None, None, None]
    std = torch.tensor(LATENTS_STD, device=device)[None, :, None, None, None]
    return mean, std


def target_size_of(
    source_size: tuple[int, int, int], spatial: bool, temporal: bool
) -> tuple[int, int, int]:
    frames, height, width = source_size
    if temporal:
        frames = 1 + 2 * (frames - 1)
    if spatial:
        height, width = 2 * height, 2 * width
    return frames, height, width


def pad_to_window_grid(latents: torch.Tensor, spatial: bool) -> torch.Tensor:
    """
    Replicate padding (bottom, right) so that the target latent grid is a multiple of the window size.

    args:
        latents: [B, C, T, H, W]
    returns:
        latents: [B, C, T, H_padded, W_padded]
    """
    multiple = WINDOW_SIZE // 2 if spatial else WINDOW_SIZE
    pad_height = -latents.shape[3] % multiple
    pad_width = -latents.shape[4] % multiple
    if pad_height == 0 and pad_width == 0:
        return latents
    return functional.pad(latents, (0, pad_width, 0, pad_height, 0, 0), mode="replicate")


@torch.no_grad()
def sample_residual(
    model: WanUpscalerTransformer,
    latents_source: torch.Tensor,
    target_size: tuple[int, int, int],
    settings: UpscaleSettings,
    seed: int,
    on_step: Callable[[], None],
) -> torch.Tensor:
    """
    Euler sampling of the residual flow for one sample; guidance: negative + cfg * (positive - negative),
    the negative branch uses a fixed hff or the null embedding.

    args:
        latents_source: [1, C, T_lr, H_lr, W_lr], normalized
    returns:
        latents: [1, C, T_hr, H_hr, W_hr], normalized, dtype: torch.float32
    """
    device = latents_source.device
    dtype = next(model.parameters()).dtype
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((1, latents_source.shape[1], *target_size), generator=generator)
    samples = settings.noise_scale * noise.to(device)
    source = latents_source.to(dtype)
    sigmas = shift_sigmas(
        torch.linspace(1.0, 0.0, settings.steps + 1, device=device), settings.shift
    )
    positive_hff = torch.full((1,), settings.hff, device=device)
    positive_mask = torch.ones(1, dtype=torch.bool, device=device)
    use_guidance = settings.cfg != 1.0
    negative_value = 0.0 if settings.negative_hff is None else settings.negative_hff
    negative_hff = torch.full((1,), negative_value, device=device)
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
    latents = from_residual(samples, latents_source)
    if not settings.spatial:
        latents[:, :, 0] = latents_source[:, :, 0].float()
    return latents


@torch.no_grad()
def upscale_latents(
    model: WanUpscalerTransformer,
    latents: torch.Tensor,
    settings: UpscaleSettings,
    device: torch.device,
    on_step: Callable[[], None],
) -> torch.Tensor:
    """
    args:
        latents: [B, 16, T, H, W], raw (unnormalized) Wan 2.1 VAE latents
    returns:
        latents: [B, 16, T_hr, H_hr, W_hr], raw Wan 2.1 VAE latents, dtype: torch.float32, on the CPU
    """
    mean, std = normalization_statistics(device)
    target_size = target_size_of(tuple(latents.shape[2:]), settings.spatial, settings.temporal)
    results = []
    for index in range(latents.shape[0]):
        normalized = (latents[index : index + 1].to(device).float() - mean) / std
        padded = pad_to_window_grid(normalized, settings.spatial)
        padded_target = target_size_of(tuple(padded.shape[2:]), settings.spatial, settings.temporal)
        upscaled = sample_residual(
            model, padded, padded_target, settings, settings.seed + index, on_step
        )
        upscaled = upscaled[:, :, : target_size[0], : target_size[1], : target_size[2]]
        results.append((upscaled * std + mean).cpu())
    return torch.cat(results)
