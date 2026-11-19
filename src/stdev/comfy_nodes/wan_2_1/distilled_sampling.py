"""
Single step Wan 2.1 latent upscaling with the distilled students: raw VAE latents are normalized, padded to
the window grid, upscaled in one forward pass at sigma 1 (guidance is a model input) and denormalized.
"""

from dataclasses import dataclass
from typing import Callable

import torch

from stdev.comfy_nodes.wan_2_1.sampling import (
    normalization_statistics,
    pad_to_window_grid,
    target_size_of,
)
from stdev.wan_distill.student import WanStudentTransformer, upscale_single_step

NOISE_SCALE_SPATIAL = 1.0
NOISE_SCALE_TEMPORAL = 0.8


@dataclass(frozen=True)
class DistilledSettings:
    spatial: bool
    temporal: bool
    guidance: float
    hff: float | None
    seed: int


@torch.no_grad()
def upscale_latents_single_step(
    model: WanStudentTransformer,
    latents: torch.Tensor,
    settings: DistilledSettings,
    device: torch.device,
    on_sample: Callable[[], None],
) -> torch.Tensor:
    """
    args:
        latents: [B, 16, T, H, W], raw (unnormalized) Wan 2.1 VAE latents
    returns:
        latents: [B, 16, T_hr, H_hr, W_hr], raw Wan 2.1 VAE latents, dtype: torch.float32, on the CPU
    """
    mean, std = normalization_statistics(device)
    dtype = next(model.parameters()).dtype
    noise_scale = NOISE_SCALE_SPATIAL if settings.spatial else NOISE_SCALE_TEMPORAL
    hff = torch.full((1,), float("nan") if settings.hff is None else settings.hff, device=device)
    guidance = torch.full((1,), settings.guidance, device=device)
    target_size = target_size_of(tuple(latents.shape[2:]), settings.spatial, settings.temporal)
    results = []
    for index in range(latents.shape[0]):
        normalized = (latents[index : index + 1].to(device).float() - mean) / std
        padded = pad_to_window_grid(normalized, settings.spatial)
        padded_target = target_size_of(tuple(padded.shape[2:]), settings.spatial, settings.temporal)
        generator = torch.Generator().manual_seed(settings.seed + index)
        noise = torch.randn((1, padded.shape[1], *padded_target), generator=generator)
        upscaled = upscale_single_step(
            model, padded, hff, guidance, noise_scale * noise.to(device), dtype
        )
        upscaled = upscaled[:, :, : target_size[0], : target_size[1], : target_size[2]]
        results.append((upscaled * std + mean).cpu())
        on_sample()
    return torch.cat(results)
