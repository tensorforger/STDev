"""
H3-Image -> FLUX.2 latent translator (autoresearch_6 run_0006, with the hff condition of Steppe-Flux.2): the
Steppe-Flux.2 window transformer, whose source path upsamples the 16x H3 latent to the 8x FLUX.2 grid, and a normalized
residual flow over a fixed linear base (least squares 3x3 H3 -> FLUX.2 map). Initialized from the Steppe-Flux.2 base.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
from safetensors.torch import load_file

from stdev.flux_pretrain.model import (
    LATENT_CHANNELS,
    FluxUpscalerTransformer,
    ModelConfig,
    SamplerConfig,
    SourceEncoder,
    shift_sigmas,
)

H3_CHANNELS = 24


class LinearBase(nn.Module):
    """
    Fixed least squares estimate of the FLUX.2 latent from the H3 latent: 3x3 convolution (replicate padding) to the
    four FLUX.2 tokens of every H3 token, pixel shuffle. Buffers are saved with the model.
    """

    def __init__(self, fit_path: Path | None = None) -> None:
        super().__init__()
        kernel = 3
        self.padding = kernel // 2
        self.register_buffer("weight", torch.zeros(4 * LATENT_CHANNELS, H3_CHANNELS, kernel, kernel))
        self.register_buffer("bias", torch.zeros(4 * LATENT_CHANNELS))
        if fit_path is not None:
            fit = torch.load(fit_path)
            weight = fit["weight"]  # [24 * k * k + 1, 128]
            self.weight.copy_(weight[:-1].T.reshape(4 * LATENT_CHANNELS, H3_CHANNELS, kernel, kernel))
            self.bias.copy_(weight[-1])

    def forward(self, latents_source: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents_source: [B, 24, h, w], normalized H3 latents
        returns:
            latents_base: [B, 32, 2h, 2w], FLUX.2 latent estimate, dtype: torch.float32
        """
        padded = functional.pad(latents_source.float(), [self.padding] * 4, mode="replicate")
        return functional.pixel_shuffle(functional.conv2d(padded, self.weight.float(), self.bias.float()), 2)


class H3TranslatorTransformer(FluxUpscalerTransformer):
    """
    Velocity model of the residual flow r = (x_flux - base - residual_mean) / residual_std: stem on [noisy residual,
    linear base], H3 source path (residual blocks on the H3 grid, pixel shuffle), Steppe-Flux.2 transformer blocks,
    sigma + hff condition (hff droppable to a learned null embedding).
    """

    def __init__(self, config: ModelConfig, residual_mean: tuple[float, ...], fit_path: Path | None = None) -> None:
        super().__init__(config)
        self.source_encoder = SourceEncoder(config.source_encoder_width, config.width, config.source_encoder_blocks, H3_CHANNELS)
        self.linear_base = LinearBase(fit_path)
        self.register_buffer("residual_mean", torch.tensor(residual_mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("residual_std", torch.tensor(config.residual_std, dtype=torch.float32).view(1, -1, 1, 1))

    def to_residual(self, latents_target: torch.Tensor, latents_base: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents_target: [B, 32, H, W], normalized FLUX.2 latents
            latents_base: [B, 32, H, W], output of linear_base
        returns:
            residual: [B, 32, H, W], per channel ~unit std, dtype: torch.float32
        """
        return (latents_target.float() - latents_base - self.residual_mean) / self.residual_std

    def from_residual(self, residual: torch.Tensor, latents_base: torch.Tensor) -> torch.Tensor:
        """
        args:
            residual: [B, 32, H, W]
            latents_base: [B, 32, H, W]
        returns:
            latents: [B, 32, H, W], normalized FLUX.2 latents, dtype: torch.float32
        """
        return latents_base + self.residual_mean + self.residual_std * residual.float()

    def forward(
        self,
        noisy_residual: torch.Tensor,
        sigmas: torch.Tensor,
        latents_source: torch.Tensor,
        latents_base: torch.Tensor,
        hff: torch.Tensor,
        hff_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            noisy_residual: [B, 32, H, W]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_source: [B, 24, H / 2, W / 2], normalized H3 latents
            latents_base: [B, 32, H, W], output of linear_base
            hff: [B], values: [0...1], high frequency feature of the target
            hff_mask: [B], dtype: torch.bool, False = hff dropped
        returns:
            velocity: [B, 32, H, W] (noise - residual)
        """
        condition = self.conditioning(sigmas, hff, hff_mask)
        stacked = torch.cat([noisy_residual, latents_base.to(noisy_residual.dtype)], dim=1)  # [B, 64, H, W]
        features = self.input_conv(stacked) + self.source_encoder(latents_source.to(noisy_residual.dtype))
        tokens = features.permute(0, 2, 3, 1)  # [B, H, W, D]
        for index, block in enumerate(self.blocks):
            if self.training and index < self.checkpointed_blocks:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, condition, use_reentrant=False)
            else:
                tokens = block(tokens, condition)
        shift, scale = self.output_modulation(condition)[:, None, None, :].chunk(2, dim=-1)
        velocity = self.output(self.output_norm(tokens) * (1 + scale) + shift)  # [B, H, W, 32]
        return velocity.permute(0, 3, 1, 2)


def initialize_from_steppe(model: H3TranslatorTransformer, checkpoint_path: Path) -> list[str]:
    """
    Copies every Steppe-Flux.2 base tensor whose name and shape match. Returns the names of the model tensors that
    keep their fresh initialization.
    """
    steppe_state = load_file(str(checkpoint_path))
    model_state = model.state_dict()
    fresh = []
    for name, tensor in model_state.items():
        if name in steppe_state and steppe_state[name].shape == tensor.shape:
            model_state[name] = steppe_state[name].to(tensor.dtype)
        elif not name.startswith(("linear_base", "residual_")):
            fresh.append(name)
    model.load_state_dict(model_state)
    return fresh


@torch.no_grad()
def translate(
    model: H3TranslatorTransformer,
    latents_source: torch.Tensor,
    hff: torch.Tensor,
    sampler: SamplerConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Euler sampling of the residual flow.

    args:
        latents_source: [B, 24, h, w], normalized H3 latents, h and w multiples of 4
        hff: [B], values: [0...1], NaN = no hff (null embedding)
    returns:
        latents: [B, 32, 2h, 2w], normalized FLUX.2 latents, dtype: torch.float32
    """
    batch_size, _, height, width = latents_source.shape
    device = latents_source.device
    generator = torch.Generator(device=device).manual_seed(sampler.seed)
    samples = sampler.noise_scale * torch.randn((batch_size, LATENT_CHANNELS, 2 * height, 2 * width), generator=generator, device=device)
    latents_base = model.linear_base(latents_source)
    source = latents_source.to(dtype)
    base_input = latents_base.to(dtype)
    hff = hff.to(device).float()
    hff_mask = ~torch.isnan(hff)
    hff = torch.nan_to_num(hff, nan=0.0)
    sigmas = shift_sigmas(torch.linspace(1.0, 0.0, sampler.steps + 1, device=device), sampler.shift)
    for current_sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        velocity = model(samples.to(dtype), current_sigma.expand(batch_size), source, base_input, hff, hff_mask)
        samples = samples + (next_sigma - current_sigma) * velocity.float()
    return model.from_residual(samples, latents_base)
