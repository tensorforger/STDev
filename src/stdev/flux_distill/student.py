"""
Single step student of the FLUX.2 latent upscaler: the pretrained window transformer with a third scalar condition,
the classifier free guidance scale the teacher used (negative = null hff). Its embedding is added to the sigma and
hff embeddings before the RMS norm; the last layer is zero initialized, so a student initialized from the teacher
computes exactly the teacher velocity. The student always runs at sigma 1: residual = noise - velocity.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional

from stdev.flux_pretrain.model import (
    HFF_SCALE,
    SIGMA_SCALE,
    ConditioningEncoder,
    FluxUpscalerTransformer,
    ModelConfig,
    from_residual,
    sinusoidal_embedding,
)

GUIDANCE_SCALE = 500.0
STUDENT_SIGMA = 1.0


class GuidedConditioningEncoder(ConditioningEncoder):
    """
    Teacher conditioning plus a zero initialized MLP embedding of (guidance - 1).
    """

    def __init__(self, condition_dim: int) -> None:
        super().__init__(condition_dim)
        self.guidance_projection = self._make_projection(condition_dim)
        nn.init.zeros_(self.guidance_projection[2].weight)
        nn.init.zeros_(self.guidance_projection[2].bias)

    def forward(self, sigmas: torch.Tensor, hff: torch.Tensor, hff_mask: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        """
        args:
            sigmas: [B], values: [0...1]
            hff: [B], values: [0...1]
            hff_mask: [B], dtype: torch.bool, False = null hff embedding
            guidance: [B], >= 1, 1 = no guidance
        returns:
            condition: [B, condition_dim]
        """
        weight_dtype = self.sigma_projection[0].weight.dtype
        sigma_embedding = self.sigma_projection(sinusoidal_embedding(sigmas, SIGMA_SCALE).to(weight_dtype))
        hff_embedding = self.hff_projection(sinusoidal_embedding(hff, HFF_SCALE).to(weight_dtype))
        hff_embedding = torch.where(hff_mask[:, None], hff_embedding, self.null_hff.to(hff_embedding.dtype)[None])
        guidance_embedding = self.guidance_projection(sinusoidal_embedding(guidance - 1.0, GUIDANCE_SCALE).to(weight_dtype))
        condition = sigma_embedding + hff_embedding + guidance_embedding
        return functional.rms_norm(condition, (condition.shape[-1],))


class FluxStudentTransformer(FluxUpscalerTransformer):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.conditioning = GuidedConditioningEncoder(config.condition_dim)

    def forward(
        self,
        noise: torch.Tensor,
        latents_source: torch.Tensor,
        hff: torch.Tensor,
        hff_mask: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            noise: [B, C, H, W], unit gaussian (residual space, sigma 1)
            latents_source: [B, C, H / 2, W / 2], normalized FLUX.2 latents
            hff: [B], values: [0...1]
            hff_mask: [B], dtype: torch.bool, False = null hff
            guidance: [B], >= 1
        returns:
            velocity: [B, C, H, W] (noise - residual)
        """
        sigmas = torch.full_like(hff, STUDENT_SIGMA)
        return self.forward_with_condition(noise, self.conditioning(sigmas, hff, hff_mask, guidance), latents_source)


def model_config_of(checkpoint: dict) -> ModelConfig:
    values = dict(checkpoint["model_config"])
    return ModelConfig(**{**values, "residual_std": tuple(values["residual_std"])})


def teacher_state(checkpoint: dict, weights: str) -> dict[str, torch.Tensor]:
    """
    returns:
        state dict of the pretrained model with the raw or EMA parameters
    """
    state = dict(checkpoint["model"])
    if weights == "ema":
        names = [name for name, _ in FluxUpscalerTransformer(model_config_of(checkpoint)).named_parameters()]
        state.update(dict(zip(names, checkpoint["ema"], strict=True)))
    return state


def load_teacher_into_student(student: FluxStudentTransformer, checkpoint_path: Path, weights: str) -> None:
    """
    Teacher weights into every student parameter except the guidance projection.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = student.load_state_dict(teacher_state(checkpoint, weights), strict=False)
    assert not unexpected and all(name.startswith("conditioning.guidance_projection.") for name in missing), (missing, unexpected)


def seeded_noise(seeds: torch.Tensor, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """
    args:
        seeds: [B], dtype: torch.int64
        shape: per sample shape (32, H, W)
    returns:
        noise: [B, *shape], unit gaussian, dtype: torch.float32, identical on every GPU for the same seeds
    """
    generator = torch.Generator(device=device)
    noise = torch.empty((len(seeds), *shape), device=device)
    for index, seed in enumerate(seeds.tolist()):
        generator.manual_seed(seed)
        noise[index] = torch.randn(shape, generator=generator, device=device)
    return noise


@torch.no_grad()
def upscale_single_step(
    student: FluxStudentTransformer,
    latents_source: torch.Tensor,
    hff: torch.Tensor,
    guidance: torch.Tensor,
    noise: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    args:
        latents_source: [B, C, h, w], normalized FLUX.2 latents
        hff: [B], values: [0...1], NaN = null hff
        guidance: [B], >= 1
        noise: [B, C, 2h, 2w], unit gaussian
    returns:
        latents: [B, C, 2h, 2w], normalized FLUX.2 latents, dtype: torch.float32
    """
    hff_mask = ~torch.isnan(hff)
    velocity = student(noise.to(dtype), latents_source.to(dtype), torch.nan_to_num(hff, nan=0.0).float(), hff_mask, guidance.float())
    return from_residual(noise.float() - velocity.float(), latents_source, student.config.residual_std)
