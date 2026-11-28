"""
Single step student of the H3 -> FLUX.2 translator: the pretrained translator with the zero initialized guidance
condition of the FLUX.2 student (`flux_distill.student.GuidedConditioningEncoder`), so a student initialized from the
teacher computes exactly the teacher velocity at sigma 1. The student always runs at sigma 1: residual = noise - velocity.
"""

from pathlib import Path

import torch

from stdev.flux_distill.student import STUDENT_SIGMA, GuidedConditioningEncoder
from stdev.flux_pretrain.model import ModelConfig
from stdev.h3_pretrain.config import LINEAR_BASE_PATH
from stdev.h3_pretrain.model import H3TranslatorTransformer


def model_config_of(checkpoint: dict) -> ModelConfig:
    values = dict(checkpoint["model_config"])
    return ModelConfig(**{**values, "residual_std": tuple(values["residual_std"])})


def build_teacher(checkpoint: dict, weights: str) -> H3TranslatorTransformer:
    """
    Pretrained translator (float32, cpu) with the raw or EMA parameters of a pretrain checkpoint.
    """
    model = H3TranslatorTransformer(model_config_of(checkpoint), tuple(checkpoint["residual_mean"]), LINEAR_BASE_PATH)
    model.load_state_dict(checkpoint["model"])
    if weights == "ema":
        with torch.no_grad():
            for parameter, ema_parameter in zip(model.parameters(), checkpoint["ema"], strict=True):
                parameter.copy_(ema_parameter)
    return model


class H3StudentTransformer(H3TranslatorTransformer):
    def __init__(self, config: ModelConfig, residual_mean: tuple[float, ...], fit_path: Path | None = None) -> None:
        super().__init__(config, residual_mean, fit_path)
        self.conditioning = GuidedConditioningEncoder(config.condition_dim)

    def forward(
        self,
        noise: torch.Tensor,
        latents_source: torch.Tensor,
        latents_base: torch.Tensor,
        hff: torch.Tensor,
        hff_mask: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            noise: [B, 32, H, W], unit gaussian (residual space, sigma 1)
            latents_source: [B, 24, H / 2, W / 2], normalized H3 latents
            latents_base: [B, 32, H, W], output of linear_base
            hff: [B], values: [0...1]
            hff_mask: [B], dtype: torch.bool, False = null hff
            guidance: [B], >= 1
        returns:
            velocity: [B, 32, H, W] (noise - residual)
        """
        sigmas = torch.full_like(hff, STUDENT_SIGMA)
        condition = self.conditioning(sigmas, hff, hff_mask, guidance)
        stacked = torch.cat([noise, latents_base.to(noise.dtype)], dim=1)  # [B, 64, H, W]
        features = self.input_conv(stacked) + self.source_encoder(latents_source.to(noise.dtype))
        tokens = features.permute(0, 2, 3, 1)  # [B, H, W, D]
        for block in self.blocks:
            tokens = block(tokens, condition)
        shift, scale = self.output_modulation(condition)[:, None, None, :].chunk(2, dim=-1)
        velocity = self.output(self.output_norm(tokens) * (1 + scale) + shift)  # [B, H, W, 32]
        return velocity.permute(0, 3, 1, 2)


def build_student_from_teacher(checkpoint_path: Path, weights: str) -> H3StudentTransformer:
    """
    Teacher weights and buffers into every student tensor except the guidance projection.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher = build_teacher(checkpoint, weights)
    student = H3StudentTransformer(teacher.config, tuple(checkpoint["residual_mean"]), LINEAR_BASE_PATH)
    missing, unexpected = student.load_state_dict(teacher.state_dict(), strict=False)
    assert not unexpected and all(name.startswith("conditioning.guidance_projection.") for name in missing), (missing, unexpected)
    return student


@torch.no_grad()
def translate_single_step(
    student: H3StudentTransformer,
    latents_source: torch.Tensor,
    hff: torch.Tensor,
    guidance: torch.Tensor,
    noise: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    args:
        latents_source: [B, 24, h, w], normalized H3 latents
        hff: [B], values: [0...1], NaN = null hff
        guidance: [B], >= 1
        noise: [B, 32, 2h, 2w], unit gaussian
    returns:
        latents: [B, 32, 2h, 2w], normalized FLUX.2 latents, dtype: torch.float32
    """
    latents_base = student.linear_base(latents_source)
    hff_mask = ~torch.isnan(hff)
    velocity = student(
        noise.to(dtype), latents_source.to(dtype), latents_base.to(dtype), torch.nan_to_num(hff, nan=0.0).float(), hff_mask, guidance.float()
    )
    return student.from_residual(noise.float() - velocity.float(), latents_base)
