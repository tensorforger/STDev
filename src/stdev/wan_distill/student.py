"""
Single step student of the Wan 2.1 latent upscaler: the pretrained window transformer with a third scalar
condition, the classifier free guidance scale the teacher used (negative = null hff). Its embedding is added
to the sigma and hff embeddings before the RMS norm; the last layer is zero initialized, so a student
initialized from the teacher computes exactly the teacher velocity. The student always runs at sigma 1:
residual = noise_scale * noise - velocity.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional

from stdev.wan_pretrain.model import (
    HFF_SCALE,
    SIGMA_SCALE,
    ConditioningEncoder,
    ModelConfig,
    WanUpscalerTransformer,
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


class WanStudentTransformer(WanUpscalerTransformer):
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
            noise: [B, C, T, H, W], noise_scale * unit gaussian (residual space, sigma 1)
            latents_source: [B, C, T_lr, H_lr, W_lr], normalized Wan 2.1 latents
            hff: [B], values: [0...1]
            hff_mask: [B], dtype: torch.bool, False = null hff
            guidance: [B], >= 1
        returns:
            velocity: [B, C, T, H, W] (noise - residual)
        """
        sigmas = torch.full_like(hff, STUDENT_SIGMA)
        condition = self.conditioning(sigmas, hff, hff_mask, guidance)
        return self.forward_with_condition(noise, condition, latents_source)


def load_teacher_into_student(student: WanStudentTransformer, checkpoint_path: Path) -> None:
    """
    EMA weights of a pretraining checkpoint into every student parameter except the guidance projection.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher = WanUpscalerTransformer(ModelConfig(**checkpoint["model_config"]))
    names = [name for name, _ in teacher.named_parameters()]
    state = dict(checkpoint["model"])
    state.update(dict(zip(names, checkpoint["ema"], strict=True)))
    missing, unexpected = student.load_state_dict(state, strict=False)
    assert not unexpected and all(name.startswith("conditioning.guidance_projection.") for name in missing), (missing, unexpected)


@torch.no_grad()
def upscale_single_step(
    student: WanStudentTransformer,
    latents_source: torch.Tensor,
    hff: torch.Tensor,
    guidance: torch.Tensor,
    noise: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    args:
        latents_source: [B, C, T_lr, H_lr, W_lr], normalized Wan 2.1 latents
        hff: [B], values: [0...1], NaN = null hff
        guidance: [B], >= 1
        noise: [B, C, T_hr, H_hr, W_hr], noise_scale * unit gaussian
    returns:
        latents: [B, C, T_hr, H_hr, W_hr], normalized Wan 2.1 latents, dtype: torch.float32
    """
    hff_mask = ~torch.isnan(hff)
    velocity = student(noise.to(dtype), latents_source.to(dtype), torch.nan_to_num(hff, nan=0.0).float(), hff_mask, guidance.float())
    latents = from_residual(noise.float() - velocity.float(), latents_source)
    if latents.shape[3:] == latents_source.shape[3:]:
        latents[:, :, 0] = latents_source[:, :, 0].float()
    return latents
