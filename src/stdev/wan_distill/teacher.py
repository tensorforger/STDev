"""
Teacher of the Wan distillation: the pretrained upscaler (EMA weights) and its Euler sampler with a per sample
hff condition and classifier free guidance scale (negative branch = null hff embedding).
"""

from pathlib import Path

import torch

from stdev.wan_pretrain.config import MODEL
from stdev.wan_pretrain.model import WanUpscalerTransformer, from_residual, shift_sigmas

INFERENCE_DTYPE = torch.bfloat16


def load_teacher(checkpoint_path: Path, device: torch.device) -> WanUpscalerTransformer:
    """
    EMA weights of a full pretraining checkpoint, bf16, compiled blocks.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = WanUpscalerTransformer(MODEL)
    model.load_state_dict(checkpoint["model"])
    names = [name for name, _ in model.named_parameters()]
    state = model.state_dict()
    for name, tensor in zip(names, checkpoint["ema"], strict=True):
        state[name] = tensor
    model.load_state_dict(state)
    del checkpoint
    model = model.to(device, INFERENCE_DTYPE).eval().requires_grad_(False)
    model.compile_blocks()
    return model


@torch.no_grad()
def sample_teacher(
    model: WanUpscalerTransformer,
    latents_source: torch.Tensor,
    initial_residual: torch.Tensor,
    hff: torch.Tensor,
    guidance: torch.Tensor,
    steps: int,
    shift: float,
) -> torch.Tensor:
    """
    Euler sampling of the residual flow from given initial noise. If any guidance differs from 1, every
    sample is evaluated with its hff and with the null hff in one doubled batch, velocity =
    negative + guidance * (positive - negative) (guidance 1 gives the positive velocity exactly).
    The first latent frame is copied from the source for temporal-only upscaling.

    args:
        latents_source: [B, 16, T_lr, H_lr, W_lr], normalized Wan 2.1 latents
        initial_residual: [B, 16, T_hr, H_hr, W_hr], noise_scale * unit noise (residual space, sigma 1)
        hff: [B], values: [0...1], NaN = null embedding
        guidance: [B], >= 1
    returns:
        latents: [B, 16, T_hr, H_hr, W_hr], normalized Wan 2.1 latents, dtype: torch.float32
    """
    batch_size = latents_source.shape[0]
    device = latents_source.device
    hff_mask = ~torch.isnan(hff)
    hff_values = torch.nan_to_num(hff, nan=0.0).float()
    guided = bool((guidance != 1.0).any())
    if guided:
        assert bool(hff_mask.all()), "guided samples need a positive hff"
        source = torch.cat([latents_source, latents_source]).to(INFERENCE_DTYPE)
        model_hff = torch.cat([hff_values, torch.zeros_like(hff_values)])
        model_mask = torch.cat([hff_mask, torch.zeros_like(hff_mask)])
    else:
        source = latents_source.to(INFERENCE_DTYPE)
        model_hff = hff_values
        model_mask = hff_mask
    scale = guidance.float().view(batch_size, 1, 1, 1, 1)
    samples = initial_residual.float()
    sigmas = shift_sigmas(torch.linspace(1.0, 0.0, steps + 1, device=device), shift)
    for current_sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        model_input = torch.cat([samples, samples]) if guided else samples
        velocity = model(model_input.to(INFERENCE_DTYPE), current_sigma.expand(model_input.shape[0]), source, model_hff, model_mask).float()
        if guided:
            positive, negative = velocity.chunk(2)
            velocity = negative + scale * (positive - negative)
        samples = samples + (next_sigma - current_sigma) * velocity
    latents = from_residual(samples, latents_source)
    if latents.shape[3:] == latents_source.shape[3:]:
        latents[:, :, 0] = latents_source[:, :, 0].float()
    return latents
