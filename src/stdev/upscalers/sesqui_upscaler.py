"""
Inference wrapper of LoganBooker/SesquiLSR (Flux2 variant, ~3M parameter attention free
regression upscaler), cloned to ./dev/SesquiLSR. Not generative: one deterministic forward pass,
no noise and no text conditioning. The model operates on raw VAE latents, so the benchmark's
batch norm normalized latents are denormalized before and renormalized after the model, as in the
official ComfyUI node.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from stdev.upscalers.base import Upscaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SESQUI_DIRECTORY = REPOSITORY_ROOT / "dev" / "SesquiLSR"
VAE_WEIGHTS_PATH = REPOSITORY_ROOT / "models/FLUX.2-klein-4B/vae/diffusion_pytorch_model.safetensors"
WEIGHTS_PATH_VARIABLE = "SESQUI_UPSCALER_WEIGHTS"
RESULT_NAME_VARIABLE = "SESQUI_UPSCALER_NAME"
MODEL_NAME = "sesqui_lsr"

LATENT_CHANNELS = 32
SCALE_FACTOR = 2
BATCH_NORM_EPS = 1e-4
PATCH_SIZE = 2

INFERENCE_DTYPE = torch.bfloat16
DEVICE = "cuda"


def patchify(latents: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents: [B, C, H, W]
    returns:
        patches: [B, 4 * C, H / 2, W / 2]
    """
    batch_size, channels, height, width = latents.shape
    patches = latents.reshape(
        batch_size, channels, height // PATCH_SIZE, PATCH_SIZE, width // PATCH_SIZE, PATCH_SIZE
    )
    return patches.permute(0, 1, 3, 5, 2, 4).reshape(
        batch_size, channels * PATCH_SIZE**2, height // PATCH_SIZE, width // PATCH_SIZE
    )


def unpatchify(patches: torch.Tensor) -> torch.Tensor:
    """
    args:
        patches: [B, 4 * C, H, W]
    returns:
        latents: [B, C, 2 * H, 2 * W]
    """
    batch_size, channels, height, width = patches.shape
    latents = patches.reshape(
        batch_size, channels // PATCH_SIZE**2, PATCH_SIZE, PATCH_SIZE, height, width
    )
    return latents.permute(0, 1, 4, 2, 5, 3).reshape(
        batch_size, channels // PATCH_SIZE**2, height * PATCH_SIZE, width * PATCH_SIZE
    )


def load_latent_upscaler_class() -> type[nn.Module]:
    sys.path.insert(0, str(SESQUI_DIRECTORY))
    try:
        from sesqui_lsr import LatentUpscaler
    finally:
        sys.path.remove(str(SESQUI_DIRECTORY))
    return LatentUpscaler


class SesquiLSRUpscaler(Upscaler):
    """
    Weights path and result name can be overridden with SESQUI_UPSCALER_WEIGHTS and
    SESQUI_UPSCALER_NAME.
    """

    def __init__(self) -> None:
        self.weights_path = Path(
            os.environ.get(WEIGHTS_PATH_VARIABLE, SESQUI_DIRECTORY / "models/upscaler_Flux2.safetensors")
        )
        self.model: nn.Module | None = None
        self.batch_norm_mean: torch.Tensor | None = None
        self.batch_norm_std: torch.Tensor | None = None

    def train(self, smoke_mode: bool = False) -> None:
        raise NotImplementedError("Pretrained external model, training is not supported")

    def prepare_to_benchmark(self) -> None:
        model = load_latent_upscaler_class()(in_channels=LATENT_CHANNELS)
        model.load_state_dict(load_file(str(self.weights_path)))
        self.model = model.to(DEVICE, INFERENCE_DTYPE).eval().requires_grad_(False)

        vae_state_dict = load_file(str(VAE_WEIGHTS_PATH))
        self.batch_norm_mean = vae_state_dict["bn.running_mean"].float().view(1, -1, 1, 1).to(DEVICE)
        self.batch_norm_std = (
            (vae_state_dict["bn.running_var"].float() + BATCH_NORM_EPS).sqrt().view(1, -1, 1, 1).to(DEVICE)
        )

    def compile(self) -> None:
        self.model = torch.compile(self.model, mode="reduce-overhead")

    def _to_raw_latents(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H, W], batch norm normalized, unpatchified
        returns:
            latents: [B, C, H, W], raw VAE latents, dtype: torch.float32
        """
        patches = patchify(tensor.float()) * self.batch_norm_std + self.batch_norm_mean
        return unpatchify(patches)

    def _to_normalized_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C, H, W], raw VAE latents
        returns:
            tensor: [B, C, H, W], batch norm normalized, unpatchified, dtype: torch.float32
        """
        patches = (patchify(latents.float()) - self.batch_norm_mean) / self.batch_norm_std
        return unpatchify(patches)

    @torch.no_grad()
    def upscale(self, tensor: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            prompt_embeds: [B, 77, 768], unused
        returns:
            tensor: [B, C, H_lr * SCALE_FACTOR, W_lr * SCALE_FACTOR], same dtype as the input
        """
        height, width = tensor.shape[-2:]
        raw_latents = self._to_raw_latents(tensor).to(INFERENCE_DTYPE)
        upscaled = self.model(raw_latents, (height * SCALE_FACTOR, width * SCALE_FACTOR))
        return self._to_normalized_latents(upscaled).to(tensor.dtype)

    @torch.no_grad()
    def single_forward(self, tensor: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            prompt_embeds: [B, 77, 768], unused
        returns:
            tensor: [B, C, H_lr * SCALE_FACTOR, W_lr * SCALE_FACTOR], dtype: INFERENCE_DTYPE
        """
        height, width = tensor.shape[-2:]
        return self.model(tensor.to(INFERENCE_DTYPE), (height * SCALE_FACTOR, width * SCALE_FACTOR))

    def get_name(self) -> str:
        return os.environ.get(RESULT_NAME_VARIABLE, MODEL_NAME)
