"""
FlowUpscaler UNet architecture trained from random initialization as a plain multi-step rectified
flow model on FLUX.2 outputs (scripts/run_flow_sft.py), sampled with a few uniform Euler steps.
No distillation, no text conditioning.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from stdev.upscalers.base import Upscaler
from stdev.upscalers.flow_upscaler import load_original_unet_class

MODEL_NAME = "flow_upscaler_sft"
WEIGHTS_PATH_VARIABLE = "FLOW_UPSCALER_SFT_WEIGHTS"
RESULT_NAME_VARIABLE = "FLOW_UPSCALER_SFT_NAME"

LATENT_CHANNELS = 32
SCALE_FACTOR = 2
TIMESTEP_SCALE = 1000.0
NUM_SAMPLING_STEPS = 4

INFERENCE_SEED = 0
INFERENCE_DTYPE = torch.bfloat16
DEVICE = "cuda"


class FlowUpscalerSFTUNet(load_original_unet_class()):
    """
    Original FlowUpscaler architecture without dropout, conditioned on the noise level sigma.
    """

    def __init__(self) -> None:
        super().__init__(dropout=0.0)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        latents_low_resolution: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            noisy_latents: [B, C, H, W]
            sigmas: [B], values: [0...1]
            latents_low_resolution: [B, C, H / 2, W / 2]
        returns:
            velocity: [B, C, H, W], flow direction noise - latents
        """
        return super().forward(
            noisy_latents, sigmas.to(noisy_latents.dtype) * TIMESTEP_SCALE, latents_low_resolution
        )


class FlowUpscalerSFT(Upscaler):
    """
    Weights path and result name are set with FLOW_UPSCALER_SFT_WEIGHTS and FLOW_UPSCALER_SFT_NAME.
    """

    def __init__(self) -> None:
        self.weights_path = Path(os.environ[WEIGHTS_PATH_VARIABLE])
        self.model: nn.Module | None = None

    def train(self, smoke_mode: bool = False) -> None:
        raise NotImplementedError("Training runs through scripts/run_flow_sft.py")

    def prepare_to_benchmark(self) -> None:
        model = FlowUpscalerSFTUNet()
        model.load_state_dict(load_file(str(self.weights_path)))
        self.model = model.to(DEVICE, INFERENCE_DTYPE).eval().requires_grad_(False)

    def compile(self) -> None:
        self.model = torch.compile(self.model, mode="reduce-overhead")

    @torch.no_grad()
    def upscale(self, tensor: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            prompt_embeds: [B, 77, 768], unused
        returns:
            tensor: [B, C, H_lr * SCALE_FACTOR, W_lr * SCALE_FACTOR], same dtype as the input
        """
        batch_size, channels, height, width = tensor.shape
        generator = torch.Generator(device=tensor.device).manual_seed(INFERENCE_SEED)
        latents = torch.randn(
            (batch_size, channels, height * SCALE_FACTOR, width * SCALE_FACTOR),
            generator=generator,
            device=tensor.device,
        )  # [B, C, H_hr, W_hr], dtype: torch.float32
        latents_low_resolution = tensor.to(INFERENCE_DTYPE)
        sigmas = torch.linspace(1.0, 0.0, NUM_SAMPLING_STEPS + 1, device=tensor.device)
        for current_sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
            velocity = self.model(
                latents.to(INFERENCE_DTYPE),
                current_sigma.expand(batch_size),
                latents_low_resolution,
            )
            latents = latents + (next_sigma - current_sigma) * velocity.float()
        return latents.to(tensor.dtype)

    @torch.no_grad()
    def single_forward(self, tensor: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            prompt_embeds: [B, 77, 768], unused
        returns:
            velocity: [B, C, H_lr * SCALE_FACTOR, W_lr * SCALE_FACTOR], dtype: INFERENCE_DTYPE
        """
        batch_size, channels, height, width = tensor.shape
        noise = torch.randn(
            (batch_size, channels, height * SCALE_FACTOR, width * SCALE_FACTOR),
            device=tensor.device,
            dtype=INFERENCE_DTYPE,
        )
        sigmas = torch.ones(batch_size, device=tensor.device)
        return self.model(noise, sigmas, tensor.to(INFERENCE_DTYPE))

    def get_name(self) -> str:
        return os.environ.get(RESULT_NAME_VARIABLE, MODEL_NAME)
