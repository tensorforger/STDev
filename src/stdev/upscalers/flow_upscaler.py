"""
Inference wrapper of the previous work TensorForger/FlowUpscaler (59M parameter attention free UNet
with a separate low resolution encoder and FiLM conditioning), cloned to ./dev/FlowUpscaler.
Single Euler step of flow matching: sample = noise - velocity, with timestep 1000 (sigma 1) as in
the official ComfyUI node. No text conditioning.
"""

import importlib.util
import os
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from stdev.upscalers.base import Upscaler

MODEL_NAME = "flow_upscaler"
REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[3] / "dev" / "FlowUpscaler"
WEIGHTS_PATH_VARIABLE = "FLOW_UPSCALER_WEIGHTS"
RESULT_NAME_VARIABLE = "FLOW_UPSCALER_NAME"

LATENT_CHANNELS = 32
SCALE_FACTOR = 2
TIMESTEP = 1000.0

INFERENCE_SEED = 0
INFERENCE_DTYPE = torch.bfloat16
DEVICE = "cuda"


def load_original_unet_class() -> type[nn.Module]:
    specification = importlib.util.spec_from_file_location(
        "flow_upscaler_unet", REPOSITORY_DIRECTORY / "upscaler_unet.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.UpscalerUNet


class FlowUpscalerUNet(load_original_unet_class()):
    """
    Original FlowUpscaler architecture (same state dict) with the interface of the single step
    distillation: always pure noise at timestep 1000, the output is the velocity.
    """

    def forward(self, noise: torch.Tensor, latents_low_resolution: torch.Tensor) -> torch.Tensor:
        """
        args:
            noise: [B, C, H, W]
            latents_low_resolution: [B, C, H / 2, W / 2]
        returns:
            velocity: [B, C, H, W], high resolution latents are noise - velocity
        """
        timestep = torch.full((noise.shape[0],), TIMESTEP, device=noise.device, dtype=noise.dtype)
        return super().forward(noise, timestep, latents_low_resolution)


class FlowUpscaler(Upscaler):
    """
    Weights path and result name can be overridden with FLOW_UPSCALER_WEIGHTS and FLOW_UPSCALER_NAME.
    """

    def __init__(self) -> None:
        self.weights_path = Path(
            os.environ.get(
                WEIGHTS_PATH_VARIABLE, REPOSITORY_DIRECTORY / "flow_upscaler.safetensors"
            )
        )
        self.model: nn.Module | None = None

    def train(self, smoke_mode: bool = False) -> None:
        raise NotImplementedError("Pretrained external model, training is not supported")

    def prepare_to_benchmark(self) -> None:
        model = FlowUpscalerUNet()
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
        noise = torch.randn(
            (batch_size, channels, height * SCALE_FACTOR, width * SCALE_FACTOR),
            generator=generator,
            device=tensor.device,
        )  # [B, C, H_hr, W_hr], dtype: torch.float32
        velocity = self.model(noise.to(INFERENCE_DTYPE), tensor.to(INFERENCE_DTYPE))
        return (noise - velocity.float()).to(tensor.dtype)

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
        return self.model(noise, tensor.to(INFERENCE_DTYPE))

    def get_name(self) -> str:
        return os.environ.get(RESULT_NAME_VARIABLE, MODEL_NAME)
