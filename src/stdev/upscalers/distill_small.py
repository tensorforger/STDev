"""
Small single step latent upscaler distilled from the rectified flow distillation pairs of the large
model (scripts/run_small_distill.py). Maps the teacher's initial noise and the bilinearly
upsampled low resolution latents to the velocity at pure noise (t = 1), so the high resolution
latents are noise - velocity. No noise level, text or high frequency conditioning.

SDXL style blocks (residual block + window transformer block) on all levels, with most of the
~50M parameters in the full resolution decoder: autoresearch found that level 0 width >= 512 and a
decoder heavy layout carry the quality, while wide or deep lower levels do not pay off.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils.checkpoint
from safetensors.torch import load_file

from stdev.upscalers.base import Upscaler

MODEL_NAME = "distill_small"
WEIGHTS_PATH_VARIABLE = "SMALL_UPSCALER_WEIGHTS"
RESULT_NAME_VARIABLE = "SMALL_UPSCALER_NAME"

LATENT_CHANNELS = 32
SCALE_FACTOR = 2

LEVEL_CHANNELS = (512, 256, 256)
ENCODER_LEVEL_BLOCKS = (1, 1, 1)
DECODER_LEVEL_BLOCKS = (3, 2, 1)
MIDDLE_BLOCKS = 1
MLP_EXPANSION = 4
MAX_NORM_GROUPS = 32
ATTENTION_HEAD_DIM = 64
WINDOW_SIZE = 8

INFERENCE_SEED = 0
INFERENCE_DTYPE = torch.bfloat16
DEVICE = "cuda"


def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(MAX_NORM_GROUPS, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=1e-6)


class ResidualBlock(nn.Module):
    """
    Two convolution residual block. The last convolution is zero initialized, so the block starts
    as the identity.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm_first = make_group_norm(in_channels)
        self.conv_first = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm_second = make_group_norm(out_channels)
        self.conv_second = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_second.weight)
        nn.init.zeros_(self.conv_second.bias)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C_in, H, W]
        returns:
            features: [B, C_out, H, W]
        """
        hidden = self.conv_first(functional.silu(self.norm_first(features)))
        hidden = self.conv_second(functional.silu(self.norm_second(hidden)))
        return self.skip(features) + hidden


class WindowAttention(nn.Module):
    """
    Multi-head self-attention inside non-overlapping WINDOW_SIZE x WINDOW_SIZE windows (linear in
    image area). The output projection is zero initialized, so the module starts as the identity.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = make_group_norm(channels)
        self.query_key_value = nn.Conv2d(channels, 3 * channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C, H, W], H and W divisible by WINDOW_SIZE
        returns:
            features: [B, C, H, W]
        """
        batch_size, channels, height, width = features.shape
        heads = channels // ATTENTION_HEAD_DIM
        rows, columns = height // WINDOW_SIZE, width // WINDOW_SIZE
        hidden = self.query_key_value(self.norm(features)).reshape(
            batch_size, 3, heads, ATTENTION_HEAD_DIM, rows, WINDOW_SIZE, columns, WINDOW_SIZE
        )
        hidden = hidden.permute(1, 0, 2, 4, 6, 5, 7, 3).reshape(
            3, batch_size, heads, rows * columns, WINDOW_SIZE * WINDOW_SIZE, ATTENTION_HEAD_DIM
        )  # [3, B, heads, windows, window tokens, D]
        attended = functional.scaled_dot_product_attention(hidden[0], hidden[1], hidden[2])
        attended = attended.reshape(
            batch_size, heads, rows, columns, WINDOW_SIZE, WINDOW_SIZE, ATTENTION_HEAD_DIM
        ).permute(0, 1, 6, 2, 4, 3, 5)  # [B, heads, D, rows, window, columns, window]
        return features + self.output(attended.reshape(batch_size, channels, height, width))


class PointwiseFeedForward(nn.Module):
    """
    Per position MLP of a transformer block. The last projection is zero initialized.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = make_group_norm(channels)
        self.expand = nn.Conv2d(channels, MLP_EXPANSION * channels, 1)
        self.contract = nn.Conv2d(MLP_EXPANSION * channels, channels, 1)
        nn.init.zeros_(self.contract.weight)
        nn.init.zeros_(self.contract.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C, H, W]
        returns:
            features: [B, C, H, W]
        """
        return features + self.contract(functional.gelu(self.expand(self.norm(features))))


class ResidualTransformerBlock(nn.Module):
    """
    SDXL style block: residual block followed by window attention and a pointwise MLP.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.residual = ResidualBlock(in_channels, out_channels)
        self.attention = WindowAttention(out_channels)
        self.feed_forward = PointwiseFeedForward(out_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C_in, H, W]
        returns:
            features: [B, C_out, H, W]
        """
        return self.feed_forward(self.attention(self.residual(features)))


class BlockSequence(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, num_blocks: int, gradient_checkpointing: bool
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList(
            ResidualTransformerBlock(in_channels if index == 0 else out_channels, out_channels)
            for index in range(num_blocks)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C_in, H, W]
        returns:
            features: [B, C_out, H, W]
        """
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                features = torch.utils.checkpoint.checkpoint(block, features, use_reentrant=False)
            else:
                features = block(features)
        return features


class SmallUpscalerUNet(nn.Module):
    """
    UNet predicting the velocity from pure noise to the high resolution latents. Latent height and
    width must be divisible by 4 * WINDOW_SIZE.
    """

    def __init__(self, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(2 * LATENT_CHANNELS, LEVEL_CHANNELS[0], 3, padding=1)
        self.encoder_stages = nn.ModuleList(
            BlockSequence(
                LEVEL_CHANNELS[level],
                LEVEL_CHANNELS[level],
                ENCODER_LEVEL_BLOCKS[level],
                gradient_checkpointing,
            )
            for level in range(len(LEVEL_CHANNELS))
        )
        self.downsamples = nn.ModuleList(
            nn.Conv2d(LEVEL_CHANNELS[level], LEVEL_CHANNELS[level + 1], 3, stride=2, padding=1)
            for level in range(len(LEVEL_CHANNELS) - 1)
        )
        self.middle_blocks = BlockSequence(
            LEVEL_CHANNELS[-1], LEVEL_CHANNELS[-1], MIDDLE_BLOCKS, gradient_checkpointing
        )
        last_level = len(LEVEL_CHANNELS) - 1
        self.decoder_stages = nn.ModuleList(
            BlockSequence(
                LEVEL_CHANNELS[min(level + 1, last_level)] + LEVEL_CHANNELS[level],
                LEVEL_CHANNELS[level],
                DECODER_LEVEL_BLOCKS[level],
                gradient_checkpointing,
            )
            for level in range(len(LEVEL_CHANNELS))
        )
        self.output_norm = make_group_norm(LEVEL_CHANNELS[0])
        self.output_conv = nn.Conv2d(LEVEL_CHANNELS[0], LATENT_CHANNELS, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, noise: torch.Tensor, latents_low_resolution: torch.Tensor) -> torch.Tensor:
        """
        args:
            noise: [B, C, H, W]
            latents_low_resolution: [B, C, H / 2, W / 2]
        returns:
            velocity: [B, C, H, W], high resolution latents are noise - velocity
        """
        latents_upsampled = functional.interpolate(
            latents_low_resolution, size=noise.shape[-2:], mode="bilinear"
        )  # [B, C, H, W]
        features = self.input_conv(torch.cat([noise, latents_upsampled], dim=1))

        skips = []
        for level, encoder_stage in enumerate(self.encoder_stages):
            features = encoder_stage(features)
            skips.append(features)
            if level < len(self.downsamples):
                features = self.downsamples[level](features)

        features = self.middle_blocks(features)
        for level in reversed(range(len(LEVEL_CHANNELS))):
            skip = skips[level]
            features = functional.interpolate(features, size=skip.shape[-2:], mode="bilinear")
            features = self.decoder_stages[level](torch.cat([features, skip], dim=1))
        return self.output_conv(functional.silu(self.output_norm(features)))


class DistillSmallUpscaler(Upscaler):
    """
    Inference wrapper of a trained checkpoint. Weights path and result name come from the
    environment variables SMALL_UPSCALER_WEIGHTS and SMALL_UPSCALER_NAME, so that the benchmark
    script can load this file directly.
    """

    def __init__(self) -> None:
        weights_path = os.environ.get(WEIGHTS_PATH_VARIABLE)
        self.weights_path = Path(weights_path) if weights_path else None
        self.model: nn.Module | None = None

    def train(self, smoke_mode: bool = False) -> None:
        raise NotImplementedError("Training runs through scripts/run_small_distill.py")

    def prepare_to_benchmark(self) -> None:
        if self.weights_path is None:
            raise ValueError(f"{WEIGHTS_PATH_VARIABLE} is required to load the model")
        model = SmallUpscalerUNet()
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
