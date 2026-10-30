"""
Large latent upscaler for pretraining: run_0077 (deep full resolution decoder) combined with
run_0078 (SDXL style window transformer block after every residual block on all levels).
Conditioned on the noise level, the mean pooled SDXL text embedding and four high frequency
features of the target image, each with a learned null embedding used for dropout and when the
value is missing.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils.checkpoint
from safetensors.torch import load_file

from stdev.upscalers.base import Upscaler

MODEL_NAME = "pretrain_large"

LATENT_CHANNELS = 32
SCALE_FACTOR = 2

LEVEL_CHANNELS = (768, 512, 1024)
LEVEL_BLOCKS = (1, 2, 2)
DECODER_LEVEL_BLOCKS = (8, 2, 2)
MIDDLE_BLOCKS = 2
MLP_EXPANSION = 4
CONDITION_DIM = 1024
TIME_FREQUENCY_DIM = 512
MAX_NORM_GROUPS = 32
TIMESTEP_SCALE = 1000.0
ATTENTION_HEAD_DIM = 64
WINDOW_SIZE = 8
TEXT_EMBEDDING_DIM = 768
HIGH_FREQUENCY_FEATURE_COUNT = 4

NUM_SAMPLING_STEPS = 20
SAMPLING_SIGMA_SHIFT = 2.0
INFERENCE_SEED = 0
INFERENCE_DTYPE = torch.bfloat16
DEVICE = "cuda"


def shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """
    Moves noise levels towards pure noise for shift > 1, as in SD3.

    args:
        sigmas: [...], values: [0...1], 1 is pure noise
    returns:
        shifted_sigmas: [...], values: [0...1], 1 is pure noise
    """
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(MAX_NORM_GROUPS, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=1e-6)


def sinusoidal_embedding(values: torch.Tensor) -> torch.Tensor:
    """
    Timestep style embedding of scalar values.

    args:
        values: [B], values: [0...1]
    returns:
        embedding: [B, TIME_FREQUENCY_DIM], dtype: torch.float32
    """
    half_dim = TIME_FREQUENCY_DIM // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half_dim, device=values.device, dtype=torch.float32)
        / half_dim
    )  # [half_dim]
    angles = values.float()[:, None] * TIMESTEP_SCALE * frequencies[None, :]  # [B, half_dim]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


def make_embedding_mlp(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, CONDITION_DIM),
        nn.SiLU(),
        nn.Linear(CONDITION_DIM, CONDITION_DIM),
    )


class ConditioningEncoder(nn.Module):
    """
    Sum of the noise level embedding, the pooled text embedding and one embedding per high
    frequency feature. Dropped or missing text / features are replaced by learned null embeddings.
    """

    def __init__(self) -> None:
        super().__init__()
        self.time_projection = make_embedding_mlp(TIME_FREQUENCY_DIM)
        self.text_projection = make_embedding_mlp(TEXT_EMBEDDING_DIM)
        self.text_null_embedding = nn.Parameter(torch.zeros(CONDITION_DIM))
        self.high_frequency_projections = nn.ModuleList(
            make_embedding_mlp(TIME_FREQUENCY_DIM) for _ in range(HIGH_FREQUENCY_FEATURE_COUNT)
        )
        self.high_frequency_null_embeddings = nn.Parameter(
            torch.zeros(HIGH_FREQUENCY_FEATURE_COUNT, CONDITION_DIM)
        )

    def forward(
        self,
        sigmas: torch.Tensor,
        text_embedding: torch.Tensor,
        text_mask: torch.Tensor,
        high_frequency_features: torch.Tensor,
        high_frequency_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            sigmas: [B], values: [0...1], 1 is pure noise
            text_embedding: [B, TEXT_EMBEDDING_DIM], mean pooled SDXL text encoder embeddings
            text_mask: [B], dtype: torch.bool, False uses the null embedding
            high_frequency_features: [B, HIGH_FREQUENCY_FEATURE_COUNT], values: [0...1]
                quantile normalized, may contain NaN where the mask is False
            high_frequency_mask: [B, HIGH_FREQUENCY_FEATURE_COUNT], dtype: torch.bool
        returns:
            condition: [B, CONDITION_DIM]
        """
        weight_dtype = self.time_projection[0].weight.dtype
        condition = self.time_projection(sinusoidal_embedding(sigmas).to(weight_dtype))
        text_condition = self.text_projection(text_embedding.to(weight_dtype))
        condition = condition + torch.where(
            text_mask[:, None], text_condition, self.text_null_embedding.to(text_condition.dtype)
        )
        features = torch.nan_to_num(high_frequency_features.float(), nan=0.0)
        for index, projection in enumerate(self.high_frequency_projections):
            feature_condition = projection(
                sinusoidal_embedding(features[:, index]).to(weight_dtype)
            )
            null_embedding = self.high_frequency_null_embeddings[index].to(feature_condition.dtype)
            condition = condition + torch.where(
                high_frequency_mask[:, index, None], feature_condition, null_embedding
            )
        return condition


class ConditionedResidualBlock(nn.Module):
    """
    Residual block with the condition injected as scale and shift of the second normalization.
    The last convolution is zero initialized, so each block starts as the identity.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm_first = make_group_norm(in_channels)
        self.conv_first = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.condition_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(CONDITION_DIM, 2 * out_channels)
        )
        self.norm_second = make_group_norm(out_channels)
        self.conv_second = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_second.weight)
        nn.init.zeros_(self.conv_second.bias)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C_in, H, W]
            condition: [B, CONDITION_DIM]
        returns:
            features: [B, C_out, H, W]
        """
        hidden = self.conv_first(functional.silu(self.norm_first(features)))
        scale, shift = self.condition_projection(condition).chunk(2, dim=1)  # [B, C_out]
        hidden = self.norm_second(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv_second(functional.silu(hidden))
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


class WindowTransformerBlock(nn.Module):
    """
    Window attention followed by a pointwise MLP (SDXL style transformer block).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = WindowAttention(channels)
        self.feed_forward = PointwiseFeedForward(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C, H, W]
        returns:
            features: [B, C, H, W]
        """
        return self.feed_forward(self.attention(features))


class ResidualBlockSequence(nn.Module):
    """
    Residual blocks, each followed by a window transformer block.
    """

    def __init__(
        self, in_channels: int, out_channels: int, num_blocks: int, gradient_checkpointing: bool
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList(
            ConditionedResidualBlock(in_channels if index == 0 else out_channels, out_channels)
            for index in range(num_blocks)
        )
        self.attentions = nn.ModuleList(
            WindowTransformerBlock(out_channels) for _ in range(num_blocks)
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C_in, H, W]
            condition: [B, CONDITION_DIM]
        returns:
            features: [B, C_out, H, W]
        """
        for block, attention in zip(self.blocks, self.attentions):
            if self.gradient_checkpointing and self.training:
                features = torch.utils.checkpoint.checkpoint(
                    block, features, condition, use_reentrant=False
                )
                features = torch.utils.checkpoint.checkpoint(
                    attention, features, use_reentrant=False
                )
            else:
                features = attention(block(features, condition))
        return features


class UpscalerUNet(nn.Module):
    """
    Convolutional UNet with window transformer blocks predicting the flow velocity of high
    resolution latents. Bilinearly upsampled low resolution latents are concatenated to the noisy
    latents. Latent height and width must be divisible by 4 * WINDOW_SIZE.
    """

    def __init__(self, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.conditioning = ConditioningEncoder()
        self.input_conv = nn.Conv2d(2 * LATENT_CHANNELS, LEVEL_CHANNELS[0], 3, padding=1)
        self.encoder_stages = nn.ModuleList(
            ResidualBlockSequence(
                LEVEL_CHANNELS[level],
                LEVEL_CHANNELS[level],
                LEVEL_BLOCKS[level],
                gradient_checkpointing,
            )
            for level in range(len(LEVEL_CHANNELS))
        )
        self.downsamples = nn.ModuleList(
            nn.Conv2d(LEVEL_CHANNELS[level], LEVEL_CHANNELS[level + 1], 3, stride=2, padding=1)
            for level in range(len(LEVEL_CHANNELS) - 1)
        )
        self.middle_blocks = ResidualBlockSequence(
            LEVEL_CHANNELS[-1], LEVEL_CHANNELS[-1], MIDDLE_BLOCKS, gradient_checkpointing
        )
        last_level = len(LEVEL_CHANNELS) - 1
        self.decoder_stages = nn.ModuleList(
            ResidualBlockSequence(
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

    def forward(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        latents_low_resolution: torch.Tensor,
        text_embedding: torch.Tensor,
        text_mask: torch.Tensor,
        high_frequency_features: torch.Tensor,
        high_frequency_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            noisy_latents: [B, C, H, W]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_low_resolution: [B, C, H / 2, W / 2]
            text_embedding: [B, TEXT_EMBEDDING_DIM]
            text_mask: [B], dtype: torch.bool
            high_frequency_features: [B, HIGH_FREQUENCY_FEATURE_COUNT], values: [0...1] or NaN
            high_frequency_mask: [B, HIGH_FREQUENCY_FEATURE_COUNT], dtype: torch.bool
        returns:
            velocity: [B, C, H, W]
        """
        condition = self.conditioning(
            sigmas, text_embedding, text_mask, high_frequency_features, high_frequency_mask
        )
        latents_upsampled = functional.interpolate(
            latents_low_resolution, size=noisy_latents.shape[-2:], mode="bilinear"
        )  # [B, C, H, W]
        features = self.input_conv(torch.cat([noisy_latents, latents_upsampled], dim=1))

        skips = []
        for level, encoder_stage in enumerate(self.encoder_stages):
            features = encoder_stage(features, condition)
            skips.append(features)
            if level < len(self.downsamples):
                features = self.downsamples[level](features)

        features = self.middle_blocks(features, condition)
        for level in reversed(range(len(LEVEL_CHANNELS))):
            skip = skips[level]
            features = functional.interpolate(features, size=skip.shape[-2:], mode="bilinear")
            features = self.decoder_stages[level](torch.cat([features, skip], dim=1), condition)
        return self.output_conv(functional.silu(self.output_norm(features)))


@torch.no_grad()
def sample_high_resolution_latents(
    model: nn.Module,
    latents_low_resolution: torch.Tensor,
    text_embedding: torch.Tensor,
    text_mask: torch.Tensor,
    high_frequency_features: torch.Tensor,
    high_frequency_mask: torch.Tensor,
    num_steps: int = NUM_SAMPLING_STEPS,
    sigma_shift: float = SAMPLING_SIGMA_SHIFT,
    seed: int = INFERENCE_SEED,
    dtype: torch.dtype = INFERENCE_DTYPE,
) -> torch.Tensor:
    """
    Euler sampling of the flow from noise to the high resolution latents.

    args:
        latents_low_resolution: [B, C, H, W]
        text_embedding: [B, TEXT_EMBEDDING_DIM]
        text_mask: [B], dtype: torch.bool
        high_frequency_features: [B, HIGH_FREQUENCY_FEATURE_COUNT]
        high_frequency_mask: [B, HIGH_FREQUENCY_FEATURE_COUNT], dtype: torch.bool
    returns:
        latents: [B, C, H * SCALE_FACTOR, W * SCALE_FACTOR], dtype: torch.float32
    """
    batch_size, channels, height, width = latents_low_resolution.shape
    device = latents_low_resolution.device
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (batch_size, channels, height * SCALE_FACTOR, width * SCALE_FACTOR),
        generator=generator,
        device=device,
    )  # [B, C, H_hr, W_hr], dtype: torch.float32
    latents_low_resolution = latents_low_resolution.to(dtype)
    text_embedding = text_embedding.to(dtype)
    sampling_sigmas = shift_sigmas(
        torch.linspace(1.0, 0.0, num_steps + 1, device=device), sigma_shift
    )
    for current_sigma, next_sigma in zip(sampling_sigmas[:-1], sampling_sigmas[1:]):
        velocity = model(
            latents.to(dtype),
            current_sigma.expand(batch_size),
            latents_low_resolution,
            text_embedding,
            text_mask,
            high_frequency_features,
            high_frequency_mask,
        )
        latents = latents + (next_sigma - current_sigma) * velocity.float()
    return latents


class PretrainLargeUpscaler(Upscaler):
    """
    Inference wrapper of a pretrained checkpoint. Training is done by the pretraining pipeline
    (scripts/run_pretrain.py). Without explicit high frequency features the model uses their null
    embeddings; set `high_frequency_features` ([B, 4], quantile normalized) to condition on them.
    """

    def __init__(self, weights_path: Path | None = None) -> None:
        self.weights_path = weights_path
        self.model: nn.Module | None = None
        self.high_frequency_features: torch.Tensor | None = None

    def train(self, smoke_mode: bool = False) -> None:
        raise NotImplementedError("Pretraining runs through scripts/run_pretrain.py")

    def prepare_to_benchmark(self) -> None:
        if self.weights_path is None:
            raise ValueError("weights_path is required to load the model")
        model = UpscalerUNet()
        model.load_state_dict(load_file(str(self.weights_path)))
        self.model = model.to(DEVICE, INFERENCE_DTYPE).eval().requires_grad_(False)

    def compile(self) -> None:
        self.model = torch.compile(self.model, mode="reduce-overhead")

    def conditioning_inputs(
        self, prompt_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        args:
            prompt_embeds: [B, 77, 768]
        returns:
            text_embedding: [B, 768], text_mask: [B], high_frequency_features: [B, 4],
            high_frequency_mask: [B, 4]
        """
        batch_size = prompt_embeds.shape[0]
        device = prompt_embeds.device
        text_embedding = prompt_embeds.float().mean(dim=1)
        text_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        if self.high_frequency_features is None:
            features = torch.zeros(batch_size, HIGH_FREQUENCY_FEATURE_COUNT, device=device)
            mask = torch.zeros_like(features, dtype=torch.bool)
        else:
            features = self.high_frequency_features.to(device).float().expand(batch_size, -1)
            mask = ~torch.isnan(features)
        return text_embedding, text_mask, features, mask

    @torch.no_grad()
    def upscale(self, tensor: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            prompt_embeds: [B, 77, 768], dtype = float32, SDXL text encoder embeddings
        returns:
            tensor: [B, C, H_lr * SCALE_FACTOR, W_lr * SCALE_FACTOR], same dtype as the input
        """
        latents = sample_high_resolution_latents(
            self.model, tensor, *self.conditioning_inputs(prompt_embeds)
        )
        return latents.to(tensor.dtype)

    @torch.no_grad()
    def single_forward(self, tensor: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            prompt_embeds: [B, 77, 768], dtype = float32
        returns:
            velocity: [B, C, H_lr * SCALE_FACTOR, W_lr * SCALE_FACTOR], dtype: INFERENCE_DTYPE
        """
        batch_size, channels, height, width = tensor.shape
        noisy_latents = torch.randn(
            (batch_size, channels, height * SCALE_FACTOR, width * SCALE_FACTOR),
            device=tensor.device,
            dtype=INFERENCE_DTYPE,
        )
        sigmas = torch.ones(batch_size, device=tensor.device)
        text_embedding, text_mask, features, mask = self.conditioning_inputs(prompt_embeds)
        return self.model(
            noisy_latents,
            sigmas,
            tensor.to(INFERENCE_DTYPE),
            text_embedding.to(INFERENCE_DTYPE),
            text_mask,
            features,
            mask,
        )

    def get_name(self) -> str:
        return MODEL_NAME
