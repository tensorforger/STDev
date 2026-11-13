"""
Small universal latent upscaler: one shared ~50M core for many VAE latent spaces (images and
videos). The core is the SDXL style window transformer UNet of distill_small, with the noise level,
the VAE identity (one hot) and the high frequency feature (HFF, quantile normalized Immerkaer)
injected as scale and shift of the residual blocks as in pretrain_large. Each VAE has its own 1x1
input and output adapter mapping its latent channels to / from the hidden width of the full
resolution level. Frames of video latents are processed independently, as a batch of images.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils.checkpoint
from safetensors.torch import load_file

from stdev.multi_vae_upscalers.base import MultiVAEUpscaler

MODEL_NAME = "small_universal"

VAE_LATENT_CHANNELS: dict[str, int] = {
    "sdxl": 4,
    "sd_3_5": 16,
    "flux_1": 16,
    "flux_2": 32,
    "qwen_image_2_1": 64,
    "wan_2_1": 16,
    "tiny_ltx_2_5": 128,
    "tiny_minimax_h3": 24,
}
VAE_NAMES = tuple(VAE_LATENT_CHANNELS)
LATENT_SPACE_BY_VAE_NAME = {"tiny_wan_2_1": "wan_2_1"}
SCALE_FACTOR = 2

LEVEL_CHANNELS = (512, 256, 256)
ENCODER_LEVEL_BLOCKS = (1, 1, 1)
DECODER_LEVEL_BLOCKS = (3, 2, 1)
MIDDLE_BLOCKS = 1
MLP_EXPANSION = 4
CONDITION_DIM = 512
TIME_FREQUENCY_DIM = 256
MAX_NORM_GROUPS = 32
TIMESTEP_SCALE = 1000.0
ATTENTION_HEAD_DIM = 64
WINDOW_SIZE = 8

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


def frames_to_batch(latents: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents: [B, C, T, H, W]
    returns:
        frames: [B * T, C, H, W]
    """
    return latents.permute(0, 2, 1, 3, 4).flatten(0, 1)


def batch_to_frames(frames: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    args:
        frames: [B * T, C, H, W]
    returns:
        latents: [B, C, T, H, W]
    """
    batch_size = frames.shape[0] // num_frames
    return frames.unflatten(0, (batch_size, num_frames)).permute(0, 2, 1, 3, 4)


class ConditioningEncoder(nn.Module):
    """
    Sum of the noise level embedding, the VAE identity embedding (one hot input) and the HFF
    embedding. A dropped or missing HFF is replaced by a learned null embedding.
    """

    def __init__(self) -> None:
        super().__init__()
        self.time_projection = make_embedding_mlp(TIME_FREQUENCY_DIM)
        self.vae_projection = make_embedding_mlp(len(VAE_NAMES))
        self.high_frequency_projection = make_embedding_mlp(TIME_FREQUENCY_DIM)
        self.high_frequency_null_embedding = nn.Parameter(torch.zeros(CONDITION_DIM))

    def forward(
        self,
        sigmas: torch.Tensor,
        vae_index: int,
        high_frequency_feature: torch.Tensor,
        high_frequency_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            sigmas: [B], values: [0...1], 1 is pure noise
            vae_index: index in VAE_NAMES
            high_frequency_feature: [B], values: [0...1], may be NaN where the mask is False
            high_frequency_mask: [B], dtype: torch.bool, False uses the null embedding
        returns:
            condition: [B, CONDITION_DIM]
        """
        weight_dtype = self.time_projection[0].weight.dtype
        batch_size = sigmas.shape[0]
        condition = self.time_projection(sinusoidal_embedding(sigmas).to(weight_dtype))
        vae_one_hot = functional.one_hot(
            torch.full((batch_size,), vae_index, device=sigmas.device), len(VAE_NAMES)
        )  # [B, num_vaes]
        condition = condition + self.vae_projection(vae_one_hot.to(weight_dtype))
        feature = torch.nan_to_num(high_frequency_feature.float(), nan=0.0)
        feature_condition = self.high_frequency_projection(
            sinusoidal_embedding(feature).to(weight_dtype)
        )
        return condition + torch.where(
            high_frequency_mask[:, None],
            feature_condition,
            self.high_frequency_null_embedding.to(feature_condition.dtype),
        )


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
    Multi-head self-attention inside non-overlapping windows of WINDOW_SIZE (linear in image
    area). Maps smaller than the window are attended globally. The output projection is zero
    initialized, so the module starts as the identity.
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
            features: [B, C, H, W], H and W divisible by min(WINDOW_SIZE, H, W)
        returns:
            features: [B, C, H, W]
        """
        batch_size, channels, height, width = features.shape
        window_height, window_width = min(WINDOW_SIZE, height), min(WINDOW_SIZE, width)
        heads = channels // ATTENTION_HEAD_DIM
        rows, columns = height // window_height, width // window_width
        window_tokens = window_height * window_width
        hidden = self.query_key_value(self.norm(features)).reshape(
            batch_size, 3, heads, ATTENTION_HEAD_DIM, rows, window_height, columns, window_width
        )
        hidden = hidden.permute(1, 0, 2, 4, 6, 5, 7, 3).reshape(
            3, batch_size, heads, rows * columns, window_tokens, ATTENTION_HEAD_DIM
        )  # [3, B, heads, windows, window tokens, D]
        attended = functional.scaled_dot_product_attention(hidden[0], hidden[1], hidden[2])
        attended = attended.reshape(
            batch_size, heads, rows, columns, window_height, window_width, ATTENTION_HEAD_DIM
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
    SDXL style block: conditioned residual block followed by window attention and a pointwise MLP.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.residual = ConditionedResidualBlock(in_channels, out_channels)
        self.attention = WindowAttention(out_channels)
        self.feed_forward = PointwiseFeedForward(out_channels)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C_in, H, W]
            condition: [B, CONDITION_DIM]
        returns:
            features: [B, C_out, H, W]
        """
        return self.feed_forward(self.attention(self.residual(features, condition)))


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

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C_in, H, W]
            condition: [B, CONDITION_DIM]
        returns:
            features: [B, C_out, H, W]
        """
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                features = torch.utils.checkpoint.checkpoint(
                    block, features, condition, use_reentrant=False
                )
            else:
                features = block(features, condition)
        return features


class UniversalUpscalerUNet(nn.Module):
    """
    UNet predicting the flow velocity of high resolution latents for any VAE in VAE_NAMES. The
    bilinearly upsampled low resolution latents are concatenated to the noisy latents before the
    VAE specific input adapter. Latent height and width must be 16, 32 or a multiple of
    4 * WINDOW_SIZE.
    """

    def __init__(self, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.conditioning = ConditioningEncoder()
        self.input_adapters = nn.ModuleDict(
            {
                name: nn.Conv2d(2 * channels, LEVEL_CHANNELS[0], 1)
                for name, channels in VAE_LATENT_CHANNELS.items()
            }
        )
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
        self.output_adapters = nn.ModuleDict(
            {
                name: nn.Conv2d(LEVEL_CHANNELS[0], channels, 1)
                for name, channels in VAE_LATENT_CHANNELS.items()
            }
        )
        for adapter in self.output_adapters.values():
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        latents_low_resolution: torch.Tensor,
        vae_name: str,
        high_frequency_feature: torch.Tensor,
        high_frequency_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            noisy_latents: [B, C, H, W], C = VAE_LATENT_CHANNELS[vae_name]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_low_resolution: [B, C, H / 2, W / 2]
            vae_name: key of VAE_LATENT_CHANNELS or of LATENT_SPACE_BY_VAE_NAME
            high_frequency_feature: [B], values: [0...1] or NaN
            high_frequency_mask: [B], dtype: torch.bool
        returns:
            velocity: [B, C, H, W]
        """
        vae_name = LATENT_SPACE_BY_VAE_NAME.get(vae_name, vae_name)
        condition = self.conditioning(
            sigmas, VAE_NAMES.index(vae_name), high_frequency_feature, high_frequency_mask
        )
        latents_upsampled = functional.interpolate(
            latents_low_resolution, size=noisy_latents.shape[-2:], mode="bilinear"
        )  # [B, C, H, W]
        features = self.input_adapters[vae_name](
            torch.cat([noisy_latents, latents_upsampled], dim=1)
        )

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
        return self.output_adapters[vae_name](functional.silu(self.output_norm(features)))


@torch.no_grad()
def sample_high_resolution_latents(
    model: nn.Module,
    latents_low_resolution: torch.Tensor,
    vae_name: str,
    high_frequency_feature: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    num_steps: int = NUM_SAMPLING_STEPS,
    sigma_shift: float = SAMPLING_SIGMA_SHIFT,
    seed: int = INFERENCE_SEED,
    dtype: torch.dtype = INFERENCE_DTYPE,
) -> torch.Tensor:
    """
    Euler sampling of the flow from noise to the high resolution latents. Guidance contrasts the
    HFF conditioned prediction with the null HFF prediction and needs `high_frequency_feature`.

    args:
        latents_low_resolution: [B, C, T, H, W]
        high_frequency_feature: [B], values: [0...1], None = unconditional
        guidance_scale: 1 disables guidance
    returns:
        latents: [B, C, T, H * SCALE_FACTOR, W * SCALE_FACTOR], dtype: torch.float32
    """
    batch_size, channels, num_frames, height, width = latents_low_resolution.shape
    device = latents_low_resolution.device
    frames_low_resolution = frames_to_batch(latents_low_resolution).to(dtype)  # [B*T, C, H, W]
    frame_count = frames_low_resolution.shape[0]
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (frame_count, channels, height * SCALE_FACTOR, width * SCALE_FACTOR),
        generator=generator,
        device=device,
    )  # [B*T, C, H_hr, W_hr], dtype: torch.float32

    if high_frequency_feature is None:
        feature = torch.zeros(frame_count, device=device)
        mask = torch.zeros(frame_count, dtype=torch.bool, device=device)
    else:
        feature = high_frequency_feature.to(device).float().repeat_interleave(num_frames)
        mask = torch.ones(frame_count, dtype=torch.bool, device=device)
    use_guidance = guidance_scale != 1.0 and high_frequency_feature is not None

    sampling_sigmas = shift_sigmas(
        torch.linspace(1.0, 0.0, num_steps + 1, device=device), sigma_shift
    )
    for current_sigma, next_sigma in zip(sampling_sigmas[:-1], sampling_sigmas[1:]):
        sigmas = current_sigma.expand(frame_count)
        velocity = model(latents.to(dtype), sigmas, frames_low_resolution, vae_name, feature, mask)
        if use_guidance:
            null_velocity = model(
                latents.to(dtype), sigmas, frames_low_resolution, vae_name, feature, ~mask
            )
            velocity = null_velocity + guidance_scale * (velocity - null_velocity)
        latents = latents + (next_sigma - current_sigma) * velocity.float()
    return batch_to_frames(latents, num_frames)


class SmallUniversalUpscaler(MultiVAEUpscaler):
    """
    Inference wrapper of a trained checkpoint. Training runs through scripts/run_multi_vae.py.
    """

    def __init__(self, guidance_scale: float = 1.0) -> None:
        self.guidance_scale = guidance_scale
        self.model: nn.Module | None = None

    def supported_vae_names(self) -> tuple[str, ...]:
        return VAE_NAMES + tuple(LATENT_SPACE_BY_VAE_NAME)

    def prepare_to_inference(self, weights_path: str) -> None:
        model = UniversalUpscalerUNet()
        model.load_state_dict(load_file(str(Path(weights_path))))
        self.model = model.to(DEVICE, INFERENCE_DTYPE).eval().requires_grad_(False)

    def compile(self) -> None:
        self.model = torch.compile(self.model, mode="reduce-overhead")

    @torch.no_grad()
    def upscale(
        self,
        latents: torch.Tensor,
        vae_name: str,
        high_frequency_feature: float | None = None,
    ) -> torch.Tensor:
        feature = (
            None
            if high_frequency_feature is None
            else torch.full((latents.shape[0],), high_frequency_feature)
        )
        upscaled = sample_high_resolution_latents(
            self.model, latents, vae_name, feature, guidance_scale=self.guidance_scale
        )
        return upscaled.to(latents.dtype)

    @torch.no_grad()
    def single_forward(self, latents: torch.Tensor, vae_name: str) -> torch.Tensor:
        frames = frames_to_batch(latents).to(INFERENCE_DTYPE)  # [B*T, C, H, W]
        frame_count, channels, height, width = frames.shape
        noisy_latents = torch.randn(
            (frame_count, channels, height * SCALE_FACTOR, width * SCALE_FACTOR),
            device=latents.device,
            dtype=INFERENCE_DTYPE,
        )
        return self.model(
            noisy_latents,
            torch.ones(frame_count, device=latents.device),
            frames,
            vae_name,
            torch.zeros(frame_count, device=latents.device),
            torch.zeros(frame_count, dtype=torch.bool, device=latents.device),
        )

    def get_name(self) -> str:
        return MODEL_NAME
