"""
FLUX.2 latent upscaler (autoresearch_3 recipe, run_0016, scaled): plain window transformer on the full resolution
latent grid with adaLN-zero blocks, shifted 8x8 window attention with relative position bias and QK-norm, GELU MLP
with a depthwise 3x3 conv, RMS normalized condition vector (sigma + hff, hff droppable to a learned null embedding
for classifier free guidance), learned source path and normalized residual flow parameterization.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils.checkpoint

LATENT_CHANNELS = 32
WINDOW_SIZE = 8
FREQUENCY_DIM = 256
SIGMA_SCALE = 1000.0
HFF_SCALE = 1000.0
MAX_NORM_GROUPS = 32


@dataclass(frozen=True)
class ModelConfig:
    width: int = 768
    depth: int = 16
    head_dim: int = 64
    mlp_hidden: int = 3072
    condition_dim: int = 512
    source_encoder_width: int = 512
    source_encoder_blocks: int = 2
    residual_std: tuple[float, ...] = field(default_factory=lambda: (1.0,) * LATENT_CHANNELS)


@dataclass(frozen=True)
class SamplerConfig:
    steps: int = 30
    shift: float = 2.0
    noise_scale: float = 1.0
    seed: int = 0


def shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """
    args:
        sigmas: [...], values: [0...1], 1 is pure noise
    returns:
        shifted_sigmas: [...], values: [0...1], moved towards 1 for shift > 1 (SD3)
    """
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def align_source(latents_source: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents_source: [B, C, h, w]
    returns:
        latents_aligned: [B, C, 2h, 2w], bilinear
    """
    return functional.interpolate(latents_source, scale_factor=2, mode="bilinear", align_corners=False)


def residual_scale(residual_std: tuple[float, ...], device: torch.device) -> torch.Tensor:
    """
    returns:
        scale: [1, C, 1, 1], dtype: torch.float32
    """
    return torch.tensor(residual_std, device=device, dtype=torch.float32)[None, :, None, None]


def to_residual(latents_target: torch.Tensor, latents_source: torch.Tensor, residual_std: tuple[float, ...]) -> torch.Tensor:
    """
    args:
        latents_target: [B, C, 2h, 2w]
        latents_source: [B, C, h, w]
    returns:
        residual: [B, C, 2h, 2w], per channel ~unit std, dtype: torch.float32
    """
    base = align_source(latents_source.float())
    return (latents_target.float() - base) / residual_scale(residual_std, latents_target.device)


def from_residual(residual: torch.Tensor, latents_source: torch.Tensor, residual_std: tuple[float, ...]) -> torch.Tensor:
    """
    args:
        residual: [B, C, 2h, 2w]
        latents_source: [B, C, h, w]
    returns:
        latents_target: [B, C, 2h, 2w], dtype: torch.float32
    """
    return align_source(latents_source.float()) + residual_scale(residual_std, residual.device) * residual.float()


def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(MAX_NORM_GROUPS, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=1e-6)


def sinusoidal_embedding(values: torch.Tensor, scale: float) -> torch.Tensor:
    """
    args:
        values: [N]
    returns:
        embedding: [N, FREQUENCY_DIM], dtype: torch.float32
    """
    half_dim = FREQUENCY_DIM // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=values.device, dtype=torch.float32) / half_dim
    )
    angles = values.float()[:, None] * scale * frequencies[None, :]  # [N, half_dim]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class ConditioningEncoder(nn.Module):
    """
    RMS normalized sum of MLP embeddings of the noise level and of the high frequency feature; dropped
    features use a learned null embedding.
    """

    def __init__(self, condition_dim: int) -> None:
        super().__init__()
        self.sigma_projection = self._make_projection(condition_dim)
        self.hff_projection = self._make_projection(condition_dim)
        self.null_hff = nn.Parameter(torch.zeros(condition_dim))

    @staticmethod
    def _make_projection(condition_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(FREQUENCY_DIM, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim))

    def forward(self, sigmas: torch.Tensor, hff: torch.Tensor, hff_mask: torch.Tensor) -> torch.Tensor:
        """
        args:
            sigmas: [B], values: [0...1], 1 is pure noise
            hff: [B], values: [0...1]
            hff_mask: [B], dtype: torch.bool, False = dropped (null embedding)
        returns:
            condition: [B, condition_dim]
        """
        weight_dtype = self.sigma_projection[0].weight.dtype
        sigma_embedding = self.sigma_projection(sinusoidal_embedding(sigmas, SIGMA_SCALE).to(weight_dtype))
        hff_embedding = self.hff_projection(sinusoidal_embedding(hff, HFF_SCALE).to(weight_dtype))
        hff_embedding = torch.where(hff_mask[:, None], hff_embedding, self.null_hff.to(hff_embedding.dtype)[None])
        condition = sigma_embedding + hff_embedding
        return functional.rms_norm(condition, (condition.shape[-1],))


class PlainResidualBlock(nn.Module):
    """
    Unconditioned residual block (source encoder), zero initialized last convolution.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm_first = make_group_norm(channels)
        self.conv_first = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm_second = make_group_norm(channels)
        self.conv_second = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv_second.weight)
        nn.init.zeros_(self.conv_second.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_first(functional.silu(self.norm_first(features)))
        return features + self.conv_second(functional.silu(self.norm_second(hidden)))


class SourceEncoder(nn.Module):
    """
    Learned 2x upsampling of the source latents: residual blocks at the source resolution, pixel shuffle,
    projection to the transformer width.
    """

    def __init__(self, channels: int, output_channels: int, num_blocks: int, input_channels: int = LATENT_CHANNELS) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(input_channels, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(PlainResidualBlock(channels) for _ in range(num_blocks)))
        self.upsample_norm = make_group_norm(channels)
        self.upsample = nn.Conv2d(channels, 4 * channels, 3, padding=1)
        self.output = nn.Conv2d(channels, output_channels, 1)

    def forward(self, latents_source: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents_source: [B, C, h, w]
        returns:
            features: [B, output_channels, 2h, 2w]
        """
        features = self.blocks(self.input_conv(latents_source))
        features = functional.pixel_shuffle(self.upsample(functional.silu(self.upsample_norm(features))), 2)
        return self.output(features)


def window_partition(tokens: torch.Tensor) -> torch.Tensor:
    """
    args:
        tokens: [N, H, W, D]
    returns:
        windows: [N, windows, WINDOW_SIZE * WINDOW_SIZE, D]
    """
    count, height, width, dim = tokens.shape
    rows, columns = height // WINDOW_SIZE, width // WINDOW_SIZE
    tokens = tokens.reshape(count, rows, WINDOW_SIZE, columns, WINDOW_SIZE, dim).permute(0, 1, 3, 2, 4, 5)
    return tokens.reshape(count, rows * columns, WINDOW_SIZE * WINDOW_SIZE, dim)


def window_merge(windows: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    args:
        windows: [N, windows, WINDOW_SIZE * WINDOW_SIZE, D]
    returns:
        tokens: [N, H, W, D]
    """
    count, _, _, dim = windows.shape
    rows, columns = height // WINDOW_SIZE, width // WINDOW_SIZE
    windows = windows.reshape(count, rows, columns, WINDOW_SIZE, WINDOW_SIZE, dim).permute(0, 1, 3, 2, 4, 5)
    return windows.reshape(count, height, width, dim)


def shifted_window_mask(height: int, width: int, shift: int, device: torch.device) -> torch.Tensor:
    """
    Swin mask for cyclically shifted windows: tokens attend only within their original region.

    returns:
        mask: [windows, WINDOW_SIZE ** 2, WINDOW_SIZE ** 2], dtype: torch.bool, True = attend
    """
    region = torch.zeros(height, width, device=device)
    label = 0
    for row_slice in (slice(0, -WINDOW_SIZE), slice(-WINDOW_SIZE, -shift), slice(-shift, None)):
        for column_slice in (slice(0, -WINDOW_SIZE), slice(-WINDOW_SIZE, -shift), slice(-shift, None)):
            region[row_slice, column_slice] = label
            label += 1
    region_windows = window_partition(region[None, :, :, None])[0, :, :, 0]  # [windows, window tokens]
    return region_windows[:, :, None] == region_windows[:, None, :]


class TransformerBlock(nn.Module):
    """
    DiT style block with adaLN-zero modulation: (shifted) window self-attention with relative position bias
    and QK-norm, GELU MLP with a depthwise 3x3 convolution on the hidden features.
    """

    def __init__(self, config: ModelConfig, shift: bool) -> None:
        super().__init__()
        dim = config.width
        self.head_dim = config.head_dim
        self.heads = dim // config.head_dim
        self.shift = WINDOW_SIZE // 2 if shift else 0
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.condition_dim, 6 * dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attention_output = nn.Linear(dim, dim)
        self.query_norm = nn.RMSNorm(config.head_dim, eps=1e-6)
        self.key_norm = nn.RMSNorm(config.head_dim, eps=1e-6)
        self.relative_bias = nn.Parameter(torch.zeros(self.heads, (2 * WINDOW_SIZE - 1) ** 2))
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(WINDOW_SIZE), torch.arange(WINDOW_SIZE), indexing="ij")
        ).flatten(1)  # [2, window tokens]
        relative = coordinates[:, :, None] - coordinates[:, None, :] + WINDOW_SIZE - 1
        self.register_buffer("relative_index", relative[0] * (2 * WINDOW_SIZE - 1) + relative[1], persistent=False)
        self.feed_forward_input = nn.Linear(dim, config.mlp_hidden)
        self.feed_forward_conv = nn.Conv2d(config.mlp_hidden, config.mlp_hidden, 3, padding=1, groups=config.mlp_hidden)
        self.feed_forward_output = nn.Linear(config.mlp_hidden, dim)

    def _attention(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        args:
            tokens: [N, H, W, D]
        returns:
            attended: [N, H, W, D]
        """
        count, height, width, dim = tokens.shape
        if self.shift:
            tokens = torch.roll(tokens, (-self.shift, -self.shift), dims=(1, 2))
        windows = window_partition(self.qkv(tokens))  # [N, windows, window tokens, 3D]
        num_windows, window_tokens = windows.shape[1], windows.shape[2]
        qkv = windows.reshape(count * num_windows, window_tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv = torch.stack([self.query_norm(qkv[0]), self.key_norm(qkv[1]), qkv[2]])
        bias = self.relative_bias[:, self.relative_index].to(qkv.dtype)[None]  # [1, heads, t, t]
        if self.shift:
            mask = shifted_window_mask(height, width, self.shift, tokens.device)  # [windows, t, t]
            bias = bias.masked_fill(~mask[:, None], float("-inf"))  # [windows, heads, t, t]
            bias = bias.expand(count, -1, -1, -1, -1).reshape(count * num_windows, self.heads, window_tokens, window_tokens)
        attended = functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=bias)  # [N * windows, heads, t, Dh]
        attended = attended.permute(0, 2, 1, 3).reshape(count, num_windows, window_tokens, dim)
        attended = window_merge(self.attention_output(attended), height, width)
        if self.shift:
            attended = torch.roll(attended, (self.shift, self.shift), dims=(1, 2))
        return attended

    def _feed_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.feed_forward_input(tokens).permute(0, 3, 1, 2)  # [N, F, H, W]
        hidden = self.feed_forward_conv(hidden).permute(0, 2, 3, 1)
        return self.feed_forward_output(functional.gelu(hidden, approximate="tanh"))

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        args:
            tokens: [B, H, W, D]
            condition: [B, condition_dim]
        returns:
            tokens: [B, H, W, D]
        """
        modulation = self.modulation(condition)[:, None, None, :].chunk(6, dim=-1)  # 6 x [B, 1, 1, D]
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation
        tokens = tokens + gate_a * self._attention(self.norm(tokens) * (1 + scale_a) + shift_a)
        return tokens + gate_f * self._feed_forward(self.norm(tokens) * (1 + scale_f) + shift_f)


class FluxUpscalerTransformer(nn.Module):
    """
    Velocity model of the residual flow: 3x3 convolution stem on the noisy residual and the aligned source
    latents plus learned source features, `depth` transformer blocks alternating regular and shifted windows,
    adaLN final layer and zero initialized linear head. Latent sides must be multiples of WINDOW_SIZE.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.width
        self.config = config
        self.checkpointed_blocks = 0
        self.conditioning = ConditioningEncoder(config.condition_dim)
        self.input_conv = nn.Conv2d(2 * LATENT_CHANNELS, dim, 3, padding=1)
        self.source_encoder = SourceEncoder(config.source_encoder_width, dim, config.source_encoder_blocks)
        self.blocks = nn.ModuleList(TransformerBlock(config, shift=index % 2 == 1) for index in range(config.depth))
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.condition_dim, 2 * dim))
        nn.init.zeros_(self.output_modulation[1].weight)
        nn.init.zeros_(self.output_modulation[1].bias)
        self.output_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.output = nn.Linear(dim, LATENT_CHANNELS)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def compile_blocks(self) -> None:
        for module in self.modules():
            if isinstance(module, (TransformerBlock, SourceEncoder)):
                module.compile(dynamic=False)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        sigmas: torch.Tensor,
        latents_source: torch.Tensor,
        hff: torch.Tensor,
        hff_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            noisy_residual: [B, C, H, W]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_source: [B, C, H / 2, W / 2], normalized FLUX.2 latents
            hff: [B], values: [0...1], high frequency feature of the target
            hff_mask: [B], dtype: torch.bool, False = hff dropped
        returns:
            velocity: [B, C, H, W] (noise - residual)
        """
        return self.forward_with_condition(noisy_residual, self.conditioning(sigmas, hff, hff_mask), latents_source)

    def forward_with_condition(
        self, noisy_residual: torch.Tensor, condition: torch.Tensor, latents_source: torch.Tensor
    ) -> torch.Tensor:
        """
        args:
            noisy_residual: [B, C, H, W]
            condition: [B, condition_dim], output of the conditioning encoder
            latents_source: [B, C, H / 2, W / 2], normalized FLUX.2 latents
        returns:
            velocity: [B, C, H, W] (noise - residual)
        """
        stacked = torch.cat([noisy_residual, align_source(latents_source).to(noisy_residual.dtype)], dim=1)
        features = self.input_conv(stacked) + self.source_encoder(latents_source.to(noisy_residual.dtype))  # [B, D, H, W]
        tokens = features.permute(0, 2, 3, 1)  # [B, H, W, D]
        for index, block in enumerate(self.blocks):
            if self.training and index < self.checkpointed_blocks:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, condition, use_reentrant=False)
            else:
                tokens = block(tokens, condition)
        shift, scale = self.output_modulation(condition)[:, None, None, :].chunk(2, dim=-1)
        velocity = self.output(self.output_norm(tokens) * (1 + scale) + shift)  # [B, H, W, C]
        return velocity.permute(0, 3, 1, 2)


@torch.no_grad()
def upscale(
    model: FluxUpscalerTransformer,
    latents_source: torch.Tensor,
    hff: torch.Tensor,
    sampler: SamplerConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Euler sampling of the residual flow.

    args:
        latents_source: [B, C, h, w], normalized FLUX.2 latents, h and w multiples of WINDOW_SIZE / 2
        hff: [B], values: [0...1], NaN = no hff (null embedding)
    returns:
        latents: [B, C, 2h, 2w], dtype: torch.float32
    """
    batch_size, channels, height, width = latents_source.shape
    device = latents_source.device
    generator = torch.Generator(device=device).manual_seed(sampler.seed)
    samples = sampler.noise_scale * torch.randn((batch_size, channels, 2 * height, 2 * width), generator=generator, device=device)
    source = latents_source.to(dtype)
    hff = hff.to(device).float()
    hff_mask = ~torch.isnan(hff)
    hff = torch.nan_to_num(hff, nan=0.0)
    sigmas = shift_sigmas(torch.linspace(1.0, 0.0, sampler.steps + 1, device=device), sampler.shift)
    for current_sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        velocity = model(samples.to(dtype), current_sigma.expand(batch_size), source, hff, hff_mask)
        samples = samples + (next_sigma - current_sigma) * velocity.float()
    return from_residual(samples, latents_source, model.config.residual_std)
