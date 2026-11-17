"""
Production Wan 2.1 latent upscaler (autoresearch_2 recipe, scaled): plain window transformer on the full
resolution latent grid with adaLN-zero blocks, shifted 8x8 window attention with relative position bias and
QK-norm, causal gated depthwise temporal convolution, GELU MLP with a depthwise 3x3 conv, RMS normalized
condition vector, learned source path and residual flow parameterization.
The high frequency feature (hff) can be dropped to a learned null embedding (classifier free guidance).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils.checkpoint

LATENT_CHANNELS = 16
WINDOW_SIZE = 8
FREQUENCY_DIM = 256
SIGMA_SCALE = 1000.0
HFF_SCALE = 1000.0
MAX_NORM_GROUPS = 32
TEMPORAL_KERNEL = 2

RESIDUAL_STD_SPATIAL = (0.129, 0.221, 0.170, 0.153, 0.334, 0.249, 0.156, 0.212, 0.141, 0.161, 0.131, 0.228, 0.207, 0.338, 0.140, 0.214)
RESIDUAL_STD_TEMPORAL = (0.211, 0.360, 0.248, 0.285, 0.417, 0.349, 0.236, 0.299, 0.222, 0.302, 0.218, 0.387, 0.375, 0.469, 0.244, 0.300)


@dataclass(frozen=True)
class ModelConfig:
    width: int = 1024
    depth: int = 24
    head_dim: int = 64
    mlp_hidden: int = 4096
    condition_dim: int = 512
    source_encoder_width: int = 512
    source_encoder_blocks: int = 2


@dataclass(frozen=True)
class SamplerConfig:
    steps: int = 20
    shift: float = 4.0
    noise_scale_spatial: float = 1.0
    noise_scale_temporal: float = 0.8
    seed: int = 0


def shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """
    args:
        sigmas: [...], values: [0...1], 1 is pure noise
    returns:
        shifted_sigmas: [...], values: [0...1], moved towards 1 for shift > 1 (SD3)
    """
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def causal_frame_indices(source_frames: int, target_frames: int, device: torch.device) -> torch.Tensor:
    """
    Source latent frame of each target latent frame for the causal Wan VAE (1 + 4k pixel frames):
    target frame j <- source frame ceil(j / 2).

    returns:
        indices: [T_hr], dtype: torch.long
    """
    if target_frames == source_frames:
        return torch.arange(target_frames, device=device)
    assert target_frames - 1 == 2 * (source_frames - 1), (source_frames, target_frames)
    return torch.div(torch.arange(target_frames, device=device) + 1, 2, rounding_mode="floor")


def align_source(latents_source: torch.Tensor, target_size: tuple[int, int, int]) -> torch.Tensor:
    """
    Causal frame index alignment and bilinear spatial resize of the source latents to the target grid.

    args:
        latents_source: [B, C, T_lr, H_lr, W_lr]
        target_size: (T_hr, H_hr, W_hr)
    returns:
        latents_aligned: [B, C, T_hr, H_hr, W_hr]
    """
    if target_size[0] != latents_source.shape[2]:
        indices = causal_frame_indices(latents_source.shape[2], target_size[0], latents_source.device)
        latents_source = latents_source.index_select(2, indices)
    if tuple(target_size[1:]) != tuple(latents_source.shape[3:]):
        latents_source = functional.interpolate(latents_source, size=target_size, mode="trilinear", align_corners=False)
    return latents_source


def residual_scale(source_shape: torch.Size, target_size: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """
    Per channel residual std: spatial table if the spatial size changes, temporal table otherwise.

    returns:
        scale: [1, C, 1, 1, 1], dtype: torch.float32
    """
    table = RESIDUAL_STD_SPATIAL if tuple(target_size[1:]) != tuple(source_shape[3:]) else RESIDUAL_STD_TEMPORAL
    return torch.tensor(table, device=device, dtype=torch.float32)[None, :, None, None, None]


def to_residual(latents_target: torch.Tensor, latents_source: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents_target: [B, C, T_hr, H_hr, W_hr]
        latents_source: [B, C, T_lr, H_lr, W_lr]
    returns:
        residual: [B, C, T_hr, H_hr, W_hr], per channel ~unit std
    """
    target_size = tuple(latents_target.shape[2:])
    base = align_source(latents_source.float(), target_size)
    return (latents_target.float() - base) / residual_scale(latents_source.shape, target_size, latents_target.device)


def from_residual(residual: torch.Tensor, latents_source: torch.Tensor) -> torch.Tensor:
    """
    args:
        residual: [B, C, T_hr, H_hr, W_hr]
        latents_source: [B, C, T_lr, H_lr, W_lr]
    returns:
        latents_target: [B, C, T_hr, H_hr, W_hr], dtype: torch.float32
    """
    target_size = tuple(residual.shape[2:])
    base = align_source(latents_source.float(), target_size)
    return base + residual_scale(latents_source.shape, target_size, residual.device) * residual.float()


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
    Unconditioned per frame residual block (source encoder), zero initialized last convolution.
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
    Learned upsampling of the source latents: residual blocks at the source resolution, pixel shuffle 2x
    spatial upsampling, causal phase temporal upsampling (target frames 2k - 1 and 2k from source frames
    k and k - 1), projection to the transformer width.
    """

    def __init__(self, channels: int, output_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(LATENT_CHANNELS, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(PlainResidualBlock(channels) for _ in range(num_blocks)))
        self.spatial_norm = make_group_norm(channels)
        self.spatial_upsample = nn.Conv2d(channels, 4 * channels, 3, padding=1)
        self.temporal_phases = nn.Conv2d(2 * channels, 2 * channels, 1)
        nn.init.zeros_(self.temporal_phases.weight)
        nn.init.zeros_(self.temporal_phases.bias)
        self.output = nn.Conv2d(channels, output_channels, 1)

    def forward(self, latents_source: torch.Tensor, target_size: tuple[int, int, int]) -> torch.Tensor:
        """
        args:
            latents_source: [B, C, T_lr, H_lr, W_lr]
            target_size: (T_hr, H_hr, W_hr)
        returns:
            features: [B * T_hr, output_channels, H_hr, W_hr]
        """
        batch_size, _, source_frames, source_height, source_width = latents_source.shape
        target_frames, target_height, target_width = target_size
        folded = latents_source.permute(0, 2, 1, 3, 4).reshape(batch_size * source_frames, LATENT_CHANNELS, source_height, source_width)
        features = self.blocks(self.input_conv(folded))  # [B * T_lr, D, h, w]
        if (target_height, target_width) != (source_height, source_width):
            assert target_height == 2 * source_height and target_width == 2 * source_width
            features = functional.pixel_shuffle(self.spatial_upsample(functional.silu(self.spatial_norm(features))), 2)
        channels = features.shape[1]
        if target_frames != source_frames:
            assert target_frames - 1 == 2 * (source_frames - 1)
            features = features.reshape(batch_size, source_frames, channels, target_height, target_width)
            previous = torch.cat([features[:, :1], features[:, :-1]], dim=1)  # frame k - 1 (replicated at 0)
            pair = torch.cat([features, previous], dim=2).reshape(batch_size * source_frames, 2 * channels, target_height, target_width)
            phases = self.temporal_phases(pair).reshape(batch_size, source_frames, 2, channels, target_height, target_width)
            odd = features[:, 1:] + phases[:, 1:, 0]  # target frames 2k - 1
            even = features[:, 1:] + phases[:, 1:, 1]  # target frames 2k
            rest = torch.stack([odd, even], dim=2).reshape(batch_size, 2 * (source_frames - 1), channels, target_height, target_width)
            features = torch.cat([features[:, :1], rest], dim=1).reshape(batch_size * target_frames, channels, target_height, target_width)
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
    and QK-norm, causal gated depthwise temporal convolution (skipped for single frames), GELU MLP with a
    depthwise 3x3 convolution on the hidden features.
    """

    def __init__(self, config: ModelConfig, shift: bool) -> None:
        super().__init__()
        dim = config.width
        self.head_dim = config.head_dim
        self.heads = dim // config.head_dim
        self.shift = WINDOW_SIZE // 2 if shift else 0
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.condition_dim, 9 * dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.spatial_qkv = nn.Linear(dim, 3 * dim)
        self.spatial_output = nn.Linear(dim, dim)
        self.query_norm = nn.RMSNorm(config.head_dim, eps=1e-6)
        self.key_norm = nn.RMSNorm(config.head_dim, eps=1e-6)
        self.relative_bias = nn.Parameter(torch.zeros(self.heads, (2 * WINDOW_SIZE - 1) ** 2))
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(WINDOW_SIZE), torch.arange(WINDOW_SIZE), indexing="ij")
        ).flatten(1)  # [2, window tokens]
        relative = coordinates[:, :, None] - coordinates[:, None, :] + WINDOW_SIZE - 1
        self.register_buffer("relative_index", relative[0] * (2 * WINDOW_SIZE - 1) + relative[1], persistent=False)
        self.temporal_input = nn.Linear(dim, 2 * dim)
        self.temporal_weight = nn.Parameter(torch.zeros(TEMPORAL_KERNEL, dim))  # [k, D], k = 0 is the current frame
        with torch.no_grad():
            self.temporal_weight[0].fill_(1.0)
            self.temporal_weight[1:].normal_(0.0, 0.5)
        self.temporal_output = nn.Linear(dim, dim)
        self.feed_forward_input = nn.Linear(dim, config.mlp_hidden)
        self.feed_forward_conv = nn.Conv2d(config.mlp_hidden, config.mlp_hidden, 3, padding=1, groups=config.mlp_hidden)
        self.feed_forward_output = nn.Linear(config.mlp_hidden, dim)

    def _spatial_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        args:
            tokens: [N, H, W, D]
        returns:
            attended: [N, H, W, D]
        """
        count, height, width, dim = tokens.shape
        if self.shift:
            tokens = torch.roll(tokens, (-self.shift, -self.shift), dims=(1, 2))
        windows = window_partition(self.spatial_qkv(tokens))  # [N, windows, window tokens, 3D]
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
        attended = window_merge(self.spatial_output(attended), height, width)
        if self.shift:
            attended = torch.roll(attended, (self.shift, self.shift), dims=(1, 2))
        return attended

    def _temporal_mixing(self, tokens: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Causal gated depthwise temporal convolution: out_t = W_o (silu(g_t) * sum_k w_k * v_{t - k}).

        args:
            tokens: [B * T, H, W, D]
        returns:
            mixed: [B * T, H, W, D]
        """
        count, height, width, dim = tokens.shape
        batch_size = count // num_frames
        gate, value = self.temporal_input(tokens).reshape(batch_size, num_frames, height, width, 2 * dim).chunk(2, dim=-1)
        padded = functional.pad(value, (0, 0, 0, 0, 0, 0, TEMPORAL_KERNEL - 1, 0))  # [B, T + k - 1, H, W, D]
        mixed = sum(
            self.temporal_weight[offset] * padded[:, TEMPORAL_KERNEL - 1 - offset : TEMPORAL_KERNEL - 1 - offset + num_frames]
            for offset in range(TEMPORAL_KERNEL)
        )
        mixed = (functional.silu(gate) * mixed).reshape(count, height, width, dim)
        return self.temporal_output(mixed)

    def _feed_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.feed_forward_input(tokens).permute(0, 3, 1, 2)  # [N, F, H, W]
        hidden = self.feed_forward_conv(hidden).permute(0, 2, 3, 1)
        return self.feed_forward_output(functional.gelu(hidden, approximate="tanh"))

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        args:
            tokens: [B * T, H, W, D]
            condition: [B * T, condition_dim]
            num_frames: T
        returns:
            tokens: [B * T, H, W, D]
        """
        modulation = self.modulation(condition)[:, None, None, :].chunk(9, dim=-1)  # 9 x [N, 1, 1, D]
        shift_a, scale_a, gate_a, shift_t, scale_t, gate_t, shift_f, scale_f, gate_f = modulation
        tokens = tokens + gate_a * self._spatial_attention(self.norm(tokens) * (1 + scale_a) + shift_a)
        if num_frames > 1:
            tokens = tokens + gate_t * self._temporal_mixing(self.norm(tokens) * (1 + scale_t) + shift_t, num_frames)
        return tokens + gate_f * self._feed_forward(self.norm(tokens) * (1 + scale_f) + shift_f)


class WanUpscalerTransformer(nn.Module):
    """
    Velocity model of the residual flow: 3x3 convolution stem on the noisy residual and the aligned source
    latents plus learned source features, `depth` transformer blocks alternating regular and shifted windows,
    adaLN final layer and zero initialized linear head.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.width
        self.config = config
        self.checkpointed_blocks = 0
        self.conditioning = ConditioningEncoder(config.condition_dim)
        self.frame_projection = nn.Sequential(nn.Linear(FREQUENCY_DIM, dim), nn.SiLU(), nn.Linear(dim, dim))
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
            noisy_residual: [B, C, T, H, W]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_source: [B, C, T_lr, H_lr, W_lr], normalized Wan 2.1 latents
            hff: [B], values: [0...1], high frequency feature of the target
            hff_mask: [B], dtype: torch.bool, False = hff dropped
        returns:
            velocity: [B, C, T, H, W] (noise - residual)
        """
        batch_size, channels, num_frames, height, width = noisy_residual.shape
        condition = self.conditioning(sigmas, hff, hff_mask).repeat_interleave(num_frames, dim=0)  # [B * T, D_c]
        latents_aligned = align_source(latents_source, (num_frames, height, width))
        stacked = torch.cat([noisy_residual, latents_aligned.to(noisy_residual.dtype)], dim=1)
        folded = stacked.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 2 * channels, height, width)
        features = self.input_conv(folded)  # [B * T, D, H, W]
        features = features + self.source_encoder(latents_source.to(features.dtype), (num_frames, height, width))
        frame_indices = torch.arange(num_frames, device=features.device).repeat(batch_size)
        frame_embedding = self.frame_projection(sinusoidal_embedding(frame_indices, 1.0).to(features.dtype))
        tokens = features.permute(0, 2, 3, 1) + frame_embedding[:, None, None, :]  # [B * T, H, W, D]
        for index, block in enumerate(self.blocks):
            if self.training and index < self.checkpointed_blocks:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, condition, num_frames, use_reentrant=False)
            else:
                tokens = block(tokens, condition, num_frames)
        shift, scale = self.output_modulation(condition)[:, None, None, :].chunk(2, dim=-1)
        velocity = self.output(self.output_norm(tokens) * (1 + scale) + shift)  # [B * T, H, W, C]
        if self.training:
            anchor = sum(parameter.reshape(-1)[0] for parameter in self.parameters())
            velocity = velocity + 0.0 * anchor.to(velocity.dtype)
        return velocity.reshape(batch_size, num_frames, height, width, channels).permute(0, 4, 1, 2, 3)


@torch.no_grad()
def upscale(
    model: nn.Module,
    latents_source: torch.Tensor,
    hff: torch.Tensor,
    output_size: tuple[int, int, int],
    sampler: SamplerConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Euler sampling of the residual flow; the first latent frame is copied from the source for temporal-only
    upscaling (identical single pixel frame in the causal VAE).

    args:
        latents_source: [B, C, T_lr, H_lr, W_lr], normalized Wan 2.1 latents
        hff: [B], values: [0...1], NaN = no hff (null embedding)
        output_size: (T_hr, H_hr, W_hr), 2x in space and / or 2x in time (1 + 2 (T_lr - 1) frames)
    returns:
        latents: [B, C, T_hr, H_hr, W_hr], dtype: torch.float32
    """
    batch_size, channels = latents_source.shape[:2]
    device = latents_source.device
    generator = torch.Generator(device=device).manual_seed(sampler.seed)
    spatial = tuple(output_size[1:]) != tuple(latents_source.shape[3:])
    noise_scale = sampler.noise_scale_spatial if spatial else sampler.noise_scale_temporal
    samples = noise_scale * torch.randn((batch_size, channels, *output_size), generator=generator, device=device)
    source = latents_source.to(dtype)
    hff = hff.to(device).float()
    hff_mask = ~torch.isnan(hff)
    hff = torch.nan_to_num(hff, nan=0.0)
    sigmas = shift_sigmas(torch.linspace(1.0, 0.0, sampler.steps + 1, device=device), sampler.shift)
    for current_sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        velocity = model(samples.to(dtype), current_sigma.expand(batch_size), source, hff, hff_mask)
        samples = samples + (next_sigma - current_sigma) * velocity.float()
    latents = from_residual(samples, latents_source)
    if not spatial:
        latents[:, :, 0] = latents_source[:, :, 0].float()
    return latents
