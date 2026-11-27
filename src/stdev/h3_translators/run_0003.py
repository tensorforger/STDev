import math
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils.checkpoint
import vaezip
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from stdev.h3_translators.base import Translator

RUN_NAME = "run_0003"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "h3_flux2_4klsdb_40k"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "trained_models" / "h3" / RUN_NAME
TENSORBOARD_DIRECTORY = REPOSITORY_ROOT / "tensorboard" / "h3" / RUN_NAME
MODEL_FILE_NAME = "model.safetensors"
# Least squares map from a 3x3 H3 token neighbourhood to the 2x2 FLUX.2 latent patch of the centre token (test R2 0.36)
# with the per channel mean / std of its residual (dev/autoresearch_6/linear_base_fit.py).
LINEAR_BASE_PATH = REPOSITORY_ROOT / "src" / "stdev" / "h3_translators" / "assets" / "linear_base_k3.pt"
H3_VAE_PATH = REPOSITORY_ROOT / "models" / "MiniMax-H3-Image" / "vae_diffusers"
STEPPE_CHECKPOINT_PATH = REPOSITORY_ROOT / "models" / "Steppe-Flux.2" / "steppe_flux_2_200M_base.safetensors"

DEVICE = "cuda"
SEED = 0
H3_CHANNELS = 24
FLUX_CHANNELS = 32
SOURCE_KEY = "latents_h3"
TARGET_KEY = "latents_flux2"

# "direct": the flow generates the FLUX.2 latent x; "linear_residual": it generates
# r = (x - linear_base(h3) - RESIDUAL_MEAN) / RESIDUAL_STD (the analogue of the Steppe residual over the bilinear source).
PARAMETERIZATION = "linear_residual"
# Initialization: None (scratch) or "steppe_flux_2_200m" (all shape compatible tensors of the Steppe-Flux.2 base model;
# its 64 channel stem sees [noisy sample, linear base] instead of [noisy residual, bilinear FLUX.2 source]).
INITIALIZATION = None

# Whole model compile, or with GRADIENT_CHECKPOINTING per block compile with checkpointing around the compiled blocks.
# Architecture: Steppe-Flux.2 window transformer (autoresearch_3 recipe) on the FLUX.2 grid; DiT style adaLN-zero blocks,
# shifted 8x8 windows with relative position bias and QK-norm, GELU MLP with a depthwise 3x3 conv, RMS normed condition
# vector. The stem sees the noisy sample and the linear base estimate; the source path (residual blocks on the H3 grid,
# pixel shuffle) learns the 2x upsampling of the H3 latent.
TRANSFORMER_WIDTH = 384
TRANSFORMER_DEPTH = 12
TRANSFORMER_HEAD_DIM = 64
MLP_HIDDEN = 4 * TRANSFORMER_WIDTH
CONDITION_DIM = 512
SOURCE_ENCODER_WIDTH = 384
SOURCE_ENCODER_BLOCKS = 2
FREQUENCY_DIM = 256
SIGMA_SCALE = 1000.0
MAX_NORM_GROUPS = 32
WINDOW_SIZE = 8
GRADIENT_CHECKPOINTING = False
# Frozen H3-Image decoder features: the first H3_FEATURE_BLOCKS ViT blocks (0 = off) on non-overlapping 16x16 token tiles
# (256 px, the released tile size), layer normed and added to the source path through a zero initialized projection.
H3_FEATURE_BLOCKS = 8
H3_TILE_TOKENS = 16
H3_FEATURE_DIM = 1024

TRAINING_TIME_SECONDS = 30 * 60
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75)
SMOKE_STEPS = 3
BATCH_SIZE = 4
NUM_WORKERS = 6
PREFETCH_FACTOR = 4
LEARNING_RATE = 1.7e-3
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
DECAY_FRACTION = 0.5
GRADIENT_CLIP_NORM = 1.0
TRAINING_SIGMA_SHIFT = 1.0
LOG_INTERVAL_STEPS = 100
TRAINING_DTYPE = torch.bfloat16
TRAINING_COMPILE_MODE = "max-autotune-no-cudagraphs"

NUM_SAMPLING_STEPS = 30
SAMPLING_SIGMA_SHIFT = 2.0
INITIAL_NOISE_SCALE = 1.0
INFERENCE_SEED = 0
INFERENCE_DTYPE = torch.bfloat16


class LatentPairDataset(Dataset):
    """
    Infinite stream of (FLUX.2 target, H3 source) pairs, deterministic for an index.
    """

    def __init__(self) -> None:
        self.block_paths = sorted(DATASET_DIRECTORY.glob("block_*.vz.safetensors"))

    def __len__(self) -> int:
        return 1_000_000_000

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
            latents_target: [32, 128, 128], dtype: torch.float32
            latents_source: [24, 64, 64], dtype: torch.float32
        """
        random_generator = np.random.default_rng([SEED, index])
        encoded_block = vaezip.load(self.block_paths[int(random_generator.integers(0, len(self.block_paths)))])
        latents_target = encoded_block[TARGET_KEY].decode(dtype=torch.float32)[0]
        latents_source = encoded_block[SOURCE_KEY].decode(dtype=torch.float32)[0]
        return latents_target, latents_source


def iterate_batches(data_loader: DataLoader) -> Iterator[tuple[torch.Tensor, ...]]:
    while True:
        yield from data_loader


def shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """
    args:
        sigmas: [...], values: [0...1], 1 is pure noise
    returns:
        shifted_sigmas: [...], values: [0...1], moved towards 1 for shift > 1 (SD3)
    """
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def sample_training_sigmas(batch_size: int, device: str) -> torch.Tensor:
    """
    returns:
        sigmas: [B], values: [0...1], logit-normal distributed and shifted
    """
    return shift_sigmas(torch.sigmoid(torch.randn(batch_size, device=device)), TRAINING_SIGMA_SHIFT)


class LinearBase(nn.Module):
    """
    Fixed least squares estimate of the FLUX.2 latent from the H3 latent: 3x3 convolution (replicate padding) to the
    four FLUX.2 tokens of every H3 token, pixel shuffle. Holds the residual statistics.
    """

    def __init__(self) -> None:
        super().__init__()
        fit = torch.load(LINEAR_BASE_PATH)
        kernel = fit["kernel"]
        weight = fit["weight"]  # [24 * k * k + 1, 128]
        self.padding = kernel // 2
        self.register_buffer("weight", weight[:-1].T.reshape(4 * FLUX_CHANNELS, H3_CHANNELS, kernel, kernel).contiguous())
        self.register_buffer("bias", weight[-1].contiguous())
        self.register_buffer("residual_mean", fit["residual_mean"].view(1, FLUX_CHANNELS, 1, 1))
        self.register_buffer("residual_std", fit["residual_std"].view(1, FLUX_CHANNELS, 1, 1))

    def forward(self, latents_source: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents_source: [B, 24, h, w], normalized H3 latents
        returns:
            latents_base: [B, 32, 2h, 2w], FLUX.2 latent estimate, same dtype
        """
        padded = functional.pad(latents_source, [self.padding] * 4, mode="replicate")
        patches = functional.conv2d(padded, self.weight.to(padded.dtype), self.bias.to(padded.dtype))
        return functional.pixel_shuffle(patches, 2)


def to_flow_space(latents_target: torch.Tensor, latents_base: torch.Tensor, linear_base: LinearBase) -> torch.Tensor:
    """
    args:
        latents_target: [B, 32, H, W]
        latents_base: [B, 32, H, W]
    returns:
        flow_target: [B, 32, H, W], the sample the flow generates
    """
    if PARAMETERIZATION == "direct":
        return latents_target
    return (latents_target - latents_base - linear_base.residual_mean) / linear_base.residual_std


def from_flow_space(samples: torch.Tensor, latents_base: torch.Tensor, linear_base: LinearBase) -> torch.Tensor:
    """
    args:
        samples: [B, 32, H, W], generated flow samples, dtype: torch.float32
        latents_base: [B, 32, H, W], dtype: torch.float32
    returns:
        latents: [B, 32, H, W], normalized FLUX.2 latents
    """
    if PARAMETERIZATION == "direct":
        return samples
    return latents_base + linear_base.residual_mean + linear_base.residual_std * samples


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
    RMS normalized MLP embedding of the noise level plus a learned constant (the null hff embedding of Steppe).
    """

    def __init__(self) -> None:
        super().__init__()
        self.sigma_projection = nn.Sequential(
            nn.Linear(FREQUENCY_DIM, CONDITION_DIM), nn.SiLU(), nn.Linear(CONDITION_DIM, CONDITION_DIM)
        )
        self.null_hff = nn.Parameter(torch.zeros(CONDITION_DIM))

    def forward(self, sigmas: torch.Tensor) -> torch.Tensor:
        """
        args:
            sigmas: [B], values: [0...1], 1 is pure noise
        returns:
            condition: [B, CONDITION_DIM]
        """
        weight_dtype = self.sigma_projection[0].weight.dtype
        condition = self.sigma_projection(sinusoidal_embedding(sigmas, SIGMA_SCALE).to(weight_dtype))
        condition = condition + self.null_hff.to(condition.dtype)[None]
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
    Learned 2x upsampling of the H3 latents (optionally plus frozen H3 decoder features): residual blocks on the H3
    grid, pixel shuffle, projection to the transformer width.
    """

    def __init__(self) -> None:
        super().__init__()
        channels = SOURCE_ENCODER_WIDTH
        self.input_conv = nn.Conv2d(H3_CHANNELS, channels, 3, padding=1)
        self.feature_projection = None
        if H3_FEATURE_BLOCKS:
            self.feature_projection = nn.Conv2d(H3_FEATURE_DIM, channels, 1)
            nn.init.zeros_(self.feature_projection.weight)
            nn.init.zeros_(self.feature_projection.bias)
        self.blocks = nn.Sequential(*(PlainResidualBlock(channels) for _ in range(SOURCE_ENCODER_BLOCKS)))
        self.upsample_norm = make_group_norm(channels)
        self.upsample = nn.Conv2d(channels, 4 * channels, 3, padding=1)
        self.output = nn.Conv2d(channels, TRANSFORMER_WIDTH, 1)

    def forward(self, latents_source: torch.Tensor, h3_features: torch.Tensor | None) -> torch.Tensor:
        """
        args:
            latents_source: [B, 24, h, w]
            h3_features: [B, 1024, h, w] layer normed H3 decoder features, or None
        returns:
            features: [B, TRANSFORMER_WIDTH, 2h, 2w]
        """
        features = self.input_conv(latents_source)
        if self.feature_projection is not None:
            features = features + self.feature_projection(h3_features.to(features.dtype))
        features = self.blocks(features)
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
    and QK-norm, GELU MLP with a depthwise 3x3 convolution on the hidden features (Steppe-Flux.2 layout).
    """

    def __init__(self, shift: bool) -> None:
        super().__init__()
        dim = TRANSFORMER_WIDTH
        self.heads = dim // TRANSFORMER_HEAD_DIM
        self.shift = WINDOW_SIZE // 2 if shift else 0
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(CONDITION_DIM, 6 * dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attention_output = nn.Linear(dim, dim)
        self.query_norm = nn.RMSNorm(TRANSFORMER_HEAD_DIM, eps=1e-6)
        self.key_norm = nn.RMSNorm(TRANSFORMER_HEAD_DIM, eps=1e-6)
        self.relative_bias = nn.Parameter(torch.zeros(self.heads, (2 * WINDOW_SIZE - 1) ** 2))
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(WINDOW_SIZE), torch.arange(WINDOW_SIZE), indexing="ij")
        ).flatten(1)  # [2, window tokens]
        relative = coordinates[:, :, None] - coordinates[:, None, :] + WINDOW_SIZE - 1
        self.register_buffer("relative_index", relative[0] * (2 * WINDOW_SIZE - 1) + relative[1], persistent=False)
        self.feed_forward_input = nn.Linear(dim, MLP_HIDDEN)
        self.feed_forward_conv = nn.Conv2d(MLP_HIDDEN, MLP_HIDDEN, 3, padding=1, groups=MLP_HIDDEN)
        self.feed_forward_output = nn.Linear(MLP_HIDDEN, dim)

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
        qkv = windows.reshape(count * num_windows, window_tokens, 3, self.heads, TRANSFORMER_HEAD_DIM).permute(
            2, 0, 3, 1, 4
        )  # [3, N * windows, heads, t, Dh]
        qkv = torch.stack([self.query_norm(qkv[0]), self.key_norm(qkv[1]), qkv[2]])
        bias = self.relative_bias[:, self.relative_index].to(qkv.dtype)[None]  # [1, heads, t, t]
        if self.shift:
            mask = shifted_window_mask(height, width, self.shift, tokens.device)  # [windows, t, t]
            bias = bias.masked_fill(~mask[:, None], float("-inf"))  # [windows, heads, t, t]
            bias = bias.expand(count, -1, -1, -1, -1).reshape(count * num_windows, self.heads, window_tokens, window_tokens)
        attended = functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=bias)
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
            condition: [B, CONDITION_DIM]
        returns:
            tokens: [B, H, W, D]
        """
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(condition)[:, None, None, :].chunk(6, dim=-1)
        tokens = tokens + gate_a * self._attention(self.norm(tokens) * (1 + scale_a) + shift_a)
        return tokens + gate_f * self._feed_forward(self.norm(tokens) * (1 + scale_f) + shift_f)


class TranslatorTransformer(nn.Module):
    """
    Velocity model: 3x3 convolution stem on [noisy sample, linear base] plus the learned source path, TRANSFORMER_DEPTH
    blocks alternating regular and shifted windows, adaLN final layer and zero initialized linear head.
    """

    def __init__(self) -> None:
        super().__init__()
        dim = TRANSFORMER_WIDTH
        self.linear_base = LinearBase()
        self.conditioning = ConditioningEncoder()
        self.input_conv = nn.Conv2d(2 * FLUX_CHANNELS, dim, 3, padding=1)
        self.source_encoder = SourceEncoder()
        self.blocks = nn.ModuleList(TransformerBlock(shift=index % 2 == 1) for index in range(TRANSFORMER_DEPTH))
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(CONDITION_DIM, 2 * dim))
        nn.init.zeros_(self.output_modulation[1].weight)
        nn.init.zeros_(self.output_modulation[1].bias)
        self.output_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.output = nn.Linear(dim, FLUX_CHANNELS)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        noisy_samples: torch.Tensor,
        sigmas: torch.Tensor,
        latents_source: torch.Tensor,
        latents_base: torch.Tensor,
        h3_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        args:
            noisy_samples: [B, 32, H, W]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_source: [B, 24, H / 2, W / 2], normalized H3 latents
            latents_base: [B, 32, H, W], linear base estimate
            h3_features: [B, 1024, H / 2, W / 2] or None
        returns:
            velocity: [B, 32, H, W] (noise - flow sample)
        """
        condition = self.conditioning(sigmas)
        stacked = torch.cat([noisy_samples, latents_base.to(noisy_samples.dtype)], dim=1)  # [B, 64, H, W]
        features = self.input_conv(stacked) + self.source_encoder(latents_source.to(noisy_samples.dtype), h3_features)
        tokens = features.permute(0, 2, 3, 1)  # [B, H, W, D]
        for block in self.blocks:
            if GRADIENT_CHECKPOINTING and self.training:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, condition, use_reentrant=False)
            else:
                tokens = block(tokens, condition)
        shift, scale = self.output_modulation(condition)[:, None, None, :].chunk(2, dim=-1)
        velocity = self.output(self.output_norm(tokens) * (1 + scale) + shift)  # [B, H, W, 32]
        return velocity.permute(0, 3, 1, 2)


class H3FeatureExtractor(nn.Module):
    """
    Frozen first H3_FEATURE_BLOCKS blocks of the H3-Image ViT decoder, run on non-overlapping H3_TILE_TOKENS tiles
    exactly like the released tiled decode (register tokens, zero token, RoPE on tile normalized coordinates).
    """

    def __init__(self) -> None:
        super().__init__()
        from diffusers import AutoencoderKLMiniMaxH3Image

        vae = AutoencoderKLMiniMaxH3Image.from_pretrained(str(H3_VAE_PATH), torch_dtype=torch.bfloat16)
        decoder = vae.decoder
        self.post_quant_conv = vae.post_quant_conv
        self.proj_in = decoder.proj_in
        self.register_tokens = decoder.register_tokens
        self.rope = decoder.rope
        self.blocks = nn.ModuleList(decoder.transformer_blocks[:H3_FEATURE_BLOCKS])
        self.register_buffer("latents_mean", torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("latents_std", torch.tensor(vae.config.latents_std).view(1, -1, 1, 1), persistent=False)
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, latents_source: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents_source: [B, 24, h, w], normalized H3 latents, h and w multiples of H3_TILE_TOKENS
        returns:
            features: [B, 1024, h, w], layer normed per token, dtype: torch.bfloat16
        """
        batch_size, channels, height, width = latents_source.shape
        tile = H3_TILE_TOKENS
        raw = latents_source.float() * self.latents_std + self.latents_mean
        raw = self.post_quant_conv(raw.to(self.post_quant_conv.weight.dtype))
        rows, columns = height // tile, width // tile
        tiles = raw.reshape(batch_size, channels, rows, tile, columns, tile).permute(0, 2, 4, 3, 5, 1)
        tiles = tiles.reshape(batch_size * rows * columns, tile * tile, channels)  # [N, t, 24]
        hidden = self.proj_in(tiles.to(self.proj_in.weight.dtype))
        count = hidden.shape[0]
        registers = self.register_tokens.expand(count, -1, -1).to(hidden.dtype)
        hidden = torch.cat([hidden, registers, torch.zeros_like(hidden[:, :1])], dim=1)
        grid = 2.0 * (torch.arange(0.5, tile, dtype=torch.float32, device=hidden.device) / tile) - 1.0
        positions = torch.stack(torch.meshgrid(grid, grid, indexing="ij"), dim=-1).flatten(0, 1)
        positions = torch.cat([positions, positions.new_zeros(self.register_tokens.shape[1] + 1, 2)])
        rotary_embedding = self.rope(positions[None].expand(count, -1, -1))
        for block in self.blocks:
            hidden = block(hidden, rotary_embedding)
        hidden = functional.layer_norm(hidden[:, : tile * tile].float(), (hidden.shape[-1],))
        hidden = hidden.reshape(batch_size, rows, columns, tile, tile, -1).permute(0, 5, 1, 3, 2, 4)
        return hidden.reshape(batch_size, -1, height, width).to(torch.bfloat16)


def build_model() -> TranslatorTransformer:
    return TranslatorTransformer()


def initialize_from_steppe(model: TranslatorTransformer) -> None:
    """
    Copies every Steppe-Flux.2 base tensor whose name and shape match (the H3 source input conv stays random).
    """
    steppe_state = load_file(str(STEPPE_CHECKPOINT_PATH))
    model_state = model.state_dict()
    loaded, skipped = [], []
    for name, tensor in steppe_state.items():
        if name in model_state and model_state[name].shape == tensor.shape:
            model_state[name] = tensor.to(model_state[name].dtype)
            loaded.append(name)
        else:
            skipped.append(name)
    model.load_state_dict(model_state)
    missing = [name for name in model_state if name not in steppe_state and not name.startswith("linear_base")]
    print(f"{RUN_NAME}: Steppe init loaded {len(loaded)} tensors, skipped {skipped}, new {missing}", flush=True)


class Run0003Translator(Translator):
    """
    run_0001 plus frozen features of the first 8 (of 24) H3-Image decoder blocks added to the source path: does the H3
    decoder's own processing of the latent help.
    """

    def __init__(self) -> None:
        self.model: TranslatorTransformer | None = None
        self.feature_extractor: H3FeatureExtractor | None = None

    def _features(self, latents_source: torch.Tensor) -> torch.Tensor | None:
        if not H3_FEATURE_BLOCKS:
            return None
        if self.feature_extractor is None:
            self.feature_extractor = H3FeatureExtractor().to(DEVICE)
        return self.feature_extractor(latents_source)

    def _compile_training_shapes(self, model: nn.Module, base_model: TranslatorTransformer) -> None:
        """
        Compiles forward and backward of the training shape before the training clock starts.
        """
        latents_source = torch.randn(BATCH_SIZE, H3_CHANNELS, 64, 64, device=DEVICE)
        noisy_samples = torch.randn(BATCH_SIZE, FLUX_CHANNELS, 128, 128, device=DEVICE)
        latents_base = base_model.linear_base(latents_source)  # float32, as in the training loop
        h3_features = self._features(latents_source)
        with torch.autocast(DEVICE, dtype=TRAINING_DTYPE):
            velocity = model(
                noisy_samples, torch.rand(BATCH_SIZE, device=DEVICE), latents_source, latents_base, h3_features
            )
        velocity.float().square().mean().backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    def train(self, smoke_mode: bool = False) -> None:
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True
        model = build_model()
        if INITIALIZATION == "steppe_flux_2_200m":
            initialize_from_steppe(model)
        model = model.to(DEVICE)
        if smoke_mode:
            forward_model = model
        elif GRADIENT_CHECKPOINTING:
            for module in model.modules():
                if isinstance(module, (TransformerBlock, SourceEncoder)):
                    module.compile(mode=TRAINING_COMPILE_MODE, dynamic=False)
            forward_model = model
        else:
            forward_model = torch.compile(model, mode=TRAINING_COMPILE_MODE, dynamic=False)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters, lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY, fused=True
        )
        data_loader = DataLoader(
            LatentPairDataset(),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=PREFETCH_FACTOR,
            drop_last=True,
        )
        parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
        print(f"{RUN_NAME}: {parameter_count / 1e6:.2f}M parameters", flush=True)

        model.train()
        if not smoke_mode:
            compile_start_time = time.perf_counter()
            self._compile_training_shapes(forward_model, model)
            print(f"{RUN_NAME}: compiled in {time.perf_counter() - compile_start_time:.0f}s", flush=True)

        summary_writer = None if smoke_mode else SummaryWriter(str(TENSORBOARD_DIRECTORY))
        step = 0
        next_checkpoint_index = 0
        loss_sum = torch.zeros((), device=DEVICE)
        gradient_norm_max = torch.zeros((), device=DEVICE)
        batches = iterate_batches(data_loader)
        training_start_time = time.perf_counter()
        while True:
            elapsed_seconds = time.perf_counter() - training_start_time
            decay_factor = min(1.0, max(0.0, 1.0 - elapsed_seconds / TRAINING_TIME_SECONDS) / DECAY_FRACTION)
            warmup_factor = min(1.0, (step + 1) / WARMUP_STEPS)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = LEARNING_RATE * warmup_factor * decay_factor

            latents_target, latents_source = (tensor.to(DEVICE, non_blocking=True) for tensor in next(batches))
            latents_base = model.linear_base(latents_source)  # [B, 32, 128, 128]
            h3_features = self._features(latents_source)
            flow_target = to_flow_space(latents_target, latents_base, model.linear_base)
            noise = torch.randn_like(flow_target)
            sigmas = sample_training_sigmas(flow_target.shape[0], DEVICE)  # [B]
            blend = sigmas[:, None, None, None]
            noisy_samples = (1.0 - blend) * flow_target + blend * noise
            with torch.autocast(DEVICE, dtype=TRAINING_DTYPE):
                velocity = forward_model(noisy_samples, sigmas, latents_source, latents_base, h3_features)
            loss = functional.mse_loss(velocity.float(), noise - flow_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, GRADIENT_CLIP_NORM)
            gradient_norm_max = torch.maximum(gradient_norm_max, gradient_norm.detach())
            optimizer.step()

            step += 1
            loss_sum += loss.detach()
            elapsed_seconds = time.perf_counter() - training_start_time
            if step % LOG_INTERVAL_STEPS == 0:
                mean_loss = loss_sum.item() / LOG_INTERVAL_STEPS
                print(f"step {step} loss {mean_loss:.4f} elapsed {elapsed_seconds:.0f}s", flush=True)
                if summary_writer is not None:
                    summary_writer.add_scalar("train/loss", mean_loss, step)
                    summary_writer.add_scalar("train/elapsed_seconds", elapsed_seconds, step)
                    summary_writer.add_scalar("train/samples", step * BATCH_SIZE, step)
                    summary_writer.add_scalar("train/grad_norm_max", gradient_norm_max.item(), step)
                loss_sum.zero_()
                gradient_norm_max.zero_()

            if (
                not smoke_mode
                and next_checkpoint_index < len(CHECKPOINT_FRACTIONS)
                and elapsed_seconds >= CHECKPOINT_FRACTIONS[next_checkpoint_index] * TRAINING_TIME_SECONDS
            ):
                percent = round(CHECKPOINT_FRACTIONS[next_checkpoint_index] * 100)
                OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
                save_file(model.state_dict(), str(OUTPUT_DIRECTORY / f"checkpoint_{percent:03d}pct.safetensors"))
                print(f"{RUN_NAME}: checkpoint {percent}% at step {step}", flush=True)
                next_checkpoint_index += 1

            if step >= SMOKE_STEPS if smoke_mode else elapsed_seconds >= TRAINING_TIME_SECONDS:
                break

        if summary_writer is not None:
            summary_writer.close()
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        save_file(model.state_dict(), str(OUTPUT_DIRECTORY / MODEL_FILE_NAME))
        print(f"{RUN_NAME}: saved weights after {step} steps to {OUTPUT_DIRECTORY}", flush=True)

    def prepare_to_benchmark(self) -> None:
        model = build_model()
        model.load_state_dict(load_file(str(OUTPUT_DIRECTORY / MODEL_FILE_NAME)))
        self.model = model.to(DEVICE, INFERENCE_DTYPE).eval().requires_grad_(False)
        self.model.linear_base.float()
        if H3_FEATURE_BLOCKS and self.feature_extractor is None:
            self.feature_extractor = H3FeatureExtractor().to(DEVICE)

    def compile(self) -> None:
        self.model = torch.compile(self.model, mode="reduce-overhead")

    def _base_model(self) -> TranslatorTransformer:
        return getattr(self.model, "_orig_mod", self.model)

    @torch.no_grad()
    def translate(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, 24, h, w], normalized H3 latents, device = "cuda"
        returns:
            latents: [B, 32, 2h, 2w], normalized FLUX.2 latents, dtype: torch.float32
        """
        batch_size, _, height, width = latents.shape
        linear_base = self._base_model().linear_base
        latents_source = latents.to(INFERENCE_DTYPE)
        latents_base = linear_base(latents.float())  # [B, 32, 2h, 2w] float32
        h3_features = self._features(latents_source)
        generator = torch.Generator(device=latents.device).manual_seed(INFERENCE_SEED)
        samples = INITIAL_NOISE_SCALE * torch.randn(
            (batch_size, FLUX_CHANNELS, 2 * height, 2 * width), generator=generator, device=latents.device
        )
        sampling_sigmas = shift_sigmas(
            torch.linspace(1.0, 0.0, NUM_SAMPLING_STEPS + 1, device=latents.device), SAMPLING_SIGMA_SHIFT
        )
        base_input = latents_base.to(INFERENCE_DTYPE)
        for current_sigma, next_sigma in zip(sampling_sigmas[:-1], sampling_sigmas[1:]):
            velocity = self.model(
                samples.to(INFERENCE_DTYPE), current_sigma.expand(batch_size), latents_source, base_input, h3_features
            )
            samples = samples + (next_sigma - current_sigma) * velocity.float()
        return from_flow_space(samples, latents_base, linear_base).float()

    @torch.no_grad()
    def single_forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, 24, h, w], device = "cuda"
        returns:
            velocity: [B, 32, 2h, 2w], dtype: INFERENCE_DTYPE
        """
        batch_size, _, height, width = latents.shape
        latents_source = latents.to(INFERENCE_DTYPE)
        latents_base = self._base_model().linear_base(latents_source.float()).to(INFERENCE_DTYPE)
        noisy_samples = torch.randn(
            (batch_size, FLUX_CHANNELS, 2 * height, 2 * width), device=latents.device, dtype=INFERENCE_DTYPE
        )
        sigmas = torch.ones(batch_size, device=latents.device)
        return self.model(noisy_samples, sigmas, latents_source, latents_base, self._features(latents_source))

    def get_name(self) -> str:
        return RUN_NAME
