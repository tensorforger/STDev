import math
import time
from dataclasses import dataclass
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

from stdev.wan_upscalers.base import Upscaler

RUN_NAME = "run_0021"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "wan_1024_i19k_v9k"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "trained_models" / RUN_NAME
TENSORBOARD_DIRECTORY = REPOSITORY_ROOT / "tensorboard" / RUN_NAME
MODEL_FILE_NAME = "model.safetensors"

DEVICE = "cuda"
SEED = 0
LATENT_CHANNELS = 16

# Architecture: plain window transformer at full latent resolution (no UNet). The UNet constants below
# are unused by this run (kept for the shared source alignment code).
TRANSFORMER_WIDTH = 128
TRANSFORMER_DEPTH = 8
TRANSFORMER_HEAD_DIM = 64
# Temporal attention only looks at the current and earlier latent frames (the Wan VAE is causal).
CAUSAL_TEMPORAL_ATTENTION = True
FEED_FORWARD_HIDDEN = 341  # SwiGLU hidden width, ~8/3 x width
LEVEL_CHANNELS = (256, 256, 512)
ENCODER_BLOCKS = (1, 1, 1)
DECODER_BLOCKS = (4, 1, 1)
MIDDLE_BLOCKS = 1
# Per level flags, the last entry is the middle stage.
WINDOW_ATTENTION_LEVELS = (False, True, True, True)
TEMPORAL_ATTENTION_LEVELS = (True, True, True, True)
FEED_FORWARD_LEVELS = (False, False, False, False)
MLP_EXPANSION = 4
GRADIENT_CHECKPOINTING = False
CONDITION_DIM = 512
FREQUENCY_DIM = 256
MAX_NORM_GROUPS = 32
SIGMA_SCALE = 1000.0
HFF_SCALE = 1000.0
ATTENTION_HEAD_DIM = 64
WINDOW_SIZE = 8

# Low resolution input path.
# "interpolate": source latents aligned to the target grid and concatenated to the noisy latents.
# "learned": additionally, a convolutional encoder at the source resolution with learned (pixel shuffle)
# spatial and causal phase temporal upsampling adds features to the full resolution input features.
SOURCE_PATH = "learned"
SOURCE_ENCODER_BLOCKS = 2
# "causal": target latent frame j takes source latent frame ceil(j / 2) (exact for the causal Wan VAE,
# target frames 2k - 1 and 2k cover the pixel frames of source frame k). "trilinear": baseline resize.
TEMPORAL_ALIGNMENT = "causal"
# Temporal-only upscaling: the first latent frame encodes the same single pixel frame in source and
# target, so it is copied from the source after sampling.
COPY_FIRST_FRAME = True

# Residual parameterization: the flow runs on r = (x - base) / RESIDUAL_STD with base = the source latents
# aligned to the target grid (causal frame index + bilinear). Wan latents are smooth: the residual has std
# 0.22 (images) / 0.33 (videos) vs 0.72 for the latents, so unit noise on x buries the detail at most sigmas.
PARAMETERIZATION = "residual"
RESIDUAL_STD_SPATIAL = (0.129, 0.221, 0.170, 0.153, 0.334, 0.249, 0.156, 0.212, 0.141, 0.161, 0.131, 0.228, 0.207, 0.338, 0.140, 0.214)
RESIDUAL_STD_TEMPORAL = (0.211, 0.360, 0.248, 0.285, 0.417, 0.349, 0.236, 0.299, 0.222, 0.302, 0.218, 0.387, 0.375, 0.469, 0.244, 0.300)

TRAINING_TIME_SECONDS = 30 * 60
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75)
BATCH_TOKEN_BUDGET = 2 * 128 * 128
MAX_BATCH_SIZE = 64
NUM_WORKERS = 8
DYNAMO_CACHE_SIZE_LIMIT = 128
PREFETCH_FACTOR = 4
NUM_BATCHES_PER_EPOCH = 10_000_000
LEARNING_RATE = 1.2e-3
ADAM_BETAS = (0.9, 0.95)
WARMUP_STEPS = 100
DECAY_FRACTION = 0.3
GRADIENT_CLIP_NORM = 1.0
TRAINING_SIGMA_SHIFT = 1.0
SAMPLING_SIGMA_SHIFT = 4.0
LOG_INTERVAL_STEPS = 100
TRAINING_DTYPE = torch.bfloat16

NUM_SAMPLING_STEPS = 14
INFERENCE_SEED = 0
# Temperature of the initial sampling noise, per task (round 3 sweep on run_0009: spatial upscaling is
# best at 0.9, temporal-only upscaling at 0.7).
INITIAL_NOISE_SCALE_SPATIAL = 0.9
INITIAL_NOISE_SCALE_TEMPORAL = 0.7
INFERENCE_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class PairType:
    """
    One kind of training pair: which block entries are the input and the target, and its sampling weight.
    `source_size` and `target_size` are latent sizes (T, H, W), the latter sets the token-budget batch size.
    """

    name: str
    block_kind: str
    source_key: str
    target_key: str
    hff_key: str
    source_size: tuple[int, int, int]
    target_size: tuple[int, int, int]
    weight: float

    @property
    def batch_size(self) -> int:
        frames, height, width = self.target_size
        return max(1, min(MAX_BATCH_SIZE, BATCH_TOKEN_BUDGET // (frames * height * width)))


PAIR_TYPES = (
    PairType("image_512_to_1024", "images", "latents_512", "latents_1024", "hff_1024", (1, 64, 64), (1, 128, 128), 5.0),
    PairType("image_256_to_512", "images", "latents_256", "latents_512", "hff_512", (1, 32, 32), (1, 64, 64), 0.0),
    PairType("image_128_to_256", "images", "latents_128", "latents_256", "hff_256", (1, 16, 16), (1, 32, 32), 0.0),
    PairType("video_17_to_33_512", "videos", "latents_17_512", "latents_33_512", "hff_33_512", (5, 64, 64), (9, 64, 64), 4.0),
    PairType("video_17_to_33_1024", "videos", "latents_17_1024", "latents_33_1024", "hff_33_1024", (5, 128, 128), (9, 128, 128), 0.0),
    PairType("video_17_512_to_17_1024", "videos", "latents_17_512", "latents_17_1024", "hff_17_1024", (5, 64, 64), (5, 128, 128), 0.0),
    PairType("video_33_512_to_33_1024", "videos", "latents_33_512", "latents_33_1024", "hff_33_1024", (9, 64, 64), (9, 128, 128), 0.0),
    PairType("video_17_512_to_33_1024", "videos", "latents_17_512", "latents_33_1024", "hff_33_1024", (5, 64, 64), (9, 128, 128), 0.0),
)


class PairBatchDataset(Dataset):
    """
    Each item is a whole batch of one pair type, so all samples of a batch share their shape.
    The pair type is drawn by weight (cycled over all types in smoke mode), deterministically from the index.
    """

    def __init__(self, dataset_directory: Path, smoke_mode: bool) -> None:
        self.smoke_mode = smoke_mode
        self.block_paths = {
            block_kind: sorted((dataset_directory / block_kind).glob("block_*.vz.safetensors"))
            for block_kind in ("images", "videos")
        }
        weights = np.array([pair_type.weight for pair_type in PAIR_TYPES], dtype=np.float64)
        self.probabilities = weights / weights.sum()

    def __len__(self) -> int:
        return len(PAIR_TYPES) if self.smoke_mode else NUM_BATCHES_PER_EPOCH

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        returns:
            latents_target: [B, C, T_hr, H_hr, W_hr], dtype: torch.float32
            latents_source: [B, C, T_lr, H_lr, W_lr], dtype: torch.float32
            hff: [B], values: [0...1], dtype: torch.float32, high frequency feature of the target
            pair_index: index of the pair type in PAIR_TYPES
        """
        random_generator = np.random.default_rng([SEED, index])
        if self.smoke_mode:
            pair_index = index % len(PAIR_TYPES)
        else:
            pair_index = int(random_generator.choice(len(PAIR_TYPES), p=self.probabilities))
        pair_type = PAIR_TYPES[pair_index]
        paths = self.block_paths[pair_type.block_kind]
        block_indices = random_generator.integers(0, len(paths), pair_type.batch_size)

        targets, sources, hffs = [], [], []
        for block_index in block_indices:
            encoded_block = vaezip.load(paths[block_index])
            targets.append(encoded_block[pair_type.target_key].decode(dtype=torch.float32)[0])
            sources.append(encoded_block[pair_type.source_key].decode(dtype=torch.float32)[0])
            hffs.append(encoded_block[pair_type.hff_key].decode(dtype=torch.float32)[0])
        return torch.stack(targets), torch.stack(sources), torch.stack(hffs), pair_index


def iterate_batches(data_loader: DataLoader) -> Iterator[tuple]:
    while True:
        yield from data_loader


def shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """
    Moves noise levels towards pure noise for shift > 1, as in SD3.

    args:
        sigmas: [...], values: [0...1], 1 is pure noise
    returns:
        shifted_sigmas: [...], values: [0...1], 1 is pure noise
    """
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def sample_training_sigmas(batch_size: int, device: str) -> torch.Tensor:
    """
    returns:
        sigmas: [B], values: [0...1], logit-normal distributed and shifted
    """
    return shift_sigmas(torch.sigmoid(torch.randn(batch_size, device=device)), TRAINING_SIGMA_SHIFT)


def causal_frame_indices(source_frames: int, target_frames: int, device) -> torch.Tensor:
    """
    Source latent frame of each target latent frame for the causal Wan VAE (1 + 4k pixel frames).

    returns:
        indices: [T_hr], dtype: torch.long
    """
    if target_frames == source_frames:
        return torch.arange(target_frames, device=device)
    assert target_frames - 1 == 2 * (source_frames - 1), (source_frames, target_frames)
    return torch.div(torch.arange(target_frames, device=device) + 1, 2, rounding_mode="floor")


def align_source(latents_source: torch.Tensor, target_size: tuple[int, int, int]) -> torch.Tensor:
    """
    Resizes the source latents to the target grid.

    args:
        latents_source: [B, C, T_lr, H_lr, W_lr]
        target_size: (T_hr, H_hr, W_hr)
    returns:
        latents_aligned: [B, C, T_hr, H_hr, W_hr]
    """
    if TEMPORAL_ALIGNMENT == "trilinear":
        return functional.interpolate(latents_source, size=target_size, mode="trilinear", align_corners=False)
    indices = causal_frame_indices(latents_source.shape[2], target_size[0], latents_source.device)
    if target_size[0] != latents_source.shape[2]:
        latents_source = latents_source.index_select(2, indices)
    if tuple(target_size[1:]) != tuple(latents_source.shape[3:]):
        latents_source = functional.interpolate(latents_source, size=target_size, mode="trilinear", align_corners=False)
    return latents_source


def residual_scale(source_shape, target_size, device) -> torch.Tensor:
    """
    Per channel residual std: spatial table if the spatial size changes, temporal table otherwise.

    returns:
        scale: [1, C, 1, 1, 1], dtype: torch.float32
    """
    table = RESIDUAL_STD_SPATIAL if tuple(target_size[1:]) != tuple(source_shape[3:]) else RESIDUAL_STD_TEMPORAL
    return torch.tensor(table, device=device, dtype=torch.float32)[None, :, None, None, None]


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
    )  # [half_dim]
    angles = values.float()[:, None] * scale * frequencies[None, :]  # [N, half_dim]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class ConditioningEncoder(nn.Module):
    """
    Sum of MLP embeddings of the noise level and of the high frequency feature.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sigma_projection = self._make_projection()
        self.hff_projection = self._make_projection()

    @staticmethod
    def _make_projection() -> nn.Sequential:
        return nn.Sequential(nn.Linear(FREQUENCY_DIM, CONDITION_DIM), nn.SiLU(), nn.Linear(CONDITION_DIM, CONDITION_DIM))

    def forward(self, sigmas: torch.Tensor, hff: torch.Tensor) -> torch.Tensor:
        """
        args:
            sigmas: [B], values: [0...1], 1 is pure noise
            hff: [B], values: [0...1]
        returns:
            condition: [B, CONDITION_DIM]
        """
        weight_dtype = self.sigma_projection[0].weight.dtype
        sigma_embedding = sinusoidal_embedding(sigmas, SIGMA_SCALE).to(weight_dtype)
        hff_embedding = sinusoidal_embedding(hff, HFF_SCALE).to(weight_dtype)
        return self.sigma_projection(sigma_embedding) + self.hff_projection(hff_embedding)


class ConditionedResidualBlock(nn.Module):
    """
    Per frame residual block with the condition injected as scale and shift of the second normalization.
    The last convolution is zero initialized, so each block starts as the identity.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm_first = make_group_norm(in_channels)
        self.conv_first = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.condition_projection = nn.Sequential(nn.SiLU(), nn.Linear(CONDITION_DIM, 2 * out_channels))
        self.norm_second = make_group_norm(out_channels)
        self.conv_second = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_second.weight)
        nn.init.zeros_(self.conv_second.bias)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B * T, C_in, H, W]
            condition: [B * T, CONDITION_DIM]
        returns:
            features: [B * T, C_out, H, W]
        """
        hidden = self.conv_first(functional.silu(self.norm_first(features)))
        scale, shift = self.condition_projection(condition).chunk(2, dim=1)  # [B * T, C_out]
        hidden = self.norm_second(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv_second(functional.silu(hidden))
        return self.skip(features) + hidden


class PlainResidualBlock(nn.Module):
    """
    Unconditioned per frame residual block (source encoder).
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


class WindowAttention(nn.Module):
    """
    Multi-head self-attention inside non-overlapping WINDOW_SIZE x WINDOW_SIZE windows of each frame.
    The output projection is zero initialized, so the module starts as the identity.
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
            features: [B * T, C, H, W], H and W divisible by WINDOW_SIZE
        returns:
            features: [B * T, C, H, W]
        """
        batch_size, channels, height, width = features.shape
        heads = channels // ATTENTION_HEAD_DIM
        rows, columns = height // WINDOW_SIZE, width // WINDOW_SIZE
        hidden = self.query_key_value(self.norm(features)).reshape(
            batch_size, 3, heads, ATTENTION_HEAD_DIM, rows, WINDOW_SIZE, columns, WINDOW_SIZE
        )
        hidden = hidden.permute(1, 0, 2, 4, 6, 5, 7, 3).reshape(
            3, batch_size, heads, rows * columns, WINDOW_SIZE * WINDOW_SIZE, ATTENTION_HEAD_DIM
        )  # [3, B * T, heads, windows, window tokens, D]
        attended = functional.scaled_dot_product_attention(hidden[0], hidden[1], hidden[2])
        attended = attended.reshape(batch_size, heads, rows, columns, WINDOW_SIZE, WINDOW_SIZE, ATTENTION_HEAD_DIM).permute(
            0, 1, 6, 2, 4, 3, 5
        )  # [B * T, heads, D, rows, window, columns, window]
        return features + self.output(attended.reshape(batch_size, channels, height, width))


class TemporalAttention(nn.Module):
    """
    Multi-head self-attention over the frames at each spatial position. Identity for single frame inputs.
    The output projection is zero initialized.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = make_group_norm(channels)
        self.query_key_value = nn.Conv2d(channels, 3 * channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        args:
            features: [B * T, C, H, W]
            num_frames: T
        returns:
            features: [B * T, C, H, W]
        """
        if num_frames == 1:
            return features
        folded_batch, channels, height, width = features.shape
        batch_size = folded_batch // num_frames
        heads = channels // ATTENTION_HEAD_DIM
        hidden = self.query_key_value(self.norm(features)).reshape(
            batch_size, num_frames, 3, heads, ATTENTION_HEAD_DIM, height * width
        )
        hidden = hidden.permute(2, 0, 3, 5, 1, 4)  # [3, B, heads, H * W, T, D]
        attended = functional.scaled_dot_product_attention(hidden[0], hidden[1], hidden[2])
        attended = attended.permute(0, 3, 1, 4, 2)  # [B, T, heads, D, H * W]
        return features + self.output(attended.reshape(folded_batch, channels, height, width))


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
        return features + self.contract(functional.gelu(self.expand(self.norm(features))))


class MixingBlock(nn.Module):
    """
    Optional window attention, temporal attention and pointwise MLP after a residual block.
    """

    def __init__(self, channels: int, window: bool, temporal: bool, feed_forward: bool) -> None:
        super().__init__()
        self.window_attention = WindowAttention(channels) if window else None
        self.temporal_attention = TemporalAttention(channels) if temporal else None
        self.feed_forward = PointwiseFeedForward(channels) if feed_forward else None

    def forward(self, features: torch.Tensor, num_frames: int) -> torch.Tensor:
        if self.window_attention is not None:
            features = self.window_attention(features)
        if self.temporal_attention is not None:
            features = self.temporal_attention(features, num_frames)
        if self.feed_forward is not None:
            features = self.feed_forward(features)
        return features


class ResidualBlockSequence(nn.Module):
    """
    Residual blocks, each followed by the mixing block of its level (if any mixing is enabled).
    """

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, level: int) -> None:
        super().__init__()
        flags = (WINDOW_ATTENTION_LEVELS[level], TEMPORAL_ATTENTION_LEVELS[level], FEED_FORWARD_LEVELS[level])
        self.blocks = nn.ModuleList(
            ConditionedResidualBlock(in_channels if index == 0 else out_channels, out_channels) for index in range(num_blocks)
        )
        self.mixers = nn.ModuleList(MixingBlock(out_channels, *flags) for _ in range(num_blocks)) if any(flags) else None

    def forward(self, features: torch.Tensor, condition: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        args:
            features: [B * T, C_in, H, W]
            condition: [B * T, CONDITION_DIM]
            num_frames: T
        returns:
            features: [B * T, C_out, H, W]
        """
        for index, block in enumerate(self.blocks):
            use_checkpoint = GRADIENT_CHECKPOINTING and self.training
            if use_checkpoint:
                features = torch.utils.checkpoint.checkpoint(block, features, condition, use_reentrant=False)
            else:
                features = block(features, condition)
            if self.mixers is not None:
                if use_checkpoint:
                    features = torch.utils.checkpoint.checkpoint(self.mixers[index], features, num_frames, use_reentrant=False)
                else:
                    features = self.mixers[index](features, num_frames)
        return features


class SourceEncoder(nn.Module):
    """
    Learned upsampling of the source latents: residual blocks at the source resolution, pixel shuffle 2x
    spatial upsampling, and causal phase temporal upsampling (target frames 2k - 1 and 2k from source
    frames k and k - 1). Output is added to the full resolution input features.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(LATENT_CHANNELS, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(PlainResidualBlock(channels) for _ in range(SOURCE_ENCODER_BLOCKS)))
        self.spatial_norm = make_group_norm(channels)
        self.spatial_upsample = nn.Conv2d(channels, 4 * channels, 3, padding=1)
        self.temporal_phases = nn.Conv2d(2 * channels, 2 * channels, 1)
        nn.init.zeros_(self.temporal_phases.weight)
        nn.init.zeros_(self.temporal_phases.bias)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, latents_source: torch.Tensor, target_size: tuple[int, int, int]) -> torch.Tensor:
        """
        args:
            latents_source: [B, C, T_lr, H_lr, W_lr]
            target_size: (T_hr, H_hr, W_hr)
        returns:
            features: [B * T_hr, channels, H_hr, W_hr]
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


class UpscalerUNet(nn.Module):
    """
    UNet on frames folded into the batch, predicting the flow velocity of the high resolution latents.
    The source latents aligned to the target grid are concatenated to the noisy latents.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conditioning = ConditioningEncoder()
        self.frame_projection = nn.Sequential(
            nn.Linear(FREQUENCY_DIM, LEVEL_CHANNELS[0]), nn.SiLU(), nn.Linear(LEVEL_CHANNELS[0], LEVEL_CHANNELS[0])
        )
        self.input_conv = nn.Conv2d(2 * LATENT_CHANNELS, LEVEL_CHANNELS[0], 3, padding=1)
        self.source_encoder = SourceEncoder(LEVEL_CHANNELS[0]) if SOURCE_PATH == "learned" else None
        num_levels = len(LEVEL_CHANNELS)
        self.encoder_stages = nn.ModuleList(
            ResidualBlockSequence(LEVEL_CHANNELS[level], LEVEL_CHANNELS[level], ENCODER_BLOCKS[level], level)
            for level in range(num_levels)
        )
        self.downsamples = nn.ModuleList(
            nn.Conv2d(LEVEL_CHANNELS[level], LEVEL_CHANNELS[level + 1], 3, stride=2, padding=1) for level in range(num_levels - 1)
        )
        self.middle_blocks = ResidualBlockSequence(LEVEL_CHANNELS[-1], LEVEL_CHANNELS[-1], MIDDLE_BLOCKS, num_levels)
        last_level = num_levels - 1
        self.decoder_stages = nn.ModuleList(
            ResidualBlockSequence(
                LEVEL_CHANNELS[min(level + 1, last_level)] + LEVEL_CHANNELS[level], LEVEL_CHANNELS[level], DECODER_BLOCKS[level], level
            )
            for level in range(num_levels)
        )
        self.output_norm = make_group_norm(LEVEL_CHANNELS[0])
        self.output_conv = nn.Conv2d(LEVEL_CHANNELS[0], LATENT_CHANNELS, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, noisy_latents: torch.Tensor, sigmas: torch.Tensor, latents_source: torch.Tensor, hff: torch.Tensor) -> torch.Tensor:
        """
        args:
            noisy_latents: [B, C, T, H, W]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_source: [B, C, T_lr, H_lr, W_lr], each size <= the size of noisy_latents
            hff: [B], values: [0...1]
        returns:
            velocity: [B, C, T, H, W]
        """
        batch_size, channels, num_frames, height, width = noisy_latents.shape
        condition = self.conditioning(sigmas, hff).repeat_interleave(num_frames, dim=0)  # [B * T, D]
        latents_aligned = align_source(latents_source, tuple(noisy_latents.shape[2:]))  # [B, C, T, H, W]
        stacked = torch.cat([noisy_latents, latents_aligned.to(noisy_latents.dtype)], dim=1)  # [B, 2C, T, H, W]
        folded = stacked.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 2 * channels, height, width)
        features = self.input_conv(folded)
        if self.source_encoder is not None:
            features = features + self.source_encoder(latents_source.to(features.dtype), tuple(noisy_latents.shape[2:]))

        frame_indices = torch.arange(num_frames, device=features.device).repeat(batch_size)  # [B * T]
        frame_embedding = self.frame_projection(sinusoidal_embedding(frame_indices, 1.0).to(features.dtype))
        features = features + frame_embedding[:, :, None, None]

        skips = []
        for level, encoder_stage in enumerate(self.encoder_stages):
            features = encoder_stage(features, condition, num_frames)
            skips.append(features)
            if level < len(self.downsamples):
                features = self.downsamples[level](features)

        features = self.middle_blocks(features, condition, num_frames)
        for level in reversed(range(len(LEVEL_CHANNELS))):
            skip = skips[level]
            features = functional.interpolate(features, size=skip.shape[-2:], mode="bilinear")
            features = self.decoder_stages[level](torch.cat([features, skip], dim=1), condition, num_frames)
        velocity = self.output_conv(functional.silu(self.output_norm(features)))  # [B * T, C, H, W]
        return velocity.reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)



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


def shifted_window_mask(height: int, width: int, shift: int, device) -> torch.Tensor:
    """
    Swin attention mask for cyclically shifted windows: tokens attend only within their original region.

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
    DiT style block with adaLN-zero modulation by the condition: (shifted) window self-attention with a
    learned relative position bias, temporal self-attention over the frames at each position (skipped for
    single frames), SwiGLU feed forward with a depthwise 3x3 convolution on the hidden features.
    """

    def __init__(self, dim: int, shift: bool) -> None:
        super().__init__()
        self.heads = dim // TRANSFORMER_HEAD_DIM
        self.shift = WINDOW_SIZE // 2 if shift else 0
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(CONDITION_DIM, 9 * dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.spatial_qkv = nn.Linear(dim, 3 * dim)
        self.spatial_output = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(torch.zeros(self.heads, (2 * WINDOW_SIZE - 1) ** 2))
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(WINDOW_SIZE), torch.arange(WINDOW_SIZE), indexing="ij")
        ).flatten(1)  # [2, window tokens]
        relative = coordinates[:, :, None] - coordinates[:, None, :] + WINDOW_SIZE - 1
        self.register_buffer("relative_index", relative[0] * (2 * WINDOW_SIZE - 1) + relative[1], persistent=False)
        self.temporal_qkv = nn.Linear(dim, 3 * dim)
        self.temporal_output = nn.Linear(dim, dim)
        self.feed_forward_input = nn.Linear(dim, 2 * FEED_FORWARD_HIDDEN)
        self.feed_forward_conv = nn.Conv2d(2 * FEED_FORWARD_HIDDEN, 2 * FEED_FORWARD_HIDDEN, 3, padding=1, groups=2 * FEED_FORWARD_HIDDEN)
        self.feed_forward_output = nn.Linear(FEED_FORWARD_HIDDEN, dim)

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
        qkv = windows.reshape(count, num_windows, window_tokens, 3, self.heads, TRANSFORMER_HEAD_DIM).permute(3, 0, 4, 1, 2, 5)
        bias = self.relative_bias[:, self.relative_index]  # [heads, window tokens, window tokens]
        bias = bias[None, :, None].to(qkv.dtype)  # [1, heads, 1, window tokens, window tokens]
        if self.shift:
            mask = shifted_window_mask(height, width, self.shift, tokens.device)  # [windows, t, t]
            bias = bias.masked_fill(~mask[None, None], float("-inf"))
        attended = functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=bias)
        attended = attended.permute(0, 2, 3, 1, 4).reshape(count, num_windows, window_tokens, dim)
        attended = window_merge(self.spatial_output(attended), height, width)
        if self.shift:
            attended = torch.roll(attended, (self.shift, self.shift), dims=(1, 2))
        return attended

    def _temporal_attention(self, tokens: torch.Tensor, num_frames: int) -> torch.Tensor:
        count, height, width, dim = tokens.shape
        batch_size = count // num_frames
        qkv = self.temporal_qkv(tokens).reshape(batch_size, num_frames, height * width, 3, self.heads, TRANSFORMER_HEAD_DIM)
        qkv = qkv.permute(3, 0, 2, 4, 1, 5)  # [3, B, H * W, heads, T, Dh]
        attended = functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=CAUSAL_TEMPORAL_ATTENTION)
        attended = attended.permute(0, 3, 1, 2, 4).reshape(count, height, width, dim)
        return self.temporal_output(attended)

    def _feed_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.feed_forward_input(tokens).permute(0, 3, 1, 2)  # [N, 2F, H, W]
        hidden = self.feed_forward_conv(hidden).permute(0, 2, 3, 1)
        gate, value = hidden.chunk(2, dim=-1)
        return self.feed_forward_output(functional.silu(gate) * value)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        args:
            tokens: [B * T, H, W, D]
            condition: [B * T, CONDITION_DIM]
            num_frames: T
        returns:
            tokens: [B * T, H, W, D]
        """
        modulation = self.modulation(condition)[:, None, None, :].chunk(9, dim=-1)  # 9 x [N, 1, 1, D]
        shift_a, scale_a, gate_a, shift_t, scale_t, gate_t, shift_f, scale_f, gate_f = modulation
        tokens = tokens + gate_a * self._spatial_attention(self.norm(tokens) * (1 + scale_a) + shift_a)
        if num_frames > 1:
            tokens = tokens + gate_t * self._temporal_attention(self.norm(tokens) * (1 + scale_t) + shift_t, num_frames)
        return tokens + gate_f * self._feed_forward(self.norm(tokens) * (1 + scale_f) + shift_f)


class UpscalerWindowTransformer(nn.Module):
    """
    Plain window transformer on the full resolution latent grid, frames folded into the batch:
    3x3 convolution stem on the noisy and aligned source latents, TRANSFORMER_DEPTH blocks alternating
    regular and shifted windows, adaLN final layer and zero initialized linear velocity head.
    """

    def __init__(self) -> None:
        super().__init__()
        dim = TRANSFORMER_WIDTH
        self.conditioning = ConditioningEncoder()
        self.frame_projection = nn.Sequential(nn.Linear(FREQUENCY_DIM, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.input_conv = nn.Conv2d(2 * LATENT_CHANNELS, dim, 3, padding=1)
        self.source_encoder = SourceEncoder(dim) if SOURCE_PATH == "learned" else None
        self.blocks = nn.ModuleList(TransformerBlock(dim, shift=index % 2 == 1) for index in range(TRANSFORMER_DEPTH))
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(CONDITION_DIM, 2 * dim))
        nn.init.zeros_(self.output_modulation[1].weight)
        nn.init.zeros_(self.output_modulation[1].bias)
        self.output_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.output = nn.Linear(dim, LATENT_CHANNELS)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy_latents: torch.Tensor, sigmas: torch.Tensor, latents_source: torch.Tensor, hff: torch.Tensor) -> torch.Tensor:
        """
        args:
            noisy_latents: [B, C, T, H, W]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_source: [B, C, T_lr, H_lr, W_lr]
            hff: [B], values: [0...1]
        returns:
            velocity: [B, C, T, H, W]
        """
        batch_size, channels, num_frames, height, width = noisy_latents.shape
        condition = self.conditioning(sigmas, hff).repeat_interleave(num_frames, dim=0)  # [B * T, D_c]
        latents_aligned = align_source(latents_source, tuple(noisy_latents.shape[2:]))
        stacked = torch.cat([noisy_latents, latents_aligned.to(noisy_latents.dtype)], dim=1)
        folded = stacked.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 2 * channels, height, width)
        features = self.input_conv(folded)  # [B * T, D, H, W]
        if self.source_encoder is not None:
            features = features + self.source_encoder(latents_source.to(features.dtype), tuple(noisy_latents.shape[2:]))
        frame_indices = torch.arange(num_frames, device=features.device).repeat(batch_size)
        frame_embedding = self.frame_projection(sinusoidal_embedding(frame_indices, 1.0).to(features.dtype))
        tokens = features.permute(0, 2, 3, 1) + frame_embedding[:, None, None, :]  # [B * T, H, W, D]
        for block in self.blocks:
            if GRADIENT_CHECKPOINTING and self.training:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, condition, num_frames, use_reentrant=False)
            else:
                tokens = block(tokens, condition, num_frames)
        shift, scale = self.output_modulation(condition)[:, None, None, :].chunk(2, dim=-1)
        velocity = self.output(self.output_norm(tokens) * (1 + scale) + shift)  # [B * T, H, W, C]
        return velocity.reshape(batch_size, num_frames, height, width, channels).permute(0, 4, 1, 2, 3)


def build_model() -> nn.Module:
    return UpscalerWindowTransformer()


COMPILED_MODULE_TYPES = (TransformerBlock, SourceEncoder)


class Run0021Upscaler(Upscaler):
    """
    run_0020 with width 128 (2 heads, SwiGLU hidden 341): lowest latency point of the front.
    """

    def __init__(self) -> None:
        self.model: nn.Module | None = None

    def _compile_training_shapes(self, model: nn.Module, compiled_model: nn.Module) -> None:
        """
        Compiles forward and backward for every batch shape before the training clock starts.
        """
        shapes = {(pair_type.batch_size, pair_type.source_size, pair_type.target_size) for pair_type in PAIR_TYPES if pair_type.weight > 0}
        for batch_size, source_size, target_size in sorted(shapes):
            noisy_latents = torch.randn(batch_size, LATENT_CHANNELS, *target_size, device=DEVICE)
            latents_source = torch.randn(batch_size, LATENT_CHANNELS, *source_size, device=DEVICE)
            with torch.autocast(DEVICE, dtype=TRAINING_DTYPE):
                velocity = compiled_model(
                    noisy_latents, torch.rand(batch_size, device=DEVICE), latents_source, torch.rand(batch_size, device=DEVICE)
                )
            velocity.float().square().mean().backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    def train(self, smoke_mode: bool = False) -> None:
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True
        model = build_model().to(DEVICE)
        if not smoke_mode:
            torch._dynamo.config.cache_size_limit = DYNAMO_CACHE_SIZE_LIMIT
            for module in model.modules():
                if isinstance(module, COMPILED_MODULE_TYPES):
                    module.compile(dynamic=False)
        compiled_model = model
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
        data_loader = DataLoader(
            PairBatchDataset(DATASET_DIRECTORY, smoke_mode),
            batch_size=None,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=PREFETCH_FACTOR,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"{RUN_NAME}: {parameter_count / 1e6:.2f}M parameters", flush=True)

        model.train()
        if not smoke_mode:
            compile_start_time = time.perf_counter()
            self._compile_training_shapes(model, compiled_model)
            print(f"{RUN_NAME}: compiled in {time.perf_counter() - compile_start_time:.0f}s", flush=True)

        summary_writer = None if smoke_mode else SummaryWriter(str(TENSORBOARD_DIRECTORY))
        step = 0
        next_checkpoint_index = 0
        loss_sum = torch.zeros((), device=DEVICE)
        pair_loss_sums = torch.zeros(len(PAIR_TYPES), device=DEVICE)
        pair_counts = torch.zeros(len(PAIR_TYPES), device=DEVICE)
        training_start_time = time.perf_counter()
        for batch in iterate_batches(data_loader):
            latents_target, latents_source, hff = (tensor.to(DEVICE, non_blocking=True) for tensor in batch[:3])
            pair_index = int(batch[3])
            if PARAMETERIZATION == "residual":
                base = align_source(latents_source, tuple(latents_target.shape[2:]))
                latents_target = (latents_target - base) / residual_scale(latents_source.shape, latents_target.shape[2:], DEVICE)
            noise = torch.randn_like(latents_target)
            sigmas = sample_training_sigmas(latents_target.shape[0], DEVICE)  # [B]
            blend = sigmas[:, None, None, None, None]
            noisy_latents = (1.0 - blend) * latents_target + blend * noise

            elapsed_seconds = time.perf_counter() - training_start_time
            decay_factor = min(1.0, max(0.0, 1.0 - elapsed_seconds / TRAINING_TIME_SECONDS) / DECAY_FRACTION)
            warmup_factor = min(1.0, (step + 1) / WARMUP_STEPS)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = LEARNING_RATE * warmup_factor * decay_factor

            with torch.autocast(DEVICE, dtype=TRAINING_DTYPE):
                velocity = compiled_model(noisy_latents, sigmas, latents_source, hff)
            loss = functional.mse_loss(velocity.float(), noise - latents_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()

            step += 1
            loss_sum += loss.detach()
            pair_loss_sums[pair_index] += loss.detach()
            pair_counts[pair_index] += 1
            elapsed_seconds = time.perf_counter() - training_start_time
            if step % LOG_INTERVAL_STEPS == 0:
                mean_loss = loss_sum.item() / LOG_INTERVAL_STEPS
                print(f"step {step} loss {mean_loss:.4f} elapsed {elapsed_seconds:.0f}s", flush=True)
                if summary_writer is not None:
                    summary_writer.add_scalar("train/loss", mean_loss, step)
                    counts = pair_counts.tolist()
                    sums = pair_loss_sums.tolist()
                    for index, pair_type in enumerate(PAIR_TYPES):
                        if counts[index] > 0:
                            summary_writer.add_scalar(f"train_pairs/{pair_type.name}", sums[index] / counts[index], step)
                    summary_writer.add_scalar("train/elapsed_seconds", elapsed_seconds, step)
                loss_sum.zero_()
                pair_loss_sums.zero_()
                pair_counts.zero_()

            if (
                not smoke_mode
                and next_checkpoint_index < len(CHECKPOINT_FRACTIONS)
                and elapsed_seconds >= CHECKPOINT_FRACTIONS[next_checkpoint_index] * TRAINING_TIME_SECONDS
            ):
                percent = round(CHECKPOINT_FRACTIONS[next_checkpoint_index] * 100)
                OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
                checkpoint_path = OUTPUT_DIRECTORY / f"checkpoint_{percent:03d}pct.safetensors"
                save_file(model.state_dict(), str(checkpoint_path))
                print(f"{RUN_NAME}: checkpoint {checkpoint_path.name} at step {step}", flush=True)
                next_checkpoint_index += 1

            if step >= len(PAIR_TYPES) if smoke_mode else elapsed_seconds >= TRAINING_TIME_SECONDS:
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

    def compile(self) -> None:
        self.model = torch.compile(self.model, mode="reduce-overhead")

    @torch.no_grad()
    def upscale(self, latents: torch.Tensor, hff: torch.Tensor, output_size: tuple[int, int, int]) -> torch.Tensor:
        """
        args:
            latents: [B, C, T_lr, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            hff: [B], values: [0...1], dtype = float32
            output_size: (T_hr, H_hr, W_hr)
        returns:
            latents: [B, C, T_hr, H_hr, W_hr], same dtype as the input
        """
        batch_size, channels = latents.shape[:2]
        generator = torch.Generator(device=latents.device).manual_seed(INFERENCE_SEED)
        spatial = tuple(output_size[1:]) != tuple(latents.shape[3:])
        noise_scale = INITIAL_NOISE_SCALE_SPATIAL if spatial else INITIAL_NOISE_SCALE_TEMPORAL
        samples = noise_scale * torch.randn(
            (batch_size, channels, *output_size), generator=generator, device=latents.device
        )  # float32
        latents_source = latents.to(INFERENCE_DTYPE)
        hff = hff.to(latents.device)
        sampling_sigmas = shift_sigmas(torch.linspace(1.0, 0.0, NUM_SAMPLING_STEPS + 1, device=latents.device), SAMPLING_SIGMA_SHIFT)
        for current_sigma, next_sigma in zip(sampling_sigmas[:-1], sampling_sigmas[1:]):
            velocity = self.model(samples.to(INFERENCE_DTYPE), current_sigma.expand(batch_size), latents_source, hff)
            samples = samples + (next_sigma - current_sigma) * velocity.float()
        if PARAMETERIZATION == "residual":
            base = align_source(latents.float(), tuple(output_size))
            samples = base + residual_scale(latents.shape, output_size, latents.device) * samples
        if COPY_FIRST_FRAME and tuple(output_size[1:]) == tuple(latents.shape[3:]):
            samples[:, :, 0] = latents[:, :, 0].float()
        return samples.to(latents.dtype)

    @torch.no_grad()
    def single_forward(self, latents: torch.Tensor, hff: torch.Tensor, output_size: tuple[int, int, int]) -> torch.Tensor:
        """
        args:
            latents: [B, C, T_lr, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            hff: [B], values: [0...1], dtype = float32
            output_size: (T_hr, H_hr, W_hr)
        returns:
            velocity: [B, C, T_hr, H_hr, W_hr], dtype: INFERENCE_DTYPE
        """
        batch_size, channels = latents.shape[:2]
        noisy_latents = torch.randn((batch_size, channels, *output_size), device=latents.device, dtype=INFERENCE_DTYPE)
        sigmas = torch.ones(batch_size, device=latents.device)
        return self.model(noisy_latents, sigmas, latents.to(INFERENCE_DTYPE), hff.to(latents.device))

    def get_name(self) -> str:
        return RUN_NAME
