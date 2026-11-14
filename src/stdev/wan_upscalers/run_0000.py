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

RUN_NAME = "run_0000"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "wan_1024_i10k_v2k"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "trained_models" / RUN_NAME
TENSORBOARD_DIRECTORY = REPOSITORY_ROOT / "tensorboard" / RUN_NAME
MODEL_FILE_NAME = "model.safetensors"

DEVICE = "cuda"
SEED = 0
LATENT_CHANNELS = 16

LEVEL_CHANNELS = (320, 320, 512)
LEVEL_BLOCKS = (1, 2, 2)
DECODER_LEVEL_BLOCKS = (3, 2, 2)
MIDDLE_BLOCKS = 2
MLP_EXPANSION = 4
GRADIENT_CHECKPOINTING = True
CONDITION_DIM = 512
FREQUENCY_DIM = 256
MAX_NORM_GROUPS = 32
SIGMA_SCALE = 1000.0
HFF_SCALE = 1000.0
ATTENTION_HEAD_DIM = 64
WINDOW_SIZE = 8

TRAINING_TIME_SECONDS = 30 * 60
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75)
BATCH_TOKEN_BUDGET = 16 * 128 * 128
MAX_BATCH_SIZE = 64
NUM_WORKERS = 8
DYNAMO_CACHE_SIZE_LIMIT = 128
PREFETCH_FACTOR = 4
NUM_BATCHES_PER_EPOCH = 10_000_000
LEARNING_RATE = 4e-4
ADAM_BETAS = (0.9, 0.95)
WARMUP_STEPS = 100
DECAY_FRACTION = 0.3
GRADIENT_CLIP_NORM = 1.0
TRAINING_SIGMA_SHIFT = 1.0
SAMPLING_SIGMA_SHIFT = 2.0
LOG_INTERVAL_STEPS = 100
TRAINING_DTYPE = torch.bfloat16

NUM_SAMPLING_STEPS = 20
INFERENCE_SEED = 0
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
        return min(MAX_BATCH_SIZE, BATCH_TOKEN_BUDGET // (frames * height * width))


PAIR_TYPES = (
    PairType(
        "image_512_to_1024",
        "images",
        "latents_512",
        "latents_1024",
        "hff_1024",
        (1, 64, 64),
        (1, 128, 128),
        4.0,
    ),
    PairType(
        "image_256_to_512",
        "images",
        "latents_256",
        "latents_512",
        "hff_512",
        (1, 32, 32),
        (1, 64, 64),
        1.0,
    ),
    PairType(
        "image_128_to_256",
        "images",
        "latents_128",
        "latents_256",
        "hff_256",
        (1, 16, 16),
        (1, 32, 32),
        0.0,
    ),
    PairType(
        "video_17_to_33_512",
        "videos",
        "latents_17_512",
        "latents_33_512",
        "hff_33_512",
        (5, 64, 64),
        (9, 64, 64),
        3.0,
    ),
    PairType(
        "video_17_to_33_1024",
        "videos",
        "latents_17_1024",
        "latents_33_1024",
        "hff_33_1024",
        (5, 128, 128),
        (9, 128, 128),
        1.0,
    ),
    PairType(
        "video_17_512_to_17_1024",
        "videos",
        "latents_17_512",
        "latents_17_1024",
        "hff_17_1024",
        (5, 64, 64),
        (5, 128, 128),
        0.0,
    ),
    PairType(
        "video_33_512_to_33_1024",
        "videos",
        "latents_33_512",
        "latents_33_1024",
        "hff_33_1024",
        (9, 64, 64),
        (9, 128, 128),
        0.0,
    ),
    PairType(
        "video_17_512_to_33_1024",
        "videos",
        "latents_17_512",
        "latents_33_1024",
        "hff_33_1024",
        (5, 64, 64),
        (9, 128, 128),
        0.0,
    ),
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
            latents_target: [B, C, T_hr, H_hr, W_hr], dtype: torch.float32
            latents_source: [B, C, T_lr, H_lr, W_lr], dtype: torch.float32
            hff: [B], values: [0...1], dtype: torch.float32, high frequency feature of the target
        """
        random_generator = np.random.default_rng([SEED, index])
        if self.smoke_mode:
            pair_type = PAIR_TYPES[index % len(PAIR_TYPES)]
        else:
            pair_type = PAIR_TYPES[random_generator.choice(len(PAIR_TYPES), p=self.probabilities)]
        paths = self.block_paths[pair_type.block_kind]
        block_indices = random_generator.integers(0, len(paths), pair_type.batch_size)

        targets, sources, hffs = [], [], []
        for block_index in block_indices:
            encoded_block = vaezip.load(paths[block_index])
            targets.append(encoded_block[pair_type.target_key].decode(dtype=torch.float32)[0])
            sources.append(encoded_block[pair_type.source_key].decode(dtype=torch.float32)[0])
            hffs.append(encoded_block[pair_type.hff_key].decode(dtype=torch.float32)[0])
        return torch.stack(targets), torch.stack(sources), torch.stack(hffs)


def iterate_batches(data_loader: DataLoader) -> Iterator[tuple[torch.Tensor, ...]]:
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
        -math.log(10000.0)
        * torch.arange(half_dim, device=values.device, dtype=torch.float32)
        / half_dim
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
        return nn.Sequential(
            nn.Linear(FREQUENCY_DIM, CONDITION_DIM),
            nn.SiLU(),
            nn.Linear(CONDITION_DIM, CONDITION_DIM),
        )

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
        attended = attended.reshape(
            batch_size, heads, rows, columns, WINDOW_SIZE, WINDOW_SIZE, ATTENTION_HEAD_DIM
        ).permute(0, 1, 6, 2, 4, 3, 5)  # [B * T, heads, D, rows, window, columns, window]
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
        """
        args:
            features: [B * T, C, H, W]
        returns:
            features: [B * T, C, H, W]
        """
        return features + self.contract(functional.gelu(self.expand(self.norm(features))))


class SpatioTemporalTransformerBlock(nn.Module):
    """
    Window attention, temporal attention and a pointwise MLP.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.window_attention = WindowAttention(channels)
        self.temporal_attention = TemporalAttention(channels)
        self.feed_forward = PointwiseFeedForward(channels)

    def forward(self, features: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        args:
            features: [B * T, C, H, W]
            num_frames: T
        returns:
            features: [B * T, C, H, W]
        """
        features = self.temporal_attention(self.window_attention(features), num_frames)
        return self.feed_forward(features)


class ResidualBlockSequence(nn.Module):
    """
    Residual blocks, each followed by a spatio-temporal transformer block.
    """

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ConditionedResidualBlock(in_channels if index == 0 else out_channels, out_channels)
            for index in range(num_blocks)
        )
        self.transformers = nn.ModuleList(
            SpatioTemporalTransformerBlock(out_channels) for _ in range(num_blocks)
        )

    def forward(
        self, features: torch.Tensor, condition: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """
        args:
            features: [B * T, C_in, H, W]
            condition: [B * T, CONDITION_DIM]
            num_frames: T
        returns:
            features: [B * T, C_out, H, W]
        """
        for block, transformer in zip(self.blocks, self.transformers):
            if GRADIENT_CHECKPOINTING and self.training:
                features = torch.utils.checkpoint.checkpoint(
                    block, features, condition, use_reentrant=False
                )
                features = torch.utils.checkpoint.checkpoint(
                    transformer, features, num_frames, use_reentrant=False
                )
            else:
                features = transformer(block(features, condition), num_frames)
        return features


class UpscalerUNet(nn.Module):
    """
    UNet on frames folded into the batch, with window and temporal attention, predicting the flow velocity
    of the high resolution latents. The trilinearly resized input latents are concatenated to the noisy latents.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conditioning = ConditioningEncoder()
        self.frame_projection = nn.Sequential(
            nn.Linear(FREQUENCY_DIM, LEVEL_CHANNELS[0]),
            nn.SiLU(),
            nn.Linear(LEVEL_CHANNELS[0], LEVEL_CHANNELS[0]),
        )
        self.input_conv = nn.Conv2d(2 * LATENT_CHANNELS, LEVEL_CHANNELS[0], 3, padding=1)
        self.encoder_stages = nn.ModuleList(
            ResidualBlockSequence(LEVEL_CHANNELS[level], LEVEL_CHANNELS[level], LEVEL_BLOCKS[level])
            for level in range(len(LEVEL_CHANNELS))
        )
        self.downsamples = nn.ModuleList(
            nn.Conv2d(LEVEL_CHANNELS[level], LEVEL_CHANNELS[level + 1], 3, stride=2, padding=1)
            for level in range(len(LEVEL_CHANNELS) - 1)
        )
        self.middle_blocks = ResidualBlockSequence(
            LEVEL_CHANNELS[-1], LEVEL_CHANNELS[-1], MIDDLE_BLOCKS
        )
        last_level = len(LEVEL_CHANNELS) - 1
        self.decoder_stages = nn.ModuleList(
            ResidualBlockSequence(
                LEVEL_CHANNELS[min(level + 1, last_level)] + LEVEL_CHANNELS[level],
                LEVEL_CHANNELS[level],
                DECODER_LEVEL_BLOCKS[level],
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
        latents_source: torch.Tensor,
        hff: torch.Tensor,
    ) -> torch.Tensor:
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
        condition = self.conditioning(sigmas, hff).repeat_interleave(
            num_frames, dim=0
        )  # [B * T, D]
        latents_resized = functional.interpolate(
            latents_source, size=noisy_latents.shape[2:], mode="trilinear", align_corners=False
        )  # [B, C, T, H, W]
        stacked = torch.cat([noisy_latents, latents_resized], dim=1)  # [B, 2C, T, H, W]
        folded = stacked.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, 2 * channels, height, width
        )
        features = self.input_conv(folded)

        frame_indices = torch.arange(num_frames, device=features.device).repeat(
            batch_size
        )  # [B * T]
        frame_embedding = self.frame_projection(
            sinusoidal_embedding(frame_indices, 1.0).to(features.dtype)
        )
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
            features = self.decoder_stages[level](
                torch.cat([features, skip], dim=1), condition, num_frames
            )
        velocity = self.output_conv(functional.silu(self.output_norm(features)))  # [B * T, C, H, W]
        return velocity.reshape(batch_size, num_frames, channels, height, width).permute(
            0, 2, 1, 3, 4
        )


class Run0000Upscaler(Upscaler):
    """
    Baseline of the Wan latent upscaler: about 100M parameter window attention UNet with temporal attention,
    rectified flow, mixed spatial (image) and temporal (video) upscaling pairs, 30 min of training.
    Recipe of the first phase: LR 4e-4 with linear decay over the last 30%, logit-normal sigmas, no EMA.
    """

    def __init__(self) -> None:
        self.model: nn.Module | None = None

    def _compile_training_shapes(self, model: nn.Module, compiled_model: nn.Module) -> None:
        """
        Compiles forward and backward for every batch shape before the training clock starts.
        """
        shapes = {
            (pair_type.batch_size, pair_type.source_size, pair_type.target_size)
            for pair_type in PAIR_TYPES
            if pair_type.weight > 0
        }
        for batch_size, source_size, target_size in sorted(shapes):
            noisy_latents = torch.randn(batch_size, LATENT_CHANNELS, *target_size, device=DEVICE)
            latents_source = torch.randn(batch_size, LATENT_CHANNELS, *source_size, device=DEVICE)
            with torch.autocast(DEVICE, dtype=TRAINING_DTYPE):
                velocity = compiled_model(
                    noisy_latents,
                    torch.rand(batch_size, device=DEVICE),
                    latents_source,
                    torch.rand(batch_size, device=DEVICE),
                )
            velocity.float().square().mean().backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    def train(self, smoke_mode: bool = False) -> None:
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True
        model = UpscalerUNet().to(DEVICE)
        if not smoke_mode:
            torch._dynamo.config.cache_size_limit = DYNAMO_CACHE_SIZE_LIMIT
            for module in model.modules():
                if isinstance(module, (ConditionedResidualBlock, SpatioTemporalTransformerBlock)):
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
            print(
                f"{RUN_NAME}: compiled in {time.perf_counter() - compile_start_time:.0f}s",
                flush=True,
            )

        summary_writer = None if smoke_mode else SummaryWriter(str(TENSORBOARD_DIRECTORY))
        step = 0
        next_checkpoint_index = 0
        loss_sum = torch.zeros((), device=DEVICE)
        training_start_time = time.perf_counter()
        for batch in iterate_batches(data_loader):
            latents_target, latents_source, hff = (
                tensor.to(DEVICE, non_blocking=True) for tensor in batch
            )  # [B, C, T, H, W], [B, C, T_lr, H_lr, W_lr], [B]
            noise = torch.randn_like(latents_target)
            sigmas = sample_training_sigmas(latents_target.shape[0], DEVICE)  # [B]
            blend = sigmas[:, None, None, None, None]
            noisy_latents = (1.0 - blend) * latents_target + blend * noise

            elapsed_seconds = time.perf_counter() - training_start_time
            decay_factor = min(
                1.0, max(0.0, 1.0 - elapsed_seconds / TRAINING_TIME_SECONDS) / DECAY_FRACTION
            )
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
            elapsed_seconds = time.perf_counter() - training_start_time
            if step % LOG_INTERVAL_STEPS == 0:
                mean_loss = loss_sum.item() / LOG_INTERVAL_STEPS
                print(
                    f"step {step} loss {mean_loss:.4f} elapsed {elapsed_seconds:.0f}s", flush=True
                )
                if summary_writer is not None:
                    summary_writer.add_scalar("train/loss", mean_loss, step)
                loss_sum.zero_()

            if (
                not smoke_mode
                and next_checkpoint_index < len(CHECKPOINT_FRACTIONS)
                and elapsed_seconds
                >= CHECKPOINT_FRACTIONS[next_checkpoint_index] * TRAINING_TIME_SECONDS
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
        model = UpscalerUNet()
        model.load_state_dict(load_file(str(OUTPUT_DIRECTORY / MODEL_FILE_NAME)))
        self.model = model.to(DEVICE, INFERENCE_DTYPE).eval().requires_grad_(False)

    def compile(self) -> None:
        self.model = torch.compile(self.model, mode="reduce-overhead")

    @torch.no_grad()
    def upscale(
        self, latents: torch.Tensor, hff: torch.Tensor, output_size: tuple[int, int, int]
    ) -> torch.Tensor:
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
        samples = torch.randn(
            (batch_size, channels, *output_size), generator=generator, device=latents.device
        )  # [B, C, T_hr, H_hr, W_hr], dtype: torch.float32
        latents_source = latents.to(INFERENCE_DTYPE)
        hff = hff.to(latents.device)
        sampling_sigmas = shift_sigmas(
            torch.linspace(1.0, 0.0, NUM_SAMPLING_STEPS + 1, device=latents.device),
            SAMPLING_SIGMA_SHIFT,
        )
        for current_sigma, next_sigma in zip(sampling_sigmas[:-1], sampling_sigmas[1:]):
            velocity = self.model(
                samples.to(INFERENCE_DTYPE), current_sigma.expand(batch_size), latents_source, hff
            )
            samples = samples + (next_sigma - current_sigma) * velocity.float()
        return samples.to(latents.dtype)

    @torch.no_grad()
    def single_forward(
        self, latents: torch.Tensor, hff: torch.Tensor, output_size: tuple[int, int, int]
    ) -> torch.Tensor:
        """
        args:
            latents: [B, C, T_lr, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            hff: [B], values: [0...1], dtype = float32
            output_size: (T_hr, H_hr, W_hr)
        returns:
            velocity: [B, C, T_hr, H_hr, W_hr], dtype: INFERENCE_DTYPE
        """
        batch_size, channels = latents.shape[:2]
        noisy_latents = torch.randn(
            (batch_size, channels, *output_size), device=latents.device, dtype=INFERENCE_DTYPE
        )
        sigmas = torch.ones(batch_size, device=latents.device)
        return self.model(
            noisy_latents, sigmas, latents.to(INFERENCE_DTYPE), hff.to(latents.device)
        )

    def get_name(self) -> str:
        return RUN_NAME
