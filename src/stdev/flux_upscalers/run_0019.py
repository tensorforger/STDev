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

from stdev.flux_upscalers.base import Upscaler

RUN_NAME = "run_0019"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIRECTORIES = (
    REPOSITORY_ROOT / "train_datasets" / "flux2_relaion_20k",
    REPOSITORY_ROOT / "train_datasets" / "flux2_4klsdb_20k",
)
# Sampling weight of each dataset directory (same order): a sample first draws its dataset, then a block uniformly.
DATASET_WEIGHTS = (1.0, 2.0)
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "trained_models" / "flux" / RUN_NAME
TENSORBOARD_DIRECTORY = REPOSITORY_ROOT / "tensorboard" / "flux" / RUN_NAME
MODEL_FILE_NAME = "model.safetensors"

DEVICE = "cuda"
SEED = 0
LATENT_CHANNELS = 32

# Source (low resolution input) of a training pair: block key -> sampling weight. `latents_512` is the bilinear
# downscale used by the benchmark; `latents_512_<variant>` are the corrupted inputs (nearest, bicubic, lanczos,
# antialias, jpeg).
SOURCE_KEY_WEIGHTS = {"latents_512": 1.0}
TARGET_KEY = "latents_1024"

# Architecture (autoresearch_1 recipe): convolutional UNet with 3 levels, capacity at full latent resolution (decoder
# heavy level 0), SDXL style window transformer block (8x8 window attention + MLP) after every residual block on all
# levels, FiLM of the noise level embedding, bilinear upsampled source concatenated to the noisy sample.
LEVEL_CHANNELS = (384, 192, 384)
ENCODER_BLOCKS = (1, 1, 1)
DECODER_BLOCKS = (4, 1, 1)
MIDDLE_BLOCKS = 1
MLP_EXPANSION = 4
ATTENTION_HEAD_DIM = 64
CONDITION_NORM = False
CONDITION_DIM = 512
FREQUENCY_DIM = 256
MAX_NORM_GROUPS = 32
SIGMA_SCALE = 1000.0
WINDOW_SIZE = 8
GRADIENT_CHECKPOINTING = False
# Learned source path (phase 2): residual blocks at the source resolution + pixel shuffle, added to the stem features.
SOURCE_ENCODER_WIDTH = 192
SOURCE_ENCODER_BLOCKS = 2

# "residual": the flow runs on r = (x - bilinear(source)) / RESIDUAL_STD (per channel); "direct": on x itself.
PARAMETERIZATION = "residual"
# Per channel std of latents_1024 - bilinear 2x(latents_512), both datasets (dev/flux2_latent_statistics.py).
RESIDUAL_STD = (0.685, 0.646, 0.700, 0.600, 0.701, 0.678, 0.563, 0.712, 0.614, 0.697, 0.716, 0.590, 0.687, 0.656, 0.738, 0.700, 0.717, 0.687, 0.701, 0.686, 0.744, 0.713, 0.658, 0.703, 0.721, 0.683, 0.663, 0.672, 0.682, 0.691, 0.687, 0.639)

TRAINING_TIME_SECONDS = 30 * 60
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75)
SMOKE_STEPS = 3
BATCH_SIZE = 16
NUM_WORKERS = 6
PREFETCH_FACTOR = 4
LEARNING_RATE = 4e-4
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
DECAY_FRACTION = 0.3
GRADIENT_CLIP_NORM = 1.0
TRAINING_SIGMA_SHIFT = 1.0
LOG_INTERVAL_STEPS = 100
TRAINING_DTYPE = torch.bfloat16

NUM_SAMPLING_STEPS = 30
SAMPLING_SIGMA_SHIFT = 2.0
INITIAL_NOISE_SCALE = 1.0
INFERENCE_SEED = 0
INFERENCE_DTYPE = torch.bfloat16


class LatentPairDataset(Dataset):
    """
    Infinite stream of (target, source) pairs; the source key is drawn per sample by SOURCE_KEY_WEIGHTS.
    Deterministic for an index: same SEED means identical batches across runs with the same data settings.
    """

    def __init__(self) -> None:
        self.dataset_block_paths = [sorted(directory.glob("block_*.vz.safetensors")) for directory in DATASET_DIRECTORIES]
        dataset_weights = np.array(DATASET_WEIGHTS, dtype=np.float64)
        self.dataset_probabilities = dataset_weights / dataset_weights.sum()
        self.source_keys = list(SOURCE_KEY_WEIGHTS)
        weights = np.array(list(SOURCE_KEY_WEIGHTS.values()), dtype=np.float64)
        self.source_probabilities = weights / weights.sum()

    def __len__(self) -> int:
        return 1_000_000_000

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
            latents_target: [C, 128, 128], dtype: torch.float32
            latents_source: [C, 64, 64], dtype: torch.float32
        """
        random_generator = np.random.default_rng([SEED, index])
        dataset_index = int(random_generator.choice(len(self.dataset_block_paths), p=self.dataset_probabilities))
        block_paths = self.dataset_block_paths[dataset_index]
        block_index = int(random_generator.integers(0, len(block_paths)))
        source_key = self.source_keys[int(random_generator.choice(len(self.source_keys), p=self.source_probabilities))]
        encoded_block = vaezip.load(block_paths[block_index])
        latents_target = encoded_block[TARGET_KEY].decode(dtype=torch.float32)[0]
        latents_source = encoded_block[source_key].decode(dtype=torch.float32)[0]
        return latents_target, latents_source


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


def align_source(latents_source: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents_source: [B, C, H, W]
    returns:
        latents_aligned: [B, C, 2H, 2W], bilinear
    """
    return functional.interpolate(latents_source, scale_factor=2, mode="bilinear", align_corners=False)


def residual_scale(device: torch.device | str) -> torch.Tensor:
    """
    returns:
        scale: [1, C, 1, 1], dtype: torch.float32
    """
    return torch.tensor(RESIDUAL_STD, device=device, dtype=torch.float32)[None, :, None, None]


def to_flow_space(latents_target: torch.Tensor, latents_source: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents_target: [B, C, 2H, 2W]
        latents_source: [B, C, H, W]
    returns:
        flow_target: [B, C, 2H, 2W], the sample the flow generates
    """
    if PARAMETERIZATION == "direct":
        return latents_target
    return (latents_target - align_source(latents_source)) / residual_scale(latents_target.device)


def from_flow_space(samples: torch.Tensor, latents_source: torch.Tensor) -> torch.Tensor:
    """
    args:
        samples: [B, C, 2H, 2W], generated flow samples, dtype: torch.float32
        latents_source: [B, C, H, W], dtype: torch.float32
    returns:
        latents: [B, C, 2H, 2W]
    """
    if PARAMETERIZATION == "direct":
        return samples
    return align_source(latents_source) + residual_scale(samples.device) * samples


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
    MLP embedding of the noise level, optionally RMS normalized.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sigma_projection = nn.Sequential(
            nn.Linear(FREQUENCY_DIM, CONDITION_DIM), nn.SiLU(), nn.Linear(CONDITION_DIM, CONDITION_DIM)
        )

    def forward(self, sigmas: torch.Tensor) -> torch.Tensor:
        """
        args:
            sigmas: [B], values: [0...1], 1 is pure noise
        returns:
            condition: [B, CONDITION_DIM]
        """
        weight_dtype = self.sigma_projection[0].weight.dtype
        condition = self.sigma_projection(sinusoidal_embedding(sigmas, SIGMA_SCALE).to(weight_dtype))
        return functional.rms_norm(condition, (condition.shape[-1],)) if CONDITION_NORM else condition


class ConditionedResidualBlock(nn.Module):
    """
    Residual block with the condition injected as scale and shift of the second normalization.
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
    Multi-head self-attention inside non-overlapping WINDOW_SIZE x WINDOW_SIZE windows, windows folded into the
    SDPA batch dimension. The output projection is zero initialized, so the module starts as the identity.
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
        hidden = hidden.permute(1, 0, 4, 6, 2, 5, 7, 3).reshape(
            3, batch_size * rows * columns, heads, WINDOW_SIZE * WINDOW_SIZE, ATTENTION_HEAD_DIM
        )  # [3, B * windows, heads, window tokens, D]
        attended = functional.scaled_dot_product_attention(hidden[0], hidden[1], hidden[2])
        attended = attended.reshape(
            batch_size, rows, columns, heads, WINDOW_SIZE, WINDOW_SIZE, ATTENTION_HEAD_DIM
        ).permute(0, 3, 6, 1, 4, 2, 5)  # [B, heads, D, rows, window, columns, window]
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
        return features + self.contract(functional.gelu(self.expand(self.norm(features))))


class PlainResidualBlock(nn.Module):
    """
    Unconditioned residual block (source encoder). The last convolution is zero initialized.
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
    Learned 2x upsampling of the source latents: residual blocks at the source resolution and pixel shuffle.
    """

    def __init__(self, channels: int, output_channels: int) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(LATENT_CHANNELS, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(PlainResidualBlock(channels) for _ in range(SOURCE_ENCODER_BLOCKS)))
        self.upsample_norm = make_group_norm(channels)
        self.upsample = nn.Conv2d(channels, 4 * channels, 3, padding=1)
        self.output = nn.Conv2d(channels, output_channels, 1)

    def forward(self, latents_source: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents_source: [B, C, H, W]
        returns:
            features: [B, output_channels, 2H, 2W]
        """
        features = self.blocks(self.input_conv(latents_source))
        features = functional.pixel_shuffle(self.upsample(functional.silu(self.upsample_norm(features))), 2)
        return self.output(features)


class UNetStage(nn.Module):
    """
    Residual blocks, each followed by an SDXL style window transformer block (window attention + MLP).
    """

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ConditionedResidualBlock(in_channels if index == 0 else out_channels, out_channels)
            for index in range(num_blocks)
        )
        self.attentions = nn.ModuleList(WindowAttention(out_channels) for _ in range(num_blocks))
        self.feed_forwards = nn.ModuleList(PointwiseFeedForward(out_channels) for _ in range(num_blocks))

    def _block(self, index: int, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.feed_forwards[index](self.attentions[index](self.blocks[index](features, condition)))

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C_in, H, W]
            condition: [B, CONDITION_DIM]
        returns:
            features: [B, C_out, H, W]
        """
        for index in range(len(self.blocks)):
            if GRADIENT_CHECKPOINTING and self.training:
                features = torch.utils.checkpoint.checkpoint(self._block, index, features, condition, use_reentrant=False)
            else:
                features = self._block(index, features, condition)
        return features


class UpscalerUNet(nn.Module):
    """
    Convolutional UNet with window transformer blocks on all levels, predicting the flow velocity at the target
    resolution. Bilinearly upsampled source latents are concatenated to the noisy sample.
    """

    def __init__(self) -> None:
        super().__init__()
        num_levels = len(LEVEL_CHANNELS)
        self.conditioning = ConditioningEncoder()
        self.input_conv = nn.Conv2d(2 * LATENT_CHANNELS, LEVEL_CHANNELS[0], 3, padding=1)
        self.source_encoder = SourceEncoder(SOURCE_ENCODER_WIDTH, LEVEL_CHANNELS[0])
        self.encoder_stages = nn.ModuleList(
            UNetStage(LEVEL_CHANNELS[level], LEVEL_CHANNELS[level], ENCODER_BLOCKS[level]) for level in range(num_levels)
        )
        self.downsamples = nn.ModuleList(
            nn.Conv2d(LEVEL_CHANNELS[level], LEVEL_CHANNELS[level + 1], 3, stride=2, padding=1)
            for level in range(num_levels - 1)
        )
        self.middle_stage = UNetStage(LEVEL_CHANNELS[-1], LEVEL_CHANNELS[-1], MIDDLE_BLOCKS)
        last_level = num_levels - 1
        self.decoder_stages = nn.ModuleList(
            UNetStage(
                LEVEL_CHANNELS[min(level + 1, last_level)] + LEVEL_CHANNELS[level],
                LEVEL_CHANNELS[level],
                DECODER_BLOCKS[level],
            )
            for level in range(num_levels)
        )
        self.output_norm = make_group_norm(LEVEL_CHANNELS[0])
        self.output_conv = nn.Conv2d(LEVEL_CHANNELS[0], LATENT_CHANNELS, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, noisy_samples: torch.Tensor, sigmas: torch.Tensor, latents_source: torch.Tensor) -> torch.Tensor:
        """
        args:
            noisy_samples: [B, C, 2H, 2W]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_source: [B, C, H, W]
        returns:
            velocity: [B, C, 2H, 2W]
        """
        condition = self.conditioning(sigmas)
        latents_aligned = align_source(latents_source.to(noisy_samples.dtype))
        features = self.input_conv(torch.cat([noisy_samples, latents_aligned], dim=1))
        features = features + self.source_encoder(latents_source.to(noisy_samples.dtype))
        skips = []
        for level, encoder_stage in enumerate(self.encoder_stages):
            features = encoder_stage(features, condition)
            skips.append(features)
            if level < len(self.downsamples):
                features = self.downsamples[level](features)
        features = self.middle_stage(features, condition)
        for level in reversed(range(len(LEVEL_CHANNELS))):
            skip = skips[level]
            features = functional.interpolate(features, size=skip.shape[-2:], mode="bilinear")
            features = self.decoder_stages[level](torch.cat([features, skip], dim=1), condition)
        return self.output_conv(functional.silu(self.output_norm(features)))


def build_model() -> nn.Module:
    return UpscalerUNet()


COMPILED_MODULE_TYPES = (ConditionedResidualBlock, WindowAttention, PointwiseFeedForward, SourceEncoder)


class Run0019Upscaler(Upscaler):
    """
    Q2 final check: the phase 1 UNet (384, 192, 384) with every shift that transferred: normalized residual, learned
    source path (width 192), ReLAION : 4KLSDB = 1 : 2, sampler 30 / 2 / 1.0; its own B16 / LR 4e-4 recipe. Clean inputs.
    Compare with run_0013 resampled at 30 / 2 / 1.0 (2.45).
    """

    def __init__(self) -> None:
        self.model: nn.Module | None = None

    def _compile_training_shapes(self, model: nn.Module) -> None:
        """
        Compiles forward and backward of the training shape before the training clock starts.
        """
        noisy_samples = torch.randn(BATCH_SIZE, LATENT_CHANNELS, 128, 128, device=DEVICE)
        latents_source = torch.randn(BATCH_SIZE, LATENT_CHANNELS, 64, 64, device=DEVICE)
        with torch.autocast(DEVICE, dtype=TRAINING_DTYPE):
            velocity = model(noisy_samples, torch.rand(BATCH_SIZE, device=DEVICE), latents_source)
        velocity.float().square().mean().backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    def train(self, smoke_mode: bool = False) -> None:
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True
        model = build_model().to(DEVICE)
        if not smoke_mode:
            for module in model.modules():
                if isinstance(module, COMPILED_MODULE_TYPES):
                    module.compile(dynamic=False)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY, fused=True
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
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"{RUN_NAME}: {parameter_count / 1e6:.2f}M parameters", flush=True)

        model.train()
        if not smoke_mode:
            compile_start_time = time.perf_counter()
            self._compile_training_shapes(model)
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
            flow_target = to_flow_space(latents_target, latents_source)  # [B, C, 128, 128]
            noise = torch.randn_like(flow_target)
            sigmas = sample_training_sigmas(flow_target.shape[0], DEVICE)  # [B]
            blend = sigmas[:, None, None, None]
            noisy_samples = (1.0 - blend) * flow_target + blend * noise
            with torch.autocast(DEVICE, dtype=TRAINING_DTYPE):
                velocity = model(noisy_samples, sigmas, latents_source)
            loss = functional.mse_loss(velocity.float(), noise - flow_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
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

    def compile(self) -> None:
        self.model = torch.compile(self.model, mode="reduce-overhead")

    @torch.no_grad()
    def upscale(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C, H, W], dtype = float32 (or float16, bfloat16), device = "cuda"
        returns:
            latents: [B, C, 2H, 2W], same dtype as the input
        """
        batch_size, channels, height, width = latents.shape
        generator = torch.Generator(device=latents.device).manual_seed(INFERENCE_SEED)
        samples = INITIAL_NOISE_SCALE * torch.randn(
            (batch_size, channels, 2 * height, 2 * width), generator=generator, device=latents.device
        )  # float32
        latents_source = latents.to(INFERENCE_DTYPE)
        sampling_sigmas = shift_sigmas(
            torch.linspace(1.0, 0.0, NUM_SAMPLING_STEPS + 1, device=latents.device), SAMPLING_SIGMA_SHIFT
        )
        for current_sigma, next_sigma in zip(sampling_sigmas[:-1], sampling_sigmas[1:]):
            velocity = self.model(samples.to(INFERENCE_DTYPE), current_sigma.expand(batch_size), latents_source)
            samples = samples + (next_sigma - current_sigma) * velocity.float()
        return from_flow_space(samples, latents.float()).to(latents.dtype)

    @torch.no_grad()
    def single_forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C, H, W], dtype = float32 (or float16, bfloat16), device = "cuda"
        returns:
            velocity: [B, C, 2H, 2W], dtype: INFERENCE_DTYPE
        """
        batch_size, channels, height, width = latents.shape
        noisy_samples = torch.randn(
            (batch_size, channels, 2 * height, 2 * width), device=latents.device, dtype=INFERENCE_DTYPE
        )
        sigmas = torch.ones(batch_size, device=latents.device)
        return self.model(noisy_samples, sigmas, latents.to(INFERENCE_DTYPE))

    def get_name(self) -> str:
        return RUN_NAME
