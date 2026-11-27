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
from diffusers import AutoencoderKLQwenImage21
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from stdev.qwen_upscalers.base import Upscaler

RUN_NAME = "run_0022"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIRECTORIES = (REPOSITORY_ROOT / "train_datasets" / "qwen_4klsdb_40k",)
# Sampling weight of each dataset directory (same order): a sample first draws its dataset, then a block uniformly.
DATASET_WEIGHTS = (1.0,)
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "trained_models" / "qwen" / RUN_NAME
TENSORBOARD_DIRECTORY = REPOSITORY_ROOT / "tensorboard" / "qwen" / RUN_NAME
MODEL_FILE_NAME = "model.safetensors"

DEVICE = "cuda"
SEED = 0
LATENT_CHANNELS = 64
BASE_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "qwen_4klsdb_40k_pixel_base"
BASE_KEY = "latents_1024_base"
DECODER_PATH = REPOSITORY_ROOT / "models" / "texture-fix-vae-for-qwen-image-2.1"

# Decoder aware loss (round 6): the frozen texture-fix decoder decodes random latent crops of the x0 prediction and of
# the target; pixel MSE on the crop centre, 2x pooled (fully determined by the input). For the flow it is applied only
# to samples with sigma < DECODER_LOSS_MAX_SIGMA, where x0 ~ the final sample (round 4: at high sigma it biases the
# velocity field).
DECODER_LOSS_WEIGHT = 1000.0
DECODER_LOSS_POOLING = 2
DECODER_LOSS_SAMPLES = 4
DECODER_LOSS_MAX_SIGMA = 1.0
DECODER_CROP_CORE = 8
DECODER_CROP_MARGIN = 2
LATENT_COMPRESSION = 16

# Source (low resolution input) of a training pair: `latents_512` is the bilinear downscale used by the benchmark.
SOURCE_KEY_WEIGHTS = {"latents_512": 1.0}
TARGET_KEY = "latents_1024"

# Architecture (autoresearch_2 recipe): plain window transformer on the full resolution latent grid. DiT style
# adaLN-zero blocks, shifted 8x8 windows with a learned relative position bias, QK-norm, GELU MLP (hidden 4D) with a
# depthwise 3x3 conv on the hidden, RMS normed condition vector, learned source path (residual blocks at the source
# resolution + pixel shuffle) added to the stem features.
TRANSFORMER_WIDTH = 384
TRANSFORMER_DEPTH = 12
TRANSFORMER_HEAD_DIM = 64
MLP_HIDDEN = 4 * TRANSFORMER_WIDTH
QK_NORM = True
CONDITION_NORM = True
SOURCE_ENCODER_BLOCKS = 2
CONDITION_DIM = 512
FREQUENCY_DIM = 256
MAX_NORM_GROUPS = 32
SIGMA_SCALE = 1000.0
WINDOW_SIZE = 8
ROPE_BASE = 100.0
GRADIENT_CHECKPOINTING = False

# "residual": the flow runs on r = (x - bilinear(source) - RESIDUAL_MEAN) / RESIDUAL_STD (per channel); "direct": on x.
PARAMETERIZATION = "residual"
# Per channel std of latents_1024 - bilinear 2x(latents_512) on qwen_4klsdb_40k (dev/autoresearch_4/latent_statistics.py).
# Per channel mean of the same residual (full dataset).
RESIDUAL_MEAN = (-0.090, -0.201, 0.245, -0.706, -0.259, 0.248, -0.441, -0.001, -0.247, -0.694, -0.222, 0.397, -0.355, -0.153, -0.033, 0.038, -0.252, -0.419, 0.029, -0.033, -0.539, 0.246, 0.198, -0.004, -0.327, 0.553, -0.363, 0.614, -0.316, 0.561, -0.072, -0.254, -0.167, -0.562, 0.428, 0.730, -0.134, 0.387, -0.404, 0.144, -0.276, 0.018, -0.080, -0.592, -0.328, -0.370, 0.111, 0.514, 0.259, 0.282, 0.126, 0.000, 0.028, 0.012, 0.769, 0.210, -0.369, 0.207, 0.847, -0.073, -0.487, -0.339, 0.786, -0.258)
RESIDUAL_STD = (0.735, 0.678, 0.635, 0.646, 0.754, 0.618, 0.598, 0.640, 0.746, 0.552, 0.608, 0.791, 0.735, 0.639, 0.788, 0.550, 0.691, 0.656, 0.752, 0.601, 0.764, 0.604, 0.750, 0.584, 0.654, 0.477, 0.709, 0.637, 0.685, 0.578, 0.740, 0.580, 0.655, 0.673, 0.594, 0.739, 0.732, 0.760, 0.773, 0.525, 0.682, 0.528, 0.839, 0.576, 0.643, 0.630, 0.508, 0.600, 0.619, 0.567, 0.723, 0.534, 0.487, 0.531, 0.626, 0.588, 0.616, 0.629, 0.741, 0.722, 0.652, 0.744, 0.718, 0.658)

TRAINING_TIME_SECONDS = 30 * 60
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75)
SMOKE_STEPS = 3
BATCH_SIZE = 16
NUM_WORKERS = 6
PREFETCH_FACTOR = 4
LEARNING_RATE = 1.7e-3
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
DECAY_FRACTION = 0.5
GRADIENT_CLIP_NORM = 1.0
TRAINING_SIGMA_SHIFT = 1.0
# Loss weight of the velocity error above HIGH_FREQUENCY_CUTOFF (cycles per latent pixel, half Nyquist): the Qwen
# decoder turns token level latent errors into colour errors (dev/autoresearch_4/error_decomposition.py).
HIGH_FREQUENCY_LOSS_WEIGHT = 16.0
HIGH_FREQUENCY_CUTOFF = 0.25
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
            latents_target: [C, 64, 64], dtype: torch.float32
            latents_source: [C, 32, 32], dtype: torch.float32
        """
        random_generator = np.random.default_rng([SEED, index])
        dataset_index = int(random_generator.choice(len(self.dataset_block_paths), p=self.dataset_probabilities))
        block_paths = self.dataset_block_paths[dataset_index]
        block_index = int(random_generator.integers(0, len(block_paths)))
        source_key = self.source_keys[int(random_generator.choice(len(self.source_keys), p=self.source_probabilities))]
        encoded_block = vaezip.load(block_paths[block_index])
        latents_target = vaezip.load(BASE_DIRECTORY / block_paths[block_index].name)[BASE_KEY].decode(
            dtype=torch.float32
        )[0]
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


def residual_offset(device: torch.device | str) -> torch.Tensor:
    """
    returns:
        offset: [1, C, 1, 1], dtype: torch.float32
    """
    return torch.tensor(RESIDUAL_MEAN, device=device, dtype=torch.float32)[None, :, None, None]


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
    return (latents_target - align_source(latents_source) - residual_offset(latents_target.device)) / residual_scale(
        latents_target.device
    )


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
    return align_source(latents_source) + residual_offset(samples.device) + residual_scale(samples.device) * samples


def frequency_weighted_mse(error: torch.Tensor) -> torch.Tensor:
    """
    MSE with the spectral energy above HIGH_FREQUENCY_CUTOFF weighted by HIGH_FREQUENCY_LOSS_WEIGHT (equal to the plain
    MSE when the weight is 1, by Parseval).

    args:
        error: [B, C, H, W], dtype: torch.float32
    returns:
        loss: [], dtype: torch.float32
    """
    height, width = error.shape[-2:]
    spectrum = torch.fft.fft2(error, norm="ortho")
    radius = torch.sqrt(
        torch.fft.fftfreq(height, device=error.device)[:, None] ** 2
        + torch.fft.fftfreq(width, device=error.device)[None, :] ** 2
    )  # [H, W]
    weights = torch.where(radius > HIGH_FREQUENCY_CUTOFF, HIGH_FREQUENCY_LOSS_WEIGHT, 1.0)
    return (spectrum.abs().square() * weights).mean()


class CropDecoder:
    """
    Frozen bf16 texture-fix Qwen-Image 2.1 decoder for latent crops, differentiable with respect to the latents.
    """

    def __init__(self) -> None:
        model = AutoencoderKLQwenImage21.from_pretrained(str(DECODER_PATH), torch_dtype=torch.bfloat16)
        self.model = model.to(DEVICE).eval().requires_grad_(False)
        shape = (1, LATENT_CHANNELS, 1, 1, 1)
        self.mean = torch.tensor(model.config.latents_mean, device=DEVICE).view(shape)
        self.std = torch.tensor(model.config.latents_std, device=DEVICE).view(shape)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C, h, w], normalized
        returns:
            pixels: [B, 3, 16h, 16w], channel_order: RGB, pixel_values: [-1...1] (not clamped), dtype: torch.bfloat16
        """
        denormalized = (latents.float().unsqueeze(2) * self.std + self.mean).to(torch.bfloat16)
        return self.model.decode(denormalized, return_dict=False)[0][:, :3, 0]


def decoder_crop_loss(
    decoder: CropDecoder, latents_prediction: torch.Tensor, latents_target: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """
    Pixel MSE between decoded crops of the prediction and the target, on the crop centre, optionally pooled.

    args:
        latents_prediction: [K, C, H, W], differentiable
        latents_target: [K, C, H, W]
    returns:
        loss: [], dtype: torch.float32
    """
    size = DECODER_CROP_CORE + 2 * DECODER_CROP_MARGIN
    height, width = latents_prediction.shape[-2:]
    tops = torch.randint(0, height - size + 1, (latents_prediction.shape[0],), generator=generator).tolist()
    lefts = torch.randint(0, width - size + 1, (latents_prediction.shape[0],), generator=generator).tolist()
    crops_prediction = torch.stack(
        [latents[:, top : top + size, left : left + size] for latents, top, left in zip(latents_prediction, tops, lefts)]
    )  # [K, C, size, size]
    crops_target = torch.stack(
        [latents[:, top : top + size, left : left + size] for latents, top, left in zip(latents_target, tops, lefts)]
    )
    start = LATENT_COMPRESSION * DECODER_CROP_MARGIN
    end = start + LATENT_COMPRESSION * DECODER_CROP_CORE
    pixels_prediction = decoder.decode(crops_prediction)[..., start:end, start:end].float()
    with torch.no_grad():
        pixels_target = decoder.decode(crops_target)[..., start:end, start:end].float()
    if DECODER_LOSS_POOLING > 1:
        pixels_prediction = functional.avg_pool2d(pixels_prediction, DECODER_LOSS_POOLING)
        pixels_target = functional.avg_pool2d(pixels_target, DECODER_LOSS_POOLING)
    return functional.mse_loss(pixels_prediction, pixels_target)


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

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(LATENT_CHANNELS, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(PlainResidualBlock(channels) for _ in range(SOURCE_ENCODER_BLOCKS)))
        self.upsample_norm = make_group_norm(channels)
        self.upsample = nn.Conv2d(channels, 4 * channels, 3, padding=1)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, latents_source: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents_source: [B, C, H, W]
        returns:
            features: [B, channels, 2H, 2W]
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
    DiT style block with adaLN-zero modulation: (shifted) window self-attention with a learned relative position
    bias and QK-norm, GELU MLP with a depthwise 3x3 convolution on the hidden features.
    """

    def __init__(self, dim: int, shift: bool) -> None:
        super().__init__()
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
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(WINDOW_SIZE), torch.arange(WINDOW_SIZE), indexing="ij")
        ).flatten(1).float()  # [2, window tokens]
        quarter = TRANSFORMER_HEAD_DIM // 4
        frequencies = ROPE_BASE ** (-torch.arange(quarter, dtype=torch.float32) / quarter)  # [Dh / 4]
        angles = torch.cat(
            [coordinates[0][:, None] * frequencies[None], coordinates[1][:, None] * frequencies[None]], dim=1
        )  # [window tokens, Dh / 2]
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)
        self.feed_forward_input = nn.Linear(dim, MLP_HIDDEN)
        self.feed_forward_conv = nn.Conv2d(MLP_HIDDEN, MLP_HIDDEN, 3, padding=1, groups=MLP_HIDDEN)
        self.feed_forward_output = nn.Linear(MLP_HIDDEN, dim)

    def _rotate(self, features: torch.Tensor) -> torch.Tensor:
        """
        Axial 2D rotary embedding of the window coordinates (rows on the first half of the pairs, columns on the second).

        args:
            features: [N, heads, window tokens, Dh]
        returns:
            features: [N, heads, window tokens, Dh], same dtype
        """
        cosine = self.rope_cos.to(features.dtype)
        sine = self.rope_sin.to(features.dtype)
        even, odd = features[..., 0::2], features[..., 1::2]
        rotated = torch.stack([even * cosine - odd * sine, even * sine + odd * cosine], dim=-1)
        return rotated.flatten(-2)

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
        query, key = (self.query_norm(qkv[0]), self.key_norm(qkv[1])) if QK_NORM else (qkv[0], qkv[1])
        attended = functional.scaled_dot_product_attention(self._rotate(query), self._rotate(key), qkv[2])
        attended = attended.permute(0, 2, 1, 3).reshape(count, num_windows, window_tokens, dim)
        attended = window_merge(self.attention_output(attended), height, width)
        if self.shift:
            attended = torch.roll(attended, (self.shift, self.shift), dims=(1, 2))
        return attended

    def _feed_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.feed_forward_input(tokens).permute(0, 3, 1, 2)  # [N, hidden, H, W]
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


class UpscalerWindowTransformer(nn.Module):
    """
    Window transformer on the full resolution latent grid: 3x3 convolution stem on the noisy sample and the bilinear
    aligned source, plus the learned source path; TRANSFORMER_DEPTH blocks alternating regular and shifted windows;
    adaLN final layer and zero initialized linear velocity head.
    """

    def __init__(self) -> None:
        super().__init__()
        dim = TRANSFORMER_WIDTH
        self.conditioning = ConditioningEncoder()
        self.input_conv = nn.Conv2d(2 * LATENT_CHANNELS, dim, 3, padding=1)
        self.source_encoder = SourceEncoder(dim)
        self.blocks = nn.ModuleList(TransformerBlock(dim, shift=index % 2 == 1) for index in range(TRANSFORMER_DEPTH))
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(CONDITION_DIM, 2 * dim))
        nn.init.zeros_(self.output_modulation[1].weight)
        nn.init.zeros_(self.output_modulation[1].bias)
        self.output_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.output = nn.Linear(dim, LATENT_CHANNELS)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

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
        latents_source = latents_source.to(noisy_samples.dtype)
        stacked = torch.cat([noisy_samples, align_source(latents_source)], dim=1)  # [B, 2C, 2H, 2W]
        features = self.input_conv(stacked) + self.source_encoder(latents_source)  # [B, D, 2H, 2W]
        tokens = features.permute(0, 2, 3, 1)  # [B, 2H, 2W, D]
        for block in self.blocks:
            if GRADIENT_CHECKPOINTING and self.training:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, condition, use_reentrant=False)
            else:
                tokens = block(tokens, condition)
        shift, scale = self.output_modulation(condition)[:, None, None, :].chunk(2, dim=-1)
        velocity = self.output(self.output_norm(tokens) * (1 + scale) + shift)  # [B, 2H, 2W, C]
        return velocity.permute(0, 3, 1, 2)


def build_model() -> nn.Module:
    return UpscalerWindowTransformer()


COMPILED_MODULE_TYPES = (TransformerBlock, SourceEncoder)


class Run0022Upscaler(Upscaler):
    """
    run_0019 (deterministic regression of the pixel re-encode base) + decoder aware loss (valid here: the target
    is deterministic), weight 1000, 2x pooled pixels: does it close the gap to the exact base.
    """

    def __init__(self) -> None:
        self.model: nn.Module | None = None

    def _compile_training_shapes(self, model: nn.Module) -> None:
        """
        Compiles forward and backward of the training shape before the training clock starts.
        """
        noisy_samples = torch.randn(BATCH_SIZE, LATENT_CHANNELS, 64, 64, device=DEVICE)
        latents_source = torch.randn(BATCH_SIZE, LATENT_CHANNELS, 32, 32, device=DEVICE)
        with torch.autocast(DEVICE, dtype=TRAINING_DTYPE):
            velocity = model(noisy_samples, torch.rand(BATCH_SIZE, device=DEVICE), latents_source)
        velocity.float().square().mean().backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    def train(self, smoke_mode: bool = False) -> None:
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True
        model = build_model().to(DEVICE)
        forward_model = model if smoke_mode else torch.compile(model, mode=TRAINING_COMPILE_MODE, dynamic=False)
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
            self._compile_training_shapes(forward_model)
            print(f"{RUN_NAME}: compiled in {time.perf_counter() - compile_start_time:.0f}s", flush=True)

        summary_writer = None if smoke_mode else SummaryWriter(str(TENSORBOARD_DIRECTORY))
        step = 0
        next_checkpoint_index = 0
        loss_sum = torch.zeros((), device=DEVICE)
        decoder_loss_sum = torch.zeros((), device=DEVICE)
        decoder = CropDecoder()
        crop_generator = torch.Generator().manual_seed(SEED)
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
            flow_target = to_flow_space(latents_target, latents_source)  # [B, C, 64, 64]
            sigmas = torch.zeros(flow_target.shape[0], device=DEVICE)
            noisy_samples = torch.zeros_like(flow_target)
            with torch.autocast(DEVICE, dtype=TRAINING_DTYPE):
                velocity = forward_model(noisy_samples, sigmas, latents_source)
            flow_loss = frequency_weighted_mse(velocity.float() - flow_target)
            latents_prediction = from_flow_space(
                velocity[:DECODER_LOSS_SAMPLES].float(), latents_source[:DECODER_LOSS_SAMPLES]
            )  # [K, C, 64, 64]
            decoder_loss = decoder_crop_loss(
                decoder, latents_prediction, latents_target[:DECODER_LOSS_SAMPLES], crop_generator
            )
            loss = flow_loss + DECODER_LOSS_WEIGHT * decoder_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            gradient_norm_max = torch.maximum(gradient_norm_max, gradient_norm.detach())
            optimizer.step()

            step += 1
            loss_sum += flow_loss.detach()
            decoder_loss_sum += decoder_loss.detach()
            elapsed_seconds = time.perf_counter() - training_start_time
            if step % LOG_INTERVAL_STEPS == 0:
                mean_loss = loss_sum.item() / LOG_INTERVAL_STEPS
                mean_decoder_loss = decoder_loss_sum.item() / LOG_INTERVAL_STEPS
                print(
                    f"step {step} loss {mean_loss:.4f} decoder_loss {mean_decoder_loss:.2e} elapsed {elapsed_seconds:.0f}s",
                    flush=True,
                )
                if summary_writer is not None:
                    summary_writer.add_scalar("train/loss", mean_loss, step)
                    summary_writer.add_scalar("train/decoder_loss", mean_decoder_loss, step)
                    summary_writer.add_scalar("train/elapsed_seconds", elapsed_seconds, step)
                    summary_writer.add_scalar("train/samples", step * BATCH_SIZE, step)
                    summary_writer.add_scalar("train/grad_norm_max", gradient_norm_max.item(), step)
                loss_sum.zero_()
                decoder_loss_sum.zero_()
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
        inputs = torch.zeros(
            (batch_size, channels, 2 * height, 2 * width), device=latents.device, dtype=INFERENCE_DTYPE
        )
        sigmas = torch.zeros(batch_size, device=latents.device)
        prediction = self.model(inputs, sigmas, latents.to(INFERENCE_DTYPE)).float()
        return from_flow_space(prediction, latents.float()).to(latents.dtype)

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
