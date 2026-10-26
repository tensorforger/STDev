import copy
import math
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as functional
import vaezip
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from stdev.upscalers.base import Upscaler

RUN_NAME = "run_0001"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "flux_1024_5k"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "trained_models" / RUN_NAME
TENSORBOARD_DIRECTORY = REPOSITORY_ROOT / "tensorboard"
MODEL_FILE_NAME = "model.safetensors"
EMA_MODEL_FILE_NAME = "ema_model.safetensors"

DEVICE = "cuda"
SEED = 0
LATENT_CHANNELS = 32
SCALE_FACTOR = 2

BASE_CHANNELS = 96
CHANNEL_MULTIPLIERS = (1, 2, 4)
NORM_GROUPS = 8
TIME_EMBEDDING_DIM = 384
SINUSOIDAL_DIM = 128

TRAINING_TIME_SECONDS = 30 * 60
SMOKE_STEPS = 5
BATCH_SIZE = 16
NUM_WORKERS = 8
LEARNING_RATE = 2e-4
WARMUP_STEPS = 200
GRADIENT_CLIP_NORM = 1.0
EMA_DECAY = 0.999
LOG_INTERVAL_STEPS = 100

NUM_SAMPLING_STEPS = 30
INFERENCE_SEED = 0
INFERENCE_DTYPE = torch.bfloat16


class LatentPairDataset(Dataset):
    """
    Pairs of full resolution latents and latents of the image downscaled by 2.
    """

    def __init__(self, dataset_directory: Path) -> None:
        self.block_paths = sorted(dataset_directory.glob("block_*.vz.safetensors"))

    def __len__(self) -> int:
        return len(self.block_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
            latents_full: [C, H, W], dtype: torch.float32
            latents_div_2: [C, H / 2, W / 2], dtype: torch.float32
        """
        encoded_block = vaezip.load(self.block_paths[index])
        latents_full = encoded_block["latents_orig"].decode(dtype=torch.float32)
        latents_div_2 = encoded_block["latents_div_2"].decode(dtype=torch.float32)
        return latents_full[0], latents_div_2[0]


def iterate_batches(
    data_loader: DataLoader,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from data_loader


def upsample_condition(latents_low_resolution: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents_low_resolution: [B, C, H, W]
    returns:
        latents_upsampled: [B, C, H * SCALE_FACTOR, W * SCALE_FACTOR]
    """
    return functional.interpolate(
        latents_low_resolution, scale_factor=SCALE_FACTOR, mode="bilinear", align_corners=False
    )


class TimeEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(SINUSOIDAL_DIM, TIME_EMBEDDING_DIM),
            nn.SiLU(),
            nn.Linear(TIME_EMBEDDING_DIM, TIME_EMBEDDING_DIM),
        )

    def forward(self, time_values: torch.Tensor) -> torch.Tensor:
        """
        args:
            time_values: [B], dtype: torch.float32, values: [0...1]
        returns:
            embedding: [B, TIME_EMBEDDING_DIM]
        """
        half_dim = SINUSOIDAL_DIM // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=time_values.device, dtype=torch.float32)
            / half_dim
        )  # [half_dim]
        angles = time_values[:, None] * 1000.0 * frequencies[None, :]  # [B, half_dim]
        sinusoidal = torch.cat([angles.sin(), angles.cos()], dim=1)  # [B, SINUSOIDAL_DIM]
        return self.projection(sinusoidal.to(self.projection[0].weight.dtype))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm_first = nn.GroupNorm(NORM_GROUPS, in_channels)
        self.conv_first = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(TIME_EMBEDDING_DIM, out_channels)
        self.norm_second = nn.GroupNorm(NORM_GROUPS, out_channels)
        self.conv_second = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, features: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C_in, H, W]
            time_embedding: [B, TIME_EMBEDDING_DIM]
        returns:
            features: [B, C_out, H, W]
        """
        hidden = self.conv_first(functional.silu(self.norm_first(features)))
        hidden = hidden + self.time_projection(time_embedding)[:, :, None, None]
        hidden = self.conv_second(functional.silu(self.norm_second(hidden)))
        return self.skip(features) + hidden


class UpsampleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C_in, H, W]
        returns:
            features: [B, C_out, 2 * H, 2 * W]
        """
        return self.conv(functional.interpolate(features, scale_factor=2.0, mode="nearest"))


class ConditionalUNet(nn.Module):
    """
    Small convolutional UNet predicting the flow velocity of high resolution latents.
    Low resolution latents are upsampled and concatenated to the noisy input channels.
    """

    def __init__(self) -> None:
        super().__init__()
        level_channels = [BASE_CHANNELS * multiplier for multiplier in CHANNEL_MULTIPLIERS]
        self.time_embedding = TimeEmbedding()
        self.input_conv = nn.Conv2d(2 * LATENT_CHANNELS, level_channels[0], 3, padding=1)

        encoder_input_channels = [level_channels[0]] + level_channels[:-1]
        self.encoder_blocks = nn.ModuleList(
            ResidualBlock(in_channels, out_channels)
            for in_channels, out_channels in zip(encoder_input_channels, level_channels)
        )
        self.downsamples = nn.ModuleList(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1)
            for channels in level_channels[:-1]
        )
        self.middle_block = ResidualBlock(level_channels[-1], level_channels[-1])
        self.upsamples = nn.ModuleList(
            UpsampleConv(deeper_channels, channels)
            for deeper_channels, channels in zip(level_channels[1:], level_channels[:-1])
        )
        self.decoder_blocks = nn.ModuleList(
            ResidualBlock(2 * channels, channels) for channels in level_channels
        )

        self.output_norm = nn.GroupNorm(NORM_GROUPS, level_channels[0])
        self.output_conv = nn.Conv2d(level_channels[0], LATENT_CHANNELS, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        condition: torch.Tensor,
        time_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            noisy_latents: [B, C, H, W]
            condition: [B, C, H, W], low resolution latents upsampled to H, W
            time_values: [B], dtype: torch.float32, values: [0...1], 1 is pure noise
        returns:
            velocity: [B, C, H, W]
        """
        time_embedding = self.time_embedding(time_values)
        features = self.input_conv(torch.cat([noisy_latents, condition], dim=1))

        skips = []
        for level, block in enumerate(self.encoder_blocks):
            features = block(features, time_embedding)
            skips.append(features)
            if level < len(self.downsamples):
                features = self.downsamples[level](features)

        features = self.middle_block(features, time_embedding)

        for level in reversed(range(len(self.decoder_blocks))):
            if level < len(self.upsamples):
                features = self.upsamples[level](features)
            features = self.decoder_blocks[level](
                torch.cat([features, skips[level]], dim=1), time_embedding
            )

        return self.output_conv(functional.silu(self.output_norm(features)))


class Run0001Upscaler(Upscaler):
    """
    Run 0000 with logit-normal time sampling replaced by uniform sampling: flow matching with a small conditional UNet, fp32 training, Euler sampling.
    """

    def __init__(self) -> None:
        self.model: nn.Module | None = None

    def train(self, smoke_mode: bool = False) -> None:
        torch.manual_seed(SEED)
        model = ConditionalUNet().to(DEVICE)
        ema_model = copy.deepcopy(model).requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
        data_loader = DataLoader(
            LatentPairDataset(DATASET_DIRECTORY),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"{RUN_NAME}: {parameter_count / 1e6:.2f}M parameters", flush=True)

        run_log_name = f"{RUN_NAME}_smoke" if smoke_mode else RUN_NAME
        summary_writer = SummaryWriter(str(TENSORBOARD_DIRECTORY / run_log_name))
        batches = iterate_batches(data_loader)
        latents_full, latents_div_2 = next(batches)
        training_start_time = time.perf_counter()
        step = 0
        loss_sum = torch.zeros((), device=DEVICE)
        while True:
            latents_full = latents_full.to(DEVICE, non_blocking=True)  # [B, C, H, W]
            condition = upsample_condition(latents_div_2.to(DEVICE, non_blocking=True))
            time_values = torch.rand(latents_full.shape[0], device=DEVICE)  # [B]
            noise = torch.randn_like(latents_full)
            blend = time_values[:, None, None, None]
            noisy_latents = (1.0 - blend) * latents_full + blend * noise

            for group in optimizer.param_groups:
                group["lr"] = LEARNING_RATE * min(1.0, (step + 1) / WARMUP_STEPS)
            velocity = model(noisy_latents, condition, time_values)
            loss = functional.mse_loss(velocity, noise - latents_full)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()

            ema_decay = min(EMA_DECAY, (1.0 + step) / (10.0 + step))
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
                    ema_parameter.lerp_(parameter, 1.0 - ema_decay)

            step += 1
            loss_sum += loss.detach()
            elapsed_seconds = time.perf_counter() - training_start_time
            if step % LOG_INTERVAL_STEPS == 0:
                mean_loss = loss_sum.item() / LOG_INTERVAL_STEPS
                print(
                    f"step {step} loss {mean_loss:.4f} elapsed {elapsed_seconds:.0f}s", flush=True
                )
                summary_writer.add_scalar("train/loss", mean_loss, step)
                loss_sum.zero_()

            if smoke_mode:
                is_finished = step >= SMOKE_STEPS
            else:
                is_finished = elapsed_seconds >= TRAINING_TIME_SECONDS
            if is_finished:
                break
            latents_full, latents_div_2 = next(batches)

        summary_writer.close()
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        save_file(model.state_dict(), str(OUTPUT_DIRECTORY / MODEL_FILE_NAME))
        save_file(ema_model.state_dict(), str(OUTPUT_DIRECTORY / EMA_MODEL_FILE_NAME))
        print(f"{RUN_NAME}: saved weights after {step} steps to {OUTPUT_DIRECTORY}", flush=True)

    def prepare_to_benchmark(self) -> None:
        model = ConditionalUNet()
        model.load_state_dict(load_file(str(OUTPUT_DIRECTORY / EMA_MODEL_FILE_NAME)))
        self.model = model.to(DEVICE, INFERENCE_DTYPE).eval().requires_grad_(False)

    def compile(self) -> None:
        self.model = torch.compile(self.model)

    @torch.no_grad()
    def upscale(self, tensor: torch.Tensor, prompt: str) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            prompt: text description of the image, unused
        returns:
            tensor: [B, C, H_lr * SCALE_FACTOR, W_lr * SCALE_FACTOR], same dtype as the input
        """
        condition = upsample_condition(tensor.to(INFERENCE_DTYPE))  # [B, C, H_hr, W_hr]
        generator = torch.Generator(device=tensor.device).manual_seed(INFERENCE_SEED)
        latents = torch.randn(
            condition.shape, generator=generator, device=tensor.device, dtype=torch.float32
        )
        sampling_times = torch.linspace(1.0, 0.0, NUM_SAMPLING_STEPS + 1, device=tensor.device)
        for current_time, next_time in zip(sampling_times[:-1], sampling_times[1:]):
            velocity = self.model(
                latents.to(INFERENCE_DTYPE), condition, current_time.expand(latents.shape[0])
            )
            latents = latents + (next_time - current_time) * velocity.float()
        return latents.to(tensor.dtype)

    def get_name(self) -> str:
        return RUN_NAME
