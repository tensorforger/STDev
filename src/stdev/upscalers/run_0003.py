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

RUN_NAME = "run_0003"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "flux_1024_1k"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "trained_models" / RUN_NAME
TENSORBOARD_DIRECTORY = REPOSITORY_ROOT / "tensorboard" / RUN_NAME
MODEL_FILE_NAME = "model.safetensors"

DEVICE = "cuda"
SEED = 0
LATENT_CHANNELS = 32
SCALE_FACTOR = 2

BASE_CHANNELS = 384
CONDITION_DIM = 1024
TIME_FREQUENCY_DIM = 512
MAX_NORM_GROUPS = 32
DROPOUT = 0.01
TIMESTEP_SCALE = 1000.0

TRAINING_TIME_SECONDS = 30 * 60
SMOKE_STEPS = 5
BATCH_SIZE = 16
NUM_WORKERS = 8
LEARNING_RATE = 1e-4
LOG_INTERVAL_STEPS = 100
TRAINING_DTYPE = torch.bfloat16

NUM_SAMPLING_STEPS = 1
INFERENCE_SEED = 0
INFERENCE_DTYPE = torch.bfloat16


class LatentTripletDataset(Dataset):
    """
    Teacher initial noise paired with the full resolution latents and the 2x downscaled latents.
    """

    def __init__(self, dataset_directory: Path) -> None:
        self.block_paths = sorted(dataset_directory.glob("block_*.vz.safetensors"))

    def __len__(self) -> int:
        return len(self.block_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
            noise: [C, H, W], dtype: torch.float32
            latents_full: [C, H, W], dtype: torch.float32
            latents_div_2: [C, H / 2, W / 2], dtype: torch.float32
        """
        encoded_block = vaezip.load(self.block_paths[index])
        noise, latents_full, latents_div_2 = (
            encoded_block[name].decode(dtype=torch.float32)[0]
            for name in ("init_latents", "latents_orig", "latents_div_2")
        )
        return noise, latents_full, latents_div_2


def iterate_batches(
    data_loader: DataLoader,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    while True:
        yield from data_loader


def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(MAX_NORM_GROUPS, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=1e-6)


class ConditioningEncoder(nn.Module):
    """
    Sinusoidal embedding of the noise level followed by an MLP.
    """

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(TIME_FREQUENCY_DIM, CONDITION_DIM),
            nn.SiLU(),
            nn.Linear(CONDITION_DIM, CONDITION_DIM),
        )

    def forward(self, sigmas: torch.Tensor) -> torch.Tensor:
        """
        args:
            sigmas: [B], values: [0...1], 1 is pure noise
        returns:
            condition: [B, CONDITION_DIM]
        """
        half_dim = TIME_FREQUENCY_DIM // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=sigmas.device, dtype=torch.float32)
            / half_dim
        )  # [half_dim]
        angles = sigmas.float()[:, None] * TIMESTEP_SCALE * frequencies[None, :]  # [B, half_dim]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)  # [B, TIME_FREQUENCY_DIM]
        return self.projection(embedding.to(self.projection[0].weight.dtype))


class ConditionedResidualBlock(nn.Module):
    """
    Residual block with the condition injected as scale and shift of the second normalization.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm_first = make_group_norm(in_channels)
        self.conv_first = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.condition_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(CONDITION_DIM, 2 * out_channels)
        )
        self.norm_second = make_group_norm(out_channels)
        self.dropout = nn.Dropout(DROPOUT)
        self.conv_second = nn.Conv2d(out_channels, out_channels, 3, padding=1)
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
        hidden = self.conv_second(self.dropout(functional.silu(hidden)))
        return self.skip(features) + hidden


class ResidualBlockSequence(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ConditionedResidualBlock(in_channels if index == 0 else out_channels, out_channels)
            for index in range(num_blocks)
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            features = block(features, condition)
        return features


class DownStage(nn.Module):
    """
    Residual blocks followed by a strided convolution, also returns the features before it.
    """

    def __init__(self, num_blocks: int) -> None:
        super().__init__()
        self.blocks = ResidualBlockSequence(BASE_CHANNELS, BASE_CHANNELS, num_blocks)
        self.downsample = nn.Conv2d(BASE_CHANNELS, BASE_CHANNELS, 3, stride=2, padding=1)

    def forward(
        self, features: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        args:
            features: [B, C, H, W]
            condition: [B, CONDITION_DIM]
        returns:
            downsampled: [B, C, H / 2, W / 2]
            skip: [B, C, H, W]
        """
        skip = self.blocks(features, condition)
        return self.downsample(skip), skip


class UpStage(nn.Module):
    """
    Bilinear upsampling to the skip resolution, concatenation with the skip and residual blocks.
    """

    def __init__(self, num_blocks: int) -> None:
        super().__init__()
        self.blocks = ResidualBlockSequence(2 * BASE_CHANNELS, BASE_CHANNELS, num_blocks)

    def forward(
        self, features: torch.Tensor, skip: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """
        args:
            features: [B, C, H / 2, W / 2]
            skip: [B, C, H, W]
            condition: [B, CONDITION_DIM]
        returns:
            features: [B, C, H, W]
        """
        features = functional.interpolate(
            features, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.blocks(torch.cat([features, skip], dim=1), condition)


class LowResolutionEncoder(nn.Module):
    """
    Encodes the low resolution latents into feature maps at three scales.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(LATENT_CHANNELS, BASE_CHANNELS, 1)
        self.block = ConditionedResidualBlock(BASE_CHANNELS, BASE_CHANNELS)
        self.downsamples = nn.ModuleList(
            nn.Conv2d(BASE_CHANNELS, BASE_CHANNELS, 3, stride=2, padding=1) for _ in range(2)
        )
        self.downsampled_blocks = nn.ModuleList(
            ConditionedResidualBlock(BASE_CHANNELS, BASE_CHANNELS) for _ in range(2)
        )

    def forward(
        self, latents_low_resolution: torch.Tensor, condition: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        args:
            latents_low_resolution: [B, C, H, W]
            condition: [B, CONDITION_DIM]
        returns:
            feature_maps: [[B, C_f, H, W], [B, C_f, H / 2, W / 2], [B, C_f, H / 4, W / 4]]
        """
        features = self.block(self.input_conv(latents_low_resolution), condition)
        feature_maps = [features]
        for downsample, block in zip(self.downsamples, self.downsampled_blocks):
            features = block(downsample(features), condition)
            feature_maps.append(features)
        return feature_maps


class FilmModulation(nn.Module):
    """
    Spatial scale and shift of the features predicted from the low resolution feature map.
    """

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.SiLU(), nn.Conv2d(BASE_CHANNELS, 2 * BASE_CHANNELS, 1))

    def forward(self, features: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C, H, W]
            guidance: [B, C, H, W]
        returns:
            features: [B, C, H, W]
        """
        scale, shift = self.projection(guidance).chunk(2, dim=1)
        return features * (1 + scale) + shift


class UpscalerUNet(nn.Module):
    """
    UNet predicting the flow velocity of high resolution latents.
    Low resolution latents are injected through FiLM modulation at three scales.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conditioning = ConditioningEncoder()
        self.input_conv = nn.Conv2d(LATENT_CHANNELS, BASE_CHANNELS, 1)
        self.low_resolution_encoder = LowResolutionEncoder()
        self.modulations = nn.ModuleList(FilmModulation() for _ in range(3))
        self.down_stages = nn.ModuleList([DownStage(num_blocks=3), DownStage(num_blocks=2)])
        self.middle_block = ConditionedResidualBlock(BASE_CHANNELS, BASE_CHANNELS)
        self.up_stages = nn.ModuleList([UpStage(num_blocks=2), UpStage(num_blocks=3)])
        self.output_conv = nn.Conv2d(BASE_CHANNELS, LATENT_CHANNELS, 1)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        latents_low_resolution: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            noisy_latents: [B, C, H, W]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_low_resolution: [B, C, H / 2, W / 2]
        returns:
            velocity: [B, C, H, W]
        """
        condition = self.conditioning(sigmas)
        feature_maps = self.low_resolution_encoder(latents_low_resolution, condition)
        height, width = noisy_latents.shape[-2:]
        guidances = [
            functional.interpolate(
                feature_map,
                size=(height // 2**level, width // 2**level),
                mode="bilinear",
                align_corners=False,
            )
            for level, feature_map in enumerate(feature_maps)
        ]

        features = self.modulations[0](self.input_conv(noisy_latents), guidances[0])
        skips = []
        for level, down_stage in enumerate(self.down_stages):
            features, skip = down_stage(features, condition)
            skips.append(skip)
            features = self.modulations[level + 1](features, guidances[level + 1])

        features = self.middle_block(features, condition)
        for up_stage in self.up_stages:
            features = up_stage(features, skips.pop(), condition)
        return self.output_conv(features)


class Run0003Upscaler(Upscaler):
    """
    Endpoint flow matching with dihedral spatial augmentation for one-step sampling.
    """

    def __init__(self) -> None:
        self.model: nn.Module | None = None

    def train(self, smoke_mode: bool = False) -> None:
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True
        model = UpscalerUNet().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        data_loader = DataLoader(
            LatentTripletDataset(DATASET_DIRECTORY),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"{RUN_NAME}: {parameter_count / 1e6:.2f}M parameters", flush=True)

        summary_writer = None if smoke_mode else SummaryWriter(str(TENSORBOARD_DIRECTORY))
        model.train()
        batches = iterate_batches(data_loader)
        batch = next(batches)
        training_start_time = time.perf_counter()
        step = 0
        loss_sum = torch.zeros((), device=DEVICE)
        while True:
            noise, latents_full, latents_div_2 = (
                tensor.to(DEVICE, non_blocking=True) for tensor in batch
            )  # [B, C, H, W], [B, C, H, W], [B, C, H / 2, W / 2]
            augmentation_index = torch.randint(0, 8, ()).item()
            if augmentation_index & 1:
                noise, latents_full, latents_div_2 = (
                    torch.flip(tensor, dims=(-1,))
                    for tensor in (noise, latents_full, latents_div_2)
                )
            if augmentation_index & 2:
                noise, latents_full, latents_div_2 = (
                    torch.flip(tensor, dims=(-2,))
                    for tensor in (noise, latents_full, latents_div_2)
                )
            if augmentation_index & 4:
                noise, latents_full, latents_div_2 = (
                    tensor.transpose(-1, -2)
                    for tensor in (noise, latents_full, latents_div_2)
                )
            sigmas = torch.ones(latents_full.shape[0], device=DEVICE)  # [B]
            blend = sigmas[:, None, None, None]
            noisy_latents = (1.0 - blend) * latents_full + blend * noise

            with torch.autocast(DEVICE, dtype=TRAINING_DTYPE):
                velocity = model(noisy_latents, sigmas, latents_div_2)
            loss = functional.mse_loss(velocity.float(), noise - latents_full)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
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

            if smoke_mode:
                is_finished = step >= SMOKE_STEPS
            else:
                is_finished = elapsed_seconds >= TRAINING_TIME_SECONDS
            if is_finished:
                break
            batch = next(batches)

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
        self.model = torch.compile(self.model)

    @torch.no_grad()
    def upscale(self, tensor: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            prompt_embeds: [B, 77, 768], dtype = float32, SDXL text encoder embeddings, unused
        returns:
            tensor: [B, C, H_lr * SCALE_FACTOR, W_lr * SCALE_FACTOR], same dtype as the input
        """
        batch_size, channels, height, width = tensor.shape
        generator = torch.Generator(device=tensor.device).manual_seed(INFERENCE_SEED)
        latents = torch.randn(
            (batch_size, channels, height * SCALE_FACTOR, width * SCALE_FACTOR),
            generator=generator,
            device=tensor.device,
        )  # [B, C, H_hr, W_hr], dtype: torch.float32
        latents_low_resolution = tensor.to(INFERENCE_DTYPE)
        sampling_sigmas = torch.linspace(1.0, 0.0, NUM_SAMPLING_STEPS + 1, device=tensor.device)
        for current_sigma, next_sigma in zip(sampling_sigmas[:-1], sampling_sigmas[1:]):
            velocity = self.model(
                latents.to(INFERENCE_DTYPE),
                current_sigma.expand(batch_size),
                latents_low_resolution,
            )
            latents = latents + (next_sigma - current_sigma) * velocity.float()
        return latents.to(tensor.dtype)

    def get_name(self) -> str:
        return RUN_NAME
