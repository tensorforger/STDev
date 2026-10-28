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
DATASET_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "flux_1024_20k"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "trained_models" / RUN_NAME
TENSORBOARD_DIRECTORY = REPOSITORY_ROOT / "tensorboard" / RUN_NAME
MODEL_FILE_NAME = "model.safetensors"
RAW_MODEL_FILE_NAME = "model_raw.safetensors"

DEVICE = "cuda"
SEED = 0
LATENT_CHANNELS = 32
SCALE_FACTOR = 2

LEVEL_CHANNELS = (192, 384, 768)
BLOCKS_PER_LEVEL = 2
MIDDLE_BLOCKS = 2
CONDITION_DIM = 1024
TIME_FREQUENCY_DIM = 512
MAX_NORM_GROUPS = 32
TIMESTEP_SCALE = 1000.0

TRAINING_TIME_SECONDS = 30 * 60
SMOKE_STEPS = 5
BATCH_SIZE = 16
NUM_WORKERS = 8
LEARNING_RATE = 2e-4
ADAM_BETAS = (0.9, 0.95)
WARMUP_STEPS = 100
GRADIENT_CLIP_NORM = 1.0
EMA_MAX_DECAY = 0.9999
SIGMA_SHIFT = 3.0
LOG_INTERVAL_STEPS = 100
TRAINING_DTYPE = torch.bfloat16

NUM_SAMPLING_STEPS = 20
INFERENCE_SEED = 0
INFERENCE_DTYPE = torch.bfloat16


class LatentPairDataset(Dataset):
    """
    Full resolution latents paired with the 2x downscaled latents.
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
        latents_full, latents_div_2 = (
            encoded_block[name].decode(dtype=torch.float32)[0]
            for name in ("latents_orig", "latents_div_2")
        )
        return latents_full, latents_div_2


def iterate_batches(data_loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from data_loader


def shift_sigmas(sigmas: torch.Tensor) -> torch.Tensor:
    """
    Moves noise levels towards pure noise, as in SD3.

    args:
        sigmas: [...], values: [0...1], 1 is pure noise
    returns:
        shifted_sigmas: [...], values: [0...1], 1 is pure noise
    """
    return SIGMA_SHIFT * sigmas / (1.0 + (SIGMA_SHIFT - 1.0) * sigmas)


def sample_training_sigmas(batch_size: int, device: str) -> torch.Tensor:
    """
    returns:
        sigmas: [B], values: [0...1], logit-normal distributed and shifted
    """
    return shift_sigmas(torch.sigmoid(torch.randn(batch_size, device=device)))


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


class UpscalerUNet(nn.Module):
    """
    Convolutional UNet without attention predicting the flow velocity of high resolution latents.
    Bilinearly upsampled low resolution latents are concatenated to the noisy latents.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conditioning = ConditioningEncoder()
        self.input_conv = nn.Conv2d(2 * LATENT_CHANNELS, LEVEL_CHANNELS[0], 3, padding=1)
        self.encoder_stages = nn.ModuleList(
            ResidualBlockSequence(LEVEL_CHANNELS[level], LEVEL_CHANNELS[level], BLOCKS_PER_LEVEL)
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
                BLOCKS_PER_LEVEL,
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
        latents_upsampled = functional.interpolate(
            latents_low_resolution, size=noisy_latents.shape[-2:], mode="bilinear"
        )  # [B, C, H, W]
        features = self.input_conv(torch.cat([noisy_latents, latents_upsampled], dim=1))

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
        return self.output_conv(functional.silu(self.output_norm(features)))


class Run0001Upscaler(Upscaler):
    """
    Baseline with an exponential moving average of the weights (ramped decay) used for inference.
    """

    def __init__(self) -> None:
        self.model: nn.Module | None = None

    def train(self, smoke_mode: bool = False) -> None:
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True
        model = UpscalerUNet().to(DEVICE)
        compiled_model = torch.compile(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / WARMUP_STEPS, total_iters=WARMUP_STEPS
        )
        data_loader = DataLoader(
            LatentPairDataset(DATASET_DIRECTORY),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        parameters = list(model.parameters())
        ema_parameters = [parameter.detach().clone() for parameter in parameters]
        parameter_count = sum(parameter.numel() for parameter in parameters)
        print(f"{RUN_NAME}: {parameter_count / 1e6:.2f}M parameters", flush=True)

        summary_writer = None if smoke_mode else SummaryWriter(str(TENSORBOARD_DIRECTORY))
        model.train()
        batches = iterate_batches(data_loader)
        training_start_time = None
        step = 0
        loss_sum = torch.zeros((), device=DEVICE)
        for batch in batches:
            latents_full, latents_div_2 = (
                tensor.to(DEVICE, non_blocking=True) for tensor in batch
            )  # [B, C, H, W], [B, C, H / 2, W / 2]
            noise = torch.randn_like(latents_full)  # [B, C, H, W]
            sigmas = sample_training_sigmas(latents_full.shape[0], DEVICE)  # [B]
            blend = sigmas[:, None, None, None]
            noisy_latents = (1.0 - blend) * latents_full + blend * noise

            with torch.autocast(DEVICE, dtype=TRAINING_DTYPE):
                velocity = compiled_model(noisy_latents, sigmas, latents_div_2)
            loss = functional.mse_loss(velocity.float(), noise - latents_full)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            warmup_scheduler.step()
            ema_decay = min(EMA_MAX_DECAY, (1.0 + step) / (10.0 + step))
            torch._foreach_lerp_(
                ema_parameters, [parameter.detach() for parameter in parameters], 1.0 - ema_decay
            )

            if training_start_time is None:
                torch.cuda.synchronize()
                training_start_time = time.perf_counter()
                continue

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

        if summary_writer is not None:
            summary_writer.close()
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        save_file(model.state_dict(), str(OUTPUT_DIRECTORY / RAW_MODEL_FILE_NAME))
        with torch.no_grad():
            for parameter, ema_parameter in zip(parameters, ema_parameters):
                parameter.copy_(ema_parameter)
        save_file(model.state_dict(), str(OUTPUT_DIRECTORY / MODEL_FILE_NAME))
        print(f"{RUN_NAME}: saved weights after {step} steps to {OUTPUT_DIRECTORY}", flush=True)

    def prepare_to_benchmark(self) -> None:
        model = UpscalerUNet()
        model.load_state_dict(load_file(str(OUTPUT_DIRECTORY / MODEL_FILE_NAME)))
        self.model = model.to(DEVICE, INFERENCE_DTYPE).eval().requires_grad_(False)

    def compile(self) -> None:
        self.model = torch.compile(self.model, mode="reduce-overhead")

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
        sampling_sigmas = shift_sigmas(
            torch.linspace(1.0, 0.0, NUM_SAMPLING_STEPS + 1, device=tensor.device)
        )
        for current_sigma, next_sigma in zip(sampling_sigmas[:-1], sampling_sigmas[1:]):
            velocity = self.model(
                latents.to(INFERENCE_DTYPE),
                current_sigma.expand(batch_size),
                latents_low_resolution,
            )
            latents = latents + (next_sigma - current_sigma) * velocity.float()
        return latents.to(tensor.dtype)

    @torch.no_grad()
    def single_forward(self, tensor: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            prompt_embeds: [B, 77, 768], dtype = float32, unused
        returns:
            velocity: [B, C, H_lr * SCALE_FACTOR, W_lr * SCALE_FACTOR], dtype: INFERENCE_DTYPE
        """
        batch_size, channels, height, width = tensor.shape
        noisy_latents = torch.randn(
            (batch_size, channels, height * SCALE_FACTOR, width * SCALE_FACTOR),
            device=tensor.device,
            dtype=INFERENCE_DTYPE,
        )
        sigmas = torch.ones(batch_size, device=tensor.device)
        return self.model(noisy_latents, sigmas, tensor.to(INFERENCE_DTYPE))

    def get_name(self) -> str:
        return RUN_NAME
