import math
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils.checkpoint
import vaezip
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from stdev.upscalers.base import Upscaler

RUN_NAME = "run_0077"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "flux_1024_20k"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "trained_models" / RUN_NAME
TENSORBOARD_DIRECTORY = REPOSITORY_ROOT / "tensorboard" / RUN_NAME
MODEL_FILE_NAME = "model.safetensors"

DEVICE = "cuda"
SEED = 0
LATENT_CHANNELS = 32
SCALE_FACTOR = 2

LEVEL_CHANNELS = (512, 256, 512)
LEVEL_BLOCKS = (1, 1, 1)
DECODER_LEVEL_BLOCKS = (8, 1, 1)
MIDDLE_BLOCKS = 1
ATTENTION_LEVELS = (False, True, True)
TRANSFORMER_MLP = False
MLP_EXPANSION = 4
GRADIENT_CHECKPOINTING = False
CONDITION_DIM = 1024
TIME_FREQUENCY_DIM = 512
MAX_NORM_GROUPS = 32
TIMESTEP_SCALE = 1000.0
ATTENTION_HEAD_DIM = 64
WINDOW_SIZE = 8
TEXT_EMBEDDING_DIM = 768

TRAINING_TIME_SECONDS = 120 * 60
CHECKPOINT_TIMES_SECONDS = (30 * 60, 60 * 60, 90 * 60)
SMOKE_STEPS = 5
BATCH_SIZE = 16
NUM_WORKERS = 8
LEARNING_RATE = 4e-4
ADAM_BETAS = (0.9, 0.95)
WARMUP_STEPS = 100
DECAY_FRACTION = 0.3
GRADIENT_CLIP_NORM = 1.0
TRAINING_SIGMA_SHIFT = 1.0
SAMPLING_SIGMA_SHIFT = 3.0
LOG_INTERVAL_STEPS = 100
TRAINING_DTYPE = torch.bfloat16

NUM_SAMPLING_STEPS = 14
INFERENCE_SEED = 0
INFERENCE_DTYPE = torch.bfloat16


class LatentPairDataset(Dataset):
    """
    Full resolution latents paired with the 2x downscaled latents and the prompt embeddings.
    """

    def __init__(self, dataset_directory: Path) -> None:
        self.block_paths = sorted(dataset_directory.glob("block_*.vz.safetensors"))

    def __len__(self) -> int:
        return len(self.block_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
            latents_full: [C, H, W], dtype: torch.float32
            latents_div_2: [C, H / 2, W / 2], dtype: torch.float32
            prompt_embeds: [77, 768], dtype: torch.float32
        """
        encoded_block = vaezip.load(self.block_paths[index])
        latents_full, latents_div_2, prompt_embeds = (
            encoded_block[name].decode(dtype=torch.float32)[0]
            for name in ("latents_orig", "latents_div_2", "prompt_embeds")
        )
        return latents_full, latents_div_2, prompt_embeds


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
    return shift_sigmas(
        torch.sigmoid(torch.randn(batch_size, device=device)), TRAINING_SIGMA_SHIFT
    )


def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(MAX_NORM_GROUPS, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=1e-6)


class ConditioningEncoder(nn.Module):
    """
    Sinusoidal embedding of the noise level followed by an MLP, plus a projection of the mean
    pooled prompt embeddings.
    """

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(TIME_FREQUENCY_DIM, CONDITION_DIM),
            nn.SiLU(),
            nn.Linear(CONDITION_DIM, CONDITION_DIM),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(TEXT_EMBEDDING_DIM, CONDITION_DIM),
            nn.SiLU(),
            nn.Linear(CONDITION_DIM, CONDITION_DIM),
        )

    def forward(self, sigmas: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        args:
            sigmas: [B], values: [0...1], 1 is pure noise
            prompt_embeds: [B, 77, TEXT_EMBEDDING_DIM]
        returns:
            condition: [B, CONDITION_DIM]
        """
        weight_dtype = self.projection[0].weight.dtype
        text_condition = self.text_projection(prompt_embeds.mean(dim=1).to(weight_dtype))
        half_dim = TIME_FREQUENCY_DIM // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=sigmas.device, dtype=torch.float32)
            / half_dim
        )  # [half_dim]
        angles = sigmas.float()[:, None] * TIMESTEP_SCALE * frequencies[None, :]  # [B, half_dim]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)  # [B, TIME_FREQUENCY_DIM]
        return self.projection(embedding.to(weight_dtype)) + text_condition


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


class WindowAttention(nn.Module):
    """
    Multi-head self-attention inside non-overlapping WINDOW_SIZE x WINDOW_SIZE windows (linear in
    image area). The output projection is zero initialized, so the module starts as the identity.
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
        hidden = hidden.permute(1, 0, 2, 4, 6, 5, 7, 3).reshape(
            3, batch_size, heads, rows * columns, WINDOW_SIZE * WINDOW_SIZE, ATTENTION_HEAD_DIM
        )  # [3, B, heads, windows, window tokens, D]
        attended = functional.scaled_dot_product_attention(hidden[0], hidden[1], hidden[2])
        attended = attended.reshape(
            batch_size, heads, rows, columns, WINDOW_SIZE, WINDOW_SIZE, ATTENTION_HEAD_DIM
        ).permute(0, 1, 6, 2, 4, 3, 5)  # [B, heads, D, rows, window, columns, window]
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
        """
        args:
            features: [B, C, H, W]
        returns:
            features: [B, C, H, W]
        """
        return features + self.contract(functional.gelu(self.expand(self.norm(features))))


class WindowTransformerBlock(nn.Module):
    """
    Window attention, optionally followed by a pointwise MLP (SDXL style transformer block).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = WindowAttention(channels)
        self.feed_forward = PointwiseFeedForward(channels) if TRANSFORMER_MLP else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C, H, W]
        returns:
            features: [B, C, H, W]
        """
        return self.feed_forward(self.attention(features))


class ResidualBlockSequence(nn.Module):
    """
    Residual blocks, each optionally followed by a window transformer block.
    """

    def __init__(
        self, in_channels: int, out_channels: int, num_blocks: int, use_attention: bool
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ConditionedResidualBlock(in_channels if index == 0 else out_channels, out_channels)
            for index in range(num_blocks)
        )
        self.attentions = nn.ModuleList(
            WindowTransformerBlock(out_channels) if use_attention else nn.Identity()
            for _ in range(num_blocks)
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C_in, H, W]
            condition: [B, CONDITION_DIM]
        returns:
            features: [B, C_out, H, W]
        """
        for block, attention in zip(self.blocks, self.attentions):
            if GRADIENT_CHECKPOINTING and self.training:
                features = torch.utils.checkpoint.checkpoint(
                    block, features, condition, use_reentrant=False
                )
                features = torch.utils.checkpoint.checkpoint(
                    attention, features, use_reentrant=False
                )
            else:
                features = attention(block(features, condition))
        return features


class UpscalerUNet(nn.Module):
    """
    Convolutional UNet with window attention predicting the flow velocity of high resolution
    latents. Bilinearly upsampled low resolution latents are concatenated to the noisy latents.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conditioning = ConditioningEncoder()
        self.input_conv = nn.Conv2d(2 * LATENT_CHANNELS, LEVEL_CHANNELS[0], 3, padding=1)
        self.encoder_stages = nn.ModuleList(
            ResidualBlockSequence(
                LEVEL_CHANNELS[level],
                LEVEL_CHANNELS[level],
                LEVEL_BLOCKS[level],
                ATTENTION_LEVELS[level],
            )
            for level in range(len(LEVEL_CHANNELS))
        )
        self.downsamples = nn.ModuleList(
            nn.Conv2d(LEVEL_CHANNELS[level], LEVEL_CHANNELS[level + 1], 3, stride=2, padding=1)
            for level in range(len(LEVEL_CHANNELS) - 1)
        )
        self.middle_blocks = ResidualBlockSequence(
            LEVEL_CHANNELS[-1], LEVEL_CHANNELS[-1], MIDDLE_BLOCKS, ATTENTION_LEVELS[-1]
        )
        last_level = len(LEVEL_CHANNELS) - 1
        self.decoder_stages = nn.ModuleList(
            ResidualBlockSequence(
                LEVEL_CHANNELS[min(level + 1, last_level)] + LEVEL_CHANNELS[level],
                LEVEL_CHANNELS[level],
                DECODER_LEVEL_BLOCKS[level],
                ATTENTION_LEVELS[level],
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
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            noisy_latents: [B, C, H, W]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_low_resolution: [B, C, H / 2, W / 2]
            prompt_embeds: [B, 77, TEXT_EMBEDDING_DIM]
        returns:
            velocity: [B, C, H, W]
        """
        condition = self.conditioning(sigmas, prompt_embeds)
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


class Run0077Upscaler(Upscaler):
    """
    2 hour scaling run (checkpoints at 30/60/90 min), stacked recipe (training shift 1, LR 4e-4,
    linear decay over the last 30%, pooled text condition, window attention after every residual
    block on ATTENTION_LEVELS): deep full resolution decoder, levels (512, 256, 512), decoder blocks
    (8, 1, 1).
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
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"{RUN_NAME}: {parameter_count / 1e6:.2f}M parameters", flush=True)

        summary_writer = None if smoke_mode else SummaryWriter(str(TENSORBOARD_DIRECTORY))
        model.train()
        batches = iterate_batches(data_loader)
        training_start_time = None
        step = 0
        next_checkpoint_index = 0
        loss_sum = torch.zeros((), device=DEVICE)
        for batch in batches:
            latents_full, latents_div_2, prompt_embeds = (
                tensor.to(DEVICE, non_blocking=True) for tensor in batch
            )  # [B, C, H, W], [B, C, H / 2, W / 2], [B, 77, 768]
            noise = torch.randn_like(latents_full)  # [B, C, H, W]
            sigmas = sample_training_sigmas(latents_full.shape[0], DEVICE)  # [B]
            blend = sigmas[:, None, None, None]
            noisy_latents = (1.0 - blend) * latents_full + blend * noise

            with torch.autocast(DEVICE, dtype=TRAINING_DTYPE):
                velocity = compiled_model(noisy_latents, sigmas, latents_div_2, prompt_embeds)
            loss = functional.mse_loss(velocity.float(), noise - latents_full)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            warmup_scheduler.step()
            if training_start_time is not None:
                remaining_fraction = 1.0 - (time.perf_counter() - training_start_time) / (
                    TRAINING_TIME_SECONDS
                )
                decay_factor = min(1.0, max(0.0, remaining_fraction) / DECAY_FRACTION)
                warmup_factor = min(1.0, (step + 1) / WARMUP_STEPS)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = LEARNING_RATE * warmup_factor * decay_factor

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

            if (
                not smoke_mode
                and next_checkpoint_index < len(CHECKPOINT_TIMES_SECONDS)
                and elapsed_seconds >= CHECKPOINT_TIMES_SECONDS[next_checkpoint_index]
            ):
                minutes = CHECKPOINT_TIMES_SECONDS[next_checkpoint_index] // 60
                OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
                checkpoint_path = OUTPUT_DIRECTORY / f"checkpoint_{minutes:03d}min.safetensors"
                save_file(model.state_dict(), str(checkpoint_path))
                print(f"{RUN_NAME}: checkpoint {checkpoint_path.name} at step {step}", flush=True)
                next_checkpoint_index += 1

            if smoke_mode:
                is_finished = step >= SMOKE_STEPS
            else:
                is_finished = elapsed_seconds >= TRAINING_TIME_SECONDS
            if is_finished:
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
    def upscale(self, tensor: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            prompt_embeds: [B, 77, 768], dtype = float32, SDXL text encoder embeddings
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
        prompt_embeds = prompt_embeds.to(INFERENCE_DTYPE)
        sampling_sigmas = shift_sigmas(
            torch.linspace(1.0, 0.0, NUM_SAMPLING_STEPS + 1, device=tensor.device),
            SAMPLING_SIGMA_SHIFT,
        )
        for current_sigma, next_sigma in zip(sampling_sigmas[:-1], sampling_sigmas[1:]):
            velocity = self.model(
                latents.to(INFERENCE_DTYPE),
                current_sigma.expand(batch_size),
                latents_low_resolution,
                prompt_embeds,
            )
            latents = latents + (next_sigma - current_sigma) * velocity.float()
        return latents.to(tensor.dtype)

    @torch.no_grad()
    def single_forward(self, tensor: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            prompt_embeds: [B, 77, 768], dtype = float32
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
        return self.model(
            noisy_latents,
            sigmas,
            tensor.to(INFERENCE_DTYPE),
            prompt_embeds.to(INFERENCE_DTYPE),
        )

    def get_name(self) -> str:
        return RUN_NAME
