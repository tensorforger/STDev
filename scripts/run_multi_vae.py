"""
Single-GPU training of the small universal upscaler (stdev.multi_vae_upscalers.small_universal)
on the universal precompute (scripts/precompute_universal_dataset.py, HFF values from
scripts/precompute_universal_hff.py): 256 -> 512 px latents of many VAEs, images and videos mixed.

Every batch holds samples of one VAE. Batches of the different VAEs are interleaved randomly by
drawing from a pool of simultaneously open shards. The last shard of every VAE is held out for the
validation loss and the previews. After every epoch the EMA weights are evaluated, exported and
used to write side by side previews (low resolution bilinearly upsampled in pixel space |
upscaled | ground truth) of fixed held out images and videos.

Learning rate: warmup, constant, then a linear decay to zero over the last part of training.
Output goes to multi_vae_runs/NAME: tensorboard, checkpoint.pt (resumed automatically at an epoch
boundary), export/{ema,raw}.safetensors and previews/epoch_NNN.

Usage: python scripts/run_multi_vae.py --run-name NAME [--epochs 10] [--smoke]
"""

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy
import torch
import torch.nn.functional as functional
import vaezip
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.tensorboard import SummaryWriter

from stdev.multi_vae_upscalers.small_universal import (
    INFERENCE_DTYPE,
    SAMPLING_SIGMA_SHIFT,
    UniversalUpscalerUNet,
    frames_to_batch,
    sample_high_resolution_latents,
    shift_sigmas,
)
from stdev.utils.model_ema import ModelEMA
from stdev.vaes.flux_1_vae import Flux1VAE
from stdev.vaes.flux_2_vae import Flux2VAE
from stdev.vaes.qwen_image_2_1_vae import QwenImage21VAE
from stdev.vaes.sd_3_5_vae import SD35VAE
from stdev.vaes.sdxl_vae import SDXLVAE
from stdev.vaes.tiny_ltx_2_5_vae import TinyLTX25VAE
from stdev.vaes.tiny_minimax_h3_vae import TinyMiniMaxH3VAE
from stdev.vaes.tiny_wan_2_1_vae import TinyWan21VAE
from stdev.vaes.vae import VAE
from stdev.vaes.wan_2_1_vae import Wan21VAE

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIRECTORY = ROOT / "precomputes" / "universal_512_dev"
RUNS_DIRECTORY = ROOT / "multi_vae_runs"
MODELS_DIRECTORY = ROOT / "models"
DEVICE = torch.device("cuda")
SEED = 0

IMAGE_BATCH_SIZE = 16
VIDEO_BATCH_SIZE = 2
ACTIVE_SHARDS = 9
LEARNING_RATE = 4e-4
WARMUP_STEPS = 100
DECAY_FRACTION = 0.3
ADAM_BETAS = (0.9, 0.95)
GRADIENT_CLIP_NORM = 1.0
EMA_DECAY = 0.999
HIGH_FREQUENCY_DROPOUT_PROBABILITY = 0.2
LOGIT_NORMAL_SIGMA_SHIFT = 1.0
LOG_INTERVAL_STEPS = 50

PREVIEW_IMAGE_COUNT = 3
PREVIEW_VIDEO_COUNT = 2
PREVIEW_VIDEO_FPS = 8
SMOKE_STEPS_PER_EPOCH = 20


@dataclass
class Shard:
    """
    args:
        low: [N, C, T, H / 2, W / 2], high: [N, C, T, H, W], T = 1 for images
        high_frequency: [N], values: [0...1]
    """

    vae_name: str
    is_video: bool
    low: torch.Tensor
    high: torch.Tensor
    high_frequency: torch.Tensor


@dataclass
class Batch:
    vae_name: str
    low: torch.Tensor
    high: torch.Tensor
    high_frequency: torch.Tensor


class OpenShard:
    """
    Shard being consumed in random order, in batches of a fixed size (the remainder is dropped).
    """

    def __init__(self, shard: Shard, batch_size: int, generator: random.Random) -> None:
        self.shard = shard
        self.batch_size = batch_size
        order = list(range(shard.low.shape[0]))
        generator.shuffle(order)
        self.order = torch.tensor(order, device=shard.low.device)
        self.cursor = 0

    def remaining_batches(self) -> int:
        return (self.order.numel() - self.cursor) // self.batch_size

    def next_batch(self) -> Batch:
        indices = self.order[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return Batch(
            self.shard.vae_name,
            self.shard.low[indices],
            self.shard.high[indices],
            self.shard.high_frequency[indices],
        )


class UniversalLatentData:
    """
    Shard index of the universal precompute. Unit = (modality, vae name, shard index).
    """

    def __init__(self, root: Path, vae_names: set[str] | None = None) -> None:
        self.root = root
        self.units: list[tuple[str, str, int]] = []
        self.shard_sizes: dict[tuple[str, str, int], int] = {}
        self.high_frequency: dict[tuple[str, str], torch.Tensor] = {}
        self.heldout_units: list[tuple[str, str, int]] = []
        for modality in ("images", "videos"):
            manifest = json.loads((root / modality / "manifest.json").read_text())
            shard_size = manifest["shard_size"]
            num_samples = len(manifest["sample_ids"])
            num_shards = -(-num_samples // shard_size)
            values = load_file(str(root / modality / "hff.safetensors"))
            for vae_name in manifest["vaes"]:
                if vae_names is not None and vae_name not in vae_names:
                    continue
                self.high_frequency[(modality, vae_name)] = (
                    values["hff"] if modality == "images" else values[vae_name]
                )
                for shard_index in range(num_shards):
                    unit = (modality, vae_name, shard_index)
                    self.shard_sizes[unit] = min(shard_size, num_samples - shard_index * shard_size)
                    if shard_index == num_shards - 1:
                        self.heldout_units.append(unit)
                    else:
                        self.units.append(unit)
        self.shard_size = shard_size

    @staticmethod
    def batch_size_of(modality: str) -> int:
        return IMAGE_BATCH_SIZE if modality == "images" else VIDEO_BATCH_SIZE

    def steps_per_epoch(self) -> int:
        return sum(self.shard_sizes[unit] // self.batch_size_of(unit[0]) for unit in self.units)

    def load_shard(self, unit: tuple[str, str, int]) -> Shard:
        modality, vae_name, shard_index = unit
        latents = {}
        for resolution in ("low", "high"):
            path = (
                self.root / modality / vae_name / resolution / f"{shard_index:05d}.vz.safetensors"
            )
            latents[resolution] = vaezip.load(str(path))["latents"].decode(
                dtype=torch.float32, device=DEVICE
            )  # [N, C, (T), H, W]
        if modality == "images":
            latents = {key: value.unsqueeze(2) for key, value in latents.items()}
        start = shard_index * self.shard_size
        high_frequency = self.high_frequency[(modality, vae_name)][
            start : start + self.shard_sizes[unit]
        ].to(DEVICE)
        return Shard(
            vae_name, modality == "videos", latents["low"], latents["high"], high_frequency
        )

    def iterate_epoch(self, seed: int):
        """
        Yields Batch objects, interleaving batches of up to ACTIVE_SHARDS open shards.
        """
        generator = random.Random(seed)
        pending = list(self.units)
        generator.shuffle(pending)
        active: list[OpenShard] = []
        while pending or active:
            while pending and len(active) < ACTIVE_SHARDS:
                unit = pending.pop()
                active.append(
                    OpenShard(self.load_shard(unit), self.batch_size_of(unit[0]), generator)
                )
            weights = [open_shard.remaining_batches() for open_shard in active]
            position = generator.choices(range(len(active)), weights=weights)[0]
            yield active[position].next_batch()
            if active[position].remaining_batches() == 0:
                active.pop(position)


def sample_training_sigmas(count: int) -> torch.Tensor:
    """
    returns:
        sigmas: [count], values: [0...1], logit-normal distributed and shifted
    """
    return shift_sigmas(torch.sigmoid(torch.randn(count, device=DEVICE)), LOGIT_NORMAL_SIGMA_SHIFT)


def flow_matching_loss(
    model: torch.nn.Module,
    vae_name: str,
    low: torch.Tensor,
    high: torch.Tensor,
    high_frequency: torch.Tensor,
    high_frequency_mask: torch.Tensor,
    sigmas: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """
    args:
        low: [F, C, H / 2, W / 2], high: [F, C, H, W], noise: [F, C, H, W]
        high_frequency: [F], high_frequency_mask: [F], dtype: torch.bool, sigmas: [F]
    returns:
        loss: scalar, mean squared error of the velocity noise - high
    """
    blend = sigmas[:, None, None, None]
    noisy_latents = (1.0 - blend) * high + blend * noise
    with torch.autocast("cuda", dtype=INFERENCE_DTYPE):
        velocity = model(noisy_latents, sigmas, low, vae_name, high_frequency, high_frequency_mask)
    return functional.mse_loss(velocity.float(), noise - high)


def training_loss(model: torch.nn.Module, batch: Batch) -> torch.Tensor:
    num_frames = batch.high.shape[2]
    high = frames_to_batch(batch.high)  # [B*T, C, H, W]
    low = frames_to_batch(batch.low)  # [B*T, C, H / 2, W / 2]
    high_frequency = batch.high_frequency.repeat_interleave(num_frames)
    mask = torch.rand(high.shape[0], device=DEVICE) >= HIGH_FREQUENCY_DROPOUT_PROBABILITY
    return flow_matching_loss(
        model,
        batch.vae_name,
        low,
        high,
        high_frequency,
        mask,
        sample_training_sigmas(high.shape[0]),
        torch.randn_like(high),
    )


@torch.no_grad()
def heldout_loss(model: torch.nn.Module, shard: Shard, batch_size: int) -> float:
    """
    Mean loss on a held out shard with fixed noise and noise levels, HFF always given.
    """
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    total, count = 0.0, 0
    for start in range(0, shard.low.shape[0], batch_size):
        high = frames_to_batch(shard.high[start : start + batch_size])
        low = frames_to_batch(shard.low[start : start + batch_size])
        num_frames = shard.high.shape[2]
        high_frequency = shard.high_frequency[start : start + batch_size].repeat_interleave(
            num_frames
        )
        loss = flow_matching_loss(
            model,
            shard.vae_name,
            low,
            high,
            high_frequency,
            torch.ones_like(high_frequency, dtype=torch.bool),
            torch.rand(high.shape[0], device=DEVICE, generator=generator),
            torch.randn(high.shape, device=DEVICE, generator=generator),
        )
        total += loss.item() * high.shape[0]
        count += high.shape[0]
    return total / count


def build_vaes(names: set[str] | None = None) -> dict[str, VAE]:
    wan_path = str(MODELS_DIRECTORY / "Wan2.1-T2V-1.3B-Diffusers/vae")
    factories = {
        "sdxl": lambda: SDXLVAE(str(MODELS_DIRECTORY / "sdxl-vae")),
        "sd_3_5": lambda: SD35VAE(str(MODELS_DIRECTORY / "stable-diffusion-3.5-large/vae")),
        "flux_1": lambda: Flux1VAE(str(MODELS_DIRECTORY / "FLUX.1-schnell/vae")),
        "flux_2": lambda: Flux2VAE(str(MODELS_DIRECTORY / "FLUX.2-klein-4B/vae")),
        "qwen_image_2_1": lambda: QwenImage21VAE(
            str(MODELS_DIRECTORY / "texture-fix-vae-for-qwen-image-2.1")
        ),
        "wan_2_1": lambda: Wan21VAE(wan_path, as_image=True),
        "tiny_wan_2_1": lambda: TinyWan21VAE(str(MODELS_DIRECTORY / "taew2_1.safetensors")),
        "tiny_ltx_2_5": lambda: TinyLTX25VAE(str(MODELS_DIRECTORY / "taeltx2_3_wide.safetensors")),
        "tiny_minimax_h3": lambda: TinyMiniMaxH3VAE(str(MODELS_DIRECTORY / "taeh3.safetensors")),
    }
    return {
        name: factory() for name, factory in factories.items() if names is None or name in names
    }


def decode_to_video_layout(vae: VAE, latents: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents: [B, C, T, H, W]
    returns:
        pixels: [B, 3, T', H', W'], pixel_values: [0...1]
    """
    pixels = vae.decode(latents)
    if not vae.is_video:
        pixels = pixels.unsqueeze(2)
    return pixels[:, :3].clamp(0.0, 1.0)


def make_comparison(
    vae: VAE,
    low: torch.Tensor,
    generated: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """
    Decodes the latents in pairs and puts bilinearly upsampled low resolution, upscaled and ground
    truth pixels side by side.

    args:
        low: [1, C, T, H / 2, W / 2], generated: [1, C, T, H, W], high: [1, C, T, H, W]
    returns:
        comparison: [T', 3, H', 3 * W'], pixel_values: [0...1]
    """
    pixels_low = decode_to_video_layout(vae, low)
    pixels_generated = decode_to_video_layout(vae, generated)
    pixels_high = decode_to_video_layout(vae, high)
    pixels_low = functional.interpolate(
        frames_to_batch(pixels_low), size=pixels_high.shape[-2:], mode="bilinear"
    )  # [T', 3, H', W']
    panels = [
        pixels_low,
        frames_to_batch(pixels_generated),
        frames_to_batch(pixels_high),
    ]
    return torch.cat(panels, dim=-1).cpu()


def write_comparison(path: Path, comparison: torch.Tensor, is_video: bool) -> None:
    """
    args:
        comparison: [T, 3, H, W], pixel_values: [0...1]
    """
    frames = (comparison * 255.0).round().byte().permute(0, 2, 3, 1).numpy()  # [T, H, W, 3]
    if not is_video:
        Image.fromarray(frames[0]).save(path.with_suffix(".png"))
        return
    writer = cv2.VideoWriter(
        str(path.with_suffix(".mp4")),
        cv2.VideoWriter_fourcc(*"mp4v"),
        PREVIEW_VIDEO_FPS,
        (frames.shape[2], frames.shape[1]),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(numpy.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))
    writer.release()


@torch.no_grad()
def write_previews(
    directory: Path,
    model: torch.nn.Module,
    vaes: dict[str, VAE],
    heldout_shards: dict[str, Shard],
    preview_indices: dict[str, list[int]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    inference_model = copy.deepcopy(model).to(INFERENCE_DTYPE)
    for vae_name, shard in heldout_shards.items():
        for sample_number, index in enumerate(
            preview_indices["videos" if shard.is_video else "images"]
        ):
            low = shard.low[index : index + 1]
            generated = sample_high_resolution_latents(
                inference_model, low, vae_name, shard.high_frequency[index : index + 1]
            )
            comparison = make_comparison(
                vaes[vae_name], low, generated, shard.high[index : index + 1]
            )
            write_comparison(directory / f"{vae_name}_{sample_number}", comparison, shard.is_video)


def learning_rate_at(step: int, total_steps: int) -> float:
    decay_start = total_steps * (1.0 - DECAY_FRACTION)
    warmup_factor = min(1.0, (step + 1) / WARMUP_STEPS)
    decay_factor = min(1.0, (total_steps - step) / max(total_steps - decay_start, 1.0))
    return LEARNING_RATE * warmup_factor * decay_factor


def export_weights(directory: Path, model: torch.nn.Module, ema_model: torch.nn.Module) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in (("raw", model), ("ema", ema_model)):
        tensors = {
            key: value.detach().float().cpu().contiguous()
            for key, value in source.state_dict().items()
        }
        save_file(tensors, str(directory / f"{name}.safetensors"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only-vae", default=None, help="train on a single VAE only")
    arguments = parser.parse_args()
    selected_vaes = None if arguments.only_vae is None else {arguments.only_vae}

    run_directory = RUNS_DIRECTORY / arguments.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data = UniversalLatentData(DATASET_DIRECTORY, selected_vaes)
    steps_per_epoch = data.steps_per_epoch()
    if arguments.smoke:
        steps_per_epoch = min(steps_per_epoch, SMOKE_STEPS_PER_EPOCH)
        arguments.epochs = 1
    total_steps = steps_per_epoch * arguments.epochs

    model = UniversalUpscalerUNet().to(DEVICE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"parameters: {parameter_count / 1e6:.1f}M, steps per epoch: {steps_per_epoch}", flush=True
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=0.0
    )
    ema = ModelEMA(model, decay=EMA_DECAY, device=DEVICE)

    first_epoch = 0
    step = 0
    checkpoint_path = run_directory / "checkpoint.pt"
    if checkpoint_path.exists() and not arguments.smoke:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        ema.ema_model.load_state_dict(checkpoint["ema"])
        ema.updates = checkpoint["ema_updates"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_epoch, step = checkpoint["epoch"], checkpoint["step"]
        print(f"resumed after epoch {first_epoch}", flush=True)

    heldout_shards = {unit[1]: data.load_shard(unit) for unit in data.heldout_units}
    preview_generator = random.Random(SEED)
    sample_counts = {
        "videos" if shard.is_video else "images": shard.low.shape[0]
        for shard in heldout_shards.values()
    }
    preview_counts = {"images": PREVIEW_IMAGE_COUNT, "videos": PREVIEW_VIDEO_COUNT}
    preview_indices = {
        modality: preview_generator.sample(range(sample_counts[modality]), preview_counts[modality])
        for modality in sorted(sample_counts)
    }
    vaes = build_vaes(selected_vaes)
    writer = SummaryWriter(str(run_directory / "tensorboard"), flush_secs=10)

    for epoch in range(first_epoch, arguments.epochs):
        model.train()
        epoch_start = time.time()
        window_losses: list[float] = []
        for epoch_step, batch in zip(range(steps_per_epoch), data.iterate_epoch(SEED + epoch)):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(step, total_steps)
            loss = training_loss(model, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            ema.update(model)
            step += 1
            loss_value = loss.item()
            window_losses.append(loss_value)
            writer.add_scalar(f"train_by_vae/{batch.vae_name}", loss_value, step)
            if step % LOG_INTERVAL_STEPS == 0:
                mean_loss = sum(window_losses) / len(window_losses)
                window_losses.clear()
                writer.add_scalar("train/loss", mean_loss, step)
                writer.add_scalar("train/lr", learning_rate_at(step, total_steps), step)
                print(
                    f"epoch {epoch + 1} step {step} loss {mean_loss:.4f} "
                    f"{time.time() - epoch_start:.0f}s",
                    flush=True,
                )

        ema.eval()
        for name, shard in heldout_shards.items():
            value = heldout_loss(
                ema.ema_model, shard, VIDEO_BATCH_SIZE if shard.is_video else IMAGE_BATCH_SIZE
            )
            writer.add_scalar(f"heldout_by_vae/{name}", value, step)
            print(f"epoch {epoch + 1} heldout {name}: {value:.4f}", flush=True)
        write_previews(
            run_directory / "previews" / f"epoch_{epoch + 1:03d}",
            ema.ema_model,
            vaes,
            heldout_shards,
            preview_indices,
        )
        export_weights(run_directory / "export", model, ema.ema_model)
        if not arguments.smoke:
            temporary_path = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.ema_model.state_dict(),
                    "ema_updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "step": step,
                },
                temporary_path,
            )
            temporary_path.replace(checkpoint_path)
        writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
