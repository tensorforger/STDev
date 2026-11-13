"""
Patch FID per VAE of a trained universal upscaler (scripts/run_multi_vae.py). The real set is the
VAE reconstruction of the ground truth high resolution latents, the fake set is the VAE decoding
of the upscaled latents, so VAE reconstruction quality cancels out. Videos are evaluated on evenly
sampled pixel frames. A bilinear baseline (low resolution latents decoded, upsampled in pixel
space) is measured against the same real set.

Default shard is the held out one (last shard of each VAE, only 16 images and 17 videos); use
--shard-index N for a full shard, which was seen in training unless it is the held out one.

Usage: python scripts/eval_multi_vae.py --run-name NAME [--shard-index N] [--max-samples 256]
"""

import argparse
import json

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from torchmetrics.image.fid import FrechetInceptionDistance

from run_multi_vae import (
    DATASET_DIRECTORY,
    DEVICE,
    RUNS_DIRECTORY,
    UniversalLatentData,
    build_vaes,
    decode_to_video_layout,
)
from stdev.multi_vae_upscalers.small_universal import (
    INFERENCE_DTYPE,
    NUM_SAMPLING_STEPS,
    UniversalUpscalerUNet,
    frames_to_batch,
    sample_high_resolution_latents,
)
from stdev.utils.fid import split_into_patches
from stdev.vaes.vae import VAE

BATCH_SIZE = 8
FRAMES_PER_VIDEO = 8


def decode_each(vae: VAE, latents: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents: [B, C, T, H, W]
    returns:
        pixels: [B, 3, T', H', W'], pixel_values: [0...1]
    """
    return torch.cat(
        [
            decode_to_video_layout(vae, latents[index : index + 1])
            for index in range(latents.shape[0])
        ]
    )


def to_patches(pixels: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    args:
        pixels: [B, 3, T, H, W], pixel_values: [0...1]
    returns:
        patches: [N, 3, patch_size, patch_size], dtype: torch.uint8
    """
    num_frames = pixels.shape[2]
    frame_indices = (
        torch.linspace(0, num_frames - 1, min(FRAMES_PER_VIDEO, num_frames)).round().long()
    )
    frames = frames_to_batch(pixels[:, :, frame_indices])  # [B * frames, 3, H, W]
    return split_into_patches(frames.clamp(0, 1).mul(255).round().to(torch.uint8), patch_size)


@torch.no_grad()
def evaluate_vae(
    model: torch.nn.Module,
    vae: VAE,
    shard,
    max_samples: int,
    patch_size: int,
    num_steps: int,
    guidance_scale: float,
) -> dict[str, float | int]:
    metrics = {
        name: FrechetInceptionDistance(feature=2048).to(DEVICE) for name in ("model", "bilinear")
    }
    sample_count = min(max_samples, shard.low.shape[0])
    for start in range(0, sample_count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, sample_count)
        low, high = shard.low[start:end], shard.high[start:end]
        generated = sample_high_resolution_latents(
            model,
            low,
            shard.vae_name,
            shard.high_frequency[start:end],
            guidance_scale=guidance_scale,
            num_steps=num_steps,
        )
        pixels_high = decode_each(vae, high)
        pixels_low = decode_each(vae, low)
        pixels_bilinear = functional.interpolate(
            frames_to_batch(pixels_low), size=pixels_high.shape[-2:], mode="bilinear"
        )  # [B * T, 3, H, W]
        pixels_bilinear = pixels_bilinear.unflatten(0, (end - start, -1)).permute(0, 2, 1, 3, 4)
        fake_pixels = {
            "model": decode_each(vae, generated),
            "bilinear": pixels_bilinear,
        }
        real_patches = to_patches(pixels_high, patch_size)
        for name, metric in metrics.items():
            metric.update(real_patches, real=True)
            metric.update(to_patches(fake_pixels[name], patch_size), real=False)
    return {
        "samples": sample_count,
        "patches": int(metrics["model"].real_features_num_samples),
        "patch_fid_model": metrics["model"].compute().item(),
        "patch_fid_bilinear": metrics["bilinear"].compute().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--weights", default="ema", choices=["ema", "raw"])
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=NUM_SAMPLING_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--only-vae", default=None)
    arguments = parser.parse_args()
    selected_vaes = None if arguments.only_vae is None else {arguments.only_vae}

    run_directory = RUNS_DIRECTORY / arguments.run_name
    model = UniversalUpscalerUNet()
    model.load_state_dict(
        load_file(str(run_directory / "export" / f"{arguments.weights}.safetensors"))
    )
    model = model.to(DEVICE, INFERENCE_DTYPE).eval().requires_grad_(False)

    data = UniversalLatentData(DATASET_DIRECTORY, selected_vaes)
    vaes = build_vaes(selected_vaes)
    results = {}
    for modality, vae_name, heldout_index in data.heldout_units:
        shard_index = heldout_index if arguments.shard_index is None else arguments.shard_index
        shard = data.load_shard((modality, vae_name, shard_index))
        results[vae_name] = evaluate_vae(
            model,
            vaes[vae_name],
            shard,
            arguments.max_samples,
            arguments.patch_size,
            arguments.num_steps,
            arguments.guidance_scale,
        )
        print(vae_name, results[vae_name], flush=True)
        del shard
        torch.cuda.empty_cache()

    shard_label = "heldout" if arguments.shard_index is None else f"shard_{arguments.shard_index}"
    output_path = run_directory / f"patch_fid_{arguments.weights}_{shard_label}.json"
    output_path.write_text(json.dumps(results, indent=1))
    print(f"{'vae':18} {'samples':>8} {'patches':>8} {'model':>8} {'bilinear':>9}")
    for vae_name, entry in results.items():
        print(
            f"{vae_name:18} {entry['samples']:8d} {entry['patches']:8d} "
            f"{entry['patch_fid_model']:8.2f} {entry['patch_fid_bilinear']:9.2f}"
        )


if __name__ == "__main__":
    main()
