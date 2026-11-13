"""
Computes the HFF conditioning value (stdev.hff.immerkaer) of every sample of the universal
precompute from the source pixels at the high resolution, and stores it next to the latents:
images/hff.safetensors {"hff": [N]} and videos/hff.safetensors {vae_name: [N]} (the clip value
depends on the frames each VAE uses).

Usage: python scripts/precompute_universal_hff.py [--output precomputes/universal_512_dev]
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from precompute_universal_dataset import (
    IMAGE_DIRECTORY,
    ROOT,
    VIDEO_DIRECTORY,
    ImageDataset,
    VideoDataset,
)
from stdev.hff.immerkaer import estimate_immerkaer


def compute_image_values(output_directory: Path, num_workers: int) -> None:
    sample_ids = json.loads((output_directory / "manifest.json").read_text())["sample_ids"]
    paths = sorted(IMAGE_DIRECTORY.iterdir())
    if [path.stem for path in paths] != sample_ids:
        raise RuntimeError("image directory does not match the manifest")
    values = []
    for batch in DataLoader(ImageDataset(paths), batch_size=16, num_workers=num_workers):
        pixels = batch.cuda().float() / 255.0  # [B, 4, H, W]
        values.append(estimate_immerkaer(pixels[:, :3]).cpu())
    save_file({"hff": torch.cat(values)}, str(output_directory / "hff.safetensors"))


def compute_video_values(output_directory: Path, num_workers: int) -> None:
    manifest = json.loads((output_directory / "manifest.json").read_text())
    paths = sorted(VIDEO_DIRECTORY.iterdir())
    if [path.stem for path in paths] != manifest["sample_ids"]:
        raise RuntimeError("video directory does not match the manifest")
    frames_by_vae = {name: entry["pixel_frames"] for name, entry in manifest["vaes"].items()}
    values = {name: [] for name in frames_by_vae}
    dataset = VideoDataset(paths, max(frames_by_vae.values()))
    for batch in DataLoader(dataset, batch_size=4, num_workers=num_workers):
        pixels = batch.cuda().float() / 255.0  # [B, 3, T, H, W]
        for name, num_frames in frames_by_vae.items():
            values[name].append(estimate_immerkaer(pixels[:, :, :num_frames]).cpu())
    save_file(
        {name: torch.cat(chunks) for name, chunks in values.items()},
        str(output_directory / "hff.safetensors"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "precomputes" / "universal_512_dev")
    parser.add_argument("--num-workers", type=int, default=8)
    arguments = parser.parse_args()
    compute_image_values(arguments.output / "images", arguments.num_workers)
    compute_video_values(arguments.output / "videos", arguments.num_workers)


if __name__ == "__main__":
    main()
