"""
Encodes the ReLAION image subset and the STRF video clips with many VAEs into vaezip shards
(high resolution 512 and low resolution 256 per sample). Layout is described in the README written
next to the shards. Finished shards are skipped, so an interrupted run can be restarted.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy
import torch
import torch.nn.functional as functional
import vaezip
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from vaezip.codecs import AutoCodec

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
IMAGE_DIRECTORY = ROOT / "datasets" / "relaion_high_resolution_subset"
VIDEO_DIRECTORY = ROOT / "datasets" / "strf-128h" / "videos"
WAN_PATH = ROOT / "models/Wan2.1-T2V-1.3B-Diffusers/vae"

HIGH_SIZE = 512
LOW_SIZE = 256
TARGET_NUM_FRAMES = 30
SHARD_SUFFIX = ".vz.safetensors"
CODEC_SHORTCUT = "lbrq_8bit"
LATENT_BLOCK_SIZE_IMAGE = (8, 8)
LATENT_BLOCK_SIZE_VIDEO = (1, 8, 8)


class ImageDataset(Dataset):
    """
    Center square crop resized to HIGH_SIZE, alpha kept (opaque where the file has none).
    """

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        returns:
            image: [4, H, W], channel_order: RGBA, pixel_values: [0...255], dtype: torch.uint8
        """
        image = ImageOps.exif_transpose(Image.open(self.paths[index])).convert("RGBA")
        side = min(image.size)
        left, top = (image.width - side) // 2, (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((HIGH_SIZE, HIGH_SIZE), Image.LANCZOS)
        return torch.from_numpy(numpy.asarray(image).copy()).permute(2, 0, 1)


class VideoDataset(Dataset):
    """
    First num_frames frames of each clip resized to HIGH_SIZE.
    """

    def __init__(self, paths: list[Path], num_frames: int) -> None:
        self.paths = paths
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        returns:
            video: [3, T, H, W], channel_order: RGB, pixel_values: [0...255], dtype: torch.uint8
        """
        cv2.setNumThreads(1)
        capture = cv2.VideoCapture(str(self.paths[index]))
        frames = []
        while len(frames) < self.num_frames:
            success, frame = capture.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(
                torch.from_numpy(
                    cv2.resize(frame, (HIGH_SIZE, HIGH_SIZE), interpolation=cv2.INTER_AREA)
                )
            )
        capture.release()
        if len(frames) < self.num_frames:
            raise RuntimeError(f"{self.paths[index]}: only {len(frames)} frames")
        return torch.stack(frames).permute(3, 0, 1, 2)


def closest_valid_num_frames(vae: VAE, available_frames: int) -> int:
    candidates = [
        count for count in range(1, available_frames + 1) if vae.is_valid_num_frames(count)
    ]
    return min(candidates, key=lambda count: (abs(count - TARGET_NUM_FRAMES), count))


def downsample_by_two(pixels: torch.Tensor) -> torch.Tensor:
    """
    args:
        pixels: [B, C, H, W] or [B, C, T, H, W]
    returns:
        pixels: same layout with H and W halved (2x2 box average)
    """
    if pixels.ndim == 4:
        return functional.avg_pool2d(pixels, kernel_size=2)
    return functional.avg_pool3d(pixels, kernel_size=(1, 2, 2))


def write_shard(path: Path, latents: torch.Tensor, block_size: tuple[int, ...]) -> None:
    """
    args:
        latents: [N, C, H, W] or [N, C, T, H, W]
    """
    codec = AutoCodec.from_config(CODEC_SHORTCUT, block_size=block_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    vaezip.save({"latents": vaezip.encode(latents.cuda(), codec=codec)}, path=temporary_path)
    temporary_path.replace(path)


def encode_modality(
    output_directory: Path,
    sample_paths: list[Path],
    dataset_factory,
    vaes: list[VAE],
    num_frames_by_vae: dict[str, int] | None,
    batch_size: int,
    shard_size: int,
    num_workers: int,
) -> None:
    is_video = num_frames_by_vae is not None
    block_size = LATENT_BLOCK_SIZE_VIDEO if is_video else LATENT_BLOCK_SIZE_IMAGE
    num_shards = -(-len(sample_paths) // shard_size)
    start_time = time.time()
    for shard_index in range(num_shards):
        first = shard_index * shard_size
        indices = list(range(first, min(first + shard_size, len(sample_paths))))
        shard_name = f"{shard_index:05d}{SHARD_SUFFIX}"
        pending_vaes = [
            vae
            for vae in vaes
            if not all(
                (output_directory / vae.get_vae_name() / resolution / shard_name).exists()
                for resolution in ("high", "low")
            )
        ]
        if not pending_vaes:
            continue

        loader = DataLoader(
            dataset_factory(indices), batch_size=batch_size, num_workers=num_workers
        )
        latents = {
            (vae.get_vae_name(), resolution): []
            for vae in pending_vaes
            for resolution in ("high", "low")
        }
        for batch in loader:
            high = batch.cuda().float() / 255.0  # [B, C, (T), H, W]
            low = downsample_by_two(high)
            for vae in pending_vaes:
                name = vae.get_vae_name()
                for resolution, pixels in (("high", high), ("low", low)):
                    pixels = pixels[:, : vae.pixel_channels]
                    if is_video:
                        pixels = pixels[:, :, : num_frames_by_vae[name]]
                    latents[(name, resolution)].append(vae.encode(pixels).cpu())

        for (name, resolution), chunks in latents.items():
            shard_latents = torch.cat(chunks)  # [N, C, T, H, W]
            if not is_video:
                shard_latents = shard_latents.squeeze(2)
            write_shard(
                output_directory / name / resolution / shard_name, shard_latents, block_size
            )
        print(
            f"shard {shard_index + 1}/{num_shards} done, {time.time() - start_time:.0f}s elapsed",
            flush=True,
        )


def build_manifest(
    sample_paths: list[Path],
    vaes: list[VAE],
    num_frames_by_vae: dict[str, int] | None,
    shard_size: int,
) -> dict:
    vae_entries = {}
    for vae in vaes:
        name = vae.get_vae_name()
        entry = {
            "latent_channels": vae.get_latent_channels(),
            "spatial_compression": vae.get_spatial_compression(),
        }
        if num_frames_by_vae is not None:
            entry["pixel_frames"] = num_frames_by_vae[name]
            entry["latent_frames"] = vae.num_latent_frames(num_frames_by_vae[name])
        vae_entries[name] = entry
    return {
        "shard_size": shard_size,
        "high_size": HIGH_SIZE,
        "low_size": LOW_SIZE,
        "codec": CODEC_SHORTCUT,
        "vaes": vae_entries,
        "sample_ids": [path.stem for path in sample_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "precomputes" / "universal_512_dev")
    parser.add_argument("--modality", choices=["images", "videos", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N samples")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    arguments = parser.parse_args()

    if arguments.modality in ("images", "all"):
        image_paths = sorted(IMAGE_DIRECTORY.iterdir())[: arguments.limit]
        image_vaes: list[VAE] = [
            SDXLVAE(str(ROOT / "models/sdxl-vae")),
            SD35VAE(str(ROOT / "models/stable-diffusion-3.5-large/vae")),
            Flux1VAE(str(ROOT / "models/FLUX.1-schnell/vae")),
            Flux2VAE(str(ROOT / "models/FLUX.2-klein-4B/vae")),
            QwenImage21VAE(str(ROOT / "models/texture-fix-vae-for-qwen-image-2.1")),
            Wan21VAE(str(WAN_PATH), as_image=True),
        ]
        output_directory = arguments.output / "images"
        output_directory.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(image_paths, image_vaes, None, arguments.shard_size)
        (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=1))
        encode_modality(
            output_directory,
            image_paths,
            lambda indices: ImageDataset([image_paths[index] for index in indices]),
            image_vaes,
            None,
            batch_size=8,
            shard_size=arguments.shard_size,
            num_workers=arguments.num_workers,
        )
        del image_vaes
        torch.cuda.empty_cache()

    if arguments.modality in ("videos", "all"):
        video_paths = sorted(VIDEO_DIRECTORY.iterdir())[: arguments.limit]
        video_vaes: list[VAE] = [
            TinyWan21VAE(str(ROOT / "models/taew2_1.safetensors")),
            TinyLTX25VAE(str(ROOT / "models/taeltx2_3_wide.safetensors")),
            TinyMiniMaxH3VAE(str(ROOT / "models/taeh3.safetensors")),
        ]
        capture = cv2.VideoCapture(str(video_paths[0]))
        available_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        num_frames_by_vae = {
            vae.get_vae_name(): closest_valid_num_frames(vae, available_frames)
            for vae in video_vaes
        }
        print("frames per vae:", num_frames_by_vae, flush=True)
        read_frames = max(num_frames_by_vae.values())
        output_directory = arguments.output / "videos"
        output_directory.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(video_paths, video_vaes, num_frames_by_vae, arguments.shard_size)
        (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=1))
        encode_modality(
            output_directory,
            video_paths,
            lambda indices: VideoDataset([video_paths[index] for index in indices], read_frames),
            video_vaes,
            num_frames_by_vae,
            batch_size=4,
            shard_size=arguments.shard_size,
            num_workers=min(arguments.num_workers, 6),
        )


if __name__ == "__main__":
    main()
