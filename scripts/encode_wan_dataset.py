"""
Encodes the Wan 2.1 VAE latent dataset for the Wan latent upscalers (see `autoresearch_2/instruction.md`).

Training dataset (default `train_datasets/wan_1024_i10k_v2k`):
    images/block_<index>.vz.safetensors   from datasets/relaion_high_resolution_subset
    videos/block_<index>.vz.safetensors   from datasets/strf-128h/videos
Benchmark datasets (`--benchmark`, taken from the first samples of the same sources for now):
    benchmark_datasets/wan_image/blocks, benchmark_datasets/wan_image/images (ground truth PNG)
    benchmark_datasets/wan_video/blocks, .../frames/<sample>/frame_<k>.png (33 ground truth frames, 512x512)

Image block keys: latents_{1024,512,256,128}, hff_{1024,512,256}. Image pyramid levels are repeated
bilinear 2x downscales of the 1024 crop.
Video block keys: latents_{33,17}_{1024,512}, hff_{33,17}_{1024,512}. A video sample is a window of 33
consecutive source frames; the 17 frame clip is every 2nd frame of that window (same time span, 2x lower
frame rate), so 17 -> 33 frames is a 2x temporal upscale.

Sharding: rank r of WORLD_SIZE handles every WORLD_SIZE-th item, existing blocks are skipped, so an
interrupted run is resumed by running the same command again.
    single GPU:  python scripts/encode_wan_dataset.py [--benchmark]
    8 GPUs:      torchrun --standalone --nproc_per_node=8 scripts/encode_wan_dataset.py [--benchmark]
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
import vaezip
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from vaezip.codecs.shortcuts import SHORTCUT_CONFIGS

from stdev.hff.immerkaer import estimate_immerkaer
from stdev.utils.image_tools import crop_maximal_rectangle
from stdev.vaes.tiny_wan_2_1_vae import TinyWan21VAE
from stdev.vaes.wan_2_1_vae import Wan21VAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SOURCE_DIRECTORY = REPOSITORY_ROOT / "datasets" / "relaion_high_resolution_subset"
VIDEO_SOURCE_DIRECTORY = REPOSITORY_ROOT / "datasets" / "strf-128h" / "videos"
TRAIN_DATASET_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "wan_1024_i10k_v2k"
BENCHMARK_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets"
BENCHMARK_IMAGE_NAME = "wan_image"
BENCHMARK_VIDEO_NAME = "wan_video"
BENCHMARK_VIDEO_SIZE = 512
VAE_PATH = REPOSITORY_ROOT / "models" / "Wan2.1-T2V-1.3B-Diffusers" / "vae"
TINY_VAE_PATH = REPOSITORY_ROOT / "models" / "taew2_1.safetensors"

NUM_TRAIN_IMAGES = 10000
MAX_READ_ATTEMPTS = 20
NUM_TRAIN_VIDEOS = 2000
NUM_BENCHMARK_IMAGES = 1000
NUM_BENCHMARK_VIDEOS = 100
NUM_SMOKE_ITEMS = 2

IMAGE_SIZE = 1024
IMAGE_LEVEL_SIZES = (1024, 512, 256, 128)
IMAGE_HFF_SIZES = (1024, 512, 256)
VIDEO_LONG_FRAMES = 33
VIDEO_SHORT_FRAMES = 17
VIDEO_SHORT_FRAME_STRIDE = 2
VIDEO_SIZES = (1024, 512)
# `lbrq_8bit` shortcut with a (T, H, W) block of (1, 8, 8): the shortcut itself only accepts 4D tensors.
LATENT_CODEC = {**SHORTCUT_CONFIGS["lbrq_8bit"], "block_size": [1, 8, 8]}
HFF_CODEC = "cast_fp16"
VAE_DTYPE = torch.bfloat16
SEED = 0
NUM_LOADER_WORKERS = 4

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encodes the Wan 2.1 VAE latent dataset.")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Write the benchmark datasets (benchmark_datasets/wan_image, wan_video).",
    )
    parser.add_argument("--smoke", action="store_true", help="Only a few items per kind.")
    parser.add_argument("--output-directory", type=Path, default=None)
    return parser.parse_args()


def block_name(index: int) -> str:
    return f"block_{index:06d}"


def downscale_half(pixels: torch.Tensor) -> torch.Tensor:
    """
    args:
        pixels: [B, 3, T, H, W], pixel_values: [0...1]
    returns:
        pixels: [B, 3, T, H / 2, W / 2], bilinear 2x downscale of every frame
    """
    batch_size, channels, frames, height, width = pixels.shape
    flat = pixels.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
    half = functional.interpolate(
        flat, size=(height // 2, width // 2), mode="bilinear", align_corners=False
    )  # [B*T, 3, H/2, W/2]
    return half.reshape(batch_size, frames, channels, height // 2, width // 2).permute(
        0, 2, 1, 3, 4
    )


def read_video_window(video_path: Path, window_start: int, num_frames: int) -> np.ndarray:
    """
    returns:
        frames: [T, H, W, 3], channel_order: RGB, dtype: uint8
    """
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    frame_index = 0
    while len(frames) < num_frames:
        is_read, frame = capture.read()
        if not is_read:
            raise RuntimeError(
                f"{video_path}: only {frame_index} frames, need {window_start + num_frames}"
            )
        if frame_index >= window_start:
            frames.append(
                cv2.cvtColor(
                    crop_maximal_rectangle(
                        frame, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE
                    ),
                    cv2.COLOR_BGR2RGB,
                )
            )
        frame_index += 1
    capture.release()
    return np.stack(frames)


def get_video_window_start(video_path: Path, item_index: int) -> int:
    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    generator = np.random.default_rng((SEED, item_index))
    return int(generator.integers(0, total_frames - VIDEO_LONG_FRAMES + 1))


class SourceDataset(Dataset):
    """
    CPU side of the pipeline: decodes the source files of (kind, item_index) items to uint8 frames.
    """

    def __init__(self, items: list[tuple[str, int]], source_paths: dict[str, list[Path]]) -> None:
        self.items = items
        self.source_paths = source_paths

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[str, int, torch.Tensor]:
        """
        returns:
            kind, item_index, frames: [T, H, W, 3], channel_order: RGB, dtype: torch.uint8
        """
        kind, item_index = self.items[index]
        paths = self.source_paths[kind]
        for attempt in range(MAX_READ_ATTEMPTS):
            source_path = paths[(item_index + attempt) % len(paths)]
            if kind == "image":
                image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"Unreadable image {source_path}, using the next one", flush=True)
                    continue
                image = crop_maximal_rectangle(
                    image, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE
                )
                frames = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None]
            else:
                try:
                    window_start = get_video_window_start(source_path, item_index)
                    frames = read_video_window(source_path, window_start, VIDEO_LONG_FRAMES)
                except (RuntimeError, ValueError) as error:
                    print(
                        f"Unreadable video {source_path} ({error}), using the next one", flush=True
                    )
                    continue
            return kind, item_index, torch.from_numpy(frames)
        raise RuntimeError(
            f"No readable {kind} source after {MAX_READ_ATTEMPTS} attempts from item {item_index}"
        )


def to_hff(pixels: torch.Tensor) -> torch.Tensor:
    """
    args:
        pixels: [1, 3, T, H, W], pixel_values: [0...1]
    returns:
        hff: [1], values: [0...1]
    """
    return estimate_immerkaer(pixels)


@torch.no_grad()
def encode_image_block(vae: Wan21VAE, frames: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    args:
        frames: [1, H, W, 3], dtype: torch.uint8
    returns:
        tensors: latents_<size> [1, 16, 1, size / 8, size / 8], hff_<size> [1]
    """
    pixels = frames.to(DEVICE).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0  # [1, 3, 1, H, W]
    tensors: dict[str, torch.Tensor] = {}
    for size in IMAGE_LEVEL_SIZES:
        if pixels.shape[-1] != size:
            pixels = downscale_half(pixels)
        tensors[f"latents_{size}"] = vae.encode(pixels)
        if size in IMAGE_HFF_SIZES:
            tensors[f"hff_{size}"] = to_hff(pixels)
    return tensors


@torch.no_grad()
def encode_video_block(vae: TinyWan21VAE, frames: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    args:
        frames: [33, H, W, 3], dtype: torch.uint8
    returns:
        tensors: latents_<33|17>_<size> [1, 16, T_latent, size / 8, size / 8], hff_<33|17>_<size> [1]
    """
    pixels_long = (
        frames.to(DEVICE).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    )  # [1, 3, 33, H, W]
    tensors: dict[str, torch.Tensor] = {}
    for size in VIDEO_SIZES:
        if pixels_long.shape[-1] != size:
            pixels_long = downscale_half(pixels_long)
        pixels_short = pixels_long[:, :, ::VIDEO_SHORT_FRAME_STRIDE]  # [1, 3, 17, H, W]
        for num_frames, pixels in (
            (VIDEO_LONG_FRAMES, pixels_long),
            (VIDEO_SHORT_FRAMES, pixels_short),
        ):
            tensors[f"latents_{num_frames}_{size}"] = vae.encode(pixels)
            tensors[f"hff_{num_frames}_{size}"] = to_hff(pixels)
    return tensors


def save_block(tensors: dict[str, torch.Tensor], path: Path) -> None:
    """
    Writes through a temporary file, so an interrupted run never leaves a partial block.
    """
    encoded_tensors = {
        name: vaezip.encode(
            tensor.float().cpu(), codec=HFF_CODEC if name.startswith("hff") else LATENT_CODEC
        )
        for name, tensor in tensors.items()
    }
    temporary_path = path.with_name(path.name + ".tmp")
    vaezip.save(encoded_tensors, temporary_path)
    temporary_path.rename(path)


def save_benchmark_ground_truth(
    kind: str, item_index: int, frames: torch.Tensor, directory: Path
) -> None:
    """
    args:
        frames: [T, H, W, 3], channel_order: RGB, dtype: torch.uint8, videos are saved at BENCHMARK_VIDEO_SIZE
    """
    if kind == "video" and frames.shape[1] != BENCHMARK_VIDEO_SIZE:
        resized = functional.interpolate(
            frames.permute(0, 3, 1, 2).float(),
            size=(BENCHMARK_VIDEO_SIZE, BENCHMARK_VIDEO_SIZE),
            mode="bilinear",
            align_corners=False,
        )  # [T, 3, S, S]
        frames = resized.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    frames_bgr = np.ascontiguousarray(frames.numpy()[..., ::-1])
    if kind == "image":
        cv2.imwrite(str(directory / "images" / f"{block_name(item_index)}.png"), frames_bgr[0])
        return
    video_directory = directory / "frames" / block_name(item_index)
    video_directory.mkdir(parents=True, exist_ok=True)
    for frame_index, frame in enumerate(frames_bgr):
        cv2.imwrite(str(video_directory / f"frame_{frame_index:02d}.png"), frame)


def build_layout(
    arguments: argparse.Namespace,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, int]]:
    """
    returns:
        block directories per kind, ground truth directories per kind (benchmark only), item counts
    """
    if arguments.benchmark:
        image_root = BENCHMARK_DIRECTORY / BENCHMARK_IMAGE_NAME
        video_root = BENCHMARK_DIRECTORY / BENCHMARK_VIDEO_NAME
        block_directories = {"image": image_root / "blocks", "video": video_root / "blocks"}
        ground_truth_directories = {"image": image_root, "video": video_root}
        counts = {"image": NUM_BENCHMARK_IMAGES, "video": NUM_BENCHMARK_VIDEOS}
    else:
        root = arguments.output_directory or TRAIN_DATASET_DIRECTORY
        block_directories = {"image": root / "images", "video": root / "videos"}
        ground_truth_directories = {}
        counts = {"image": NUM_TRAIN_IMAGES, "video": NUM_TRAIN_VIDEOS}
    if arguments.smoke:
        counts = {kind: NUM_SMOKE_ITEMS for kind in counts}
    return block_directories, ground_truth_directories, counts


def main() -> None:
    torch.cuda.set_device(DEVICE)
    arguments = parse_arguments()
    block_directories, ground_truth_directories, counts = build_layout(arguments)
    source_paths = {
        "image": sorted(IMAGE_SOURCE_DIRECTORY.glob("*")),
        "video": sorted(VIDEO_SOURCE_DIRECTORY.glob("*.mp4")),
    }
    for kind, directory in block_directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        if counts[kind] > len(source_paths[kind]):
            raise RuntimeError(
                f"Need {counts[kind]} {kind} sources, found {len(source_paths[kind])}"
            )
    for directory in ground_truth_directories.values():
        (directory / "images").mkdir(exist_ok=True)
        (directory / "frames").mkdir(exist_ok=True)

    all_items = [(kind, index) for kind in ("image", "video") for index in range(counts[kind])]
    pending_items = [
        item
        for item in all_items[RANK::WORLD_SIZE]
        if not (block_directories[item[0]] / f"{block_name(item[1])}.vz.safetensors").exists()
    ]
    if not pending_items:
        print(f"[rank {RANK}] Everything already exists", flush=True)
        return

    vaes = {
        "image": Wan21VAE(str(VAE_PATH), device=DEVICE, dtype=VAE_DTYPE),
        "video": TinyWan21VAE(str(TINY_VAE_PATH), device=DEVICE, dtype=VAE_DTYPE),
    }
    data_loader = DataLoader(
        SourceDataset(pending_items, source_paths),
        batch_size=None,
        shuffle=False,
        num_workers=NUM_LOADER_WORKERS,
        prefetch_factor=2,
    )
    for kind, item_index, frames in tqdm(
        data_loader, total=len(pending_items), desc=f"Encoding [rank {RANK}]", position=RANK
    ):
        encode_function = encode_image_block if kind == "image" else encode_video_block
        tensors = encode_function(vaes[kind], frames)
        if arguments.benchmark:
            save_benchmark_ground_truth(kind, item_index, frames, ground_truth_directories[kind])
        save_block(tensors, block_directories[kind] / f"{block_name(item_index)}.vz.safetensors")


if __name__ == "__main__":
    main()
