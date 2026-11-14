"""
Downloads the source data and encodes the Wan 2.1 VAE latent datasets for the Wan latent upscalers
(see `autoresearch_2/instruction.md`). Run it once, after the environment is set and before autoresearch_2.

Prerequisites:
    - Python environment of this repository installed (`pyproject.toml` dependencies).
    - Full Wan 2.1 VAE in `models/Wan2.1-T2V-1.3B-Diffusers/vae` and the tiny one in `models/taew2_1.safetensors`.
    - Hugging Face login with accepted terms of the gated dataset `laion/relaion-high-resolution` (`hf auth login`).
    - `.env` in the repository root with S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY
      (read access to the bucket `vk-10k`, generated with `src/stdev/video_downloader/config-vk-10k.yaml`).
    - Immerkaer quantiles calibrated (`dev/calibrate_immerkaer.py`), they are used for the `hff_*` keys.

Run (8 GPUs; every rank downloads and encodes its share, an interrupted run is resumed by the same command):
    torchrun --standalone --nproc_per_node=8 scripts/download_and_encode_wan_dataset.py
Single GPU:
    python scripts/download_and_encode_wan_dataset.py
Quick check of the whole pipeline into `dev/smoke_wan_dataset` (nothing is written to the real directories):
    torchrun --standalone --nproc_per_node=8 scripts/download_and_encode_wan_dataset.py --smoke

Pipeline:
    1. 20000 images are downloaded from ReLAION-high-resolution into `datasets/relaion_high_resolution_20k`
       (each rank downloads its own slice of the shuffled urls).
    2. Images are encoded with the full Wan 2.1 VAE, 10000 video clips of the S3 bucket `vk-10k` are streamed
       from S3 and encoded with the tiny Wan 2.1 VAE. Clips are not stored locally.
    3. Sources are sorted by name; the last 1000 images and last 1000 videos form the benchmark, the rest is
       the training set (19000 images, 9000 videos).

Training dataset `train_datasets/wan_1024_i19k_v9k`:
    images/block_<index>.vz.safetensors, videos/block_<index>.vz.safetensors
Benchmark datasets: benchmark_datasets/wan_image/blocks, benchmark_datasets/wan_image/images (ground truth PNG),
    benchmark_datasets/wan_video/blocks, benchmark_datasets/wan_video/frames/<sample>/frame_<k>.png
    (33 ground truth frames, 512x512)

Image block keys: latents_{1024,512,256,128}, hff_{1024,512,256}. Image pyramid levels are repeated
bilinear 2x downscales of the 1024 crop.
Video block keys: latents_{33,17}_{1024,512}, hff_{33,17}_{1024,512}. A video sample is a window of 33
consecutive source frames; the 17 frame clip is every 2nd frame of that window (same time span, 2x lower
frame rate), so 17 -> 33 frames is a 2x temporal upscale.
"""

import argparse
import asyncio
import hashlib
import io
import os
import random
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

import aiohttp
import boto3
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as distributed
import torch.nn.functional as functional
import vaezip
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from vaezip.codecs.shortcuts import SHORTCUT_CONFIGS

from stdev.hff.immerkaer import estimate_immerkaer
from stdev.utils.image_tools import crop_maximal_rectangle
from stdev.vaes.tiny_wan_2_1_vae import TinyWan21VAE
from stdev.vaes.wan_2_1_vae import Wan21VAE
from stdev.video_downloader.stream_videos import REQUIRED_S3_VARIABLES, load_env_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DOWNLOAD_DIRECTORY = REPOSITORY_ROOT / "datasets" / "relaion_high_resolution_20k"
TRAIN_DATASET_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "wan_1024_i19k_v9k"
BENCHMARK_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets"
SMOKE_DIRECTORY = REPOSITORY_ROOT / "dev" / "smoke_wan_dataset"
BENCHMARK_IMAGE_NAME = "wan_image"
BENCHMARK_VIDEO_NAME = "wan_video"
BENCHMARK_VIDEO_SIZE = 512
VAE_PATH = REPOSITORY_ROOT / "models" / "Wan2.1-T2V-1.3B-Diffusers" / "vae"
TINY_VAE_PATH = REPOSITORY_ROOT / "models" / "taew2_1.safetensors"

RELAION_DATASET_NAME = "laion/relaion-high-resolution"
RELAION_SHARD_INDEX = 0
VIDEO_BUCKET = "vk-10k"
VIDEO_KEY_PREFIX = "videos/"

NUM_IMAGES = 20000
NUM_VIDEOS = 10000
NUM_BENCHMARK_IMAGES = 1000
NUM_BENCHMARK_VIDEOS = 1000
NUM_SMOKE_TRAIN_ITEMS = 2
NUM_SMOKE_BENCHMARK_ITEMS = 2

DOWNLOAD_CONCURRENCY_PER_RANK = 64
MAX_CONNECTIONS_PER_HOST = 4
CONNECT_TIMEOUT_SECONDS = 5.0
TOTAL_TIMEOUT_SECONDS = 20.0
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MIN_IMAGE_SIDE = 1024
SUPPORTED_IMAGE_FORMATS = {"jpeg": "jpg", "png": "png", "webp": "webp"}
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) research-dataset-downloader"
MAX_READ_ATTEMPTS = 20
BARRIER_TIMEOUT = timedelta(hours=6)

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
NUM_LOADER_WORKERS = 6

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downloads and encodes the Wan 2.1 VAE datasets.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only a few items per kind, written to dev/smoke_wan_dataset.",
    )
    return parser.parse_args()


def block_name(index: int) -> str:
    return f"block_{index:06d}"


def build_s3_client() -> Any:
    load_env_file(REPOSITORY_ROOT / ".env")
    missing = [name for name in REQUIRED_S3_VARIABLES if name not in os.environ]
    if missing:
        raise RuntimeError(f"Missing S3 variables: {', '.join(missing)}")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(
            retries={"max_attempts": 5, "mode": "standard"}, connect_timeout=10, read_timeout=60
        ),
    )


def list_video_keys(total_count: int) -> list[str]:
    client = build_s3_client()
    keys: list[str] = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=VIDEO_BUCKET, Prefix=VIDEO_KEY_PREFIX
    ):
        keys.extend(
            item["Key"] for item in page.get("Contents", []) if item["Key"].endswith(".mp4")
        )
    keys.sort()
    if len(keys) < total_count:
        raise RuntimeError(f"Need {total_count} videos in bucket {VIDEO_BUCKET}, found {len(keys)}")
    return keys[:total_count]


async def fetch_image(session: aiohttp.ClientSession, url: str) -> tuple[str, bytes, str]:
    """
    returns:
        outcome: "ok" or a failure reason, image bytes, file extension
    """
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return f"http_{response.status}", b"", ""
            if (response.content_length or 0) > MAX_IMAGE_BYTES:
                return "too_large", b"", ""
            buffer = bytearray()
            async for chunk in response.content.iter_chunked(64 * 1024):
                buffer.extend(chunk)
                if len(buffer) > MAX_IMAGE_BYTES:
                    return "too_large", b"", ""
    except Exception:
        return "network_error", b"", ""
    try:
        with Image.open(io.BytesIO(buffer)) as image:
            image.verify()
            width, height = image.size
            image_format = (image.format or "").lower()
    except Exception:
        return "not_a_valid_image", b"", ""
    if image_format not in SUPPORTED_IMAGE_FORMATS:
        return "unsupported_format", b"", ""
    if min(width, height) < MIN_IMAGE_SIDE:
        return "too_small", b"", ""
    return "ok", bytes(buffer), SUPPORTED_IMAGE_FORMATS[image_format]


async def download_rank_images(urls: list[str], quota: int, directory: Path) -> None:
    """
    Downloads until this rank holds `quota` images. Files are named `<url hash>-<rank>.<extension>`.
    """
    existing_hashes = {
        path.stem.rsplit("-", 1)[0]
        for path in directory.glob(f"*-{RANK}.*")
        if path.suffix != ".tmp"
    }
    pending_count = quota - len(existing_hashes)
    if pending_count <= 0:
        return
    url_iterator = iter(
        url for url in urls if hashlib.sha1(url.encode()).hexdigest()[:16] not in existing_hashes
    )
    success_count = 0
    progress_bar = tqdm(
        total=pending_count, desc=f"Downloading images [rank {RANK}]", position=RANK
    )

    async def worker(session: aiohttp.ClientSession) -> None:
        nonlocal success_count
        for url in url_iterator:
            if success_count >= pending_count:
                return
            outcome, content, extension = await fetch_image(session, url)
            if outcome != "ok" or success_count >= pending_count:
                continue
            file_path = (
                directory / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}-{RANK}.{extension}"
            )
            temporary_path = file_path.with_name(file_path.name + ".tmp")
            temporary_path.write_bytes(content)
            temporary_path.rename(file_path)
            success_count += 1
            progress_bar.update(1)

    connector = aiohttp.TCPConnector(
        limit=DOWNLOAD_CONCURRENCY_PER_RANK,
        limit_per_host=MAX_CONNECTIONS_PER_HOST,
        ttl_dns_cache=600,
        ssl=False,
    )
    timeout = aiohttp.ClientTimeout(total=TOTAL_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, headers={"User-Agent": USER_AGENT}
    ) as session:
        await asyncio.gather(*(worker(session) for _ in range(DOWNLOAD_CONCURRENCY_PER_RANK)))
    progress_bar.close()
    if success_count < pending_count:
        raise RuntimeError(
            f"[rank {RANK}] urls ran out, downloaded {success_count}/{pending_count}"
        )


def download_images(total_count: int, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    shard_name = sorted(
        name
        for name in HfApi().list_repo_files(RELAION_DATASET_NAME, repo_type="dataset")
        if name.endswith(".parquet")
    )[RELAION_SHARD_INDEX]
    shard_path = hf_hub_download(RELAION_DATASET_NAME, shard_name, repo_type="dataset")
    shard_table = pq.read_table(shard_path)
    url_column = next(name for name in shard_table.column_names if name.lower() == "url")
    urls: list[str] = shard_table.column(url_column).to_pylist()
    random.Random(SEED).shuffle(urls)
    rank_urls = urls[RANK::WORLD_SIZE]
    quota = -(-total_count // WORLD_SIZE)
    asyncio.run(download_rank_images(rank_urls, quota, directory))


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
            capture.release()
            raise RuntimeError(f"only {frame_index} frames, need {window_start + num_frames}")
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


def get_video_window_start(video_path: Path, global_index: int) -> int:
    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    generator = np.random.default_rng((SEED, global_index))
    return int(generator.integers(0, total_frames - VIDEO_LONG_FRAMES + 1))


class SourceDataset(Dataset):
    """
    CPU side of the pipeline: reads (downloads, for videos) and decodes the sources of
    (split, kind, index) items to uint8 frames.
    """

    def __init__(
        self,
        items: list[tuple[str, str, int]],
        image_paths: list[Path],
        video_keys: list[str],
        split_ranges: dict[tuple[str, str], tuple[int, int]],
    ) -> None:
        self.items = items
        self.sources = {"image": image_paths, "video": video_keys}
        self.split_ranges = split_ranges
        self.s3_client = None

    def __len__(self) -> int:
        return len(self.items)

    def read_image(self, source_path: Path) -> np.ndarray | None:
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        image = crop_maximal_rectangle(image, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None]

    def read_video(self, key: str, global_index: int) -> np.ndarray:
        if self.s3_client is None:
            self.s3_client = build_s3_client()
        body = self.s3_client.get_object(Bucket=VIDEO_BUCKET, Key=key)["Body"].read()
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video_file:
            video_file.write(body)
            video_file.flush()
            video_path = Path(video_file.name)
            window_start = get_video_window_start(video_path, global_index)
            return read_video_window(video_path, window_start, VIDEO_LONG_FRAMES)

    def __getitem__(self, index: int) -> tuple[str, str, int, torch.Tensor]:
        """
        returns:
            split, kind, item_index, frames: [T, H, W, 3], channel_order: RGB, dtype: torch.uint8
        """
        split, kind, item_index = self.items[index]
        range_start, range_end = self.split_ranges[(split, kind)]
        for attempt in range(MAX_READ_ATTEMPTS):
            global_index = range_start + (item_index + attempt) % (range_end - range_start)
            source = self.sources[kind][global_index]
            try:
                if kind == "image":
                    frames = self.read_image(source)
                else:
                    frames = self.read_video(source, global_index)
            except (RuntimeError, ValueError, ClientError, BotoCoreError) as error:
                print(f"Unreadable {kind} {source} ({error}), using the next one", flush=True)
                continue
            if frames is None:
                print(f"Unreadable {kind} {source}, using the next one", flush=True)
                continue
            return split, kind, item_index, torch.from_numpy(frames)
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


def main() -> None:
    torch.cuda.set_device(DEVICE)
    arguments = parse_arguments()
    if WORLD_SIZE > 1:
        distributed.init_process_group(
            "nccl", timeout=BARRIER_TIMEOUT, device_id=torch.device(DEVICE)
        )

    if arguments.smoke:
        train_root = SMOKE_DIRECTORY / "train"
        benchmark_root = SMOKE_DIRECTORY / "benchmark"
        image_download_directory = SMOKE_DIRECTORY / "relaion"
        benchmark_counts = {"image": NUM_SMOKE_BENCHMARK_ITEMS, "video": NUM_SMOKE_BENCHMARK_ITEMS}
        train_counts = {"image": NUM_SMOKE_TRAIN_ITEMS, "video": NUM_SMOKE_TRAIN_ITEMS}
    else:
        train_root = TRAIN_DATASET_DIRECTORY
        benchmark_root = BENCHMARK_DIRECTORY
        image_download_directory = IMAGE_DOWNLOAD_DIRECTORY
        benchmark_counts = {"image": NUM_BENCHMARK_IMAGES, "video": NUM_BENCHMARK_VIDEOS}
        train_counts = {
            "image": NUM_IMAGES - NUM_BENCHMARK_IMAGES,
            "video": NUM_VIDEOS - NUM_BENCHMARK_VIDEOS,
        }
    total_counts = {kind: train_counts[kind] + benchmark_counts[kind] for kind in train_counts}

    image_root = benchmark_root / BENCHMARK_IMAGE_NAME
    video_root = benchmark_root / BENCHMARK_VIDEO_NAME
    block_directories = {
        ("train", "image"): train_root / "images",
        ("train", "video"): train_root / "videos",
        ("benchmark", "image"): image_root / "blocks",
        ("benchmark", "video"): video_root / "blocks",
    }
    ground_truth_directories = {"image": image_root, "video": video_root}
    split_ranges = {("train", kind): (0, train_counts[kind]) for kind in train_counts} | {
        ("benchmark", kind): (train_counts[kind], total_counts[kind]) for kind in train_counts
    }
    for directory in block_directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    for directory in ground_truth_directories.values():
        (directory / "images").mkdir(exist_ok=True)
        (directory / "frames").mkdir(exist_ok=True)

    download_images(total_counts["image"], image_download_directory)
    video_keys = list_video_keys(total_counts["video"])
    if WORLD_SIZE > 1:
        distributed.barrier()
    image_paths = sorted(
        path for path in image_download_directory.glob("*") if path.suffix != ".tmp"
    )
    if len(image_paths) < total_counts["image"]:
        raise RuntimeError(f"Need {total_counts['image']} images, found {len(image_paths)}")
    image_paths = image_paths[: total_counts["image"]]

    all_items = [
        (split, kind, index)
        for split in ("train", "benchmark")
        for kind in ("image", "video")
        for index in range(split_ranges[(split, kind)][1] - split_ranges[(split, kind)][0])
    ]
    pending_items = [
        item
        for item in all_items[RANK::WORLD_SIZE]
        if not (
            block_directories[(item[0], item[1])] / f"{block_name(item[2])}.vz.safetensors"
        ).exists()
    ]
    if not pending_items:
        print(f"[rank {RANK}] Everything already exists", flush=True)
        return

    vaes = {
        "image": Wan21VAE(str(VAE_PATH), device=DEVICE, dtype=VAE_DTYPE),
        "video": TinyWan21VAE(str(TINY_VAE_PATH), device=DEVICE, dtype=VAE_DTYPE),
    }
    data_loader = DataLoader(
        SourceDataset(pending_items, image_paths, video_keys, split_ranges),
        batch_size=None,
        shuffle=False,
        num_workers=NUM_LOADER_WORKERS,
        prefetch_factor=4,
    )
    for split, kind, item_index, frames in tqdm(
        data_loader, total=len(pending_items), desc=f"Encoding [rank {RANK}]", position=RANK
    ):
        encode_function = encode_image_block if kind == "image" else encode_video_block
        tensors = encode_function(vaes[kind], frames)
        if split == "benchmark":
            save_benchmark_ground_truth(kind, item_index, frames, ground_truth_directories[kind])
        save_block(
            tensors, block_directories[(split, kind)] / f"{block_name(item_index)}.vz.safetensors"
        )


if __name__ == "__main__":
    main()
