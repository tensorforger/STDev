"""
Encodes the FLUX.2 VAE latent datasets of autoresearch_3 (512 -> 1024 image upscaling).

Prerequisites: sources downloaded by `scripts/download_flux2_sources.py` and Axis-V1 in `datasets/axis-v1_1k`
(`hf download refine-axis/axis-v1_1k --repo-type dataset --local-dir datasets/axis-v1_1k`), FLUX.2 VAE in
`models/FLUX.2-klein-4B/vae`.

Run (4 GPUs; the benchmark is built by rank 0 first, an interrupted run is resumed by the same command):
    torchrun --standalone --nproc_per_node=4 scripts/encode_flux2_datasets.py
Quick check into `dev/smoke_flux2_dataset`:
    python scripts/encode_flux2_datasets.py --smoke

Every image is center cropped and resized to 1024x1024 (`crop_maximal_rectangle`, antialiased) and the 512x512 input
is its uint8 2x downscale. The benchmark input uses `bilinear` (cv2 INTER_LINEAR, as all benchmarks); training blocks
additionally hold slightly corrupted inputs (other downscaling filters, JPEG artifacts).

Benchmark `benchmark_datasets/axis_v1_1k_flux2_2x`: ground_truth/<name>.png (1024), low_resolution/<name>.png (512),
latents.pt ([N, 32, 64, 64], latents of the low resolution images), latents_hr.pt ([N, 32, 128, 128]),
hff.pt ([N], Immerkaer hff of the ground truth), manifest.json.

Training datasets `train_datasets/flux2_relaion_20k`, `train_datasets/flux2_4klsdb_20k`: block_<index>.vz.safetensors
with normalized FLUX.2 latents (`lbrq_8bit`) and hff (`cast_fp16`):
    latents_1024            [1, 32, 128, 128]   target
    latents_512             [1, 32, 64, 64]     bilinear 2x downscale (benchmark input)
    latents_512_<variant>   [1, 32, 64, 64]     LR_VARIANTS other than bilinear
    hff_1024, hff_512       [1]
"""

import argparse
import io
import json
import os
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as distributed
import torch.nn.functional as functional
import vaezip
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stdev.hff.immerkaer import estimate_immerkaer
from stdev.upscale_benchmark.dataset import list_source_images
from stdev.utils.image_tools import crop_maximal_rectangle
from stdev.vaes.flux_2_vae import Flux2VAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VAE_PATH = REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B" / "vae"
AXIS_SOURCE_DIRECTORY = REPOSITORY_ROOT / "datasets" / "axis-v1_1k"
BENCHMARK_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "axis_v1_1k_flux2_2x"
TRAIN_SOURCES = {
    "flux2_relaion_20k": REPOSITORY_ROOT / "datasets" / "relaion_high_resolution_20k",
    "flux2_4klsdb_20k": REPOSITORY_ROOT / "datasets" / "4klsdb_1024",
}
TRAIN_DATASETS_DIRECTORY = REPOSITORY_ROOT / "train_datasets"
SMOKE_DIRECTORY = REPOSITORY_ROOT / "dev" / "smoke_flux2_dataset"

HIGH_RESOLUTION_SIZE = 1024
LOW_RESOLUTION_SIZE = 512
NUM_TRAIN_SAMPLES_PER_SOURCE = 20000
NUM_SMOKE_SAMPLES = 4
LR_VARIANTS = ("bilinear", "nearest", "bicubic", "lanczos", "antialias", "jpeg")
JPEG_QUALITY_RANGE = (20, 50)
LATENT_CODEC = "lbrq_8bit"
HFF_CODEC = "cast_fp16"
VAE_DTYPE = torch.bfloat16
SEED = 0
BATCH_SIZE = 4
NUM_LOADER_WORKERS = 8
BENCHMARK_ENCODE_BATCH_SIZE = 8
BARRIER_TIMEOUT = timedelta(hours=3)

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encodes the FLUX.2 latent datasets.")
    parser.add_argument("--smoke", action="store_true", help="A few samples into dev/smoke_flux2_dataset.")
    return parser.parse_args()


def downscale_variant(image: np.ndarray, variant: str, seed: int) -> np.ndarray:
    """
    2x downscale of a uint8 image with one of LR_VARIANTS.

    args:
        image: [1024, 1024, 3], channel_order: RGB, dtype: uint8
    returns:
        image: [512, 512, 3], channel_order: RGB, dtype: uint8
    """
    size = (LOW_RESOLUTION_SIZE, LOW_RESOLUTION_SIZE)
    if variant == "bilinear":
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if variant == "nearest":
        return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    if variant == "bicubic":
        return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    if variant == "lanczos":
        return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
    if variant == "antialias":
        return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BILINEAR))
    if variant == "jpeg":
        bilinear = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        quality = int(np.random.default_rng([SEED, seed]).integers(*JPEG_QUALITY_RANGE, endpoint=True))
        buffer = io.BytesIO()
        Image.fromarray(bilinear).save(buffer, format="JPEG", quality=quality)
        return np.asarray(Image.open(buffer).convert("RGB"))
    raise ValueError(f"Unknown variant {variant}")


def to_pixels(images: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    args:
        images: [B, H, W, 3], channel_order: RGB, dtype: uint8
    returns:
        pixels: [B, 3, H, W], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32, on DEVICE
    """
    images = torch.as_tensor(images).to(DEVICE)
    return images.permute(0, 3, 1, 2).float() / 255.0


def encode(vae: Flux2VAE, pixels: torch.Tensor) -> torch.Tensor:
    """
    args:
        pixels: [B, 3, H, W], pixel_values: [0...1]
    returns:
        latents: [B, 32, H / 8, W / 8], normalized FLUX.2 latents, dtype: torch.float32
    """
    return vae.encode(pixels)[:, :, 0]


class TrainSourceDataset(Dataset):
    """
    CPU side: reads a source image, crops it to 1024 and produces all low resolution variants.
    """

    def __init__(self, source_paths: list[Path], indices: list[int]) -> None:
        self.source_paths = source_paths
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        """
        returns:
            index, valid, image_1024: [1024, 1024, 3] uint8 RGB, images_512: [V, 512, 512, 3] uint8 RGB
        """
        index = self.indices[position]
        image = cv2.imread(str(self.source_paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            return {
                "index": index,
                "valid": False,
                "image_1024": torch.zeros(HIGH_RESOLUTION_SIZE, HIGH_RESOLUTION_SIZE, 3, dtype=torch.uint8),
                "images_512": torch.zeros(
                    len(LR_VARIANTS), LOW_RESOLUTION_SIZE, LOW_RESOLUTION_SIZE, 3, dtype=torch.uint8
                ),
            }
        image = crop_maximal_rectangle(image, HIGH_RESOLUTION_SIZE, HIGH_RESOLUTION_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images_512 = np.stack([downscale_variant(image, variant, index) for variant in LR_VARIANTS])
        return {
            "index": index,
            "valid": True,
            "image_1024": torch.from_numpy(image),
            "images_512": torch.from_numpy(images_512),
        }


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


def block_path(directory: Path, index: int) -> Path:
    return directory / f"block_{index:06d}.vz.safetensors"


@torch.no_grad()
def encode_train_dataset(vae: Flux2VAE, source_directory: Path, output_directory: Path, num_samples: int) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    source_paths = list_source_images(source_directory)[:num_samples]
    if len(source_paths) < num_samples:
        print(f"[rank {RANK}] {source_directory.name}: only {len(source_paths)} sources", flush=True)
    pending = [
        index
        for index in range(RANK, len(source_paths), WORLD_SIZE)
        if not block_path(output_directory, index).exists()
    ]
    if RANK == 0:
        (output_directory / "sources.json").write_text(
            json.dumps([path.name for path in source_paths], indent=0)
        )
    loader = DataLoader(
        TrainSourceDataset(source_paths, pending),
        batch_size=BATCH_SIZE,
        num_workers=NUM_LOADER_WORKERS,
        pin_memory=True,
    )
    invalid_count = 0
    for batch in tqdm(loader, desc=f"{output_directory.name} [rank {RANK}]", position=RANK):
        latents_1024 = encode(vae, to_pixels(batch["image_1024"]))  # [B, 32, 128, 128]
        hff_1024 = estimate_immerkaer(to_pixels(batch["image_1024"]))  # [B]
        batch_size, num_variants = batch["images_512"].shape[:2]
        pixels_512 = to_pixels(batch["images_512"].flatten(0, 1))  # [B * V, 3, 512, 512]
        latents_512 = encode(vae, pixels_512).unflatten(0, (batch_size, num_variants))
        hff_512 = estimate_immerkaer(pixels_512[:: num_variants])  # bilinear variant, [B]
        for sample in range(batch_size):
            index = int(batch["index"][sample])
            if not bool(batch["valid"][sample]):
                invalid_count += 1
                print(f"[rank {RANK}] unreadable source {source_paths[index]}", flush=True)
                continue
            tensors = {
                "latents_1024": latents_1024[sample : sample + 1],
                "hff_1024": hff_1024[sample : sample + 1],
                "hff_512": hff_512[sample : sample + 1],
            }
            for variant_index, variant in enumerate(LR_VARIANTS):
                key = "latents_512" if variant == "bilinear" else f"latents_512_{variant}"
                tensors[key] = latents_512[sample, variant_index : variant_index + 1]
            save_block(tensors, block_path(output_directory, index))
    if invalid_count:
        print(f"[rank {RANK}] {output_directory.name}: {invalid_count} unreadable sources skipped", flush=True)


def prepare_benchmark_image(source_path: Path, ground_truth_path: Path, low_resolution_path: Path) -> None:
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    ground_truth = crop_maximal_rectangle(image, HIGH_RESOLUTION_SIZE, HIGH_RESOLUTION_SIZE)
    low_resolution = cv2.resize(
        ground_truth, (LOW_RESOLUTION_SIZE, LOW_RESOLUTION_SIZE), interpolation=cv2.INTER_LINEAR
    )
    cv2.imwrite(str(ground_truth_path), ground_truth)
    cv2.imwrite(str(low_resolution_path), low_resolution)


def load_png_batch(directory: Path, names: list[str]) -> torch.Tensor:
    """
    returns:
        pixels: [B, 3, H, W], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32, on DEVICE
    """
    images = np.stack(
        [cv2.cvtColor(cv2.imread(str(directory / f"{name}.png")), cv2.COLOR_BGR2RGB) for name in names]
    )
    return to_pixels(images)


@torch.no_grad()
def build_benchmark(vae: Flux2VAE, output_directory: Path, max_samples: int | None) -> None:
    if (output_directory / "manifest.json").exists():
        return
    source_paths = list_source_images(AXIS_SOURCE_DIRECTORY)[:max_samples]
    names = [f"{index:04d}" for index in range(len(source_paths))]
    ground_truth_directory = output_directory / "ground_truth"
    low_resolution_directory = output_directory / "low_resolution"
    ground_truth_directory.mkdir(parents=True, exist_ok=True)
    low_resolution_directory.mkdir(parents=True, exist_ok=True)
    for source_path, name in tqdm(list(zip(source_paths, names)), desc="Benchmark images"):
        prepare_benchmark_image(
            source_path, ground_truth_directory / f"{name}.png", low_resolution_directory / f"{name}.png"
        )
    latents_low, latents_high, hff = [], [], []
    for start in tqdm(range(0, len(names), BENCHMARK_ENCODE_BATCH_SIZE), desc="Benchmark encode"):
        batch_names = names[start : start + BENCHMARK_ENCODE_BATCH_SIZE]
        pixels_high = load_png_batch(ground_truth_directory, batch_names)
        latents_low.append(encode(vae, load_png_batch(low_resolution_directory, batch_names)).cpu())
        latents_high.append(encode(vae, pixels_high).cpu())
        hff.append(estimate_immerkaer(pixels_high).cpu())
    torch.save(torch.cat(latents_low), output_directory / "latents.pt")
    torch.save(torch.cat(latents_high), output_directory / "latents_hr.pt")
    torch.save(torch.cat(hff), output_directory / "hff.pt")
    manifest = {
        "sample_names": names,
        "source_paths": [path.relative_to(AXIS_SOURCE_DIRECTORY).as_posix() for path in source_paths],
    }
    (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    arguments = parse_arguments()
    if WORLD_SIZE > 1:
        distributed.init_process_group("nccl", timeout=BARRIER_TIMEOUT, device_id=torch.device(DEVICE))
    torch.cuda.set_device(DEVICE)
    vae = Flux2VAE(str(VAE_PATH), device=DEVICE, dtype=VAE_DTYPE)

    benchmark_directory = SMOKE_DIRECTORY / "benchmark" if arguments.smoke else BENCHMARK_DIRECTORY
    if RANK == 0:
        build_benchmark(vae, benchmark_directory, NUM_SMOKE_SAMPLES if arguments.smoke else None)
    for dataset_name, source_directory in TRAIN_SOURCES.items():
        output_directory = (SMOKE_DIRECTORY if arguments.smoke else TRAIN_DATASETS_DIRECTORY) / dataset_name
        encode_train_dataset(
            vae,
            source_directory,
            output_directory,
            NUM_SMOKE_SAMPLES if arguments.smoke else NUM_TRAIN_SAMPLES_PER_SOURCE,
        )
    if WORLD_SIZE > 1:
        distributed.barrier()
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
