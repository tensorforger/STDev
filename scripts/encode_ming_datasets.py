"""
Encodes the Ming-Image VAE latent datasets of autoresearch_5 (512 -> 1024 image upscaling). The Ming-Image 0.1 / 0.2
Design VAE is byte-identical to the Qwen-Image-Layered RGBA VAE (`src/stdev/vaes/ming_image_vae.py`).

Prerequisites: 4KLSDB sources downloaded by `scripts/download_flux2_sources.py 4klsdb --num-files 65` and Axis-V1 in
`datasets/axis-v1_1k` (`hf download refine-axis/axis-v1_1k --repo-type dataset --local-dir datasets/axis-v1_1k`),
the VAE in `models/Ming-Image-0.2-Design/vae` (`hf download inclusionAI/Ming-Image-0.2-Design vae/config.json
vae/diffusion_pytorch_model.safetensors`). Existing ground_truth / low_resolution PNGs of the benchmark are reused.

Run (4 GPUs; the benchmark is built by rank 0 first, an interrupted run is resumed by the same command):
    torchrun --standalone --nproc_per_node=4 scripts/encode_ming_datasets.py
Quick check into `dev/smoke_ming_dataset`:
    python scripts/encode_ming_datasets.py --smoke

Every image is center cropped and resized to 1024x1024 (`crop_maximal_rectangle`, antialiased); the 512x512 input is
its uint8 bilinear 2x downscale (cv2 INTER_LINEAR, as all benchmarks). Images are encoded as opaque RGBA (alpha 1),
bf16 VAE, posterior mode, per channel normalized. Latents are stored as fp16 (`cast_fp16`).

Benchmark `benchmark_datasets/axis_v1_1k_ming_2x`: ground_truth/<name>.png (1024), low_resolution/<name>.png (512),
latents.pt ([N, 16, 64, 64], latents of the low resolution images), latents_hr.pt ([N, 16, 128, 128]),
hff.pt ([N], Immerkaer hff of the ground truth), manifest.json.

Training dataset `train_datasets/ming_4klsdb_40k`: block_<index>.vz.safetensors with normalized latents:
    latents_1024            [1, 16, 128, 128]   target
    latents_512             [1, 16, 64, 64]     bilinear 2x downscale (benchmark input)
    hff_1024, hff_512       [1]                 Immerkaer hff (unused in research)
"""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as distributed
import vaezip
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stdev.hff.immerkaer import estimate_immerkaer
from stdev.upscale_benchmark.dataset import list_source_images
from stdev.utils.image_tools import crop_maximal_rectangle
from stdev.vaes.ming_image_vae import MingImageVAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VAE_PATH = REPOSITORY_ROOT / "models" / "Ming-Image-0.2-Design" / "vae"
AXIS_SOURCE_DIRECTORY = REPOSITORY_ROOT / "datasets" / "axis-v1_1k"
BENCHMARK_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "axis_v1_1k_ming_2x"
TRAIN_SOURCES = {
    "ming_4klsdb_40k": REPOSITORY_ROOT / "datasets" / "4klsdb_1024",
}
TRAIN_DATASETS_DIRECTORY = REPOSITORY_ROOT / "train_datasets"
SMOKE_DIRECTORY = REPOSITORY_ROOT / "dev" / "smoke_ming_dataset"

HIGH_RESOLUTION_SIZE = 1024
LOW_RESOLUTION_SIZE = 512
NUM_TRAIN_SAMPLES_PER_SOURCE = 40000
NUM_SMOKE_SAMPLES = 4
LATENT_CODEC = "cast_fp16"
HFF_CODEC = "cast_fp16"
VAE_DTYPE = torch.bfloat16
BATCH_SIZE = 8
NUM_LOADER_WORKERS = 8
BENCHMARK_ENCODE_BATCH_SIZE = 8
BARRIER_TIMEOUT = timedelta(hours=3)

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encodes the Ming-Image latent datasets.")
    parser.add_argument("--smoke", action="store_true", help="A few samples into dev/smoke_ming_dataset.")
    return parser.parse_args()


def to_pixels(images: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    args:
        images: [B, H, W, 3], channel_order: RGB, dtype: uint8
    returns:
        pixels: [B, 3, H, W], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32, on DEVICE
    """
    images = torch.as_tensor(images).to(DEVICE)
    return images.permute(0, 3, 1, 2).float() / 255.0


def encode(vae: MingImageVAE, pixels: torch.Tensor) -> torch.Tensor:
    """
    args:
        pixels: [B, 3, H, W], channel_order: RGB, pixel_values: [0...1]
    returns:
        latents: [B, 16, H / 8, W / 8], normalized Ming-Image latents of the opaque image, dtype: torch.float32
    """
    opaque_pixels = torch.cat([pixels, torch.ones_like(pixels[:, :1])], dim=1)  # [B, 4, H, W] RGBA
    return vae.encode(opaque_pixels)[:, :, 0]


class TrainSourceDataset(Dataset):
    """
    CPU side: reads a source image, crops it to 1024 and produces the bilinear 512 input.
    """

    def __init__(self, source_paths: list[Path], indices: list[int]) -> None:
        self.source_paths = source_paths
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        """
        returns:
            index, valid, image_1024: [1024, 1024, 3] uint8 RGB, image_512: [512, 512, 3] uint8 RGB
        """
        index = self.indices[position]
        image = cv2.imread(str(self.source_paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            return {
                "index": index,
                "valid": False,
                "image_1024": torch.zeros(HIGH_RESOLUTION_SIZE, HIGH_RESOLUTION_SIZE, 3, dtype=torch.uint8),
                "image_512": torch.zeros(LOW_RESOLUTION_SIZE, LOW_RESOLUTION_SIZE, 3, dtype=torch.uint8),
            }
        image = crop_maximal_rectangle(image, HIGH_RESOLUTION_SIZE, HIGH_RESOLUTION_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_512 = cv2.resize(image, (LOW_RESOLUTION_SIZE, LOW_RESOLUTION_SIZE), interpolation=cv2.INTER_LINEAR)
        return {
            "index": index,
            "valid": True,
            "image_1024": torch.from_numpy(image),
            "image_512": torch.from_numpy(image_512),
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
def encode_train_dataset(
    vae: MingImageVAE, source_directory: Path, output_directory: Path, num_samples: int
) -> None:
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
        pixels_1024 = to_pixels(batch["image_1024"])  # [B, 3, 1024, 1024]
        pixels_512 = to_pixels(batch["image_512"])  # [B, 3, 512, 512]
        latents_1024 = encode(vae, pixels_1024)  # [B, 16, 128, 128]
        latents_512 = encode(vae, pixels_512)  # [B, 16, 64, 64]
        hff_1024 = estimate_immerkaer(pixels_1024)  # [B]
        hff_512 = estimate_immerkaer(pixels_512)  # [B]
        for sample in range(pixels_1024.shape[0]):
            index = int(batch["index"][sample])
            if not bool(batch["valid"][sample]):
                invalid_count += 1
                print(f"[rank {RANK}] unreadable source {source_paths[index]}", flush=True)
                continue
            tensors = {
                "latents_1024": latents_1024[sample : sample + 1],
                "latents_512": latents_512[sample : sample + 1],
                "hff_1024": hff_1024[sample : sample + 1],
                "hff_512": hff_512[sample : sample + 1],
            }
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
def build_benchmark(vae: MingImageVAE, output_directory: Path, max_samples: int | None) -> None:
    if (output_directory / "manifest.json").exists():
        return
    source_paths = list_source_images(AXIS_SOURCE_DIRECTORY)[:max_samples]
    names = [f"{index:04d}" for index in range(len(source_paths))]
    ground_truth_directory = output_directory / "ground_truth"
    low_resolution_directory = output_directory / "low_resolution"
    ground_truth_directory.mkdir(parents=True, exist_ok=True)
    low_resolution_directory.mkdir(parents=True, exist_ok=True)
    for source_path, name in tqdm(list(zip(source_paths, names)), desc="Benchmark images"):
        if (ground_truth_directory / f"{name}.png").exists() and (low_resolution_directory / f"{name}.png").exists():
            continue
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
    vae = MingImageVAE(str(VAE_PATH), device=DEVICE, dtype=VAE_DTYPE)

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
