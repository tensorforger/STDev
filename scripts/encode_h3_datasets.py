"""
Encodes the datasets of autoresearch_6: MiniMax-H3-Image latents (16x, 24 channels) as the condition and FLUX.2
latents (8x, 32 channels) of the same 1024 image as the target ("H3 -> FLUX.2 latent translator", no upscaling in
pixel space).

Prerequisites: 4KLSDB sources from `scripts/download_flux2_sources.py 4klsdb --num-files 65` (may still be running:
`--watch` encodes every finished parquet file and waits for the rest), Axis-V1 in `datasets/axis-v1_1k`, the VAEs in
`models/MiniMax-H3-Image/vae_diffusers` and `models/FLUX.2-klein-4B/vae`.

Run (4 GPUs; the benchmark is built by rank 0 first, an interrupted run is resumed by the same command):
    torchrun --standalone --nproc_per_node=4 scripts/encode_h3_datasets.py --watch
Quick check into `dev/smoke_h3_dataset`:
    python scripts/encode_h3_datasets.py --smoke

Every image is center cropped and resized to 1024x1024 (`crop_maximal_rectangle`, antialiased). The H3 encoder runs
in fp32 with the released tiling (256 px tiles, posterior mode); the FLUX.2 VAE in bf16 (posterior mean). Both are
normalized with their wrapper statistics and stored as fp16 (`cast_fp16`).

Benchmark `benchmark_datasets/axis_v1_1k_h3_flux2`: ground_truth/<name>.png (1024), low_resolution/<name>.png (512,
bilinear, only for the roundtrip metrics), latents_h3.pt ([N, 24, 64, 64]), latents_flux2.pt ([N, 32, 128, 128]),
hff.pt ([N]), manifest.json.

Training dataset `train_datasets/h3_flux2_4klsdb_40k`: block_<source stem>.vz.safetensors with normalized latents:
    latents_h3              [1, 24, 64, 64]     condition
    latents_flux2           [1, 32, 128, 128]   target
    hff_1024                [1]                 Immerkaer hff (unused in research)
"""

import argparse
import json
import os
import time
import zlib
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
from stdev.vaes.flux_2_vae import Flux2VAE
from stdev.vaes.minimax_h3_image_vae import MiniMaxH3ImageVAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
H3_VAE_PATH = REPOSITORY_ROOT / "models" / "MiniMax-H3-Image" / "vae_diffusers"
FLUX2_VAE_PATH = REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B" / "vae"
AXIS_SOURCE_DIRECTORY = REPOSITORY_ROOT / "datasets" / "axis-v1_1k"
TRAIN_SOURCE_DIRECTORY = REPOSITORY_ROOT / "datasets" / "4klsdb_1024"
BENCHMARK_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "axis_v1_1k_h3_flux2"
TRAIN_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "h3_flux2_4klsdb_40k"
SMOKE_DIRECTORY = REPOSITORY_ROOT / "dev" / "smoke_h3_dataset"

NUM_SOURCE_FILES = 65
IMAGE_SIZE = 1024
LOW_RESOLUTION_SIZE = 512
NUM_SMOKE_SAMPLES = 4
LATENT_CODEC = "cast_fp16"
HFF_CODEC = "cast_fp16"
FLUX2_DTYPE = torch.bfloat16
H3_DECODER_DTYPE = torch.bfloat16
BATCH_SIZE = 8
NUM_LOADER_WORKERS = 8
WATCH_INTERVAL_SECONDS = 60
BARRIER_TIMEOUT = timedelta(hours=6)

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encodes the H3 -> FLUX.2 latent datasets.")
    parser.add_argument("--smoke", action="store_true", help="A few samples into dev/smoke_h3_dataset.")
    parser.add_argument("--watch", action="store_true", help="Wait for all 4KLSDB source files.")
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


class TrainSourceDataset(Dataset):
    """
    CPU side: reads a source image and crops it to 1024.
    """

    def __init__(self, source_paths: list[Path]) -> None:
        self.source_paths = source_paths

    def __len__(self) -> int:
        return len(self.source_paths)

    def __getitem__(self, position: int) -> dict:
        """
        returns:
            position, valid, image: [1024, 1024, 3] uint8 RGB
        """
        image = cv2.imread(str(self.source_paths[position]), cv2.IMREAD_COLOR)
        if image is None:
            return {
                "position": position,
                "valid": False,
                "image": torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 3, dtype=torch.uint8),
            }
        image = cv2.cvtColor(crop_maximal_rectangle(image, IMAGE_SIZE, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
        return {"position": position, "valid": True, "image": torch.from_numpy(image)}


def save_block(tensors: dict[str, torch.Tensor], path: Path) -> None:
    """
    Writes through a temporary file, so an interrupted run never leaves a partial block.
    """
    encoded_tensors = {
        name: vaezip.encode(tensor.float().cpu(), codec=HFF_CODEC if name.startswith("hff") else LATENT_CODEC)
        for name, tensor in tensors.items()
    }
    temporary_path = path.with_name(path.name + ".tmp")
    vaezip.save(encoded_tensors, temporary_path)
    temporary_path.rename(path)


def block_path(directory: Path, source_path: Path) -> Path:
    return directory / f"block_{source_path.stem}.vz.safetensors"


def finished_source_paths(source_directory: Path) -> list[Path]:
    """
    returns:
        sorted images of the parquet files whose download finished (`.file_<index>.done` markers)
    """
    finished_files = {
        int(marker.name.split("_")[1].split(".")[0]) for marker in source_directory.glob(".file_*.done")
    }
    return [path for path in list_source_images(source_directory) if int(path.stem.split("_")[0]) in finished_files]


def count_finished_files(source_directory: Path) -> int:
    return len(list(source_directory.glob(".file_*.done")))


@torch.no_grad()
def encode_train_sources(
    h3_vae: MiniMaxH3ImageVAE, flux2_vae: Flux2VAE, source_paths: list[Path], output_directory: Path
) -> int:
    """
    Encodes the not yet encoded sources of this rank (assigned by a hash of the file name).

    returns:
        number of new blocks
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    pending = [
        path
        for path in source_paths
        if zlib.crc32(path.name.encode()) % WORLD_SIZE == RANK and not block_path(output_directory, path).exists()
    ]
    if not pending:
        return 0
    loader = DataLoader(
        TrainSourceDataset(pending), batch_size=BATCH_SIZE, num_workers=NUM_LOADER_WORKERS, pin_memory=True
    )
    written = 0
    for batch in tqdm(loader, desc=f"{output_directory.name} [rank {RANK}]", position=RANK):
        pixels = to_pixels(batch["image"])  # [B, 3, 1024, 1024]
        latents_h3 = h3_vae.encode(pixels)[:, :, 0]  # [B, 24, 64, 64]
        latents_flux2 = flux2_vae.encode(pixels)[:, :, 0]  # [B, 32, 128, 128]
        hff = estimate_immerkaer(pixels)  # [B]
        for sample in range(pixels.shape[0]):
            source_path = pending[int(batch["position"][sample])]
            if not bool(batch["valid"][sample]):
                print(f"[rank {RANK}] unreadable source {source_path}", flush=True)
                continue
            tensors = {
                "latents_h3": latents_h3[sample : sample + 1],
                "latents_flux2": latents_flux2[sample : sample + 1],
                "hff_1024": hff[sample : sample + 1],
            }
            save_block(tensors, block_path(output_directory, source_path))
            written += 1
    return written


def prepare_benchmark_image(source_path: Path, ground_truth_path: Path, low_resolution_path: Path) -> None:
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    ground_truth = crop_maximal_rectangle(image, IMAGE_SIZE, IMAGE_SIZE)
    low_resolution = cv2.resize(ground_truth, (LOW_RESOLUTION_SIZE, LOW_RESOLUTION_SIZE), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(ground_truth_path), ground_truth)
    cv2.imwrite(str(low_resolution_path), low_resolution)


def load_png_batch(directory: Path, names: list[str]) -> torch.Tensor:
    """
    returns:
        pixels: [B, 3, H, W], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32, on DEVICE
    """
    images = np.stack([cv2.cvtColor(cv2.imread(str(directory / f"{name}.png")), cv2.COLOR_BGR2RGB) for name in names])
    return to_pixels(images)


@torch.no_grad()
def build_benchmark(
    h3_vae: MiniMaxH3ImageVAE, flux2_vae: Flux2VAE, output_directory: Path, max_samples: int | None
) -> None:
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
    latents_h3, latents_flux2, hff = [], [], []
    for start in tqdm(range(0, len(names), BATCH_SIZE), desc="Benchmark encode"):
        pixels = load_png_batch(ground_truth_directory, names[start : start + BATCH_SIZE])
        latents_h3.append(h3_vae.encode(pixels)[:, :, 0].cpu())
        latents_flux2.append(flux2_vae.encode(pixels)[:, :, 0].cpu())
        hff.append(estimate_immerkaer(pixels).cpu())
    torch.save(torch.cat(latents_h3), output_directory / "latents_h3.pt")
    torch.save(torch.cat(latents_flux2), output_directory / "latents_flux2.pt")
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
    h3_vae = MiniMaxH3ImageVAE(str(H3_VAE_PATH), device=DEVICE, dtype=H3_DECODER_DTYPE)
    flux2_vae = Flux2VAE(str(FLUX2_VAE_PATH), device=DEVICE, dtype=FLUX2_DTYPE)

    if arguments.smoke:
        build_benchmark(h3_vae, flux2_vae, SMOKE_DIRECTORY / "benchmark", NUM_SMOKE_SAMPLES)
        source_paths = list_source_images(TRAIN_SOURCE_DIRECTORY)[:NUM_SMOKE_SAMPLES]
        encode_train_sources(h3_vae, flux2_vae, source_paths, SMOKE_DIRECTORY / TRAIN_DIRECTORY.name)
        return

    if RANK == 0:
        build_benchmark(h3_vae, flux2_vae, BENCHMARK_DIRECTORY, None)
    while True:
        all_files_finished = count_finished_files(TRAIN_SOURCE_DIRECTORY) >= NUM_SOURCE_FILES
        written = encode_train_sources(h3_vae, flux2_vae, finished_source_paths(TRAIN_SOURCE_DIRECTORY), TRAIN_DIRECTORY)
        if all_files_finished or not arguments.watch:
            break
        if written == 0:
            time.sleep(WATCH_INTERVAL_SECONDS)
    if WORLD_SIZE > 1:
        distributed.barrier()
        distributed.destroy_process_group()
    if RANK == 0:
        print(f"{len(list(TRAIN_DIRECTORY.glob('block_*.vz.safetensors')))} training blocks", flush=True)


if __name__ == "__main__":
    main()
