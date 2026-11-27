"""
Benchmark of autoresearch_6 (MiniMax-H3-Image latents -> FLUX.2 latents, same pixel resolution) on the fixed Axis-V1
1k set built by `scripts/encode_h3_datasets.py`.

Quality (1024 px, all 1000 images): patch_fid (native 256 tiles, main), fid, psnr against the ground truth,
low_res_mse / chroma_ciede2000 (roundtrip to the 512 bilinear downscale, as the earlier benchmarks).
Speed: batch 1 latency and peak VRAM of the full latent -> pixels path at 1024, 2048 and 4096 px.
"""

import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from stdev.upscale_benchmark.metrics import roundtrip_errors
from stdev.utils.fid import FIDEvaluator, FIDReference

H3_LATENT_CHANNELS = 24
FLUX2_LATENT_CHANNELS = 32
IMAGE_SIZE = 1024
FID_PATCH_SIZE = 256
QUALITY_BATCH_SIZE = 8
SPEED_RESOLUTIONS = (1024, 2048, 4096)
BYTES_PER_MEGABYTE = 2**20


class H3BenchmarkDataset:
    """
    `ground_truth/<name>.png` (1024), `low_resolution/<name>.png` (512 bilinear), `latents_h3.pt` ([N, 24, 64, 64]),
    `latents_flux2.pt` ([N, 32, 128, 128]), `manifest.json`.
    """

    def __init__(self, dataset_directory: Path, max_samples: int | None = None) -> None:
        manifest = json.loads((dataset_directory / "manifest.json").read_text())
        self.ground_truth_directory = dataset_directory / "ground_truth"
        self.low_resolution_directory = dataset_directory / "low_resolution"
        self.sample_names: list[str] = manifest["sample_names"][:max_samples]
        self.latents_h3 = torch.load(dataset_directory / "latents_h3.pt")[: len(self.sample_names)]
        self.latents_flux2 = torch.load(dataset_directory / "latents_flux2.pt")[: len(self.sample_names)]

    def __len__(self) -> int:
        return len(self.sample_names)

    def load_pixels(self, directory: Path, names: list[str]) -> torch.Tensor:
        """
        returns:
            pixels: [B, 3, H, W], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32, on cuda
        """
        images = np.stack(
            [cv2.cvtColor(cv2.imread(str(directory / f"{name}.png")), cv2.COLOR_BGR2RGB) for name in names]
        )
        return torch.from_numpy(images).cuda().permute(0, 3, 1, 2).float() / 255.0


def build_fid_evaluator(dataset: H3BenchmarkDataset, statistics_directory: Path) -> FIDEvaluator:
    """
    `fid`: whole images resized to 299 by Inception, `patch_fid`: native resolution FID_PATCH_SIZE tiles.
    """
    common_arguments = {
        "image_dir": str(dataset.ground_truth_directory),
        "file_names": [f"{name}.png" for name in dataset.sample_names],
    }
    return FIDEvaluator(
        references=[
            FIDReference(
                name="fid", stats_path=str(statistics_directory / f"fid_stats_{len(dataset)}.pt"), **common_arguments
            ),
            FIDReference(
                name="patch_fid",
                stats_path=str(statistics_directory / f"patch_fid_stats_{len(dataset)}.pt"),
                patch_size=FID_PATCH_SIZE,
                **common_arguments,
            ),
        ],
        image_size=IMAGE_SIZE,
        batch_size=16,
    )


def save_image(pixels: torch.Tensor, path: Path) -> None:
    """
    args:
        pixels: [3, H, W], channel_order: RGB, pixel_values: [0...1]
    """
    image = pixels.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def evaluate_quality(
    predict_pixels: Callable[[torch.Tensor], torch.Tensor],
    dataset: H3BenchmarkDataset,
    fid_evaluator: FIDEvaluator,
    samples_directory: Path | None,
    num_saved_samples: int,
) -> dict[str, float]:
    """
    args:
        predict_pixels: indices [B] -> pixels [B, 3, 1024, 1024] in [0...1]
    returns:
        fid, patch_fid, psnr, low_res_mse, chroma_ciede2000
    """
    if samples_directory is not None:
        samples_directory.mkdir(parents=True, exist_ok=True)
    mse_values: list[float] = []
    color_difference_values: list[float] = []
    psnr_values: list[float] = []
    for start in tqdm(range(0, len(dataset), QUALITY_BATCH_SIZE), desc="Quality"):
        indices = torch.arange(start, min(start + QUALITY_BATCH_SIZE, len(dataset)))
        names = [dataset.sample_names[index] for index in indices]
        pixels = predict_pixels(indices)
        pixels = pixels.clamp(0, 1).mul(255).round().div(255)  # same quantization as a saved PNG
        if tuple(pixels.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Prediction shape {tuple(pixels.shape)}")
        fid_evaluator.update(pixels)
        ground_truth = dataset.load_pixels(dataset.ground_truth_directory, names)
        full_mse = (pixels.double() - ground_truth.double()).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
        psnr_values.extend((10 * torch.log10(1.0 / full_mse)).tolist())
        mse, color_difference = roundtrip_errors(pixels, dataset.load_pixels(dataset.low_resolution_directory, names))
        mse_values.extend(mse.tolist())
        color_difference_values.extend(color_difference.tolist())
        if samples_directory is not None:
            for offset, index in enumerate(indices.tolist()):
                if index < num_saved_samples:
                    save_image(pixels[offset], samples_directory / f"{dataset.sample_names[index]}.png")
    quality = fid_evaluator.compute()
    quality["psnr"] = float(np.mean(psnr_values))
    quality["low_res_mse"] = float(np.mean(mse_values))
    quality["chroma_ciede2000"] = float(np.mean(color_difference_values))
    return quality


@torch.no_grad()
def measure_latency(function: Callable[[], object], warmup_iterations: int, iterations: int) -> dict[str, float]:
    """
    returns:
        mean and p95 synchronized wall time per call in ms
    """
    for _ in range(warmup_iterations):
        function()
    torch.cuda.synchronize()
    latencies_ms = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        function()
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start_time) * 1000)
    return {"mean": float(np.mean(latencies_ms)), "p95": float(np.percentile(latencies_ms, 95))}


@torch.no_grad()
def measure_peak_memory(function: Callable[[], object]) -> float:
    """
    returns:
        peak allocated CUDA memory during one call above the level before it (weights excluded), MB
    """
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    function()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline_bytes) / BYTES_PER_MEGABYTE
