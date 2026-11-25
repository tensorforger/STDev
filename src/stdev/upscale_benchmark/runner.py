import gc
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from stdev.upscale_benchmark.dataset import BenchmarkDataset, load_image_batch
from stdev.upscale_benchmark.metrics import roundtrip_errors
from stdev.upscale_benchmark.solution import (
    INPUT_LATENT_SIZE,
    LATENT_CHANNELS,
    OUTPUT_PIXEL_SIZE,
    UpscaleSolution,
)
from stdev.utils.fid import FIDEvaluator, FIDReference
from stdev.utils.image_tools import np_to_cv2

BYTES_PER_MEGABYTE = 2**20
FID_PATCH_SIZE = 256
EVALUATION_BATCH_SIZE = 8


def release_gpu_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def save_prediction(pixels: torch.Tensor, path: Path) -> None:
    """
    args:
        pixels: [1, 3, H, W], channel_order: RGB, pixel_values: [0...1]
    """
    image = pixels.clamp(0, 1).mul(255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(str(path), np_to_cv2(image))


@torch.no_grad()
def run_inference(
    solution: UpscaleSolution,
    dataset: BenchmarkDataset,
    predictions_directory: Path,
    seed: int,
    num_warmup_samples: int,
    input_shape: tuple[int, ...] = (LATENT_CHANNELS, 1, INPUT_LATENT_SIZE, INPUT_LATENT_SIZE),
) -> dict[str, float]:
    """
    Loads the solution, warms it up on random latents, then upscales every dataset sample with batch size 1 and
    saves the 1024x1024 predictions as PNG. `input_shape` is the per sample latent shape of the warmup inputs.
    Latency is the synchronized wall time of `upscale` (decoding included),
    peak VRAM is the peak of allocated CUDA memory above the level before loading (weights included).

    returns:
        latency and VRAM statistics
    """
    predictions_directory.mkdir(parents=True, exist_ok=True)
    release_gpu_memory()
    baseline_bytes = torch.cuda.memory_allocated()
    solution.load()
    torch.cuda.synchronize()
    weights_bytes = torch.cuda.memory_allocated() - baseline_bytes

    generator = torch.Generator(device="cuda").manual_seed(seed)
    for _ in tqdm(range(num_warmup_samples), desc=f"{solution.name} warmup", leave=False):
        fake_latents = torch.randn(1, *input_shape, generator=generator, device="cuda")
        solution.upscale(fake_latents, seed, torch.rand(1, generator=generator, device="cuda"))
    torch.cuda.synchronize()

    latencies_ms: list[float] = []
    peak_bytes = 0
    for index in tqdm(range(len(dataset)), desc=solution.name):
        latents = dataset.load_latents(index).to("cuda")
        hff = dataset.load_hff(index).to("cuda") if solution.uses_hff else None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        pixels = solution.upscale(latents, seed + index, hff)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start_time) * 1000)
        peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated())

        expected_shape = (1, 3, OUTPUT_PIXEL_SIZE, OUTPUT_PIXEL_SIZE)
        if tuple(pixels.shape) != expected_shape:
            raise ValueError(
                f"{solution.name} returned {tuple(pixels.shape)}, expected {expected_shape}"
            )
        save_prediction(pixels, predictions_directory / f"{dataset.sample_names[index]}.png")
        del pixels

    return {
        "latency_ms": float(np.mean(latencies_ms)),
        "latency_ms_median": float(np.median(latencies_ms)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
        "peak_vram_mb": (peak_bytes - baseline_bytes) / BYTES_PER_MEGABYTE,
        "weights_vram_mb": weights_bytes / BYTES_PER_MEGABYTE,
    }


def build_fid_evaluator(dataset: BenchmarkDataset, statistics_directory: Path) -> FIDEvaluator:
    """
    `fid` compares whole images resized to 299 by Inception, `patch_fid` compares native resolution
    FID_PATCH_SIZE tiles. Statistics of the ground truth are cached per number of samples.
    """
    common_arguments = {
        "image_dir": str(dataset.ground_truth_directory),
        "file_names": [f"{name}.png" for name in dataset.sample_names],
    }
    return FIDEvaluator(
        references=[
            FIDReference(
                name="fid",
                stats_path=str(statistics_directory / f"fid_stats_{len(dataset)}.pt"),
                **common_arguments,
            ),
            FIDReference(
                name="patch_fid",
                stats_path=str(statistics_directory / f"patch_fid_stats_{len(dataset)}.pt"),
                patch_size=FID_PATCH_SIZE,
                **common_arguments,
            ),
        ],
        image_size=OUTPUT_PIXEL_SIZE,
        batch_size=16,
    )


@torch.no_grad()
def evaluate_predictions(
    solution_name: str,
    predictions_directory: Path,
    dataset: BenchmarkDataset,
    statistics_directory: Path,
) -> dict[str, float]:
    """
    Quality of the saved predictions: FID and patch FID against the ground truth, and the roundtrip errors
    (prediction downsampled to 512, compared with the initial 512 image) as MSE and CIEDE2000.
    """
    fid_evaluator = build_fid_evaluator(dataset, statistics_directory)
    mse_values: list[float] = []
    color_difference_values: list[float] = []
    for start in tqdm(
        range(0, len(dataset), EVALUATION_BATCH_SIZE), desc=f"{solution_name} evaluation"
    ):
        sample_names = dataset.sample_names[start : start + EVALUATION_BATCH_SIZE]
        predicted_pixels = load_image_batch(
            predictions_directory, sample_names
        )  # [B, 3, 1024, 1024]
        initial_pixels = load_image_batch(
            dataset.low_resolution_directory, sample_names
        )  # [B, 3, 512, 512]
        fid_evaluator.update(predicted_pixels)
        mse, color_difference = roundtrip_errors(predicted_pixels, initial_pixels)
        mse_values.extend(mse.tolist())
        color_difference_values.extend(color_difference.tolist())
    quality = fid_evaluator.compute()
    quality["low_res_mse"] = float(np.mean(mse_values))
    quality["chroma_shift_ciede2000"] = float(np.mean(color_difference_values))
    return quality
