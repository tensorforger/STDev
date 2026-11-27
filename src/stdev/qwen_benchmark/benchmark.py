"""
Benchmark of Qwen-Image 2.1 latent upscalers (autoresearch_4): 512 -> 1024 px (32x32 -> 64x64 latents, 64 channels)
on the fixed Axis-V1 1k set built by `scripts/encode_qwen_datasets.py`. Predictions are decoded with the texture-fix
Qwen-Image 2.1 decoder (same latent space as the official VAE); the RGB channels of the RGBA output are evaluated.

Main metrics: patch_fid, fid, low_res_mse, chroma_ciede2000 (quality of the decoded 1024 predictions, same definitions
as `stdev.upscale_benchmark`), latency_ms_p95 and peak_vram_mb (single model forward, batch 1).
"""

import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from tqdm import tqdm

from stdev.qwen_upscalers.base import Upscaler
from stdev.upscale_benchmark.metrics import roundtrip_errors
from stdev.utils.fid import FIDEvaluator, FIDReference
from stdev.vaes.qwen_image_2_1_vae import QwenImage21VAE

LATENT_CHANNELS = 64
INPUT_LATENT_SIZE = 32
OUTPUT_PIXEL_SIZE = 1024
FID_PATCH_SIZE = 256
QUALITY_BATCH_SIZE = 8
DECODE_BATCH_SIZE = 1
BYTES_PER_MEGABYTE = 2**20


class QwenBenchmarkDataset:
    """
    `ground_truth/<name>.png` (1024), `low_resolution/<name>.png` (512, bilinear downscale of the ground truth),
    `latents.pt` ([N, 64, 32, 32], low resolution latents), `latents_hr.pt` ([N, 64, 64, 64]), `manifest.json`.
    """

    def __init__(self, dataset_directory: Path, max_samples: int | None = None) -> None:
        manifest = json.loads((dataset_directory / "manifest.json").read_text())
        self.dataset_directory = dataset_directory
        self.ground_truth_directory = dataset_directory / "ground_truth"
        self.low_resolution_directory = dataset_directory / "low_resolution"
        self.sample_names: list[str] = manifest["sample_names"][:max_samples]
        self.latents = torch.load(dataset_directory / "latents.pt")[: len(self.sample_names)]
        self.latents_high = torch.load(dataset_directory / "latents_hr.pt")[: len(self.sample_names)]

    def __len__(self) -> int:
        return len(self.sample_names)

    def load_low_resolution_pixels(self, names: list[str]) -> torch.Tensor:
        """
        returns:
            pixels: [B, 3, 512, 512], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32, on cuda
        """
        images = np.stack(
            [
                cv2.cvtColor(cv2.imread(str(self.low_resolution_directory / f"{name}.png")), cv2.COLOR_BGR2RGB)
                for name in names
            ]
        )
        return torch.from_numpy(images).cuda().permute(0, 3, 1, 2).float() / 255.0


def build_fid_evaluator(dataset: QwenBenchmarkDataset, statistics_directory: Path) -> FIDEvaluator:
    """
    `fid`: whole images resized to 299 by Inception, `patch_fid`: native resolution FID_PATCH_SIZE tiles.
    Ground truth statistics are cached per number of samples.
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
        image_size=OUTPUT_PIXEL_SIZE,
        batch_size=16,
    )


def decode_rgb(vae: QwenImage21VAE, latents: torch.Tensor) -> torch.Tensor:
    """
    Decodes in chunks of DECODE_BATCH_SIZE (the 1024 decode peaks at ~6.7 GB per image in bf16).

    args:
        latents: [B, 64, h, w], normalized Qwen-Image 2.1 latents
    returns:
        pixels: [B, 3, 16h, 16w], channel_order: RGB, pixel_values: [0...1] (not clamped), dtype: torch.float32
    """
    pixels = [
        vae.decode(latents[start : start + DECODE_BATCH_SIZE].unsqueeze(2))[:, :3]
        for start in range(0, latents.shape[0], DECODE_BATCH_SIZE)
    ]
    return torch.cat(pixels)


def save_image(pixels: torch.Tensor, path: Path) -> None:
    """
    args:
        pixels: [3, H, W], channel_order: RGB, pixel_values: [0...1]
    """
    image = pixels.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def evaluate_quality(
    predict_pixels: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataset: QwenBenchmarkDataset,
    fid_evaluator: FIDEvaluator,
    samples_directory: Path | None,
    num_saved_samples: int,
) -> dict[str, float]:
    """
    args:
        predict_pixels: (latents [B, 64, 32, 32], indices [B]) -> pixels [B, 3, 1024, 1024] in [0...1]
    returns:
        fid, patch_fid, low_res_mse, chroma_ciede2000
    """
    if samples_directory is not None:
        samples_directory.mkdir(parents=True, exist_ok=True)
    mse_values: list[float] = []
    color_difference_values: list[float] = []
    for start in tqdm(range(0, len(dataset), QUALITY_BATCH_SIZE), desc="Quality"):
        indices = torch.arange(start, min(start + QUALITY_BATCH_SIZE, len(dataset)))
        names = [dataset.sample_names[index] for index in indices]
        pixels = predict_pixels(dataset.latents[indices].cuda(), indices)
        pixels = pixels.clamp(0, 1).mul(255).round().div(255)  # same quantization as a saved PNG
        if tuple(pixels.shape[1:]) != (3, OUTPUT_PIXEL_SIZE, OUTPUT_PIXEL_SIZE):
            raise ValueError(f"Prediction shape {tuple(pixels.shape)}")
        fid_evaluator.update(pixels)
        mse, color_difference = roundtrip_errors(pixels, dataset.load_low_resolution_pixels(names))
        mse_values.extend(mse.tolist())
        color_difference_values.extend(color_difference.tolist())
        if samples_directory is not None:
            for offset, index in enumerate(indices.tolist()):
                if index < num_saved_samples:
                    save_image(pixels[offset], samples_directory / f"{dataset.sample_names[index]}.png")
    quality = fid_evaluator.compute()
    quality["low_res_mse"] = float(np.mean(mse_values))
    quality["chroma_ciede2000"] = float(np.mean(color_difference_values))
    return quality


def upscaler_pixel_predictor(
    upscaler: Upscaler, vae: QwenImage21VAE, dataset: QwenBenchmarkDataset, latent_errors: list[float]
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Upscales latents with the upscaler and decodes them; collects the latent MSE to the ground truth latents.
    """

    def predict(latents: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        upscaled = upscaler.upscale(latents).float()  # [B, 64, 64, 64]
        expected_shape = (latents.shape[0], LATENT_CHANNELS, 2 * INPUT_LATENT_SIZE, 2 * INPUT_LATENT_SIZE)
        if tuple(upscaled.shape) != expected_shape:
            raise ValueError(f"upscale returned {tuple(upscaled.shape)}, expected {expected_shape}")
        latent_errors.extend(
            (upscaled - dataset.latents_high[indices].cuda()).square().mean(dim=(1, 2, 3)).tolist()
        )
        return decode_rgb(vae, upscaled)

    return predict


def reference_pixel_predictor(
    reference: str, vae: QwenImage21VAE, dataset: QwenBenchmarkDataset
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Non-learned references: `oracle` (VAE roundtrip of the ground truth), `pixel_bicubic` (decoded 512 image,
    bicubic 2x), `latent_bilinear` (bilinear 2x latents, decoded).
    """

    def predict(latents: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if reference == "oracle":
            return decode_rgb(vae, dataset.latents_high[indices].cuda())
        if reference == "pixel_bicubic":
            pixels = decode_rgb(vae, latents)
            return functional.interpolate(pixels, scale_factor=2, mode="bicubic", align_corners=False)
        if reference == "latent_bilinear":
            upscaled = functional.interpolate(latents, scale_factor=2, mode="bilinear", align_corners=False)
            return decode_rgb(vae, upscaled)
        raise ValueError(f"Unknown reference {reference}")

    return predict


REFERENCES = ("oracle", "pixel_bicubic", "latent_bilinear")


@torch.no_grad()
def measure_peak_memory(upscaler: Upscaler, iterations: int) -> dict[str, float]:
    """
    Peak allocated CUDA memory of the eager single forward (batch 1, 32x32 -> 64x64), weights included.
    The upscaler weights must not be loaded yet: memory is counted from the level before `prepare_to_benchmark`.
    """
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline_bytes = torch.cuda.memory_allocated()
    upscaler.prepare_to_benchmark()
    torch.cuda.synchronize()
    weights_bytes = torch.cuda.memory_allocated() - baseline_bytes
    latents = torch.randn(1, LATENT_CHANNELS, INPUT_LATENT_SIZE, INPUT_LATENT_SIZE, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iterations):
        output = upscaler.single_forward(latents)
        del output
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated() - baseline_bytes
    return {
        "peak_vram_mb": peak_bytes / BYTES_PER_MEGABYTE,
        "weights_vram_mb": weights_bytes / BYTES_PER_MEGABYTE,
    }


@torch.no_grad()
def measure_latency(
    function: Callable[[], object], warmup_iterations: int, iterations: int
) -> dict[str, float]:
    """
    Synchronized wall time per call after warmup.

    returns:
        mean and p95 latency in ms
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
