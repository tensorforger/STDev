from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from tqdm import tqdm

from stdev.benchmark.cache import LatentCache
from stdev.benchmark.dataset import BenchmarkDataset
from stdev.upscalers.base import Upscaler
from stdev.utils.fid import FIDEvaluator, FIDReference
from stdev.utils.image_tools import torch_to_cv2
from stdev.vaes.image_vae import ImageVAE

FID_PATCH_SIZE = 256


def resize_bilinear(image: torch.Tensor, scale: float) -> torch.Tensor:
    """
    args:
        image: [1, 3, H, W], pixel_values: [0...1], dtype: torch.float32
    returns:
        image: [1, 3, H * scale, W * scale], pixel_values: [0...1], dtype: torch.float32
    """
    target_size = (round(image.shape[2] * scale), round(image.shape[3] * scale))
    return functional.interpolate(image, size=target_size, mode="bilinear", align_corners=False)


@torch.no_grad()
def populate_latent_cache(
    dataset: BenchmarkDataset,
    vae: ImageVAE,
    cache: LatentCache,
    scale_factor: int,
    seed: int,
) -> None:
    """
    Encodes ground truth `latents_hr_gt` and `latents_lr_gt` of the samples missing in the cache.
    """
    for sample_name in tqdm(dataset.sample_names, desc="Caching latents"):
        if cache.has_latents(sample_name):
            continue
        image_hr_gt = dataset.load_image(sample_name)
        image_lr_gt = resize_bilinear(image_hr_gt, 1 / scale_factor)
        for kind, image in (("hr_gt", image_hr_gt), ("lr_gt", image_lr_gt)):
            torch.manual_seed(seed)
            cache.save_latents(kind, sample_name, vae.encode(image))


def build_fid_evaluator(
    dataset: BenchmarkDataset,
    cache: LatentCache,
    image_size: int,
) -> FIDEvaluator:
    """
    `fid` compares whole images resized to 299 by Inception (global structure and color).
    `fid_patch` compares native resolution FID_PATCH_SIZE tiles (generated high frequency detail).
    """
    return FIDEvaluator(
        references=[
            FIDReference(
                name="fid",
                image_dir=str(dataset.images_directory),
                stats_path=str(cache.fid_statistics_path),
                file_names=dataset.get_image_file_names(),
            ),
            FIDReference(
                name="fid_patch",
                image_dir=str(dataset.images_directory),
                stats_path=str(cache.fid_patch_statistics_path),
                file_names=dataset.get_image_file_names(),
                patch_size=FID_PATCH_SIZE,
            ),
        ],
        image_size=image_size,
        batch_size=16,
    )


def _save_image(image: torch.Tensor, path: Path) -> None:
    """
    args:
        image: [1, 3, H, W], channel_order: RGB, pixel_values: [0...1]
    """
    cv2.imwrite(str(path), torch_to_cv2(image))


@torch.no_grad()
def evaluate_quality(
    upscaler: Upscaler,
    vae: ImageVAE,
    dataset: BenchmarkDataset,
    cache: LatentCache,
    fid_evaluator: FIDEvaluator,
    scale_factor: int,
    upscaler_dtype: torch.dtype,
    images_directory: Path,
    num_saved_samples: int,
) -> dict[str, float]:
    """
    Roundtrip of the ground truth through downscale, encode, upscale, decode, and the metrics.
    Saves `<sample_name>_in.png` (pixel upscaled ground truth) and `<sample_name>_out.png` pairs.
    """
    images_directory.mkdir(parents=True, exist_ok=True)
    saved_indices = set(
        np.linspace(0, len(dataset) - 1, min(num_saved_samples, len(dataset)))
        .round()
        .astype(int)
        .tolist()
    )
    metric_values: dict[str, list[float]] = {
        "mse_hr_image": [],
        "mse_lr_image": [],
        "mse_hr_latents": [],
    }

    for index, sample_name in enumerate(tqdm(dataset.sample_names, desc="Quality")):
        image_hr_gt = dataset.load_image(sample_name)  # [1, 3, H, W]
        image_lr_gt = resize_bilinear(image_hr_gt, 1 / scale_factor)
        latents_lr_gt = cache.load_latents("lr_gt", sample_name)
        latents_hr_gt = cache.load_latents("hr_gt", sample_name)

        latents_hr_pred = upscaler.upscale(
            latents_lr_gt.to(upscaler_dtype), dataset.load_prompt_embeds(sample_name)
        ).float()
        image_hr_pred = vae.decode(latents_hr_pred).float().clamp(0, 1)
        image_lr_pred = resize_bilinear(image_hr_pred, 1 / scale_factor)

        metric_values["mse_hr_image"].append(functional.mse_loss(image_hr_pred, image_hr_gt).item())
        metric_values["mse_lr_image"].append(functional.mse_loss(image_lr_pred, image_lr_gt).item())
        metric_values["mse_hr_latents"].append(
            functional.mse_loss(latents_hr_pred, latents_hr_gt).item()
        )
        fid_evaluator.update(image_hr_pred)

        if index in saved_indices:
            _save_image(
                resize_bilinear(image_lr_gt, scale_factor),
                images_directory / f"{sample_name}_in.png",
            )
            _save_image(image_hr_pred, images_directory / f"{sample_name}_out.png")

    quality = {name: float(np.mean(values)) for name, values in metric_values.items()}
    quality.update(fid_evaluator.compute())
    return quality
