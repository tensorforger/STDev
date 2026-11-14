from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from tqdm import tqdm

from stdev.utils.fid import FIDEvaluator, FIDReference
from stdev.utils.image_tools import torch_to_cv2
from stdev.vaes.vae import VAE
from stdev.wan_benchmark.dataset import WanBenchmarkDataset
from stdev.wan_upscalers.base import Upscaler

FID_PATCH_SIZE = 256
VIDEO_SAVE_FPS = 8


@dataclass(frozen=True)
class QualityTask:
    """
    Which block entries form the input, the ground truth and the conditioning.
    `lr_frame_stride` and `lr_spatial_scale` map high resolution pixels onto the pixels of the input.
    `image_size` is the side of the ground truth frames.
    """

    kind: str
    source_key: str
    target_key: str
    hff_key: str
    lr_frame_stride: int
    lr_spatial_scale: int
    image_size: int


QUALITY_TASKS = {
    "wan_image": QualityTask(
        kind="image",
        source_key="latents_512",
        target_key="latents_1024",
        hff_key="hff_1024",
        lr_frame_stride=1,
        lr_spatial_scale=2,
        image_size=1024,
    ),
    "wan_video": QualityTask(
        kind="video",
        source_key="latents_17_512",
        target_key="latents_33_512",
        hff_key="hff_33_512",
        lr_frame_stride=2,
        lr_spatial_scale=1,
        image_size=512,
    ),
}


def project_to_input_pixels(pixels: torch.Tensor, task: QualityTask) -> torch.Tensor:
    """
    Pixels of the high resolution sample at the frames and resolution of the input sample.

    args:
        pixels: [1, 3, T, H, W]
    returns:
        pixels: [1, 3, ceil(T / lr_frame_stride), H / lr_spatial_scale, W / lr_spatial_scale]
    """
    pixels = pixels[:, :, :: task.lr_frame_stride]
    if task.lr_spatial_scale == 1:
        return pixels
    frames = pixels.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [T, 3, H, W]
    frames = functional.interpolate(
        frames, scale_factor=1 / task.lr_spatial_scale, mode="bilinear", align_corners=False
    )
    return frames.unsqueeze(0).permute(0, 2, 1, 3, 4)


def build_fid_evaluator(
    dataset: WanBenchmarkDataset, statistics_directory: Path, image_size: int
) -> FIDEvaluator:
    """
    `fid` compares whole frames resized to 299 by Inception (global structure and color).
    `fid_patch` compares native resolution FID_PATCH_SIZE tiles (generated high frequency detail).
    Statistics are cached per ground truth fingerprint.
    """
    fingerprint = dataset.get_fingerprint()
    common_arguments = {
        "image_dir": str(dataset.get_ground_truth_directory()),
        "file_names": dataset.get_ground_truth_file_names(),
    }
    return FIDEvaluator(
        references=[
            FIDReference(
                name="fid",
                stats_path=str(statistics_directory / f"fid_stats_{fingerprint}.pt"),
                **common_arguments,
            ),
            FIDReference(
                name="fid_patch",
                stats_path=str(statistics_directory / f"fid_patch_stats_{fingerprint}.pt"),
                patch_size=FID_PATCH_SIZE,
                **common_arguments,
            ),
        ],
        image_size=image_size,
        batch_size=16,
    )


def _save_samples(
    task: QualityTask,
    sample_name: str,
    ground_truth_pixels: torch.Tensor,
    predicted_pixels: torch.Tensor,
    samples_directory: Path,
) -> None:
    """
    Images: `<name>_in.png` (pixel upscaled ground truth input) and `<name>_out.png`.
    Videos: `<name>_in.mp4` (ground truth input frames repeated over time) and `<name>_out.mp4`.

    args:
        ground_truth_pixels, predicted_pixels: [1, 3, T, H, W], pixel_values: [0...1]
    """
    input_pixels = project_to_input_pixels(ground_truth_pixels, task)
    if task.kind == "image":
        upscaled = functional.interpolate(
            input_pixels[:, :, 0], scale_factor=task.lr_spatial_scale, mode="bilinear"
        )
        cv2.imwrite(str(samples_directory / f"{sample_name}_in.png"), torch_to_cv2(upscaled))
        cv2.imwrite(
            str(samples_directory / f"{sample_name}_out.png"),
            torch_to_cv2(predicted_pixels[:, :, 0]),
        )
        return

    repeated = input_pixels.repeat_interleave(task.lr_frame_stride, dim=2)[
        :, :, : predicted_pixels.shape[2]
    ]
    for suffix, pixels in (("in", repeated), ("out", predicted_pixels)):
        height, width = pixels.shape[-2:]
        writer = cv2.VideoWriter(
            str(samples_directory / f"{sample_name}_{suffix}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            VIDEO_SAVE_FPS,
            (width, height),
        )
        for frame_index in range(pixels.shape[2]):
            writer.write(torch_to_cv2(pixels[:, :, frame_index]))
        writer.release()


@torch.no_grad()
def evaluate_quality(
    upscaler: Upscaler,
    vae: VAE,
    dataset: WanBenchmarkDataset,
    task: QualityTask,
    fid_evaluator: FIDEvaluator,
    upscaler_dtype: torch.dtype,
    samples_directory: Path,
    num_saved_samples: int,
) -> dict[str, float]:
    """
    Upscales the input latents of every sample, decodes, and compares with the ground truth pixels.
    `mse_hr_*` is against the ground truth, `mse_lr_*` against the input after projecting the prediction
    onto the input frames and resolution, `mse_hr_latents` is in latent space.
    """
    samples_directory.mkdir(parents=True, exist_ok=True)
    saved_indices = set(
        np.linspace(0, len(dataset) - 1, min(num_saved_samples, len(dataset)))
        .round()
        .astype(int)
        .tolist()
    )
    metric_values: dict[str, list[float]] = {
        f"mse_hr_{dataset.kind}": [],
        f"mse_lr_{dataset.kind}": [],
        "mse_hr_latents": [],
    }

    for index, sample_name in enumerate(tqdm(dataset.sample_names, desc="Quality")):
        block = dataset.load_block(sample_name)
        latents_hr_gt = block[task.target_key]  # [1, C, T_hr, H_hr, W_hr]
        ground_truth_pixels = dataset.load_ground_truth_pixels(sample_name)  # [1, 3, T, H, W]

        latents_hr_pred = upscaler.upscale(
            block[task.source_key].to(upscaler_dtype),
            block[task.hff_key],
            tuple(latents_hr_gt.shape[2:]),
        ).float()
        predicted_pixels = vae.decode(latents_hr_pred).float().clamp(0, 1)  # [1, 3, T, H, W]

        metric_values[f"mse_hr_{dataset.kind}"].append(
            functional.mse_loss(predicted_pixels, ground_truth_pixels).item()
        )
        metric_values[f"mse_lr_{dataset.kind}"].append(
            functional.mse_loss(
                project_to_input_pixels(predicted_pixels, task),
                project_to_input_pixels(ground_truth_pixels, task),
            ).item()
        )
        metric_values["mse_hr_latents"].append(
            functional.mse_loss(latents_hr_pred, latents_hr_gt).item()
        )
        fid_evaluator.update(predicted_pixels.squeeze(0).permute(1, 0, 2, 3))  # [T, 3, H, W]

        if index in saved_indices:
            _save_samples(
                task, sample_name, ground_truth_pixels, predicted_pixels, samples_directory
            )

    quality = {name: float(np.mean(values)) for name, values in metric_values.items()}
    quality.update(fid_evaluator.compute())
    return quality
