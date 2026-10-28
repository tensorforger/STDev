import os
from dataclasses import dataclass

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from stdev.utils.image_tools import crop_maximal_rectangle, cv2_to_np

REAL_STATE_NAMES = (
    "real_features_sum",
    "real_features_cov_sum",
    "real_features_num_samples",
)


def split_into_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Non-overlapping tiles at native resolution; the remainder at the right and bottom is dropped.

    args:
        images: [B, 3, H, W]
    returns:
        patches: [B * (H // patch_size) * (W // patch_size), 3, patch_size, patch_size]
    """
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, images.shape[1], patch_size, patch_size)


class RealImageDataset(Dataset):
    """Raw images with the same preprocessing as in scripts/encode_dataset.py."""

    def __init__(self, image_dir: str, image_size: int, file_names: list[str] | None = None):
        self.image_dir = image_dir
        self.image_size = image_size
        self.files = sorted(os.listdir(image_dir)) if file_names is None else file_names

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        returns:
            image: [3, H, W], dtype = uint8, channel_order = RGB
        """
        image = cv2.imread(os.path.join(self.image_dir, self.files[idx]))
        image = crop_maximal_rectangle(
            image, target_height=self.image_size, target_width=self.image_size
        )
        image = cv2_to_np(image)
        return torch.from_numpy(image).permute(2, 0, 1)


@dataclass
class FIDReference:
    """
    patch_size: None compares whole images (resized to 299 by Inception), otherwise
    non-overlapping native resolution tiles of this size.
    """

    name: str
    image_dir: str
    stats_path: str
    file_names: list[str] | None = None
    patch_size: int | None = None


class FIDEvaluator:
    """
    FID against the statistics of several reference image sets.
    Reference statistics are computed once and cached at stats_path, so all runs share them.
    """

    def __init__(
        self,
        references: list[FIDReference],
        image_size: int = 256,
        device: str = "cuda",
        batch_size: int = 256,
        num_workers: int = 8,
    ):
        self.device = device
        self.metrics: dict[int | None, FrechetInceptionDistance] = {}
        self.reference_patch_sizes: dict[str, int | None] = {}
        self.real_statistics: dict[str, dict[str, torch.Tensor]] = {}

        for reference in references:
            if reference.patch_size not in self.metrics:
                self.metrics[reference.patch_size] = FrechetInceptionDistance(
                    feature=2048, reset_real_features=False
                ).to(device)
            metric = self.metrics[reference.patch_size]
            for name in REAL_STATE_NAMES:
                getattr(metric, name).zero_()
            if os.path.exists(reference.stats_path):
                self._load_real_statistics(metric, reference.stats_path)
            else:
                self._compute_real_statistics(
                    metric,
                    RealImageDataset(reference.image_dir, image_size, reference.file_names),
                    reference.patch_size,
                    batch_size,
                    num_workers,
                )
                self._save_real_statistics(metric, reference.stats_path)
            self.reference_patch_sizes[reference.name] = reference.patch_size
            self.real_statistics[reference.name] = {
                name: getattr(metric, name).clone() for name in REAL_STATE_NAMES
            }

    def _compute_real_statistics(self, metric, dataset, patch_size, batch_size, num_workers):
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        for images in tqdm(loader, desc="FID real statistics"):
            images = images.to(self.device, non_blocking=True)
            if patch_size is not None:
                images = split_into_patches(images, patch_size)
            metric.update(images, real=True)

    def _save_real_statistics(self, metric, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {name: getattr(metric, name).cpu() for name in REAL_STATE_NAMES}
        torch.save(state, path)

    def _load_real_statistics(self, metric, path: str):
        state = torch.load(path, map_location=self.device)
        for name in REAL_STATE_NAMES:
            setattr(metric, name, state[name])

    def update(self, images: torch.Tensor):
        """
        args:
            images: [B, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        images = images.clamp(0, 1).mul(255).round().to(torch.uint8)
        for patch_size, metric in self.metrics.items():
            if patch_size is None:
                metric.update(images, real=False)
            else:
                metric.update(split_into_patches(images, patch_size), real=False)

    def compute(self) -> dict[str, float]:
        """Returns FID per reference name of all images passed to update() and clears them."""
        fids: dict[str, float] = {}
        for reference_name, state in self.real_statistics.items():
            metric = self.metrics[self.reference_patch_sizes[reference_name]]
            for name in REAL_STATE_NAMES:
                setattr(metric, name, state[name].clone())
            metric._computed = None
            fids[reference_name] = metric.compute().item()
        for metric in self.metrics.values():
            metric.reset()
        return fids
