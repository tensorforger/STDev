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
    name: str
    image_dir: str
    stats_path: str
    file_names: list[str] | None = None


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
        self.metric = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)
        self.real_statistics: dict[str, dict[str, torch.Tensor]] = {}

        for reference in references:
            for name in REAL_STATE_NAMES:
                getattr(self.metric, name).zero_()
            if os.path.exists(reference.stats_path):
                self._load_real_statistics(reference.stats_path)
            else:
                self._compute_real_statistics(
                    RealImageDataset(reference.image_dir, image_size, reference.file_names),
                    batch_size,
                    num_workers,
                )
                self._save_real_statistics(reference.stats_path)
            self.real_statistics[reference.name] = {
                name: getattr(self.metric, name).clone() for name in REAL_STATE_NAMES
            }

    def _compute_real_statistics(self, dataset, batch_size, num_workers):
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        for images in tqdm(loader, desc="FID real statistics"):
            self.metric.update(images.to(self.device, non_blocking=True), real=True)

    def _save_real_statistics(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {name: getattr(self.metric, name).cpu() for name in REAL_STATE_NAMES}
        torch.save(state, path)

    def _load_real_statistics(self, path: str):
        state = torch.load(path, map_location=self.device)
        for name in REAL_STATE_NAMES:
            setattr(self.metric, name, state[name])

    def update(self, images: torch.Tensor):
        """
        args:
            images: [B, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        images = images.clamp(0, 1).mul(255).round().to(torch.uint8)
        self.metric.update(images, real=False)

    def compute(self) -> dict[str, float]:
        """Returns FID per reference name of all images passed to update() and clears them."""
        fids: dict[str, float] = {}
        for reference_name, state in self.real_statistics.items():
            for name in REAL_STATE_NAMES:
                setattr(self.metric, name, state[name].clone())
            fids[reference_name] = self.metric.compute().item()
        self.metric.reset()
        return fids
