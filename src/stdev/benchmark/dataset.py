from pathlib import Path

import cv2
import torch

from stdev.utils.image_tools import cv2_to_torch


class BenchmarkDataset:
    """
    Directory with `images/<name>.png` and `prompts/<name>.txt` files.
    """

    def __init__(self, dataset_directory: Path, max_samples: int | None = None) -> None:
        self.dataset_directory = dataset_directory
        self.images_directory = dataset_directory / "images"
        self.prompts_directory = dataset_directory / "prompts"
        self.sample_names: list[str] = sorted(
            image_path.stem for image_path in self.images_directory.glob("*.png")
        )
        if max_samples is not None:
            self.sample_names = self.sample_names[:max_samples]

    def __len__(self) -> int:
        return len(self.sample_names)

    def get_image_file_names(self) -> list[str]:
        return [f"{sample_name}.png" for sample_name in self.sample_names]

    def load_image(self, sample_name: str, device: str = "cuda") -> torch.Tensor:
        """
        returns:
            image: [1, 3, H, W], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32
        """
        image = cv2.imread(str(self.images_directory / f"{sample_name}.png"))
        return cv2_to_torch(image, device=device)

    def load_prompt(self, sample_name: str) -> str:
        return (self.prompts_directory / f"{sample_name}.txt").read_text().strip()
