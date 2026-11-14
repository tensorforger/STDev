import hashlib
from pathlib import Path

import cv2
import torch
import vaezip

from stdev.utils.image_tools import cv2_to_torch

VIDEO_NUM_FRAMES = 33


class WanBenchmarkDataset:
    """
    Directory written by `scripts/encode_wan_dataset.py --benchmark`: `blocks/<name>.vz.safetensors` with
    the encoded latents and high frequency features, plus the ground truth pixels as
    `images/<name>.png` (kind "image") or `frames/<name>/frame_<k>.png` (kind "video").
    """

    def __init__(self, dataset_directory: Path, kind: str, max_samples: int | None = None) -> None:
        if kind not in ("image", "video"):
            raise ValueError(f"kind must be 'image' or 'video', got {kind!r}")
        self.dataset_directory = dataset_directory
        self.kind = kind
        self.blocks_directory = dataset_directory / "blocks"
        self.images_directory = dataset_directory / "images"
        self.frames_directory = dataset_directory / "frames"
        self.sample_names: list[str] = sorted(
            block_path.name.removesuffix(".vz.safetensors")
            for block_path in self.blocks_directory.glob("*.vz.safetensors")
        )
        if max_samples is not None:
            self.sample_names = self.sample_names[:max_samples]

    def __len__(self) -> int:
        return len(self.sample_names)

    def get_ground_truth_directory(self) -> Path:
        return self.images_directory if self.kind == "image" else self.frames_directory

    def get_ground_truth_file_names(self) -> list[str]:
        """
        returns:
            ground truth pixel file names relative to `get_ground_truth_directory()`
        """
        if self.kind == "image":
            return [f"{sample_name}.png" for sample_name in self.sample_names]
        return [
            f"{sample_name}/frame_{frame_index:02d}.png"
            for sample_name in self.sample_names
            for frame_index in range(VIDEO_NUM_FRAMES)
        ]

    def get_fingerprint(self) -> str:
        """
        Identifies the ground truth files, so that cached FID statistics are not reused after the
        dataset changes.
        """
        directory = self.get_ground_truth_directory()
        digest = hashlib.md5()
        for file_name in self.get_ground_truth_file_names():
            digest.update(f"{file_name}:{(directory / file_name).stat().st_size};".encode())
        return digest.hexdigest()[:16]

    def load_block(self, sample_name: str, device: str = "cuda") -> dict[str, torch.Tensor]:
        """
        returns:
            tensors: latents_* [1, C, T, H, W] and hff_* [1], dtype: torch.float32
        """
        encoded_tensors = vaezip.load(self.blocks_directory / f"{sample_name}.vz.safetensors")
        return {
            name: encoded_tensor.decode(dtype=torch.float32, device=device)
            for name, encoded_tensor in encoded_tensors.items()
        }

    def load_ground_truth_pixels(self, sample_name: str, device: str = "cuda") -> torch.Tensor:
        """
        returns:
            pixels: [1, 3, T, H, W], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32,
            T = 1 for images
        """
        directory = self.get_ground_truth_directory()
        if self.kind == "image":
            paths = [directory / f"{sample_name}.png"]
        else:
            paths = [
                directory / sample_name / f"frame_{frame_index:02d}.png"
                for frame_index in range(VIDEO_NUM_FRAMES)
            ]
        frames = [
            cv2_to_torch(cv2.imread(str(path)), device=device) for path in paths
        ]  # T x [1, 3, H, W]
        return torch.stack(frames, dim=2)
