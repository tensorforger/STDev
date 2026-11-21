import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from stdev.hff.immerkaer import estimate_immerkaer
from stdev.utils.image_tools import crop_maximal_rectangle, cv2_to_torch, resize_cv2
from stdev.vaes.wan_2_1_vae import Wan21VAE

HIGH_RESOLUTION_SIZE = 1024
LOW_RESOLUTION_SIZE = 512
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
MANIFEST_NAME = "manifest.json"
LATENTS_NAME = "latents.pt"
HFF_NAME = "hff.pt"
ENCODE_BATCH_SIZE = 16
NUM_PREPROCESS_THREADS = 8


def list_source_images(source_directory: Path) -> list[Path]:
    """
    returns:
        image paths of the source dataset sorted by relative path, repository metadata excluded
    """
    paths = [
        path
        for path in source_directory.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
        and ".git" not in path.relative_to(source_directory).parts
    ]
    return sorted(paths, key=lambda path: path.relative_to(source_directory).as_posix())


def load_image_batch(
    directory: Path, sample_names: list[str], device: str = "cuda"
) -> torch.Tensor:
    """
    args:
        sample_names: file names without the .png extension
    returns:
        pixels: [B, 3, H, W], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32
    """
    return torch.cat(
        [
            cv2_to_torch(cv2.imread(str(directory / f"{name}.png")), device=device)
            for name in sample_names
        ]
    )


class BenchmarkDataset:
    """
    Directory written by `build_dataset`: `ground_truth/<name>.png` (1024), `low_resolution/<name>.png` (512, the
    bilinear downsampled ground truth) and `latents.pt` (Wan 2.1 VAE latents of the low resolution images).
    `hff.pt` (Immerkaer high frequency feature of the ground truth images) is computed on first use.
    """

    def __init__(self, dataset_directory: Path, max_samples: int | None = None) -> None:
        manifest = json.loads((dataset_directory / MANIFEST_NAME).read_text())
        self.dataset_directory = dataset_directory
        self.ground_truth_directory = dataset_directory / "ground_truth"
        self.low_resolution_directory = dataset_directory / "low_resolution"
        self.sample_names: list[str] = manifest["sample_names"][:max_samples]
        self.latents = torch.load(dataset_directory / LATENTS_NAME)[: len(self.sample_names)]
        self.hff = self._load_or_compute_hff(manifest["sample_names"])[: len(self.sample_names)]

    def _load_or_compute_hff(self, all_sample_names: list[str]) -> torch.Tensor:
        hff_path = self.dataset_directory / HFF_NAME
        if not hff_path.exists():
            hff_values = [
                estimate_immerkaer(load_image_batch(self.ground_truth_directory, [name]))
                for name in tqdm(all_sample_names, desc="Ground truth hff")
            ]
            torch.save(torch.cat(hff_values).cpu(), hff_path)
        return torch.load(hff_path)

    def __len__(self) -> int:
        return len(self.sample_names)

    def load_hff(self, index: int) -> torch.Tensor:
        """
        returns:
            hff: [1], values: [0...1], Immerkaer feature of the ground truth image, dtype: torch.float32, on the CPU
        """
        return self.hff[index : index + 1]

    def load_latents(self, index: int) -> torch.Tensor:
        """
        returns:
            latents: [1, 16, 1, 64, 64], normalized Wan 2.1 VAE latents, dtype: torch.float32, on the CPU
        """
        return self.latents[index : index + 1]


def _prepare_image(source_path: Path, ground_truth_path: Path, low_resolution_path: Path) -> None:
    source_image = cv2.imread(str(source_path))
    ground_truth = crop_maximal_rectangle(source_image, HIGH_RESOLUTION_SIZE, HIGH_RESOLUTION_SIZE)
    low_resolution = resize_cv2(ground_truth, LOW_RESOLUTION_SIZE, LOW_RESOLUTION_SIZE, "bilinear")
    cv2.imwrite(str(ground_truth_path), ground_truth)
    cv2.imwrite(str(low_resolution_path), low_resolution)


def build_dataset(source_directory: Path, dataset_directory: Path, vae_path: Path) -> None:
    """
    Center crops every source image to 1024x1024 (`crop_maximal_rectangle`), downsamples it bilinearly to 512x512
    and encodes it with the full Wan 2.1 VAE. Does nothing if the dataset already exists.
    """
    if (dataset_directory / MANIFEST_NAME).exists():
        return
    source_paths = list_source_images(source_directory)
    if not source_paths:
        raise FileNotFoundError(f"No images found in {source_directory}")
    sample_names = [f"{index:04d}" for index in range(len(source_paths))]
    ground_truth_directory = dataset_directory / "ground_truth"
    low_resolution_directory = dataset_directory / "low_resolution"
    ground_truth_directory.mkdir(parents=True, exist_ok=True)
    low_resolution_directory.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(NUM_PREPROCESS_THREADS) as executor:
        futures = [
            executor.submit(
                _prepare_image,
                source_path,
                ground_truth_directory / f"{name}.png",
                low_resolution_directory / f"{name}.png",
            )
            for source_path, name in zip(source_paths, sample_names)
        ]
        for future in tqdm(futures, desc="Crop and downsample"):
            future.result()

    vae = Wan21VAE(str(vae_path), device="cuda", dtype=torch.bfloat16, as_image=True)
    latents = []
    for start in tqdm(range(0, len(sample_names), ENCODE_BATCH_SIZE), desc="Encode"):
        pixels = load_image_batch(
            low_resolution_directory, sample_names[start : start + ENCODE_BATCH_SIZE]
        )
        latents.append(vae.encode(pixels).cpu())  # [B, 16, 1, 64, 64]
    torch.save(torch.cat(latents), dataset_directory / LATENTS_NAME)

    manifest = {
        "sample_names": sample_names,
        "source_paths": [path.relative_to(source_directory).as_posix() for path in source_paths],
    }
    (dataset_directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
