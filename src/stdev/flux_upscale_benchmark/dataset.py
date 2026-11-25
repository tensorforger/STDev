import json
from pathlib import Path

import torch


class FluxUpscaleBenchmarkDataset:
    """
    The autoresearch_3 FLUX.2 benchmark set (`benchmark_datasets/axis_v1_1k_flux2_2x`, built by
    `build_benchmark` of `scripts/encode_flux2_datasets.py`): `ground_truth/<name>.png` (1024, antialiased center crop),
    `low_resolution/<name>.png` (512, cv2 bilinear downscale of the ground truth) and `latents.pt` (normalized FLUX.2
    latents of the low resolution images). Ground truth latents and hff exist in the directory but are never used.
    """

    def __init__(self, dataset_directory: Path, max_samples: int | None = None) -> None:
        manifest = json.loads((dataset_directory / "manifest.json").read_text())
        self.dataset_directory = dataset_directory
        self.ground_truth_directory = dataset_directory / "ground_truth"
        self.low_resolution_directory = dataset_directory / "low_resolution"
        self.sample_names: list[str] = manifest["sample_names"][:max_samples]
        self.latents = torch.load(dataset_directory / "latents.pt")[: len(self.sample_names)]

    def __len__(self) -> int:
        return len(self.sample_names)

    def load_latents(self, index: int) -> torch.Tensor:
        """
        returns:
            latents: [1, 32, 64, 64], normalized FLUX.2 VAE latents, dtype: torch.float32, on the CPU
        """
        return self.latents[index : index + 1].float()

    def load_hff(self, index: int) -> torch.Tensor:
        raise RuntimeError("The FLUX.2 upscale benchmark gives no ground truth information to solutions")
