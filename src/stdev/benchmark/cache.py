import json
from pathlib import Path
from typing import Any

import torch
import vaezip

LATENT_KINDS = ("lr_gt", "hr_gt")


class LatentCache:
    """
    Per-sample ground truth latents compressed with vaezip and FID statistics of the real and
    teacher images.
    """

    def __init__(self, cache_directory: Path, metadata: dict[str, Any], codec: str) -> None:
        self.cache_directory = cache_directory
        self.codec = codec
        self.fid_statistics_paths = {
            "real": cache_directory / "fid_real_stats.pt",
            "teacher": cache_directory / "fid_teacher_stats.pt",
        }
        cache_directory.mkdir(parents=True, exist_ok=True)
        for kind in LATENT_KINDS:
            (cache_directory / f"latents_{kind}").mkdir(exist_ok=True)

        metadata_path = cache_directory / "meta.json"
        if metadata_path.exists():
            existing_metadata = json.loads(metadata_path.read_text())
            if existing_metadata != metadata:
                raise RuntimeError(
                    f"Cache metadata mismatch in {cache_directory}: "
                    f"{existing_metadata} != {metadata}. Delete the cache directory."
                )
        else:
            metadata_path.write_text(json.dumps(metadata, indent=2))

    def _get_path(self, kind: str, sample_name: str) -> Path:
        return self.cache_directory / f"latents_{kind}" / f"{sample_name}.vz.safetensors"

    def has_latents(self, sample_name: str) -> bool:
        return all(self._get_path(kind, sample_name).exists() for kind in LATENT_KINDS)

    def save_latents(self, kind: str, sample_name: str, latents: torch.Tensor) -> None:
        """
        args:
            latents: [1, C, H, W], dtype: any floating point
        """
        vaezip.save_tensor(latents.float(), self._get_path(kind, sample_name), codec=self.codec)

    def load_latents(self, kind: str, sample_name: str, device: str = "cuda") -> torch.Tensor:
        """
        returns:
            latents: [1, C, H, W], dtype: torch.float32
        """
        return vaezip.load_tensor(self._get_path(kind, sample_name), device=device)
