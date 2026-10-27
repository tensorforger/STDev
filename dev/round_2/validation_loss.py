import json
from pathlib import Path

import torch
import torch.nn.functional as functional
import vaezip
from torch.utils.data import DataLoader, Dataset

from stdev.upscalers.loading import load_upscaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIRECTORY = REPOSITORY_ROOT / "benchmark_cache" / "axis_1024"
UPSCALERS_DIRECTORY = REPOSITORY_ROOT / "src" / "stdev" / "upscalers"
RESULTS_FILE = REPOSITORY_ROOT / "dev" / "round_2" / "benchmark_flow_loss.json"
RUN_IDS = ("0000", "0002", "0003")
BATCH_SIZE = 16
NUM_WORKERS = 8
EVALUATION_SEED = 123


class BenchmarkLatentPairDataset(Dataset):
    """Cached high- and low-resolution latent pairs from the held-out benchmark."""

    def __init__(self, cache_directory: Path) -> None:
        self.high_resolution_paths = sorted(
            (cache_directory / "latents_hr_gt").glob("*.vz.safetensors")
        )
        self.low_resolution_directory = cache_directory / "latents_lr_gt"

    def __len__(self) -> int:
        return len(self.high_resolution_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
            latents_full: [32, 128, 128], normalized FLUX latents, dtype: torch.float32
            latents_low_resolution: [32, 64, 64], normalized FLUX latents, dtype: torch.float32
        """
        high_resolution_path = self.high_resolution_paths[index]
        low_resolution_path = self.low_resolution_directory / high_resolution_path.name
        latents_full = vaezip.load_tensor(high_resolution_path, device="cpu")
        latents_low_resolution = vaezip.load_tensor(low_resolution_path, device="cpu")
        return latents_full[0], latents_low_resolution[0]


@torch.no_grad()
def evaluate_flow_loss(
    model: torch.nn.Module,
    data_loader: DataLoader,
) -> float:
    """Evaluate logit-normal conditional flow loss with a fixed random sequence."""
    generator = torch.Generator(device="cuda").manual_seed(EVALUATION_SEED)
    weighted_loss_sum = 0.0
    sample_count = 0
    for latents_full, latents_low_resolution in data_loader:
        latents_full = latents_full.to("cuda", non_blocking=True)
        latents_low_resolution = latents_low_resolution.to("cuda", non_blocking=True)
        condition = functional.interpolate(
            latents_low_resolution.to(torch.bfloat16),
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        time_values = torch.sigmoid(
            torch.randn(
                latents_full.shape[0],
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            )
        )
        noise = torch.randn(
            latents_full.shape,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        blend = time_values[:, None, None, None]
        noisy_latents = (1.0 - blend) * latents_full + blend * noise
        velocity = model(
            noisy_latents.to(torch.bfloat16),
            condition,
            time_values,
        )
        loss = functional.mse_loss(velocity.float(), noise - latents_full)
        batch_size = latents_full.shape[0]
        weighted_loss_sum += loss.item() * batch_size
        sample_count += batch_size
    return weighted_loss_sum / sample_count


def main() -> None:
    data_loader = DataLoader(
        BenchmarkLatentPairDataset(CACHE_DIRECTORY),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    results: dict[str, float] = {}
    for run_id in RUN_IDS:
        upscaler = load_upscaler(UPSCALERS_DIRECTORY / f"run_{run_id}.py")
        upscaler.prepare_to_benchmark()
        assert upscaler.model is not None
        loss = evaluate_flow_loss(upscaler.model, data_loader)
        results[run_id] = loss
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        print(run_id, loss, flush=True)
        del upscaler
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
