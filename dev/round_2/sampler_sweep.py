import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from tqdm import tqdm

from stdev.benchmark.cache import LatentCache
from stdev.benchmark.dataset import BenchmarkDataset
from stdev.benchmark.quality import build_fid_evaluator
from stdev.upscalers.loading import load_upscaler
from stdev.vaes.flux_2_vae import Flux2VAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "axis_1024"
CACHE_DIRECTORY = REPOSITORY_ROOT / "benchmark_cache" / "axis_1024"
VAE_DIRECTORY = REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B" / "vae"
RUN_FILE = REPOSITORY_ROOT / "src" / "stdev" / "upscalers" / "run_0003.py"
RESULTS_FILE = REPOSITORY_ROOT / "dev" / "round_2" / "sampler_run_0003_results.json"

INFERENCE_DTYPE = torch.bfloat16
INFERENCE_SEED = 0
SCALE_FACTOR = 2
VARIANTS = (
    ("full_8", 1.0, 8),
    ("full_6", 1.0, 6),
    ("full_4", 1.0, 4),
    ("full_3", 1.0, 3),
    ("full_2", 1.0, 2),
)


@torch.no_grad()
def sample_latents(
    model: torch.nn.Module,
    latents_low_resolution: torch.Tensor,
    start_time: float,
    num_steps: int,
) -> torch.Tensor:
    """
    args:
        model: conditional flow model accepting noisy latents, condition, and time
        latents_low_resolution: [B, 32, H, W], dtype: torch.float32, device: cuda
        start_time: initial flow time in (0...1]
        num_steps: positive number of Euler steps
    returns:
        latents_high_resolution: [B, 32, 2*H, 2*W], dtype: torch.float32, device: cuda
    """
    condition = functional.interpolate(
        latents_low_resolution.to(INFERENCE_DTYPE),
        scale_factor=SCALE_FACTOR,
        mode="bilinear",
        align_corners=False,
    )
    generator = torch.Generator(device=latents_low_resolution.device).manual_seed(INFERENCE_SEED)
    noise = torch.randn(
        condition.shape,
        generator=generator,
        device=latents_low_resolution.device,
        dtype=torch.float32,
    )
    latents = (1.0 - start_time) * condition.float() + start_time * noise
    sampling_times = torch.linspace(
        start_time, 0.0, num_steps + 1, device=latents_low_resolution.device
    )
    for current_time, next_time in zip(sampling_times[:-1], sampling_times[1:]):
        velocity = model(
            latents.to(INFERENCE_DTYPE),
            condition,
            current_time.expand(latents.shape[0]),
        )
        latents = latents + (next_time - current_time) * velocity.float()
    return latents


@torch.no_grad()
def evaluate_variant(
    model: torch.nn.Module,
    vae: Flux2VAE,
    dataset: BenchmarkDataset,
    cache: LatentCache,
    start_time: float,
    num_steps: int,
) -> dict[str, float]:
    """Evaluate latent reconstruction and image FID for one sampler configuration."""
    fid_evaluator = build_fid_evaluator(dataset, cache, image_size=1024)
    latent_squared_error_sum = 0.0
    image_squared_error_sum = 0.0
    for sample_name in tqdm(dataset.sample_names, desc=f"t={start_time}, steps={num_steps}"):
        latents_low_resolution = cache.load_latents("lr_gt", sample_name)
        latents_high_resolution_target = cache.load_latents("hr_gt", sample_name)
        latents_high_resolution = sample_latents(
            model, latents_low_resolution, start_time, num_steps
        )
        latent_squared_error_sum += functional.mse_loss(
            latents_high_resolution, latents_high_resolution_target
        ).item()
        image_prediction = vae.decode(latents_high_resolution).float().clamp(0, 1)
        image_target = dataset.load_image(sample_name)
        image_squared_error_sum += functional.mse_loss(image_prediction, image_target).item()
        fid_evaluator.update(image_prediction)
    sample_count = len(dataset)
    return {
        "fid": fid_evaluator.compute(),
        "mse_hr_latents": latent_squared_error_sum / sample_count,
        "mse_hr_image": image_squared_error_sum / sample_count,
    }


def main() -> None:
    torch.manual_seed(0)
    dataset = BenchmarkDataset(DATASET_DIRECTORY)
    cache = LatentCache(
        CACHE_DIRECTORY,
        metadata={
            "vae": "flux_2",
            "vae_dtype": "torch.bfloat16",
            "scale_factor": SCALE_FACTOR,
            "codec": "lbrq_8bit",
            "seed": 0,
            "image_size": 1024,
        },
        codec="lbrq_8bit",
    )
    upscaler = load_upscaler(RUN_FILE)
    upscaler.prepare_to_benchmark()
    upscaler.compile()
    assert upscaler.model is not None
    vae = Flux2VAE(str(VAE_DIRECTORY), device="cuda", dtype=INFERENCE_DTYPE)

    results: dict[str, dict[str, float]] = {}
    for variant_name, start_time, num_steps in VARIANTS:
        variant_start_time = time.perf_counter()
        variant_metrics = evaluate_variant(
            upscaler.model, vae, dataset, cache, start_time, num_steps
        )
        variant_metrics["evaluation_seconds"] = time.perf_counter() - variant_start_time
        results[variant_name] = variant_metrics
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        print(variant_name, json.dumps(variant_metrics), flush=True)


if __name__ == "__main__":
    main()
