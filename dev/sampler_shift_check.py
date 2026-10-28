import json
from pathlib import Path

import torch

from stdev.benchmark.cache import LatentCache
from stdev.benchmark.dataset import BenchmarkDataset
from stdev.benchmark.quality import (
    build_fid_evaluator,
    evaluate_quality,
    populate_latent_cache,
)
from stdev.upscalers import run_0004
from stdev.upscalers.base import Upscaler
from stdev.vaes.flux_2_vae import Flux2VAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = Path(__file__).resolve().parent / "sampler_shift_results.json"
BENCHMARK_CACHE_DIRECTORY = REPOSITORY_ROOT / "benchmark_cache"
VAE_PATH = REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B" / "vae"
DATASET_PATHS = (
    REPOSITORY_ROOT / "benchmark_datasets" / "axis_1024",
    REPOSITORY_ROOT / "benchmark_datasets" / "flux_1024",
)
SAMPLING_SHIFTS = (1.0, 2.0, 4.0)


def evaluate_dataset(
    upscaler: Upscaler, vae: Flux2VAE, dataset_path: Path, sampling_shift: float
) -> dict[str, float]:
    """
    Evaluate one sampler shift on a benchmark dataset without changing model weights.
    """
    dataset = BenchmarkDataset(dataset_path)
    image_size = dataset.load_image(dataset.sample_names[0]).shape[-1]
    cache = LatentCache(
        BENCHMARK_CACHE_DIRECTORY / dataset_path.name,
        metadata={
            "vae": vae.get_vae_name(),
            "vae_dtype": str(torch.bfloat16),
            "scale_factor": 2,
            "codec": "lbrq_8bit",
            "seed": 0,
            "image_size": image_size,
        },
        codec="lbrq_8bit",
    )
    populate_latent_cache(dataset, vae, cache, 2, 0)
    fid_evaluator = build_fid_evaluator(dataset, cache, image_size)
    return evaluate_quality(
        upscaler=upscaler,
        vae=vae,
        dataset=dataset,
        cache=cache,
        fid_evaluator=fid_evaluator,
        scale_factor=2,
        upscaler_dtype=torch.bfloat16,
        images_directory=(
            RESULTS_PATH.parent / "sampler_shift_images" / str(sampling_shift) / dataset_path.name
        ),
        num_saved_samples=0,
    )


def main() -> None:
    upscaler = run_0004.Run0004Upscaler()
    upscaler.prepare_to_benchmark()
    vae = Flux2VAE(str(VAE_PATH), device="cuda", dtype=run_0004.INFERENCE_DTYPE)
    all_results: dict[str, dict[str, dict[str, float]]] = {}
    for sampling_shift in SAMPLING_SHIFTS:
        run_0004.SIGMA_SHIFT = sampling_shift
        shift_results: dict[str, dict[str, float]] = {}
        for dataset_path in DATASET_PATHS:
            shift_results[dataset_path.name] = evaluate_dataset(
                upscaler, vae, dataset_path, sampling_shift
            )
        all_results[str(sampling_shift)] = shift_results
        print(f"shift {sampling_shift}: {json.dumps(shift_results)}", flush=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
