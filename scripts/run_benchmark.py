import argparse
import json
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

import torch
from tqdm import tqdm

from stdev.benchmark.cache import LatentCache
from stdev.benchmark.dataset import BenchmarkDataset
from stdev.benchmark.efficiency import measure_latency, measure_memory
from stdev.benchmark.quality import build_fid_evaluator, evaluate_quality, populate_latent_cache
from stdev.upscalers.base import Upscaler
from stdev.upscalers.loading import load_upscaler
from stdev.vaes.flux_2_vae import Flux2VAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CACHE_DIRECTORY = REPOSITORY_ROOT / "benchmark_cache"
RAW_RESULTS_DIRECTORY = REPOSITORY_ROOT / "raw_results"
VAE_PATH = REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B" / "vae"

VAE_DTYPE = torch.bfloat16
UPSCALER_DTYPE = torch.bfloat16
SCALE_FACTOR = 2
RESOLUTIONS = [512, 1024, 2048]
MEMORY_WARMUP_ITERATIONS = 3
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50
SMOKE_ITERATIONS = 2
SMOKE_NUM_SAMPLES = 2
NUM_SAVED_SAMPLES = 32
LATENT_CODEC = "lbrq_8bit"
PROMPT_EMBEDS_SHAPE = (1, 77, 768)
SEED = 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        required=True,
        help="Benchmark dataset directories, e.g. real images and teacher generations.",
    )
    parser.add_argument("--upscaler", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    with ExitStack() as exit_stack:
        run_benchmark(exit_stack)


def measure_efficiency(
    upscaler: Upscaler, vae: Flux2VAE, results_directory: Path, smoke: bool
) -> dict[str, float]:
    """
    Memory is profiled on the eager model, then latency is measured on the compiled model.
    """
    common_arguments = {
        "upscaler": upscaler,
        "latent_channels": vae.get_latent_channels(),
        "scale_factor": SCALE_FACTOR,
        "dtype": UPSCALER_DTYPE,
        "prompt_embeds": torch.zeros(PROMPT_EMBEDS_SHAPE, device="cuda"),
        "seed": SEED,
    }
    efficiency: dict[str, float] = {}
    for resolution in tqdm(RESOLUTIONS, desc="Memory"):
        efficiency.update(
            measure_memory(
                resolution=resolution,
                latent_size=resolution // vae.get_spatial_compression(),
                warmup_iterations=0 if smoke else MEMORY_WARMUP_ITERATIONS,
                profile_directory=results_directory / "profiles" / str(resolution),
                **common_arguments,
            )
        )

    if not smoke:
        upscaler.compile()
    for resolution in tqdm(RESOLUTIONS, desc="Latency"):
        if not smoke:
            torch._dynamo.reset()
        efficiency.update(
            measure_latency(
                resolution=resolution,
                latent_size=resolution // vae.get_spatial_compression(),
                warmup_iterations=0 if smoke else WARMUP_ITERATIONS,
                benchmark_iterations=SMOKE_ITERATIONS if smoke else BENCHMARK_ITERATIONS,
                profile_directory=results_directory / "profiles" / str(resolution),
                **common_arguments,
            )
        )
    return efficiency


def measure_quality(
    upscaler: Upscaler,
    vae: Flux2VAE,
    dataset: BenchmarkDataset,
    cache_directory: Path,
    images_directory: Path,
) -> dict[str, float]:
    """
    FID is computed against the images of the same dataset.
    """
    image_size = dataset.load_image(dataset.sample_names[0]).shape[-1]
    cache = LatentCache(
        cache_directory,
        metadata={
            "vae": vae.get_vae_name(),
            "vae_dtype": str(VAE_DTYPE),
            "scale_factor": SCALE_FACTOR,
            "codec": LATENT_CODEC,
            "seed": SEED,
            "image_size": image_size,
        },
        codec=LATENT_CODEC,
    )
    populate_latent_cache(dataset, vae, cache, SCALE_FACTOR, SEED)
    fid_evaluator = build_fid_evaluator(dataset, cache, image_size)

    return evaluate_quality(
        upscaler=upscaler,
        vae=vae,
        dataset=dataset,
        cache=cache,
        fid_evaluator=fid_evaluator,
        scale_factor=SCALE_FACTOR,
        upscaler_dtype=UPSCALER_DTYPE,
        images_directory=images_directory,
        num_saved_samples=NUM_SAVED_SAMPLES,
    )


def run_benchmark(exit_stack: ExitStack) -> None:
    arguments = parse_arguments()
    smoke = arguments.smoke
    dataset_names = [dataset_path.name for dataset_path in arguments.datasets]

    upscaler = load_upscaler(arguments.upscaler)
    vae = Flux2VAE(str(VAE_PATH), device="cuda", dtype=VAE_DTYPE)

    results_directory = RAW_RESULTS_DIRECTORY / upscaler.get_name()
    cache_root_directory = BENCHMARK_CACHE_DIRECTORY
    if smoke:
        temporary_directory = Path(exit_stack.enter_context(tempfile.TemporaryDirectory()))
        results_directory = temporary_directory / "results"
        cache_root_directory = temporary_directory / "cache"
    elif results_directory.exists():
        shutil.rmtree(results_directory)
    results_directory.mkdir(parents=True)

    upscaler.prepare_to_benchmark()
    efficiency = measure_efficiency(upscaler, vae, results_directory, smoke)

    if not smoke:
        torch._dynamo.reset()
    quality: dict[str, dict[str, float]] = {}
    for dataset_path, dataset_name in zip(arguments.datasets, dataset_names):
        dataset = BenchmarkDataset(dataset_path, max_samples=SMOKE_NUM_SAMPLES if smoke else None)
        quality[dataset_name] = measure_quality(
            upscaler=upscaler,
            vae=vae,
            dataset=dataset,
            cache_directory=cache_root_directory / dataset_name,
            images_directory=results_directory / f"images_{dataset_name}",
        )

    metrics = {
        "settings": {"datasets": dataset_names, "upscaler": upscaler.get_name(), "smoke": smoke},
        "efficiency": efficiency,
        "quality": quality,
    }
    if not smoke:
        (results_directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
