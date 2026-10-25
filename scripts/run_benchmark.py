import argparse
import importlib.util
import inspect
import json
import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from stdev.benchmark.cache import LatentCache
from stdev.benchmark.dataset import BenchmarkDataset
from stdev.benchmark.efficiency import measure_efficiency
from stdev.benchmark.quality import build_fid_evaluator, evaluate_quality, populate_latent_cache
from stdev.upscalers.base import Upscaler
from stdev.vaes.flux_2_vae import Flux2VAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CACHE_DIRECTORY = REPOSITORY_ROOT / "benchmark_cache"
RAW_RESULTS_DIRECTORY = REPOSITORY_ROOT / "raw_results"
VAE_PATH = REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B" / "vae"

VAE_DTYPE = torch.bfloat16
UPSCALER_DTYPE = torch.bfloat16
SCALE_FACTOR = 2
RESOLUTIONS = [512, 1024, 2048]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50
SMOKE_ITERATIONS = 2
SMOKE_NUM_SAMPLES = 2
NUM_SAVED_SAMPLES = 8
LATENT_CODEC = "lbrq_8bit"
DUMMY_PROMPT = "A photo of an object."
SEED = 0


def load_upscaler(upscaler_path: Path) -> Upscaler:
    """
    Instantiates the single concrete `Upscaler` subclass defined in the file.
    """
    spec = importlib.util.spec_from_file_location(upscaler_path.stem, upscaler_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[upscaler_path.stem] = module
    spec.loader.exec_module(module)
    upscaler_classes = [
        member
        for _, member in inspect.getmembers(module, inspect.isclass)
        if issubclass(member, Upscaler)
        and member.__module__ == module.__name__
        and not inspect.isabstract(member)
    ]
    if len(upscaler_classes) != 1:
        raise RuntimeError(f"Expected one Upscaler in {upscaler_path}, found {upscaler_classes}")
    return upscaler_classes[0]()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--upscaler", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    smoke = arguments.smoke
    dataset_name = arguments.dataset.name
    run_name = f"{dataset_name}_smoke" if smoke else dataset_name

    dataset = BenchmarkDataset(arguments.dataset, max_samples=SMOKE_NUM_SAMPLES if smoke else None)
    upscaler = load_upscaler(arguments.upscaler)
    vae = Flux2VAE(str(VAE_PATH), device="cuda", dtype=VAE_DTYPE)
    image_size = dataset.load_image(dataset.sample_names[0]).shape[-1]

    results_directory = RAW_RESULTS_DIRECTORY / run_name / upscaler.get_name()
    if results_directory.exists():
        shutil.rmtree(results_directory)
    results_directory.mkdir(parents=True)

    upscaler.prepare_to_benchmark()
    if not smoke:
        upscaler.compile()

    efficiency: dict[str, float] = {}
    for resolution in tqdm(RESOLUTIONS, desc="Efficiency"):
        if not smoke:
            torch._dynamo.reset()
        efficiency.update(
            measure_efficiency(
                upscaler=upscaler,
                resolution=resolution,
                latent_channels=vae.get_latent_channels(),
                latent_size=resolution // vae.get_spatial_compression(),
                scale_factor=SCALE_FACTOR,
                dtype=UPSCALER_DTYPE,
                prompt=DUMMY_PROMPT,
                warmup_iterations=0 if smoke else WARMUP_ITERATIONS,
                benchmark_iterations=SMOKE_ITERATIONS if smoke else BENCHMARK_ITERATIONS,
                profile_directory=results_directory / "profiles" / str(resolution),
                seed=SEED,
            )
        )

    cache = LatentCache(
        BENCHMARK_CACHE_DIRECTORY / run_name,
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

    if not smoke:
        torch._dynamo.reset()
    quality = evaluate_quality(
        upscaler=upscaler,
        vae=vae,
        dataset=dataset,
        cache=cache,
        fid_evaluator=fid_evaluator,
        scale_factor=SCALE_FACTOR,
        upscaler_dtype=UPSCALER_DTYPE,
        images_directory=results_directory / "images",
        num_saved_samples=NUM_SAVED_SAMPLES,
    )

    metrics = {
        "settings": {"dataset": dataset_name, "upscaler": upscaler.get_name(), "smoke": smoke},
        "efficiency": efficiency,
        "quality": quality,
    }
    (results_directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
