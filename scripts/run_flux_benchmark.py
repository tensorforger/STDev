"""
Benchmark of one FLUX.2 latent upscaler run file (or a non-learned reference) on `axis_v1_1k_flux2_2x`.

Order: peak memory of the eager single forward -> compile -> single forward latency (batch 1) -> end-to-end latency
(full `upscale` + FLUX.2 decode, batch 1, info only) -> eager reload -> quality on all 1000 images.

Results: `raw_results/flux/<run name>/metrics.json` and the first NUM_SAVED_IMAGES predictions in `images/`.

Run:
    python scripts/run_flux_benchmark.py --upscaler src/stdev/flux_upscalers/run_0000.py
    python scripts/run_flux_benchmark.py --reference oracle
    python scripts/run_flux_benchmark.py --prepare-cache
    python scripts/run_flux_benchmark.py --upscaler ... --smoke     # 16 images, nothing is written
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch

from stdev.flux_benchmark.benchmark import (
    INPUT_LATENT_SIZE,
    LATENT_CHANNELS,
    REFERENCES,
    FluxBenchmarkDataset,
    build_fid_evaluator,
    evaluate_quality,
    measure_latency,
    measure_peak_memory,
    reference_pixel_predictor,
    upscaler_pixel_predictor,
)
from stdev.flux_upscalers.loading import load_upscaler
from stdev.vaes.flux_2_vae import Flux2VAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "axis_v1_1k_flux2_2x"
STATISTICS_DIRECTORY = REPOSITORY_ROOT / "benchmark_cache" / "axis_v1_1k_flux2_2x"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "raw_results" / "flux"
VAE_PATH = REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B" / "vae"
VAE_DTYPE = torch.bfloat16

MEMORY_ITERATIONS = 3
WARMUP_ITERATIONS = 10
LATENCY_ITERATIONS = 100
END_TO_END_ITERATIONS = 20
SMOKE_NUM_SAMPLES = 16
SMOKE_ITERATIONS = 2
NUM_SAVED_IMAGES = 32


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--upscaler", type=Path)
    group.add_argument("--reference", choices=REFERENCES)
    group.add_argument("--prepare-cache", action="store_true", help="Only fill the FID statistics cache.")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    smoke = arguments.smoke
    dataset = FluxBenchmarkDataset(DATASET_DIRECTORY, max_samples=SMOKE_NUM_SAMPLES if smoke else None)
    with tempfile.TemporaryDirectory() as temporary_directory:
        statistics_directory = Path(temporary_directory) / "cache" if smoke else STATISTICS_DIRECTORY
        if arguments.prepare_cache:
            build_fid_evaluator(dataset, statistics_directory)
            return

        efficiency: dict[str, float] = {}
        upscaler = None
        if arguments.upscaler is not None:
            upscaler = load_upscaler(arguments.upscaler)
            name = upscaler.get_name()
            iterations = SMOKE_ITERATIONS if smoke else MEMORY_ITERATIONS
            efficiency.update(measure_peak_memory(upscaler, iterations))
            if not smoke:
                upscaler.compile()
            latents = torch.randn(1, LATENT_CHANNELS, INPUT_LATENT_SIZE, INPUT_LATENT_SIZE, device="cuda")
            single = measure_latency(
                lambda: upscaler.single_forward(latents),
                0 if smoke else WARMUP_ITERATIONS,
                SMOKE_ITERATIONS if smoke else LATENCY_ITERATIONS,
            )
            efficiency["latency_ms_p95"] = single["p95"]
            efficiency["latency_ms_mean"] = single["mean"]
        else:
            name = f"reference_{arguments.reference}"

        vae = Flux2VAE(str(VAE_PATH), device="cuda", dtype=VAE_DTYPE)
        if upscaler is not None:
            end_to_end = measure_latency(
                lambda: vae.decode(upscaler.upscale(latents).unsqueeze(2)),
                0 if smoke else 3,
                SMOKE_ITERATIONS if smoke else END_TO_END_ITERATIONS,
            )
            efficiency["end_to_end_latency_ms_p95"] = end_to_end["p95"]
            if not smoke:
                torch._dynamo.reset()
            upscaler.prepare_to_benchmark()

        results_directory = Path(temporary_directory) / "results" if smoke else RESULTS_DIRECTORY / name
        if results_directory.exists():
            shutil.rmtree(results_directory)
        results_directory.mkdir(parents=True)
        latent_errors: list[float] = []
        if upscaler is not None:
            predictor = upscaler_pixel_predictor(upscaler, vae, dataset, latent_errors)
        else:
            predictor = reference_pixel_predictor(arguments.reference, vae, dataset)
        quality = evaluate_quality(
            predictor,
            dataset,
            build_fid_evaluator(dataset, statistics_directory),
            results_directory / "images",
            NUM_SAVED_IMAGES,
        )
        if latent_errors:
            quality["latent_mse_hr"] = sum(latent_errors) / len(latent_errors)

        metrics = {
            "settings": {"name": name, "smoke": smoke, "num_samples": len(dataset)},
            "quality": quality,
            "efficiency": efficiency,
        }
        if not smoke:
            (results_directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
