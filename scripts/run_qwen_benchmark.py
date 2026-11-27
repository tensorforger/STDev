"""
Benchmark of one Qwen-Image 2.1 latent upscaler run file (or a non-learned reference) on `axis_v1_1k_qwen_2x`
(autoresearch_4).

Order: peak memory of the eager single forward -> compile -> single forward latency (batch 1) -> end-to-end latency
(full `upscale` + texture-fix decode, batch 1, info only) -> eager reload -> quality on all 1000 images.

Results: `raw_results/qwen/<run name>/metrics.json` and the first NUM_SAVED_IMAGES predictions in `images/`.

Run:
    python scripts/run_qwen_benchmark.py --upscaler src/stdev/qwen_upscalers/run_0000.py
    python scripts/run_qwen_benchmark.py --reference oracle
    python scripts/run_qwen_benchmark.py --reference oracle --decoder official    # official decoder, info only
    python scripts/run_qwen_benchmark.py --prepare-cache
    python scripts/run_qwen_benchmark.py --upscaler ... --smoke     # 16 images, nothing is written
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch

from stdev.qwen_benchmark.benchmark import (
    INPUT_LATENT_SIZE,
    LATENT_CHANNELS,
    REFERENCES,
    QwenBenchmarkDataset,
    build_fid_evaluator,
    decode_rgb,
    evaluate_quality,
    measure_latency,
    measure_peak_memory,
    reference_pixel_predictor,
    upscaler_pixel_predictor,
)
from stdev.qwen_upscalers.loading import load_upscaler
from stdev.vaes.qwen_image_2_1_vae import QwenImage21VAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "axis_v1_1k_qwen_2x"
STATISTICS_DIRECTORY = REPOSITORY_ROOT / "benchmark_cache" / "axis_v1_1k_qwen_2x"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "raw_results" / "qwen"
VAE_PATHS = {
    "texture_fix": REPOSITORY_ROOT / "models" / "texture-fix-vae-for-qwen-image-2.1",
    "official": REPOSITORY_ROOT / "models" / "Qwen-Image-2.1" / "vae",
}
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
    parser.add_argument(
        "--decoder",
        choices=tuple(VAE_PATHS),
        default="texture_fix",
        help="Benchmark decoder; only texture_fix results are comparable, others add a suffix to the name.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    smoke = arguments.smoke
    dataset = QwenBenchmarkDataset(DATASET_DIRECTORY, max_samples=SMOKE_NUM_SAMPLES if smoke else None)
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
        if arguments.decoder != "texture_fix":
            name = f"{name}_{arguments.decoder}_decoder"

        vae = QwenImage21VAE(str(VAE_PATHS[arguments.decoder]), device="cuda", dtype=VAE_DTYPE)
        if upscaler is not None:
            end_to_end = measure_latency(
                lambda: decode_rgb(vae, upscaler.upscale(latents)),
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
            "settings": {
                "name": name,
                "smoke": smoke,
                "num_samples": len(dataset),
                "decoder": arguments.decoder,
            },
            "quality": quality,
            "efficiency": efficiency,
        }
        if not smoke:
            (results_directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
