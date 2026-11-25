"""
Unified benchmark of 2x upscaling solutions for the FLUX.2 VAE on 1000 images of Axis-V1 (same pattern as the Wan 2.1
benchmark `scripts/run_upscale_benchmark.py`).

Task of every solution: the normalized FLUX.2 latent of a 512x512 image (`[1, 32, 64, 64]`, nothing else: no prompt,
no hff, no ground truth information) in, the decoded 1024x1024 image out. Data: the autoresearch_3 set
`benchmark_datasets/axis_v1_1k_flux2_2x` (Axis-V1 center crops to 1024, antialiased; cv2 bilinear 2x downscale;
full FLUX.2 VAE encode), built here if missing.

Metrics per solution: fid, patch_fid (256 px tiles), low_res_mse and chroma_shift_ciede2000 (roundtrip: the 1024
prediction is downsampled to 512 and compared with the initial 512 image), latency_ms (batch size 1, after 10 warmup
passes on fake latents, decode included), peak_vram_mb (peak allocated CUDA memory, weights included).

Prerequisites (repository root, project venv activated; models go to `models/`):
    1. Dataset: hf download refine-axis/axis-v1_1k --repo-type dataset --local-dir datasets/axis-v1_1k
    2. FLUX.2 VAE: models/FLUX.2-klein-4B/vae (black-forest-labs/FLUX.2-klein-4B, `vae/*`)
    3. Our checkpoints (raw weights): python model_ops/download_checkpoint.py --weights raw
           --run flux_pretrain_v1 flux_distill_v1 flux_small_s20m
    4. TAEF2: hf download madebyollin/taef2 taef2.safetensors --local-dir models/taef2
    5. FlowUpscaler: hf download TensorForger/FlowUpscaler flow_upscaler.safetensors upscaler_unet.py
           --local-dir models/FlowUpscaler
    6. SesquiLSR: git clone https://github.com/LoganBooker/SesquiLSR models/SesquiLSR (upscaler_Flux2.safetensors)
    7. SPAN / Real-ESRGAN: see `scripts/run_upscale_benchmark.py` (models/2xNomosUni_span_multijpg,
           models/RealESRGAN_x2plus.pth)
    8. VOSR (needs timm): git clone https://github.com/cswry/VOSR models/VOSR
           hf download CSWRY/VOSR --local-dir models/VOSR_weights --include "VOSR_0.5B_os/*" "VOSR_1.4B_os/*"
               "Qwen-Image-vae-2d/*" sd21_lwdecoder.pth "stable-diffusion-2-1-base/vae/config.json"
               "stable-diffusion-2-1-base/vae/diffusion_pytorch_model.safetensors" "torch_cache/checkpoints/dinov2_vit*"
               "torch_cache/facebookresearch_dinov2_main/*"

Run:
    python scripts/run_flux_upscale_benchmark.py
    python scripts/run_flux_upscale_benchmark.py --solutions decode_bicubic sesquilsr --force
    python scripts/run_flux_upscale_benchmark.py --max-samples 8     # quick check, writes to dev/flux_upscale_benchmark_smoke

Results: `raw_results/flux_upscale_benchmark_axis_v1_1k/` (`results.json` of all solutions, `<solution>/results.json`
and `<solution>/images/<sample>.png`). Solutions with existing results are skipped unless `--force` is set.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from stdev.flux_upscale_benchmark.dataset import FluxUpscaleBenchmarkDataset
from stdev.flux_upscale_benchmark.registry import build_solution_factories
from stdev.flux_upscale_benchmark.solution import INPUT_SHAPE
from stdev.upscale_benchmark.dataset import load_image_batch
from stdev.upscale_benchmark.metrics import roundtrip_errors
from stdev.upscale_benchmark.runner import evaluate_predictions, release_gpu_memory, run_inference
from stdev.vaes.flux_2_vae import Flux2VAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIRECTORY = REPOSITORY_ROOT / "models"
DATASET_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "axis_v1_1k_flux2_2x"
STATISTICS_DIRECTORY = REPOSITORY_ROOT / "benchmark_cache" / "axis_v1_1k_flux2_2x_upscale"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "raw_results" / "flux_upscale_benchmark_axis_v1_1k"
SMOKE_RESULTS_DIRECTORY = REPOSITORY_ROOT / "dev" / "flux_upscale_benchmark_smoke"
VAE_PATH = MODELS_DIRECTORY / "FLUX.2-klein-4B" / "vae"

SEED = 0
NUM_WARMUP_SAMPLES = 10
ROUNDTRIP_BATCH_SIZE = 8
SUMMARY_COLUMNS = (
    "fid",
    "patch_fid",
    "low_res_mse",
    "chroma_shift_ciede2000",
    "latency_ms",
    "peak_vram_mb",
)


def parse_arguments(solution_names: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solutions", nargs="+", choices=solution_names, default=solution_names)
    parser.add_argument(
        "--force", action="store_true", help="Rerun solutions that already have results."
    )
    parser.add_argument(
        "--max-samples", type=int, help="Use only the first N samples (quick check)."
    )
    parser.add_argument(
        "--external",
        nargs="+",
        default=[],
        help="Only evaluate solutions whose predictions and `external.json` (description, settings, efficiency)"
        " were written by another script (e.g. scripts/run_hildegard_benchmark.py).",
    )
    return parser.parse_args()


def ensure_dataset() -> None:
    if (DATASET_DIRECTORY / "manifest.json").exists():
        return
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
    from encode_flux2_datasets import build_benchmark

    vae = Flux2VAE(str(VAE_PATH), device="cuda", dtype=torch.bfloat16)
    build_benchmark(vae, DATASET_DIRECTORY, None)
    del vae
    release_gpu_memory()


def collect_results(results_directory: Path) -> dict[str, dict]:
    return {
        path.parent.name: json.loads(path.read_text())
        for path in sorted(results_directory.glob("*/results.json"))
    }


def evaluate_roundtrip(predictions_directory: Path, sample_names: list[str]) -> dict[str, float]:
    """
    Per image metrics only (roundtrip MSE and CIEDE2000), for solutions evaluated on a subset where FID is not
    meaningful.
    """
    mse_values: list[float] = []
    color_difference_values: list[float] = []
    for start in range(0, len(sample_names), ROUNDTRIP_BATCH_SIZE):
        names = sample_names[start : start + ROUNDTRIP_BATCH_SIZE]
        mse, color_difference = roundtrip_errors(
            load_image_batch(predictions_directory, names),
            load_image_batch(DATASET_DIRECTORY / "low_resolution", names),
        )
        mse_values.extend(mse.tolist())
        color_difference_values.extend(color_difference.tolist())
    return {
        "low_res_mse": float(np.mean(mse_values)),
        "chroma_shift_ciede2000": float(np.mean(color_difference_values)),
    }


def evaluate_external(
    name: str, results_directory: Path, dataset: FluxUpscaleBenchmarkDataset
) -> dict:
    external = json.loads((results_directory / name / "external.json").read_text())
    num_predictions = len(list((results_directory / name / "images").glob("*.png")))
    if num_predictions < len(dataset):
        sample_names = dataset.sample_names[:num_predictions]
        quality = {
            "fid": None,
            "patch_fid": None,
            **evaluate_roundtrip(results_directory / name / "images", sample_names),
        }
        external["settings"]["subset"] = (
            f"first {num_predictions} samples; fid / patch_fid not measured (too slow for the full set)"
        )
    else:
        num_predictions = len(dataset)
        quality = evaluate_predictions(
            name, results_directory / name / "images", dataset, STATISTICS_DIRECTORY
        )
    return {
        "description": external["description"],
        "settings": external["settings"],
        "num_samples": num_predictions,
        "seed": SEED,
        "metrics": {**quality, **external["efficiency"]},
    }


def print_summary(results: dict[str, dict]) -> None:
    name_width = max(len(name) for name in results)
    header = f"{'solution':<{name_width}}  " + "  ".join(
        f"{column:>22}" for column in SUMMARY_COLUMNS
    )
    print(header)
    for name, result in sorted(
        results.items(), key=lambda item: item[1]["metrics"]["patch_fid"] or float("inf")
    ):
        values = "  ".join(
            f"{result['metrics'][column]:>22.4f}"
            if result["metrics"][column] is not None
            else f"{'not measured':>22}"
            for column in SUMMARY_COLUMNS
        )
        print(f"{name:<{name_width}}  {values}")


def run_benchmark() -> None:
    factories = build_solution_factories(MODELS_DIRECTORY)
    arguments = parse_arguments(list(factories))
    results_directory = SMOKE_RESULTS_DIRECTORY if arguments.max_samples else RESULTS_DIRECTORY

    ensure_dataset()
    dataset = FluxUpscaleBenchmarkDataset(DATASET_DIRECTORY, arguments.max_samples)
    print(f"Benchmark dataset: {len(dataset)} samples, results: {results_directory}")

    for name in arguments.external:
        result = evaluate_external(name, results_directory, dataset)
        (results_directory / name / "results.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({name: result}, indent=2))

    for name in [] if arguments.external else arguments.solutions:
        result_path = results_directory / name / "results.json"
        if result_path.exists() and not arguments.force:
            print(f"Skipping {name}: results exist (use --force to rerun)")
            continue
        solution = factories[name]()
        efficiency = run_inference(
            solution,
            dataset,
            results_directory / name / "images",
            SEED,
            NUM_WARMUP_SAMPLES,
            INPUT_SHAPE,
        )
        description, settings = solution.description, solution.get_settings()
        del solution
        release_gpu_memory()
        quality = evaluate_predictions(
            name, results_directory / name / "images", dataset, STATISTICS_DIRECTORY
        )

        result = {
            "description": description,
            "settings": settings,
            "num_samples": len(dataset),
            "seed": SEED,
            "metrics": {**quality, **efficiency},
        }
        result_path.write_text(json.dumps(result, indent=2))
        print(json.dumps({name: result}, indent=2))

    all_results = collect_results(results_directory)
    if all_results:
        (results_directory / "results.json").write_text(json.dumps(all_results, indent=2))
        print_summary(all_results)


if __name__ == "__main__":
    run_benchmark()
