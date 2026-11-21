"""
Unified benchmark of 2x latent upscaling solutions for the Wan 2.1 VAE on 1000 images of Axis-V1.

Task of every solution: the latent (64x64, normalized Wan 2.1 VAE latent, nothing else) of a 512x512 image in,
the decoded 1024x1024 image out. Ground truth: Axis-V1 images center cropped to 1024x1024 (`crop_maximal_rectangle`,
bilinear), inputs: ground truth bilinearly downsampled to 512x512 and encoded with the full Wan 2.1 VAE.

Metrics per solution: fid, patch_fid (256 px tiles), low_res_mse and chroma_shift_ciede2000 (roundtrip: the 1024
prediction is downsampled to 512 and compared with the initial 512 image), latency_ms (batch size 1, after 10 warmup
passes on fake latents, decode included), peak_vram_mb (peak allocated CUDA memory during processing, weights included).
All 1024x1024 predictions are saved as PNG.

Our solutions run twice: with the null hff (the model decides the sharpness itself, the fair setting) and, as
`<name>_gt_hff`, with the Immerkaer hff of the ground truth image (privileged, upper bound). Guidance is 1.0 and the
noise seed of every image is identical in both runs.

Prerequisites (repository root, project venv activated; models go to `models/`):
    1. Environment: `uv pip install -e .` (includes spandrel, which loads the SPAN and Real-ESRGAN weights).
    2. Dataset (needs git-lfs):
           git clone https://huggingface.co/datasets/refine-axis/axis-v1_1k datasets/axis-v1_1k
    3. Full Wan 2.1 VAE (all solutions):
           hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --include "vae/*" --local-dir models/Wan2.1-T2V-1.3B-Diffusers
    4. Tiny Wan 2.1 VAE (tiny VAE solutions):
           curl -L -o models/taew2_1.safetensors https://github.com/madebyollin/taehv/raw/main/safetensors/taew2_1.safetensors
    5. Our checkpoints are already in `models/latent_upscale_models` (newest step of each run is used):
           wan_pretrain_v1 (base), wan_distill_v1 (200M), wan_small_s20m (20M), wan_small_s2m (2M)
    6. spacepxl/Wan2.1-VAE-upscale2x (decoder with integrated 2x upscaling):
           hf download spacepxl/Wan2.1-VAE-upscale2x --include "diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1/*" --local-dir models/Wan2.1-VAE-upscale2x
    7. WanNNLatentUpscaler (neural latent upscaler of ComfyUI-VAE-Utils):
           curl -L -o models/wan21_latent_upscale_2x.safetensors https://github.com/spacepxl/ComfyUI-VAE-Utils/raw/main/latent_upscale/wan21_latent_upscale_2x.safetensors
    8. SPAN 2x (Phips, 2xNomosUni_span_multijpg):
           hf download Phips/2xNomosUni_span_multijpg 2xNomosUni_span_multijpg.safetensors --local-dir models/2xNomosUni_span_multijpg
    9. Real-ESRGAN x2plus:
           curl -L -o models/RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
    10. SesquiLSR (code and Wan 2.1 weights `models/upscaler_Wan21.safetensors` in the repository):
           git clone https://github.com/LoganBooker/SesquiLSR models/SesquiLSR

Run:
    python scripts/run_upscale_benchmark.py
    python scripts/run_upscale_benchmark.py --solutions decode_bicubic decode_span --force
    python scripts/run_upscale_benchmark.py --max-samples 8     # quick check, writes to dev/upscale_benchmark_smoke

The dataset is built once (`benchmark_datasets/axis_v1_1k_wan_2x`). Results: `raw_results/upscale_benchmark_axis_v1_1k/`
(`results.json` of all solutions, `<solution>/results.json` and `<solution>/images/<sample>.png`). Solutions with
existing results are skipped unless `--force` is set.
"""

import argparse
import json
from pathlib import Path

from stdev.upscale_benchmark.dataset import BenchmarkDataset, build_dataset
from stdev.upscale_benchmark.registry import build_solution_factories
from stdev.upscale_benchmark.runner import evaluate_predictions, release_gpu_memory, run_inference

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIRECTORY = REPOSITORY_ROOT / "models"
SOURCE_DIRECTORY = REPOSITORY_ROOT / "datasets" / "axis-v1_1k"
DATASET_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "axis_v1_1k_wan_2x"
STATISTICS_DIRECTORY = REPOSITORY_ROOT / "benchmark_cache" / "axis_v1_1k_wan_2x"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "raw_results" / "upscale_benchmark_axis_v1_1k"
SMOKE_RESULTS_DIRECTORY = REPOSITORY_ROOT / "dev" / "upscale_benchmark_smoke"
VAE_PATH = MODELS_DIRECTORY / "Wan2.1-T2V-1.3B-Diffusers" / "vae"

SEED = 0
NUM_WARMUP_SAMPLES = 10
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
    return parser.parse_args()


def collect_results(results_directory: Path, solution_names: list[str]) -> dict[str, dict]:
    return {
        name: json.loads((results_directory / name / "results.json").read_text())
        for name in solution_names
        if (results_directory / name / "results.json").exists()
    }


def print_summary(results: dict[str, dict]) -> None:
    name_width = max(len(name) for name in results)
    header = f"{'solution':<{name_width}}  " + "  ".join(
        f"{column:>22}" for column in SUMMARY_COLUMNS
    )
    print(header)
    for name, result in results.items():
        values = "  ".join(f"{result['metrics'][column]:>22.4f}" for column in SUMMARY_COLUMNS)
        print(f"{name:<{name_width}}  {values}")


def run_benchmark() -> None:
    factories = build_solution_factories(MODELS_DIRECTORY)
    arguments = parse_arguments(list(factories))
    results_directory = SMOKE_RESULTS_DIRECTORY if arguments.max_samples else RESULTS_DIRECTORY

    build_dataset(SOURCE_DIRECTORY, DATASET_DIRECTORY, VAE_PATH)
    dataset = BenchmarkDataset(DATASET_DIRECTORY, arguments.max_samples)
    print(f"Benchmark dataset: {len(dataset)} samples, results: {results_directory}")

    for name in arguments.solutions:
        result_path = results_directory / name / "results.json"
        if result_path.exists() and not arguments.force:
            print(f"Skipping {name}: results exist (use --force to rerun)")
            continue
        solution = factories[name]()
        efficiency = run_inference(
            solution, dataset, results_directory / name / "images", SEED, NUM_WARMUP_SAMPLES
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

        all_results = collect_results(results_directory, list(factories))
        (results_directory / "results.json").write_text(json.dumps(all_results, indent=2))

    all_results = collect_results(results_directory, list(factories))
    if all_results:
        print_summary(all_results)


if __name__ == "__main__":
    run_benchmark()
