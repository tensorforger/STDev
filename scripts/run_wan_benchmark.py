import argparse
import json
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

import torch
from tqdm import tqdm

from stdev.vaes.tiny_wan_2_1_vae import TinyWan21VAE
from stdev.vaes.wan_2_1_vae import Wan21VAE
from stdev.wan_benchmark.dataset import WanBenchmarkDataset
from stdev.wan_benchmark.efficiency import EFFICIENCY_SCENARIOS, measure_latency, measure_memory
from stdev.wan_benchmark.quality import QUALITY_TASKS, build_fid_evaluator, evaluate_quality
from stdev.wan_upscalers.base import Upscaler
from stdev.wan_upscalers.loading import load_upscaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CACHE_DIRECTORY = REPOSITORY_ROOT / "benchmark_cache"
RAW_RESULTS_DIRECTORY = REPOSITORY_ROOT / "raw_results"
VAE_PATH = REPOSITORY_ROOT / "models" / "Wan2.1-T2V-1.3B-Diffusers" / "vae"
TINY_VAE_PATH = REPOSITORY_ROOT / "models" / "taew2_1.safetensors"

VAE_DTYPE = torch.bfloat16
UPSCALER_DTYPE = torch.bfloat16
MEMORY_WARMUP_ITERATIONS = 3
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50
SMOKE_ITERATIONS = 2
SMOKE_NUM_SAMPLES = 2
NUM_SAVED_IMAGES = 32
NUM_SAVED_VIDEOS = 8
SEED = 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        required=True,
        help=f"Benchmark dataset directories, names must be in {sorted(QUALITY_TASKS)}.",
    )
    parser.add_argument("--upscaler", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--prepare-cache",
        action="store_true",
        help="Only fill the FID statistics caches, then exit. Run once before benchmarking in parallel "
        "to avoid concurrent cache writes.",
    )
    arguments = parser.parse_args()
    if arguments.upscaler is None and not arguments.prepare_cache:
        parser.error("--upscaler is required unless --prepare-cache is set")
    for dataset_path in arguments.datasets:
        if dataset_path.name not in QUALITY_TASKS:
            parser.error(
                f"Unknown dataset {dataset_path.name}, expected one of {sorted(QUALITY_TASKS)}"
            )
    return arguments


def measure_efficiency(
    upscaler: Upscaler, vae: Wan21VAE, results_directory: Path, smoke: bool
) -> dict[str, float]:
    """
    Memory is profiled on the eager model, then latency is measured on the compiled model.
    """
    common_arguments = {
        "upscaler": upscaler,
        "latent_channels": vae.get_latent_channels(),
        "dtype": UPSCALER_DTYPE,
        "seed": SEED,
    }
    efficiency: dict[str, float] = {}
    for scenario in tqdm(EFFICIENCY_SCENARIOS, desc="Memory"):
        efficiency.update(
            measure_memory(
                scenario=scenario,
                warmup_iterations=0 if smoke else MEMORY_WARMUP_ITERATIONS,
                profile_directory=results_directory / "profiles" / scenario.name,
                **common_arguments,
            )
        )

    if not smoke:
        upscaler.compile()
    for scenario in tqdm(EFFICIENCY_SCENARIOS, desc="Latency"):
        if not smoke:
            torch._dynamo.reset()
        efficiency.update(
            measure_latency(
                scenario=scenario,
                warmup_iterations=0 if smoke else WARMUP_ITERATIONS,
                benchmark_iterations=SMOKE_ITERATIONS if smoke else BENCHMARK_ITERATIONS,
                profile_directory=results_directory / "profiles" / scenario.name,
                **common_arguments,
            )
        )
    return efficiency


def run_benchmark(exit_stack: ExitStack) -> None:
    arguments = parse_arguments()
    smoke = arguments.smoke
    dataset_names = [dataset_path.name for dataset_path in arguments.datasets]

    def open_dataset(dataset_path: Path) -> WanBenchmarkDataset:
        return WanBenchmarkDataset(
            dataset_path,
            kind=QUALITY_TASKS[dataset_path.name].kind,
            max_samples=SMOKE_NUM_SAMPLES if smoke else None,
        )

    if arguments.prepare_cache:
        for dataset_path in arguments.datasets:
            build_fid_evaluator(
                open_dataset(dataset_path),
                BENCHMARK_CACHE_DIRECTORY / dataset_path.name,
                QUALITY_TASKS[dataset_path.name].image_size,
            )
        return

    vae = Wan21VAE(str(VAE_PATH), device="cuda", dtype=VAE_DTYPE)
    tiny_vae = TinyWan21VAE(str(TINY_VAE_PATH), device="cuda", dtype=VAE_DTYPE)
    upscaler = load_upscaler(arguments.upscaler)

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
    for dataset_path in arguments.datasets:
        dataset = open_dataset(dataset_path)
        task = QUALITY_TASKS[dataset_path.name]
        quality[dataset_path.name] = evaluate_quality(
            upscaler=upscaler,
            vae=tiny_vae if task.kind == "video" else vae,
            dataset=dataset,
            task=task,
            fid_evaluator=build_fid_evaluator(
                dataset, cache_root_directory / dataset_path.name, task.image_size
            ),
            upscaler_dtype=UPSCALER_DTYPE,
            samples_directory=results_directory
            / f"{'images' if task.kind == 'image' else 'videos'}_{dataset_path.name}",
            num_saved_samples=NUM_SAVED_IMAGES if task.kind == "image" else NUM_SAVED_VIDEOS,
        )

    metrics = {
        "settings": {"datasets": dataset_names, "upscaler": upscaler.get_name(), "smoke": smoke},
        "efficiency": efficiency,
        "quality": quality,
    }
    if not smoke:
        (results_directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def main() -> None:
    with ExitStack() as exit_stack:
        run_benchmark(exit_stack)


if __name__ == "__main__":
    main()
