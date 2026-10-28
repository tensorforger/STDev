"""
Trains a run briefly into a temporary directory, then reports parameter count, training throughput,
memory and compiled latency the way the benchmark measures them, and optionally a quick quality
check on a few benchmark samples.

Usage: python dev/profile_run.py RUN [--train-seconds 60] [--override NAME=EXPR ...]
           [--quality-samples 0] [--weights-output DIR] [--no-latency]
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tools import REPOSITORY_ROOT, instantiate_upscaler, load_run_module, resolve_run_file

TEMPORARY_ROOT = REPOSITORY_ROOT / "dev" / "temporary"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--train-seconds", type=float, default=60)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--quality-samples", type=int, default=0)
    parser.add_argument("--weights-output", type=Path)
    parser.add_argument("--no-latency", action="store_true")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[512, 1024, 2048])
    arguments = parser.parse_args()

    TEMPORARY_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(tempfile.mkdtemp(prefix="profile_", dir=TEMPORARY_ROOT))
    weights_directory = arguments.weights_output or temporary_directory / "weights"
    overrides = [
        f"OUTPUT_DIRECTORY=__import__('pathlib').Path({str(weights_directory)!r})",
        f"TENSORBOARD_DIRECTORY=__import__('pathlib').Path({str(temporary_directory / 'tb')!r})",
        f"TRAINING_TIME_SECONDS={arguments.train_seconds}",
        *arguments.override,
    ]
    module = load_run_module(resolve_run_file(arguments.run), overrides)
    upscaler = instantiate_upscaler(module)
    report: dict[str, float] = {}

    start_time = time.perf_counter()
    upscaler.train(smoke_mode=False)
    report["train_wall_seconds"] = time.perf_counter() - start_time
    torch.cuda.empty_cache()
    report["train_peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3

    upscaler.prepare_to_benchmark()
    report["parameters_m"] = sum(p.numel() for p in upscaler.model.parameters()) / 1e6

    from stdev.benchmark.efficiency import measure_latency, measure_memory

    common = {
        "upscaler": upscaler,
        "latent_channels": 32,
        "scale_factor": 2,
        "dtype": torch.bfloat16,
        "prompt_embeds": torch.zeros((1, 77, 768), device="cuda"),
        "seed": 0,
    }
    for resolution in arguments.resolutions:
        report.update(
            measure_memory(
                resolution=resolution,
                latent_size=resolution // 8,
                warmup_iterations=1,
                profile_directory=temporary_directory / "profiles" / str(resolution),
                **common,
            )
        )
    if not arguments.no_latency:
        upscaler.compile()
        for resolution in arguments.resolutions:
            torch._dynamo.reset()
            report.update(
                measure_latency(
                    resolution=resolution,
                    latent_size=resolution // 8,
                    warmup_iterations=10,
                    benchmark_iterations=50,
                    profile_directory=temporary_directory / "profiles" / str(resolution),
                    **common,
                )
            )
        torch._dynamo.reset()

    if arguments.quality_samples > 0:
        from evaluate_quality import evaluate_upscaler

        report.update(evaluate_upscaler(upscaler, arguments.quality_samples, fid=False))

    print(json.dumps({key: round(value, 5) for key, value in report.items()}, indent=2))
    shutil.rmtree(temporary_directory, ignore_errors=True)


if __name__ == "__main__":
    main()
