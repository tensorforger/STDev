"""
Benchmark quality of a trained run with constant overrides (sampling steps, shift, seed, ...),
optionally on a subset of benchmark samples and from another weights directory.
FID on a subset is biased upwards; compare only evaluations with the same sample count.

Usage: python dev/evaluate_quality.py RUN [--samples 1000] [--override NAME=EXPR ...]
           [--weights-directory DIR] [--tag NAME]
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tools import REPOSITORY_ROOT, instantiate_upscaler, load_run_module, resolve_run_file

BENCHMARK_DATASET = REPOSITORY_ROOT / "benchmark_datasets" / "flux_1024"
BENCHMARK_CACHE = REPOSITORY_ROOT / "benchmark_cache" / "flux_1024"
EVALUATIONS_DIRECTORY = REPOSITORY_ROOT / "dev" / "evaluations"


def evaluate_upscaler(upscaler, num_samples: int, fid: bool = True) -> dict[str, float]:
    from stdev.benchmark.cache import LatentCache
    from stdev.benchmark.dataset import BenchmarkDataset
    from stdev.benchmark.quality import build_fid_evaluator, evaluate_quality
    from stdev.vaes.flux_2_vae import Flux2VAE

    import json as json_module

    vae = Flux2VAE(
        str(REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B" / "vae"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    metadata = json_module.loads((BENCHMARK_CACHE / "meta.json").read_text())
    cache = LatentCache(BENCHMARK_CACHE, metadata=metadata, codec=metadata["codec"])
    dataset = BenchmarkDataset(BENCHMARK_DATASET, max_samples=num_samples)
    full_dataset = BenchmarkDataset(BENCHMARK_DATASET)

    class NullEvaluator:
        def update(self, image: torch.Tensor) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {}

    fid_evaluator = (
        build_fid_evaluator(full_dataset, cache, metadata["image_size"]) if fid else NullEvaluator()
    )
    with tempfile.TemporaryDirectory() as images_directory:
        return evaluate_quality(
            upscaler=upscaler,
            vae=vae,
            dataset=dataset,
            cache=cache,
            fid_evaluator=fid_evaluator,
            scale_factor=2,
            upscaler_dtype=torch.bfloat16,
            images_directory=Path(images_directory),
            num_saved_samples=0,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--weights-directory", type=Path)
    parser.add_argument("--tag", default="")
    parser.add_argument("--compile", action="store_true")
    arguments = parser.parse_args()

    overrides = list(arguments.override)
    if arguments.weights_directory is not None:
        overrides.append(
            f"OUTPUT_DIRECTORY=__import__('pathlib').Path({str(arguments.weights_directory)!r})"
        )
    run_file = resolve_run_file(arguments.run)
    module = load_run_module(run_file, overrides)
    upscaler = instantiate_upscaler(module)
    upscaler.prepare_to_benchmark()
    if arguments.compile:
        upscaler.compile()
    quality = evaluate_upscaler(upscaler, arguments.samples)

    result = {
        "run": run_file.stem,
        "samples": arguments.samples,
        "overrides": arguments.override,
        "weights_directory": str(arguments.weights_directory or ""),
        "quality": quality,
    }
    EVALUATIONS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    name = f"{run_file.stem}_{arguments.tag or 'eval'}_{arguments.samples}.json"
    (EVALUATIONS_DIRECTORY / name).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
