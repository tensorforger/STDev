"""
Full autoresearch_3 benchmark (axis_v1_1k_flux2_2x: 1000 images, patch_fid, fid, low_res_mse, chroma_ciede2000,
latent_mse_hr) of a FLUX.2 pretrain checkpoint, or of a reference. Builds the benchmark set if missing (same code as
scripts/encode_flux2_datasets.py). Results: raw_results/flux/<name>/metrics.json (+ first 32 images).

Usage: CUDA_VISIBLE_DEVICES=7 python scripts/flux_tools/flux_pretrain_full_benchmark.py --checkpoint PATH [--weights ema|raw]
           [--hff gt|null|VALUE] [--steps 30] [--shift 2.0] [--batch 4] [--max-samples N]
       CUDA_VISIBLE_DEVICES=7 python scripts/flux_tools/flux_pretrain_full_benchmark.py --reference oracle|pixel_bicubic
"""

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

import torch

from stdev.flux_benchmark.benchmark import (
    FluxBenchmarkDataset,
    build_fid_evaluator,
    evaluate_quality,
    reference_pixel_predictor,
)
from stdev.flux_pretrain.config import MODEL, REPOSITORY_ROOT, VAE_DIRECTORY
from stdev.flux_pretrain.model import FluxUpscalerTransformer, ModelConfig, SamplerConfig, upscale
from stdev.vaes.flux_2_vae import Flux2VAE

DATASET_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "axis_v1_1k_flux2_2x"
STATISTICS_DIRECTORY = REPOSITORY_ROOT / "benchmark_cache" / "axis_v1_1k_flux2_2x"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "raw_results" / "flux"
NUM_SAVED_IMAGES = 32
ENCODE_BATCH_SIZE = 2


def build_dataset_if_missing(vae: Flux2VAE) -> None:
    if (DATASET_DIRECTORY / "manifest.json").exists():
        return
    specification = importlib.util.spec_from_file_location("encode_flux2_datasets", REPOSITORY_ROOT / "scripts" / "encode_flux2_datasets.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.BENCHMARK_ENCODE_BATCH_SIZE = ENCODE_BATCH_SIZE
    module.DEVICE = "cuda"
    module.build_benchmark(vae, DATASET_DIRECTORY, None)


def load_model(checkpoint_path: Path, weights: str) -> FluxUpscalerTransformer:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ModelConfig(**{**checkpoint["model_config"], "residual_std": tuple(checkpoint["model_config"]["residual_std"])})
    assert config == MODEL, "checkpoint config differs from the current MODEL"
    model = FluxUpscalerTransformer(config)
    model.load_state_dict(checkpoint["model"])
    if weights == "ema":
        with torch.no_grad():
            for parameter, ema_parameter in zip(model.parameters(), checkpoint["ema"]):
                parameter.copy_(ema_parameter)
    print(f"loaded {checkpoint_path.name} (step {checkpoint['state']['step']}, {checkpoint['state']['training_seconds'] / 3600:.2f} h), {weights}", flush=True)
    return model.to("cuda", torch.bfloat16).eval().requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", choices=("oracle", "pixel_bicubic"))
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--hff", default="gt")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--shift", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--name", default=None)
    arguments = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    vae = Flux2VAE(str(VAE_DIRECTORY), device="cuda", dtype=torch.bfloat16)
    build_dataset_if_missing(vae)
    dataset = FluxBenchmarkDataset(DATASET_DIRECTORY, max_samples=arguments.max_samples)
    hff_values = torch.load(DATASET_DIRECTORY / "hff.pt").float()[: len(dataset)]
    latent_errors: list[float] = []

    if arguments.reference is not None:
        name = arguments.name or f"reference_{arguments.reference}"
        reference_predictor = reference_pixel_predictor(arguments.reference, vae, dataset)

        def predictor(latents: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [reference_predictor(chunk, index_chunk) for chunk, index_chunk in zip(latents.split(arguments.batch), indices.split(arguments.batch))]
            )
    else:
        model = load_model(arguments.checkpoint, arguments.weights)
        model.compile_blocks()
        sampler = SamplerConfig(steps=arguments.steps, shift=arguments.shift, noise_scale=1.0)
        step_name = arguments.checkpoint.stem.split("_")[-1].lstrip("0")
        name = arguments.name or f"flux_pretrain_v1_{step_name}_{arguments.weights}_hff_{arguments.hff}_{arguments.steps}_{arguments.shift:g}"

        def predictor(latents: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            if arguments.hff == "gt":
                hff = hff_values[indices]
            elif arguments.hff == "null":
                hff = torch.full((len(indices),), float("nan"))
            else:
                hff = torch.full((len(indices),), float(arguments.hff))
            upscaled = torch.cat(
                [
                    upscale(model, chunk, hff_chunk.cuda(), sampler)
                    for chunk, hff_chunk in zip(latents.split(arguments.batch), hff.split(arguments.batch))
                ]
            )
            latent_errors.extend((upscaled - dataset.latents_high[indices].cuda()).square().mean(dim=(1, 2, 3)).tolist())
            return torch.cat([vae.decode(chunk.unsqueeze(2)) for chunk in upscaled.split(arguments.batch)])

    results_directory = RESULTS_DIRECTORY / name
    if results_directory.exists():
        shutil.rmtree(results_directory)
    results_directory.mkdir(parents=True)
    start_time = time.perf_counter()
    quality = evaluate_quality(
        predictor, dataset, build_fid_evaluator(dataset, STATISTICS_DIRECTORY), results_directory / "images", NUM_SAVED_IMAGES
    )
    if latent_errors:
        quality["latent_mse_hr"] = sum(latent_errors) / len(latent_errors)
    metrics = {
        "settings": {**{key: str(value) for key, value in vars(arguments).items()}, "name": name, "num_samples": len(dataset)},
        "quality": quality,
        "seconds": time.perf_counter() - start_time,
    }
    (results_directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
