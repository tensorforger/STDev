"""
Statistics of upscaled outputs as a function of sampling settings, against the teacher:
latent std / mean error, image high frequency energy (Laplacian), saturation and brightness.

Usage: python dev/sampling_diagnostics.py RUN --variant "NAME=EXPR;NAME=EXPR" ... [--samples 32]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tools import REPOSITORY_ROOT, instantiate_upscaler, load_run_module, resolve_run_file

BENCHMARK_DATASET = REPOSITORY_ROOT / "benchmark_datasets" / "flux_1024"
BENCHMARK_CACHE = REPOSITORY_ROOT / "benchmark_cache" / "flux_1024"
LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def image_statistics(image: torch.Tensor) -> dict[str, float]:
    """
    args:
        image: [1, 3, H, W], pixel_values: [0...1]
    """
    gray = image.mean(dim=1, keepdim=True)
    laplacian = functional.conv2d(gray, LAPLACIAN.to(image)[None, None])
    saturation = image.max(dim=1).values - image.min(dim=1).values
    return {
        "laplacian_energy": laplacian.pow(2).mean().item() * 1e3,
        "saturation": saturation.mean().item(),
        "brightness": gray.mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--samples", type=int, default=32)
    arguments = parser.parse_args()

    from stdev.benchmark.cache import LatentCache
    from stdev.benchmark.dataset import BenchmarkDataset
    from stdev.vaes.flux_2_vae import Flux2VAE

    vae = Flux2VAE(
        str(REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B" / "vae"), device="cuda", dtype=torch.bfloat16
    )
    metadata = json.loads((BENCHMARK_CACHE / "meta.json").read_text())
    cache = LatentCache(BENCHMARK_CACHE, metadata=metadata, codec=metadata["codec"])
    dataset = BenchmarkDataset(BENCHMARK_DATASET)
    sample_names = dataset.sample_names[:: len(dataset.sample_names) // arguments.samples][: arguments.samples]

    rows = {}
    teacher_statistics: dict[str, float] = {}
    for sample_name in sample_names:
        for key, value in image_statistics(dataset.load_image(sample_name)).items():
            teacher_statistics[key] = teacher_statistics.get(key, 0.0) + value / len(sample_names)
        latents = cache.load_latents("hr_gt", sample_name)
        teacher_statistics["latent_std"] = teacher_statistics.get("latent_std", 0.0) + latents.std().item() / len(sample_names)
    rows["teacher"] = teacher_statistics

    for variant in arguments.variant:
        module = load_run_module(resolve_run_file(arguments.run), [item for item in variant.split(";") if item])
        upscaler = instantiate_upscaler(module)
        upscaler.prepare_to_benchmark()
        statistics: dict[str, float] = {}
        for sample_name in sample_names:
            latents_low = cache.load_latents("lr_gt", sample_name).to(torch.bfloat16)
            latents_high = cache.load_latents("hr_gt", sample_name)
            prediction = upscaler.upscale(latents_low, dataset.load_prompt_embeds(sample_name)).float()
            image = vae.decode(prediction).float().clamp(0, 1)
            values = image_statistics(image)
            values["latent_std"] = prediction.std().item()
            values["latent_mse"] = (prediction - latents_high).pow(2).mean().item()
            values["latent_lowpass_mse"] = (
                functional.avg_pool2d(prediction, 8) - functional.avg_pool2d(latents_high, 8)
            ).pow(2).mean().item()
            for key, value in values.items():
                statistics[key] = statistics.get(key, 0.0) + value / len(sample_names)
        rows[variant] = statistics

    for name, statistics in rows.items():
        print(f"{name:45s} " + " ".join(f"{key}={value:.4f}" for key, value in statistics.items()))


if __name__ == "__main__":
    main()
