"""
Relative velocity change of a FLUX.2 pretrain checkpoint between hff conditions (ground truth vs null, 0.9 vs 0.1)
on visual benchmark images at several noise levels (CPU, fp32).
Usage: python scripts/flux_tools/flux_hff_sensitivity.py DATA_RUN CHECKPOINT [--weights ema] [--device cpu]
"""

import argparse
from pathlib import Path

import torch

from stdev.flux_pretrain.config import run_paths
from stdev.flux_pretrain.model import FluxUpscalerTransformer, ModelConfig, to_residual

SIGMAS = (0.1, 0.5, 0.9)
NUM_IMAGES = 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_run")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    torch.set_num_threads(64)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**{**checkpoint["model_config"], "residual_std": tuple(checkpoint["model_config"]["residual_std"])})
    model = FluxUpscalerTransformer(config)
    model.load_state_dict(checkpoint["model"])
    if arguments.weights == "ema":
        with torch.no_grad():
            for parameter, ema_parameter in zip(model.parameters(), checkpoint["ema"]):
                parameter.copy_(ema_parameter)
    model = model.to(arguments.device).eval().requires_grad_(False)
    tensors = torch.load(run_paths(arguments.data_run).benchmark_set / "latents.pt")
    latents_source = tensors["latents_source"][:NUM_IMAGES].to(arguments.device)
    latents_target = tensors["latents_target"][:NUM_IMAGES].to(arguments.device)
    hff_truth = tensors["hff"][:NUM_IMAGES].to(arguments.device)
    print(f"ground truth hff {hff_truth.tolist()}")
    residual = to_residual(latents_target, latents_source, config.residual_std)
    noise = torch.randn(residual.shape, generator=torch.Generator().manual_seed(0)).to(arguments.device)
    ones = torch.ones(NUM_IMAGES, dtype=torch.bool, device=arguments.device)
    with torch.no_grad():
        for sigma in SIGMAS:
            sigmas = torch.full((NUM_IMAGES,), sigma, device=arguments.device)
            noisy = (1 - sigma) * residual + sigma * noise
            velocity = {
                "truth": model(noisy, sigmas, latents_source, hff_truth, ones),
                "null": model(noisy, sigmas, latents_source, hff_truth, ~ones),
                "low": model(noisy, sigmas, latents_source, torch.full_like(hff_truth, 0.1), ones),
                "high": model(noisy, sigmas, latents_source, torch.full_like(hff_truth, 0.9), ones),
            }
            norm = velocity["truth"].norm()
            print(
                f"sigma {sigma}: |v(truth) - v(null)| / |v| {((velocity['truth'] - velocity['null']).norm() / norm):.4f}, "
                f"|v(0.9) - v(0.1)| / |v| {((velocity['high'] - velocity['low']).norm() / norm):.4f}"
            )


if __name__ == "__main__":
    main()
