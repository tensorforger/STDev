"""
1-step evaluation of FLUX.2 student snapshots on the autoresearch_3 benchmark (axis_v1_1k_flux2_2x, 1000 images):
patch_fid, fid, low_res_mse, chroma_ciede2000, latent_mse_hr; ground truth hff, a fixed noise seed per image.
Results: <run>/evaluations/<snapshot>_<weights>_cfg<guidance>/metrics.json (+ first 16 images).

Usage: CUDA_VISIBLE_DEVICES=0 python -m stdev.flux_distill.evaluation --snapshot PATH --weights ema --guidance 1.0
           [--model large|small]
"""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from stdev.flux_benchmark.benchmark import FluxBenchmarkDataset, build_fid_evaluator, evaluate_quality
from stdev.flux_distill.small_config import SMALL_MODEL
from stdev.flux_distill.student import FluxStudentTransformer, seeded_noise, upscale_single_step
from stdev.flux_pretrain.config import MODEL, REPOSITORY_ROOT, VAE_DIRECTORY
from stdev.flux_pretrain.model import LATENT_CHANNELS
from stdev.vaes.flux_2_vae import Flux2VAE

DATASET_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "axis_v1_1k_flux2_2x"
STATISTICS_DIRECTORY = REPOSITORY_ROOT / "benchmark_cache" / "axis_v1_1k_flux2_2x"
SAVED_IMAGES = 16
BATCH_SIZE = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--hff", default="gt", help="gt, null or a value")
    parser.add_argument("--model", choices=("large", "small"), default="large")
    arguments = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    tensors = load_file(str(arguments.snapshot))
    prefix = f"{arguments.weights}."
    model = FluxStudentTransformer(SMALL_MODEL if arguments.model == "small" else MODEL)
    model.load_state_dict({name.removeprefix(prefix): tensor for name, tensor in tensors.items() if name.startswith(prefix)})
    model = model.to("cuda", torch.bfloat16).eval().requires_grad_(False)
    vae = Flux2VAE(str(VAE_DIRECTORY), device="cuda", dtype=torch.bfloat16)
    dataset = FluxBenchmarkDataset(DATASET_DIRECTORY)
    hff_values = torch.load(DATASET_DIRECTORY / "hff.pt").float()
    latent_errors: list[float] = []

    def predictor(latents: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if arguments.hff == "gt":
            hff = hff_values[indices]
        elif arguments.hff == "null":
            hff = torch.full((len(indices),), float("nan"))
        else:
            hff = torch.full((len(indices),), float(arguments.hff))
        noise = seeded_noise(indices.long(), (LATENT_CHANNELS, 2 * latents.shape[2], 2 * latents.shape[3]), latents.device)
        upscaled = upscale_single_step(model, latents, hff.cuda(), torch.full((len(indices),), arguments.guidance, device="cuda"), noise)
        latent_errors.extend((upscaled - dataset.latents_high[indices].cuda()).square().mean(dim=(1, 2, 3)).tolist())
        return vae.decode(upscaled.unsqueeze(2))

    name = f"{arguments.snapshot.stem}_{arguments.weights}_cfg{arguments.guidance:g}" + ("" if arguments.hff == "gt" else f"_hff_{arguments.hff}")
    output_directory = arguments.snapshot.parent.parent / "evaluations" / name
    output_directory.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    quality = evaluate_quality(predictor, dataset, build_fid_evaluator(dataset, STATISTICS_DIRECTORY), output_directory / "images", SAVED_IMAGES)
    quality["latent_mse_hr"] = sum(latent_errors) / len(latent_errors)
    info = json.loads(arguments.snapshot.with_suffix(".json").read_text())
    metrics = {"snapshot": info, "settings": {key: str(value) for key, value in vars(arguments).items()}, "quality": quality, "seconds": time.perf_counter() - start_time}
    (output_directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
