"""
Upscales a few benchmark samples with several constant overrides of one run and saves zoomed
native resolution crops next to the teacher image.

Usage: python dev/sample_variants.py OUTPUT.png RUN --variant "NAME=EXPR;NAME=EXPR" ...
           [--samples 3] [--crop 128] [--zoom 2] [--weights-directory DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tools import REPOSITORY_ROOT, instantiate_upscaler, load_run_module, resolve_run_file

BENCHMARK_DATASET = REPOSITORY_ROOT / "benchmark_datasets" / "flux_1024"
BENCHMARK_CACHE = REPOSITORY_ROOT / "benchmark_cache" / "flux_1024"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("run")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--first-sample", type=int, default=0)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--zoom", type=int, default=2)
    parser.add_argument("--weights-directory", type=Path)
    arguments = parser.parse_args()

    from stdev.benchmark.cache import LatentCache
    from stdev.benchmark.dataset import BenchmarkDataset
    from stdev.utils.image_tools import torch_to_cv2
    from stdev.vaes.flux_2_vae import Flux2VAE

    vae = Flux2VAE(
        str(REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B" / "vae"), device="cuda", dtype=torch.bfloat16
    )
    metadata = json.loads((BENCHMARK_CACHE / "meta.json").read_text())
    cache = LatentCache(BENCHMARK_CACHE, metadata=metadata, codec=metadata["codec"])
    dataset = BenchmarkDataset(BENCHMARK_DATASET)
    sample_names = dataset.sample_names[arguments.first_sample :: 97][: arguments.samples]

    variants = arguments.variant or [""]
    columns: list[list[np.ndarray]] = [[] for _ in sample_names]
    for sample_index, sample_name in enumerate(sample_names):
        teacher = torch_to_cv2(dataset.load_image(sample_name))
        columns[sample_index].append(teacher)
    for variant in variants:
        overrides = [item for item in variant.split(";") if item]
        if arguments.weights_directory is not None:
            overrides.append(
                f"OUTPUT_DIRECTORY=__import__('pathlib').Path({str(arguments.weights_directory)!r})"
            )
        module = load_run_module(resolve_run_file(arguments.run), overrides)
        upscaler = instantiate_upscaler(module)
        upscaler.prepare_to_benchmark()
        for sample_index, sample_name in enumerate(sample_names):
            latents = cache.load_latents("lr_gt", sample_name).to(torch.bfloat16)
            prediction = upscaler.upscale(latents, dataset.load_prompt_embeds(sample_name)).float()
            image = vae.decode(prediction).float().clamp(0, 1)
            columns[sample_index].append(torch_to_cv2(image))

    rows = []
    for images in columns:
        height, width = images[0].shape[:2]
        top, left = height // 2 - arguments.crop // 2, width // 3 - arguments.crop // 2
        crops = [
            cv2.resize(
                image[top : top + arguments.crop, left : left + arguments.crop],
                None,
                fx=arguments.zoom,
                fy=arguments.zoom,
                interpolation=cv2.INTER_NEAREST,
            )
            for image in images
        ]
        rows.append(np.concatenate(crops, axis=1))
    montage = np.concatenate(rows, axis=0)
    size = arguments.crop * arguments.zoom
    for index, label in enumerate(["teacher"] + variants):
        cv2.putText(montage, label[:40], (index * size + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(arguments.output), montage)


if __name__ == "__main__":
    main()
