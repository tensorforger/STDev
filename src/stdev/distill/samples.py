"""
Sample images of the distillation (no metrics). The fixed sample set is SAMPLE_IMAGE_COUNT crops
of the 4KLSDB validation split (one per image, deterministic). For a weights file it upscales the
set with full conditioning and fixed noise per image at SAMPLE_STEPS uniform Euler steps and at the
teacher schedule (TEACHER_STEPS at TEACHER_SIGMA_SHIFT), and saves every image plus one strip per
image: low resolution (bicubic) | steps ... | teacher schedule | ground truth.

Usage:
    python -m stdev.distill.samples --run-name NAME --build-set
    python -m stdev.distill.samples --run-name NAME --weights PATH --output-name NAME
"""

import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as parquet
import torch
from safetensors.torch import load_file, save_file

from stdev.distill.config import (
    CAPTION_COLUMN,
    SAMPLE_IMAGE_COUNT,
    SAMPLE_SEED,
    SAMPLE_SET_FILE,
    SAMPLE_STEPS,
    SAMPLE_STRIP_TILE,
    TEACHER_SIGMA_SHIFT,
    TEACHER_STEPS,
    run_paths,
)
from stdev.distill.data import ensure_downloaded, make_noise, preprocess_image
from stdev.pretrain.benchmark import decode_in_chunks
from stdev.pretrain.config import SDXL_DIRECTORY, VAE_DIRECTORY
from stdev.pretrain.high_frequency import QuantileNormalizer
from stdev.upscalers.pretrain_large import (
    INFERENCE_DTYPE,
    SCALE_FACTOR,
    UpscalerUNet,
    sample_high_resolution_latents,
)
from stdev.vaes.flux_2_vae import Flux2VAE

BATCH_SIZE = 8


def build_sample_set(run_name: str) -> None:
    from stdev.pretrain.producer import encode_in_chunks
    from stdev.text_encoders.sdxl_text_encoder import SDXLTextEncoder

    paths = run_paths(run_name)
    device = torch.device("cuda")
    parquet_file = parquet.ParquetFile(ensure_downloaded(SAMPLE_SET_FILE))
    table = parquet_file.read_row_group(0, columns=["hr", CAPTION_COLUMN])
    images = table.column("hr").to_pylist()
    captions = table.column(CAPTION_COLUMN).to_pylist()
    crops, crop_captions = [], []
    for index in itertools.count():
        if len(crops) == SAMPLE_IMAGE_COUNT:
            break
        crop = preprocess_image(images[index]["bytes"], f"sample-{index}", crop_count=1)
        if crop:
            crops.append(crop[0])
            crop_captions.append(captions[index] or "")

    vae = Flux2VAE(str(VAE_DIRECTORY), device=device, dtype=torch.bfloat16)
    text_encoder = SDXLTextEncoder(str(SDXL_DIRECTORY), device=device, dtype=torch.float16)
    normalizer = QuantileNormalizer.load(paths.root / "high_frequency_quantiles.json")
    with torch.no_grad():
        latents_low = encode_in_chunks(vae, [crop["image_low"] for crop in crops], device)
        text_embedding = text_encoder.encode_text_batch(crop_captions).float().mean(dim=1)
    features = normalizer.normalize(np.stack([crop["high_frequency_raw"] for crop in crops]))
    for index, crop in enumerate(crops):
        for kind in ("high", "low"):
            cv2.imwrite(
                str(paths.sample_set / f"{index:02d}_{kind}.png"),
                cv2.cvtColor(crop[f"image_{kind}"], cv2.COLOR_RGB2BGR),
            )
    (paths.sample_set / "captions.txt").write_text("\n".join(crop_captions))
    save_file(
        {
            "latents_low": latents_low.float().cpu().contiguous(),
            "text_embedding": text_embedding.cpu().contiguous(),
            "high_frequency_features": torch.from_numpy(features).float().contiguous(),
        },
        str(paths.sample_set / "sample_set.safetensors"),
    )
    print(f"samples: sample set of {len(crops)} crops built", flush=True)


def to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    args:
        tensor: [3, H, W], channel_order: RGB, pixel_values: [0...1]
    returns:
        image: [H, W, 3], channel_order: BGR, dtype: uint8
    """
    image = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def render_samples(run_name: str, weights_path: Path, output_name: str) -> None:
    paths = run_paths(run_name)
    device = torch.device("cuda")
    sample_set = load_file(str(paths.sample_set / "sample_set.safetensors"))
    vae = Flux2VAE(str(VAE_DIRECTORY), device=device, dtype=torch.bfloat16)
    model = UpscalerUNet()
    model.load_state_dict(load_file(str(weights_path)))
    model = model.to(device, INFERENCE_DTYPE).eval().requires_grad_(False)
    output_directory = paths.samples / output_name
    output_directory.mkdir(parents=True, exist_ok=True)

    schedules = [(f"steps{steps:02d}", steps, 1.0) for steps in SAMPLE_STEPS]
    schedules.append((f"teacher{TEACHER_STEPS:02d}", TEACHER_STEPS, TEACHER_SIGMA_SHIFT))
    count = sample_set["latents_low"].shape[0]
    columns: dict[str, list[np.ndarray]] = {}
    for name, steps, shift in schedules:
        columns[name] = []
        for start in range(0, count, BATCH_SIZE):
            latents_low = sample_set["latents_low"][start : start + BATCH_SIZE].to(device)
            features = sample_set["high_frequency_features"][start : start + BATCH_SIZE].to(device)
            height, width = (size * SCALE_FACTOR for size in latents_low.shape[-2:])
            noise = torch.stack(
                [make_noise(SAMPLE_SEED + index, height, width) for index in range(start, start + len(latents_low))]
            )
            with torch.no_grad():
                latents = sample_high_resolution_latents(
                    model,
                    latents_low,
                    sample_set["text_embedding"][start : start + BATCH_SIZE].to(device),
                    torch.ones(len(latents_low), dtype=torch.bool, device=device),
                    features,
                    ~torch.isnan(features),
                    num_steps=steps,
                    sigma_shift=shift,
                    noise=noise,
                )
                images = decode_in_chunks(vae, latents)
            for offset, image in enumerate(images):
                image = to_image(image)
                cv2.imwrite(str(output_directory / f"{start + offset:02d}_{name}.png"), image)
                columns[name].append(image)

    tile = (SAMPLE_STRIP_TILE, SAMPLE_STRIP_TILE)
    for index in range(count):
        low = cv2.imread(str(paths.sample_set / f"{index:02d}_low.png"))
        high = cv2.imread(str(paths.sample_set / f"{index:02d}_high.png"))
        strip = [cv2.resize(low, tile, interpolation=cv2.INTER_CUBIC)]
        strip += [cv2.resize(columns[name][index], tile, interpolation=cv2.INTER_AREA) for name, _, _ in schedules]
        strip.append(cv2.resize(high, tile, interpolation=cv2.INTER_AREA))
        cv2.imwrite(str(output_directory / f"strip_{index:02d}.jpg"), np.hstack(strip), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"samples: {output_name} saved ({', '.join(name for name, _, _ in schedules)})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--build-set", action="store_true")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-name")
    arguments = parser.parse_args()
    if arguments.build_set:
        build_sample_set(arguments.run_name)
    else:
        render_samples(arguments.run_name, arguments.weights, arguments.output_name)


if __name__ == "__main__":
    main()
