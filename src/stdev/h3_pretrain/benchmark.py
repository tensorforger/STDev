"""
Visual benchmark of the H3 -> FLUX.2 translator pretraining: BENCHMARK_IMAGE_COUNT fixed Axis-V1 1024 crops (evenly
spaced over the first 1000 of the autoresearch order, same `NNNN` names as the FLUX.2 pretrain benchmark), H3 latents
from the diffusers fp32 tiled encode (as the autoresearch_6 benchmark), ground truth hff, guidance 1,
BENCHMARK_SAMPLER, decoded with the FLUX.2 small decoder. Out of the training distribution (Axis photos only).

Metrics: psnr and mse against the 1024 ground truth, low_res_mse / chroma_ciede2000 (roundtrip to the 512 bilinear
image, as autoresearch_6), lowpass_ciede2000 (both images area resized to LOW_FREQUENCY_SIDE: local colour drift),
mean_lab_shift (Euclidean distance of the image mean Lab colours: global colour cast), latent_mse. No FID.

--build   prepares the set in the data run (ground truth / low resolution PNG, latents, hff)
watcher   (default) evaluates every snapshot written by the trainer with the raw and the EMA weights, logs to
          tensorboard (benchmark_raw/*, benchmark_ema/*) and benchmarks/results.jsonl and saves all generated images
          (benchmarks/<snapshot>/<name>_<raw|ema>.png). The first run logs and saves the references: FLUX.2 small
          decoder oracle, native H3 decoder (released tiling) and the linear base alone. Exits after training is done.

Usage: CUDA_VISIBLE_DEVICES=7 python -m stdev.h3_pretrain.benchmark --run-name NAME --data-run-name DATA [--build]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from torch.utils.tensorboard import SummaryWriter

from stdev.flux_benchmark.benchmark import save_image
from stdev.h3_pretrain.config import (
    AXIS_DIRECTORY,
    BENCHMARK_BATCH,
    BENCHMARK_IMAGE_COUNT,
    BENCHMARK_SAMPLER,
    ENV_PATH,
    FLUX_SMALL_DECODER_DIRECTORY,
    FLUX_VAE_DIRECTORY,
    H3_VAE_DIRECTORY,
    LINEAR_BASE_PATH,
    LOW_FREQUENCY_SIDE,
    MODEL,
    RESIDUAL_MEAN,
    run_paths,
)
from stdev.h3_pretrain.model import H3TranslatorTransformer, translate
from stdev.hff.immerkaer import estimate_immerkaer
from stdev.pretrain.config import PretrainPaths
from stdev.upscale_benchmark.dataset import list_source_images
from stdev.upscale_benchmark.metrics import ciede2000, rgb_to_lab, roundtrip_errors
from stdev.utils.image_tools import crop_maximal_rectangle
from stdev.vaes.flux_2_vae import Flux2VAE
from stdev.vaes.minimax_h3_image_vae import MiniMaxH3ImageVAE
from stdev.video_downloader.stream_videos import load_env_file

IMAGE_SIZE = 1024
LOW_RESOLUTION_SIZE = 512
AXIS_BENCHMARK_SIZE = 1000
POLL_INTERVAL_SECONDS = 30.0
WEIGHT_VARIANTS = ("raw", "ema")
INFERENCE_DTYPE = torch.bfloat16


def load_pixels(paths: list[Path]) -> torch.Tensor:
    """
    returns:
        pixels: [B, 3, H, W], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32, on cuda
    """
    images = np.stack([cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in paths])
    return torch.from_numpy(images).cuda().permute(0, 3, 1, 2).float() / 255.0


@torch.no_grad()
def build_benchmark_set(data_paths: PretrainPaths) -> None:
    directory = data_paths.benchmark_set
    if (directory / "manifest.json").exists():
        return
    source_paths = list_source_images(AXIS_DIRECTORY)[:AXIS_BENCHMARK_SIZE]
    indices = [round(position * len(source_paths) / BENCHMARK_IMAGE_COUNT) for position in range(BENCHMARK_IMAGE_COUNT)]
    names = [f"{index:04d}" for index in indices]
    (directory / "ground_truth").mkdir(parents=True, exist_ok=True)
    (directory / "low_resolution").mkdir(parents=True, exist_ok=True)
    for index, name in zip(indices, names):
        ground_truth = crop_maximal_rectangle(cv2.imread(str(source_paths[index]), cv2.IMREAD_COLOR), IMAGE_SIZE, IMAGE_SIZE)
        low_resolution = cv2.resize(ground_truth, (LOW_RESOLUTION_SIZE, LOW_RESOLUTION_SIZE), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(directory / "ground_truth" / f"{name}.png"), ground_truth)
        cv2.imwrite(str(directory / "low_resolution" / f"{name}.png"), low_resolution)
    ground_truth_pixels = load_pixels([directory / "ground_truth" / f"{name}.png" for name in names])
    flux_vae = Flux2VAE(str(FLUX_VAE_DIRECTORY), device="cuda", dtype=torch.bfloat16)
    latents_target = torch.cat([flux_vae.encode(chunk)[:, :, 0].cpu() for chunk in ground_truth_pixels.split(BENCHMARK_BATCH)])
    del flux_vae
    h3_vae = MiniMaxH3ImageVAE(str(H3_VAE_DIRECTORY), device="cuda", dtype=torch.bfloat16, use_tiling=True)
    latents_source = torch.cat([h3_vae.encode(chunk)[:, :, 0].cpu() for chunk in ground_truth_pixels.split(BENCHMARK_BATCH)])
    torch.save(
        {
            "latents_source": latents_source.float(),
            "latents_target": latents_target.float(),
            "hff": estimate_immerkaer(ground_truth_pixels).float().cpu(),
        },
        directory / "latents.pt",
    )
    manifest = {
        "sample_names": names,
        "source_paths": [source_paths[index].relative_to(AXIS_DIRECTORY).as_posix() for index in indices],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"benchmark: built {len(names)} images", flush=True)


class BenchmarkSet:
    """
    latents_source: [N, 24, 64, 64] (H3), latents_target: [N, 32, 128, 128] (FLUX.2), hff: [N], ground truth (1024)
    and low resolution (512 bilinear) PNGs.
    """

    def __init__(self, data_paths: PretrainPaths) -> None:
        directory = data_paths.benchmark_set
        self.names: list[str] = json.loads((directory / "manifest.json").read_text())["sample_names"]
        tensors = torch.load(directory / "latents.pt")
        self.latents_source = tensors["latents_source"]
        self.latents_target = tensors["latents_target"]
        self.hff = tensors["hff"]
        self.low_resolution_paths = [directory / "low_resolution" / f"{name}.png" for name in self.names]
        self.ground_truth_paths = [directory / "ground_truth" / f"{name}.png" for name in self.names]

    def __len__(self) -> int:
        return len(self.names)


def quantize(pixels: torch.Tensor) -> torch.Tensor:
    """
    args:
        pixels: [B, 3, H, W]
    returns:
        pixels: [B, 3, H, W], pixel_values: [0...1], dtype: torch.float32, quantized like a saved PNG
    """
    return pixels.float().clamp(0, 1).mul(255).round().div(255)


def image_errors(predicted_pixels: torch.Tensor, ground_truth_pixels: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    args:
        predicted_pixels: [B, 3, H, W], pixel_values: [0...1]
        ground_truth_pixels: [B, 3, H, W], pixel_values: [0...1]
    returns:
        per image values [B]: mse, psnr, lowpass_ciede2000, mean_lab_shift
    """
    mse = (predicted_pixels.double() - ground_truth_pixels.double()).square().mean(dim=(1, 2, 3))
    size = (LOW_FREQUENCY_SIDE, LOW_FREQUENCY_SIDE)
    predicted_lab = rgb_to_lab(functional.interpolate(predicted_pixels.double(), size=size, mode="area"))
    ground_truth_lab = rgb_to_lab(functional.interpolate(ground_truth_pixels.double(), size=size, mode="area"))
    return {
        "mse": mse,
        "psnr": 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12)),
        "lowpass_ciede2000": ciede2000(predicted_lab, ground_truth_lab).mean(dim=(1, 2)),
        "mean_lab_shift": (predicted_lab.mean(dim=(2, 3)) - ground_truth_lab.mean(dim=(2, 3))).norm(dim=1),
    }


@torch.no_grad()
def evaluate(
    produce_pixels: Callable[[slice], torch.Tensor],
    benchmark_set: BenchmarkSet,
    output_directory: Path | None,
    tag: str,
    produce_latents: Callable[[slice], torch.Tensor] | None = None,
    decoder: Flux2VAE | None = None,
) -> dict[str, float]:
    """
    args:
        produce_pixels: function(slice) -> pixels [B, 3, 1024, 1024] in [0...1] (unused with produce_latents)
        produce_latents: optional function(slice) -> normalized FLUX.2 latents [B, 32, 128, 128], decoded by decoder
    returns:
        mean metrics over the set
    """
    values: dict[str, list[float]] = {}
    for start in range(0, len(benchmark_set), BENCHMARK_BATCH):
        batch = slice(start, min(start + BENCHMARK_BATCH, len(benchmark_set)))
        if produce_latents is not None:
            latents = produce_latents(batch)
            latent_errors = (latents.cpu() - benchmark_set.latents_target[batch]).square().mean(dim=(1, 2, 3))
            values.setdefault("latent_mse", []).extend(latent_errors.tolist())
            pixels = quantize(decoder.decode(latents.cuda()[:, :, None]))
        else:
            pixels = quantize(produce_pixels(batch))
        low_res_mse, chroma_difference = roundtrip_errors(pixels, load_pixels(benchmark_set.low_resolution_paths[batch]))
        errors = image_errors(pixels, load_pixels(benchmark_set.ground_truth_paths[batch]))
        errors["low_res_mse"] = low_res_mse
        errors["chroma_ciede2000"] = chroma_difference
        for name, per_image in errors.items():
            values.setdefault(name, []).extend(per_image.tolist())
        if output_directory is not None:
            for offset, name in enumerate(benchmark_set.names[batch]):
                save_image(pixels[offset], output_directory / f"{name}_{tag}.png")
    return {name: float(np.mean(per_image)) for name, per_image in values.items()}


def build_model() -> H3TranslatorTransformer:
    """
    Inference model in INFERENCE_DTYPE with the linear base and residual statistics kept in float32.
    """
    model = H3TranslatorTransformer(MODEL, RESIDUAL_MEAN, LINEAR_BASE_PATH).to("cuda", INFERENCE_DTYPE)
    model.linear_base.float()
    model.residual_mean = model.residual_mean.float()
    model.residual_std = model.residual_std.float()
    return model.eval().requires_grad_(False)


def evaluate_h3_native(benchmark_set: BenchmarkSet, output_directory: Path) -> dict[str, float]:
    """
    The native H3-Image decoder (released tiling, bf16) on the H3 latents: the baseline to beat.
    """
    h3_vae = MiniMaxH3ImageVAE(str(H3_VAE_DIRECTORY), device="cuda", dtype=torch.bfloat16, use_tiling=True)

    def decode(batch: slice) -> torch.Tensor:
        return h3_vae.decode(benchmark_set.latents_source[batch].cuda()[:, :, None]).float()

    return evaluate(decode, benchmark_set, output_directory, "h3")


def log_references(paths: PretrainPaths, decoder: Flux2VAE, benchmark_set: BenchmarkSet, summary_writer: SummaryWriter) -> None:
    references_path = paths.benchmarks / "references.json"
    if references_path.exists():
        return
    references_directory = paths.benchmarks / "references"
    references_directory.mkdir(parents=True, exist_ok=True)
    references = {
        "flux2_small_oracle": evaluate(
            None, benchmark_set, references_directory, "oracle", lambda batch: benchmark_set.latents_target[batch], decoder
        ),
    }
    linear_base = build_model().linear_base
    references["linear_base"] = evaluate(
        None, benchmark_set, references_directory, "linear_base", lambda batch: linear_base(benchmark_set.latents_source[batch].cuda()), decoder
    )
    references["h3_native"] = evaluate_h3_native(benchmark_set, references_directory)
    torch.cuda.empty_cache()
    for name, metrics in references.items():
        for metric, value in metrics.items():
            summary_writer.add_text(f"references/{name}/{metric}", f"{value:.6f}", 0)
    references_path.write_text(json.dumps(references, indent=1))
    print(f"benchmark: references {json.dumps(references)}", flush=True)


def snapshot_weights(tensors: dict[str, torch.Tensor], variant: str) -> dict[str, torch.Tensor]:
    prefix = f"{variant}."
    return {name.removeprefix(prefix): tensor for name, tensor in tensors.items() if name.startswith(prefix)}


def run_watcher(paths: PretrainPaths, data_paths: PretrainPaths) -> None:
    torch.backends.cudnn.benchmark = True
    benchmark_set = BenchmarkSet(data_paths)
    decoder = Flux2VAE(str(FLUX_SMALL_DECODER_DIRECTORY), device="cuda", dtype=torch.bfloat16)
    model = build_model()
    buffer_names = {name for name, _ in model.named_buffers()}
    summary_writer = SummaryWriter(str(paths.tensorboard / "benchmark"))
    paths.benchmarks.mkdir(parents=True, exist_ok=True)
    log_references(paths, decoder, benchmark_set, summary_writer)
    print(f"benchmark: {len(benchmark_set)} images ready", flush=True)

    def generated(batch: slice) -> torch.Tensor:
        return translate(model, benchmark_set.latents_source[batch].cuda(), benchmark_set.hff[batch].cuda(), BENCHMARK_SAMPLER, INFERENCE_DTYPE)

    while True:
        snapshot_paths = sorted(paths.snapshots.glob("*.safetensors"), key=lambda path: path.stat().st_mtime)
        if not snapshot_paths:
            if paths.training_done.exists():
                break
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        snapshot_path = snapshot_paths[0]
        snapshot_info = json.loads(snapshot_path.with_suffix(".json").read_text())
        step = snapshot_info["step"]
        output_directory = paths.benchmarks / snapshot_path.stem
        output_directory.mkdir(parents=True, exist_ok=True)
        tensors = load_file(str(snapshot_path))
        start_time = time.perf_counter()
        results: dict[str, object] = dict(snapshot_info)
        for variant in WEIGHT_VARIANTS:
            missing, unexpected = model.load_state_dict(snapshot_weights(tensors, variant), strict=False)
            if unexpected or set(missing) - buffer_names:
                raise RuntimeError(f"snapshot mismatch: missing {missing}, unexpected {unexpected}")
            metrics = evaluate(None, benchmark_set, output_directory, variant, generated, decoder)
            for name, value in metrics.items():
                summary_writer.add_scalar(f"benchmark_{variant}/{name}", value, step)
                results[f"{variant}_{name}"] = value
        summary_writer.add_scalar("benchmark/units_seen", snapshot_info["units_seen"], step)
        summary_writer.add_scalar("benchmark/training_hours", snapshot_info["training_seconds"] / 3600, step)
        summary_writer.flush()
        results["benchmark_seconds"] = time.perf_counter() - start_time
        with (paths.benchmarks / "results.jsonl").open("a") as results_file:
            results_file.write(json.dumps(results) + "\n")
        print(f"benchmark: {json.dumps(results)}", flush=True)
        if snapshot_info["tag"] == "final":
            snapshot_path.rename(snapshot_path.with_suffix(".done"))
        else:
            snapshot_path.unlink()
            snapshot_path.with_suffix(".json").unlink()
    summary_writer.close()


def main() -> None:
    load_env_file(ENV_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-run-name", required=True)
    parser.add_argument("--build", action="store_true")
    arguments = parser.parse_args()
    data_paths = run_paths(arguments.data_run_name)
    data_paths.create()
    if arguments.build:
        build_benchmark_set(data_paths)
        return
    paths = run_paths(arguments.run_name)
    paths.create()
    run_watcher(paths, data_paths)


if __name__ == "__main__":
    main()
