"""
Visual benchmark of the FLUX.2 pretraining: BENCHMARK_IMAGE_COUNT fixed Axis-V1 images (evenly spaced over the
first 1000 of the autoresearch_3 benchmark order, same `NNNN` names), 512 -> 1024 with the bilinear input of the
benchmark, ground truth hff, guidance 1, BENCHMARK_SAMPLER. Metrics: low_res_mse and chroma_ciede2000 (roundtrip
of the prediction against the 512 input, as in autoresearch_3) and latent_mse_hr; no FID (set too small).

--build   prepares the set in the data run (ground truth / low resolution PNG, latents, hff)
watcher   (default) evaluates every snapshot written by the trainer with the raw and the EMA weights, logs to
          tensorboard (benchmark_raw/*, benchmark_ema/*) and benchmarks/results.jsonl and saves all generated
          images (benchmarks/<snapshot>/<name>_<raw|ema>.png). The first run logs the VAE oracle and pixel
          bicubic references and saves the bicubic inputs. Exits after training is done.

Usage: CUDA_VISIBLE_DEVICES=7 python -m stdev.flux_pretrain.benchmark --run-name NAME --data-run-name DATA [--build]
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from torch.utils.tensorboard import SummaryWriter

from stdev.flux_benchmark.benchmark import save_image
from stdev.flux_pretrain.config import (
    AXIS_DIRECTORY,
    BENCHMARK_BATCH,
    BENCHMARK_IMAGE_COUNT,
    BENCHMARK_SAMPLER,
    ENV_PATH,
    MODEL,
    VAE_DIRECTORY,
    run_paths,
)
from stdev.flux_pretrain.model import FluxUpscalerTransformer, upscale
from stdev.hff.immerkaer import estimate_immerkaer
from stdev.pretrain.config import PretrainPaths
from stdev.upscale_benchmark.dataset import list_source_images
from stdev.upscale_benchmark.metrics import roundtrip_errors
from stdev.utils.image_tools import crop_maximal_rectangle
from stdev.vaes.flux_2_vae import Flux2VAE
from stdev.video_downloader.stream_videos import load_env_file

HIGH_RESOLUTION_SIZE = 1024
LOW_RESOLUTION_SIZE = 512
AXIS_BENCHMARK_SIZE = 1000
POLL_INTERVAL_SECONDS = 30.0
WEIGHT_VARIANTS = ("raw", "ema")
INFERENCE_DTYPE = torch.bfloat16


def benchmark_directory(data_paths: PretrainPaths) -> Path:
    return data_paths.benchmark_set


def load_pixels(paths: list[Path]) -> torch.Tensor:
    """
    returns:
        pixels: [B, 3, H, W], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32, on cuda
    """
    images = np.stack([cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in paths])
    return torch.from_numpy(images).cuda().permute(0, 3, 1, 2).float() / 255.0


@torch.no_grad()
def build_benchmark_set(data_paths: PretrainPaths) -> None:
    directory = benchmark_directory(data_paths)
    if (directory / "manifest.json").exists():
        return
    source_paths = list_source_images(AXIS_DIRECTORY)[:AXIS_BENCHMARK_SIZE]
    indices = [round(position * len(source_paths) / BENCHMARK_IMAGE_COUNT) for position in range(BENCHMARK_IMAGE_COUNT)]
    names = [f"{index:04d}" for index in indices]
    (directory / "ground_truth").mkdir(parents=True, exist_ok=True)
    (directory / "low_resolution").mkdir(parents=True, exist_ok=True)
    for index, name in zip(indices, names):
        image = cv2.imread(str(source_paths[index]), cv2.IMREAD_COLOR)
        ground_truth = crop_maximal_rectangle(image, HIGH_RESOLUTION_SIZE, HIGH_RESOLUTION_SIZE)
        low_resolution = cv2.resize(ground_truth, (LOW_RESOLUTION_SIZE, LOW_RESOLUTION_SIZE), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(directory / "ground_truth" / f"{name}.png"), ground_truth)
        cv2.imwrite(str(directory / "low_resolution" / f"{name}.png"), low_resolution)
    vae = Flux2VAE(str(VAE_DIRECTORY), device="cuda", dtype=torch.bfloat16)
    ground_truth_pixels = load_pixels([directory / "ground_truth" / f"{name}.png" for name in names])
    low_resolution_pixels = load_pixels([directory / "low_resolution" / f"{name}.png" for name in names])
    latents_target = torch.cat([vae.encode(chunk)[:, :, 0].cpu() for chunk in ground_truth_pixels.split(BENCHMARK_BATCH)])
    latents_source = torch.cat([vae.encode(chunk)[:, :, 0].cpu() for chunk in low_resolution_pixels.split(BENCHMARK_BATCH)])
    torch.save(
        {"latents_source": latents_source, "latents_target": latents_target, "hff": estimate_immerkaer(ground_truth_pixels).float().cpu()},
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
    latents_source: [N, 32, 64, 64], latents_target: [N, 32, 128, 128], hff: [N], ground truth / low resolution PNGs.
    """

    def __init__(self, data_paths: PretrainPaths) -> None:
        directory = benchmark_directory(data_paths)
        self.names: list[str] = json.loads((directory / "manifest.json").read_text())["sample_names"]
        tensors = torch.load(directory / "latents.pt")
        self.latents_source = tensors["latents_source"]
        self.latents_target = tensors["latents_target"]
        self.hff = tensors["hff"]
        self.low_resolution_paths = [directory / "low_resolution" / f"{name}.png" for name in self.names]
        self.ground_truth_paths = [directory / "ground_truth" / f"{name}.png" for name in self.names]

    def __len__(self) -> int:
        return len(self.names)


@torch.no_grad()
def decode(vae: Flux2VAE, latents: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents: [B, 32, h, w]
    returns:
        pixels: [B, 3, 8h, 8w], pixel_values: [0...1], dtype: torch.float32, quantized like a saved PNG
    """
    pixels = vae.decode(latents.cuda()[:, :, None]).float()
    return pixels.clamp(0, 1).mul(255).round().div(255)


@torch.no_grad()
def evaluate(
    produce_latents,
    vae: Flux2VAE,
    benchmark_set: BenchmarkSet,
    output_directory: Path | None,
    tag: str,
    produce_pixels=None,
) -> dict[str, float]:
    """
    args:
        produce_latents: function(slice) -> latents [B, 32, 128, 128] (or None with produce_pixels)
        produce_pixels: optional function(slice) -> pixels [B, 3, 1024, 1024] for pixel space references
    returns:
        low_res_mse, chroma_ciede2000, latent_mse_hr (only for latent predictions)
    """
    mse_values: list[float] = []
    color_values: list[float] = []
    latent_errors: list[float] = []
    for start in range(0, len(benchmark_set), BENCHMARK_BATCH):
        batch = slice(start, min(start + BENCHMARK_BATCH, len(benchmark_set)))
        if produce_pixels is None:
            latents = produce_latents(batch)
            latent_errors.extend((latents.cpu() - benchmark_set.latents_target[batch]).square().mean(dim=(1, 2, 3)).tolist())
            pixels = decode(vae, latents)
        else:
            pixels = produce_pixels(batch)
        mse, color_difference = roundtrip_errors(pixels, load_pixels(benchmark_set.low_resolution_paths[batch]))
        mse_values.extend(mse.tolist())
        color_values.extend(color_difference.tolist())
        if output_directory is not None:
            for offset, name in enumerate(benchmark_set.names[batch]):
                save_image(pixels[offset], output_directory / f"{name}_{tag}.png")
    metrics = {"low_res_mse": float(np.mean(mse_values)), "chroma_ciede2000": float(np.mean(color_values))}
    if latent_errors:
        metrics["latent_mse_hr"] = float(np.mean(latent_errors))
    return metrics


def log_references(paths: PretrainPaths, vae: Flux2VAE, benchmark_set: BenchmarkSet, summary_writer: SummaryWriter) -> None:
    references_path = paths.benchmarks / "references.json"
    if references_path.exists():
        return

    def bicubic(batch: slice) -> torch.Tensor:
        low = load_pixels(benchmark_set.low_resolution_paths[batch])
        pixels = functional.interpolate(low, scale_factor=2, mode="bicubic", align_corners=False)
        return pixels.clamp(0, 1).mul(255).round().div(255)

    inputs_directory = paths.benchmarks / "inputs"
    inputs_directory.mkdir(parents=True, exist_ok=True)
    references = {
        "vae_oracle": evaluate(lambda batch: benchmark_set.latents_target[batch], vae, benchmark_set, None, "oracle"),
        "pixel_bicubic": evaluate(None, vae, benchmark_set, inputs_directory, "bicubic", produce_pixels=bicubic),
    }
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
    vae = Flux2VAE(str(VAE_DIRECTORY), device="cuda", dtype=torch.bfloat16)
    model = FluxUpscalerTransformer(MODEL).to("cuda", INFERENCE_DTYPE).eval().requires_grad_(False)
    summary_writer = SummaryWriter(str(paths.tensorboard / "benchmark"))
    paths.benchmarks.mkdir(parents=True, exist_ok=True)
    log_references(paths, vae, benchmark_set, summary_writer)
    print(f"benchmark: {len(benchmark_set)} images ready", flush=True)

    def generated(batch: slice) -> torch.Tensor:
        return upscale(model, benchmark_set.latents_source[batch].cuda(), benchmark_set.hff[batch].cuda(), BENCHMARK_SAMPLER, INFERENCE_DTYPE)

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
            model.load_state_dict(snapshot_weights(tensors, variant))
            metrics = evaluate(generated, vae, benchmark_set, output_directory, variant)
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
