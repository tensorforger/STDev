"""
Benchmark watcher of the pretraining pipeline. Runs on the encoder GPU: for every snapshot saved
by the trainer it pauses the encoder, upscales the fixed held-out benchmark set with the EMA and
the raw weights (with text and high frequency features), and logs fid_patch, fid and mse to
tensorboard and benchmarks/results.jsonl. Saves BENCHMARK_SAVED_PAIRS (low resolution input,
generated) image pairs per benchmark. Exits when training is done and no snapshot is left.

Usage: CUDA_VISIBLE_DEVICES=7 python -m stdev.pretrain.benchmark --run-name NAME
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

from stdev.pretrain.cache import list_shards, read_shard
from stdev.pretrain.config import (
    BENCHMARK_BATCH_SIZE,
    BENCHMARK_SAVED_PAIRS,
    VAE_DIRECTORY,
    PretrainPaths,
    run_paths,
)
from stdev.pretrain.high_frequency import QuantileNormalizer
from stdev.upscalers.pretrain_large import (
    INFERENCE_DTYPE,
    UpscalerUNet,
    sample_high_resolution_latents,
)
from stdev.utils.fid import FIDEvaluator, FIDReference
from stdev.utils.image_tools import torch_to_cv2
from stdev.vaes.flux_2_vae import Flux2VAE

FID_PATCH_SIZE = 256
POLL_INTERVAL_SECONDS = 10.0
WEIGHT_VARIANTS = ("ema", "raw")


class BenchmarkSet:
    """
    Held-out samples in production order: latents, text embeddings, normalized high frequency
    features and the low / high resolution PNG files.
    """

    def __init__(self, paths: PretrainPaths) -> None:
        shards = [read_shard(path) for path in list_shards(paths.benchmark_set)]
        self.latents_low = torch.cat([shard.latents_low for shard in shards])
        self.text_embedding = torch.cat([shard.text_embedding for shard in shards])
        raw_features = torch.cat([shard.high_frequency_raw for shard in shards]).numpy()
        normalizer = QuantileNormalizer.load(paths.high_frequency_quantiles)
        self.high_frequency_features = torch.from_numpy(normalizer.normalize(raw_features))
        self.image_high_directory = paths.benchmark_set / "images_high"
        self.image_low_directory = paths.benchmark_set / "images_low"
        self.file_names = [f"{index:05d}.png" for index in range(len(self.latents_low))]
        self.high_resolution = cv2.imread(str(self.image_high_directory / self.file_names[0])).shape[0]

    def __len__(self) -> int:
        return len(self.file_names)

    def load_low_resolution_images(self, indices: range) -> torch.Tensor:
        """
        returns:
            images: [B, 3, H_lr, W_lr], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32
        """
        images = [
            cv2.cvtColor(cv2.imread(str(self.image_low_directory / self.file_names[index])), cv2.COLOR_BGR2RGB)
            for index in indices
        ]
        return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div(255.0)


def build_fid_evaluator(paths: PretrainPaths, benchmark_set: BenchmarkSet) -> FIDEvaluator:
    image_directory = str(benchmark_set.image_high_directory)
    return FIDEvaluator(
        references=[
            FIDReference(
                name="fid",
                image_dir=image_directory,
                stats_path=str(paths.statistics / "fid_statistics.pt"),
                file_names=benchmark_set.file_names,
            ),
            FIDReference(
                name="fid_patch",
                image_dir=image_directory,
                stats_path=str(paths.statistics / "fid_patch_statistics.pt"),
                file_names=benchmark_set.file_names,
                patch_size=FID_PATCH_SIZE,
            ),
        ],
        image_size=benchmark_set.high_resolution,
        batch_size=16,
    )


@torch.no_grad()
def evaluate_weights(
    model: torch.nn.Module,
    vae: Flux2VAE,
    benchmark_set: BenchmarkSet,
    fid_evaluator: FIDEvaluator,
    saved_indices: set[int],
    output_directory: Path,
    variant: str,
) -> dict[str, float]:
    """
    returns:
        fid, fid_patch, mse_lr_image (generated image downscaled 2x vs low resolution input)
    """
    device = torch.device("cuda")
    squared_errors: list[float] = []
    for start in range(0, len(benchmark_set), BENCHMARK_BATCH_SIZE):
        indices = range(start, min(start + BENCHMARK_BATCH_SIZE, len(benchmark_set)))
        features = benchmark_set.high_frequency_features[start : indices.stop].to(device)
        latents_high = sample_high_resolution_latents(
            model,
            benchmark_set.latents_low[start : indices.stop].to(device),
            benchmark_set.text_embedding[start : indices.stop].to(device),
            torch.ones(len(indices), dtype=torch.bool, device=device),
            features,
            ~torch.isnan(features),
        )
        images_high = vae.decode(latents_high).float().clamp(0, 1)  # [B, 3, H, W]
        fid_evaluator.update(images_high)
        images_low = benchmark_set.load_low_resolution_images(indices).to(device)
        downscaled = functional.avg_pool2d(images_high, 2)
        squared_errors.extend(((downscaled - images_low) ** 2).mean(dim=(1, 2, 3)).tolist())
        for offset, index in enumerate(indices):
            if index in saved_indices:
                cv2.imwrite(
                    str(output_directory / f"{index:05d}_{variant}.png"),
                    torch_to_cv2(images_high[offset : offset + 1]),
                )
    metrics = fid_evaluator.compute()
    metrics["mse_lr_image"] = float(np.mean(squared_errors))
    return metrics


def snapshot_weights(tensors: dict[str, torch.Tensor], variant: str) -> dict[str, torch.Tensor]:
    prefix = f"{variant}."
    return {name.removeprefix(prefix): tensor for name, tensor in tensors.items() if name.startswith(prefix)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    arguments = parser.parse_args()
    paths = run_paths(arguments.run_name)
    torch.backends.cudnn.benchmark = True

    benchmark_set = BenchmarkSet(paths)
    fid_evaluator = build_fid_evaluator(paths, benchmark_set)
    vae = Flux2VAE(str(VAE_DIRECTORY), device="cuda", dtype=torch.bfloat16)
    model = UpscalerUNet().to("cuda", INFERENCE_DTYPE).eval().requires_grad_(False)
    compiled_model = torch.compile(model)
    saved_indices = set(
        np.linspace(0, len(benchmark_set) - 1, BENCHMARK_SAVED_PAIRS).round().astype(int).tolist()
    )
    summary_writer = SummaryWriter(str(paths.tensorboard / "benchmark"))
    print(f"benchmark: {len(benchmark_set)} samples ready", flush=True)

    while True:
        snapshot_paths = sorted(
            paths.snapshots.glob("*.safetensors"), key=lambda path: path.stat().st_mtime
        )
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
        for index in saved_indices:
            source = benchmark_set.image_low_directory / benchmark_set.file_names[index]
            (output_directory / f"{index:05d}_low.png").write_bytes(source.read_bytes())

        tensors = load_file(str(snapshot_path))
        paths.encoder_pause.touch()
        start_time = time.perf_counter()
        try:
            results: dict[str, object] = dict(snapshot_info)
            for variant in WEIGHT_VARIANTS:
                model.load_state_dict(snapshot_weights(tensors, variant))
                metrics = evaluate_weights(
                    compiled_model,
                    vae,
                    benchmark_set,
                    fid_evaluator,
                    saved_indices,
                    output_directory,
                    variant,
                )
                for name, value in metrics.items():
                    summary_writer.add_scalar(f"benchmark_{variant}/{name}", value, step)
                    results[f"{variant}_{name}"] = value
            summary_writer.add_scalar("benchmark/samples_seen", snapshot_info["samples_seen"], step)
            summary_writer.flush()
        finally:
            paths.encoder_pause.unlink(missing_ok=True)
        results["benchmark_seconds"] = time.perf_counter() - start_time
        with (paths.benchmarks / "results.jsonl").open("a") as results_file:
            results_file.write(json.dumps(results) + "\n")
        print(f"benchmark: {json.dumps(results)}", flush=True)
        if snapshot_info["tag"] != "final":
            snapshot_path.unlink()
            snapshot_path.with_suffix(".json").unlink()
        else:
            snapshot_path.rename(snapshot_path.with_suffix(".done"))
    summary_writer.close()


if __name__ == "__main__":
    main()
