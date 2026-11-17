"""
Benchmark of the Wan pretraining: BENCHMARK_IMAGE_COUNT held-out ReLAION images (benchmark metadata shard,
maximal centered square, 1024 px) upscaled 512 -> 1024 in latent space (full Wan 2.1 VAE), patch FID on
native 256 px tiles. Small set: for training dynamics only, not comparable with 1000 sample benchmarks.

--build   downloads and encodes the benchmark set (ground truth PNG, source / target latents, target hff)
watcher   (default) evaluates every snapshot written by the trainer with the raw and the EMA weights, logs
          to tensorboard (benchmark_raw/*, benchmark_ema/*) and benchmarks/results.jsonl, saves all generated
          images (benchmarks/<snapshot>/<index>_<raw|ema>.png; bicubic inputs once in benchmarks/inputs,
          ground truth in the data run benchmark_set/images).
          The first run also logs the VAE oracle and pixel bicubic references. Exits after training is done.

Usage: CUDA_VISIBLE_DEVICES=7 python -m stdev.wan_pretrain.benchmark --run-name NAME --data-run-name DATA [--build]
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from torch.utils.tensorboard import SummaryWriter

from stdev.hff.immerkaer import estimate_immerkaer
from stdev.pretrain.config import PretrainPaths
from stdev.pretrain.producer import fetch_image
from stdev.pretrain.sources import list_metadata_files, load_filtered_shard
from stdev.utils.fid import FIDEvaluator, FIDReference
from stdev.utils.image_tools import crop_maximal_rectangle, torch_to_cv2
from stdev.vaes.wan_2_1_vae import Wan21VAE
from stdev.wan_pretrain.cache import EncodedShard, list_shards, read_shard, write_shard
from stdev.video_downloader.stream_videos import load_env_file
from stdev.wan_pretrain.config import (
    BENCHMARK_DECODE_BATCH,
    ENV_PATH,
    BENCHMARK_IMAGE_COUNT,
    BENCHMARK_SAMPLER,
    BENCHMARK_SAVED_IMAGES,
    BENCHMARK_SHARD_INDEX,
    FULL_VAE_DIRECTORY,
    MIN_IMAGE_SOURCE_SIDE,
    MODEL,
    run_paths,
)
from stdev.wan_pretrain.model import WanUpscalerTransformer, upscale

IMAGE_SIDE = 1024
FID_PATCH_SIZE = 256
UPSCALE_BATCH = 4
DOWNLOAD_BATCH = 64
POLL_INTERVAL_SECONDS = 30.0
WEIGHT_VARIANTS = ("raw", "ema")
INFERENCE_DTYPE = torch.bfloat16


async def download_first_images(urls: list[str], count: int) -> list[bytes]:
    """
    returns:
        contents of the first `count` urls (in url order) that load as images with min side >= MIN_IMAGE_SOURCE_SIDE
    """
    results: list[bytes] = []
    timeout = aiohttp.ClientTimeout(total=20.0, connect=5.0)
    connector = aiohttp.TCPConnector(limit=DOWNLOAD_BATCH, ssl=False)
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-dataset-downloader"}
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
        for start in range(0, len(urls), DOWNLOAD_BATCH):
            contents = await asyncio.gather(*(fetch_image(session, url) for url in urls[start : start + DOWNLOAD_BATCH]))
            for content in contents:
                if content is None:
                    continue
                image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or min(image.shape[:2]) < MIN_IMAGE_SOURCE_SIDE:
                    continue
                results.append(content)
                if len(results) == count:
                    return results
    raise RuntimeError(f"only {len(results)} benchmark images available")


@torch.no_grad()
def build_benchmark_set(paths: PretrainPaths) -> None:
    done_marker = paths.benchmark_set / "done"
    if done_marker.exists():
        return
    images_directory = paths.benchmark_set / "images"
    images_directory.mkdir(parents=True, exist_ok=True)
    file_name = list_metadata_files()[BENCHMARK_SHARD_INDEX]
    jobs = load_filtered_shard(file_name, MIN_IMAGE_SOURCE_SIDE)
    contents = asyncio.run(download_first_images([job.url for job in jobs[: 20 * BENCHMARK_IMAGE_COUNT]], BENCHMARK_IMAGE_COUNT))
    vae = Wan21VAE(str(FULL_VAE_DIRECTORY), device="cuda", dtype=torch.bfloat16, as_image=True)
    sources, targets, hffs = [], [], []
    for index, content in enumerate(contents):
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = crop_maximal_rectangle(image, IMAGE_SIDE, IMAGE_SIDE)
        cv2.imwrite(str(images_directory / f"{index:05d}.png"), image)
        pixels = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cuda().permute(2, 0, 1)[None].float() / 255.0
        targets.append(vae.encode(pixels).cpu())
        sources.append(vae.encode(functional.avg_pool2d(pixels, 2)).cpu())
        hffs.append(estimate_immerkaer(pixels).float().cpu())
    write_shard(paths.benchmark_set, "benchmark", EncodedShard(torch.cat(sources), torch.cat(targets), torch.cat(hffs)))
    done_marker.touch()
    print(f"benchmark: built {len(contents)} images", flush=True)


class BenchmarkSet:
    def __init__(self, paths: PretrainPaths) -> None:
        shard = read_shard(list_shards(paths.benchmark_set)[0])
        self.latents_source = shard.latents_source  # [N, 16, 1, 64, 64]
        self.latents_target = shard.latents_target  # [N, 16, 1, 128, 128]
        self.hff = shard.hff  # [N]
        self.images_directory = paths.benchmark_set / "images"
        self.file_names = [f"{index:05d}.png" for index in range(len(self.hff))]

    def __len__(self) -> int:
        return len(self.file_names)

    def load_images(self, indices: range) -> torch.Tensor:
        """
        returns:
            images: [B, 3, 1024, 1024], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32
        """
        images = [cv2.cvtColor(cv2.imread(str(self.images_directory / self.file_names[index])), cv2.COLOR_BGR2RGB) for index in indices]
        return torch.from_numpy(np.stack(images)).cuda().permute(0, 3, 1, 2).float() / 255.0


def build_fid_evaluator(paths: PretrainPaths, benchmark_set: BenchmarkSet) -> FIDEvaluator:
    common = {"image_dir": str(benchmark_set.images_directory), "file_names": benchmark_set.file_names}
    return FIDEvaluator(
        references=[
            FIDReference(name="fid", stats_path=str(paths.statistics / "fid_statistics.pt"), **common),
            FIDReference(
                name="fid_patch", stats_path=str(paths.statistics / "fid_patch_statistics.pt"), patch_size=FID_PATCH_SIZE, **common
            ),
        ],
        image_size=IMAGE_SIDE,
        batch_size=16,
    )


@torch.no_grad()
def decode(vae: Wan21VAE, latents: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents: [B, 16, 1, 128, 128]
    returns:
        images: [B, 3, 1024, 1024], pixel_values: [0...1], dtype: torch.float32
    """
    return torch.cat([vae.decode(chunk.cuda()).float().clamp(0, 1) for chunk in latents.split(BENCHMARK_DECODE_BATCH)])


@torch.no_grad()
def evaluate_images(
    produce_batch,
    vae: Wan21VAE,
    benchmark_set: BenchmarkSet,
    fid_evaluator: FIDEvaluator,
    output_directory: Path | None,
    tag: str,
) -> dict[str, float]:
    """
    args:
        produce_batch: function(indices) -> images [B, 3, 1024, 1024], pixel_values: [0...1]
    returns:
        fid, fid_patch, mse_lr (generated image box-downscaled 2x vs ground truth downscaled 2x)
    """
    squared_errors: list[float] = []
    for start in range(0, len(benchmark_set), UPSCALE_BATCH):
        indices = range(start, min(start + UPSCALE_BATCH, len(benchmark_set)))
        images = produce_batch(indices)
        fid_evaluator.update(images)
        ground_truth = benchmark_set.load_images(indices)
        squared_errors.extend(
            ((functional.avg_pool2d(images, 2) - functional.avg_pool2d(ground_truth, 2)) ** 2).mean(dim=(1, 2, 3)).tolist()
        )
        if output_directory is not None:
            for offset, index in enumerate(indices):
                if index < BENCHMARK_SAVED_IMAGES:
                    cv2.imwrite(str(output_directory / f"{index:05d}_{tag}.png"), torch_to_cv2(images[offset : offset + 1]))
    metrics = fid_evaluator.compute()
    metrics["mse_lr"] = float(np.mean(squared_errors))
    return metrics


def log_references(
    paths: PretrainPaths, vae: Wan21VAE, benchmark_set: BenchmarkSet, fid_evaluator: FIDEvaluator, summary_writer: SummaryWriter
) -> None:
    references_path = paths.benchmarks / "references.json"
    if references_path.exists():
        return

    def oracle(indices: range) -> torch.Tensor:
        return decode(vae, benchmark_set.latents_target[indices.start : indices.stop])

    def bicubic(indices: range) -> torch.Tensor:
        low = functional.avg_pool2d(benchmark_set.load_images(indices), 2)
        return functional.interpolate(low, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)

    inputs_directory = paths.benchmarks / "inputs"
    inputs_directory.mkdir(parents=True, exist_ok=True)
    references = {
        "vae_oracle": evaluate_images(oracle, vae, benchmark_set, fid_evaluator, None, "oracle"),
        "pixel_bicubic": evaluate_images(bicubic, vae, benchmark_set, fid_evaluator, inputs_directory, "bicubic"),
    }
    for name, metrics in references.items():
        for metric, value in metrics.items():
            summary_writer.add_text(f"references/{name}/{metric}", f"{value:.4f}", 0)
    references_path.write_text(json.dumps(references, indent=1))
    print(f"benchmark: references {json.dumps(references)}", flush=True)


def snapshot_weights(tensors: dict[str, torch.Tensor], variant: str) -> dict[str, torch.Tensor]:
    prefix = f"{variant}."
    return {name.removeprefix(prefix): tensor for name, tensor in tensors.items() if name.startswith(prefix)}


def run_watcher(paths: PretrainPaths, data_paths: PretrainPaths) -> None:
    torch.backends.cudnn.benchmark = True
    benchmark_set = BenchmarkSet(data_paths)
    fid_evaluator = build_fid_evaluator(data_paths, benchmark_set)
    vae = Wan21VAE(str(FULL_VAE_DIRECTORY), device="cuda", dtype=torch.bfloat16, as_image=True)
    model = WanUpscalerTransformer(MODEL).to("cuda", INFERENCE_DTYPE).eval().requires_grad_(False)
    model.compile_blocks()
    summary_writer = SummaryWriter(str(paths.tensorboard / "benchmark"))
    paths.benchmarks.mkdir(parents=True, exist_ok=True)
    log_references(paths, vae, benchmark_set, fid_evaluator, summary_writer)
    print(f"benchmark: {len(benchmark_set)} images ready", flush=True)

    def generated(indices: range) -> torch.Tensor:
        latents = upscale(
            model,
            benchmark_set.latents_source[indices.start : indices.stop].cuda(),
            benchmark_set.hff[indices.start : indices.stop].cuda(),
            tuple(benchmark_set.latents_target.shape[2:]),
            BENCHMARK_SAMPLER,
            INFERENCE_DTYPE,
        )
        return decode(vae, latents)

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
            metrics = evaluate_images(generated, vae, benchmark_set, fid_evaluator, output_directory, variant)
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
