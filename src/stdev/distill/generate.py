"""
Teacher pair generation of one distillation round on one GPU: reads 4KLSDB images from the rank's
position, makes random scale crops, encodes the low resolution crop and the caption, samples the
teacher (TEACHER_STEPS Euler steps at TEACHER_SIGMA_SHIFT) from seeded noise with dropped out
conditioning, and writes pair shards. Generates for GENERATION_SECONDS after the first batch
(compilation excluded) and writes a result json with the new source position.

Usage: CUDA_VISIBLE_DEVICES=G python -m stdev.distill.generate --run-name NAME --round R
           --rank G --world-size 8 --file-index F --row-offset O [--seconds S]
"""

import argparse
import json
import multiprocessing
import threading
import time
from typing import Iterator

import numpy as np
import torch
from safetensors.torch import load_file

from stdev.distill.config import (
    GENERATION_SECONDS,
    HIGH_FREQUENCY_DROPOUT_PROBABILITY,
    PAIR_SHARD_SAMPLES,
    PREPROCESS_LOOKAHEAD_IMAGES,
    PREPROCESS_WORKERS_PER_GPU,
    TEACHER_BATCH_SIZE,
    TEACHER_SIGMA_SHIFT,
    TEACHER_STEPS,
    TEACHER_WEIGHTS_PATH,
    TEXT_DROPOUT_PROBABILITY,
    run_paths,
)
from stdev.distill.data import (
    PairShard,
    SourcePosition,
    iterate_rows,
    list_train_files,
    make_noise,
    preprocess_image,
    rank_files,
    write_pair_shard,
)
from stdev.pretrain.config import SDXL_DIRECTORY, VAE_DIRECTORY
from stdev.pretrain.high_frequency import QuantileNormalizer
from stdev.pretrain.producer import encode_in_chunks
from stdev.upscalers.pretrain_large import (
    INFERENCE_DTYPE,
    SCALE_FACTOR,
    UpscalerUNet,
    sample_high_resolution_latents,
)

SEEDS_PER_ROUND = 1 << 32
SEEDS_PER_RANK = 1 << 28


def preprocess_job(job: tuple[SourcePosition, bytes, str]) -> tuple[SourcePosition, str, list[dict]]:
    position, content, caption = job
    return position, caption, preprocess_image(content, f"{position.file_index}-{position.row_offset}")


def throttled(jobs: Iterator, semaphore: threading.Semaphore, stop_event: threading.Event) -> Iterator:
    """
    Feeds at most the semaphore's count of jobs ahead of the consumer; ends when stop_event is set.
    """
    for job in jobs:
        semaphore.acquire()
        if stop_event.is_set():
            return
        yield job


def load_teacher(device: torch.device) -> torch.nn.Module:
    model = UpscalerUNet()
    model.load_state_dict(load_file(str(TEACHER_WEIGHTS_PATH)))
    return torch.compile(model.to(device, INFERENCE_DTYPE).eval().requires_grad_(False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--file-index", type=int, required=True)
    parser.add_argument("--row-offset", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=GENERATION_SECONDS)
    arguments = parser.parse_args()

    from stdev.text_encoders.sdxl_text_encoder import SDXLTextEncoder
    from stdev.vaes.flux_2_vae import Flux2VAE

    paths = run_paths(arguments.run_name)
    output_directory = paths.pairs(arguments.round)
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(arguments.round * 1000 + arguments.rank)

    vae = Flux2VAE(str(VAE_DIRECTORY), device=device, dtype=torch.bfloat16)
    text_encoder = SDXLTextEncoder(str(SDXL_DIRECTORY), device=device, dtype=torch.float16)
    teacher = load_teacher(device)
    normalizer = QuantileNormalizer.load(paths.root / "high_frequency_quantiles.json")
    files = rank_files(list_train_files(), arguments.rank, arguments.world_size)
    start_position = SourcePosition(arguments.file_index, arguments.row_offset)
    seed_base = arguments.round * SEEDS_PER_ROUND + arguments.rank * SEEDS_PER_RANK
    writer_name = f"r{arguments.round:04d}_g{arguments.rank}"

    semaphore = threading.Semaphore(PREPROCESS_LOOKAHEAD_IMAGES)
    stop_event = threading.Event()
    context = multiprocessing.get_context("spawn")
    pool = context.Pool(PREPROCESS_WORKERS_PER_GPU)
    results = pool.imap(
        preprocess_job, throttled(iterate_rows(files, start_position), semaphore, stop_event)
    )

    position = start_position
    pending: list[dict] = []
    shard_parts: dict[str, list] = {name: [] for name in ("low", "teacher", "text", "features", "text_mask", "feature_mask")}
    shard_seeds: list[int] = []
    shard_captions: list[str] = []
    generated = 0
    first_batch_done_time: float | None = None
    start_time = time.perf_counter()

    def flush() -> None:
        if not shard_seeds:
            return
        write_pair_shard(
            output_directory,
            writer_name,
            PairShard(
                torch.cat(shard_parts["low"]),
                torch.cat(shard_parts["teacher"]),
                torch.cat(shard_parts["text"]),
                torch.cat(shard_parts["features"]),
                torch.cat(shard_parts["text_mask"]),
                torch.cat(shard_parts["feature_mask"]),
                list(shard_seeds),
            ),
            list(shard_captions),
        )
        for values in shard_parts.values():
            values.clear()
        shard_seeds.clear()
        shard_captions.clear()

    for position, caption, crops in results:
        semaphore.release()
        pending.extend({**crop, "caption": caption} for crop in crops)
        if len(pending) < TEACHER_BATCH_SIZE:
            continue
        batch, pending = pending[:TEACHER_BATCH_SIZE], pending[TEACHER_BATCH_SIZE:]
        with torch.no_grad():
            latents_low = encode_in_chunks(vae, [sample["image_low"] for sample in batch], device)
            text_embedding = text_encoder.encode_text_batch(
                [sample["caption"] for sample in batch]
            ).float().mean(dim=1)  # [B, 768]
            raw_features = np.stack([sample["high_frequency_raw"] for sample in batch])
            features = torch.from_numpy(normalizer.normalize(raw_features)).float().to(device)
            text_mask = torch.rand(len(batch), device=device) >= TEXT_DROPOUT_PROBABILITY
            feature_mask = (
                torch.rand(features.shape, device=device) >= HIGH_FREQUENCY_DROPOUT_PROBABILITY
            ) & ~torch.isnan(features)
            seeds = [seed_base + generated + offset for offset in range(len(batch))]
            height, width = (size * SCALE_FACTOR for size in latents_low.shape[-2:])
            noise = torch.stack([make_noise(seed, height, width) for seed in seeds]).to(device)
            latents_teacher = sample_high_resolution_latents(
                teacher,
                latents_low,
                text_embedding,
                text_mask,
                features,
                feature_mask,
                num_steps=TEACHER_STEPS,
                sigma_shift=TEACHER_SIGMA_SHIFT,
                noise=noise,
            )
        shard_parts["low"].append(latents_low.float().cpu())
        shard_parts["teacher"].append(latents_teacher.cpu())
        shard_parts["text"].append(text_embedding.cpu())
        shard_parts["features"].append(features.cpu())
        shard_parts["text_mask"].append(text_mask.cpu())
        shard_parts["feature_mask"].append(feature_mask.cpu())
        shard_seeds.extend(seeds)
        shard_captions.extend(sample["caption"] for sample in batch)
        generated += len(batch)
        if len(shard_seeds) >= PAIR_SHARD_SAMPLES:
            flush()

        now = time.perf_counter()
        if first_batch_done_time is None:
            first_batch_done_time = now
            print(f"generate: first batch after {now - start_time:.0f} s", flush=True)
        if now - first_batch_done_time >= arguments.seconds:
            break
    flush()
    stop_event.set()
    semaphore.release(PREPROCESS_LOOKAHEAD_IMAGES + 1)
    pool.terminate()

    seconds = time.perf_counter() - (first_batch_done_time or start_time)
    result = {
        "rank": arguments.rank,
        "pairs": generated,
        "seconds": seconds,
        "pairs_per_second": generated / max(seconds, 1e-6),
        "file_index": position.file_index,
        "row_offset": position.row_offset,
        "files_exhausted": position.file_index >= len(files),
    }
    (paths.round_results(arguments.round) / f"generate_{arguments.rank}.json").write_text(json.dumps(result))
    print(f"generate: {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
