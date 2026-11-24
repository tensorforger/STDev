"""
Online pairs of the large single step FLUX.2 student for the small student: generator processes (one per GPU) take
low resolution latents from the pretraining cache (pretraining count shares, with corruptions), draw the conditions as
for the teacher pairs (true / random / null hff, CFG in GUIDANCE_RANGE for half of the non-null samples) and a noise
seed, run the large student once and write shards into a rolling pair cache (the first generator evicts the oldest
shards above PAIR_CACHE_SIZE_LIMIT_BYTES).

Usage: CUDA_VISIBLE_DEVICES=5 python -m stdev.flux_distill.small_pairs --run-name RUN --generator-index 0 [--evict]
           [--max-seconds S]
"""

import argparse
import json
import os
import queue
import random
import threading
import time
from collections import Counter
from pathlib import Path

import torch

from stdev.flux_distill.config import PRETRAIN_CACHE, RUNS_DIRECTORY
from stdev.flux_distill.pairs import load_job, write_pair_shard
from stdev.flux_distill.small_config import (
    GENERATOR_BATCH_SIZE,
    LARGE_STUDENT_CHECKPOINT,
    LARGE_STUDENT_WEIGHTS,
    PAIR_CACHE_SIZE_LIMIT_BYTES,
    SHARD_SAMPLES,
)
from stdev.flux_distill.student import FluxStudentTransformer, model_config_of, seeded_noise, upscale_single_step
from stdev.flux_pretrain.cache import cache_size_bytes, list_shards
from stdev.flux_pretrain.config import PAIR_TYPES
from stdev.flux_pretrain.model import LATENT_CHANNELS

PREFETCHED_JOBS = 6
PREFETCH_THREADS = 2
INPUT_SHARDS_PER_JOB = 1
EVICTION_INTERVAL_SECONDS = 60.0
STATISTICS_INTERVAL_SECONDS = 30.0


def evict_oldest_pairs(directory: Path, size_limit_bytes: int) -> int:
    """
    Deletes the oldest pair shards (over all pair types) until the directory fits the limit.

    returns:
        number of deleted shards
    """
    total = cache_size_bytes(directory)
    deleted = 0
    for shard_path in sorted(directory.glob("*/*.safetensors"), key=lambda path: path.name):
        if total <= size_limit_bytes:
            break
        try:
            total -= shard_path.stat().st_size
            shard_path.unlink()
            deleted += 1
        except FileNotFoundError:
            continue
    return deleted


def small_pairs_directory(run_name: str) -> Path:
    return RUNS_DIRECTORY / run_name / "pairs"


def load_large_student(device: torch.device) -> FluxStudentTransformer:
    """
    LARGE_STUDENT_WEIGHTS of the distillation checkpoint, bf16, compiled blocks.
    """
    checkpoint = torch.load(LARGE_STUDENT_CHECKPOINT, map_location="cpu", weights_only=False)
    model = FluxStudentTransformer(model_config_of(checkpoint))
    if LARGE_STUDENT_WEIGHTS == "raw":
        model.load_state_dict(checkpoint["model"])
    else:
        with torch.no_grad():
            for parameter, weight in zip(model.parameters(), checkpoint["ema"], strict=True):
                parameter.copy_(weight)
    del checkpoint
    model = model.to(device, torch.bfloat16).eval().requires_grad_(False)
    model.compile_blocks()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--generator-index", type=int, required=True)
    parser.add_argument("--evict", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    arguments = parser.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 64
    output_directory = small_pairs_directory(arguments.run_name)
    output_directory.mkdir(parents=True, exist_ok=True)
    writer_name = f"generator{arguments.generator_index}"
    statistics_path = output_directory / f"statistics_{writer_name}.json"
    model = load_large_student(device)
    shard_paths = {pair_type.name: list_shards(PRETRAIN_CACHE / pair_type.name) for pair_type in PAIR_TYPES}
    generator = random.Random(int.from_bytes(os.urandom(8), "little"))
    produced: Counter = Counter()
    if statistics_path.exists():
        produced.update(json.loads(statistics_path.read_text()).get("produced", {}))
    jobs: queue.Queue = queue.Queue(maxsize=PREFETCHED_JOBS)

    def prefetch() -> None:
        planned: Counter = Counter(produced)
        while True:
            total = sum(planned.values()) + 1
            pair_type = max(PAIR_TYPES, key=lambda candidate: candidate.count_share - planned[candidate.name] / total)
            planned[pair_type.name] += SHARD_SAMPLES[pair_type.name]
            try:
                job = load_job(pair_type, shard_paths[pair_type.name], generator, SHARD_SAMPLES[pair_type.name], INPUT_SHARDS_PER_JOB)
                jobs.put((pair_type, job))
            except Exception as error:
                jobs.put(error)
                return

    for _ in range(PREFETCH_THREADS):
        threading.Thread(target=prefetch, daemon=True).start()
    start_time = time.perf_counter()
    last_statistics = 0.0
    last_eviction = 0.0
    session_samples = 0
    print(f"{writer_name}: started", flush=True)
    while arguments.max_seconds <= 0 or time.perf_counter() - start_time < arguments.max_seconds:
        item = jobs.get()
        if isinstance(item, Exception):
            raise item
        pair_type, job = item
        batch_size = GENERATOR_BATCH_SIZE[pair_type.name]
        generated: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(job["hff"]), batch_size):
                part = slice(start, start + batch_size)
                noise = seeded_noise(job["noise_seed"][part], (LATENT_CHANNELS, *pair_type.target_latent_size), device)
                latents = upscale_single_step(
                    model, job["latents_source"][part].to(device), job["hff"][part].to(device), job["guidance"][part].to(device), noise
                )
                generated.append(latents.half().cpu())
        write_pair_shard(
            output_directory / pair_type.name,
            writer_name,
            {**job, "latents_source": job["latents_source"].half(), "latents_generated": torch.cat(generated)},
        )
        produced[pair_type.name] += len(job["hff"])
        session_samples += len(job["hff"])
        now = time.perf_counter()
        if arguments.evict and now - last_eviction > EVICTION_INTERVAL_SECONDS:
            evict_oldest_pairs(output_directory, PAIR_CACHE_SIZE_LIMIT_BYTES)
            last_eviction = now
        if now - last_statistics > STATISTICS_INTERVAL_SECONDS:
            rate = session_samples / (now - start_time)
            statistics_path.write_text(json.dumps({"produced": dict(produced), "samples_per_second": rate}))
            print(
                f"{writer_name}: {sum(produced.values())} samples, {rate:.1f} samples/s, cache {cache_size_bytes(output_directory) / 2**30:.0f} GB",
                flush=True,
            )
            last_statistics = now


if __name__ == "__main__":
    main()
