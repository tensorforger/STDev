"""
Online distillation pairs of the large single step student for the small students: generator processes (one per
GPU) sample LR latents from the pretraining cache in the pretraining production shares, draw the conditions as for
the teacher pairs (true / random / null hff, CFG in [1, 3] for half of the non-null samples) and a noise seed, run
the large student once and write safetensors shards into a rolling cache (oldest shards evicted above
CACHE_SIZE_LIMIT_BYTES). The noise is not stored: it is regenerated bit exactly from the seed on any GPU
(seeded_noise).

Usage: CUDA_VISIBLE_DEVICES=6 python -m stdev.wan_distill.online_pairs --generator-index 0 [--data-root DIR] [--evict]
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
from typing import Iterator

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import IterableDataset, get_worker_info

from stdev.wan_distill.config import DEFAULT_DATA_ROOT
from stdev.wan_distill.pairs import GUIDANCE_RANGE, GUIDED_PROBABILITY, NULL_HFF_PROBABILITY, RANDOM_HFF_PROBABILITY, noise_scale_of
from stdev.wan_distill.small_config import (
    CACHE_SIZE_LIMIT_BYTES,
    GENERATOR_BATCH_SIZE,
    LARGE_STUDENT_CHECKPOINT,
    ONLINE_PAIRS_DIRECTORY,
    PRETRAIN_CACHE,
    SHARD_SAMPLES,
)
from stdev.wan_distill.student import WanStudentTransformer, upscale_single_step
from stdev.wan_pretrain.cache import cache_size_bytes, list_shards, read_shard
from stdev.wan_pretrain.config import PAIR_TYPES, PairType, production_shares
from stdev.wan_pretrain.model import ModelConfig

SHARD_SUFFIX = ".safetensors"
INPUT_SHARDS_PER_OUTPUT = 4
PREFETCHED_JOBS = 3
EVICTION_INTERVAL_SECONDS = 60.0
STATISTICS_INTERVAL_SECONDS = 60.0
DYNAMO_CACHE_SIZE_LIMIT = 64


def seeded_noise(seeds: torch.Tensor, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """
    args:
        seeds: [B], dtype: torch.int64
        shape: per sample shape (16, T, H, W)
    returns:
        noise: [B, *shape], unit gaussian, dtype: torch.float32, identical on every GPU for the same seeds
    """
    generator = torch.Generator(device=device)
    noise = torch.empty((len(seeds), *shape), device=device)
    for index, seed in enumerate(seeds.tolist()):
        generator.manual_seed(seed)
        noise[index] = torch.randn(shape, generator=generator, device=device)
    return noise


def load_large_student(checkpoint_path: Path, device: torch.device) -> WanStudentTransformer:
    """
    EMA weights of a distillation checkpoint, bf16, compiled blocks.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = WanStudentTransformer(ModelConfig(**checkpoint["model_config"]))
    with torch.no_grad():
        for parameter, weight in zip(model.parameters(), checkpoint["ema"], strict=True):
            parameter.copy_(weight)
    del checkpoint
    model = model.to(device, torch.bfloat16).eval().requires_grad_(False)
    model.compile_blocks()
    return model


def draw_conditions(generator: random.Random, hff_true: float) -> tuple[float, float]:
    """
    returns:
        hff (NaN = null), guidance
    """
    mode_draw = generator.random()
    if mode_draw < NULL_HFF_PROBABILITY:
        return float("nan"), 1.0
    hff = generator.random() if mode_draw < NULL_HFF_PROBABILITY + RANDOM_HFF_PROBABILITY else hff_true
    guidance = generator.uniform(*GUIDANCE_RANGE) if generator.random() < GUIDED_PROBABILITY else 1.0
    return hff, guidance


def load_job(pair_type: PairType, shard_paths: list[Path], generator: random.Random) -> dict[str, torch.Tensor]:
    """
    SHARD_SAMPLES random samples from INPUT_SHARDS_PER_OUTPUT random cache shards of the type, with conditions
    and noise seeds.

    returns:
        latents_source: [N, 16, T_lr, h_lr, w_lr], hff, guidance, noise_scale: [N], noise_seed: [N] (int64)
    """
    count = SHARD_SAMPLES[pair_type.name]
    sources: list[torch.Tensor] = []
    hffs: list[float] = []
    guidances: list[float] = []
    per_shard = -(-count // INPUT_SHARDS_PER_OUTPUT)
    while len(sources) < count:
        try:
            shard = read_shard(generator.choice(shard_paths))
        except (FileNotFoundError, OSError, KeyError, RuntimeError):
            continue
        size = shard.hff.shape[0]
        for index in generator.sample(range(size), min(per_shard, size, count - len(sources))):
            sources.append(shard.latents_source[index])
            hff, guidance = draw_conditions(generator, shard.hff[index].item())
            hffs.append(hff)
            guidances.append(guidance)
    return {
        "latents_source": torch.stack(sources),
        "hff": torch.tensor(hffs, dtype=torch.float32),
        "guidance": torch.tensor(guidances, dtype=torch.float32),
        "noise_scale": torch.full((count,), noise_scale_of(pair_type), dtype=torch.float32),
        "noise_seed": torch.tensor([generator.getrandbits(62) for _ in range(count)], dtype=torch.int64),
    }


def write_online_shard(directory: Path, writer_name: str, tensors: dict[str, torch.Tensor]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{time.time_ns()}_{writer_name}"
    temporary_path = directory / f".{stem}.tmp"
    save_file({name: tensor.contiguous() for name, tensor in tensors.items()}, str(temporary_path))
    path = directory / f"{stem}{SHARD_SUFFIX}"
    temporary_path.replace(path)
    return path


def list_online_shards(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob(f"*{SHARD_SUFFIX}"))


def evict_oldest(cache_directory: Path, size_limit_bytes: int) -> int:
    total = cache_size_bytes(cache_directory)
    deleted = 0
    for shard_path in sorted(cache_directory.glob(f"*/*{SHARD_SUFFIX}"), key=lambda path: path.name):
        if total <= size_limit_bytes:
            break
        try:
            total -= shard_path.stat().st_size
            shard_path.unlink()
            deleted += 1
        except FileNotFoundError:
            continue
    return deleted


class OnlinePairDataset(IterableDataset):
    """
    Endless stream of one pair type from the rolling cache: each worker loads a random current shard and yields
    its samples in random order.
    """

    def __init__(self, directory: Path, seed: str) -> None:
        self.directory = directory
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        yields:
            latents_source: [16, T_lr, h_lr, w_lr] fp16, latents_generated: [16, T_hr, h_hr, w_hr] fp16, hff, guidance,
            noise_scale: [] fp32, noise_seed: [] int64
        """
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        generator = random.Random(f"{self.seed}-{worker_id}-{time.time_ns()}")
        while True:
            shard_paths = list_online_shards(self.directory)
            if not shard_paths:
                time.sleep(5.0)
                continue
            try:
                shard = load_file(str(generator.choice(shard_paths)))
            except (FileNotFoundError, OSError, RuntimeError):
                continue
            order = list(range(shard["hff"].shape[0]))
            generator.shuffle(order)
            for index in order:
                yield {name: tensor[index] for name, tensor in shard.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-index", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--evict", action="store_true", help="this generator also evicts the oldest shards")
    arguments = parser.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = DYNAMO_CACHE_SIZE_LIMIT
    output_directory = arguments.data_root / ONLINE_PAIRS_DIRECTORY
    writer_name = f"generator{arguments.generator_index}"
    statistics_path = output_directory / f"statistics_{writer_name}.json"
    model = load_large_student(arguments.data_root / LARGE_STUDENT_CHECKPOINT, device)
    cache_directory = arguments.data_root / PRETRAIN_CACHE
    shard_paths = {pair_type.name: list_shards(cache_directory / pair_type.name) for pair_type in PAIR_TYPES}
    shares = production_shares()
    generator = random.Random(int.from_bytes(os.urandom(8), "little"))
    produced: Counter = Counter()
    if statistics_path.exists():
        produced.update(json.loads(statistics_path.read_text()).get("produced", {}))

    jobs: queue.Queue = queue.Queue(maxsize=PREFETCHED_JOBS)

    def prefetch() -> None:
        planned: Counter = Counter(produced)
        while True:
            total = sum(planned.values()) + 1
            pair_type = max(PAIR_TYPES, key=lambda candidate: shares[candidate.name] - planned[candidate.name] / total)
            planned[pair_type.name] += SHARD_SAMPLES[pair_type.name]
            try:
                jobs.put((pair_type, load_job(pair_type, shard_paths[pair_type.name], generator)))
            except Exception as error:
                jobs.put(error)
                return

    threading.Thread(target=prefetch, daemon=True).start()
    start_time = time.perf_counter()
    last_eviction = 0.0
    last_statistics = 0.0
    session_samples = 0
    print(f"{writer_name}: started", flush=True)
    while True:
        item = jobs.get()
        if isinstance(item, Exception):
            raise item
        pair_type, job = item
        batch_size = GENERATOR_BATCH_SIZE[pair_type.name]
        generated: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(job["hff"]), batch_size):
                part = slice(start, start + batch_size)
                noise = seeded_noise(job["noise_seed"][part], (16, *pair_type.target_latent_size), device)
                latents = upscale_single_step(
                    model,
                    job["latents_source"][part].to(device),
                    job["hff"][part].to(device),
                    job["guidance"][part].to(device),
                    job["noise_scale"][part].to(device)[:, None, None, None, None] * noise,
                )
                generated.append(latents.half().cpu())
        write_online_shard(
            output_directory / pair_type.name,
            writer_name,
            {**job, "latents_source": job["latents_source"].half(), "latents_generated": torch.cat(generated)},
        )
        produced[pair_type.name] += len(job["hff"])
        session_samples += len(job["hff"])
        now = time.perf_counter()
        if arguments.evict and now - last_eviction > EVICTION_INTERVAL_SECONDS:
            evict_oldest(output_directory, CACHE_SIZE_LIMIT_BYTES)
            last_eviction = now
        if now - last_statistics > STATISTICS_INTERVAL_SECONDS:
            rate = session_samples / (now - start_time)
            statistics_path.write_text(json.dumps({"produced": dict(produced), "samples_per_second": rate}))
            print(f"{writer_name}: {sum(produced.values())} samples, {rate:.1f} samples/s, {json.dumps(dict(produced))}", flush=True)
            last_statistics = now


if __name__ == "__main__":
    main()
