"""
Online rectified flow distillation pairs of the FLUX.2 latent upscaler. Generator processes (one per GPU) take low
resolution latents from the pretraining cache (pair types in the pretraining count shares, inputs with their
corruptions), draw a condition (hff: true / uniform random / null; CFG scale in GUIDANCE_RANGE for GUIDED_PROBABILITY
of the non-null samples, negative = null hff) and a noise seed, sample the teacher (TEACHER_STEPS Euler steps,
TEACHER_SHIFT) and write safetensors shards into the run's pair directory. The noise is not stored: it is regenerated
bit exactly from the seed (seeded_noise).

Usage: CUDA_VISIBLE_DEVICES=0 python -m stdev.flux_distill.pairs --run-name RUN --generator-index 0
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

from stdev.flux_distill.config import (
    GUIDANCE_RANGE,
    GUIDED_PROBABILITY,
    NULL_HFF_PROBABILITY,
    PRETRAIN_CACHE,
    RANDOM_HFF_PROBABILITY,
    RUNS_DIRECTORY,
    SHARD_SAMPLES,
    TEACHER_BATCH_SIZE,
    TEACHER_CHECKPOINT,
    TEACHER_SHIFT,
    TEACHER_STEPS,
    TEACHER_WEIGHTS,
)
from stdev.flux_distill.student import model_config_of, seeded_noise, teacher_state
import vaezip

from stdev.flux_pretrain.cache import list_shards
from stdev.flux_pretrain.config import PAIR_TYPES, PairType
from stdev.flux_pretrain.model import LATENT_CHANNELS, FluxUpscalerTransformer, from_residual, shift_sigmas

SHARD_SUFFIX = ".safetensors"
INPUT_SHARDS_PER_OUTPUT = 4
PREFETCHED_JOBS = 3
STATISTICS_INTERVAL_SECONDS = 30.0
INFERENCE_DTYPE = torch.bfloat16
DYNAMO_CACHE_SIZE_LIMIT = 64


def pairs_directory(run_name: str) -> Path:
    return RUNS_DIRECTORY / run_name / "pairs"


def load_teacher(device: torch.device) -> FluxUpscalerTransformer:
    """
    Pretrained teacher (TEACHER_WEIGHTS of TEACHER_CHECKPOINT), bf16, compiled blocks.
    """
    checkpoint = torch.load(TEACHER_CHECKPOINT, map_location="cpu", weights_only=False)
    model = FluxUpscalerTransformer(model_config_of(checkpoint))
    model.load_state_dict(teacher_state(checkpoint, TEACHER_WEIGHTS))
    del checkpoint
    model = model.to(device, INFERENCE_DTYPE).eval().requires_grad_(False)
    model.compile_blocks()
    return model


@torch.no_grad()
def sample_teacher(
    model: FluxUpscalerTransformer,
    latents_source: torch.Tensor,
    noise: torch.Tensor,
    hff: torch.Tensor,
    guidance: torch.Tensor,
    steps: int = TEACHER_STEPS,
    shift: float = TEACHER_SHIFT,
) -> torch.Tensor:
    """
    Euler sampling of the residual flow from given noise. If any guidance differs from 1, every sample is evaluated
    with its hff and with the null hff in one doubled batch: velocity = negative + guidance * (positive - negative).

    args:
        latents_source: [B, 32, h, w], normalized FLUX.2 latents
        noise: [B, 32, 2h, 2w], unit gaussian (residual space, sigma 1)
        hff: [B], values: [0...1], NaN = null embedding
        guidance: [B], >= 1
    returns:
        latents: [B, 32, 2h, 2w], normalized FLUX.2 latents, dtype: torch.float32
    """
    batch_size = latents_source.shape[0]
    hff_mask = ~torch.isnan(hff)
    hff_values = torch.nan_to_num(hff, nan=0.0).float()
    guided = bool((guidance != 1.0).any())
    if guided:
        assert bool(hff_mask.all()), "guided samples need a positive hff"
        source = torch.cat([latents_source, latents_source]).to(INFERENCE_DTYPE)
        model_hff = torch.cat([hff_values, torch.zeros_like(hff_values)])
        model_mask = torch.cat([hff_mask, torch.zeros_like(hff_mask)])
    else:
        source = latents_source.to(INFERENCE_DTYPE)
        model_hff = hff_values
        model_mask = hff_mask
    scale = guidance.float().view(batch_size, 1, 1, 1)
    samples = noise.float()
    sigmas = shift_sigmas(torch.linspace(1.0, 0.0, steps + 1, device=noise.device), shift)
    for current_sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        model_input = torch.cat([samples, samples]) if guided else samples
        velocity = model(model_input.to(INFERENCE_DTYPE), current_sigma.expand(model_input.shape[0]), source, model_hff, model_mask).float()
        if guided:
            positive, negative = velocity.chunk(2)
            velocity = negative + scale * (positive - negative)
        samples = samples + (next_sigma - current_sigma) * velocity
    return from_residual(samples, latents_source, model.config.residual_std)


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


def read_shard_inputs(path: Path) -> dict[str, torch.Tensor]:
    """
    Only the fields pair generation needs from a pretraining cache shard (the target latents are not decoded).

    returns:
        latents_source: [N, 32, h, w], hff: [N], variant_index, dataset_index: [N] int64
    """
    encoded = vaezip.load(path)
    return {
        "latents_source": encoded["latents_source"].decode(dtype=torch.float32)[:, :, 0],
        "hff": encoded["hff"].decode(dtype=torch.float32),
        "variant_index": encoded["variant_index"].decode(dtype=torch.float32).round().long(),
        "dataset_index": encoded["dataset_index"].decode(dtype=torch.float32).round().long(),
    }


def load_job(
    pair_type: PairType, shard_paths: list[Path], generator: random.Random, count: int | None = None, input_shards: int = INPUT_SHARDS_PER_OUTPUT
) -> dict[str, torch.Tensor]:
    """
    count (default SHARD_SAMPLES) random samples from input_shards random cache shards of the type, with conditions and
    noise seeds, sorted unguided first (guided samples need a doubled batch).

    returns:
        latents_source: [N, 32, h, w], hff, hff_true, guidance: [N], noise_seed, variant_index, dataset_index: [N] int64
    """
    count = count or SHARD_SAMPLES[pair_type.name]
    per_shard = -(-count // input_shards)
    rows: list[tuple] = []
    while len(rows) < count:
        try:
            shard = read_shard_inputs(generator.choice(shard_paths))
        except (FileNotFoundError, OSError, KeyError, RuntimeError):
            continue
        size = shard["hff"].shape[0]
        for index in generator.sample(range(size), min(per_shard, size, count - len(rows))):
            hff_true = shard["hff"][index].item()
            hff, guidance = draw_conditions(generator, hff_true)
            rows.append(
                (shard["latents_source"][index], hff, hff_true, guidance, generator.getrandbits(62), int(shard["variant_index"][index]), int(shard["dataset_index"][index]))
            )
    rows.sort(key=lambda row: row[3] != 1.0)
    sources, hffs, hff_trues, guidances, seeds, variants, datasets = zip(*rows)
    return {
        "latents_source": torch.stack(sources),
        "hff": torch.tensor(hffs, dtype=torch.float32),
        "hff_true": torch.tensor(hff_trues, dtype=torch.float32),
        "guidance": torch.tensor(guidances, dtype=torch.float32),
        "noise_seed": torch.tensor(seeds, dtype=torch.int64),
        "variant_index": torch.tensor(variants, dtype=torch.int64),
        "dataset_index": torch.tensor(datasets, dtype=torch.int64),
    }


def write_pair_shard(directory: Path, writer_name: str, tensors: dict[str, torch.Tensor]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{time.time_ns()}_{writer_name}"
    temporary_path = directory / f".{stem}.tmp"
    save_file({name: tensor.contiguous() for name, tensor in tensors.items()}, str(temporary_path))
    path = directory / f"{stem}{SHARD_SUFFIX}"
    temporary_path.replace(path)
    return path


def list_pair_shards(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob(f"*{SHARD_SUFFIX}"))


def count_pairs(run_name: str) -> int:
    total = 0
    for statistics_path in pairs_directory(run_name).glob("statistics_*.json"):
        try:
            total += sum(json.loads(statistics_path.read_text())["produced"].values())
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            continue
    return total


class PairDataset(IterableDataset):
    """
    Endless stream of one pair type from the pair directory: each worker loads a random current shard and yields
    its samples in random order.
    """

    def __init__(self, directory: Path, seed: str) -> None:
        self.directory = directory
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        yields:
            latents_source: [32, h, w] fp16, latents_generated: [32, 2h, 2w] fp16, hff, hff_true, guidance: [] fp32,
            noise_seed, variant_index, dataset_index: [] int64
        """
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        generator = random.Random(f"{self.seed}-{worker_id}-{time.time_ns()}")
        while True:
            shard_paths = list_pair_shards(self.directory)
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
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--generator-index", type=int, required=True)
    arguments = parser.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = DYNAMO_CACHE_SIZE_LIMIT
    output_directory = pairs_directory(arguments.run_name)
    output_directory.mkdir(parents=True, exist_ok=True)
    writer_name = f"generator{arguments.generator_index}"
    statistics_path = output_directory / f"statistics_{writer_name}.json"
    model = load_teacher(device)
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
                jobs.put((pair_type, load_job(pair_type, shard_paths[pair_type.name], generator)))
            except Exception as error:
                jobs.put(error)
                return

    threading.Thread(target=prefetch, daemon=True).start()
    start_time = time.perf_counter()
    last_statistics = 0.0
    session_samples = 0
    print(f"{writer_name}: started", flush=True)
    while True:
        item = jobs.get()
        if isinstance(item, Exception):
            raise item
        pair_type, job = item
        batch_size = TEACHER_BATCH_SIZE[pair_type.name]
        is_guided = job["guidance"] != 1.0
        generated: list[torch.Tensor] = []
        for group in (~is_guided, is_guided):
            indices = torch.nonzero(group).flatten()
            group_batch = batch_size if group is not is_guided else max(1, batch_size // 2)
            for start in range(0, len(indices), group_batch):
                part = indices[start : start + group_batch]
                noise = seeded_noise(job["noise_seed"][part], (LATENT_CHANNELS, *pair_type.target_latent_size), device)
                latents = sample_teacher(
                    model, job["latents_source"][part].to(device), noise, job["hff"][part].to(device), job["guidance"][part].to(device)
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
        if now - last_statistics > STATISTICS_INTERVAL_SECONDS:
            rate = session_samples / (now - start_time)
            statistics_path.write_text(json.dumps({"produced": dict(produced), "samples_per_second": rate}))
            print(f"{writer_name}: {sum(produced.values())} samples, {rate:.2f} samples/s, {json.dumps(dict(produced))}", flush=True)
            last_statistics = now


if __name__ == "__main__":
    main()
