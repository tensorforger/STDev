"""
Online rectified flow distillation pairs of the H3 -> FLUX.2 translator (the FLUX.2 pair generator with the
translator teacher). Generator processes (one per GPU) take H3 latents from the pretraining cache (pair types in the
pretraining count shares, inputs with their corruptions), draw a condition (hff: true / uniform random / null; CFG scale
for part of the non-null samples, negative = null hff) and a noise seed, sample the teacher and write safetensors
shards (latents_source: H3, latents_generated: FLUX.2 fp16) into the run's pair directory.

Usage: CUDA_VISIBLE_DEVICES=0 python -m stdev.h3_distill.pairs --run-name RUN --generator-index 0
"""

import argparse
import json
import os
import queue
import random
import threading
import time
from collections import Counter

import torch

from stdev.flux_distill.pairs import load_job, pairs_directory, write_pair_shard
from stdev.flux_distill.student import seeded_noise
from stdev.flux_pretrain.cache import list_shards
from stdev.flux_pretrain.model import LATENT_CHANNELS, shift_sigmas
from stdev.h3_distill.config import (
    PRETRAIN_CACHE,
    SHARD_SAMPLES,
    TEACHER_BATCH_SIZE,
    TEACHER_CHECKPOINT,
    TEACHER_SHIFT,
    TEACHER_STEPS,
    TEACHER_WEIGHTS,
)
from stdev.h3_distill.student import build_teacher
from stdev.h3_pretrain.config import PAIR_TYPES
from stdev.h3_pretrain.model import H3TranslatorTransformer

PREFETCHED_JOBS = 3
STATISTICS_INTERVAL_SECONDS = 30.0
INFERENCE_DTYPE = torch.bfloat16
DYNAMO_CACHE_SIZE_LIMIT = 64


def load_teacher(device: torch.device) -> H3TranslatorTransformer:
    """
    Pretrained translator (TEACHER_WEIGHTS of TEACHER_CHECKPOINT), bf16 with float32 linear base and residual
    statistics, compiled blocks.
    """
    checkpoint = torch.load(TEACHER_CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_teacher(checkpoint, TEACHER_WEIGHTS)
    del checkpoint
    model = model.to(device, INFERENCE_DTYPE).eval().requires_grad_(False)
    model.linear_base.float()
    model.residual_mean = model.residual_mean.float()
    model.residual_std = model.residual_std.float()
    model.compile_blocks()
    return model


@torch.no_grad()
def sample_teacher(
    model: H3TranslatorTransformer,
    latents_source: torch.Tensor,
    noise: torch.Tensor,
    hff: torch.Tensor,
    guidance: torch.Tensor,
    steps: int = TEACHER_STEPS,
    shift: float = TEACHER_SHIFT,
) -> torch.Tensor:
    """
    Euler sampling of the residual flow from given noise; with any guidance != 1 every sample is evaluated with its hff
    and the null hff in one doubled batch: velocity = negative + guidance * (positive - negative).

    args:
        latents_source: [B, 24, h, w], normalized H3 latents
        noise: [B, 32, 2h, 2w], unit gaussian (residual space, sigma 1)
        hff: [B], values: [0...1], NaN = null embedding
        guidance: [B], >= 1
    returns:
        latents: [B, 32, 2h, 2w], normalized FLUX.2 latents, dtype: torch.float32
    """
    batch_size = latents_source.shape[0]
    latents_base = model.linear_base(latents_source)
    hff_mask = ~torch.isnan(hff)
    hff_values = torch.nan_to_num(hff, nan=0.0).float()
    guided = bool((guidance != 1.0).any())
    repeats = 2 if guided else 1
    if guided:
        assert bool(hff_mask.all()), "guided samples need a positive hff"
        model_hff = torch.cat([hff_values, torch.zeros_like(hff_values)])
        model_mask = torch.cat([hff_mask, torch.zeros_like(hff_mask)])
    else:
        model_hff = hff_values
        model_mask = hff_mask
    source = latents_source.repeat(repeats, 1, 1, 1).to(INFERENCE_DTYPE)
    base_input = latents_base.repeat(repeats, 1, 1, 1).to(INFERENCE_DTYPE)
    scale = guidance.float().view(batch_size, 1, 1, 1)
    samples = noise.float()
    sigmas = shift_sigmas(torch.linspace(1.0, 0.0, steps + 1, device=noise.device), shift)
    for current_sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        model_input = samples.repeat(repeats, 1, 1, 1)
        velocity = model(
            model_input.to(INFERENCE_DTYPE), current_sigma.expand(model_input.shape[0]), source, base_input, model_hff, model_mask
        ).float()
        if guided:
            positive, negative = velocity.chunk(2)
            velocity = negative + scale * (positive - negative)
        samples = samples + (next_sigma - current_sigma) * velocity
    return model.from_residual(samples, latents_base)


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
                jobs.put((pair_type, load_job(pair_type, shard_paths[pair_type.name], generator, count=SHARD_SAMPLES[pair_type.name])))
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
