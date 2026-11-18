"""
Generates the rectified flow distillation pairs of the Wan 2.1 latent upscaler (stdev.wan_distill.pairs) from
the pretraining latent cache with the pretrained teacher (EMA weights). Pair types follow the pretraining
production shares; every rank writes its cost balanced share of chunks, existing chunks are skipped (rerun the
same command to resume).

--measure     times the teacher per pair type on one GPU and prints the sample count for --budget-hours on
              --gpus GPUs (no output written)

Usage: torchrun --standalone --nproc-per-node 8 scripts/generate_wan_distill_pairs.py --total-samples N
       python scripts/generate_wan_distill_pairs.py --measure [--budget-hours 2 --gpus 8]
       python scripts/generate_wan_distill_pairs.py --total-samples N --smoke   (one chunk per type, separate directory)
"""

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import torch

from stdev.wan_distill.pairs import (
    GUIDANCE_RANGE,
    GUIDED_PROBABILITY,
    NULL_HFF_PROBABILITY,
    TEACHER_BATCH_SIZE,
    TEACHER_SHIFT,
    TEACHER_STEPS,
    Chunk,
    PairChunk,
    PlannedSample,
    assign_chunks,
    chunk_cost,
    noise_scale_of,
    plan_chunks,
    production_shares,
    quantize_noise,
    sample_counts,
    write_pair_chunk,
)
from stdev.wan_distill.teacher import load_teacher, sample_teacher
from stdev.wan_pretrain.cache import EncodedShard, read_shard
from stdev.wan_pretrain.config import PAIR_TYPE_BY_NAME, PAIR_TYPES, PRETRAIN_RUNS_DIRECTORY

DEFAULT_CHECKPOINT = PRETRAIN_RUNS_DIRECTORY / "wan_pretrain_v1" / "checkpoints" / "checkpoint_000161590.pt"
DEFAULT_CACHE = PRETRAIN_RUNS_DIRECTORY / "wan_data" / "cache"
DEFAULT_OUTPUT = PRETRAIN_RUNS_DIRECTORY / "wan_distill_pairs"
DYNAMO_CACHE_SIZE_LIMIT = 64
PREFETCHED_CHUNKS = 2
MEASURE_REPEATS = 2

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-samples", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-directory", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--budget-hours", type=float, default=2.0)
    parser.add_argument("--gpus", type=int, default=8)
    return parser.parse_args()


def unit_noise(seed: int, shape: tuple[int, ...]) -> torch.Tensor:
    """
    returns:
        noise: shape, unit gaussian, dtype: torch.float32
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator)


def load_chunk_inputs(chunk: Chunk, cache_directory: Path) -> tuple[dict[str, torch.Tensor], object]:
    """
    returns:
        latents_source: [N, 16, T_lr, h_lr, w_lr], hff_true: [N], noise: [N, 16, T_hr, h_hr, w_hr] (quantized
        unit gaussian); encoded noise
    """
    shards: dict[str, EncodedShard] = {}
    parts: dict[str, list[torch.Tensor]] = {"latents_source": [], "latents_target": [], "hff_true": []}
    for sample in chunk.samples:
        if sample.shard_name not in shards:
            shards = {sample.shard_name: read_shard(cache_directory / chunk.pair_type / sample.shard_name)}
        shard = shards[sample.shard_name]
        parts["latents_source"].append(shard.latents_source[sample.shard_index])
        parts["latents_target"].append(shard.latents_target[sample.shard_index])
        parts["hff_true"].append(shard.hff[sample.shard_index])
    inputs = {name: torch.stack(values) for name, values in parts.items()}
    target_shape = tuple(inputs.pop("latents_target").shape[1:])
    encoded_noise, inputs["noise"] = quantize_noise(
        torch.stack([unit_noise(sample.noise_seed, target_shape) for sample in chunk.samples])
    )
    return inputs, encoded_noise


def conditioning_hff(sample: PlannedSample, hff_true: float) -> float:
    if sample.hff_mode == "null":
        return float("nan")
    return hff_true if sample.hff_mode == "true" else sample.random_hff


def run_batches(
    teacher: torch.nn.Module,
    latents_source: torch.Tensor,
    initial_residual: torch.Tensor,
    hff: torch.Tensor,
    guidance: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """
    Teacher sampling in fixed size batches (the tail is padded by repeating its last sample, so compiled shapes
    stay static); guided samples run in half size batches (doubled inside the sampler).

    args:
        latents_source: [N, 16, T_lr, h_lr, w_lr] (CPU), initial_residual: [N, 16, T_hr, h_hr, w_hr] (CPU),
        hff: [N] (NaN = null), guidance: [N]
    returns:
        latents_generated: [N, 16, T_hr, h_hr, w_hr], dtype: torch.float32, CPU
    """
    outputs = torch.empty_like(initial_residual)
    guided_indices = torch.nonzero(guidance != 1.0).flatten()
    unguided_indices = torch.nonzero(guidance == 1.0).flatten()
    for indices, size in ((unguided_indices, batch_size), (guided_indices, max(1, batch_size // 2))):
        for start in range(0, len(indices), size):
            batch_indices = indices[start : start + size]
            padded = torch.cat([batch_indices, batch_indices[-1:].expand(size - len(batch_indices))])
            latents = sample_teacher(
                teacher,
                latents_source[padded].to(DEVICE),
                initial_residual[padded].to(DEVICE),
                hff[padded].to(DEVICE),
                guidance[padded].to(DEVICE),
                TEACHER_STEPS,
                TEACHER_SHIFT,
            )
            outputs[batch_indices] = latents[: len(batch_indices)].cpu()
    return outputs


def generate_chunk(teacher: torch.nn.Module, chunk: Chunk, inputs: dict[str, torch.Tensor]) -> PairChunk:
    pair_type = PAIR_TYPE_BY_NAME[chunk.pair_type]
    noise = inputs["noise"]
    noise_scale = noise_scale_of(pair_type)
    hff = torch.tensor([conditioning_hff(sample, value) for sample, value in zip(chunk.samples, inputs["hff_true"].tolist())])
    guidance = torch.tensor([sample.guidance for sample in chunk.samples])
    latents_generated = run_batches(
        teacher, inputs["latents_source"], noise_scale * noise, hff, guidance, TEACHER_BATCH_SIZE[chunk.pair_type]
    )
    return PairChunk(
        latents_source=inputs["latents_source"],
        noise=noise,
        latents_generated=latents_generated,
        hff=hff,
        hff_true=inputs["hff_true"],
        guidance=guidance,
        noise_scale=torch.full((len(chunk.samples),), noise_scale),
    )


def measure(arguments: argparse.Namespace, teacher: torch.nn.Module) -> None:
    """
    Seconds per sample of every pair type with and without guidance, and the sample count of the budget.
    """
    seconds_per_sample: dict[str, dict[str, float]] = {}
    for pair_type in PAIR_TYPES:
        batch_size = TEACHER_BATCH_SIZE[pair_type.name]
        source = torch.randn(batch_size, 16, *pair_type.source_latent_size)
        noise = torch.randn(batch_size, 16, *pair_type.target_latent_size)
        hff = torch.rand(batch_size)
        timings = {}
        for name, guidance in (("unguided", torch.ones(batch_size)), ("guided", torch.full((batch_size,), 2.0))):
            for _ in range(MEASURE_REPEATS):
                torch.cuda.synchronize()
                start = time.perf_counter()
                run_batches(teacher, source, noise, hff, guidance, batch_size)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            timings[name] = elapsed / batch_size
        seconds_per_sample[pair_type.name] = timings
        print(f"measure: {pair_type.name} {json.dumps({key: round(value, 4) for key, value in timings.items()})} s/sample", flush=True)
    guided_share = (1.0 - NULL_HFF_PROBABILITY) * GUIDED_PROBABILITY
    shares = production_shares()
    mean_seconds = sum(
        shares[name] * ((1.0 - guided_share) * timing["unguided"] + guided_share * timing["guided"])
        for name, timing in seconds_per_sample.items()
    )
    total = int(arguments.budget_hours * 3600 * arguments.gpus / mean_seconds)
    print(
        f"measure: mean {mean_seconds:.4f} s/sample/GPU (guided share {guided_share:.2f}, CFG {GUIDANCE_RANGE}) -> "
        f"{total} samples in {arguments.budget_hours} h on {arguments.gpus} GPUs: {json.dumps(sample_counts(total))}",
        flush=True,
    )


def prefetch(chunks: list[Chunk], cache_directory: Path, output: queue.Queue) -> None:
    """
    Puts (chunk, inputs, encoded noise) per chunk; an exception is put instead and re-raised by the consumer.
    """
    try:
        for chunk in chunks:
            output.put((chunk, *load_chunk_inputs(chunk, cache_directory)))
    except Exception as error:
        output.put(error)


def main() -> None:
    arguments = parse_arguments()
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = DYNAMO_CACHE_SIZE_LIMIT
    teacher = load_teacher(arguments.checkpoint, DEVICE)
    if arguments.measure:
        measure(arguments, teacher)
        return
    assert arguments.total_samples > 0, "--total-samples is required"

    output_directory = arguments.output_directory or (
        DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.name + "_smoke") if arguments.smoke else DEFAULT_OUTPUT
    )
    chunks = plan_chunks(arguments.total_samples, arguments.cache_directory)
    if arguments.smoke:
        chunks = [chunk for chunk in chunks if chunk.chunk_index == 0]
    if RANK == 0:
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "config.json").write_text(
            json.dumps(
                {
                    "total_samples": arguments.total_samples,
                    "sample_counts": sample_counts(arguments.total_samples),
                    "checkpoint": str(arguments.checkpoint),
                    "cache_directory": str(arguments.cache_directory),
                    "teacher_steps": TEACHER_STEPS,
                    "teacher_shift": TEACHER_SHIFT,
                    "chunks": len(chunks),
                },
                indent=1,
            )
        )
    rank_chunks = [
        chunk
        for chunk in assign_chunks(chunks, WORLD_SIZE)[RANK]
        if not (output_directory / chunk.pair_type / chunk.file_name).exists()
    ]
    total_rank_samples = sum(len(chunk.samples) for chunk in rank_chunks)
    total_rank_cost = sum(chunk_cost(chunk) for chunk in rank_chunks)
    print(f"rank {RANK}: {len(rank_chunks)} chunks, {total_rank_samples} samples pending", flush=True)

    inputs_queue: queue.Queue = queue.Queue(maxsize=PREFETCHED_CHUNKS)
    threading.Thread(target=prefetch, args=(rank_chunks, arguments.cache_directory, inputs_queue), daemon=True).start()
    start_time = time.perf_counter()
    done_samples = 0
    done_cost = 0.0
    for chunk_number in range(len(rank_chunks)):
        item = inputs_queue.get()
        if isinstance(item, Exception):
            raise item
        chunk, inputs, encoded_noise = item
        chunk_start = time.perf_counter()
        pair_chunk = generate_chunk(teacher, chunk, inputs)
        write_pair_chunk(output_directory / chunk.pair_type, chunk, pair_chunk, encoded_noise)
        done_samples += len(chunk.samples)
        done_cost += chunk_cost(chunk)
        elapsed = time.perf_counter() - start_time
        remaining_hours = elapsed / done_cost * (total_rank_cost - done_cost) / 3600
        print(
            f"rank {RANK}: {chunk.pair_type}/{chunk.file_name} {len(chunk.samples)} samples in "
            f"{time.perf_counter() - chunk_start:.0f}s, chunk {chunk_number + 1}/{len(rank_chunks)}, "
            f"{done_samples}/{total_rank_samples} samples, {elapsed / 3600:.2f} h, ~{remaining_hours:.2f} h left",
            flush=True,
        )
    print(f"rank {RANK}: done", flush=True)


if __name__ == "__main__":
    main()
