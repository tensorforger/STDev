"""
Rectified flow distillation pairs of the Wan 2.1 latent upscaler: the pretrained teacher sampled from stored
noise on low resolution latents of the pretraining cache, with a per sample hff condition (true, random or
null) and classifier free guidance scale (negative = null hff). One directory per pretraining pair type,
chunks of consecutive planned samples per file.
"""

import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
import vaezip

from stdev.wan_pretrain.cache import LATENT_CODEC, list_shards
from stdev.wan_pretrain.config import BENCHMARK_SAMPLER, PAIR_TYPES, PairType, production_shares

PAIR_SUFFIX = ".vz.safetensors"
EXACT_CODEC = "cast_fp16"
PLAN_SEED = 0
SHARD_COUNT_THREADS = 16

NULL_HFF_PROBABILITY = 0.1
RANDOM_HFF_PROBABILITY = 0.45
GUIDED_PROBABILITY = 0.5
GUIDANCE_RANGE = (1.0, 3.0)

TEACHER_STEPS = BENCHMARK_SAMPLER.steps
TEACHER_SHIFT = BENCHMARK_SAMPLER.shift

CHUNK_SAMPLES = {
    "image_512_1024": 64,
    "image_256_512": 64,
    "video_17_33_512": 32,
    "video_9_17_1024": 16,
    "video_17_512_1024": 16,
    "video_9_512_17_1024": 16,
}
TEACHER_BATCH_SIZE = {
    "image_512_1024": 8,
    "image_256_512": 32,
    "video_17_33_512": 4,
    "video_9_17_1024": 2,
    "video_17_512_1024": 2,
    "video_9_512_17_1024": 2,
}


@dataclass(frozen=True)
class PlannedSample:
    """
    One distillation sample: the cache sample it is built from and its generation settings.
    hff_mode: "true" (Immerkaer hff of the target), "random" (uniform [0, 1]) or "null" (null embedding);
    hff is NaN for "null". guidance 1.0 = no CFG. noise_seed seeds the unit gaussian noise.
    """

    pair_type: str
    shard_name: str
    shard_index: int
    noise_seed: int
    hff_mode: str
    random_hff: float
    guidance: float


@dataclass(frozen=True)
class Chunk:
    pair_type: str
    chunk_index: int
    samples: tuple[PlannedSample, ...]

    @property
    def file_name(self) -> str:
        return f"chunk_{self.chunk_index:05d}{PAIR_SUFFIX}"


@dataclass
class PairChunk:
    """
    latents_source: [N, 16, T_lr, h_lr, w_lr], normalized Wan 2.1 latents (pretraining cache)
    noise: [N, 16, T_hr, h_hr, w_hr], unit gaussian in residual space, quantized by NOISE_CODEC before
        sampling (stored values are exactly the teacher input); the teacher starts from noise_scale * noise at sigma 1
    latents_generated: [N, 16, T_hr, h_hr, w_hr], normalized Wan 2.1 latents sampled by the teacher
    hff: [N], values: [0...1] or NaN (null embedding), the hff the teacher was conditioned on
    hff_true: [N], values: [0...1], Immerkaer hff of the real target (the real target latents stay in the cache,
        referenced by the plan json next to each chunk)
    guidance: [N], CFG scale (1.0 = none), negative = null hff
    noise_scale: [N]
    """

    latents_source: torch.Tensor
    noise: torch.Tensor
    latents_generated: torch.Tensor
    hff: torch.Tensor
    hff_true: torch.Tensor
    guidance: torch.Tensor
    noise_scale: torch.Tensor


EXACT_FIELDS = ("latents_source", "latents_generated")
SCALAR_FIELDS = ("hff", "hff_true", "guidance", "noise_scale")
SCALAR_CODEC = "cast_fp32"
NOISE_CODEC = LATENT_CODEC


def sample_counts(total_samples: int) -> dict[str, int]:
    """
    Samples per pair type in the pretraining production shares (at least one chunk per type).
    """
    shares = production_shares()
    return {
        pair_type.name: max(CHUNK_SAMPLES[pair_type.name], round(total_samples * shares[pair_type.name]))
        for pair_type in PAIR_TYPES
    }


def noise_scale_of(pair_type: PairType) -> float:
    return BENCHMARK_SAMPLER.noise_scale_spatial if pair_type.is_spatial else BENCHMARK_SAMPLER.noise_scale_temporal


def shard_sample_count(shard_path: Path) -> int:
    return int(vaezip.load(shard_path)["hff"].decode(dtype=torch.float32).shape[0])


def plan_pair_type(pair_type: PairType, count: int, cache_directory: Path, seed_offset: int) -> list[PlannedSample]:
    """
    Spreads the samples over all cache shards of the type (random shard order, a few random samples per shard,
    real shard sizes: the last shards of a writer can be partial) and draws the generation settings.
    """
    generator = random.Random(f"{PLAN_SEED}-{pair_type.name}")
    shard_paths = list_shards(cache_directory / pair_type.name)
    if not shard_paths:
        raise RuntimeError(f"no cache shards for {pair_type.name}")
    with ThreadPoolExecutor(SHARD_COUNT_THREADS) as executor:
        sizes = dict(zip(shard_paths, executor.map(shard_sample_count, shard_paths)))
    if sum(sizes.values()) < count:
        raise RuntimeError(f"{pair_type.name}: {count} samples exceed the cache ({sum(sizes.values())})")
    generator.shuffle(shard_paths)
    per_shard = math.ceil(count / len(shard_paths))
    while sum(min(per_shard, size) for size in sizes.values()) < count:
        per_shard += 1
    samples: list[PlannedSample] = []
    for shard_path in shard_paths:
        for shard_index in sorted(generator.sample(range(sizes[shard_path]), min(per_shard, sizes[shard_path]))):
            if len(samples) == count:
                break
            mode_draw = generator.random()
            hff_mode = "null" if mode_draw < NULL_HFF_PROBABILITY else (
                "random" if mode_draw < NULL_HFF_PROBABILITY + RANDOM_HFF_PROBABILITY else "true"
            )
            random_hff = generator.random()
            guided = hff_mode != "null" and generator.random() < GUIDED_PROBABILITY
            guidance = generator.uniform(*GUIDANCE_RANGE) if guided else 1.0
            samples.append(
                PlannedSample(pair_type.name, shard_path.name, shard_index, seed_offset + len(samples), hff_mode, random_hff, guidance)
            )
    return samples


def plan_chunks(total_samples: int, cache_directory: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for type_index, pair_type in enumerate(PAIR_TYPES):
        count = sample_counts(total_samples)[pair_type.name]
        samples = plan_pair_type(pair_type, count, cache_directory, seed_offset=type_index * 10_000_000)
        size = CHUNK_SAMPLES[pair_type.name]
        for chunk_index, start in enumerate(range(0, len(samples), size)):
            chunks.append(Chunk(pair_type.name, chunk_index, tuple(samples[start : start + size])))
    return chunks


def chunk_cost(chunk: Chunk) -> float:
    """
    Teacher target tokens times model evaluations (CFG doubles them).
    """
    tokens = next(pair_type for pair_type in PAIR_TYPES if pair_type.name == chunk.pair_type).tokens_per_sample
    return sum(tokens * TEACHER_STEPS * (2 if sample.guidance != 1.0 else 1) for sample in chunk.samples)


def assign_chunks(chunks: list[Chunk], world_size: int) -> list[list[Chunk]]:
    """
    Greedy cost balance over ranks, most expensive chunks first.
    """
    assignments: list[list[Chunk]] = [[] for _ in range(world_size)]
    loads = [0.0] * world_size
    for chunk in sorted(chunks, key=chunk_cost, reverse=True):
        rank = loads.index(min(loads))
        assignments[rank].append(chunk)
        loads[rank] += chunk_cost(chunk)
    return assignments


def quantize_noise(noise: torch.Tensor) -> tuple[object, torch.Tensor]:
    """
    args:
        noise: [N, 16, T, H, W], unit gaussian
    returns:
        encoded noise (NOISE_CODEC), decoded noise: [N, 16, T, H, W], dtype: torch.float32 (the teacher input)
    """
    encoded = vaezip.encode(noise.float().cpu().contiguous(), codec=NOISE_CODEC)
    return encoded, encoded.decode(dtype=torch.float32)


def write_pair_chunk(directory: Path, chunk: Chunk, pair_chunk: PairChunk, encoded_noise: object) -> Path:
    """
    encoded_noise: the NOISE_CODEC encoding whose decoded values are pair_chunk.noise.
    """
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: vaezip.encode(getattr(pair_chunk, name).float().cpu().contiguous(), codec=EXACT_CODEC) for name in EXACT_FIELDS
    }
    tensors.update(
        {name: vaezip.encode(getattr(pair_chunk, name).float().cpu().contiguous(), codec=SCALAR_CODEC) for name in SCALAR_FIELDS}
    )
    tensors["noise"] = encoded_noise
    path = directory / chunk.file_name
    temporary_path = directory / f".{chunk.file_name}.tmp"
    vaezip.save(tensors, path=temporary_path)
    plan_path(path).write_text(json.dumps([sample.__dict__ for sample in chunk.samples]))
    temporary_path.replace(path)
    return path


def plan_path(chunk_path: Path) -> Path:
    return chunk_path.with_name(chunk_path.name.removesuffix(PAIR_SUFFIX) + ".json")


def read_pair_chunk(path: Path) -> PairChunk:
    encoded = vaezip.load(path)
    return PairChunk(**{name: encoded[name].decode(dtype=torch.float32) for name in PairChunk.__dataclass_fields__})
