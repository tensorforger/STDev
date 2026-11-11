"""
Blocks of the SFT dataset: one FLUX.2 teacher sample per block (high resolution latents, their
bilinear 2x downscale re-encoded, the pooled SDXL text embedding and raw high frequency features
of the decoded high resolution image), stored per resolution bucket directory.
"""

import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import vaezip
from torch.utils.data import Dataset

from stdev.sft.config import (
    HIGH_FREQUENCY_CODEC,
    LATENT_CODEC,
    RESOLUTION_BUCKETS,
    TEXT_EMBEDDING_CODEC,
    ResolutionBucket,
)

BLOCK_SUFFIX = ".vz.safetensors"
BLOCK_TENSOR_NAMES = ("latents_orig", "latents_div_2", "text_embedding", "high_frequency_raw")


def bucket_directory(dataset_directory: Path, bucket: ResolutionBucket) -> Path:
    return dataset_directory / bucket.name


def block_path(dataset_directory: Path, bucket: ResolutionBucket, block_index: int) -> Path:
    return bucket_directory(dataset_directory, bucket) / f"block_{block_index:06d}{BLOCK_SUFFIX}"


def write_block(
    dataset_directory: Path,
    bucket: ResolutionBucket,
    block_index: int,
    latents_orig: torch.Tensor,
    latents_div_2: torch.Tensor,
    text_embedding: torch.Tensor,
    high_frequency_raw: torch.Tensor,
) -> None:
    """
    args:
        latents_orig: [1, 32, height/8, width/8]
        latents_div_2: [1, 32, height/16, width/16]
        text_embedding: [1, 768], mean pooled SDXL text encoder embedding
        high_frequency_raw: [1, 4], raw (not quantile normalized) high frequency features
    """
    path = block_path(dataset_directory, bucket, block_index)
    encoded_tensors = {
        "latents_orig": vaezip.encode(latents_orig.float(), codec=LATENT_CODEC),
        "latents_div_2": vaezip.encode(latents_div_2.float(), codec=LATENT_CODEC),
        "text_embedding": vaezip.encode(text_embedding.float(), codec=TEXT_EMBEDDING_CODEC),
        "high_frequency_raw": vaezip.encode(high_frequency_raw.float(), codec=HIGH_FREQUENCY_CODEC),
    }
    temporary_path = path.with_name(path.name + ".tmp")
    vaezip.save(encoded_tensors, path=temporary_path)
    temporary_path.replace(path)


def read_block(path: Path) -> dict[str, torch.Tensor]:
    """
    returns:
        latents_orig: [32, h, w], latents_div_2: [32, h/2, w/2], text_embedding: [768],
        high_frequency_raw: [4]
    """
    encoded = vaezip.load(path)
    return {name: encoded[name].decode(dtype=torch.float32)[0] for name in BLOCK_TENSOR_NAMES}


def list_block_paths(dataset_directory: Path, bucket: ResolutionBucket) -> list[Path]:
    return sorted(bucket_directory(dataset_directory, bucket).glob(f"*{BLOCK_SUFFIX}"))


def collect_raw_high_frequency_features(dataset_directory: Path, sample_limit: int) -> np.ndarray:
    """
    Reads raw high frequency features from up to sample_limit blocks spread evenly across the
    buckets that already have generated blocks.

    returns:
        raw_features: [N, 4]
    """
    buckets_with_blocks = [
        bucket for bucket in RESOLUTION_BUCKETS if list_block_paths(dataset_directory, bucket)
    ]
    per_bucket_limit = max(1, sample_limit // max(1, len(buckets_with_blocks)))
    features = [
        read_block(path)["high_frequency_raw"].numpy()
        for bucket in buckets_with_blocks
        for path in list_block_paths(dataset_directory, bucket)[:per_bucket_limit]
    ]
    return np.stack(features)


class SFTBlockDataset(Dataset):
    """
    Map-style dataset over all generated blocks, indexed by (bucket name, position in the
    bucket's sorted file list). Meant to be used with BucketBatchSampler so that every batch
    comes from a single bucket (same latent resolution).
    """

    def __init__(self, dataset_directory: Path) -> None:
        self.block_paths: dict[str, list[Path]] = {
            bucket.name: list_block_paths(dataset_directory, bucket)
            for bucket in RESOLUTION_BUCKETS
        }

    def bucket_sizes(self) -> dict[str, int]:
        return {name: len(paths) for name, paths in self.block_paths.items()}

    def __len__(self) -> int:
        return sum(len(paths) for paths in self.block_paths.values())

    def __getitem__(self, key: tuple[str, int]) -> dict[str, torch.Tensor]:
        bucket_name, position = key
        return read_block(self.block_paths[bucket_name][position])


class BucketBatchSampler:
    """
    Infinite stream of batches of (bucket name, position) keys, one bucket per batch (chosen with
    probability proportional to its block count), reshuffled whenever a bucket's queue empties.
    """

    def __init__(self, bucket_sizes: dict[str, int], batch_size: int, seed: int) -> None:
        self.bucket_sizes = {name: size for name, size in bucket_sizes.items() if size > 0}
        if not self.bucket_sizes:
            raise ValueError("no generated blocks found in any resolution bucket")
        self.batch_size = batch_size
        self.generator = random.Random(seed)
        self.queues: dict[str, list[int]] = {name: [] for name in self.bucket_sizes}

    def _next_index(self, bucket_name: str) -> int:
        if not self.queues[bucket_name]:
            self.queues[bucket_name] = list(range(self.bucket_sizes[bucket_name]))
            self.generator.shuffle(self.queues[bucket_name])
        return self.queues[bucket_name].pop()

    def __iter__(self) -> Iterator[list[tuple[str, int]]]:
        bucket_names = list(self.bucket_sizes)
        weights = [self.bucket_sizes[name] for name in bucket_names]
        while True:
            bucket_name = self.generator.choices(bucket_names, weights=weights)[0]
            yield [(bucket_name, self._next_index(bucket_name)) for _ in range(self.batch_size)]


class EpochBucketBatchSampler:
    """
    One pass over all blocks: single-bucket batches in random order, incomplete batches dropped.
    """

    def __init__(self, bucket_sizes: dict[str, int], batch_size: int, seed: int) -> None:
        self.bucket_sizes = bucket_sizes
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return sum(size // self.batch_size for size in self.bucket_sizes.values())

    def __iter__(self) -> Iterator[list[tuple[str, int]]]:
        generator = random.Random(f"{self.seed}-{self.epoch}")
        batches = []
        for bucket_name, size in self.bucket_sizes.items():
            positions = list(range(size))
            generator.shuffle(positions)
            for start in range(0, size - self.batch_size + 1, self.batch_size):
                batches.append(
                    [
                        (bucket_name, position)
                        for position in positions[start : start + self.batch_size]
                    ]
                )
        generator.shuffle(batches)
        yield from batches


def collate_blocks(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: torch.stack([sample[name] for sample in samples]) for name in samples[0]}
