"""
Rolling latent cache: shards of encoded samples written atomically by the encoders, read at
random by the trainer, oldest shards evicted above a size limit.
"""

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import vaezip
from torch.utils.data import IterableDataset, get_worker_info

from stdev.pretrain.high_frequency import QuantileNormalizer

SHARD_SUFFIX = ".vz.safetensors"
LATENT_CODEC = "lbrq_8bit"
TEXT_CODEC = "cast_fp16"
FEATURE_CODEC = "cast_fp32"


@dataclass
class EncodedShard:
    """
    latents_low: [N, 32, H_lr / 8, W_lr / 8], latents_high: [N, 32, H_hr / 8, W_hr / 8],
    text_embedding: [N, 768] mean pooled SDXL text embeddings,
    high_frequency_raw: [N, 4] raw high frequency features of the high resolution image.
    """

    latents_low: torch.Tensor
    latents_high: torch.Tensor
    text_embedding: torch.Tensor
    high_frequency_raw: torch.Tensor


def write_shard(
    directory: Path, writer_name: str, shard: EncodedShard, metadata: list[dict[str, str]]
) -> Path:
    """
    Writes the shard and a json sidecar (urls, captions), the shard last and atomically so that
    readers never see partial files. Names sort by creation time.
    """
    stem = f"{time.time_ns()}_{writer_name}"
    (directory / f"{stem}.json").write_text(json.dumps(metadata))
    temporary_path = directory / f".{stem}.tmp"
    vaezip.save(
        {
            "latents_low": vaezip.encode(shard.latents_low.float(), codec=LATENT_CODEC),
            "latents_high": vaezip.encode(shard.latents_high.float(), codec=LATENT_CODEC),
            "text_embedding": vaezip.encode(shard.text_embedding.float(), codec=TEXT_CODEC),
            "high_frequency_raw": vaezip.encode(
                shard.high_frequency_raw.float(), codec=FEATURE_CODEC
            ),
        },
        path=temporary_path,
    )
    shard_path = directory / f"{stem}{SHARD_SUFFIX}"
    temporary_path.replace(shard_path)
    return shard_path


def read_shard(path: Path) -> EncodedShard:
    encoded = vaezip.load(path)
    return EncodedShard(
        *(
            encoded[name].decode(dtype=torch.float32)
            for name in ("latents_low", "latents_high", "text_embedding", "high_frequency_raw")
        )
    )


def list_shards(directory: Path) -> list[Path]:
    """
    returns:
        shard paths, oldest first
    """
    return sorted(directory.glob(f"*{SHARD_SUFFIX}"))


def cache_size_bytes(directory: Path) -> int:
    total = 0
    for entry in os.scandir(directory):
        try:
            total += entry.stat().st_size
        except FileNotFoundError:
            continue
    return total


def evict_oldest_shards(directory: Path, size_limit_bytes: int) -> int:
    """
    Deletes the oldest shards (and their sidecars) until the cache fits the limit.

    returns:
        number of deleted shards
    """
    total = cache_size_bytes(directory)
    deleted = 0
    for shard_path in list_shards(directory):
        if total <= size_limit_bytes:
            break
        sidecar_path = shard_path.with_name(shard_path.name.removesuffix(SHARD_SUFFIX) + ".json")
        for path in (shard_path, sidecar_path):
            try:
                total -= path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                continue
        deleted += 1
    return deleted


def read_high_frequency_features(directory: Path, sample_limit: int) -> np.ndarray:
    """
    returns:
        raw features of the oldest shards: [N, 4], N >= sample_limit if available
    """
    features: list[np.ndarray] = []
    count = 0
    for shard_path in list_shards(directory):
        encoded = vaezip.load(shard_path)
        shard_features = encoded["high_frequency_raw"].decode(dtype=torch.float32).numpy()
        features.append(shard_features)
        count += len(shard_features)
        if count >= sample_limit:
            break
    return np.concatenate(features)[:sample_limit]


class CachedLatentDataset(IterableDataset):
    """
    Endless stream of samples from random shards of the cache. Each data loader worker picks
    shards uniformly from the current cache content, so every sample is reused on average
    (consumed samples / produced samples) times.
    """

    def __init__(self, cache_directory: Path, quantiles_path: Path, seed: int) -> None:
        self.cache_directory = cache_directory
        self.quantiles_path = quantiles_path
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        yields:
            latents_low: [C, h, w], latents_high: [C, H, W], text_embedding: [768],
            high_frequency_features: [4], values: [0...1] or NaN
        """
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        generator = random.Random(f"{self.seed}-{worker_id}-{time.time_ns()}")
        normalizer = QuantileNormalizer.load(self.quantiles_path)
        while True:
            shard_paths = list_shards(self.cache_directory)
            if not shard_paths:
                time.sleep(5.0)
                continue
            try:
                shard = read_shard(generator.choice(shard_paths))
            except (FileNotFoundError, OSError):
                continue
            features = torch.from_numpy(normalizer.normalize(shard.high_frequency_raw.numpy()))
            order = list(range(shard.latents_high.shape[0]))
            generator.shuffle(order)
            for index in order:
                yield {
                    "latents_low": shard.latents_low[index],
                    "latents_high": shard.latents_high[index],
                    "text_embedding": shard.text_embedding[index],
                    "high_frequency_features": features[index],
                }
