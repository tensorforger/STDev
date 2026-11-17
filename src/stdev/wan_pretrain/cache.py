"""
Rolling latent cache of the Wan pretraining: one directory per pair type with shards of encoded pairs
(source latents, target latents, target hff), written atomically by the producer, read at random by the
trainer, oldest shards evicted above a total size limit.
"""

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import vaezip
from torch.utils.data import IterableDataset, get_worker_info
from vaezip.codecs.shortcuts import SHORTCUT_CONFIGS

SHARD_SUFFIX = ".vz.safetensors"
LATENT_CODEC = {**SHORTCUT_CONFIGS["lbrq_8bit"], "block_size": [1, 8, 8]}
HFF_CODEC = "cast_fp16"


@dataclass
class EncodedShard:
    """
    latents_source: [N, 16, T_lr, h_lr, w_lr], latents_target: [N, 16, T_hr, h_hr, w_hr] (normalized Wan 2.1
    latents), hff: [N], values: [0...1], Immerkaer feature of the target pixels.
    """

    latents_source: torch.Tensor
    latents_target: torch.Tensor
    hff: torch.Tensor


def write_shard(directory: Path, writer_name: str, shard: EncodedShard) -> Path:
    """
    Writes through a hidden temporary file; names sort by creation time.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{time.time_ns()}_{writer_name}"
    temporary_path = directory / f".{stem}.tmp"
    vaezip.save(
        {
            "latents_source": vaezip.encode(shard.latents_source.float().cpu(), codec=LATENT_CODEC),
            "latents_target": vaezip.encode(shard.latents_target.float().cpu(), codec=LATENT_CODEC),
            "hff": vaezip.encode(shard.hff.float().cpu(), codec=HFF_CODEC),
        },
        path=temporary_path,
    )
    shard_path = directory / f"{stem}{SHARD_SUFFIX}"
    temporary_path.replace(shard_path)
    return shard_path


def read_shard(path: Path) -> EncodedShard:
    encoded = vaezip.load(path)
    return EncodedShard(*(encoded[name].decode(dtype=torch.float32) for name in ("latents_source", "latents_target", "hff")))


def list_shards(directory: Path) -> list[Path]:
    """
    returns:
        shard paths, oldest first
    """
    if not directory.exists():
        return []
    return sorted(directory.glob(f"*{SHARD_SUFFIX}"))


def cache_size_bytes(cache_directory: Path) -> int:
    total = 0
    for root, _, file_names in os.walk(cache_directory):
        for file_name in file_names:
            try:
                total += os.stat(os.path.join(root, file_name)).st_size
            except FileNotFoundError:
                continue
    return total


def evict_oldest_shards(cache_directory: Path, size_limit_bytes: int) -> int:
    """
    Deletes the oldest shards over all pair types until the cache fits the limit.

    returns:
        number of deleted shards
    """
    total = cache_size_bytes(cache_directory)
    if total <= size_limit_bytes:
        return 0
    shards = sorted(cache_directory.glob(f"*/*{SHARD_SUFFIX}"), key=lambda path: path.name)
    deleted = 0
    for shard_path in shards:
        if total <= size_limit_bytes:
            break
        try:
            total -= shard_path.stat().st_size
            shard_path.unlink()
            deleted += 1
        except FileNotFoundError:
            continue
    return deleted



class PairTypeDataset(IterableDataset):
    """
    Endless stream of samples of one pair type: each worker loads a random shard of the current cache
    content and yields its samples in random order.
    """

    def __init__(self, directory: Path, seed: str) -> None:
        self.directory = directory
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        yields:
            latents_source: [16, T_lr, h_lr, w_lr], latents_target: [16, T_hr, h_hr, w_hr], hff: []
        """
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        generator = random.Random(f"{self.seed}-{worker_id}-{time.time_ns()}")
        while True:
            shard_paths = list_shards(self.directory)
            if not shard_paths:
                time.sleep(5.0)
                continue
            try:
                shard = read_shard(generator.choice(shard_paths))
            except (FileNotFoundError, OSError, KeyError, RuntimeError):
                continue
            order = list(range(shard.latents_target.shape[0]))
            generator.shuffle(order)
            for index in order:
                yield {
                    "latents_source": shard.latents_source[index],
                    "latents_target": shard.latents_target[index],
                    "hff": shard.hff[index],
                }
