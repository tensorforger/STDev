"""
Rolling latent cache of the FLUX.2 pretraining: one directory per pair type with shards of encoded pairs
(source latents, target latents, target hff, dataset and source variant ids), written atomically by the
producer, read at random by the trainer, oldest shards evicted above a total size limit.
"""

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import vaezip
from torch.utils.data import IterableDataset, get_worker_info
from vaezip.codecs.shortcuts import SHORTCUT_CONFIGS

from stdev.wan_pretrain.cache import SHARD_SUFFIX, cache_size_bytes, evict_oldest_shards, list_shards

LATENT_CODEC = {**SHORTCUT_CONFIGS["lbrq_8bit"], "block_size": [1, 8, 8]}
SCALAR_CODEC = "cast_fp16"
SHARD_KEYS = ("latents_source", "latents_target", "hff", "dataset_index", "variant_index")

__all__ = ["EncodedShard", "write_shard", "read_shard", "list_shards", "cache_size_bytes", "evict_oldest_shards", "PairTypeDataset"]


@dataclass
class EncodedShard:
    """
    latents_source: [N, 32, h, w], latents_target: [N, 32, 2h, 2w] (normalized FLUX.2 latents),
    hff: [N], values: [0...1], Immerkaer feature of the target pixels,
    dataset_index: [N], index into DATASETS, variant_index: [N], index into SOURCE_VARIANTS.
    """

    latents_source: torch.Tensor
    latents_target: torch.Tensor
    hff: torch.Tensor
    dataset_index: torch.Tensor
    variant_index: torch.Tensor


def write_shard(directory: Path, writer_name: str, shard: EncodedShard) -> Path:
    """
    Writes through a hidden temporary file; names sort by creation time.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{time.time_ns()}_{writer_name}"
    temporary_path = directory / f".{stem}.tmp"
    vaezip.save(
        {
            "latents_source": vaezip.encode(shard.latents_source.float().cpu()[:, :, None], codec=LATENT_CODEC),
            "latents_target": vaezip.encode(shard.latents_target.float().cpu()[:, :, None], codec=LATENT_CODEC),
            "hff": vaezip.encode(shard.hff.float().cpu(), codec=SCALAR_CODEC),
            "dataset_index": vaezip.encode(shard.dataset_index.float().cpu(), codec=SCALAR_CODEC),
            "variant_index": vaezip.encode(shard.variant_index.float().cpu(), codec=SCALAR_CODEC),
        },
        path=temporary_path,
    )
    shard_path = directory / f"{stem}{SHARD_SUFFIX}"
    temporary_path.replace(shard_path)
    return shard_path


def read_shard(path: Path) -> EncodedShard:
    encoded = vaezip.load(path)
    values = [encoded[name].decode(dtype=torch.float32) for name in SHARD_KEYS]
    return EncodedShard(values[0][:, :, 0], values[1][:, :, 0], values[2], values[3].round().long(), values[4].round().long())


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
            latents_source: [32, h, w], latents_target: [32, 2h, 2w], hff: [], dataset_index: [], variant_index: []
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
                yield {name: getattr(shard, name)[index] for name in SHARD_KEYS}
