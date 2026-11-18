"""
Training data of the Wan distillation: every rank holds a fixed partition of the pair chunks (chunk i of a pair
type goes to rank i mod world size) in memory. An epoch is a deterministic, rank independent sequence of
micro-steps (one pair type per micro-step on all ranks, as in pretraining) in which every rank uses each of its
samples at most once; the number of micro-steps per type is the minimum over ranks.
"""

import random
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from stdev.wan_distill.pairs import PAIR_SUFFIX, read_pair_chunk
from stdev.wan_pretrain.config import PAIR_TYPES, PairType

LOAD_THREADS = 8
TENSOR_FIELDS = ("latents_source", "noise", "latents_generated")
SCALAR_FIELDS = ("hff", "guidance", "noise_scale")


class RankPairs:
    """
    Per pair type tensors of one rank: latents_source [N, 16, T_lr, h_lr, w_lr], noise and latents_generated
    [N, 16, T_hr, h_hr, w_hr] (fp16), hff [N] (NaN = null), guidance [N], noise_scale [N] (fp32).
    """

    def __init__(self, pairs_directory: Path, rank: int, world_size: int) -> None:
        self.tensors: dict[str, dict[str, torch.Tensor]] = {}
        for pair_type in PAIR_TYPES:
            chunk_paths = sorted((pairs_directory / pair_type.name).glob(f"*{PAIR_SUFFIX}"))[rank::world_size]
            if not chunk_paths:
                self.tensors[pair_type.name] = {}
                continue
            with ThreadPoolExecutor(LOAD_THREADS) as executor:
                chunks = list(executor.map(read_pair_chunk, chunk_paths))
            fields = {name: torch.cat([getattr(chunk, name) for chunk in chunks]).half() for name in TENSOR_FIELDS}
            fields.update({name: torch.cat([getattr(chunk, name) for chunk in chunks]).float() for name in SCALAR_FIELDS})
            self.tensors[pair_type.name] = fields

    def count(self, pair_type: PairType) -> int:
        fields = self.tensors[pair_type.name]
        return int(fields["hff"].shape[0]) if fields else 0

    def batch(self, pair_type: PairType, epoch: int, rank: int, batch_index: int) -> dict[str, torch.Tensor]:
        """
        Micro-batch `batch_index` of the epoch's per rank permutation of this type.

        returns:
            the TENSOR_FIELDS and SCALAR_FIELDS of micro_batch_size samples (CPU)
        """
        size = pair_type.micro_batch_size
        generator = torch.Generator().manual_seed(zlib.crc32(f"{epoch}-{rank}-{pair_type.name}".encode()))
        order = torch.randperm(self.count(pair_type), generator=generator)
        indices = order[batch_index * size : (batch_index + 1) * size]
        return {name: tensor[indices] for name, tensor in self.tensors[pair_type.name].items()}


def epoch_schedule(micro_batches_per_type: dict[str, int], epoch: int, seed: int) -> list[tuple[PairType, int]]:
    """
    returns:
        (pair type, micro-batch index within the type) for every micro-step of the epoch, same on all ranks
    """
    names = [name for name, count in micro_batches_per_type.items() for _ in range(count)]
    random.Random(f"{seed}-{epoch}").shuffle(names)
    by_name = {pair_type.name: pair_type for pair_type in PAIR_TYPES}
    occurrences: dict[str, int] = {}
    schedule: list[tuple[PairType, int]] = []
    for name in names:
        schedule.append((by_name[name], occurrences.get(name, 0)))
        occurrences[name] = occurrences.get(name, 0) + 1
    return schedule
