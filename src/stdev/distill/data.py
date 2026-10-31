"""
Data of the distillation: the 4KLSDB source (local parquet files downloaded from the Hub, row
groups read in order from a resumable position per generation rank), image preprocessing, the
teacher pair shards and the training stream over the pairs of one round.
"""

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pyarrow.parquet as parquet
import torch
import vaezip
from huggingface_hub import HfApi, hf_hub_download
from torch.utils.data import IterableDataset, get_worker_info

from stdev.distill.config import (
    CAPTION_COLUMN,
    CROPS_PER_IMAGE,
    DATASET_DIRECTORY,
    DATASET_NAME,
    PREFETCHED_FILES_PER_RANK,
    STAGE,
    TRAIN_FILE_PREFIX,
)
from stdev.pretrain.high_frequency import compute_feature_vector
from stdev.pretrain.producer import crop_and_resize

PAIR_SUFFIX = ".vz.safetensors"
LATENT_CODEC = "lbrq_8bit"
LATENT_CHANNELS = 32
VAE_SPATIAL_COMPRESSION = 8


@dataclass
class SourcePosition:
    """
    Next row of one generation rank: index into the rank's file list and row offset in the file.
    """

    file_index: int = 0
    row_offset: int = 0


def list_train_files() -> list[str]:
    return sorted(
        file_name
        for file_name in HfApi().list_repo_files(DATASET_NAME, repo_type="dataset")
        if file_name.startswith(TRAIN_FILE_PREFIX) and file_name.endswith(".parquet")
    )


def rank_files(train_files: list[str], rank: int, world_size: int) -> list[str]:
    return train_files[rank::world_size]


def local_path(file_name: str) -> Path:
    return DATASET_DIRECTORY / file_name


def ensure_downloaded(file_name: str) -> Path:
    path = local_path(file_name)
    if not path.exists():
        hf_hub_download(
            DATASET_NAME, file_name, repo_type="dataset", local_dir=str(DATASET_DIRECTORY)
        )
    return path


def files_to_prefetch(files: list[str], position: SourcePosition) -> list[str]:
    return files[position.file_index : position.file_index + PREFETCHED_FILES_PER_RANK]


def iterate_rows(
    files: list[str], position: SourcePosition
) -> Iterator[tuple[SourcePosition, bytes, str]]:
    """
    yields:
        position after the row, encoded HR image, caption
    """
    file_index, row_offset = position.file_index, position.row_offset
    while file_index < len(files):
        parquet_file = parquet.ParquetFile(ensure_downloaded(files[file_index]))
        group_start = 0
        for group_index in range(parquet_file.metadata.num_row_groups):
            group_rows = parquet_file.metadata.row_group(group_index).num_rows
            if group_start + group_rows <= row_offset:
                group_start += group_rows
                continue
            table = parquet_file.read_row_group(group_index, columns=["hr", CAPTION_COLUMN])
            images = table.column("hr").to_pylist()
            captions = table.column(CAPTION_COLUMN).to_pylist()
            for index in range(max(0, row_offset - group_start), group_rows):
                yield (
                    SourcePosition(file_index, group_start + index + 1),
                    images[index]["bytes"],
                    captions[index] or "",
                )
            group_start += group_rows
        file_index, row_offset = file_index + 1, 0


def preprocess_image(content: bytes, seed: str, crop_count: int = CROPS_PER_IMAGE) -> list[dict]:
    """
    Random scale crops of one image (as in pretraining), deterministic for a seed.

    returns:
        crops: dicts with image_high / image_low ([H, W, 3], RGB, uint8) and
        high_frequency_raw ([4], float32); empty when the image is unreadable or too small
    """
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < STAGE.high_resolution:
        return []
    generator = random.Random(seed)
    crops = []
    for _ in range(crop_count):
        image_high, image_low = crop_and_resize(image, STAGE, generator)
        crops.append(
            {
                "image_high": cv2.cvtColor(image_high, cv2.COLOR_BGR2RGB),
                "image_low": cv2.cvtColor(image_low, cv2.COLOR_BGR2RGB),
                "high_frequency_raw": compute_feature_vector(image_high),
            }
        )
    return crops


def make_noise(seed: int, height: int, width: int) -> torch.Tensor:
    """
    Initial noise of one teacher pair, reproducible on CPU from its seed.

    returns:
        noise: [LATENT_CHANNELS, height, width], dtype: torch.float32
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((LATENT_CHANNELS, height, width), generator=generator)


@dataclass
class PairShard:
    """
    Teacher pairs: latents_low [N, 32, h, w], latents_teacher [N, 32, H, W] (teacher output from
    make_noise(seeds[i])), text_embedding [N, 768], high_frequency_features [N, 4] (quantile
    normalized, NaN where undefined), text_mask [N], high_frequency_mask [N, 4] (conditioning the
    teacher used, after dropout), seeds.
    """

    latents_low: torch.Tensor
    latents_teacher: torch.Tensor
    text_embedding: torch.Tensor
    high_frequency_features: torch.Tensor
    text_mask: torch.Tensor
    high_frequency_mask: torch.Tensor
    seeds: list[int]


def write_pair_shard(directory: Path, writer_name: str, shard: PairShard, captions: list[str]) -> Path:
    stem = f"{time.time_ns()}_{writer_name}"
    (directory / f"{stem}.json").write_text(json.dumps({"seeds": shard.seeds, "captions": captions}))
    temporary_path = directory / f".{stem}.tmp"
    vaezip.save(
        {
            "latents_low": vaezip.encode(shard.latents_low.float(), codec=LATENT_CODEC),
            "latents_teacher": vaezip.encode(shard.latents_teacher.float(), codec=LATENT_CODEC),
            "text_embedding": vaezip.encode(shard.text_embedding.float(), codec="cast_fp16"),
            "high_frequency_features": vaezip.encode(
                shard.high_frequency_features.float(), codec="cast_fp32"
            ),
            "text_mask": vaezip.encode(shard.text_mask.float()[:, None], codec="cast_fp32"),
            "high_frequency_mask": vaezip.encode(shard.high_frequency_mask.float(), codec="cast_fp32"),
        },
        path=temporary_path,
    )
    shard_path = directory / f"{stem}{PAIR_SUFFIX}"
    temporary_path.replace(shard_path)
    return shard_path


def read_pair_shard(path: Path) -> PairShard:
    encoded = vaezip.load(path)
    tensors = {name: value.decode(dtype=torch.float32) for name, value in encoded.items()}
    sidecar = json.loads(path.with_name(path.name.removesuffix(PAIR_SUFFIX) + ".json").read_text())
    return PairShard(
        tensors["latents_low"],
        tensors["latents_teacher"],
        tensors["text_embedding"],
        tensors["high_frequency_features"],
        tensors["text_mask"][:, 0] > 0.5,
        tensors["high_frequency_mask"] > 0.5,
        sidecar["seeds"],
    )


def list_pair_shards(directory: Path) -> list[Path]:
    return sorted(directory.glob(f"*{PAIR_SUFFIX}"))


def count_pairs(directory: Path) -> int:
    return sum(
        len(json.loads(path.with_name(path.name.removesuffix(PAIR_SUFFIX) + ".json").read_text())["seeds"])
        for path in list_pair_shards(directory)
    )


class RoundPairDataset(IterableDataset):
    """
    Endless stream over the pair shards of one round. Every pass shuffles the shards (same order on
    all ranks), each rank takes every world_size-th shard and each loader worker every
    num_workers-th of these; samples are shuffled within a shard.
    """

    def __init__(self, directory: Path, rank: int, world_size: int, seed: int) -> None:
        self.shard_paths = list_pair_shards(directory)
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        yields:
            latents_low: [C, h, w], latents_teacher: [C, H, W], noise: [C, H, W],
            text_embedding: [768], high_frequency_features: [4], text_mask: [],
            high_frequency_mask: [4]
        """
        worker_info = get_worker_info()
        worker_id, worker_count = (0, 1) if worker_info is None else (worker_info.id, worker_info.num_workers)
        pass_index = 0
        while True:
            order = list(self.shard_paths)
            random.Random(f"{self.seed}-{pass_index}").shuffle(order)
            rank_shards = order[self.rank :: self.world_size][worker_id::worker_count]
            if not rank_shards:
                rank_shards = [order[(self.rank * worker_count + worker_id) % len(order)]]
            generator = random.Random(f"{self.seed}-{pass_index}-{self.rank}-{worker_id}")
            for shard_path in rank_shards:
                shard = read_pair_shard(shard_path)
                height, width = shard.latents_teacher.shape[-2:]
                indices = list(range(len(shard.seeds)))
                generator.shuffle(indices)
                for index in indices:
                    yield {
                        "latents_low": shard.latents_low[index],
                        "latents_teacher": shard.latents_teacher[index],
                        "noise": make_noise(shard.seeds[index], height, width),
                        "text_embedding": shard.text_embedding[index],
                        "high_frequency_features": shard.high_frequency_features[index],
                        "text_mask": shard.text_mask[index],
                        "high_frequency_mask": shard.high_frequency_mask[index],
                    }
            pass_index += 1
