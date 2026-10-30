"""
ReLAION-high-resolution url source: metadata shards from the Hugging Face Hub, filtered and
shuffled per shard, iterated from a resumable position.
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as parquet
from huggingface_hub import HfApi, hf_hub_download

from stdev.pretrain.config import (
    MAX_UNSAFE_PROBABILITY,
    MAX_WATERMARK_PROBABILITY,
    METADATA_SHUFFLE_SEED,
    RELAION_DATASET_NAME,
)


@dataclass
class ImageJob:
    url: str
    caption: str


@dataclass
class SourcePosition:
    """
    Position of the next row to read: list index into the shard order and row offset inside it.
    """

    shard_order_index: int = 0
    row_offset: int = 0

    def save(self, path: Path) -> None:
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(self.__dict__))
        temporary_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SourcePosition":
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text()))


def list_metadata_files() -> list[str]:
    return sorted(
        file_name
        for file_name in HfApi().list_repo_files(RELAION_DATASET_NAME, repo_type="dataset")
        if file_name.endswith(".parquet")
    )


def load_filtered_shard(file_name: str, min_source_side: int) -> list[ImageJob]:
    """
    Rows of one metadata shard that pass the size, watermark and safety filters, shuffled with a
    fixed seed.
    """
    shard_path = hf_hub_download(RELAION_DATASET_NAME, file_name, repo_type="dataset")
    table = parquet.read_table(
        shard_path, columns=["URL", "TEXT", "WIDTH", "HEIGHT", "pwatermark", "punsafe"]
    ).to_pydict()
    jobs: list[ImageJob] = []
    for url, caption, width, height, watermark, unsafe in zip(
        table["URL"],
        table["TEXT"],
        table["WIDTH"],
        table["HEIGHT"],
        table["pwatermark"],
        table["punsafe"],
    ):
        if url is None or width is None or height is None:
            continue
        if min(width, height) < min_source_side:
            continue
        if watermark is None or math.isnan(watermark) or watermark >= MAX_WATERMARK_PROBABILITY:
            continue
        if unsafe is None or math.isnan(unsafe) or unsafe >= MAX_UNSAFE_PROBABILITY:
            continue
        jobs.append(ImageJob(url, caption or ""))
    random.Random(f"{METADATA_SHUFFLE_SEED}-{file_name}").shuffle(jobs)
    return jobs


def iterate_jobs(
    shard_indices: list[int], min_source_side: int, position: SourcePosition
) -> Iterator[tuple[ImageJob, SourcePosition]]:
    """
    Yields jobs from the shards in the given order, starting at position, with the position of
    the row after each job.
    """
    file_names = list_metadata_files()
    for shard_order_index in range(position.shard_order_index, len(shard_indices)):
        jobs = load_filtered_shard(file_names[shard_indices[shard_order_index]], min_source_side)
        start_offset = position.row_offset if shard_order_index == position.shard_order_index else 0
        for row_offset in range(start_offset, len(jobs)):
            yield jobs[row_offset], SourcePosition(shard_order_index, row_offset + 1)
