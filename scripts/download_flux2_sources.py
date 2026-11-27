"""
Downloads the image sources of the FLUX.2 latent datasets (autoresearch_3).

Sources:
    relaion: 20000 images (min side >= 1024) from shard 0 of the gated `laion/relaion-high-resolution`, shuffled
        with seed 0 (same url order as the Wan 2.1 dataset). Raw files go to `datasets/relaion_high_resolution_20k`.
    4klsdb: the `hr` images of the first `train_x4` parquet files of `SingleBicycle/4KLSDB` (about 629 rows per
        file). The 4K images are center cropped and downscaled to 1024x1024 (`crop_maximal_rectangle`) on the fly
        and stored as PNG in `datasets/4klsdb_1024`; parquet files are deleted after processing.

Each process handles every `world_size`-th url / parquet file; an interrupted run is resumed by the same command.

Run (repository root, `.env` with HF_TOKEN):
    for rank in 0 1 2 3 4 5 6 7; do python scripts/download_flux2_sources.py relaion --rank $rank --world-size 8 & done
    for rank in 0 1 2 3 4 5 6 7; do python scripts/download_flux2_sources.py 4klsdb --rank $rank --world-size 8 & done
Autoresearch_4 (Qwen-Image 2.1, 40k 4KLSDB images) uses `--num-files 65`.
"""

import argparse
import asyncio
import hashlib
import io
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp
import cv2
import numpy as np
import pyarrow.parquet as parquet
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from tqdm import tqdm

from stdev.utils.image_tools import crop_maximal_rectangle

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELAION_DIRECTORY = REPOSITORY_ROOT / "datasets" / "relaion_high_resolution_20k"
LSDB_DIRECTORY = REPOSITORY_ROOT / "datasets" / "4klsdb_1024"
LSDB_PARQUET_DIRECTORY = REPOSITORY_ROOT / "datasets" / "4klsdb_parquet"

RELAION_DATASET_NAME = "laion/relaion-high-resolution"
RELAION_SHARD_INDEX = 0
LSDB_DATASET_NAME = "SingleBicycle/4KLSDB"
LSDB_TRAIN_FILE_PREFIX = "data/train_x4-"
LSDB_NUM_FILES = 33

NUM_RELAION_IMAGES = 20000
IMAGE_SIZE = 1024
SEED = 0

DOWNLOAD_CONCURRENCY = 64
MAX_CONNECTIONS_PER_HOST = 4
CONNECT_TIMEOUT_SECONDS = 5.0
TOTAL_TIMEOUT_SECONDS = 20.0
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MIN_IMAGE_SIDE = 1024
SUPPORTED_IMAGE_FORMATS = {"jpeg": "jpg", "png": "png", "webp": "webp"}
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) research-dataset-downloader"
LSDB_DECODE_THREADS = 6


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downloads the FLUX.2 dataset image sources.")
    parser.add_argument("source", choices=["relaion", "4klsdb"])
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--num-files", type=int, default=LSDB_NUM_FILES, help="4klsdb: number of train_x4 parquet files."
    )
    return parser.parse_args()


def url_hash(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


async def fetch_image(session: aiohttp.ClientSession, url: str) -> tuple[str, bytes, str]:
    """
    returns:
        outcome: "ok" or a failure reason, image bytes, file extension
    """
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return f"http_{response.status}", b"", ""
            if (response.content_length or 0) > MAX_IMAGE_BYTES:
                return "too_large", b"", ""
            buffer = bytearray()
            async for chunk in response.content.iter_chunked(64 * 1024):
                buffer.extend(chunk)
                if len(buffer) > MAX_IMAGE_BYTES:
                    return "too_large", b"", ""
    except Exception:
        return "network_error", b"", ""
    try:
        with Image.open(io.BytesIO(buffer)) as image:
            image.verify()
            width, height = image.size
            image_format = (image.format or "").lower()
    except Exception:
        return "not_a_valid_image", b"", ""
    if image_format not in SUPPORTED_IMAGE_FORMATS:
        return "unsupported_format", b"", ""
    if min(width, height) < MIN_IMAGE_SIDE:
        return "too_small", b"", ""
    return "ok", bytes(buffer), SUPPORTED_IMAGE_FORMATS[image_format]


async def download_relaion_rank(urls: list[str], quota: int, rank: int) -> None:
    """
    Downloads until this rank holds `quota` images. Files are named `<url hash>-<rank>.<extension>`.
    """
    existing_hashes = {
        path.stem.rsplit("-", 1)[0]
        for path in RELAION_DIRECTORY.glob(f"*-{rank}.*")
        if path.suffix != ".tmp"
    }
    pending_count = quota - len(existing_hashes)
    if pending_count <= 0:
        return
    url_iterator = iter(url for url in urls if url_hash(url) not in existing_hashes)
    success_count = 0
    progress_bar = tqdm(total=pending_count, desc=f"relaion [rank {rank}]", position=rank)

    async def worker(session: aiohttp.ClientSession) -> None:
        nonlocal success_count
        for url in url_iterator:
            if success_count >= pending_count:
                return
            outcome, content, extension = await fetch_image(session, url)
            if outcome != "ok" or success_count >= pending_count:
                continue
            file_path = RELAION_DIRECTORY / f"{url_hash(url)}-{rank}.{extension}"
            temporary_path = file_path.with_name(file_path.name + ".tmp")
            temporary_path.write_bytes(content)
            temporary_path.rename(file_path)
            success_count += 1
            progress_bar.update(1)

    connector = aiohttp.TCPConnector(
        limit=DOWNLOAD_CONCURRENCY,
        limit_per_host=MAX_CONNECTIONS_PER_HOST,
        ttl_dns_cache=600,
        ssl=False,
    )
    timeout = aiohttp.ClientTimeout(total=TOTAL_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, headers={"User-Agent": USER_AGENT}
    ) as session:
        await asyncio.gather(*(worker(session) for _ in range(DOWNLOAD_CONCURRENCY)))
    progress_bar.close()
    if success_count < pending_count:
        raise RuntimeError(f"[rank {rank}] urls ran out, downloaded {success_count}/{pending_count}")


def download_relaion(rank: int, world_size: int) -> None:
    RELAION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    shard_name = sorted(
        name
        for name in HfApi().list_repo_files(RELAION_DATASET_NAME, repo_type="dataset")
        if name.endswith(".parquet")
    )[RELAION_SHARD_INDEX]
    shard_path = hf_hub_download(RELAION_DATASET_NAME, shard_name, repo_type="dataset")
    shard_table = parquet.read_table(shard_path)
    url_column = next(name for name in shard_table.column_names if name.lower() == "url")
    urls: list[str] = shard_table.column(url_column).to_pylist()
    random.Random(SEED).shuffle(urls)
    quota = -(-NUM_RELAION_IMAGES // world_size)
    asyncio.run(download_relaion_rank(urls[rank::world_size], quota, rank))


def save_lsdb_image(content: bytes, output_path: Path) -> bool:
    """
    Decodes one 4K image, center crops and downscales it to IMAGE_SIZE and writes it as PNG.

    returns:
        whether the image was readable and large enough
    """
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < IMAGE_SIZE:
        return False
    image = crop_maximal_rectangle(image, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE)
    temporary_path = output_path.with_name(output_path.stem + ".tmp.png")
    cv2.imwrite(str(temporary_path), image)
    temporary_path.rename(output_path)
    return True


def process_lsdb_file(file_index: int, file_name: str, executor: ThreadPoolExecutor) -> None:
    """
    Converts every row of one parquet file into `<file index>_<row index>.png`; a `.done` marker makes
    the file skippable on resume.
    """
    done_marker = LSDB_DIRECTORY / f".file_{file_index:03d}.done"
    if done_marker.exists():
        return
    parquet_path = Path(
        hf_hub_download(
            LSDB_DATASET_NAME,
            file_name,
            repo_type="dataset",
            local_dir=str(LSDB_PARQUET_DIRECTORY),
        )
    )
    parquet_file = parquet.ParquetFile(parquet_path)
    row_index = 0
    futures = []
    for group_index in range(parquet_file.metadata.num_row_groups):
        table = parquet_file.read_row_group(group_index, columns=["hr"])
        for item in table.column("hr").to_pylist():
            output_path = LSDB_DIRECTORY / f"{file_index:03d}_{row_index:04d}.png"
            if not output_path.exists():
                futures.append(executor.submit(save_lsdb_image, item["bytes"], output_path))
            row_index += 1
        del table
    failures = sum(not future.result() for future in futures)
    if failures:
        print(f"{file_name}: {failures} unreadable images skipped", flush=True)
    done_marker.touch()
    parquet_path.unlink()


def download_lsdb(rank: int, world_size: int, num_files: int) -> None:
    LSDB_DIRECTORY.mkdir(parents=True, exist_ok=True)
    train_files = sorted(
        name
        for name in HfApi().list_repo_files(LSDB_DATASET_NAME, repo_type="dataset")
        if name.startswith(LSDB_TRAIN_FILE_PREFIX) and name.endswith(".parquet")
    )[:num_files]
    rank_files = list(enumerate(train_files))[rank::world_size]
    with ThreadPoolExecutor(LSDB_DECODE_THREADS) as executor:
        for file_index, file_name in tqdm(rank_files, desc=f"4klsdb [rank {rank}]", position=rank):
            process_lsdb_file(file_index, file_name, executor)


def main() -> None:
    arguments = parse_arguments()
    load_dotenv(REPOSITORY_ROOT / ".env")
    if arguments.source == "relaion":
        download_relaion(arguments.rank, arguments.world_size)
    else:
        download_lsdb(arguments.rank, arguments.world_size, arguments.num_files)


if __name__ == "__main__":
    main()
