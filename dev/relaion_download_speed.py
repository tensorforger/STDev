"""
ReLAION-high-resolution download throughput on this machine for a fixed duration, with the
downloader of notebooks/download_relaion_high_resolution_subset.ipynb split over several processes.
Images are validated (PIL verify) and counted, not saved.

Usage: python dev/relaion_download_speed.py --processes 4 --concurrency 512 [--seconds 90]
           [--url-offset 0]
"""

import argparse
import asyncio
import io
import json
import multiprocessing
import random
import time
from collections import Counter

import aiohttp
import pyarrow.parquet as parquet
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

DATASET_NAME = "laion/relaion-high-resolution"
SHARD_INDEX = 0
SEED = 0
MAX_CONNECTIONS_PER_HOST = 4
CONNECT_TIMEOUT_SECONDS = 5.0
TOTAL_TIMEOUT_SECONDS = 20.0
MAX_IMAGE_BYTES = 30 * 1024 * 1024
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) research-dataset-downloader"


def load_shuffled_urls() -> list[str]:
    repository_files = sorted(
        file_name
        for file_name in HfApi().list_repo_files(DATASET_NAME, repo_type="dataset")
        if file_name.endswith(".parquet")
    )
    shard_path = hf_hub_download(DATASET_NAME, repository_files[SHARD_INDEX], repo_type="dataset")
    urls = parquet.read_table(shard_path, columns=["URL"]).column("URL").to_pylist()
    random.Random(SEED).shuffle(urls)
    return urls


async def fetch_image(session: aiohttp.ClientSession, url: str) -> tuple[str, int]:
    """
    returns:
        outcome ("ok" or a failure reason), byte count
    """
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return f"http_{response.status}", 0
            buffer = bytearray()
            async for chunk in response.content.iter_chunked(64 * 1024):
                buffer.extend(chunk)
                if len(buffer) > MAX_IMAGE_BYTES:
                    return "too_large", 0
    except asyncio.TimeoutError:
        return "timeout", 0
    except aiohttp.ClientConnectorError:
        return "connection_or_dns_error", 0
    except aiohttp.ClientError:
        return "client_error", 0
    except Exception:
        return "other_error", 0
    try:
        with Image.open(io.BytesIO(buffer)) as image:
            image.verify()
    except Exception:
        return "not_a_valid_image", 0
    return "ok", len(buffer)


async def download_for(urls: list[str], concurrency: int, seconds: float) -> dict[str, object]:
    connector = aiohttp.TCPConnector(
        limit=concurrency,
        limit_per_host=MAX_CONNECTIONS_PER_HOST,
        ttl_dns_cache=600,
        ssl=False,
        enable_cleanup_closed=True,
    )
    timeout = aiohttp.ClientTimeout(total=TOTAL_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
    url_iterator = iter(urls)
    outcomes: Counter[str] = Counter()
    byte_total = 0
    deadline = time.perf_counter() + seconds

    async def worker(session: aiohttp.ClientSession) -> None:
        nonlocal byte_total
        for url in url_iterator:
            if time.perf_counter() >= deadline:
                return
            outcome, byte_count = await fetch_image(session, url)
            if time.perf_counter() < deadline:
                outcomes[outcome] += 1
                byte_total += byte_count

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, headers={"User-Agent": USER_AGENT}
    ) as session:
        workers = [asyncio.create_task(worker(session)) for _ in range(concurrency)]
        await asyncio.wait(workers, timeout=seconds + 1)
        for task in workers:
            task.cancel()
    return {"outcomes": dict(outcomes), "bytes": byte_total}


def run_process(arguments: tuple[list[str], int, float]) -> dict[str, object]:
    urls, concurrency, seconds = arguments
    return asyncio.run(download_for(urls, concurrency, seconds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=256, help="per process")
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--url-offset", type=int, default=0)
    arguments = parser.parse_args()

    urls = load_shuffled_urls()[arguments.url_offset :]
    chunk = len(urls) // arguments.processes
    jobs = [
        (urls[index * chunk : (index + 1) * chunk], arguments.concurrency, arguments.seconds)
        for index in range(arguments.processes)
    ]
    with multiprocessing.Pool(arguments.processes) as pool:
        results = pool.map(run_process, jobs)

    outcomes: Counter[str] = Counter()
    for result in results:
        outcomes.update(result["outcomes"])
    byte_total = sum(result["bytes"] for result in results)
    attempts = sum(outcomes.values())
    success = outcomes["ok"]
    summary = {
        "processes": arguments.processes,
        "concurrency_per_process": arguments.concurrency,
        "total_concurrency": arguments.processes * arguments.concurrency,
        "seconds": arguments.seconds,
        "images_per_second": success / arguments.seconds,
        "attempts_per_second": attempts / arguments.seconds,
        "megabytes_per_second": byte_total / 1e6 / arguments.seconds,
        "alive_fraction": success / max(attempts, 1),
        "mean_image_megabytes": byte_total / 1e6 / max(success, 1),
        "outcomes": dict(outcomes.most_common()),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
