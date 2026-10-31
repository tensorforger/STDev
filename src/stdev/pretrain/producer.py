"""
Online data producer: downloads ReLAION images, crops and resizes them to the stage resolution
pair, measures high frequency features of the high resolution image, encodes both resolutions
with the FLUX.2 VAE and the caption with the SDXL text encoder, and writes shards to the cache.

Modes:
    train      endless production into the rolling cache (training shards), resumable position
    benchmark  fixed held-out set from the benchmark shard, also saves the images as PNG

Usage: python -m stdev.pretrain.producer --run-name NAME --stage stage_1 --mode train
           --encoder-gpus 7 [--sample-limit N]
"""

import argparse
import asyncio
import io
import json
import multiprocessing
import queue
import random
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
import cv2
import numpy as np
import torch
from PIL import Image

from stdev.pretrain.cache import EncodedShard, evict_oldest_shards, list_shards, write_shard
from stdev.pretrain.config import (
    BENCHMARK_SHARD_INDEX,
    CACHE_SIZE_LIMIT_BYTES,
    CONNECT_TIMEOUT_SECONDS,
    DOWNLOAD_CONCURRENCY_PER_PROCESS,
    DOWNLOAD_PROCESSES,
    ENCODE_BATCH_SIZE,
    ENCODE_CHUNK_PIXELS,
    EVICTION_INTERVAL_SECONDS,
    IMAGE_QUEUE_SIZE,
    MAX_CONNECTIONS_PER_HOST,
    MAX_IMAGE_BYTES,
    PREPROCESS_PROCESSES,
    RELAION_SHARD_COUNT,
    SAMPLE_QUEUE_SIZE,
    SDXL_DIRECTORY,
    SHARD_SAMPLE_COUNT,
    STAGES,
    TOTAL_TIMEOUT_SECONDS,
    URL_QUEUE_SIZE,
    USER_AGENT,
    VAE_DIRECTORY,
    PretrainPaths,
    StageConfig,
    run_paths,
)
from stdev.pretrain.high_frequency import compute_feature_vector
from stdev.pretrain.sources import SourcePosition, iterate_jobs

if TYPE_CHECKING:
    from stdev.vaes.flux_2_vae import Flux2VAE

QUEUE_TIMEOUT_SECONDS = 1.0
STATE_SAVE_INTERVAL_SECONDS = 30.0
ENCODER_SHUTDOWN_TIMEOUT_SECONDS = 300.0


async def fetch_image(session: aiohttp.ClientSession, url: str) -> bytes | None:
    """
    returns:
        raw bytes of a valid image, None for any failure
    """
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return None
            buffer = bytearray()
            async for chunk in response.content.iter_chunked(64 * 1024):
                buffer.extend(chunk)
                if len(buffer) > MAX_IMAGE_BYTES:
                    return None
    except Exception:
        return None
    try:
        with Image.open(io.BytesIO(buffer)) as image:
            image.verify()
    except Exception:
        return None
    return bytes(buffer)


async def download_loop(
    job_queue: multiprocessing.Queue,
    image_queue: multiprocessing.Queue,
    stop_event: Event,
    attempt_counter: Synchronized,
    success_counter: Synchronized,
) -> None:
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=4)
    pending_jobs: asyncio.Queue = asyncio.Queue(maxsize=2 * DOWNLOAD_CONCURRENCY_PER_PROCESS)

    def get_job() -> tuple[str, str] | None:
        try:
            return job_queue.get(timeout=QUEUE_TIMEOUT_SECONDS)
        except queue.Empty:
            return None

    def put_image(item: tuple[str, str, bytes]) -> None:
        while not stop_event.is_set():
            try:
                image_queue.put(item, timeout=QUEUE_TIMEOUT_SECONDS)
                return
            except queue.Full:
                continue

    async def feeder() -> None:
        while not stop_event.is_set():
            job = await loop.run_in_executor(executor, get_job)
            if job is not None:
                await pending_jobs.put(job)

    async def worker(session: aiohttp.ClientSession) -> None:
        while not stop_event.is_set():
            try:
                url, caption = await asyncio.wait_for(pending_jobs.get(), QUEUE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                continue
            content = await fetch_image(session, url)
            with attempt_counter.get_lock():
                attempt_counter.value += 1
            if content is None:
                continue
            with success_counter.get_lock():
                success_counter.value += 1
            await loop.run_in_executor(executor, put_image, (url, caption, content))

    connector = aiohttp.TCPConnector(
        limit=DOWNLOAD_CONCURRENCY_PER_PROCESS,
        limit_per_host=MAX_CONNECTIONS_PER_HOST,
        ttl_dns_cache=600,
        ssl=False,
        enable_cleanup_closed=True,
    )
    timeout = aiohttp.ClientTimeout(total=TOTAL_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, headers={"User-Agent": USER_AGENT}
    ) as session:
        tasks = [asyncio.create_task(feeder())] + [
            asyncio.create_task(worker(session)) for _ in range(DOWNLOAD_CONCURRENCY_PER_PROCESS)
        ]
        await asyncio.gather(*tasks)
    executor.shutdown(wait=False, cancel_futures=True)


def run_download_worker(
    job_queue: multiprocessing.Queue,
    image_queue: multiprocessing.Queue,
    stop_event: Event,
    attempt_counter: Synchronized,
    success_counter: Synchronized,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    image_queue.cancel_join_thread()
    asyncio.run(download_loop(job_queue, image_queue, stop_event, attempt_counter, success_counter))


def crop_and_resize(
    image: np.ndarray, stage: StageConfig, generator: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random square crop with side uniform in [high_resolution, short side], area resized to the
    high resolution; the low resolution image is the area 2x downscale of the high resolution one.

    args:
        image: [H, W, 3], channel_order: BGR, dtype: uint8
    returns:
        image_high: [high_resolution, high_resolution, 3], channel_order: BGR, dtype: uint8
        image_low: [low_resolution, low_resolution, 3], channel_order: BGR, dtype: uint8
    """
    height, width = image.shape[:2]
    side = generator.randint(stage.high_resolution, min(height, width))
    top = generator.randint(0, height - side)
    left = generator.randint(0, width - side)
    crop = image[top : top + side, left : left + side]
    size_high = (stage.high_resolution, stage.high_resolution)
    image_high = (
        np.ascontiguousarray(crop)
        if side == stage.high_resolution
        else cv2.resize(crop, size_high, interpolation=cv2.INTER_AREA)
    )
    size_low = (stage.low_resolution, stage.low_resolution)
    image_low = cv2.resize(image_high, size_low, interpolation=cv2.INTER_AREA)
    return image_high, image_low


def run_preprocess_worker(
    image_queue: multiprocessing.Queue,
    sample_queue: multiprocessing.Queue,
    stage_name: str,
    deterministic: bool,
    stop_event: Event,
) -> None:
    """
    Sample: dict with url, caption, image_high / image_low ([H, W, 3], RGB, uint8) and
    high_frequency_raw ([4], float32). Deterministic mode seeds the crop with the url.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cv2.setNumThreads(1)
    sample_queue.cancel_join_thread()
    stage = STAGES[stage_name]
    generator = random.Random()
    while not stop_event.is_set():
        try:
            url, caption, content = image_queue.get(timeout=QUEUE_TIMEOUT_SECONDS)
        except queue.Empty:
            continue
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < stage.min_source_side:
            continue
        if deterministic:
            generator.seed(url)
        image_high, image_low = crop_and_resize(image, stage, generator)
        sample = {
            "url": url,
            "caption": caption,
            "image_high": cv2.cvtColor(image_high, cv2.COLOR_BGR2RGB),
            "image_low": cv2.cvtColor(image_low, cv2.COLOR_BGR2RGB),
            "high_frequency_raw": compute_feature_vector(image_high),
        }
        while not stop_event.is_set():
            try:
                sample_queue.put(sample, timeout=QUEUE_TIMEOUT_SECONDS)
                break
            except queue.Full:
                continue


def images_to_tensor(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """
    args:
        images: B x [H, W, 3], channel_order: RGB, dtype: uint8
    returns:
        images: [B, 3, H, W], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32
    """
    stacked = torch.from_numpy(np.stack(images)).to(device, non_blocking=True)
    return stacked.permute(0, 3, 1, 2).float().div(255.0)


def encode_in_chunks(vae: "Flux2VAE", images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """
    Encodes in chunks of at most ENCODE_CHUNK_PIXELS pixels (fixed chunk size per resolution).
    returns:
        latents: [B, 32, H / 8, W / 8]
    """
    height, width = images[0].shape[:2]
    chunk_size = max(1, ENCODE_CHUNK_PIXELS // (height * width))
    return torch.cat(
        [
            vae.encode(images_to_tensor(images[start : start + chunk_size], device))
            for start in range(0, len(images), chunk_size)
        ]
    )


def run_encoder_worker(
    gpu_index: int,
    sample_queue: multiprocessing.Queue,
    run_root: str,
    output_directory: str,
    writer_name: str,
    stop_event: Event,
    produced_counter: Synchronized,
    sample_limit: int,
    save_images: bool,
) -> None:
    """
    Encodes batches of samples on one GPU and writes shards of SHARD_SAMPLE_COUNT samples.
    Pauses while the encoder pause file exists (benchmark running on the same GPU). On stop, the
    partial shard is written before exit.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from stdev.text_encoders.sdxl_text_encoder import SDXLTextEncoder
    from stdev.vaes.flux_2_vae import Flux2VAE

    paths = PretrainPaths(Path(run_root))
    output_path = Path(output_directory)
    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda", gpu_index)
    torch.backends.cudnn.benchmark = True
    vae = Flux2VAE(str(VAE_DIRECTORY), device=device, dtype=torch.bfloat16)
    vae.model.encoder = torch.compile(vae.model.encoder)
    text_encoder = SDXLTextEncoder(str(SDXL_DIRECTORY), device=device, dtype=torch.float16)

    pending: dict[str, list] = {
        "latents_low": [],
        "latents_high": [],
        "text_embedding": [],
        "high_frequency_raw": [],
        "metadata": [],
    }

    def flush() -> None:
        if not pending["metadata"]:
            return
        shard = EncodedShard(
            torch.cat(pending["latents_low"]),
            torch.cat(pending["latents_high"]),
            torch.cat(pending["text_embedding"]),
            torch.from_numpy(np.stack(pending["high_frequency_raw"])),
        )
        write_shard(output_path, writer_name, shard, pending["metadata"])
        with produced_counter.get_lock():
            produced_counter.value += len(pending["metadata"])
        for values in pending.values():
            values.clear()

    def limit_reached() -> bool:
        return sample_limit > 0 and produced_counter.value + len(pending["metadata"]) >= sample_limit

    while not stop_event.is_set() and not limit_reached():
        if paths.encoder_pause.exists():
            time.sleep(1.0)
            continue
        batch: list[dict] = []
        deadline = time.monotonic() + 5.0
        while len(batch) < ENCODE_BATCH_SIZE and time.monotonic() < deadline:
            if stop_event.is_set():
                break
            try:
                batch.append(sample_queue.get(timeout=QUEUE_TIMEOUT_SECONDS))
            except queue.Empty:
                continue
        if not batch:
            continue
        if sample_limit > 0:
            batch = batch[: sample_limit - produced_counter.value - len(pending["metadata"])]
        padding = ENCODE_BATCH_SIZE - len(batch)
        images_high = [sample["image_high"] for sample in batch]
        images_low = [sample["image_low"] for sample in batch]
        images_high += [images_high[-1]] * padding
        images_low += [images_low[-1]] * padding
        latents_high = encode_in_chunks(vae, images_high, device)[: len(batch)]
        latents_low = encode_in_chunks(vae, images_low, device)[: len(batch)]
        text_embedding = text_encoder.encode_text_batch(
            [sample["caption"] for sample in batch]
        ).float().mean(dim=1)  # [B, 768]

        if save_images:
            first_index = produced_counter.value + len(pending["metadata"])
            for offset, sample in enumerate(batch):
                for kind in ("high", "low"):
                    image_path = paths.benchmark_set / f"images_{kind}" / f"{first_index + offset:05d}.png"
                    cv2.imwrite(str(image_path), cv2.cvtColor(sample[f"image_{kind}"], cv2.COLOR_RGB2BGR))

        pending["latents_low"].append(latents_low.float().cpu())
        pending["latents_high"].append(latents_high.float().cpu())
        pending["text_embedding"].append(text_embedding.cpu())
        pending["high_frequency_raw"].extend(sample["high_frequency_raw"] for sample in batch)
        pending["metadata"].extend(
            {"url": sample["url"], "caption": sample["caption"]} for sample in batch
        )
        if len(pending["metadata"]) >= SHARD_SAMPLE_COUNT:
            flush()
    flush()


def load_produced_total(paths: PretrainPaths) -> int:
    if not paths.producer_statistics.exists():
        return 0
    return int(json.loads(paths.producer_statistics.read_text()).get("produced_samples_total", 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", default="stage_1", choices=STAGES)
    parser.add_argument("--mode", default="train", choices=("train", "benchmark"))
    parser.add_argument("--encoder-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    arguments = parser.parse_args()

    paths = run_paths(arguments.run_name)
    paths.create()
    stage = STAGES[arguments.stage]
    is_benchmark = arguments.mode == "benchmark"
    if is_benchmark:
        output_directory = paths.benchmark_set
        for kind in ("high", "low"):
            (paths.benchmark_set / f"images_{kind}").mkdir(parents=True, exist_ok=True)
        shard_indices = [BENCHMARK_SHARD_INDEX]
        position = SourcePosition()
        sample_limit = arguments.sample_limit
        if len(arguments.encoder_gpus) != 1:
            raise ValueError("benchmark mode uses exactly one encoder GPU (ordered image indices)")
    else:
        output_directory = paths.cache
        shard_indices = [index for index in range(RELAION_SHARD_COUNT) if index != BENCHMARK_SHARD_INDEX]
        position = SourcePosition.load(paths.producer_state)
        sample_limit = 0

    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    job_queue = context.Queue(maxsize=URL_QUEUE_SIZE)
    image_queue = context.Queue(maxsize=IMAGE_QUEUE_SIZE)
    sample_queue = context.Queue(maxsize=SAMPLE_QUEUE_SIZE)
    attempt_counter = context.Value("q", 0)
    success_counter = context.Value("q", 0)
    produced_counter = context.Value("q", 0)

    download_processes = [
        context.Process(
            target=run_download_worker,
            args=(job_queue, image_queue, stop_event, attempt_counter, success_counter),
            daemon=True,
        )
        for _ in range(DOWNLOAD_PROCESSES)
    ]
    preprocess_processes = [
        context.Process(
            target=run_preprocess_worker,
            args=(image_queue, sample_queue, stage.name, is_benchmark, stop_event),
            daemon=True,
        )
        for _ in range(PREPROCESS_PROCESSES)
    ]
    encoder_processes = [
        context.Process(
            target=run_encoder_worker,
            args=(
                gpu_index,
                sample_queue,
                str(paths.root),
                str(output_directory),
                f"{arguments.mode}{gpu_index}",
                stop_event,
                produced_counter,
                sample_limit,
                is_benchmark,
            ),
        )
        for gpu_index in arguments.encoder_gpus
    ]
    for process in download_processes + preprocess_processes + encoder_processes:
        process.start()

    def request_stop(signal_number: int, frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    produced_base = 0 if is_benchmark else load_produced_total(paths)
    start_time = time.monotonic()
    last_state_time = start_time
    last_eviction_time = start_time
    current_position = position

    def save_state() -> None:
        if is_benchmark:
            return
        current_position.save(paths.producer_state)
        elapsed = time.monotonic() - start_time
        statistics = {
            "produced_samples_total": produced_base + produced_counter.value,
            "produced_samples_this_session": produced_counter.value,
            "download_attempts": attempt_counter.value,
            "download_successes": success_counter.value,
            "session_seconds": elapsed,
            "samples_per_second": produced_counter.value / max(elapsed, 1e-6),
            "cache_shards": len(list_shards(paths.cache)),
            "encoder_gpus": arguments.encoder_gpus,
            "updated_at": time.time(),
        }
        temporary_path = paths.producer_statistics.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(statistics, indent=1))
        temporary_path.replace(paths.producer_statistics)

    def should_stop() -> bool:
        if stop_event.is_set() or paths.stop_request.exists():
            return True
        if arguments.max_seconds > 0 and time.monotonic() - start_time >= arguments.max_seconds:
            return True
        return not any(process.is_alive() for process in encoder_processes)

    def periodic() -> None:
        nonlocal last_state_time, last_eviction_time
        now = time.monotonic()
        if now - last_state_time >= STATE_SAVE_INTERVAL_SECONDS:
            save_state()
            print(
                f"producer: {produced_counter.value} samples, downloads "
                f"{success_counter.value}/{attempt_counter.value}, "
                f"{produced_counter.value / max(now - start_time, 1e-6):.1f} samples/s",
                flush=True,
            )
            last_state_time = now
        if not is_benchmark and now - last_eviction_time >= EVICTION_INTERVAL_SECONDS:
            deleted = evict_oldest_shards(paths.cache, CACHE_SIZE_LIMIT_BYTES)
            if deleted:
                print(f"producer: evicted {deleted} shards", flush=True)
            last_eviction_time = now

    for job, next_position in iterate_jobs(shard_indices, stage.min_source_side, position):
        while not should_stop():
            try:
                job_queue.put((job.url, job.caption), timeout=QUEUE_TIMEOUT_SECONDS)
                break
            except queue.Full:
                periodic()
        if should_stop():
            break
        current_position = next_position
        periodic()

    while not should_stop():
        periodic()
        time.sleep(QUEUE_TIMEOUT_SECONDS)

    stop_event.set()
    for process_queue in (job_queue, image_queue, sample_queue):
        process_queue.cancel_join_thread()
    for process in encoder_processes:
        process.join(timeout=ENCODER_SHUTDOWN_TIMEOUT_SECONDS)
    save_state()
    for process in download_processes + preprocess_processes + encoder_processes:
        if process.is_alive():
            process.terminate()
    print(f"producer: stopped after {produced_counter.value} samples", flush=True)


if __name__ == "__main__":
    main()
