"""
Online data producer of the Wan pretraining. CPU processes download ReLAION images (random square crop, area
resize to the target side of a randomly drawn image pair type) and stream vk-50k clips from S3 (random window
and crop for a randomly drawn video pair type). Encoder processes (one per GPU) build the source from the
target pixels (2x box downscale and / or every 2nd frame), encode images with the full Wan 2.1 VAE and videos
with the tiny one, compute the target hff and write shards per pair type into the rolling cache.
The encoder takes images or videos by deficit against the production shares, so every pair type is produced
in proportion to its training consumption.

Usage: python -m stdev.wan_pretrain.producer --run-name NAME --encoder-gpus 7 [--max-seconds S]
"""

import argparse
import json
import multiprocessing
import os
import queue
import random
import signal
import tempfile
import time
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.synchronize import Event
from pathlib import Path

import boto3
import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from botocore.config import Config

from stdev.hff.immerkaer import estimate_immerkaer
from stdev.pretrain.config import PretrainPaths
from stdev.pretrain.producer import run_download_worker
from stdev.pretrain.sources import SourcePosition, iterate_jobs
from stdev.utils.image_tools import crop_maximal_rectangle
from stdev.video_downloader.stream_videos import load_env_file
from stdev.wan_pretrain.cache import EncodedShard, evict_oldest_shards, list_shards, write_shard
from stdev.wan_pretrain.config import (
    BENCHMARK_SHARD_INDEX,
    CACHE_SIZE_LIMIT_BYTES,
    DOWNLOAD_PROCESSES,
    ENV_PATH,
    EVICTION_INTERVAL_SECONDS,
    FULL_VAE_DIRECTORY,
    HELD_OUT_VIDEOS,
    IMAGE_BYTES_QUEUE_SIZE,
    IMAGE_JOB_QUEUE_SIZE,
    IMAGE_PREPROCESS_PROCESSES,
    IMAGE_SAMPLE_QUEUE_SIZE,
    MIN_IMAGE_SOURCE_SIDE,
    PAIR_TYPE_BY_NAME,
    PAIR_TYPES,
    RELAION_SHARD_COUNT,
    TINY_VAE_PATH,
    VIDEO_BUCKET,
    VIDEO_KEY_PREFIX,
    VIDEO_PROCESSES,
    VIDEO_SAMPLE_QUEUE_SIZE,
    VIDEO_SEED,
    PairType,
    production_shares,
    run_paths,
)

QUEUE_TIMEOUT_SECONDS = 1.0
STATE_SAVE_INTERVAL_SECONDS = 30.0
ENCODER_SHUTDOWN_TIMEOUT_SECONDS = 120.0
PAIR_NAMES = [pair_type.name for pair_type in PAIR_TYPES]


def draw_pair_type(kind: str, generator: random.Random) -> PairType:
    shares = production_shares()
    kind_types = [pair_type for pair_type in PAIR_TYPES if pair_type.kind == kind]
    return generator.choices(kind_types, weights=[shares[pair_type.name] for pair_type in kind_types])[0]


def put_until_stopped(target_queue: multiprocessing.Queue, item: object, stop_event: Event) -> None:
    while not stop_event.is_set():
        try:
            target_queue.put(item, timeout=QUEUE_TIMEOUT_SECONDS)
            return
        except queue.Full:
            continue


def random_square_crop(image: np.ndarray, side: int, generator: random.Random) -> np.ndarray:
    """
    Random square crop with side uniform in [side, short side], area resized to side.

    args:
        image: [H, W, 3], dtype: uint8
    returns:
        image: [side, side, 3], dtype: uint8
    """
    height, width = image.shape[:2]
    crop_side = generator.randint(side, min(height, width))
    top = generator.randint(0, height - crop_side)
    left = generator.randint(0, width - crop_side)
    crop = image[top : top + crop_side, left : left + crop_side]
    if crop_side == side:
        return np.ascontiguousarray(crop)
    return cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)


def run_image_preprocess_worker(
    image_queue: multiprocessing.Queue, sample_queue: multiprocessing.Queue, stop_event: Event
) -> None:
    """
    Sample: dict with pair_name and pixels ([S, S, 3], RGB, uint8, S = target side).
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cv2.setNumThreads(1)
    sample_queue.cancel_join_thread()
    generator = random.Random(f"image-{os.getpid()}-{time.time_ns()}")
    while not stop_event.is_set():
        try:
            _, _, content = image_queue.get(timeout=QUEUE_TIMEOUT_SECONDS)
        except queue.Empty:
            continue
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < MIN_IMAGE_SOURCE_SIDE:
            continue
        pair_type = draw_pair_type("image", generator)
        pixels = cv2.cvtColor(random_square_crop(image, pair_type.target_side, generator), cv2.COLOR_BGR2RGB)
        put_until_stopped(sample_queue, {"pair_name": pair_type.name, "pixels": pixels}, stop_event)


def build_s3_client() -> "boto3.client":
    load_env_file(ENV_PATH)
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(retries={"max_attempts": 5, "mode": "standard"}, connect_timeout=10, read_timeout=60),
    )


def list_training_video_keys() -> list[str]:
    """
    returns:
        sorted video keys of vk-50k without the last HELD_OUT_VIDEOS (reserved for evaluation)
    """
    client = build_s3_client()
    keys: list[str] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=VIDEO_BUCKET, Prefix=VIDEO_KEY_PREFIX):
        keys.extend(item["Key"] for item in page.get("Contents", []) if item["Key"].endswith(".mp4"))
    keys.sort()
    return keys[:-HELD_OUT_VIDEOS]


def read_clip_window(video_path: Path, num_frames: int, generator: random.Random) -> np.ndarray | None:
    """
    returns:
        frames: [T, H, W, 3], channel_order: BGR, dtype: uint8, a random window of consecutive frames, or None
    """
    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        capture.release()
        return None
    start = generator.randint(0, total_frames - num_frames)
    frames: list[np.ndarray] = []
    frame_index = 0
    while len(frames) < num_frames:
        if frame_index < start:
            is_read = capture.grab()
        else:
            is_read, frame = capture.read()
            if is_read:
                frames.append(frame)
        if not is_read:
            capture.release()
            return None
        frame_index += 1
    capture.release()
    return np.stack(frames)


def crop_clip(frames: np.ndarray, side: int, generator: random.Random) -> np.ndarray:
    """
    Same random square crop (side uniform in [side, short side]) for all frames, area resized to side; for
    side >= short side the maximal centered square, resized.

    args:
        frames: [T, H, W, 3], dtype: uint8
    returns:
        frames: [T, side, side, 3], dtype: uint8
    """
    height, width = frames.shape[1:3]
    if min(height, width) <= side:
        return np.stack([crop_maximal_rectangle(frame, side, side) for frame in frames])
    crop_side = generator.randint(side, min(height, width))
    top = generator.randint(0, height - crop_side)
    left = generator.randint(0, width - crop_side)
    crops = frames[:, top : top + crop_side, left : left + crop_side]
    if crop_side == side:
        return np.ascontiguousarray(crops)
    return np.stack([cv2.resize(frame, (side, side), interpolation=cv2.INTER_AREA) for frame in crops])


def run_video_worker(
    keys: list[str], worker_index: int, sample_queue: multiprocessing.Queue, stop_event: Event
) -> None:
    """
    Sample: dict with pair_name and pixels ([T, S, S, 3], RGB, uint8, target frames and side).
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cv2.setNumThreads(1)
    sample_queue.cancel_join_thread()
    generator = random.Random(f"{VIDEO_SEED}-{worker_index}-{time.time_ns()}")
    client = build_s3_client()
    order = list(keys)
    position = len(order)
    while not stop_event.is_set():
        if position >= len(order):
            generator.shuffle(order)
            position = 0
        key = order[position]
        position += 1
        pair_type = draw_pair_type("video", generator)
        try:
            body = client.get_object(Bucket=VIDEO_BUCKET, Key=key)["Body"].read()
        except Exception as error:
            print(f"producer: video {key} download failed ({error})", flush=True)
            time.sleep(1.0)
            continue
        with tempfile.NamedTemporaryFile(suffix=".mp4", dir="/dev/shm") as video_file:
            video_file.write(body)
            video_file.flush()
            frames = read_clip_window(Path(video_file.name), pair_type.target_frames, generator)
        if frames is None:
            continue
        frames = crop_clip(frames, pair_type.target_side, generator)
        pixels = np.ascontiguousarray(frames[..., ::-1])
        put_until_stopped(sample_queue, {"pair_name": pair_type.name, "pixels": pixels}, stop_event)


def to_pixels(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    args:
        frames: [H, W, 3] or [T, H, W, 3], channel_order: RGB, dtype: uint8
    returns:
        pixels: [1, 3, T, H, W], pixel_values: [0...1], dtype: torch.float32
    """
    tensor = torch.from_numpy(frames).to(device, non_blocking=True)
    if tensor.ndim == 3:
        tensor = tensor[None]
    return tensor.permute(3, 0, 1, 2)[None].float().div(255.0)


def make_source_pixels(pixels: torch.Tensor, pair_type: PairType) -> torch.Tensor:
    """
    args:
        pixels: [1, 3, T_hr, S_hr, S_hr], pixel_values: [0...1]
    returns:
        pixels: [1, 3, T_lr, S_lr, S_lr], every 2nd frame and / or 2x box downscale
    """
    if pair_type.is_temporal:
        pixels = pixels[:, :, ::2]
    if pair_type.is_spatial:
        pixels = functional.avg_pool3d(pixels, kernel_size=(1, 2, 2))
    return pixels


def load_produced_counts(paths: PretrainPaths) -> dict[str, int]:
    if not paths.producer_statistics.exists():
        return {name: 0 for name in PAIR_NAMES}
    produced = json.loads(paths.producer_statistics.read_text()).get("produced", {})
    return {name: int(produced.get(name, 0)) for name in PAIR_NAMES}


def run_encoder_worker(
    gpu_index: int,
    image_sample_queue: multiprocessing.Queue,
    video_sample_queue: multiprocessing.Queue,
    run_root: str,
    stop_event: Event,
    produced_counts: SynchronizedArray,
) -> None:
    """
    Encodes one sample at a time (no throughput gain from batching the full VAE) and writes a shard of a pair
    type when it holds shard_samples samples. Waits while the encoder pause file exists. Partial shards are
    written on stop.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from stdev.vaes.tiny_wan_2_1_vae import TinyWan21VAE
    from stdev.vaes.wan_2_1_vae import Wan21VAE

    paths = PretrainPaths(Path(run_root))
    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda", gpu_index)
    torch.backends.cudnn.benchmark = True
    full_vae = Wan21VAE(str(FULL_VAE_DIRECTORY), device=device, dtype=torch.bfloat16, as_image=True)
    tiny_vae = TinyWan21VAE(str(TINY_VAE_PATH), device=device, dtype=torch.bfloat16)
    shares = production_shares()
    image_share = sum(shares[pair_type.name] for pair_type in PAIR_TYPES if pair_type.kind == "image")
    pending: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {name: [] for name in PAIR_NAMES}

    def flush(name: str) -> None:
        if not pending[name]:
            return
        sources, targets, hffs = zip(*pending[name])
        write_shard(paths.cache / name, f"encoder{gpu_index}", EncodedShard(torch.cat(sources), torch.cat(targets), torch.cat(hffs)))
        with produced_counts.get_lock():
            produced_counts[PAIR_NAMES.index(name)] += len(pending[name])
        pending[name].clear()

    def next_sample() -> dict | None:
        with produced_counts.get_lock():
            counts = list(produced_counts)
        image_count = sum(count for name, count in zip(PAIR_NAMES, counts) if PAIR_TYPE_BY_NAME[name].kind == "image")
        image_count += sum(len(pending[name]) for name in PAIR_NAMES if PAIR_TYPE_BY_NAME[name].kind == "image")
        total = sum(counts) + sum(len(samples) for samples in pending.values())
        wants_image = image_count <= image_share * total
        preferred, other = (image_sample_queue, video_sample_queue) if wants_image else (video_sample_queue, image_sample_queue)
        for sample_queue, timeout in ((preferred, 0.5), (other, 0.1)):
            try:
                return sample_queue.get(timeout=timeout)
            except queue.Empty:
                continue
        return None

    with torch.no_grad():
        while not stop_event.is_set():
            if paths.encoder_pause.exists():
                time.sleep(1.0)
                continue
            sample = next_sample()
            if sample is None:
                continue
            pair_type = PAIR_TYPE_BY_NAME[sample["pair_name"]]
            target_pixels = to_pixels(sample["pixels"], device)  # [1, 3, T, S, S]
            source_pixels = make_source_pixels(target_pixels, pair_type)
            if pair_type.kind == "image":
                latents_target = full_vae.encode(target_pixels[:, :, 0])
                latents_source = full_vae.encode(source_pixels[:, :, 0])
            else:
                latents_target = tiny_vae.encode(target_pixels)
                latents_source = tiny_vae.encode(source_pixels)
            hff = estimate_immerkaer(target_pixels if pair_type.kind == "video" else target_pixels[:, :, 0])
            pending[pair_type.name].append((latents_source.cpu(), latents_target.cpu(), hff.float().cpu()))
            if len(pending[pair_type.name]) >= pair_type.shard_samples:
                flush(pair_type.name)
        for name in PAIR_NAMES:
            flush(name)


def main() -> None:
    load_env_file(ENV_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--encoder-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    arguments = parser.parse_args()

    paths = run_paths(arguments.run_name)
    paths.create()
    shard_indices = [index for index in range(RELAION_SHARD_COUNT) if index != BENCHMARK_SHARD_INDEX]
    position = SourcePosition.load(paths.producer_state)
    video_keys = list_training_video_keys()
    print(f"producer: {len(video_keys)} training videos", flush=True)

    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    job_queue = context.Queue(maxsize=IMAGE_JOB_QUEUE_SIZE)
    image_bytes_queue = context.Queue(maxsize=IMAGE_BYTES_QUEUE_SIZE)
    image_sample_queue = context.Queue(maxsize=IMAGE_SAMPLE_QUEUE_SIZE)
    video_sample_queue = context.Queue(maxsize=VIDEO_SAMPLE_QUEUE_SIZE)
    attempt_counter = context.Value("q", 0)
    success_counter = context.Value("q", 0)
    base_counts = load_produced_counts(paths)
    produced_counts = context.Array("q", [base_counts[name] for name in PAIR_NAMES])

    worker_processes = [
        context.Process(
            target=run_download_worker,
            args=(job_queue, image_bytes_queue, stop_event, attempt_counter, success_counter),
            daemon=True,
        )
        for _ in range(DOWNLOAD_PROCESSES)
    ]
    worker_processes += [
        context.Process(target=run_image_preprocess_worker, args=(image_bytes_queue, image_sample_queue, stop_event), daemon=True)
        for _ in range(IMAGE_PREPROCESS_PROCESSES)
    ]
    worker_processes += [
        context.Process(target=run_video_worker, args=(video_keys, index, video_sample_queue, stop_event), daemon=True)
        for index in range(VIDEO_PROCESSES)
    ]
    encoder_processes = [
        context.Process(
            target=run_encoder_worker,
            args=(gpu_index, image_sample_queue, video_sample_queue, str(paths.root), stop_event, produced_counts),
        )
        for gpu_index in arguments.encoder_gpus
    ]
    for process in worker_processes + encoder_processes:
        process.start()

    def request_stop(signal_number: int, frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    start_time = time.monotonic()
    last_state_time = start_time
    last_eviction_time = start_time
    current_position = position
    session_start_counts = list(produced_counts)

    def save_state() -> None:
        current_position.save(paths.producer_state)
        elapsed = time.monotonic() - start_time
        counts = dict(zip(PAIR_NAMES, list(produced_counts)))
        session_samples = sum(counts.values()) - sum(session_start_counts)
        statistics = {
            "produced": counts,
            "produced_samples_total": sum(counts.values()),
            "produced_samples_this_session": session_samples,
            "download_attempts": attempt_counter.value,
            "download_successes": success_counter.value,
            "session_seconds": elapsed,
            "samples_per_second": session_samples / max(elapsed, 1e-6),
            "image_sample_queue": image_sample_queue.qsize(),
            "video_sample_queue": video_sample_queue.qsize(),
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
        return not all(process.is_alive() for process in encoder_processes)

    def periodic() -> None:
        nonlocal last_state_time, last_eviction_time
        now = time.monotonic()
        if now - last_state_time >= STATE_SAVE_INTERVAL_SECONDS:
            save_state()
            counts = list(produced_counts)
            print(
                f"producer: {sum(counts) - sum(session_start_counts)} samples this session "
                f"({dict(zip(PAIR_NAMES, counts))}), downloads {success_counter.value}/{attempt_counter.value}, "
                f"queues image {image_sample_queue.qsize()} video {video_sample_queue.qsize()}",
                flush=True,
            )
            last_state_time = now
        if now - last_eviction_time >= EVICTION_INTERVAL_SECONDS:
            deleted = evict_oldest_shards(paths.cache, CACHE_SIZE_LIMIT_BYTES)
            if deleted:
                print(f"producer: evicted {deleted} shards", flush=True)
            last_eviction_time = now

    for job, next_position in iterate_jobs(shard_indices, MIN_IMAGE_SOURCE_SIDE, position):
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
    for process_queue in (job_queue, image_bytes_queue, image_sample_queue, video_sample_queue):
        process_queue.cancel_join_thread()
    for process in encoder_processes:
        process.join(timeout=ENCODER_SHUTDOWN_TIMEOUT_SECONDS)
    save_state()
    for process in worker_processes + encoder_processes:
        if process.is_alive():
            process.terminate()
    failed = [process.exitcode for process in encoder_processes if process.exitcode not in (0, None)]
    print(f"producer: stopped, shards {sum(len(list_shards(paths.cache / name)) for name in PAIR_NAMES)}", flush=True)
    if failed:
        raise SystemExit(f"producer: encoder exited with {failed}")


if __name__ == "__main__":
    main()
