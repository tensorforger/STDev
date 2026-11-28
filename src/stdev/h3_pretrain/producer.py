"""
Online data producer of the H3 -> FLUX.2 translator pretraining. CPU processes download ReLAION images (online,
from urls) and stream 4KLSDB train_x4 parquet files from the Hugging Face Hub (readers of the FLUX.2 pretrain).
Preprocess workers draw a pair type, take a large random square crop and build the clean target and the (possibly
corrupted) H3 encoder input of the same side. The encoder (one process per GPU) takes samples from the two datasets by
deficit against DATASET_SHARES, encodes the target with the FLUX.2 VAE and the source with the H3-Image encoder
(released tiling, batched tiles, single frame 2D convolutions, bf16), computes the target hff and writes shards per
pair type into the rolling cache (latents_source: H3 [N, 24, s/16, s/16], latents_target: FLUX.2 [N, 32, s/8, s/8]).

Usage: python -m stdev.h3_pretrain.producer --run-name NAME --encoder-gpus 7 [--max-seconds S]
"""

import argparse
import json
import multiprocessing
import os
import queue
import random
import signal
import threading
import time
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.synchronize import Event
from pathlib import Path

import cv2
import numpy as np
import torch

from stdev.flux_pretrain.cache import EncodedShard, evict_oldest_shards, list_shards, write_shard
from stdev.flux_pretrain.producer import list_lsdb_files, put_until_stopped, run_lsdb_reader
from stdev.h3_pretrain.config import (
    CACHE_SIZE_LIMIT_BYTES,
    DATASET_SHARES,
    DATASETS,
    DOWNLOAD_PROCESSES,
    ENV_PATH,
    EVICTION_INTERVAL_SECONDS,
    EXCLUDED_RELAION_SHARD_INDEX,
    FLUX_VAE_DIRECTORY,
    H3_ENCODER_BFLOAT16,
    H3_VAE_DIRECTORY,
    IMAGE_BYTES_QUEUE_SIZE,
    IMAGE_JOB_QUEUE_SIZE,
    LSDB_READER_PROCESSES,
    MIN_IMAGE_SOURCE_SIDE,
    PAIR_TYPE_BY_NAME,
    PAIR_TYPES,
    PREPROCESS_PROCESSES_PER_DATASET,
    RELAION_SHARD_COUNT,
    SAMPLE_QUEUE_SIZE,
    SOURCE_VARIANTS,
    PairType,
    run_paths,
)
from stdev.h3_pretrain.corruption import make_pair
from stdev.hff.immerkaer import estimate_immerkaer
from stdev.pretrain.config import PretrainPaths
from stdev.pretrain.producer import run_download_worker
from stdev.pretrain.sources import SourcePosition, iterate_jobs
from stdev.video_downloader.stream_videos import load_env_file

QUEUE_TIMEOUT_SECONDS = 1.0
STATE_SAVE_INTERVAL_SECONDS = 30.0
ENCODER_SHUTDOWN_TIMEOUT_SECONDS = 120.0
ENCODER_READY_BATCHES = 8
ENCODER_PENDING_SHARDS = 4
PAIR_NAMES = [pair_type.name for pair_type in PAIR_TYPES]
COUNTER_NAMES = PAIR_NAMES + [f"dataset_{name}" for name in DATASETS] + [f"variant_{name}" for name in SOURCE_VARIANTS]


def draw_pair_type(generator: random.Random) -> PairType:
    return generator.choices(PAIR_TYPES, weights=[pair_type.count_share for pair_type in PAIR_TYPES])[0]


def run_preprocess_worker(
    dataset_name: str, bytes_queue: multiprocessing.Queue, sample_queue: multiprocessing.Queue, stop_event: Event
) -> None:
    """
    Sample: dict with pair_name, dataset_index, variant_index, target ([S, S, 3], RGB, uint8, clean) and
    source ([S, S, 3], RGB, uint8, H3 encoder input).
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cv2.setNumThreads(1)
    sample_queue.cancel_join_thread()
    generator = random.Random(f"{dataset_name}-{os.getpid()}-{time.time_ns()}")
    while not stop_event.is_set():
        try:
            item = bytes_queue.get(timeout=QUEUE_TIMEOUT_SECONDS)
        except queue.Empty:
            continue
        image = cv2.imdecode(np.frombuffer(item[-1], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < MIN_IMAGE_SOURCE_SIDE:
            continue
        pair_type = draw_pair_type(generator)
        target, source, variant = make_pair(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), pair_type.side, generator)
        sample = {
            "pair_name": pair_type.name,
            "dataset_index": DATASETS.index(dataset_name),
            "variant_index": SOURCE_VARIANTS.index(variant),
            "target": target,
            "source": source,
        }
        put_until_stopped(sample_queue, sample, stop_event)


def load_produced_counts(paths: PretrainPaths) -> dict[str, int]:
    if not paths.producer_statistics.exists():
        return {name: 0 for name in COUNTER_NAMES}
    produced = json.loads(paths.producer_statistics.read_text()).get("produced", {})
    return {name: int(produced.get(name, 0)) for name in COUNTER_NAMES}


def run_encoder_worker(
    gpu_index: int,
    sample_queues: dict[str, multiprocessing.Queue],
    run_root: str,
    stop_event: Event,
    produced_counts: SynchronizedArray,
) -> None:
    """
    A feeder thread takes samples by dataset deficit and groups them per pair type into encode batches; the
    main thread encodes them (compiled FLUX.2 and H3 encoders, fixed batch shapes); a writer thread writes a shard of a
    pair type when it holds shard_samples samples. Waits while the encoder pause file exists.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from stdev.h3_pretrain.h3_encoder import BatchedTileH3Encoder
    from stdev.vaes.flux_2_vae import Flux2VAE
    from stdev.vaes.minimax_h3_image_vae import MiniMaxH3ImageVAE

    paths = PretrainPaths(Path(run_root))
    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda", gpu_index)
    torch.backends.cudnn.benchmark = True
    vae = Flux2VAE(str(FLUX_VAE_DIRECTORY), device=device, dtype=torch.bfloat16)
    vae.model.encoder.compile(dynamic=False)
    h3_vae = MiniMaxH3ImageVAE(str(H3_VAE_DIRECTORY), device=device, dtype=torch.bfloat16, use_tiling=True)
    h3_encoder = BatchedTileH3Encoder(h3_vae, use_bfloat16=H3_ENCODER_BFLOAT16, compile_encoder=True)
    batch_queue: queue.Queue = queue.Queue(maxsize=ENCODER_READY_BATCHES)
    shard_queue: queue.Queue = queue.Queue(maxsize=ENCODER_PENDING_SHARDS)
    feeder_done = threading.Event()

    def take_sample(taken_counts: dict[str, int]) -> dict | None:
        total = sum(taken_counts.values())
        deficits = sorted(DATASETS, key=lambda name: taken_counts[name] - DATASET_SHARES[name] * total)
        for dataset_name, timeout in zip(deficits, (0.5,) + (0.05,) * (len(deficits) - 1)):
            try:
                sample = sample_queues[dataset_name].get(timeout=timeout)
            except queue.Empty:
                continue
            taken_counts[dataset_name] += 1
            return sample
        return None

    def run_feeder() -> None:
        taken_counts = {name: 0 for name in DATASETS}
        batches: dict[str, list[dict]] = {name: [] for name in PAIR_NAMES}
        while not stop_event.is_set():
            if paths.encoder_pause.exists():
                time.sleep(1.0)
                continue
            sample = take_sample(taken_counts)
            if sample is None:
                continue
            pair_type = PAIR_TYPE_BY_NAME[sample["pair_name"]]
            batch = batches[pair_type.name]
            batch.append(sample)
            if len(batch) < pair_type.encode_batch_size:
                continue
            ready = (
                pair_type,
                torch.from_numpy(np.stack([item["target"] for item in batch])).pin_memory(),
                torch.from_numpy(np.stack([item["source"] for item in batch])).pin_memory(),
                torch.tensor([item["dataset_index"] for item in batch]),
                torch.tensor([item["variant_index"] for item in batch]),
            )
            batches[pair_type.name] = []
            while not stop_event.is_set():
                try:
                    batch_queue.put(ready, timeout=QUEUE_TIMEOUT_SECONDS)
                    break
                except queue.Full:
                    continue
        feeder_done.set()

    def run_writer() -> None:
        while True:
            item = shard_queue.get()
            if item is None:
                return
            name, parts = item
            shard = EncodedShard(*(torch.cat([getattr(part, field) for part in parts]) for field in EncodedShard.__dataclass_fields__))
            write_shard(paths.cache / name, f"encoder{gpu_index}", shard)
            with produced_counts.get_lock():
                produced_counts[COUNTER_NAMES.index(name)] += shard.hff.shape[0]
                for dataset_index in shard.dataset_index.tolist():
                    produced_counts[COUNTER_NAMES.index(f"dataset_{DATASETS[dataset_index]}")] += 1
                for variant_index in shard.variant_index.tolist():
                    produced_counts[COUNTER_NAMES.index(f"variant_{SOURCE_VARIANTS[variant_index]}")] += 1

    def to_device_pixels(images: torch.Tensor) -> torch.Tensor:
        """
        args:
            images: [B, S, S, 3], channel_order: RGB, dtype: uint8, pinned
        returns:
            pixels: [B, 3, S, S], pixel_values: [0...1], dtype: torch.float32
        """
        return images.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div(255.0)

    feeder = threading.Thread(target=run_feeder, daemon=True)
    writer = threading.Thread(target=run_writer, daemon=True)
    feeder.start()
    writer.start()
    pending: dict[str, list[EncodedShard]] = {name: [] for name in PAIR_NAMES}
    with torch.no_grad():
        while not (feeder_done.is_set() and batch_queue.empty()):
            try:
                pair_type, targets, sources, dataset_index, variant_index = batch_queue.get(timeout=QUEUE_TIMEOUT_SECONDS)
            except queue.Empty:
                continue
            target_pixels = to_device_pixels(targets)
            latents_target = vae.encode(target_pixels)[:, :, 0]  # [B, 32, S / 8, S / 8]
            latents_source = h3_encoder.encode(to_device_pixels(sources))  # [B, 24, S / 16, S / 16]
            hff = estimate_immerkaer(target_pixels).float()
            pending[pair_type.name].append(
                EncodedShard(latents_source.cpu(), latents_target.cpu(), hff.cpu(), dataset_index, variant_index)
            )
            if sum(part.hff.shape[0] for part in pending[pair_type.name]) >= pair_type.shard_samples:
                shard_queue.put((pair_type.name, pending[pair_type.name]))
                pending[pair_type.name] = []
    for name in PAIR_NAMES:
        if pending[name]:
            shard_queue.put((name, pending[name]))
    shard_queue.put(None)
    writer.join()


def main() -> None:
    load_env_file(ENV_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--encoder-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    arguments = parser.parse_args()

    paths = run_paths(arguments.run_name)
    paths.create()
    shard_indices = [index for index in range(RELAION_SHARD_COUNT) if index != EXCLUDED_RELAION_SHARD_INDEX]
    position = SourcePosition.load(paths.producer_state)
    lsdb_files = list_lsdb_files()
    print(f"producer: {len(lsdb_files)} 4KLSDB files", flush=True)

    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    job_queue = context.Queue(maxsize=IMAGE_JOB_QUEUE_SIZE)
    bytes_queues = {name: context.Queue(maxsize=IMAGE_BYTES_QUEUE_SIZE) for name in DATASETS}
    sample_queues = {name: context.Queue(maxsize=SAMPLE_QUEUE_SIZE) for name in DATASETS}
    attempt_counter = context.Value("q", 0)
    success_counter = context.Value("q", 0)
    base_counts = load_produced_counts(paths)
    produced_counts = context.Array("q", [base_counts[name] for name in COUNTER_NAMES])

    worker_processes = [
        context.Process(
            target=run_download_worker,
            args=(job_queue, bytes_queues["relaion"], stop_event, attempt_counter, success_counter),
            daemon=True,
        )
        for _ in range(DOWNLOAD_PROCESSES)
    ]
    worker_processes += [
        context.Process(
            target=run_lsdb_reader,
            args=(lsdb_files[index::LSDB_READER_PROCESSES], index, str(paths.root / "lsdb_download"), bytes_queues["lsdb"], stop_event),
            daemon=True,
        )
        for index in range(LSDB_READER_PROCESSES)
    ]
    worker_processes += [
        context.Process(target=run_preprocess_worker, args=(name, bytes_queues[name], sample_queues[name], stop_event), daemon=True)
        for name in DATASETS
        for _ in range(PREPROCESS_PROCESSES_PER_DATASET)
    ]
    encoder_processes = [
        context.Process(
            target=run_encoder_worker, args=(gpu_index, sample_queues, str(paths.root), stop_event, produced_counts)
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
    session_start_total = sum(produced_counts[COUNTER_NAMES.index(name)] for name in PAIR_NAMES)

    def counts() -> dict[str, int]:
        return dict(zip(COUNTER_NAMES, list(produced_counts)))

    def save_state() -> None:
        current_position.save(paths.producer_state)
        elapsed = time.monotonic() - start_time
        current_counts = counts()
        total = sum(current_counts[name] for name in PAIR_NAMES)
        statistics = {
            "produced": current_counts,
            "produced_samples_total": total,
            "produced_samples_this_session": total - session_start_total,
            "download_attempts": attempt_counter.value,
            "download_successes": success_counter.value,
            "session_seconds": elapsed,
            "samples_per_second": (total - session_start_total) / max(elapsed, 1e-6),
            "sample_queues": {name: sample_queues[name].qsize() for name in DATASETS},
            "bytes_queues": {name: bytes_queues[name].qsize() for name in DATASETS},
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
            current_counts = counts()
            total = sum(current_counts[name] for name in PAIR_NAMES)
            print(
                f"producer: {total - session_start_total} samples this session "
                f"({(total - session_start_total) / max(now - start_time, 1e-6):.1f}/s), {current_counts}, "
                f"downloads {success_counter.value}/{attempt_counter.value}, "
                f"bytes queues {[bytes_queues[name].qsize() for name in DATASETS]}, "
                f"sample queues {[sample_queues[name].qsize() for name in DATASETS]}",
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
    for process_queue in (job_queue, *bytes_queues.values(), *sample_queues.values()):
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
