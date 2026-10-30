"""
Launches one pretraining stage end to end on this machine:

1. benchmark set: BENCHMARK_SAMPLE_COUNT held-out images from the benchmark metadata shard
2. warmup: all GPUs encode into the cache for WARMUP_SECONDS, then high frequency feature
   quantiles are fitted on the first HIGH_FREQUENCY_STATISTICS_SAMPLES cached samples
3. training: producer on ENCODER_GPUS, benchmark watcher on the first encoder GPU, DDP trainer on
   TRAINING_GPUS; crashed producer / benchmark processes are restarted

Each step is skipped when already done, so rerunning the same command resumes the run. Control
files: `touch pretrain_runs/NAME/control/decay` starts the final LR decay (optional content:
number of decay steps), `touch pretrain_runs/NAME/control/stop` saves a checkpoint and stops.

Usage: python scripts/run_pretrain.py --run-name NAME [--stage stage_1] [--init-checkpoint PATH]
           [--warmup-seconds S] [--benchmark-samples N] [--statistics-samples N] [--max-steps N]
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from stdev.pretrain.cache import list_shards, read_high_frequency_features
from stdev.pretrain.config import (
    BENCHMARK_SAMPLE_COUNT,
    ENCODER_GPUS,
    HIGH_FREQUENCY_QUANTILE_COUNT,
    HIGH_FREQUENCY_STATISTICS_SAMPLES,
    REPOSITORY_ROOT,
    STAGES,
    TRAINING_GPUS,
    WARMUP_ENCODER_GPUS,
    WARMUP_SECONDS,
    PretrainPaths,
    run_paths,
)
from stdev.pretrain.high_frequency import QuantileNormalizer

POLL_INTERVAL_SECONDS = 10.0
MAX_SERVICE_RESTARTS = 20
SHUTDOWN_TIMEOUT_SECONDS = 600.0


def start_process(
    arguments: list[str], log_path: Path, visible_gpus: tuple[int, ...] | None = None
) -> subprocess.Popen:
    environment = dict(os.environ)
    if visible_gpus is not None:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in visible_gpus)
    log_file = log_path.open("a")
    print(f"launcher: {' '.join(arguments)} > {log_path}", flush=True)
    return subprocess.Popen(
        arguments, cwd=REPOSITORY_ROOT, env=environment, stdout=log_file, stderr=subprocess.STDOUT
    )


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_to_completion(arguments: list[str], log_path: Path) -> None:
    process = start_process(arguments, log_path)
    if process.wait() != 0:
        raise RuntimeError(f"{' '.join(arguments)} failed, see {log_path}")


def producer_command(
    run_name: str,
    stage: str,
    mode: str,
    encoder_gpus: tuple[int, ...],
    extra_arguments: list[str],
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "stdev.pretrain.producer",
        "--run-name",
        run_name,
        "--stage",
        stage,
        "--mode",
        mode,
        "--encoder-gpus",
        *(str(gpu) for gpu in encoder_gpus),
        *extra_arguments,
    ]


def build_benchmark_set(paths: PretrainPaths, run_name: str, stage: str, sample_count: int) -> None:
    done_marker = paths.benchmark_set / "done"
    if done_marker.exists():
        return
    for path in paths.benchmark_set.rglob("*"):
        if path.is_file():
            path.unlink()
    run_to_completion(
        producer_command(
            run_name, stage, "benchmark", (ENCODER_GPUS[0],), ["--sample-limit", str(sample_count)]
        ),
        paths.logs / "producer_benchmark_set.log",
    )
    built_count = len(list((paths.benchmark_set / "images_high").glob("*.png")))
    if built_count != sample_count:
        raise RuntimeError(f"benchmark set has {built_count} images, expected {sample_count}")
    done_marker.touch()


def run_warmup(
    paths: PretrainPaths,
    run_name: str,
    stage: str,
    warmup_seconds: float,
    statistics_samples: int,
) -> None:
    if paths.warmup_done.exists():
        return
    while True:
        run_to_completion(
            producer_command(
                run_name,
                stage,
                "train",
                WARMUP_ENCODER_GPUS,
                ["--max-seconds", str(warmup_seconds)],
            ),
            paths.logs / "producer_warmup.log",
        )
        raw_features = read_high_frequency_features(paths.cache, statistics_samples)
        if len(raw_features) >= statistics_samples:
            break
        print(f"launcher: only {len(raw_features)} cached samples, extending warmup", flush=True)
    normalizer = QuantileNormalizer.fit(raw_features, HIGH_FREQUENCY_QUANTILE_COUNT)
    normalizer.save(paths.high_frequency_quantiles, len(raw_features))
    paths.warmup_done.touch()
    print(
        f"launcher: warmup done, {len(list_shards(paths.cache))} shards, "
        f"quantiles from {len(raw_features)} samples",
        flush=True,
    )


def run_training(
    paths: PretrainPaths, run_name: str, stage: str, trainer_extra_arguments: list[str]
) -> None:
    producer_arguments = producer_command(run_name, stage, "train", ENCODER_GPUS, [])
    benchmark_arguments = [sys.executable, "-m", "stdev.pretrain.benchmark", "--run-name", run_name]
    trainer_arguments = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(len(TRAINING_GPUS)),
        "-m",
        "stdev.pretrain.trainer",
        "--run-name",
        run_name,
        "--stage",
        stage,
        *trainer_extra_arguments,
    ]
    services = {
        "producer": (producer_arguments, None),
        "benchmark": (benchmark_arguments, (ENCODER_GPUS[0],)),
    }
    processes = {
        name: start_process(arguments, paths.logs / f"{name}.log", gpus)
        for name, (arguments, gpus) in services.items()
    }
    restarts = {name: 0 for name in services}
    trainer = start_process(trainer_arguments, paths.logs / "trainer.log", TRAINING_GPUS)

    def handle_signal(signal_number: int, frame: object) -> None:
        trainer.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while trainer.poll() is None:
        time.sleep(POLL_INTERVAL_SECONDS)
        for name, (arguments, gpus) in services.items():
            if processes[name].poll() is None or restarts[name] >= MAX_SERVICE_RESTARTS:
                continue
            restarts[name] += 1
            print(f"launcher: {name} exited with {processes[name].returncode}, restart {restarts[name]}", flush=True)
            processes[name] = start_process(arguments, paths.logs / f"{name}.log", gpus)
    print(f"launcher: trainer exited with {trainer.returncode}", flush=True)

    stop_process(processes["producer"])
    while any(paths.snapshots.glob("*.safetensors")) and processes["benchmark"].poll() is None:
        time.sleep(POLL_INTERVAL_SECONDS)
    stop_process(processes["benchmark"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", default="stage_1", choices=STAGES)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--warmup-seconds", type=float, default=WARMUP_SECONDS)
    parser.add_argument("--benchmark-samples", type=int, default=BENCHMARK_SAMPLE_COUNT)
    parser.add_argument("--statistics-samples", type=int, default=HIGH_FREQUENCY_STATISTICS_SAMPLES)
    parser.add_argument("--max-steps", type=int, default=0)
    arguments = parser.parse_args()

    paths = run_paths(arguments.run_name)
    paths.create()
    build_benchmark_set(paths, arguments.run_name, arguments.stage, arguments.benchmark_samples)
    run_warmup(
        paths,
        arguments.run_name,
        arguments.stage,
        arguments.warmup_seconds,
        arguments.statistics_samples,
    )
    trainer_extra_arguments: list[str] = []
    if arguments.init_checkpoint is not None:
        trainer_extra_arguments += ["--init-checkpoint", str(arguments.init_checkpoint)]
    if arguments.max_steps > 0:
        trainer_extra_arguments += ["--max-steps", str(arguments.max_steps)]
    run_training(paths, arguments.run_name, arguments.stage, trainer_extra_arguments)


if __name__ == "__main__":
    main()
