"""
Launches the FLUX.2 latent upscaler pretraining on this machine:

1. visual benchmark set (30 Axis-V1 images) in the data run, built once
2. producer on ENCODER_GPUS writing the rolling latent cache of the data run; waits until every pair type has
   WARMUP_MIN_SHARDS_PER_TYPE shards and WARMUP_MIN_SAMPLES samples were produced
3. benchmark watcher on the first encoder GPU, DDP trainer on TRAINING_GPUS (fixed TRAINING_HOURS schedule)

Crashed producer / benchmark processes are restarted; a crashed or hung trainer is restarted from its latest
checkpoint up to MAX_TRAINER_RESTARTS times. Exits 0 when training is done, 1 when the trainer keeps failing.
`touch pretrain_runs/RUN/control/stop` saves a checkpoint and stops; relaunching the same command resumes.

Usage: python scripts/run_flux_pretrain.py --run-name RUN --data-run-name DATA [--learning-rate LR]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from stdev.flux_pretrain.cache import list_shards
from stdev.flux_pretrain.config import (
    BASE_LEARNING_RATE,
    ENCODER_GPUS,
    ENV_PATH,
    PAIR_TYPES,
    REPOSITORY_ROOT,
    TRAINING_GPUS,
    WARMUP_MIN_SAMPLES,
    WARMUP_MIN_SHARDS_PER_TYPE,
    run_paths,
)
from stdev.pretrain.config import PretrainPaths
from stdev.video_downloader.stream_videos import load_env_file

POLL_INTERVAL_SECONDS = 10.0
MAX_SERVICE_RESTARTS = 50
MAX_TRAINER_RESTARTS = 5
TRAINER_RESTART_DELAY_SECONDS = 60.0
TRAINER_SILENCE_LIMIT_SECONDS = 20 * 60.0
SHUTDOWN_TIMEOUT_SECONDS = 900.0


def start_process(arguments: list[str], log_path: Path, visible_gpus: tuple[int, ...] | None = None) -> subprocess.Popen:
    environment = dict(os.environ)
    if visible_gpus is not None:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in visible_gpus)
    log_file = log_path.open("a")
    print(f"launcher: {' '.join(arguments)} > {log_path}", flush=True)
    return subprocess.Popen(
        arguments, cwd=REPOSITORY_ROOT, env=environment, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
    )


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def kill_hung_trainer(trainer: subprocess.Popen) -> None:
    """
    Dumps the Python stacks of all trainer ranks (faulthandler on SIGUSR1) into the log, then kills the whole
    torchrun process group.
    """
    for child in subprocess.run(["pgrep", "-P", str(trainer.pid)], capture_output=True, text=True).stdout.split():
        try:
            os.kill(int(child), signal.SIGUSR1)
        except ProcessLookupError:
            continue
    time.sleep(15.0)
    os.killpg(trainer.pid, signal.SIGKILL)
    trainer.wait()


def cache_is_warm(data_paths: PretrainPaths) -> bool:
    if not all(len(list_shards(data_paths.cache / pair_type.name)) >= WARMUP_MIN_SHARDS_PER_TYPE for pair_type in PAIR_TYPES):
        return False
    try:
        produced = json.loads(data_paths.producer_statistics.read_text())["produced_samples_total"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return False
    return produced >= WARMUP_MIN_SAMPLES


def main() -> None:
    load_env_file(ENV_PATH)
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-run-name", required=True)
    parser.add_argument("--learning-rate", type=float, default=BASE_LEARNING_RATE)
    arguments = parser.parse_args()

    paths = run_paths(arguments.run_name)
    data_paths = run_paths(arguments.data_run_name)
    paths.create()
    data_paths.create()
    python = sys.executable
    if not (data_paths.benchmark_set / "manifest.json").exists():
        build = start_process(
            [python, "-m", "stdev.flux_pretrain.benchmark", "--run-name", arguments.run_name, "--data-run-name", arguments.data_run_name, "--build"],
            paths.logs / "benchmark_build.log",
            ENCODER_GPUS[:1],
        )
        if build.wait() != 0:
            raise SystemExit("launcher: benchmark set build failed")

    services = {
        "producer": (
            [python, "-m", "stdev.flux_pretrain.producer", "--run-name", arguments.data_run_name, "--encoder-gpus", *(str(gpu) for gpu in ENCODER_GPUS)],
            None,
        ),
        "benchmark": (
            [python, "-m", "stdev.flux_pretrain.benchmark", "--run-name", arguments.run_name, "--data-run-name", arguments.data_run_name],
            ENCODER_GPUS[:1],
        ),
    }
    trainer_arguments = [
        python,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(len(TRAINING_GPUS)),
        "--master-port",
        "29514",
        "-m",
        "stdev.flux_pretrain.trainer",
        "--run-name",
        arguments.run_name,
        "--data-run-name",
        arguments.data_run_name,
        "--learning-rate",
        str(arguments.learning_rate),
    ]
    processes = {name: start_process(command, paths.logs / f"{name}.log", gpus) for name, (command, gpus) in services.items()}
    restarts = {name: 0 for name in services}

    def check_services() -> None:
        for name, (command, gpus) in services.items():
            if processes[name].poll() is None or restarts[name] >= MAX_SERVICE_RESTARTS:
                continue
            if name == "benchmark" and paths.training_done.exists():
                continue
            restarts[name] += 1
            print(f"launcher: {name} exited with {processes[name].returncode}, restart {restarts[name]}", flush=True)
            processes[name] = start_process(command, paths.logs / f"{name}.log", gpus)

    while not cache_is_warm(data_paths):
        check_services()
        time.sleep(POLL_INTERVAL_SECONDS)
    print("launcher: cache warm, starting trainer", flush=True)

    trainer = start_process(trainer_arguments, paths.logs / "trainer.log", TRAINING_GPUS)
    stopping = False
    trainer_start_time = time.time()

    def handle_signal(signal_number: int, frame: object) -> None:
        nonlocal stopping
        stopping = True
        trainer.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    trainer_restarts = 0
    exit_code = 0
    while True:
        time.sleep(POLL_INTERVAL_SECONDS)
        check_services()
        if trainer.poll() is None:
            trainer_log = paths.logs / "trainer.log"
            silent_seconds = time.time() - trainer_log.stat().st_mtime
            if silent_seconds > TRAINER_SILENCE_LIMIT_SECONDS and time.time() - trainer_start_time > TRAINER_SILENCE_LIMIT_SECONDS:
                print(f"launcher: TRAINER HUNG (log silent for {silent_seconds:.0f}s), dumping stacks and killing", flush=True)
                kill_hung_trainer(trainer)
            else:
                continue
        if trainer.returncode == 0 or stopping or paths.training_done.exists():
            print(f"launcher: trainer exited with {trainer.returncode}", flush=True)
            break
        if trainer_restarts >= MAX_TRAINER_RESTARTS:
            print(f"launcher: TRAINER FAILED with {trainer.returncode} after {trainer_restarts} restarts", flush=True)
            exit_code = 1
            break
        trainer_restarts += 1
        print(f"launcher: TRAINER CRASHED with {trainer.returncode}, restart {trainer_restarts} from the latest checkpoint", flush=True)
        time.sleep(TRAINER_RESTART_DELAY_SECONDS)
        trainer = start_process(trainer_arguments, paths.logs / "trainer.log", TRAINING_GPUS)
        trainer_start_time = time.time()

    stop_process(processes["producer"])
    while paths.training_done.exists() and any(paths.snapshots.glob("*.safetensors")) and processes["benchmark"].poll() is None:
        time.sleep(POLL_INTERVAL_SECONDS)
    stop_process(processes["benchmark"])
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
