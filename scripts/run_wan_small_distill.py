"""
Launches the small student size ladder (stdev.wan_distill.small_config.STUDENTS): online pair generators of the large
single step student on GENERATOR_GPUS (the first one also evicts the rolling cache) and one single GPU trainer per
student on TRAINING_GPUS. Crashed processes are restarted (trainers resume from their latest checkpoint) up to
MAX_RESTARTS times each; generators are stopped when all trainers are done.
Control files per run (distill_runs/wan_small_<student>/control): `decay` starts the final LR decay (optional
content: steps, default 30% of the steps done), `stop` checkpoints and stops that trainer.

Usage: python scripts/run_wan_small_distill.py [--students s20m s10m ...] [--no-upload]
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from stdev.wan_distill.config import DEFAULT_DATA_ROOT, RUNS_DIRECTORY
from stdev.wan_distill.small_config import GENERATOR_GPUS, ONLINE_PAIRS_DIRECTORY, RUN_PREFIX, STUDENTS, TRAINING_GPUS

POLL_INTERVAL_SECONDS = 30.0
MAX_RESTARTS = 5
SHUTDOWN_TIMEOUT_SECONDS = 600.0


def start_process(arguments: list[str], log_path: Path, gpu: int) -> subprocess.Popen:
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a")
    print(f"launcher: GPU {gpu}: {' '.join(arguments)} > {log_path}", flush=True)
    return subprocess.Popen(arguments, cwd=DEFAULT_DATA_ROOT, env=environment, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students", nargs="+", default=[student.name for student in STUDENTS])
    parser.add_argument("--no-upload", action="store_true")
    arguments = parser.parse_args()
    assert len(arguments.students) <= len(TRAINING_GPUS)
    python = sys.executable
    pairs_logs = DEFAULT_DATA_ROOT / ONLINE_PAIRS_DIRECTORY / "logs"

    commands: dict[str, tuple[list[str], Path, int]] = {}
    for index, gpu in enumerate(GENERATOR_GPUS):
        command = [python, "-m", "stdev.wan_distill.online_pairs", "--generator-index", str(index)] + (["--evict"] if index == 0 else [])
        commands[f"generator{index}"] = (command, pairs_logs / f"generator{index}.log", gpu)
    trainer_names = []
    for name, gpu in zip(arguments.students, TRAINING_GPUS):
        command = [python, "-m", "stdev.wan_distill.small_trainer", "--student", name] + (["--no-upload"] if arguments.no_upload else [])
        run_directory = DEFAULT_DATA_ROOT / RUNS_DIRECTORY / f"{RUN_PREFIX}{name}"
        commands[name] = (command, run_directory / "logs" / "trainer.log", gpu)
        trainer_names.append(name)

    processes = {name: start_process(*specification) for name, specification in commands.items()}
    restarts = {name: 0 for name in commands}
    finished: set[str] = set()
    given_up: set[str] = set()

    def shutdown(signal_number: int, frame: object) -> None:
        for process in processes.values():
            stop_process(process)
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while len(finished) < len(trainer_names):
        time.sleep(POLL_INTERVAL_SECONDS)
        for name, process in processes.items():
            if name in finished or name in given_up or process.poll() is None:
                continue
            run_directory = DEFAULT_DATA_ROOT / RUNS_DIRECTORY / f"{RUN_PREFIX}{name}"
            if name in trainer_names and process.returncode == 0:
                finished.add(name)
                print(f"launcher: {name} exited ({'done' if (run_directory / 'training_done').exists() else 'stopped'})", flush=True)
                continue
            if restarts[name] >= MAX_RESTARTS:
                print(f"launcher: {name} failed {restarts[name] + 1} times (exit {process.returncode}), giving up", flush=True)
                given_up.add(name)
                if name in trainer_names:
                    finished.add(name)
                continue
            restarts[name] += 1
            print(f"launcher: {name} exited with {process.returncode}, restart {restarts[name]}", flush=True)
            processes[name] = start_process(*commands[name])
    for name, process in processes.items():
        stop_process(process)
    print("launcher: all trainers finished", flush=True)


if __name__ == "__main__":
    main()
