"""
Launches the small FLUX.2 student distillation on this machine:

1. large student pair generators on GENERATOR_GPUS (the first one evicts the rolling pair cache), restarted on crash
2. DDP trainer on TRAINING_GPUS (fixed TRAINING_HOURS), restarted from its latest checkpoint on crash
3. every new snapshot is evaluated (1000-image 1-step benchmark, EMA, cfg 1) on the last generator GPU, next to its
   generator; after training the generators stop and the final snapshot is also evaluated with raw weights and cfg 2.
   Summary: distill_runs/RUN/evaluations/summary.md

`touch distill_runs/RUN/control/stop` checkpoints and stops the trainer; relaunching resumes.

Usage: python scripts/run_flux_small_distill.py --run-name RUN [--training-hours H]
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_flux_distill import write_summary

from stdev.flux_distill.config import RUNS_DIRECTORY
from stdev.flux_distill.small_config import GENERATOR_GPUS, TRAINING_GPUS, TRAINING_HOURS
from stdev.flux_pretrain.config import ENV_PATH, REPOSITORY_ROOT
from stdev.video_downloader.stream_videos import load_env_file

POLL_INTERVAL_SECONDS = 15.0
MAX_RESTARTS = 5
TRAINER_SILENCE_LIMIT_SECONDS = 20 * 60.0


def start_process(arguments: list[str], log_path: Path, gpus: tuple[int, ...]) -> subprocess.Popen:
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(gpu) for gpu in gpus))
    print(f"launcher: gpus {gpus}: {' '.join(arguments)} > {log_path}", flush=True)
    return subprocess.Popen(arguments, cwd=REPOSITORY_ROOT, env=environment, stdout=log_path.open("a"), stderr=subprocess.STDOUT, start_new_session=True)


def stop_process(process: subprocess.Popen, timeout: float = 120.0) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main() -> None:
    load_env_file(ENV_PATH)
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--training-hours", type=float, default=TRAINING_HOURS)
    arguments = parser.parse_args()
    run_directory = RUNS_DIRECTORY / arguments.run_name
    logs = run_directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    def generator_command(index: int) -> list[str]:
        command = [python, "-m", "stdev.flux_distill.small_pairs", "--run-name", arguments.run_name, "--generator-index", str(index)]
        return command + (["--evict"] if index == 0 else [])

    generators = {index: start_process(generator_command(index), logs / f"generator{index}.log", (gpu,)) for index, gpu in enumerate(GENERATOR_GPUS)}
    generator_restarts = {index: 0 for index in generators}
    trainer_command = [
        python, "-m", "torch.distributed.run", "--nproc-per-node", str(len(TRAINING_GPUS)), "--master-port", "29515",
        "-m", "stdev.flux_distill.small_trainer", "--run-name", arguments.run_name, "--training-hours", str(arguments.training_hours),
    ]
    trainer = start_process(trainer_command, logs / "trainer.log", TRAINING_GPUS)
    trainer_restarts = 0
    trainer_start = time.time()
    evaluated: set[str] = set()
    evaluation: subprocess.Popen | None = None
    evaluation_gpu = GENERATOR_GPUS[-1]

    def evaluate(snapshot: Path, weights: str, guidance: float, gpu: int) -> subprocess.Popen:
        return start_process(
            [python, "-m", "stdev.flux_distill.evaluation", "--snapshot", str(snapshot), "--weights", weights, "--guidance", str(guidance), "--model", "small"],
            logs / f"evaluation_{snapshot.stem}_{weights}_cfg{guidance:g}.log",
            (gpu,),
        )

    while True:
        time.sleep(POLL_INTERVAL_SECONDS)
        for index, process in generators.items():
            if process.poll() is not None and generator_restarts[index] < MAX_RESTARTS:
                generator_restarts[index] += 1
                print(f"launcher: generator {index} exited with {process.returncode}, restart {generator_restarts[index]}", flush=True)
                generators[index] = start_process(generator_command(index), logs / f"generator{index}.log", (GENERATOR_GPUS[index],))
        if evaluation is None or evaluation.poll() is not None:
            if evaluation is not None and evaluation.returncode != 0:
                print(f"launcher: EVALUATION FAILED with {evaluation.returncode}", flush=True)
            pending = [path for path in sorted((run_directory / "snapshots").glob("snapshot_*.safetensors")) if path.stem not in evaluated]
            if pending and not (run_directory / "training_done").exists():
                evaluated.add(pending[0].stem)
                evaluation = evaluate(pending[0], "ema", 1.0, evaluation_gpu)
        if trainer.poll() is None:
            silent = time.time() - (logs / "trainer.log").stat().st_mtime
            if silent > TRAINER_SILENCE_LIMIT_SECONDS and time.time() - trainer_start > TRAINER_SILENCE_LIMIT_SECONDS:
                print(f"launcher: TRAINER HUNG (log silent {silent:.0f}s), killing", flush=True)
                os.killpg(trainer.pid, signal.SIGKILL)
                trainer.wait()
            else:
                continue
        if trainer.returncode == 0 or (run_directory / "training_done").exists():
            print(f"launcher: trainer exited with {trainer.returncode}", flush=True)
            break
        if trainer_restarts >= MAX_RESTARTS:
            print(f"launcher: TRAINER FAILED with {trainer.returncode} after {trainer_restarts} restarts", flush=True)
            for process in generators.values():
                stop_process(process)
            raise SystemExit(1)
        trainer_restarts += 1
        print(f"launcher: TRAINER CRASHED with {trainer.returncode}, restart {trainer_restarts}", flush=True)
        time.sleep(60.0)
        trainer = start_process(trainer_command, logs / "trainer.log", TRAINING_GPUS)
        trainer_start = time.time()

    for process in generators.values():
        stop_process(process)
    if evaluation is not None:
        evaluation.wait()
    snapshots = sorted((run_directory / "snapshots").glob("snapshot_*.safetensors"))
    jobs = [(path, "ema", 1.0) for path in snapshots if path.stem not in evaluated]
    jobs += [(snapshots[-1], "raw", 1.0), (snapshots[-1], "ema", 2.0)]
    free_gpus = list(TRAINING_GPUS) + list(GENERATOR_GPUS)
    running: dict[int, subprocess.Popen] = {}
    while jobs or running:
        for gpu, process in list(running.items()):
            if process.poll() is not None:
                if process.returncode != 0:
                    print(f"launcher: EVALUATION FAILED with {process.returncode} on gpu {gpu}", flush=True)
                del running[gpu]
                free_gpus.append(gpu)
        while jobs and free_gpus:
            gpu = free_gpus.pop(0)
            running[gpu] = evaluate(*jobs.pop(0), gpu)
        time.sleep(POLL_INTERVAL_SECONDS)
    write_summary(run_directory)


if __name__ == "__main__":
    main()
