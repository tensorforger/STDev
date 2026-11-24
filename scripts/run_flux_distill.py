"""
Launches the FLUX.2 single step rectified flow distillation on this machine:

1. teacher pair generators on GENERATOR_GPUS (online, restarted on crash)
2. student trainer on TRAINER_GPU once MIN_PAIRS_TO_START pairs exist (fixed training time)
3. after training: generators stop, the 1-step benchmark runs on every snapshot (EMA, cfg 1), plus raw and cfg 2
   for the final snapshot, in parallel on all GPUs; summary in distill_runs/RUN/evaluations/summary.md

Usage: python scripts/run_flux_distill.py --run-name RUN [--training-minutes M]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from stdev.flux_distill.config import GENERATOR_GPUS, MIN_PAIRS_TO_START, RUNS_DIRECTORY, TRAINER_GPU, TRAINING_MINUTES
from stdev.flux_distill.pairs import count_pairs
from stdev.flux_pretrain.config import ENV_PATH, REPOSITORY_ROOT
from stdev.video_downloader.stream_videos import load_env_file

POLL_INTERVAL_SECONDS = 10.0
MAX_GENERATOR_RESTARTS = 10


def start_process(arguments: list[str], log_path: Path, gpu: int) -> subprocess.Popen:
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    print(f"launcher: gpu {gpu}: {' '.join(arguments)} > {log_path}", flush=True)
    return subprocess.Popen(arguments, cwd=REPOSITORY_ROOT, env=environment, stdout=log_path.open("a"), stderr=subprocess.STDOUT, start_new_session=True)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def evaluation_jobs(snapshots_directory: Path) -> list[tuple[Path, str, float]]:
    snapshots = sorted(snapshots_directory.glob("snapshot_*.safetensors"))
    final = snapshots[-1]
    jobs = [(snapshots[0], "raw", 1.0)]
    jobs += [(snapshot, "ema", 1.0) for snapshot in snapshots[1:]]
    jobs += [(final, "raw", 1.0), (final, "ema", 2.0), (final, "raw", 2.0)]
    return jobs


def write_summary(run_directory: Path) -> None:
    rows = []
    for metrics_path in sorted((run_directory / "evaluations").glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        quality = metrics["quality"]
        rows.append(
            f"| {metrics_path.parent.name} | {metrics['snapshot']['training_seconds'] / 60:.1f} | {metrics['snapshot']['samples_seen']} | "
            f"{quality['patch_fid']:.3f} | {quality['fid']:.3f} | {1e4 * quality['low_res_mse']:.2f} | {quality['chroma_ciede2000']:.3f} | {quality['latent_mse_hr']:.4f} |"
        )
    header = "| evaluation | minutes | samples | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | latent_mse_hr |\n|---|---|---|---|---|---|---|---|\n"
    (run_directory / "evaluations" / "summary.md").write_text(header + "\n".join(rows) + "\n")
    print(header + "\n".join(rows), flush=True)


def main() -> None:
    load_env_file(ENV_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--training-minutes", type=float, default=TRAINING_MINUTES)
    arguments = parser.parse_args()
    run_directory = RUNS_DIRECTORY / arguments.run_name
    logs = run_directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    def generator_command(index: int) -> list[str]:
        return [python, "-m", "stdev.flux_distill.pairs", "--run-name", arguments.run_name, "--generator-index", str(index)]

    generators = {index: start_process(generator_command(index), logs / f"generator{index}.log", gpu) for index, gpu in enumerate(GENERATOR_GPUS)}
    restarts = {index: 0 for index in generators}

    def check_generators() -> None:
        for index, process in generators.items():
            if process.poll() is not None and restarts[index] < MAX_GENERATOR_RESTARTS:
                restarts[index] += 1
                print(f"launcher: generator {index} exited with {process.returncode}, restart {restarts[index]}", flush=True)
                generators[index] = start_process(generator_command(index), logs / f"generator{index}.log", GENERATOR_GPUS[index])

    while count_pairs(arguments.run_name) < MIN_PAIRS_TO_START:
        check_generators()
        time.sleep(POLL_INTERVAL_SECONDS)
    print(f"launcher: {count_pairs(arguments.run_name)} pairs, starting trainer", flush=True)
    trainer = start_process(
        [python, "-m", "stdev.flux_distill.trainer", "--run-name", arguments.run_name, "--training-minutes", str(arguments.training_minutes)],
        logs / "trainer.log",
        TRAINER_GPU,
    )
    while trainer.poll() is None:
        check_generators()
        time.sleep(POLL_INTERVAL_SECONDS)
    for process in generators.values():
        stop_process(process)
    print(f"launcher: trainer exited with {trainer.returncode}, {count_pairs(arguments.run_name)} pairs produced", flush=True)
    if trainer.returncode != 0:
        raise SystemExit("launcher: TRAINER FAILED")

    pending = evaluation_jobs(run_directory / "snapshots")
    free_gpus = list(GENERATOR_GPUS) + [TRAINER_GPU]
    running: dict[int, subprocess.Popen] = {}
    failures = 0
    while pending or running:
        for gpu, process in list(running.items()):
            if process.poll() is not None:
                failures += process.returncode != 0
                del running[gpu]
                free_gpus.append(gpu)
        while pending and free_gpus:
            snapshot, weights, guidance = pending.pop(0)
            gpu = free_gpus.pop(0)
            running[gpu] = start_process(
                [python, "-m", "stdev.flux_distill.evaluation", "--snapshot", str(snapshot), "--weights", weights, "--guidance", str(guidance)],
                logs / f"evaluation_{snapshot.stem}_{weights}_cfg{guidance:g}.log",
                gpu,
            )
        time.sleep(POLL_INTERVAL_SECONDS)
    write_summary(run_directory)
    if failures:
        raise SystemExit(f"launcher: {failures} EVALUATIONS FAILED")


if __name__ == "__main__":
    main()
