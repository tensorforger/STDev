"""
Launches the H3 -> FLUX.2 translator single step rectified flow distillation on this machine:

1. teacher pair generators on GENERATOR_GPUS (online, restarted on crash)
2. student trainer on TRAINER_GPU once MIN_PAIRS_TO_START pairs exist (fixed training time)
3. after training: generators stop, the 1-step benchmark (autoresearch_6 Axis 1k, null hff) runs on every snapshot
   (raw), plus EMA, ground truth hff, cfg 2 and TAEF2 for the final snapshot, in parallel on all GPUs; summary in
   distill_runs/RUN/evaluations/summary.md

Usage: python scripts/run_h3_distill.py --run-name RUN [--training-minutes M] [--decay-fraction F]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from stdev.flux_distill.pairs import count_pairs
from stdev.h3_distill.config import DECAY_FRACTION, GENERATOR_GPUS, MIN_PAIRS_TO_START, RUNS_DIRECTORY, TRAINER_GPU, TRAINING_MINUTES
from stdev.h3_pretrain.config import ENV_PATH, REPOSITORY_ROOT
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


def evaluation_jobs(snapshots_directory: Path) -> list[list[str]]:
    """
    Evaluation arguments (after --snapshot) per job; the final snapshot runs first variants first.
    """
    snapshots = sorted(snapshots_directory.glob("snapshot_*.safetensors"))
    final = snapshots[-1]
    jobs = [
        [str(final), "--weights", "raw", "--taef2"],
        [str(final), "--weights", "ema"],
        [str(final), "--weights", "raw", "--hff", "truth"],
        [str(final), "--weights", "raw", "--hff", "truth", "--guidance", "2"],
    ]
    jobs += [[str(snapshot), "--weights", "raw"] for snapshot in snapshots[:-1]]
    return jobs


def write_summary(run_directory: Path) -> None:
    rows = []
    for metrics_path in sorted((run_directory / "evaluations").glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        quality = metrics["quality"]
        taef2 = metrics.get("quality_taef2")
        taef2_text = f"{taef2['patch_fid']:.3f} / {taef2['fid']:.3f}" if taef2 else "-"
        rows.append(
            f"| {metrics_path.parent.name} | {metrics['snapshot']['training_seconds'] / 60:.1f} | {metrics['snapshot']['samples_seen']} | "
            f"{quality['patch_fid']:.3f} | {quality['fid']:.3f} | {quality['psnr']:.2f} | {1e4 * quality['low_res_mse']:.2f} | "
            f"{quality['chroma_ciede2000']:.3f} | {taef2_text} | {metrics['latent_mse']:.4f} |"
        )
    header = (
        "| evaluation | minutes | samples | patch_fid | fid | psnr | low_res_mse 1e-4 | chroma dE00 | TAEF2 patch_fid / fid | latent_mse |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    (run_directory / "evaluations" / "summary.md").write_text(header + "\n".join(rows) + "\n")
    print(header + "\n".join(rows), flush=True)


def main() -> None:
    load_env_file(ENV_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--training-minutes", type=float, default=TRAINING_MINUTES)
    parser.add_argument("--decay-fraction", type=float, default=DECAY_FRACTION)
    arguments = parser.parse_args()
    run_directory = RUNS_DIRECTORY / arguments.run_name
    logs = run_directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    def generator_command(index: int) -> list[str]:
        return [python, "-m", "stdev.h3_distill.pairs", "--run-name", arguments.run_name, "--generator-index", str(index)]

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
        [python, "-m", "stdev.h3_distill.trainer", "--run-name", arguments.run_name, "--training-minutes", str(arguments.training_minutes), "--decay-fraction", str(arguments.decay_fraction)],
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
            job = pending.pop(0)
            gpu = free_gpus.pop(0)
            running[gpu] = start_process(
                [python, "-m", "stdev.h3_distill.evaluation", "--snapshot", *job],
                logs / f"evaluation_{Path(job[0]).stem}_{'_'.join(argument.strip('-') for argument in job[1:])}.log",
                gpu,
            )
        time.sleep(POLL_INTERVAL_SECONDS)
    write_summary(run_directory)
    if failures:
        raise SystemExit(f"launcher: {failures} EVALUATIONS FAILED")


if __name__ == "__main__":
    main()
