"""
Runs dev/evaluate_quality.py jobs on all free GPUs, one job per GPU at a time.

Usage: python dev/sweep.py JOB [JOB ...] [--gpus 0 1 2 ...] [--samples 1000]
JOB is "RUN:TAG:NAME=EXPR,NAME=EXPR" (overrides optional), e.g. "0009:s20h3:NUM_SAMPLING_STEPS=20".
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(REPOSITORY_ROOT / ".venv" / "bin" / "python")


def build_command(job: str, samples: int) -> tuple[list[str], str]:
    run, tag, *rest = job.split(":")
    command = [
        PYTHON,
        str(REPOSITORY_ROOT / "dev" / "evaluate_quality.py"),
        run,
        "--tag",
        tag,
        "--samples",
        str(samples),
    ]
    overrides = rest[0].split(",") if rest and rest[0] else []
    for override in overrides:
        command += ["--override", override]
    return command, f"{run}_{tag}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", nargs="+")
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--samples", type=int, default=1000)
    arguments = parser.parse_args()

    pending = list(arguments.jobs)
    running: dict[int, tuple[subprocess.Popen, str]] = {}
    (REPOSITORY_ROOT / "dev" / "logs").mkdir(parents=True, exist_ok=True)
    while pending or running:
        for gpu, (process, name) in list(running.items()):
            if process.poll() is not None:
                print(f"done {name} on GPU {gpu} exit {process.returncode}", flush=True)
                del running[gpu]
        for gpu in arguments.gpus:
            if gpu not in running and pending:
                command, name = build_command(pending.pop(0), arguments.samples)
                log = open(REPOSITORY_ROOT / "dev" / "logs" / f"eval_{name}.log", "w")
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                )
                running[gpu] = (process, name)
        time.sleep(5)
    sys.exit(0)


if __name__ == "__main__":
    main()
