"""
Runs the reflow distillation of the pretrained upscaler on this machine as a loop of rounds:

1. generate: every GPU generates teacher pairs on its own 4KLSDB files for GENERATION_SECONDS
2. train: DDP over all GPUs, each pair used TRAINING_REPEATS times on average
3. samples: student images (EMA and raw) at 1, 2, 4, 8 steps and the teacher schedule

Setup once: the sample set and the teacher's samples (`samples/teacher`). The next round's 4KLSDB
files are downloaded while the student trains; consumed files are deleted. The run state is saved
after every phase, so rerunning the same command resumes at the interrupted phase.

Control files: `touch distill_runs/NAME/control/decay` starts the final LR decay (optional
content: number of decay steps; default 15% of the steps done), the run ends after it.
`touch distill_runs/NAME/control/stop` stops after the current round.

Usage: python scripts/run_distill.py --run-name NAME [--generation-seconds S] [--max-rounds N]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from stdev.distill.config import (
    GENERATION_SECONDS,
    GPUS,
    TEACHER_QUANTILES_PATH,
    TEACHER_WEIGHTS_PATH,
    DistillPaths,
    run_paths,
)
from stdev.distill.data import (
    SourcePosition,
    ensure_downloaded,
    files_to_prefetch,
    list_train_files,
    local_path,
    rank_files,
)
from stdev.pretrain.config import REPOSITORY_ROOT

PHASES = ("generate", "train", "samples")


def log(message: str) -> None:
    print(f"distill: {time.strftime('%H:%M:%S')} {message}", flush=True)


def start_process(arguments: list[str], log_path: Path, gpus: tuple[int, ...] | None) -> subprocess.Popen:
    environment = dict(os.environ)
    if gpus is not None:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
    return subprocess.Popen(
        arguments, cwd=REPOSITORY_ROOT, env=environment, stdout=log_path.open("a"), stderr=subprocess.STDOUT
    )


def run_processes(commands: list[tuple[list[str], Path, tuple[int, ...] | None]]) -> None:
    processes = [start_process(*command) for command in commands]
    return_codes = [process.wait() for process in processes]
    failed = [str(command[1]) for command, code in zip(commands, return_codes) if code != 0]
    if failed:
        raise RuntimeError(f"failed: {', '.join(failed)}")


def load_state(paths: DistillPaths) -> dict:
    if paths.state.exists():
        return json.loads(paths.state.read_text())
    return {"round": 0, "phase": "generate", "positions": [[0, 0] for _ in GPUS], "setup_done": False}


def save_state(paths: DistillPaths, state: dict) -> None:
    temporary_path = paths.state.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(state, indent=1))
    temporary_path.replace(paths.state)


def needed_files(state: dict, files_per_rank: list[list[str]]) -> list[str]:
    return [
        file_name
        for rank, files in enumerate(files_per_rank)
        for file_name in files_to_prefetch(files, SourcePosition(*state["positions"][rank]))
    ]


def download_files(file_names: list[str]) -> None:
    threads = [threading.Thread(target=ensure_downloaded, args=(name,)) for name in file_names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def delete_consumed_files(state: dict, files_per_rank: list[list[str]]) -> None:
    for rank, files in enumerate(files_per_rank):
        for file_name in files[: state["positions"][rank][0]]:
            local_path(file_name).unlink(missing_ok=True)


def samples_command(run_name: str, weights: Path, output_name: str) -> list[str]:
    return [
        sys.executable, "-m", "stdev.distill.samples", "--run-name", run_name,
        "--weights", str(weights), "--output-name", output_name,
    ]


def setup(paths: DistillPaths, run_name: str) -> None:
    shutil.copy(TEACHER_QUANTILES_PATH, paths.root / "high_frequency_quantiles.json")
    run_processes([(
        [sys.executable, "-m", "stdev.distill.samples", "--run-name", run_name, "--build-set"],
        paths.logs / "samples.log",
        (GPUS[0],),
    )])
    run_processes([(samples_command(run_name, TEACHER_WEIGHTS_PATH, "teacher"), paths.logs / "samples.log", (GPUS[0],))])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--generation-seconds", type=float, default=GENERATION_SECONDS)
    parser.add_argument("--max-rounds", type=int, default=0, help="smoke test: stop after N rounds")
    arguments = parser.parse_args()
    run_name = arguments.run_name

    paths = run_paths(run_name)
    paths.create()
    state = load_state(paths)
    if not state["setup_done"]:
        setup(paths, run_name)
        state["setup_done"] = True
        save_state(paths, state)
        log("setup done: sample set and teacher samples")

    files_per_rank = [rank_files(list_train_files(), rank, len(GPUS)) for rank in range(len(GPUS))]
    prefetch_thread: threading.Thread | None = None
    rounds_this_session = 0
    while not paths.training_done.exists():
        round_index = state["round"]
        round_directory = paths.round_results(round_index)
        round_start = time.perf_counter()

        if state["phase"] == "generate":
            if prefetch_thread is not None:
                prefetch_thread.join()
            delete_consumed_files(state, files_per_rank)
            download_files(needed_files(state, files_per_rank))
            shutil.rmtree(round_directory, ignore_errors=True)
            paths.pairs(round_index).mkdir(parents=True)
            run_processes([
                (
                    [
                        sys.executable, "-m", "stdev.distill.generate", "--run-name", run_name,
                        "--round", str(round_index), "--rank", str(rank), "--world-size", str(len(GPUS)),
                        "--file-index", str(state["positions"][rank][0]),
                        "--row-offset", str(state["positions"][rank][1]),
                        "--seconds", str(arguments.generation_seconds),
                    ],
                    paths.logs / f"generate_{rank}.log",
                    (gpu,),
                )
                for rank, gpu in enumerate(GPUS)
            ])
            results = [json.loads((round_directory / f"generate_{rank}.json").read_text()) for rank in range(len(GPUS))]
            state["positions"] = [[result["file_index"], result["row_offset"]] for result in results]
            state["phase"] = "train"
            save_state(paths, state)
            pairs = sum(result["pairs"] for result in results)
            rate = sum(result["pairs_per_second"] for result in results)
            log(f"round {round_index} generate: {pairs} pairs, {rate:.1f} pairs/s, {time.perf_counter() - round_start:.0f} s")
            if any(result["files_exhausted"] for result in results):
                log("a rank used up its 4KLSDB files")

        if state["phase"] == "train":
            phase_start = time.perf_counter()
            prefetch_thread = threading.Thread(target=download_files, args=(needed_files(state, files_per_rank),))
            prefetch_thread.start()
            run_processes([(
                [
                    sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(len(GPUS)),
                    "-m", "stdev.distill.train", "--run-name", run_name, "--round", str(round_index),
                ],
                paths.logs / "train.log",
                GPUS,
            )])
            result = json.loads((round_directory / "train.json").read_text())
            state["phase"] = "samples"
            save_state(paths, state)
            log(
                f"round {round_index} train: {result['steps']} steps to step {result['step']}, loss "
                f"{result['loss']:.4f}, lr {result['learning_rate']:.2e}, {time.perf_counter() - phase_start:.0f} s"
            )

        if state["phase"] == "samples":
            phase_start = time.perf_counter()
            run_processes([
                (
                    samples_command(run_name, paths.export / f"student_{variant}.safetensors", f"round_{round_index:04d}/{variant}"),
                    paths.logs / "samples.log",
                    (gpu,),
                )
                for variant, gpu in (("ema", GPUS[0]), ("raw", GPUS[1]))
            ])
            shutil.rmtree(paths.pairs(round_index), ignore_errors=True)
            state["round"] = round_index + 1
            state["phase"] = "generate"
            save_state(paths, state)
            rounds_this_session += 1
            log(f"round {round_index} samples saved, {time.perf_counter() - phase_start:.0f} s")

        if paths.stop_request.exists():
            paths.stop_request.unlink()
            log("stopped on request")
            break
        if arguments.max_rounds > 0 and rounds_this_session >= arguments.max_rounds:
            log(f"stopped after {rounds_this_session} rounds (--max-rounds)")
            break
    if prefetch_thread is not None:
        prefetch_thread.join()
    if paths.training_done.exists():
        log("training done")


if __name__ == "__main__":
    main()
