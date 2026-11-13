"""
Single-VAE baselines for a multi-VAE run: for every VAE trains a fresh model (same seed, hence the
same initial core and adapters as the reference run) on that VAE only, for exactly as many steps
as the reference model spent on it, then evaluates patch FID with scripts/eval_multi_vae.py and
prints the comparison with the reference run. Finished runs are resumed or skipped.

Usage: python scripts/run_single_vae_baselines.py [--reference-run test_2] [--epochs 10]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from stdev.multi_vae_upscalers.small_universal import LATENT_SPACE_BY_VAE_NAME, VAE_NAMES

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIRECTORY = ROOT / "scripts"
RUNS_DIRECTORY = ROOT / "multi_vae_runs"


def count_training_steps(run_name: str, vae_name: str) -> int:
    accumulator = EventAccumulator(
        str(RUNS_DIRECTORY / run_name / "tensorboard"), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    return len(accumulator.Scalars(f"train_by_vae/{vae_name}"))


def run_script(script_name: str, arguments: list[str]) -> None:
    subprocess.run([sys.executable, str(SCRIPTS_DIRECTORY / script_name), *arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", default="test_2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--prefix", default="single")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--vaes", nargs="*", default=None)
    arguments = parser.parse_args()

    vae_names = arguments.vaes or list(VAE_NAMES) + list(LATENT_SPACE_BY_VAE_NAME)
    result_name = f"patch_fid_ema_shard_{arguments.shard_index}.json"
    reference_results = json.loads(
        (RUNS_DIRECTORY / arguments.reference_run / result_name).read_text()
    )

    rows = []
    for vae_name in vae_names:
        run_name = f"{arguments.prefix}_{vae_name}"
        run_script(
            "run_multi_vae.py",
            [
                "--run-name",
                run_name,
                "--epochs",
                str(arguments.epochs),
                "--only-vae",
                vae_name,
            ],
        )
        if not (RUNS_DIRECTORY / run_name / result_name).exists():
            run_script(
                "eval_multi_vae.py",
                [
                    "--run-name",
                    run_name,
                    "--only-vae",
                    vae_name,
                    "--shard-index",
                    str(arguments.shard_index),
                ],
            )
        single_results = json.loads((RUNS_DIRECTORY / run_name / result_name).read_text())[vae_name]
        rows.append(
            (
                vae_name,
                count_training_steps(arguments.reference_run, vae_name),
                count_training_steps(run_name, vae_name),
                reference_results[vae_name]["patch_fid_model"],
                single_results["patch_fid_model"],
                single_results["patch_fid_bilinear"],
            )
        )

    print(
        f"{'vae':18} {'ref steps':>10} {'single steps':>13} "
        f"{'multi FID':>10} {'single FID':>11} {'bilinear':>9}"
    )
    for vae_name, reference_steps, single_steps, multi_fid, single_fid, bilinear_fid in rows:
        print(
            f"{vae_name:18} {reference_steps:10d} {single_steps:13d} "
            f"{multi_fid:10.2f} {single_fid:11.2f} {bilinear_fid:9.2f}"
        )


if __name__ == "__main__":
    main()
