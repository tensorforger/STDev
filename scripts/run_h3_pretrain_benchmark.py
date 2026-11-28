"""
Full autoresearch_6 benchmark (1000 Axis-V1 1024 crops, `benchmark_datasets/axis_v1_1k_h3_flux2`) of an H3 -> FLUX.2
translator pretrain checkpoint on all GPUs: the multi step translation is sharded over the GPUs, then every setting is
decoded and scored on one GPU each with `stdev.h3_benchmark.evaluate_quality` (patch_fid, fid, psnr, low_res_mse,
chroma_ciede2000) for the FLUX.2 small decoder (`quality`) and TAEF2 (`quality_taef2`), plus latent_mse. The H3 and
FLUX.2 small decoder oracle reference rows run alongside (`scripts/run_h3_benchmark.py --reference`).

Settings: `<weights>_<hff>[_s<steps>]`, weights raw | ema, hff null (null embedding, the practical case: the true hff
of a generated latent is unknown) | truth (Immerkaer hff of the ground truth). Results:
`raw_results/h3/<run>_<step>_<setting>/metrics.json`, summary `raw_results/h3/<run>_<step>_summary.md`.

Usage:
    python scripts/run_h3_pretrain_benchmark.py --checkpoint pretrain_runs/h3_translator_v1/checkpoints/checkpoint_000050590.pt \
        --settings raw_null raw_truth ema_null [--gpus 0 1 2 3 4 5 6 7] [--references h3 flux2_small]
Internal: --translate-shard SETTING INDEX COUNT, --score SETTING (one GPU, CUDA_VISIBLE_DEVICES set by the scheduler).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from run_h3_benchmark import (  # noqa: E402
    DATASET_DIRECTORY,
    NUM_SAVED_IMAGES,
    RESULTS_DIRECTORY,
    STATISTICS_DIRECTORY,
    TRANSLATOR_DECODERS,
    load_decoder,
    make_latent_predictor,
)

from stdev.flux_pretrain.model import SamplerConfig  # noqa: E402
from stdev.h3_benchmark.benchmark import H3BenchmarkDataset, build_fid_evaluator, evaluate_quality  # noqa: E402
from stdev.h3_pretrain.benchmark import build_model  # noqa: E402
from stdev.h3_pretrain.config import BENCHMARK_SAMPLER  # noqa: E402
from stdev.h3_pretrain.model import translate  # noqa: E402

TRANSLATE_BATCH_SIZE = 8
POLL_SECONDS = 5.0
METRIC_COLUMNS = ("patch_fid", "fid", "psnr", "low_res_mse", "chroma_ciede2000")


def parse_setting(setting: str) -> tuple[str, str, SamplerConfig]:
    parts = setting.split("_")
    weights, hff_mode = parts[0], parts[1]
    if weights not in ("raw", "ema") or hff_mode not in ("null", "truth"):
        raise ValueError(f"unknown setting {setting}")
    sampler = BENCHMARK_SAMPLER
    if len(parts) > 2:
        sampler = SamplerConfig(steps=int(parts[2].removeprefix("s")), shift=BENCHMARK_SAMPLER.shift, noise_scale=BENCHMARK_SAMPLER.noise_scale)
    return weights, hff_mode, sampler


def result_name(checkpoint_path: Path, setting: str) -> str:
    run_name = checkpoint_path.parent.parent.name
    step = int(checkpoint_path.stem.split("_")[-1])
    return f"{run_name}_{step}_{setting}"


def shard_path(checkpoint_path: Path, setting: str, index: int) -> Path:
    return RESULTS_DIRECTORY / result_name(checkpoint_path, setting) / "shards" / f"latents_{index:02d}.pt"


@torch.no_grad()
def translate_shard(checkpoint_path: Path, setting: str, index: int, count: int) -> None:
    """
    Translates the index-th of count contiguous slices of the benchmark set and saves the FLUX.2 latents.
    """
    weights, hff_mode, sampler = parse_setting(setting)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(checkpoint["model"])
    if weights == "ema":
        for parameter, ema_parameter in zip(model.parameters(), checkpoint["ema"]):
            parameter.copy_(ema_parameter.to(parameter.dtype))
    del checkpoint
    dataset = H3BenchmarkDataset(DATASET_DIRECTORY)
    hff_values = torch.load(DATASET_DIRECTORY / "hff.pt").float().flatten()
    slice_size = -(-len(dataset) // count)
    start, end = index * slice_size, min(len(dataset), (index + 1) * slice_size)
    predictions = []
    for batch_start in tqdm(range(start, end, TRANSLATE_BATCH_SIZE), desc=f"{setting} shard {index}"):
        batch_end = min(end, batch_start + TRANSLATE_BATCH_SIZE)
        latents_source = dataset.latents_h3[batch_start:batch_end].cuda().float()
        hff = hff_values[batch_start:batch_end] if hff_mode == "truth" else torch.full((batch_end - batch_start,), float("nan"))
        predictions.append(translate(model, latents_source, hff.cuda(), sampler).cpu())
    output_path = shard_path(checkpoint_path, setting, index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"start": start, "end": end, "latents": torch.cat(predictions)}, output_path)


@torch.no_grad()
def score(checkpoint_path: Path, setting: str, count: int) -> None:
    """
    Gathers the shards of a setting and scores them with both decoders.
    """
    dataset = H3BenchmarkDataset(DATASET_DIRECTORY)
    shards = sorted((torch.load(shard_path(checkpoint_path, setting, index)) for index in range(count)), key=lambda shard: shard["start"])
    predicted_latents = torch.cat([shard["latents"] for shard in shards])
    if predicted_latents.shape[0] != len(dataset):
        raise RuntimeError(f"{setting}: {predicted_latents.shape[0]} latents for {len(dataset)} images")
    name = result_name(checkpoint_path, setting)
    weights, hff_mode, sampler = parse_setting(setting)
    metrics: dict = {
        "settings": {"name": name, "checkpoint": str(checkpoint_path), "weights": weights, "hff": hff_mode, "sampler": sampler.__dict__, "num_samples": len(dataset)},
        "latent_mse": float((predicted_latents - dataset.latents_flux2.float()).square().mean()),
    }
    results_directory = RESULTS_DIRECTORY / name
    for key, reference in TRANSLATOR_DECODERS.items():
        vae, _ = load_decoder(reference)
        metrics[key] = evaluate_quality(
            make_latent_predictor(vae, predicted_latents),
            dataset,
            build_fid_evaluator(dataset, STATISTICS_DIRECTORY),
            results_directory / f"images_{reference}",
            NUM_SAVED_IMAGES,
        )
        del vae
        torch.cuda.empty_cache()
    (results_directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


class GpuScheduler:
    """
    Runs queued commands, one per free GPU, and fails fast on a non-zero exit.
    """

    def __init__(self, gpus: list[int], log_directory: Path) -> None:
        self.free_gpus = list(gpus)
        self.running: dict[int, tuple[subprocess.Popen, str]] = {}
        self.log_directory = log_directory
        log_directory.mkdir(parents=True, exist_ok=True)

    def run_all(self, tasks: list[tuple[str, list[str]]]) -> None:
        pending = list(tasks)
        while pending or self.running:
            while pending and self.free_gpus:
                label, command = pending.pop(0)
                gpu = self.free_gpus.pop(0)
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                log_file = (self.log_directory / f"{label}.log").open("w")
                process = subprocess.Popen(command, cwd=REPOSITORY_ROOT, env=environment, stdout=log_file, stderr=subprocess.STDOUT)
                self.running[gpu] = (process, label)
                print(f"benchmark: started {label} on GPU {gpu}", flush=True)
            time.sleep(POLL_SECONDS)
            for gpu, (process, label) in list(self.running.items()):
                if process.poll() is None:
                    continue
                del self.running[gpu]
                self.free_gpus.append(gpu)
                if process.returncode != 0:
                    for other_process, _ in self.running.values():
                        other_process.kill()
                    raise SystemExit(f"benchmark: FAILED {label} with {process.returncode}, see {self.log_directory / label}.log")
                print(f"benchmark: finished {label}", flush=True)


def write_summary(checkpoint_path: Path, settings: list[str], references: list[str]) -> None:
    rows = []
    for reference in references:
        metrics = json.loads((RESULTS_DIRECTORY / f"reference_{reference}" / "metrics.json").read_text())
        rows.append((f"reference {reference}", metrics["quality"], None, None))
    for setting in settings:
        metrics = json.loads((RESULTS_DIRECTORY / result_name(checkpoint_path, setting) / "metrics.json").read_text())
        rows.append((setting, metrics["quality"], metrics["quality_taef2"], metrics["latent_mse"]))
    lines = [
        f"# {checkpoint_path.parent.parent.name} step {checkpoint_path.stem.split('_')[-1]}: Axis-V1 1k (autoresearch_6 benchmark)",
        "",
        "| row | " + " | ".join(METRIC_COLUMNS) + " | TAEF2 patch_fid / fid | latent_mse |",
        "|---|" + "---|" * (len(METRIC_COLUMNS) + 2),
    ]
    for label, quality, quality_taef2, latent_mse in rows:
        values = " | ".join(f"{quality[column]:.4g}" for column in METRIC_COLUMNS)
        taef2 = f"{quality_taef2['patch_fid']:.3f} / {quality_taef2['fid']:.3f}" if quality_taef2 else "-"
        mse = f"{latent_mse:.4f}" if latent_mse is not None else "-"
        lines.append(f"| {label} | {values} | {taef2} | {mse} |")
    summary_path = RESULTS_DIRECTORY / f"{result_name(checkpoint_path, 'summary')}.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", default=["raw_null"])
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(torch.cuda.device_count())))
    parser.add_argument("--references", nargs="*", default=["h3", "flux2_small"])
    parser.add_argument("--translate-shard", nargs=3, metavar=("SETTING", "INDEX", "COUNT"))
    parser.add_argument("--score", nargs=2, metavar=("SETTING", "COUNT"))
    arguments = parser.parse_args()
    checkpoint_path = arguments.checkpoint.resolve()
    if arguments.translate_shard:
        setting, index, count = arguments.translate_shard
        translate_shard(checkpoint_path, setting, int(index), int(count))
        return
    if arguments.score:
        score(checkpoint_path, arguments.score[0], int(arguments.score[1]))
        return

    for setting in arguments.settings:
        parse_setting(setting)
    python = sys.executable
    script = str(Path(__file__).resolve())
    shard_count = len(arguments.gpus)
    log_directory = RESULTS_DIRECTORY / f"{result_name(checkpoint_path, 'logs')}"
    scheduler = GpuScheduler(arguments.gpus, log_directory)
    statistics_ready = (STATISTICS_DIRECTORY / "patch_fid_stats_1000.pt").exists() and (STATISTICS_DIRECTORY / "fid_stats_1000.pt").exists()
    first_tasks = [] if statistics_ready else [("fid_statistics", [python, "scripts/run_h3_benchmark.py", "--prepare-cache"])]
    translate_tasks = [
        (f"translate_{setting}_{index}", [python, script, "--checkpoint", str(checkpoint_path), "--translate-shard", setting, str(index), str(shard_count)])
        for setting in arguments.settings
        for index in range(shard_count)
        if not shard_path(checkpoint_path, setting, index).exists()
    ]
    scheduler.run_all(first_tasks + translate_tasks)
    score_tasks = [
        (f"score_{setting}", [python, script, "--checkpoint", str(checkpoint_path), "--score", setting, str(shard_count)])
        for setting in arguments.settings
    ]
    reference_tasks = [
        (f"reference_{reference}", [python, "scripts/run_h3_benchmark.py", "--reference", reference])
        for reference in arguments.references
        if not (RESULTS_DIRECTORY / f"reference_{reference}" / "metrics.json").exists()
    ]
    scheduler.run_all(score_tasks + reference_tasks)
    write_summary(checkpoint_path, arguments.settings, arguments.references)


if __name__ == "__main__":
    main()
