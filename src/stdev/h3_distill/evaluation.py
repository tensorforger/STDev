"""
1-step benchmark of an H3 -> FLUX.2 student snapshot on the autoresearch_6 Axis 1k set (`stdev.h3_benchmark`:
patch_fid, fid, psnr, low_res_mse, chroma_ciede2000 with the FLUX.2 small decoder; optionally TAEF2), fixed noise seed
per image (its index), hff null (default, the practical case) or the ground truth. Results:
`distill_runs/RUN/evaluations/<snapshot>_<weights>_cfg<guidance>_<hff>/metrics.json`.

Usage: CUDA_VISIBLE_DEVICES=0 python -m stdev.h3_distill.evaluation --snapshot distill_runs/RUN/snapshots/snapshot_000100.safetensors
           [--weights raw|ema] [--guidance 1.0] [--hff null|truth] [--taef2]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from stdev.flux_distill.student import seeded_noise
from stdev.flux_pretrain.model import LATENT_CHANNELS
from stdev.h3_benchmark.benchmark import H3BenchmarkDataset, build_fid_evaluator, evaluate_quality
from stdev.h3_distill.config import TEACHER_CHECKPOINT, TEACHER_WEIGHTS
from stdev.h3_distill.student import build_student_from_teacher, translate_single_step
from stdev.h3_pretrain.config import REPOSITORY_ROOT

sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
from run_h3_benchmark import DATASET_DIRECTORY, NUM_SAVED_IMAGES, STATISTICS_DIRECTORY, load_decoder, make_latent_predictor  # noqa: E402

TRANSLATE_BATCH_SIZE = 16
INFERENCE_DTYPE = torch.bfloat16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--hff", choices=("null", "truth"), default="null")
    parser.add_argument("--taef2", action="store_true")
    arguments = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    student = build_student_from_teacher(TEACHER_CHECKPOINT, TEACHER_WEIGHTS)
    snapshot_tensors = load_file(str(arguments.snapshot))
    snapshot = {}
    for prefix in (f"{arguments.weights}.", "buffers."):
        snapshot |= {name.removeprefix(prefix): tensor for name, tensor in snapshot_tensors.items() if name.startswith(prefix)}
    missing, unexpected = student.load_state_dict(snapshot, strict=False)
    buffer_names = {name for name, _ in student.named_buffers()}
    assert not unexpected and set(missing) <= buffer_names, (missing, unexpected)
    student = student.to("cuda", INFERENCE_DTYPE).eval().requires_grad_(False)
    student.linear_base.float()
    student.residual_mean = student.residual_mean.float()
    student.residual_std = student.residual_std.float()

    dataset = H3BenchmarkDataset(DATASET_DIRECTORY)
    hff_values = torch.load(DATASET_DIRECTORY / "hff.pt").float().flatten()
    predictions = []
    with torch.no_grad():
        for start in tqdm(range(0, len(dataset), TRANSLATE_BATCH_SIZE), desc="Translate"):
            end = min(len(dataset), start + TRANSLATE_BATCH_SIZE)
            latents_source = dataset.latents_h3[start:end].cuda().float()
            count = end - start
            hff = hff_values[start:end].cuda() if arguments.hff == "truth" else torch.full((count,), float("nan"), device="cuda")
            noise = seeded_noise(torch.arange(start, end), (LATENT_CHANNELS, 2 * latents_source.shape[2], 2 * latents_source.shape[3]), torch.device("cuda"))
            guidance = torch.full((count,), arguments.guidance, device="cuda")
            predictions.append(translate_single_step(student, latents_source, hff, guidance, noise, INFERENCE_DTYPE).cpu())
    predicted_latents = torch.cat(predictions)
    del student
    torch.cuda.empty_cache()

    snapshot_info = json.loads(arguments.snapshot.with_suffix(".json").read_text())
    name = f"{arguments.snapshot.stem}_{arguments.weights}_cfg{arguments.guidance:g}_{arguments.hff}"
    output_directory = arguments.snapshot.parent.parent / "evaluations" / name
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics: dict = {
        "snapshot": snapshot_info,
        "settings": vars(arguments) | {"snapshot": str(arguments.snapshot)},
        "latent_mse": float((predicted_latents - dataset.latents_flux2.float()).square().mean()),
    }
    decoders = {"quality": "flux2_small"} | ({"quality_taef2": "taef2"} if arguments.taef2 else {})
    for key, reference in decoders.items():
        vae, _ = load_decoder(reference)
        metrics[key] = evaluate_quality(
            make_latent_predictor(vae, predicted_latents),
            dataset,
            build_fid_evaluator(dataset, STATISTICS_DIRECTORY),
            output_directory / f"images_{reference}",
            NUM_SAVED_IMAGES,
        )
        del vae
        torch.cuda.empty_cache()
    (output_directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
