"""
Single GPU trainer of the H3 -> FLUX.2 single step student (rectified flow distillation, FLUX.2 recipe): the student,
initialized from the teacher, maps the noise at sigma 1 directly to the teacher's sample: velocity target = noise -
residual(teacher latents) over the linear base, conditioned on the hff (NaN = null) and the guidance scale the teacher used. Pairs come from the online
generators (rolling pair directory, reused when the trainer is faster). Fixed training time: warmup, constant LR,
linear decay over the last DECAY_FRACTION. Snapshots (raw + EMA, bf16) every SNAPSHOT_INTERVAL_SECONDS and at the end
for the evaluation; a full checkpoint at the end (uploaded to R2).

Usage: CUDA_VISIBLE_DEVICES=7 python -m stdev.h3_distill.trainer --run-name RUN [--training-minutes M] [--max-steps N]
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from stdev.h3_distill.config import (
    ACCUMULATION,
    ADAM_BETAS,
    DATA_WORKERS_PER_TYPE,
    DECAY_FRACTION,
    EMA_DECAY,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    LOG_INTERVAL_STEPS,
    RUNS_DIRECTORY,
    SEED,
    SNAPSHOT_INTERVAL_SECONDS,
    TEACHER_CHECKPOINT,
    TEACHER_WEIGHTS,
    TRAINING_MINUTES,
    UPLOAD_BUCKET,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from stdev.flux_distill.pairs import PairDataset, count_pairs, pairs_directory
from stdev.flux_distill.student import seeded_noise
from stdev.flux_pretrain.model import LATENT_CHANNELS
from stdev.h3_distill.student import build_student_from_teacher
from stdev.h3_pretrain.config import ENV_PATH, PAIR_TYPES, micro_step_probabilities
from stdev.h3_pretrain.model import H3_CHANNELS

TRAINING_DTYPE = torch.bfloat16
LOSS_GROUPS = ("null_hff", "hff_unguided", "hff_guided")


def save_snapshot(directory: Path, model: torch.nn.Module, ema_parameters: list[torch.Tensor], info: dict) -> None:
    """
    Raw and EMA parameters in bf16 and the persistent buffers (linear base, residual statistics) in float32 under
    `buffers.`, with a json sidecar.
    """
    tensors: dict[str, torch.Tensor] = {}
    for (name, parameter), ema_parameter in zip(model.named_parameters(), ema_parameters):
        tensors[f"raw.{name}"] = parameter.detach().to(torch.bfloat16).cpu().contiguous()
        tensors[f"ema.{name}"] = ema_parameter.to(torch.bfloat16).cpu().contiguous()
    for name, buffer in model.named_buffers(remove_duplicate=True):
        if name in model.state_dict():
            tensors[f"buffers.{name}"] = buffer.detach().float().cpu().contiguous()
    stem = f"snapshot_{info['step']:06d}"
    save_file(tensors, str(directory / f"{stem}.safetensors"))
    (directory / f"{stem}.json").write_text(json.dumps(info))
    print(f"trainer: {stem}", flush=True)


def upload(path: Path, run_name: str) -> None:
    try:
        from stdev.s3_tools.client import get_s3_client
        from stdev.video_downloader.stream_videos import load_env_file

        load_env_file(ENV_PATH)
        get_s3_client().upload_file(str(path), UPLOAD_BUCKET, f"{run_name}/{path.name}")
        print(f"trainer: uploaded {path.name} to s3://{UPLOAD_BUCKET}/{run_name}/", flush=True)
    except Exception as error:
        print(f"trainer: upload of {path.name} failed: {error!r}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--training-minutes", type=float, default=TRAINING_MINUTES)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--decay-fraction", type=float, default=DECAY_FRACTION)
    parser.add_argument("--max-steps", type=int, default=0, help="smoke test")
    parser.add_argument("--no-upload", action="store_true")
    arguments = parser.parse_args()
    total_seconds = arguments.training_minutes * 60.0
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 64
    run_directory = RUNS_DIRECTORY / arguments.run_name
    snapshots_directory = run_directory / "snapshots"
    snapshots_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "checkpoints").mkdir(exist_ok=True)

    torch.manual_seed(SEED)
    model = build_student_from_teacher(TEACHER_CHECKPOINT, TEACHER_WEIGHTS)
    model = model.to(device).train()
    ema_parameters = [parameter.detach().float().clone() for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY, fused=True)
    save_snapshot(snapshots_directory, model, ema_parameters, {"step": 0, "training_seconds": 0.0, "samples_seen": 0})
    model.compile_blocks()

    compile_start = time.perf_counter()
    for pair_type in PAIR_TYPES:
        batch_size = pair_type.micro_batch_size
        with torch.autocast("cuda", dtype=TRAINING_DTYPE):
            velocity = model(
                torch.randn(batch_size, LATENT_CHANNELS, *pair_type.target_latent_size, device=device),
                torch.randn(batch_size, H3_CHANNELS, *pair_type.source_latent_size, device=device),
                torch.randn(batch_size, LATENT_CHANNELS, *pair_type.target_latent_size, device=device),
                torch.rand(batch_size, device=device),
                torch.ones(batch_size, dtype=torch.bool, device=device),
                torch.ones(batch_size, device=device),
            )
        velocity.float().square().mean().backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"trainer: compiled in {time.perf_counter() - compile_start:.0f}s, {count_pairs(arguments.run_name)} pairs available", flush=True)

    probabilities = micro_step_probabilities()
    loaders = {
        pair_type.name: iter(
            DataLoader(
                PairDataset(pairs_directory(arguments.run_name) / pair_type.name, seed=f"{SEED}-{pair_type.name}"),
                batch_size=pair_type.micro_batch_size,
                num_workers=DATA_WORKERS_PER_TYPE,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
            )
        )
        for pair_type in PAIR_TYPES
    }
    summary_writer = SummaryWriter(str(run_directory / "tensorboard"))
    summary_writer.add_text("config", json.dumps({**vars(arguments), "teacher": str(TEACHER_CHECKPOINT), "teacher_weights": TEACHER_WEIGHTS}), 0)
    type_names = [pair_type.name for pair_type in PAIR_TYPES]
    step = 0
    samples_seen = 0
    training_seconds = 0.0
    next_snapshot_seconds = SNAPSHOT_INTERVAL_SECONDS
    statistics = torch.zeros(2 * len(PAIR_TYPES) + 2 * len(LOSS_GROUPS), device=device)
    gradient_norm_sum = 0.0
    interval_samples = 0
    interval_start = time.perf_counter()
    step_start = time.perf_counter()
    while True:
        warmup_factor = min(1.0, (step + 1) / WARMUP_STEPS)
        decay_factor = min(1.0, max(0.0, 1.0 - training_seconds / total_seconds) / arguments.decay_fraction)
        learning_rate = arguments.learning_rate * warmup_factor * decay_factor
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        for micro_step in range(ACCUMULATION):
            pair_type = random.Random(f"{SEED}-{step}-{micro_step}").choices(PAIR_TYPES, weights=[probabilities[name] for name in type_names])[0]
            batch = next(loaders[pair_type.name])
            latents_source = batch["latents_source"].to(device, non_blocking=True).float()
            latents_generated = batch["latents_generated"].to(device, non_blocking=True).float()
            hff = batch["hff"].to(device, non_blocking=True)
            guidance = batch["guidance"].to(device, non_blocking=True)
            hff_mask = ~torch.isnan(hff)
            noise = seeded_noise(batch["noise_seed"], (LATENT_CHANNELS, *pair_type.target_latent_size), device)
            latents_base = model.linear_base(latents_source)
            residual = model.to_residual(latents_generated, latents_base)
            with torch.autocast("cuda", dtype=TRAINING_DTYPE):
                velocity = model(noise, latents_source, latents_base, torch.nan_to_num(hff, nan=0.0), hff_mask, guidance)
            sample_losses = (velocity.float() - (noise - residual)).square().mean(dim=(1, 2, 3))  # [B]
            (sample_losses.mean() / ACCUMULATION).backward()
            type_index = type_names.index(pair_type.name)
            statistics[2 * type_index] += sample_losses.detach().mean()
            statistics[2 * type_index + 1] += 1
            groups = torch.where(~hff_mask, 0, torch.where(guidance != 1.0, 2, 1))
            offset = 2 * len(PAIR_TYPES)
            statistics[offset : offset + len(LOSS_GROUPS)] += torch.zeros(len(LOSS_GROUPS), device=device).index_add_(0, groups, sample_losses.detach())
            statistics[offset + len(LOSS_GROUPS) :] += torch.bincount(groups, minlength=len(LOSS_GROUPS)).float()
            samples_seen += latents_source.shape[0]
            interval_samples += latents_source.shape[0]
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch._foreach_lerp_(ema_parameters, [parameter.detach() for parameter in model.parameters()], 1.0 - EMA_DECAY)
        gradient_norm_sum += gradient_norm.item()
        now = time.perf_counter()
        if step > 0:
            training_seconds += now - step_start
        step_start = now
        step += 1

        if step % LOG_INTERVAL_STEPS == 0:
            elapsed = now - interval_start
            values = statistics.tolist()
            offset = 2 * len(PAIR_TYPES)
            group_sums, group_counts = values[offset : offset + len(LOSS_GROUPS)], values[offset + len(LOSS_GROUPS) :]
            scalars = {
                "train/loss": sum(group_sums) / max(1.0, sum(group_counts)),
                "train/learning_rate": learning_rate,
                "train/grad_norm_mean": gradient_norm_sum / LOG_INTERVAL_STEPS,
                "train/samples_per_second": interval_samples / max(elapsed, 1e-6),
                "train/samples_seen": samples_seen,
                "train/training_minutes": training_seconds / 60,
                "data/pairs_produced": count_pairs(arguments.run_name),
            }
            scalars["data/mean_reuse"] = samples_seen / max(1, scalars["data/pairs_produced"])
            for index, name in enumerate(type_names):
                if values[2 * index + 1] > 0:
                    scalars[f"loss_by_type/{name}"] = values[2 * index] / values[2 * index + 1]
            for name, total, count in zip(LOSS_GROUPS, group_sums, group_counts):
                if count > 0:
                    scalars[f"loss_by_condition/{name}"] = total / count
            for name, value in scalars.items():
                summary_writer.add_scalar(name, value, step)
            print(
                f"trainer: step {step} loss {scalars['train/loss']:.4f} lr {learning_rate:.2e} grad {scalars['train/grad_norm_mean']:.3f} "
                f"samples {samples_seen} pairs {scalars['data/pairs_produced']} reuse {scalars['data/mean_reuse']:.2f} "
                f"{training_seconds / 60:.1f} min",
                flush=True,
            )
            statistics.zero_()
            gradient_norm_sum = 0.0
            interval_samples = 0
            interval_start = time.perf_counter()

        is_finished = training_seconds >= total_seconds or (arguments.max_steps > 0 and step >= arguments.max_steps)
        info = {"step": step, "training_seconds": training_seconds, "samples_seen": samples_seen}
        if not is_finished and training_seconds >= next_snapshot_seconds:
            save_snapshot(snapshots_directory, model, ema_parameters, info)
            next_snapshot_seconds += SNAPSHOT_INTERVAL_SECONDS
        if is_finished:
            save_snapshot(snapshots_directory, model, ema_parameters, {**info, "final": True})
            checkpoint_path = run_directory / "checkpoints" / f"checkpoint_{step:06d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": [parameter.cpu() for parameter in ema_parameters],
                    "state": info,
                    "model_config": model.config.__dict__,
                    "residual_mean": model.residual_mean.flatten().tolist(),
                    "architecture": "H3StudentTransformer",
                    "teacher": str(TEACHER_CHECKPOINT),
                },
                checkpoint_path,
            )
            print(f"trainer: finished at step {step}, {checkpoint_path.name}", flush=True)
            if not arguments.no_upload:
                upload(checkpoint_path, arguments.run_name)
            (run_directory / "training_done").write_text(json.dumps(info))
            break
    summary_writer.close()


if __name__ == "__main__":
    main()
