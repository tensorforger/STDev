"""
DDP trainer of the small single step FLUX.2 student (SMALL_MODEL), from scratch on the online pairs of the large
student (stdev.flux_distill.small_pairs). Objective as the large distillation: sigma 1, input = seeded unit noise,
target velocity = input - residual(large student latents). Every step all ranks train one pair type (drawn with the
pretraining micro-step probabilities, MICRO_BATCH_SIZE per rank, no accumulation). Fixed training time
(TRAINING_HOURS, synchronized across ranks): warmup, constant LR (square root scaled to the global batch), linear
decay over the last DECAY_FRACTION. Rank 0 keeps an EMA, saves snapshots (raw + EMA, for the launcher's evaluation)
every SNAPSHOT_INTERVAL_SECONDS and resumable checkpoints; `control/stop` or SIGTERM checkpoints and exits.

Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 -m stdev.flux_distill.small_trainer --run-name RUN
           [--training-hours H] [--max-steps N] [--no-upload]
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")

import argparse
import json
import math
import random
import signal
import threading
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as distributed
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from stdev.flux_distill.config import RUNS_DIRECTORY
from stdev.flux_distill.pairs import PairDataset, list_pair_shards
from stdev.flux_distill.small_config import (
    ADAM_BETAS,
    CHECKPOINT_INTERVAL_SECONDS,
    CONTROL_CHECK_INTERVAL_STEPS,
    DATA_WORKERS_PER_TYPE,
    DECAY_FRACTION,
    EMA_DECAY,
    GRADIENT_CLIP_NORM,
    LOG_INTERVAL_STEPS,
    MICRO_BATCH_SIZE,
    SEED,
    SMALL_MODEL,
    SNAPSHOT_INTERVAL_SECONDS,
    TRAINING_HOURS,
    WARMUP_MIN_SHARDS_PER_TYPE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    learning_rate_for,
)
from stdev.flux_distill.small_pairs import small_pairs_directory
from stdev.flux_distill.student import FluxStudentTransformer, seeded_noise
from stdev.flux_distill.trainer import upload
from stdev.flux_pretrain.config import PAIR_TYPES, TOKENS_PER_UNIT, micro_step_probabilities
from stdev.flux_pretrain.model import LATENT_CHANNELS, to_residual

TRAINING_DTYPE = torch.bfloat16
LOSS_GROUPS = ("null_hff", "hff_unguided", "hff_guided")


def count_produced(pairs_directory: Path) -> int:
    total = 0
    for statistics_path in pairs_directory.glob("statistics_*.json"):
        try:
            total += sum(json.loads(statistics_path.read_text())["produced"].values())
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            continue
    return total


def save_snapshot(directory: Path, model: torch.nn.Module, ema: list[torch.Tensor], info: dict) -> None:
    tensors: dict[str, torch.Tensor] = {}
    for (name, parameter), ema_parameter in zip(model.named_parameters(), ema):
        tensors[f"raw.{name}"] = parameter.detach().to(torch.bfloat16).cpu().contiguous()
        tensors[f"ema.{name}"] = ema_parameter.to(torch.bfloat16).cpu().contiguous()
    stem = f"snapshot_{info['step']:07d}"
    temporary_path = directory / f".{stem}.tmp"
    save_file(tensors, str(temporary_path))
    (directory / f"{stem}.json").write_text(json.dumps(info))
    temporary_path.replace(directory / f"{stem}.safetensors")
    print(f"trainer: {stem}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--training-hours", type=float, default=TRAINING_HOURS)
    parser.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps of this session")
    parser.add_argument("--no-upload", action="store_true")
    arguments = parser.parse_args()
    total_seconds = arguments.training_hours * 3600.0

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed.init_process_group("nccl", timeout=timedelta(minutes=15), device_id=device)
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    is_main = rank == 0
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 64
    run_directory = RUNS_DIRECTORY / arguments.run_name
    snapshots_directory = run_directory / "snapshots"
    checkpoints_directory = run_directory / "checkpoints"
    control_directory = run_directory / "control"
    if is_main:
        for directory in (snapshots_directory, checkpoints_directory, control_directory):
            directory.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    pairs_directory = small_pairs_directory(arguments.run_name)
    global_batch = world_size * MICRO_BATCH_SIZE[PAIR_TYPES[0].name]
    base_learning_rate = learning_rate_for(global_batch)

    torch.manual_seed(SEED)
    model = FluxStudentTransformer(SMALL_MODEL).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_learning_rate, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY, fused=True)
    state = {
        "step": 0,
        "samples_seen": 0,
        "units_seen": 0.0,
        "training_seconds": 0.0,
        "next_snapshot_seconds": 0.0,
        "next_checkpoint_seconds": CHECKPOINT_INTERVAL_SECONDS,
        "ema_updates": 0,
    }
    ema: list[torch.Tensor] = []
    checkpoints = sorted(checkpoints_directory.glob("checkpoint_*.pt"))
    if checkpoints:
        checkpoint = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        state.update(checkpoint["state"])
        if is_main:
            ema = [parameter.to(device) for parameter in checkpoint["ema"]]
            print(f"trainer: resumed from {checkpoints[-1].name} at step {state['step']}", flush=True)
        del checkpoint
    if is_main and not ema:
        ema = [parameter.detach().float().clone() for parameter in model.parameters()]
    parameters = list(model.parameters())
    model.compile_blocks()
    distributed_model = DistributedDataParallel(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    def training_loss(batch: dict[str, torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
        """
        returns:
            sample_losses: [B], velocity MSE
        """
        latents_source = batch["latents_source"].float()
        noise = seeded_noise(batch["noise_seed"], (LATENT_CHANNELS, *target_size), device)
        residual = to_residual(batch["latents_generated"].float(), latents_source, SMALL_MODEL.residual_std)
        hff_mask = ~torch.isnan(batch["hff"])
        with torch.autocast("cuda", dtype=TRAINING_DTYPE):
            velocity = distributed_model(noise, latents_source, torch.nan_to_num(batch["hff"], nan=0.0), hff_mask, batch["guidance"])
        return (velocity.float() - (noise - residual)).square().mean(dim=(1, 2, 3))

    compile_start = time.perf_counter()
    for pair_type in PAIR_TYPES:
        size = MICRO_BATCH_SIZE[pair_type.name]
        dummy = {
            "latents_source": torch.randn(size, LATENT_CHANNELS, *pair_type.source_latent_size, device=device),
            "latents_generated": torch.randn(size, LATENT_CHANNELS, *pair_type.target_latent_size, device=device),
            "hff": torch.rand(size, device=device),
            "guidance": torch.ones(size, device=device),
            "noise_seed": torch.arange(size, dtype=torch.int64),
        }
        with distributed_model.no_sync():
            training_loss(dummy, pair_type.target_latent_size).mean().backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if is_main:
        print(
            f"trainer: {sum(parameter.numel() for parameter in parameters) / 1e6:.2f}M parameters, {world_size} ranks, global batch {global_batch} @1024, "
            f"lr {base_learning_rate:.2e}, compiled in {time.perf_counter() - compile_start:.0f}s, {arguments.training_hours} h schedule",
            flush=True,
        )
    while not all(len(list_pair_shards(pairs_directory / pair_type.name)) >= WARMUP_MIN_SHARDS_PER_TYPE for pair_type in PAIR_TYPES):
        time.sleep(10.0)
    probabilities = micro_step_probabilities()
    loaders = {
        pair_type.name: iter(
            DataLoader(
                PairDataset(pairs_directory / pair_type.name, seed=f"{SEED}-{rank}-{pair_type.name}-{state['step']}"),
                batch_size=MICRO_BATCH_SIZE[pair_type.name],
                num_workers=DATA_WORKERS_PER_TYPE if probabilities[pair_type.name] > 0.05 else 2,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
            )
        )
        for pair_type in PAIR_TYPES
    }
    summary_writer = SummaryWriter(str(run_directory / "tensorboard")) if is_main else None
    if is_main:
        summary_writer.add_text("config", json.dumps({**vars(arguments), "model": SMALL_MODEL.__dict__, "lr": base_learning_rate, "global_batch": global_batch}), state["step"])
        if state["step"] == 0:
            save_snapshot(snapshots_directory, model, ema, {"step": 0, "training_seconds": 0.0, "samples_seen": 0})
            state["next_snapshot_seconds"] = SNAPSHOT_INTERVAL_SECONDS

    termination_requested = False

    def request_termination(signal_number: int, frame: object) -> None:
        nonlocal termination_requested
        termination_requested = True

    signal.signal(signal.SIGTERM, request_termination)

    def save_checkpoint() -> Path:
        path = checkpoints_directory / f"checkpoint_{state['step']:07d}.pt"
        temporary_path = path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "ema": [parameter.cpu() for parameter in ema],
                "optimizer": optimizer.state_dict(),
                "state": dict(state),
                "model_config": SMALL_MODEL.__dict__,
                "architecture": "FluxStudentTransformer",
            },
            temporary_path,
        )
        temporary_path.replace(path)
        for old_path in sorted(checkpoints_directory.glob("checkpoint_*.pt"))[:-2]:
            old_path.unlink()
        print(f"trainer: checkpoint {path.name}", flush=True)
        return path

    type_names = [pair_type.name for pair_type in PAIR_TYPES]
    statistics = torch.zeros(2 * len(PAIR_TYPES) + 2 * len(LOSS_GROUPS), device=device)
    gradient_norm_sum = torch.zeros((), device=device)
    gradient_norm_max = torch.zeros((), device=device)
    schedule_seconds = state["training_seconds"]
    is_finished = False
    steps_this_session = 0
    interval_samples = 0
    interval_start = time.perf_counter()
    step_start = time.perf_counter()
    upload_thread: threading.Thread | None = None
    while True:
        warmup_factor = min(1.0, (state["step"] + 1) / WARMUP_STEPS)
        decay_factor = min(1.0, max(0.0, 1.0 - schedule_seconds / total_seconds) / DECAY_FRACTION)
        learning_rate = base_learning_rate * warmup_factor * decay_factor
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        pair_type = random.Random(f"{SEED}-{state['step']}").choices(PAIR_TYPES, weights=[probabilities[name] for name in type_names])[0]
        batch = {name: tensor.to(device, non_blocking=True) for name, tensor in next(loaders[pair_type.name]).items()}
        sample_losses = training_loss(batch, pair_type.target_latent_size)
        sample_losses.mean().backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if is_main:
            state["ema_updates"] += 1
            decay = min(EMA_DECAY, (1.0 + state["ema_updates"]) / (10.0 + state["ema_updates"]))
            torch._foreach_lerp_(ema, [parameter.detach() for parameter in parameters], 1.0 - decay)
        detached = sample_losses.detach()
        type_index = type_names.index(pair_type.name)
        statistics[2 * type_index] += detached.mean()
        statistics[2 * type_index + 1] += 1
        hff_mask = ~torch.isnan(batch["hff"])
        groups = torch.where(~hff_mask, 0, torch.where(batch["guidance"] != 1.0, 2, 1))
        offset = 2 * len(PAIR_TYPES)
        statistics[offset : offset + len(LOSS_GROUPS)] += torch.zeros(len(LOSS_GROUPS), device=device).index_add_(0, groups, detached)
        statistics[offset + len(LOSS_GROUPS) :] += torch.bincount(groups, minlength=len(LOSS_GROUPS)).float()
        gradient_norm_sum += gradient_norm.detach()
        gradient_norm_max = torch.maximum(gradient_norm_max, gradient_norm.detach())
        step_samples = world_size * detached.shape[0]
        now = time.perf_counter()
        if steps_this_session > 0:
            state["training_seconds"] += now - step_start
        step_start = now
        steps_this_session += 1
        state["step"] += 1
        state["samples_seen"] += step_samples
        state["units_seen"] += step_samples * pair_type.tokens_per_sample / TOKENS_PER_UNIT
        interval_samples += step_samples

        if state["step"] % CONTROL_CHECK_INTERVAL_STEPS == 0 or termination_requested:
            control = torch.zeros(2, device=device)
            if is_main:
                control[0] = float((control_directory / "stop").exists())
                control[1] = state["training_seconds"]
            control[0] += float(termination_requested)
            distributed.all_reduce(control, op=distributed.ReduceOp.MAX)
            state["training_seconds"] = control[1].item()
            schedule_seconds = state["training_seconds"]
            is_finished = schedule_seconds >= total_seconds
            if control[0] > 0:
                if is_main:
                    save_checkpoint()
                    (control_directory / "stop").unlink(missing_ok=True)
                    print("trainer: stopped on request", flush=True)
                break

        if state["step"] % LOG_INTERVAL_STEPS == 0:
            distributed.all_reduce(statistics)
            if is_main:
                elapsed = time.perf_counter() - interval_start
                values = statistics.tolist()
                group_sums, group_counts = values[offset : offset + len(LOSS_GROUPS)], values[offset + len(LOSS_GROUPS) :]
                produced = count_produced(pairs_directory)
                scalars = {
                    "train/loss": sum(group_sums) / max(1.0, sum(group_counts)),
                    "train/learning_rate": learning_rate,
                    "train/grad_norm_mean": gradient_norm_sum.item() / LOG_INTERVAL_STEPS,
                    "train/grad_norm_max": gradient_norm_max.item(),
                    "train/samples_per_second": interval_samples / max(elapsed, 1e-6),
                    "train/samples_seen": state["samples_seen"],
                    "train/training_hours": state["training_seconds"] / 3600,
                    "system/gpu_memory_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "data/pairs_produced": produced,
                    "data/mean_reuse": state["samples_seen"] / max(1, produced),
                }
                for index, name in enumerate(type_names):
                    if values[2 * index + 1] > 0:
                        scalars[f"loss_by_type/{name}"] = values[2 * index] / values[2 * index + 1]
                for name, total, count in zip(LOSS_GROUPS, group_sums, group_counts):
                    if count > 0:
                        scalars[f"loss_by_condition/{name}"] = total / count
                for name, value in scalars.items():
                    summary_writer.add_scalar(name, value, state["step"])
                print(
                    f"trainer: step {state['step']} loss {scalars['train/loss']:.4f} lr {learning_rate:.2e} grad "
                    f"{scalars['train/grad_norm_mean']:.3f}/{scalars['train/grad_norm_max']:.3f} {scalars['train/samples_per_second']:.1f} samples/s, "
                    f"reuse {scalars['data/mean_reuse']:.2f}, mem {scalars['system/gpu_memory_peak_gb']:.1f} GB, {state['training_seconds'] / 3600:.2f} h",
                    flush=True,
                )
                torch.cuda.reset_peak_memory_stats()
            statistics.zero_()
            gradient_norm_sum.zero_()
            gradient_norm_max.zero_()
            interval_samples = 0
            interval_start = time.perf_counter()

        info = {"step": state["step"], "training_seconds": state["training_seconds"], "samples_seen": state["samples_seen"]}
        is_smoke_finished = arguments.max_steps > 0 and steps_this_session >= arguments.max_steps
        if is_main and not is_finished and state["training_seconds"] >= state["next_snapshot_seconds"]:
            save_snapshot(snapshots_directory, model, ema, info)
            state["next_snapshot_seconds"] = state["training_seconds"] + SNAPSHOT_INTERVAL_SECONDS
        if is_main and not is_finished and state["training_seconds"] >= state["next_checkpoint_seconds"]:
            save_checkpoint()
            state["next_checkpoint_seconds"] = state["training_seconds"] + CHECKPOINT_INTERVAL_SECONDS
        if is_finished or is_smoke_finished:
            if is_main:
                checkpoint_path = save_checkpoint()
                save_snapshot(snapshots_directory, model, ema, {**info, "final": True})
                if is_finished:
                    (run_directory / "training_done").write_text(json.dumps(info))
                    if not arguments.no_upload:
                        upload_thread = threading.Thread(target=upload, args=(checkpoint_path, arguments.run_name))
                        upload_thread.start()
                print(f"trainer: finished at step {state['step']}", flush=True)
            break

    if summary_writer is not None:
        summary_writer.close()
    if upload_thread is not None:
        upload_thread.join()
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
