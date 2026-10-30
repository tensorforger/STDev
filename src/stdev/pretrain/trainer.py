"""
DDP trainer of the pretraining pipeline. Reads the rolling latent cache, trains with a constant
learning rate after warmup and keeps an EMA of the weights on rank 0. The final linear decay to
zero starts when the control file `control/decay` appears (optional content: decay steps,
default DEFAULT_DECAY_FRACTION of the steps done); training ends when the decay is done.
`control/stop` (or SIGTERM) saves a checkpoint and exits; a restart resumes from the latest
checkpoint. Rank 0 saves benchmark snapshots (EMA and raw weights) on the benchmark schedule.

Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc-per-node 7 -m stdev.pretrain.trainer
           --run-name NAME --stage stage_1 [--init-checkpoint PATH] [--max-steps N]
"""

import argparse
import json
import os
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from safetensors.torch import save_file
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from stdev.pretrain.cache import CachedLatentDataset, list_shards
from stdev.pretrain.config import (
    ADAM_BETAS,
    BENCHMARK_EARLY_INTERVAL_SECONDS,
    BENCHMARK_EARLY_PERIOD_SECONDS,
    BENCHMARK_LATE_INTERVAL_SECONDS,
    CHECKPOINT_INTERVAL_SECONDS,
    CONTROL_CHECK_INTERVAL_STEPS,
    DATA_WORKERS_PER_RANK,
    DEFAULT_DECAY_FRACTION,
    EMA_DECAY,
    GRADIENT_CLIP_NORM,
    HIGH_FREQUENCY_DROPOUT_PROBABILITY,
    KEPT_CHECKPOINTS,
    LEARNING_RATE,
    LOG_INTERVAL_STEPS,
    SEED,
    STAGES,
    TEXT_DROPOUT_PROBABILITY,
    TRAINING_SIGMA_SHIFT,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    PretrainPaths,
    run_paths,
)
from stdev.upscalers.pretrain_large import UpscalerUNet, shift_sigmas

TRAINING_DTYPE = torch.bfloat16


def next_benchmark_time(training_seconds: float) -> float:
    """
    Every BENCHMARK_EARLY_INTERVAL_SECONDS during the first BENCHMARK_EARLY_PERIOD_SECONDS of
    training, then every BENCHMARK_LATE_INTERVAL_SECONDS.
    """
    if training_seconds < BENCHMARK_EARLY_PERIOD_SECONDS:
        intervals = int(training_seconds // BENCHMARK_EARLY_INTERVAL_SECONDS) + 1
        return min(intervals * BENCHMARK_EARLY_INTERVAL_SECONDS, BENCHMARK_EARLY_PERIOD_SECONDS)
    late_seconds = training_seconds - BENCHMARK_EARLY_PERIOD_SECONDS
    intervals = int(late_seconds // BENCHMARK_LATE_INTERVAL_SECONDS) + 1
    return BENCHMARK_EARLY_PERIOD_SECONDS + intervals * BENCHMARK_LATE_INTERVAL_SECONDS


def sample_training_sigmas(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    returns:
        sigmas: [B], values: [0...1], logit-normal distributed and shifted
    """
    return shift_sigmas(torch.sigmoid(torch.randn(batch_size, device=device)), TRAINING_SIGMA_SHIFT)


class TrainingState:
    """
    Resumable counters and learning rate schedule state.
    """

    def __init__(self) -> None:
        self.stage = ""
        self.step = 0
        self.samples_seen = 0
        self.training_seconds = 0.0
        self.decay_start_step = -1
        self.decay_steps = 0
        self.next_benchmark_seconds = next_benchmark_time(0.0)
        self.next_checkpoint_seconds = float(CHECKPOINT_INTERVAL_SECONDS)
        self.ema_updates = 0

    def learning_rate(self) -> float:
        warmup_factor = min(1.0, (self.step + 1) / WARMUP_STEPS)
        if self.decay_start_step < 0:
            return LEARNING_RATE * warmup_factor
        decay_progress = (self.step - self.decay_start_step) / max(self.decay_steps, 1)
        return LEARNING_RATE * warmup_factor * max(0.0, 1.0 - decay_progress)

    def decay_finished(self) -> bool:
        return self.decay_start_step >= 0 and self.step >= self.decay_start_step + self.decay_steps

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def load_dict(self, values: dict) -> None:
        self.__dict__.update(values)


def latest_checkpoint(paths: PretrainPaths) -> Path | None:
    checkpoints = sorted(paths.checkpoints.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(
    paths: PretrainPaths,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_parameters: list[torch.Tensor],
    state: TrainingState,
) -> None:
    """
    Full resumable checkpoint (weights, EMA, optimizer, schedule state); keeps the newest
    KEPT_CHECKPOINTS.
    """
    path = paths.checkpoints / f"checkpoint_{state.step:09d}.pt"
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": [parameter.cpu() for parameter in ema_parameters],
            "optimizer": optimizer.state_dict(),
            "state": state.to_dict(),
        },
        temporary_path,
    )
    temporary_path.replace(path)
    for old_path in sorted(paths.checkpoints.glob("checkpoint_*.pt"))[:-KEPT_CHECKPOINTS]:
        old_path.unlink()
    print(f"trainer: checkpoint {path.name}", flush=True)


def save_snapshot(
    paths: PretrainPaths,
    model: torch.nn.Module,
    ema_parameters: list[torch.Tensor],
    state: TrainingState,
    tag: str,
) -> None:
    """
    EMA and raw weights in bf16 for the benchmark process, with a json sidecar.
    """
    tensors: dict[str, torch.Tensor] = {}
    for (name, parameter), ema_parameter in zip(model.named_parameters(), ema_parameters):
        tensors[f"raw.{name}"] = parameter.detach().to(torch.bfloat16).cpu().contiguous()
        tensors[f"ema.{name}"] = ema_parameter.to(torch.bfloat16).cpu().contiguous()
    for name, buffer in model.named_buffers():
        tensors[f"raw.{name}"] = buffer.detach().cpu().contiguous()
        tensors[f"ema.{name}"] = buffer.detach().cpu().contiguous()
    stem = f"{tag}_{state.step:09d}"
    temporary_path = paths.snapshots / f".{stem}.tmp"
    save_file(tensors, str(temporary_path))
    (paths.snapshots / f"{stem}.json").write_text(
        json.dumps(
            {
                "stage": state.stage,
                "step": state.step,
                "samples_seen": state.samples_seen,
                "training_seconds": state.training_seconds,
                "tag": tag,
            }
        )
    )
    temporary_path.replace(paths.snapshots / f"{stem}.safetensors")
    print(f"trainer: snapshot {stem}", flush=True)


def read_control(paths: PretrainPaths, state: TrainingState) -> tuple[int, int]:
    """
    returns:
        decay_steps (0 when no new decay request), stop_requested (0 or 1)
    """
    decay_steps = 0
    if state.decay_start_step < 0 and paths.decay_request.exists():
        content = paths.decay_request.read_text().strip()
        decay_steps = int(content) if content else max(1, round(DEFAULT_DECAY_FRACTION * state.step))
        paths.decay_request.rename(paths.control / "decay.accepted")
    return decay_steps, int(paths.stop_request.exists())


def read_producer_total(paths: PretrainPaths) -> int:
    try:
        return int(json.loads(paths.producer_statistics.read_text())["produced_samples_total"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", default="stage_1", choices=STAGES)
    parser.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps")
    parser.add_argument(
        "--init-checkpoint", type=Path, help="weights of the previous stage for a new run"
    )
    arguments = parser.parse_args()

    distributed.init_process_group("nccl")
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(SEED + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    paths = run_paths(arguments.run_name)
    stage = STAGES[arguments.stage]
    is_main = rank == 0

    model = UpscalerUNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    state = TrainingState()
    state.stage = stage.name
    ema_parameters: list[torch.Tensor] = []

    checkpoint_path = latest_checkpoint(paths)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        state.load_dict(checkpoint["state"])
        if is_main:
            ema_parameters = [parameter.to(device) for parameter in checkpoint["ema"]]
            print(f"trainer: resumed from {checkpoint_path.name} at step {state.step}", flush=True)
        del checkpoint
    elif arguments.init_checkpoint is not None:
        checkpoint = torch.load(arguments.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        if is_main:
            print(f"trainer: initialized from {arguments.init_checkpoint}", flush=True)
    if is_main and not ema_parameters:
        ema_parameters = [parameter.detach().float().clone() for parameter in model.parameters()]

    distributed_model = DistributedDataParallel(
        model, device_ids=[local_rank], gradient_as_bucket_view=True
    )
    distributed_model.register_comm_hook(None, default_hooks.bf16_compress_hook)
    compiled_model = torch.compile(distributed_model)
    parameters = list(model.parameters())

    data_loader = DataLoader(
        CachedLatentDataset(paths.cache, paths.high_frequency_quantiles, seed=SEED * 1000 + rank),
        batch_size=stage.micro_batch_size,
        num_workers=DATA_WORKERS_PER_RANK,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    batches = iter(data_loader)
    samples_per_step = stage.micro_batch_size * stage.gradient_accumulation_steps * world_size
    summary_writer = SummaryWriter(str(paths.tensorboard / "train")) if is_main else None
    if is_main:
        parameter_count = sum(parameter.numel() for parameter in parameters)
        print(
            f"trainer: {parameter_count / 1e6:.1f}M parameters, {world_size} ranks, "
            f"{samples_per_step} samples per step",
            flush=True,
        )

    termination_requested = False

    def request_termination(signal_number: int, frame: object) -> None:
        nonlocal termination_requested
        termination_requested = True

    signal.signal(signal.SIGTERM, request_termination)

    loss_sum = torch.zeros((), device=device)
    loss_count = 0
    interval_start_time = time.perf_counter()
    interval_start_step = state.step
    step_start_time = time.perf_counter()
    steps_this_session = 0

    while True:
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = state.learning_rate()
        for micro_step in range(stage.gradient_accumulation_steps):
            batch = next(batches)
            latents_high = batch["latents_high"].to(device, non_blocking=True)
            latents_low = batch["latents_low"].to(device, non_blocking=True)
            text_embedding = batch["text_embedding"].to(device, non_blocking=True)
            features = batch["high_frequency_features"].to(device, non_blocking=True)
            batch_size = latents_high.shape[0]
            text_mask = torch.rand(batch_size, device=device) >= TEXT_DROPOUT_PROBABILITY
            feature_mask = (
                torch.rand(features.shape, device=device) >= HIGH_FREQUENCY_DROPOUT_PROBABILITY
            ) & ~torch.isnan(features)
            noise = torch.randn_like(latents_high)
            sigmas = sample_training_sigmas(batch_size, device)
            blend = sigmas[:, None, None, None]
            noisy_latents = (1.0 - blend) * latents_high + blend * noise
            is_last_micro_step = micro_step == stage.gradient_accumulation_steps - 1
            synchronization = nullcontext() if is_last_micro_step else distributed_model.no_sync()
            with synchronization:
                with torch.autocast("cuda", dtype=TRAINING_DTYPE):
                    velocity = compiled_model(
                        noisy_latents,
                        sigmas,
                        latents_low,
                        text_embedding,
                        text_mask,
                        features,
                        feature_mask,
                    )
                loss = functional.mse_loss(velocity.float(), noise - latents_high)
                (loss / stage.gradient_accumulation_steps).backward()
            loss_sum += loss.detach()
            loss_count += 1
        torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if is_main:
            state.ema_updates += 1
            decay = min(EMA_DECAY, (1.0 + state.ema_updates) / (10.0 + state.ema_updates))
            torch._foreach_lerp_(
                ema_parameters, [parameter.detach() for parameter in parameters], 1.0 - decay
            )

        now = time.perf_counter()
        if steps_this_session > 0:
            state.training_seconds += now - step_start_time
        else:
            interval_start_time = now
            interval_start_step = state.step + 1
        step_start_time = now
        state.step += 1
        state.samples_seen += samples_per_step
        steps_this_session += 1

        if state.step % CONTROL_CHECK_INTERVAL_STEPS == 0 or termination_requested:
            control = torch.zeros(3, device=device)
            if is_main:
                decay_steps, stop_requested = read_control(paths, state)
                control[0] = decay_steps
                control[1] = stop_requested
                control[2] = state.training_seconds
            control[1] += float(termination_requested)
            distributed.all_reduce(control, op=distributed.ReduceOp.MAX)
            state.training_seconds = control[2].item()
            if control[0] > 0 and state.decay_start_step < 0:
                state.decay_start_step = state.step
                state.decay_steps = int(control[0].item())
                if is_main:
                    print(f"trainer: decay over {state.decay_steps} steps from step {state.step}", flush=True)
            if control[1] > 0:
                if is_main:
                    save_checkpoint(paths, model, optimizer, ema_parameters, state)
                    paths.stop_request.unlink(missing_ok=True)
                    print("trainer: stopped on request", flush=True)
                break

        if is_main and state.step % LOG_INTERVAL_STEPS == 0:
            elapsed = time.perf_counter() - interval_start_time
            steps = state.step - interval_start_step
            mean_loss = loss_sum.item() / max(loss_count, 1)
            samples_per_second = steps * samples_per_step / max(elapsed, 1e-6)
            produced_total = read_producer_total(paths)
            summary_writer.add_scalar("train/loss", mean_loss, state.step)
            summary_writer.add_scalar("train/learning_rate", state.learning_rate(), state.step)
            summary_writer.add_scalar("train/samples_per_second", samples_per_second, state.step)
            summary_writer.add_scalar("train/samples_seen", state.samples_seen, state.step)
            summary_writer.add_scalar("train/training_hours", state.training_seconds / 3600, state.step)
            summary_writer.add_scalar("data/produced_samples", produced_total, state.step)
            summary_writer.add_scalar("data/cache_shards", len(list_shards(paths.cache)), state.step)
            if produced_total > 0:
                summary_writer.add_scalar(
                    "data/mean_reuse", state.samples_seen / produced_total, state.step
                )
            print(
                f"trainer: step {state.step} loss {mean_loss:.4f} lr {state.learning_rate():.2e} "
                f"{samples_per_second:.0f} samples/s, {state.training_seconds / 3600:.2f} h",
                flush=True,
            )
            loss_sum.zero_()
            loss_count = 0
            interval_start_time = time.perf_counter()
            interval_start_step = state.step

        if is_main and state.training_seconds >= state.next_benchmark_seconds:
            save_snapshot(paths, model, ema_parameters, state, "scheduled")
            state.next_benchmark_seconds = next_benchmark_time(state.training_seconds)
        if is_main and state.training_seconds >= state.next_checkpoint_seconds:
            save_checkpoint(paths, model, optimizer, ema_parameters, state)
            state.next_checkpoint_seconds = state.training_seconds + CHECKPOINT_INTERVAL_SECONDS

        is_smoke_finished = arguments.max_steps > 0 and steps_this_session >= arguments.max_steps
        if state.decay_finished() or is_smoke_finished:
            if is_main:
                save_checkpoint(paths, model, optimizer, ema_parameters, state)
                save_snapshot(paths, model, ema_parameters, state, "final")
                if state.decay_finished():
                    paths.training_done.write_text(json.dumps(state.to_dict()))
                print(f"trainer: finished at step {state.step}", flush=True)
            break

    if summary_writer is not None:
        summary_writer.close()
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
