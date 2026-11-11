"""
Single-GPU training of the small single step upscaler (stdev.upscalers.distill_small) on the
rectified flow distillation pairs of the large model (scripts/generate_distill_pairs.py).

The model always takes the teacher's initial noise of a pair (pure noise, t = 1) and the low
resolution latents and predicts the velocity noise - teacher_output, so the sample is
noise - velocity. After every epoch over all pairs the EMA weights are benchmarked with the
unmodified scripts/run_benchmark.py on flux_1024; images and metrics are kept in
raw_results/NAME_epoch_NNN, loss and benchmark metrics go to tensorboard (x axis: optimizer step).

Learning rate: constant after warmup. Creating `distill_runs/NAME/control/decay` (optional
content: number of decay epochs, default 1) starts a linear decay to zero over the whole next
epoch(s), after which training finishes. `control/stop` ends training after the current epoch.
Control files are read after the benchmark of an epoch. A restart resumes from the last epoch.

With --architecture flow the same recipe trains a random init copy of the FlowUpscaler UNet
(stdev.upscalers.flow_upscaler) instead, to separate architecture from training setup.

Usage: python scripts/run_small_distill.py --run-name NAME [--architecture small|flow] [--smoke]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from stdev.distill.config import (
    ADAM_BETAS,
    DATA_WORKERS_PER_RANK,
    GRADIENT_CLIP_NORM,
    KEPT_CHECKPOINTS,
    LOG_INTERVAL_STEPS,
    SEED,
    SFT_PAIRS_DIRECTORY,
    DistillPaths,
    run_paths,
)
from stdev.distill.data import EpochPairDataset, list_bucket_pair_shards, shard_pair_count
from stdev.pretrain.config import REPOSITORY_ROOT
from stdev.upscalers import distill_small, flow_upscaler

TRAINING_DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
COMPILE_CACHE_SIZE_LIMIT = 64

BATCH_SIZE = 16
MICRO_BATCH_SIZE = 4
GRADIENT_CHECKPOINTING = False
POOL_SHARDS = 4
LEARNING_RATE = 4e-4
WARMUP_STEPS = 100
EMA_DECAY = 0.999
WEIGHT_DECAY = 0.0

SMOKE_STEPS_PER_EPOCH = 20

ARCHITECTURE_MODULES = {"small": distill_small, "flow": flow_upscaler}
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "run_benchmark.py"
BENCHMARK_DATASET = REPOSITORY_ROOT / "benchmark_datasets" / "flux_1024"
BENCHMARK_DATASET_NAME = BENCHMARK_DATASET.name
RAW_RESULTS_DIRECTORY = REPOSITORY_ROOT / "raw_results"


class TrainingState:
    """
    Resumable counters and learning rate schedule state.
    """

    def __init__(self) -> None:
        self.epoch = 0
        self.step = 0
        self.samples_seen = 0
        self.training_seconds = 0.0
        self.decay_start_step = -1
        self.decay_steps = 0
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


def latest_checkpoint(checkpoints_directory: Path) -> Path | None:
    checkpoints = sorted(checkpoints_directory.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(
    checkpoints_directory: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_parameters: list[torch.Tensor],
    state: TrainingState,
) -> None:
    path = checkpoints_directory / f"checkpoint_epoch_{state.epoch:04d}.pt"
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
    for old_path in sorted(checkpoints_directory.glob("checkpoint_*.pt"))[:-KEPT_CHECKPOINTS]:
        old_path.unlink()


def export_weights(
    paths: DistillPaths, model: torch.nn.Module, ema_parameters: list[torch.Tensor]
) -> None:
    """
    Latest raw and EMA weights (fp32) as export/student_{raw,ema}.safetensors.
    """
    raw = {
        name: tensor.detach().float().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    ema = dict(raw)
    for (name, _), ema_parameter in zip(model.named_parameters(), ema_parameters):
        ema[name] = ema_parameter.float().cpu().contiguous()
    for variant, tensors in (("raw", raw), ("ema", ema)):
        temporary_path = paths.export / f".student_{variant}.tmp"
        save_file(tensors, str(temporary_path))
        temporary_path.replace(paths.export / f"student_{variant}.safetensors")


def read_control(paths: DistillPaths, state: TrainingState, steps_per_epoch: int) -> bool:
    """
    Starts a decay on request.

    returns:
        True when a decay was started
    """
    if state.decay_start_step >= 0 or not paths.decay_request.exists():
        return False
    content = paths.decay_request.read_text().strip()
    decay_epochs = int(content) if content else 1
    paths.decay_request.rename(paths.control / "decay.accepted")
    state.decay_start_step = state.step
    state.decay_steps = decay_epochs * steps_per_epoch
    return True


def build_model(architecture: str) -> torch.nn.Module:
    if architecture == "flow":
        return flow_upscaler.FlowUpscalerUNet()
    return distill_small.SmallUpscalerUNet(gradient_checkpointing=GRADIENT_CHECKPOINTING)


def run_epoch_benchmark(
    paths: DistillPaths, run_name: str, epoch: int, smoke: bool, architecture: str
) -> dict | None:
    """
    Benchmarks the exported EMA weights with scripts/run_benchmark.py as a separate process.

    returns:
        parsed metrics.json, None in smoke mode (the benchmark writes no results then)
    """
    result_name = f"{run_name}_epoch_{epoch:03d}"
    upscaler_module = ARCHITECTURE_MODULES[architecture]
    environment = {
        **os.environ,
        upscaler_module.WEIGHTS_PATH_VARIABLE: str(paths.export / "student_ema.safetensors"),
        upscaler_module.RESULT_NAME_VARIABLE: result_name,
    }
    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--datasets",
        str(BENCHMARK_DATASET),
        "--upscaler",
        str(upscaler_module.__file__),
    ]
    if smoke:
        command.append("--smoke")
    torch.cuda.empty_cache()
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)
    if smoke:
        return None
    return json.loads((RAW_RESULTS_DIRECTORY / result_name / "metrics.json").read_text())


def log_benchmark(summary_writer: SummaryWriter, metrics: dict, state: TrainingState) -> None:
    quality = metrics["quality"][BENCHMARK_DATASET_NAME]
    for name, value in quality.items():
        summary_writer.add_scalar(f"benchmark/{name}", value, state.step)
    for name, value in metrics["efficiency"].items():
        if name.startswith("latency_ms_avg"):
            summary_writer.add_scalar(f"benchmark/{name}", value, state.step)
    summary_writer.add_scalar("benchmark/epoch", state.epoch, state.step)
    summary_writer.flush()
    print(
        f"run_small_distill: epoch {state.epoch} benchmark "
        + " ".join(f"{name} {value:.4f}" for name, value in quality.items()),
        flush=True,
    )


def run_epoch(
    compiled_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_parameters: list[torch.Tensor],
    state: TrainingState,
    shard_paths: list[Path],
    steps_per_epoch: int,
    summary_writer: SummaryWriter,
) -> float:
    """
    returns:
        mean training loss over the epoch
    """
    parameters = list(compiled_model.parameters())
    accumulation_steps = BATCH_SIZE // MICRO_BATCH_SIZE
    data_loader = DataLoader(
        EpochPairDataset(shard_paths, MICRO_BATCH_SIZE, POOL_SHARDS, seed=SEED + state.epoch),
        batch_size=None,
        num_workers=DATA_WORKERS_PER_RANK,
        pin_memory=True,
        prefetch_factor=4,
    )
    batches = iter(data_loader)
    loss_sum = torch.zeros((), device=DEVICE)
    loss_count = 0
    epoch_loss_sum = torch.zeros((), device=DEVICE)
    epoch_loss_count = 0
    interval_start_step = state.step
    interval_start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    for step_in_epoch in range(steps_per_epoch):
        learning_rate = state.learning_rate()
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        for _ in range(accumulation_steps):
            batch = next(batches)
            noise = batch["noise"].to(DEVICE, non_blocking=True)
            latents_teacher = batch["latents_teacher"].to(DEVICE, non_blocking=True)
            latents_low = batch["latents_low"].to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=TRAINING_DTYPE):
                velocity = compiled_model(noise, latents_low)
            loss = functional.mse_loss(velocity.float(), noise - latents_teacher)
            sample_weight = noise.shape[0] / MICRO_BATCH_SIZE
            (loss * sample_weight / accumulation_steps).backward()
            loss_sum += loss.detach()
            loss_count += 1
            epoch_loss_sum += loss.detach()
            epoch_loss_count += 1
            state.samples_seen += noise.shape[0]
        torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        state.ema_updates += 1
        decay = min(EMA_DECAY, (1.0 + state.ema_updates) / (10.0 + state.ema_updates))
        torch._foreach_lerp_(
            ema_parameters, [parameter.detach() for parameter in parameters], 1.0 - decay
        )
        state.step += 1

        is_last_step = step_in_epoch == steps_per_epoch - 1
        if state.step % LOG_INTERVAL_STEPS == 0 or is_last_step:
            elapsed = time.perf_counter() - interval_start_time
            steps_per_second = (state.step - interval_start_step) / max(elapsed, 1e-6)
            mean_loss = loss_sum.item() / max(loss_count, 1)
            peak_memory_gigabytes = torch.cuda.max_memory_allocated() / 1024**3
            summary_writer.add_scalar("train/loss", mean_loss, state.step)
            summary_writer.add_scalar("train/learning_rate", learning_rate, state.step)
            summary_writer.add_scalar("train/steps_per_second", steps_per_second, state.step)
            print(
                f"run_small_distill: epoch {state.epoch + 1} step {state.step} "
                f"({step_in_epoch + 1}/{steps_per_epoch}) loss {mean_loss:.4f} "
                f"lr {learning_rate:.2e} {steps_per_second:.2f} steps/s "
                f"peak {peak_memory_gigabytes:.1f} GB",
                flush=True,
            )
            loss_sum.zero_()
            loss_count = 0
            interval_start_time = time.perf_counter()
            interval_start_step = state.step
    return epoch_loss_sum.item() / max(epoch_loss_count, 1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--pairs-directory", type=Path, default=SFT_PAIRS_DIRECTORY)
    parser.add_argument(
        "--architecture",
        choices=sorted(ARCHITECTURE_MODULES),
        default="small",
        help="small: distill_small UNet, flow: random init FlowUpscaler UNet",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=0, help="stop after this many epochs in total"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"{SMOKE_STEPS_PER_EPOCH} steps per epoch and a smoke benchmark without results",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = COMPILE_CACHE_SIZE_LIMIT

    paths = run_paths(arguments.run_name)
    paths.create()

    shard_paths = list_bucket_pair_shards(arguments.pairs_directory)
    if not shard_paths:
        raise FileNotFoundError(f"no pair shards in {arguments.pairs_directory}")
    pair_count = sum(shard_pair_count(path) for path in shard_paths)
    micro_batches_per_epoch = sum(
        math.ceil(shard_pair_count(path) / MICRO_BATCH_SIZE) for path in shard_paths
    )
    steps_per_epoch = micro_batches_per_epoch // (BATCH_SIZE // MICRO_BATCH_SIZE)
    if arguments.smoke:
        steps_per_epoch = min(steps_per_epoch, SMOKE_STEPS_PER_EPOCH)

    model = build_model(arguments.architecture).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    ema_parameters = [parameter.detach().float().clone() for parameter in model.parameters()]
    state = TrainingState()
    checkpoint_path = latest_checkpoint(paths.checkpoints)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema_parameters = [parameter.to(DEVICE) for parameter in checkpoint["ema"]]
        state.load_dict(checkpoint["state"])
        del checkpoint
        print(
            f"run_small_distill: resumed from {checkpoint_path.name} at step {state.step}",
            flush=True,
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"run_small_distill: {parameter_count / 1e6:.1f}M parameters, {pair_count} pairs, "
        f"{steps_per_epoch} steps of {BATCH_SIZE} samples per epoch",
        flush=True,
    )
    compiled_model = torch.compile(model, dynamic=False)
    summary_writer = SummaryWriter(str(paths.tensorboard / "small_distill"))

    last_metrics_path = (
        RAW_RESULTS_DIRECTORY / f"{arguments.run_name}_epoch_{state.epoch:03d}" / "metrics.json"
    )
    if state.epoch > 0 and not arguments.smoke and not last_metrics_path.exists():
        metrics = run_epoch_benchmark(
            paths, arguments.run_name, state.epoch, arguments.smoke, arguments.architecture
        )
        log_benchmark(summary_writer, metrics, state)

    while not state.decay_finished() and not paths.stop_request.exists():
        if arguments.max_epochs > 0 and state.epoch >= arguments.max_epochs:
            break
        epoch_start_time = time.perf_counter()
        epoch_loss = run_epoch(
            compiled_model,
            optimizer,
            ema_parameters,
            state,
            shard_paths,
            steps_per_epoch,
            summary_writer,
        )
        state.epoch += 1
        state.training_seconds += time.perf_counter() - epoch_start_time
        summary_writer.add_scalar("epoch/train_loss", epoch_loss, state.step)
        summary_writer.add_scalar("epoch/training_hours", state.training_seconds / 3600, state.step)
        print(
            f"run_small_distill: epoch {state.epoch} finished, mean loss {epoch_loss:.4f}",
            flush=True,
        )

        save_checkpoint(paths.checkpoints, model, optimizer, ema_parameters, state)
        export_weights(paths, model, ema_parameters)
        metrics = run_epoch_benchmark(
            paths, arguments.run_name, state.epoch, arguments.smoke, arguments.architecture
        )
        if metrics is not None:
            log_benchmark(summary_writer, metrics, state)

        if read_control(paths, state, steps_per_epoch):
            save_checkpoint(paths.checkpoints, model, optimizer, ema_parameters, state)
            print(
                f"run_small_distill: decay over {state.decay_steps} steps from step {state.step}",
                flush=True,
            )

    if paths.stop_request.exists():
        paths.stop_request.unlink()
        print("run_small_distill: stopped on request", flush=True)
    elif state.decay_finished():
        paths.training_done.write_text(json.dumps(state.to_dict()))
        print(f"run_small_distill: finished at step {state.step}", flush=True)
    summary_writer.close()


if __name__ == "__main__":
    main()
