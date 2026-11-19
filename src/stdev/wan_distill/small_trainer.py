"""
Single GPU trainer of a small single step student (stdev.wan_distill.small_config.STUDENTS), from scratch on the
online pairs of the large student (stdev.wan_distill.online_pairs). Same objective as the large distillation: sigma 1,
input = noise_scale * seeded noise, target velocity = input - residual(large student latents). Every step trains one
pair type (drawn with the pretraining micro-step probabilities) with STEP_BATCH_MULTIPLIER x the pretraining micro
batch, split into the largest micro-batches that fit (probed at start). Warmup, then constant LR; `control/decay`
in the run directory starts a linear decay to zero (content: steps, default DEFAULT_DECAY_FRACTION of the steps done),
the run ends when it is done; `control/stop` (or SIGTERM) checkpoints and exits, a restart resumes. Every
BENCHMARK_INTERVAL_SECONDS of training the 1000-image single step benchmark (EMA weights) runs on the same
GPU, then a checkpoint is saved and uploaded to R2.

Usage: CUDA_VISIBLE_DEVICES=0 python -m stdev.wan_distill.small_trainer --student s20m [--data-root DIR] [--no-upload]
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import random
import signal
import threading
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from stdev.pretrain.config import PretrainPaths
from stdev.wan_distill.config import DEFAULT_DATA_ROOT, EVALUATION_DATA_RUN, FULL_VAE_DIRECTORY, RUNS_DIRECTORY
from stdev.wan_distill.evaluation import DistributedEvaluation
from stdev.wan_distill.online_pairs import OnlinePairDataset, list_online_shards, seeded_noise
from stdev.wan_distill.small_config import (
    ADAM_BETAS,
    BENCHMARK_INTERVAL_SECONDS,
    CONTROL_CHECK_INTERVAL_STEPS,
    DATA_WORKERS_PER_TYPE,
    DEFAULT_DECAY_FRACTION,
    EMA_DECAY,
    GRADIENT_CLIP_NORM,
    KEPT_CHECKPOINTS,
    KEPT_UPLOADS,
    LOG_INTERVAL_STEPS,
    ONLINE_PAIRS_DIRECTORY,
    RUN_PREFIX,
    STEP_BATCH_MULTIPLIER,
    STUDENT_BY_NAME,
    WARMUP_MIN_SHARDS_PER_TYPE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    SmallStudent,
)
from stdev.wan_distill.student import WanStudentTransformer
from stdev.wan_distill.trainer import start_upload
from stdev.wan_pretrain.config import PAIR_TYPES, TOKENS_PER_UNIT, PairType, micro_step_probabilities
from stdev.wan_pretrain.model import to_residual

TRAINING_DTYPE = torch.bfloat16
SEED = 0
EVALUATION_VARIANTS = (("ema_cfg1", "ema"),)
DYNAMO_CACHE_SIZE_LIMIT = 64
MEMORY_LIMIT_FRACTION = 0.8


class TrainingState:
    def __init__(self) -> None:
        self.step = 0
        self.samples_seen = 0
        self.units_seen = 0.0
        self.training_seconds = 0.0
        self.next_benchmark_seconds = 0.0
        self.decay_start_step = -1
        self.decay_steps = 0
        self.ema_updates = 0

    def learning_rate(self, base_learning_rate: float) -> float:
        learning_rate = base_learning_rate * min(1.0, (self.step + 1) / WARMUP_STEPS)
        if self.decay_start_step < 0:
            return learning_rate
        return learning_rate * max(0.0, 1.0 - (self.step - self.decay_start_step) / max(self.decay_steps, 1))

    def decay_finished(self) -> bool:
        return self.decay_start_step >= 0 and self.step >= self.decay_start_step + self.decay_steps


def step_batch_size(pair_type: PairType) -> int:
    return STEP_BATCH_MULTIPLIER * pair_type.micro_batch_size


def choose_pair_type(step: int) -> PairType:
    probabilities = micro_step_probabilities()
    generator = random.Random(f"{SEED}-{step}")
    return generator.choices(PAIR_TYPES, weights=[probabilities[pair_type.name] for pair_type in PAIR_TYPES])[0]


def training_loss(model: WanStudentTransformer, batch: dict[str, torch.Tensor], pair_type: PairType, device: torch.device) -> torch.Tensor:
    """
    returns:
        sample_losses: [B], velocity MSE
    """
    noise = batch["noise_scale"][:, None, None, None, None] * seeded_noise(batch["noise_seed"], (16, *pair_type.target_latent_size), device)
    residual = to_residual(batch["latents_generated"].float(), batch["latents_source"].float())
    hff_mask = ~torch.isnan(batch["hff"])
    with torch.autocast("cuda", dtype=TRAINING_DTYPE):
        velocity = model(noise, batch["latents_source"], torch.nan_to_num(batch["hff"], nan=0.0), hff_mask, batch["guidance"])
    return (velocity.float() - (noise - residual)).square().mean(dim=(1, 2, 3, 4))


def probe_micro_batch_sizes(model: WanStudentTransformer, device: torch.device) -> dict[str, int]:
    """
    Largest micro-batch (step batch / 2^k) per pair type whose forward and backward stay below
    MEMORY_LIMIT_FRACTION of the GPU memory; compiles every used shape.
    """
    limit = MEMORY_LIMIT_FRACTION * torch.cuda.get_device_properties(device).total_memory
    sizes: dict[str, int] = {}
    for pair_type in PAIR_TYPES:
        size = step_batch_size(pair_type)
        while True:
            batch = {
                "latents_source": torch.randn(size, 16, *pair_type.source_latent_size, device=device).half(),
                "latents_generated": torch.randn(size, 16, *pair_type.target_latent_size, device=device).half(),
                "hff": torch.rand(size, device=device),
                "guidance": torch.ones(size, device=device),
                "noise_scale": torch.ones(size, device=device),
                "noise_seed": torch.arange(size, dtype=torch.int64),
            }
            torch.cuda.reset_peak_memory_stats()
            try:
                training_loss(model, batch, pair_type, device).mean().backward()
                fits = torch.cuda.max_memory_allocated() < limit
            except torch.OutOfMemoryError:
                fits = False
            model.zero_grad(set_to_none=True)
            del batch
            torch.cuda.empty_cache()
            if fits or size == 1:
                sizes[pair_type.name] = size
                break
            size //= 2
    return sizes


def save_checkpoint(paths: PretrainPaths, student: SmallStudent, model: torch.nn.Module, optimizer: torch.optim.Optimizer, ema: list[torch.Tensor], state: TrainingState) -> Path:
    path = paths.checkpoints / f"checkpoint_{state.step:08d}.pt"
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": [parameter.cpu() for parameter in ema],
            "optimizer": optimizer.state_dict(),
            "state": dict(state.__dict__),
            "model_config": student.config.__dict__,
            "architecture": "WanStudentTransformer",
            "student": student.name,
        },
        temporary_path,
    )
    temporary_path.replace(path)
    for old_path in sorted(paths.checkpoints.glob("checkpoint_*.pt"))[:-KEPT_CHECKPOINTS]:
        old_path.unlink()
    return path


def make_inference_copy(model: WanStudentTransformer, weights: list[torch.Tensor] | None) -> WanStudentTransformer:
    inference_model = WanStudentTransformer(model.config).to(next(model.parameters()).device).eval().requires_grad_(False)
    with torch.no_grad():
        source = weights if weights is not None else [parameter.detach() for parameter in model.parameters()]
        for parameter, weight in zip(inference_model.parameters(), source, strict=True):
            parameter.copy_(weight)
    return inference_model.to(torch.bfloat16)


def run_benchmark(
    evaluation: DistributedEvaluation, model: WanStudentTransformer, ema: list[torch.Tensor], state: TrainingState, paths: PretrainPaths, summary_writer: SummaryWriter
) -> None:
    start_time = time.perf_counter()
    tag = f"step_{state.step:08d}"
    output_directory = paths.benchmarks / tag
    output_directory.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {"tag": tag, "step": state.step, "samples_seen": state.samples_seen, "training_hours": state.training_seconds / 3600}
    for variant_name, weights in EVALUATION_VARIANTS:
        inference_model = make_inference_copy(model, ema if weights == "ema" else None)
        metrics = evaluation.evaluate(inference_model, 1.0, output_directory, variant_name)
        del inference_model
        for name, value in metrics.items():
            results[f"{variant_name}_{name}"] = value
            summary_writer.add_scalar(f"benchmark_{variant_name}/{name}", value, state.step)
    torch.cuda.empty_cache()
    results["benchmark_seconds"] = time.perf_counter() - start_time
    summary_writer.add_scalar("benchmark/training_hours", state.training_seconds / 3600, state.step)
    summary_writer.flush()
    with (paths.benchmarks / "results.jsonl").open("a") as results_file:
        results_file.write(json.dumps(results) + "\n")
    print(f"benchmark: {json.dumps(results)}", flush=True)


def wait_for_pairs(pairs_directory: Path) -> None:
    while not all(len(list_online_shards(pairs_directory / pair_type.name)) >= WARMUP_MIN_SHARDS_PER_TYPE for pair_type in PAIR_TYPES):
        time.sleep(30.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True, choices=sorted(STUDENT_BY_NAME))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--learning-rate", type=float, default=0.0, help="0 = the student's default")
    parser.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps of this session")
    parser.add_argument("--evaluation-count", type=int, default=0, help="smoke test: benchmark only the first N images")
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--no-upload", action="store_true")
    arguments = parser.parse_args()
    student = STUDENT_BY_NAME[arguments.student]
    base_learning_rate = arguments.learning_rate or student.learning_rate
    run_name = f"{RUN_PREFIX}{student.name}{arguments.run_suffix}"
    data_root = arguments.data_root
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = DYNAMO_CACHE_SIZE_LIMIT

    paths = PretrainPaths(data_root / RUNS_DIRECTORY / run_name)
    paths.create()
    paths.benchmarks.mkdir(parents=True, exist_ok=True)
    model = WanStudentTransformer(student.config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_learning_rate, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY, fused=True)
    state = TrainingState()
    checkpoints = sorted(paths.checkpoints.glob("checkpoint_*.pt"))
    if checkpoints:
        checkpoint = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        state.__dict__.update(checkpoint["state"])
        ema = [parameter.to(device) for parameter in checkpoint["ema"]]
        print(f"trainer: resumed from {checkpoints[-1].name} at step {state.step}", flush=True)
        del checkpoint
    else:
        ema = [parameter.detach().float().clone() for parameter in model.parameters()]
    parameters = list(model.parameters())
    model.compile_blocks()
    model.train()
    probe_start = time.perf_counter()
    micro_batch_sizes = probe_micro_batch_sizes(model, device)
    parameter_count = sum(parameter.numel() for parameter in parameters)
    print(
        f"trainer: {run_name} {parameter_count / 1e6:.2f}M parameters, config {json.dumps(student.config.__dict__)}, lr {base_learning_rate:.2e}, "
        f"micro-batches {json.dumps(micro_batch_sizes)} (step batch x{STEP_BATCH_MULTIPLIER} of pretraining), probed in {time.perf_counter() - probe_start:.0f}s",
        flush=True,
    )

    pairs_directory = data_root / ONLINE_PAIRS_DIRECTORY
    wait_for_pairs(pairs_directory)
    probabilities = micro_step_probabilities()
    loaders = {
        pair_type.name: iter(
            DataLoader(
                OnlinePairDataset(pairs_directory / pair_type.name, seed=f"{run_name}-{pair_type.name}"),
                batch_size=micro_batch_sizes[pair_type.name],
                num_workers=DATA_WORKERS_PER_TYPE if probabilities[pair_type.name] > 0.05 else 1,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
            )
        )
        for pair_type in PAIR_TYPES
    }
    evaluation = DistributedEvaluation(PretrainPaths(data_root / EVALUATION_DATA_RUN), data_root / FULL_VAE_DIRECTORY, 0, 1, arguments.evaluation_count)
    summary_writer = SummaryWriter(str(paths.tensorboard / "train"))
    summary_writer.add_text("config", json.dumps({"student": student.name, "config": student.config.__dict__, "learning_rate": base_learning_rate, "micro_batches": micro_batch_sizes}), state.step)
    if state.step == 0 and state.next_benchmark_seconds == 0.0:
        run_benchmark(evaluation, model, ema, state, paths, summary_writer)
        state.next_benchmark_seconds = BENCHMARK_INTERVAL_SECONDS

    termination_requested = False

    def request_termination(signal_number: int, frame: object) -> None:
        nonlocal termination_requested
        termination_requested = True

    signal.signal(signal.SIGTERM, request_termination)

    type_names = [pair_type.name for pair_type in PAIR_TYPES]
    statistics = torch.zeros(2 * len(PAIR_TYPES), device=device)
    gradient_norm_sum = torch.zeros((), device=device)
    gradient_norm_max = torch.zeros((), device=device)
    interval_start = time.perf_counter()
    interval_samples = 0
    interval_steps = 0
    steps_this_session = 0
    upload_thread: threading.Thread | None = None
    env_path = data_root / ".env"

    while True:
        step_start = time.perf_counter()
        learning_rate = state.learning_rate(base_learning_rate)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        pair_type = choose_pair_type(state.step)
        micro_size = micro_batch_sizes[pair_type.name]
        accumulation = step_batch_size(pair_type) // micro_size
        step_loss = torch.zeros((), device=device)
        for _ in range(accumulation):
            batch = {name: tensor.to(device, non_blocking=True) for name, tensor in next(loaders[pair_type.name]).items()}
            loss = training_loss(model, batch, pair_type, device).mean()
            (loss / accumulation).backward()
            step_loss += loss.detach() / accumulation
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state.ema_updates += 1
        decay = min(EMA_DECAY, (1.0 + state.ema_updates) / (10.0 + state.ema_updates))
        torch._foreach_lerp_(ema, [parameter.detach() for parameter in parameters], 1.0 - decay)
        type_index = type_names.index(pair_type.name)
        statistics[2 * type_index] += step_loss
        statistics[2 * type_index + 1] += 1
        gradient_norm_sum += gradient_norm.detach()
        gradient_norm_max = torch.maximum(gradient_norm_max, gradient_norm.detach())

        state.step += 1
        step_samples = step_batch_size(pair_type)
        state.samples_seen += step_samples
        state.units_seen += step_samples * pair_type.tokens_per_sample / TOKENS_PER_UNIT
        steps_this_session += 1
        interval_samples += step_samples
        interval_steps += 1
        if steps_this_session > 1:
            state.training_seconds += time.perf_counter() - step_start

        stop_requested = termination_requested
        if state.step % CONTROL_CHECK_INTERVAL_STEPS == 0:
            if state.decay_start_step < 0 and paths.decay_request.exists():
                content = paths.decay_request.read_text().strip()
                state.decay_steps = int(content) if content else max(1, round(DEFAULT_DECAY_FRACTION * state.step))
                state.decay_start_step = state.step
                paths.decay_request.rename(paths.control / "decay.accepted")
                print(f"trainer: decay over {state.decay_steps} steps from step {state.step}", flush=True)
            stop_requested = stop_requested or paths.stop_request.exists()

        if state.step % LOG_INTERVAL_STEPS == 0:
            elapsed = time.perf_counter() - interval_start
            losses = statistics.view(-1, 2).tolist()
            total_loss = sum(loss_sum for loss_sum, _ in losses) / max(1.0, sum(count for _, count in losses))
            scalars = {
                "train/loss": total_loss,
                "train/learning_rate": learning_rate,
                "train/samples_per_second": interval_samples / max(elapsed, 1e-6),
                "train/step_seconds": elapsed / max(interval_steps, 1),
                "train/grad_norm_mean": gradient_norm_sum.item() / LOG_INTERVAL_STEPS,
                "train/grad_norm_max": gradient_norm_max.item(),
                "train/training_hours": state.training_seconds / 3600,
                "train/samples_seen": state.samples_seen,
                "system/gpu_memory_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            }
            for name, (loss_sum, count) in zip(type_names, losses):
                if count > 0:
                    scalars[f"loss_by_type/{name}"] = loss_sum / count
            for name, value in scalars.items():
                summary_writer.add_scalar(name, value, state.step)
            print(
                f"trainer: step {state.step} loss {total_loss:.4f} lr {learning_rate:.2e} grad {scalars['train/grad_norm_mean']:.3f}/"
                f"{scalars['train/grad_norm_max']:.3f} {scalars['train/samples_per_second']:.1f} samples/s, {state.training_seconds / 3600:.2f} h",
                flush=True,
            )
            statistics.zero_()
            gradient_norm_sum.zero_()
            gradient_norm_max.zero_()
            interval_samples = 0
            interval_steps = 0
            torch.cuda.reset_peak_memory_stats()
            interval_start = time.perf_counter()

        finished = state.decay_finished()
        smoke_finished = arguments.max_steps > 0 and steps_this_session >= arguments.max_steps
        if finished or state.training_seconds >= state.next_benchmark_seconds:
            run_benchmark(evaluation, model, ema, state, paths, summary_writer)
            state.next_benchmark_seconds = state.training_seconds + BENCHMARK_INTERVAL_SECONDS
            checkpoint_path = save_checkpoint(paths, student, model, optimizer, ema, state)
            if not arguments.no_upload and (upload_thread is None or not upload_thread.is_alive()):
                upload_thread = start_upload(checkpoint_path, run_name, env_path)
            interval_start = time.perf_counter()
        if finished:
            paths.training_done.write_text(json.dumps(state.__dict__))
            print(f"trainer: finished at step {state.step}", flush=True)
            break
        if stop_requested or smoke_finished:
            save_checkpoint(paths, student, model, optimizer, ema, state)
            paths.stop_request.unlink(missing_ok=True)
            print(f"trainer: stopped at step {state.step}", flush=True)
            break

    summary_writer.close()
    if upload_thread is not None:
        upload_thread.join()


if __name__ == "__main__":
    main()
