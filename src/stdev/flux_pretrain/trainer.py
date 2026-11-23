"""
DDP trainer of the FLUX.2 pretraining. Every micro-step all ranks train on the same pair type (drawn from a step
seeded generator by micro_step_probabilities), so micro-steps take equal time on all ranks. The schedule is fixed
in training time (TRAINING_HOURS): LR warmup over WARMUP_UNITS, gradient accumulation ramp by ACCUMULATION_SCHEDULE
(fractions of the training time) with square root LR scaling, linear LR decay to zero over the last DECAY_FRACTION
of the time, end at TRAINING_HOURS. `control/stop` (or SIGTERM) saves a checkpoint and exits; a restart resumes
from the latest checkpoint. Rank 0 keeps an EMA, saves benchmark snapshots every BENCHMARK_INTERVAL_SECONDS and
checkpoints, and uploads a checkpoint to R2 every UPLOAD_INTERVAL_SECONDS.

Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc-per-node 7 -m stdev.flux_pretrain.trainer
           --run-name NAME --data-run-name DATA [--learning-rate LR] [--training-hours H] [--max-steps N]
           [--no-upload] [--benchmark-interval-seconds S]
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")

import argparse
import faulthandler
import json
import math
import random
import signal
import threading
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as distributed
from safetensors.torch import save_file
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from stdev.flux_pretrain.cache import PairTypeDataset, cache_size_bytes
from stdev.flux_pretrain.config import (
    ACCUMULATION_SCHEDULE,
    ADAM_BETAS,
    BASE_LEARNING_RATE,
    BENCHMARK_INTERVAL_SECONDS,
    CHECKPOINT_INTERVAL_SECONDS,
    CONTROL_CHECK_INTERVAL_STEPS,
    DATA_WORKERS_PER_TYPE,
    DATASETS,
    DECAY_FRACTION,
    DYNAMO_CACHE_SIZE_LIMIT,
    EMA_DECAY,
    ENV_PATH,
    GRADIENT_CLIP_NORM,
    HFF_DROPOUT_PROBABILITY,
    KEPT_CHECKPOINTS,
    KEPT_UPLOADS,
    LOG_INTERVAL_STEPS,
    MODEL,
    PAIR_TYPES,
    SEED,
    SOURCE_VARIANTS,
    TOKENS_PER_UNIT,
    TRAINING_HOURS,
    TRAINING_SIGMA_SHIFT,
    UPLOAD_BUCKET,
    UPLOAD_INTERVAL_SECONDS,
    WARMUP_UNITS,
    WEIGHT_DECAY,
    PairType,
    micro_step_probabilities,
    run_paths,
)
from stdev.flux_pretrain.model import LATENT_CHANNELS, FluxUpscalerTransformer, shift_sigmas, to_residual
from stdev.pretrain.config import PretrainPaths

TRAINING_DTYPE = torch.bfloat16
SIGMA_BIN_EDGES = (0.25, 0.5, 0.75)
GRADIENT_GROUPS = ("conditioning", "input_conv", "source_encoder", "blocks_early", "blocks_middle", "blocks_late", "output")


class TrainingState:
    """
    Resumable counters. The schedule is a function of the training time as synchronized across ranks at the
    last control check (schedule_seconds), so all ranks agree on accumulation, LR and the end.
    """

    def __init__(self) -> None:
        self.step = 0
        self.units_seen = 0.0
        self.samples_seen = 0
        self.training_seconds = 0.0
        self.next_benchmark_seconds = BENCHMARK_INTERVAL_SECONDS
        self.next_checkpoint_seconds = CHECKPOINT_INTERVAL_SECONDS
        self.next_upload_seconds = UPLOAD_INTERVAL_SECONDS
        self.ema_updates = 0

    def accumulation_factor(self, schedule_seconds: float, total_seconds: float) -> int:
        progress = schedule_seconds / total_seconds
        return [factor for fraction, factor in ACCUMULATION_SCHEDULE if progress >= fraction][-1]

    def learning_rate(self, base_learning_rate: float, schedule_seconds: float, total_seconds: float) -> float:
        warmup_factor = min(1.0, max(self.units_seen, 1.0) / WARMUP_UNITS)
        remaining_fraction = max(0.0, 1.0 - schedule_seconds / total_seconds)
        decay_factor = min(1.0, remaining_fraction / DECAY_FRACTION)
        accumulation = self.accumulation_factor(schedule_seconds, total_seconds)
        return base_learning_rate * math.sqrt(accumulation) * warmup_factor * decay_factor

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def load_dict(self, values: dict) -> None:
        self.__dict__.update(values)


def sample_training_sigmas(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    returns:
        sigmas: [B], values: [0...1], logit-normal distributed and shifted
    """
    return shift_sigmas(torch.sigmoid(torch.randn(batch_size, device=device)), TRAINING_SIGMA_SHIFT)


def choose_pair_type(step: int, micro_step: int) -> PairType:
    probabilities = micro_step_probabilities()
    generator = random.Random(f"{SEED}-{step}-{micro_step}")
    return generator.choices(PAIR_TYPES, weights=[probabilities[pair_type.name] for pair_type in PAIR_TYPES])[0]


def gradient_group(name: str, depth: int) -> str:
    if name.startswith("blocks."):
        index = int(name.split(".")[1])
        return ("blocks_early", "blocks_middle", "blocks_late")[min(2, 3 * index // depth)]
    if name.startswith(("output", "output_modulation", "output_norm")):
        return "output"
    return name.split(".")[0]


def latest_checkpoint(paths: PretrainPaths) -> Path | None:
    checkpoints = sorted(paths.checkpoints.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(
    paths: PretrainPaths,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_parameters: list[torch.Tensor],
    state: TrainingState,
) -> Path:
    """
    Full resumable checkpoint; keeps the newest KEPT_CHECKPOINTS (files linked for upload stay until uploaded).
    """
    path = paths.checkpoints / f"checkpoint_{state.step:09d}.pt"
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": [parameter.cpu() for parameter in ema_parameters],
            "optimizer": optimizer.state_dict(),
            "state": state.to_dict(),
            "model_config": MODEL.__dict__,
        },
        temporary_path,
    )
    temporary_path.replace(path)
    for old_path in sorted(paths.checkpoints.glob("checkpoint_*.pt"))[:-KEPT_CHECKPOINTS]:
        old_path.unlink()
    print(f"trainer: checkpoint {path.name}", flush=True)
    return path


def upload_checkpoint(local_path: Path, run_name: str) -> None:
    """
    Uploads a (hard linked) checkpoint to R2 and deletes the older remote ones beyond KEPT_UPLOADS.
    Runs in a background thread; failures are logged, never raised.
    """
    try:
        from stdev.video_downloader.stream_videos import load_env_file

        load_env_file(ENV_PATH)
        from stdev.s3_tools.client import get_s3_client

        client = get_s3_client()
        prefix = f"{run_name}/"
        client.upload_file(str(local_path), UPLOAD_BUCKET, prefix + local_path.name)
        listed = client.list_objects_v2(Bucket=UPLOAD_BUCKET, Prefix=prefix).get("Contents", [])
        remote_checkpoints = sorted(item["Key"] for item in listed if item["Key"].endswith(".pt"))
        for key in remote_checkpoints[:-KEPT_UPLOADS]:
            client.delete_object(Bucket=UPLOAD_BUCKET, Key=key)
        print(f"trainer: uploaded {local_path.name} to s3://{UPLOAD_BUCKET}/{prefix}", flush=True)
    except Exception as error:
        print(f"trainer: upload of {local_path.name} failed: {error!r}", flush=True)
    finally:
        local_path.unlink(missing_ok=True)


def start_upload(checkpoint_path: Path, paths: PretrainPaths, run_name: str) -> threading.Thread:
    upload_directory = paths.checkpoints / "upload"
    upload_directory.mkdir(exist_ok=True)
    linked_path = upload_directory / checkpoint_path.name
    linked_path.unlink(missing_ok=True)
    os.link(checkpoint_path, linked_path)
    thread = threading.Thread(target=upload_checkpoint, args=(linked_path, run_name), daemon=True)
    thread.start()
    return thread


def save_snapshot(
    paths: PretrainPaths, model: torch.nn.Module, ema_parameters: list[torch.Tensor], state: TrainingState, tag: str
) -> None:
    """
    Raw and EMA weights in bf16 for the benchmark process, with a json sidecar.
    """
    tensors: dict[str, torch.Tensor] = {}
    for (name, parameter), ema_parameter in zip(model.named_parameters(), ema_parameters):
        tensors[f"raw.{name}"] = parameter.detach().to(torch.bfloat16).cpu().contiguous()
        tensors[f"ema.{name}"] = ema_parameter.to(torch.bfloat16).cpu().contiguous()
    stem = f"{tag}_{state.step:09d}"
    temporary_path = paths.snapshots / f".{stem}.tmp"
    save_file(tensors, str(temporary_path))
    (paths.snapshots / f"{stem}.json").write_text(
        json.dumps(
            {
                "step": state.step,
                "units_seen": state.units_seen,
                "samples_seen": state.samples_seen,
                "training_seconds": state.training_seconds,
                "tag": tag,
            }
        )
    )
    temporary_path.replace(paths.snapshots / f"{stem}.safetensors")
    print(f"trainer: snapshot {stem}", flush=True)


def read_producer_statistics(data_paths: PretrainPaths) -> dict:
    try:
        return json.loads(data_paths.producer_statistics.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def make_data_loaders(data_paths: PretrainPaths, rank: int) -> dict[str, DataLoader]:
    probabilities = micro_step_probabilities()
    loaders: dict[str, DataLoader] = {}
    for pair_type in PAIR_TYPES:
        workers = DATA_WORKERS_PER_TYPE if probabilities[pair_type.name] > 0.05 else 1
        loaders[pair_type.name] = DataLoader(
            PairTypeDataset(data_paths.cache / pair_type.name, seed=f"{SEED}-{rank}-{pair_type.name}"),
            batch_size=pair_type.micro_batch_size,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
    return loaders


def warm_compile(model: FluxUpscalerTransformer, distributed_model: DistributedDataParallel, device: torch.device) -> None:
    """
    Compiles forward and backward of every pair type shape before the clock starts (no gradient sync).
    """
    for pair_type in PAIR_TYPES:
        batch_size = pair_type.micro_batch_size
        model.checkpointed_blocks = pair_type.checkpointed_blocks
        with distributed_model.no_sync():
            with torch.autocast("cuda", dtype=TRAINING_DTYPE):
                velocity = distributed_model(
                    torch.randn(batch_size, LATENT_CHANNELS, *pair_type.target_latent_size, device=device),
                    torch.rand(batch_size, device=device),
                    torch.randn(batch_size, LATENT_CHANNELS, *pair_type.source_latent_size, device=device),
                    torch.rand(batch_size, device=device),
                    torch.ones(batch_size, dtype=torch.bool, device=device),
                )
            velocity.float().square().mean().backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-run-name", required=True)
    parser.add_argument("--learning-rate", type=float, default=BASE_LEARNING_RATE)
    parser.add_argument("--training-hours", type=float, default=TRAINING_HOURS)
    parser.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps of this session")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--benchmark-interval-seconds", type=float, default=BENCHMARK_INTERVAL_SECONDS)
    arguments = parser.parse_args()
    total_seconds = arguments.training_hours * 3600.0

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed.init_process_group("nccl", timeout=timedelta(minutes=15), device_id=device)
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    is_main = rank == 0
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = DYNAMO_CACHE_SIZE_LIMIT

    paths = run_paths(arguments.run_name)
    data_paths = run_paths(arguments.data_run_name)
    if is_main:
        paths.create()
    distributed.barrier()

    torch.manual_seed(SEED)
    model = FluxUpscalerTransformer(MODEL).to(device)
    torch.manual_seed(SEED * 1000 + rank + int(time.time()) % 100000)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=arguments.learning_rate, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY, fused=True
    )
    state = TrainingState()
    state.next_benchmark_seconds = arguments.benchmark_interval_seconds
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
    if is_main and not ema_parameters:
        ema_parameters = [parameter.detach().float().clone() for parameter in model.parameters()]

    model.compile_blocks()
    distributed_model = DistributedDataParallel(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    distributed_model.register_comm_hook(None, default_hooks.bf16_compress_hook)
    parameters = list(model.parameters())
    parameter_names = [name for name, _ in model.named_parameters()]
    parameter_groups = [GRADIENT_GROUPS.index(gradient_group(name, MODEL.depth)) for name in parameter_names]

    compile_start = time.perf_counter()
    warm_compile(model, distributed_model, device)
    if is_main:
        parameter_count = sum(parameter.numel() for parameter in parameters)
        print(
            f"trainer: {parameter_count / 1e6:.1f}M parameters, {world_size} ranks, compiled in "
            f"{time.perf_counter() - compile_start:.0f}s, lr {arguments.learning_rate:.2e}, "
            f"{arguments.training_hours} h schedule",
            flush=True,
        )
    loaders = make_data_loaders(data_paths, rank)
    iterators = {name: iter(loader) for name, loader in loaders.items()}
    summary_writer = SummaryWriter(str(paths.tensorboard / "train")) if is_main else None
    if is_main:
        summary_writer.add_text("config", json.dumps({**vars(arguments), "model": MODEL.__dict__}), state.step)

    termination_requested = False

    def request_termination(signal_number: int, frame: object) -> None:
        nonlocal termination_requested
        termination_requested = True

    signal.signal(signal.SIGTERM, request_termination)

    type_names = [pair_type.name for pair_type in PAIR_TYPES]
    sigma_bins = len(SIGMA_BIN_EDGES) + 1
    type_offset = 0
    sigma_offset = type_offset + 2 * len(PAIR_TYPES)
    variant_offset = sigma_offset + 2 * sigma_bins
    dataset_offset = variant_offset + 2 * len(SOURCE_VARIANTS)
    statistics = torch.zeros(dataset_offset + 2 * len(DATASETS), device=device)
    sigma_edges = torch.tensor(SIGMA_BIN_EDGES, device=device)
    gradient_norm_sum = torch.zeros((), device=device)
    gradient_norm_max = torch.zeros((), device=device)
    group_norm_sums = torch.zeros(len(GRADIENT_GROUPS), device=device)
    parameter_group_indices = torch.tensor(parameter_groups, device=device)
    interval_steps = 0
    interval_units = 0.0
    interval_samples = 0
    interval_start_time = time.perf_counter()
    step_start_time = time.perf_counter()
    upload_thread: threading.Thread | None = None
    steps_this_session = 0
    torch.cuda.reset_peak_memory_stats()
    schedule_seconds = state.training_seconds
    is_finished = False

    def add_grouped(offset: int, count: int, indices: torch.Tensor, values: torch.Tensor) -> None:
        statistics[offset : offset + count] += torch.zeros(count, device=device).index_add_(0, indices, values)
        statistics[offset + count : offset + 2 * count] += torch.bincount(indices, minlength=count).float()

    while True:
        accumulation = state.accumulation_factor(schedule_seconds, total_seconds)
        learning_rate = state.learning_rate(arguments.learning_rate, schedule_seconds, total_seconds)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        step_units = 0.0
        step_samples = 0
        for micro_step in range(accumulation):
            pair_type = choose_pair_type(state.step, micro_step)
            batch = next(iterators[pair_type.name])
            latents_source = batch["latents_source"].to(device, non_blocking=True)
            latents_target = batch["latents_target"].to(device, non_blocking=True)
            hff = batch["hff"].to(device, non_blocking=True)
            variant_index = batch["variant_index"].to(device, non_blocking=True)
            dataset_index = batch["dataset_index"].to(device, non_blocking=True)
            batch_size = latents_target.shape[0]
            residual = to_residual(latents_target, latents_source, MODEL.residual_std)
            noise = torch.randn_like(residual)
            sigmas = sample_training_sigmas(batch_size, device)
            blend = sigmas[:, None, None, None]
            noisy_residual = (1.0 - blend) * residual + blend * noise
            hff_mask = torch.rand(batch_size, device=device) >= HFF_DROPOUT_PROBABILITY
            model.checkpointed_blocks = pair_type.checkpointed_blocks
            is_last_micro_step = micro_step == accumulation - 1
            with nullcontext() if is_last_micro_step else distributed_model.no_sync():
                with torch.autocast("cuda", dtype=TRAINING_DTYPE):
                    velocity = distributed_model(noisy_residual, sigmas, latents_source, hff, hff_mask)
                sample_losses = (velocity.float() - (noise - residual)).square().mean(dim=(1, 2, 3))  # [B]
                loss = sample_losses.mean()
                (loss / accumulation).backward()
            detached_losses = sample_losses.detach()
            type_index = type_names.index(pair_type.name)
            statistics[type_offset + 2 * type_index] += detached_losses.mean()
            statistics[type_offset + 2 * type_index + 1] += 1
            add_grouped(sigma_offset, sigma_bins, torch.bucketize(sigmas, sigma_edges), detached_losses)
            if pair_type is PAIR_TYPES[0]:
                add_grouped(variant_offset, len(SOURCE_VARIANTS), variant_index, detached_losses)
                add_grouped(dataset_offset, len(DATASETS), dataset_index, detached_losses)
            step_units += world_size * batch_size * pair_type.tokens_per_sample / TOKENS_PER_UNIT
            step_samples += world_size * batch_size

        is_log_step = (state.step + 1) % LOG_INTERVAL_STEPS == 0
        if is_log_step and is_main:
            gradient_norms = torch.stack(torch._foreach_norm([parameter.grad for parameter in parameters]))
            group_norm_sums.zero_().index_add_(0, parameter_group_indices, gradient_norms.square())
            group_norm_sums.sqrt_()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        gradient_norm_sum += gradient_norm.detach()
        gradient_norm_max = torch.maximum(gradient_norm_max, gradient_norm.detach())

        if is_main:
            state.ema_updates += 1
            decay = min(EMA_DECAY, (1.0 + state.ema_updates) / (10.0 + state.ema_updates))
            torch._foreach_lerp_(ema_parameters, [parameter.detach() for parameter in parameters], 1.0 - decay)

        now = time.perf_counter()
        if steps_this_session > 0:
            state.training_seconds += now - step_start_time
        else:
            interval_start_time = now
        step_start_time = now
        state.step += 1
        state.units_seen += step_units
        state.samples_seen += step_samples
        steps_this_session += 1
        if steps_this_session > 1:
            interval_steps += 1
            interval_units += step_units
            interval_samples += step_samples

        if state.step % CONTROL_CHECK_INTERVAL_STEPS == 0 or termination_requested:
            control = torch.zeros(2, device=device)
            if is_main:
                control[0] = float(paths.stop_request.exists())
                control[1] = state.training_seconds
            control[0] += float(termination_requested)
            distributed.all_reduce(control, op=distributed.ReduceOp.MAX)
            state.training_seconds = control[1].item()
            schedule_seconds = state.training_seconds
            is_finished = schedule_seconds >= total_seconds
            if control[0] > 0:
                if is_main:
                    save_checkpoint(paths, model, optimizer, ema_parameters, state)
                    paths.stop_request.unlink(missing_ok=True)
                    print("trainer: stopped on request", flush=True)
                break

        if is_log_step:
            distributed.all_reduce(statistics)
            if is_main:
                elapsed = time.perf_counter() - interval_start_time

                def grouped_means(offset: int, names: list[str] | tuple[str, ...]) -> dict[str, float]:
                    values = statistics[offset : offset + 2 * len(names)].tolist()
                    sums, counts = values[: len(names)], values[len(names) :]
                    return {name: total / count for name, total, count in zip(names, sums, counts) if count > 0}

                type_losses = statistics[type_offset : type_offset + 2 * len(PAIR_TYPES)].view(-1, 2).tolist()
                total_loss = sum(loss_sum for loss_sum, _ in type_losses) / max(1.0, sum(count for _, count in type_losses))
                producer = read_producer_statistics(data_paths)
                produced_total = producer.get("produced_samples_total", 0)
                scalars = {
                    "train/loss": total_loss,
                    "train/learning_rate": learning_rate,
                    "train/accumulation": accumulation,
                    "train/global_batch_samples": step_samples,
                    "train/global_batch_units": step_units,
                    "train/units_seen": state.units_seen,
                    "train/samples_seen": state.samples_seen,
                    "train/units_per_second": interval_units / max(elapsed, 1e-6),
                    "train/samples_per_second": interval_samples / max(elapsed, 1e-6),
                    "train/step_seconds": elapsed / max(interval_steps, 1),
                    "train/training_hours": state.training_seconds / 3600,
                    "train/grad_norm_mean": gradient_norm_sum.item() / LOG_INTERVAL_STEPS,
                    "train/grad_norm_max": gradient_norm_max.item(),
                    "train/weight_norm": torch.linalg.vector_norm(torch.stack(torch._foreach_norm(parameters))).item(),
                    "system/gpu_memory_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "data/produced_total": produced_total,
                    "data/producer_samples_per_second": producer.get("samples_per_second", 0.0),
                    "data/cache_gb": cache_size_bytes(data_paths.cache) / 2**30,
                }
                if produced_total > 0:
                    scalars["data/mean_reuse"] = state.samples_seen / produced_total
                for name, (loss_sum, count) in zip(type_names, type_losses):
                    if count > 0:
                        scalars[f"loss_by_type/{name}"] = loss_sum / count
                bin_names = [
                    f"{lower:.2f}_{upper:.2f}"
                    for lower, upper in zip((0.0,) + SIGMA_BIN_EDGES, SIGMA_BIN_EDGES + (1.0,))
                ]
                for name, value in grouped_means(sigma_offset, bin_names).items():
                    scalars[f"loss_by_sigma/{name}"] = value
                for name, value in grouped_means(variant_offset, SOURCE_VARIANTS).items():
                    scalars[f"loss_by_variant_{PAIR_TYPES[0].name}/{name}"] = value
                for name, value in grouped_means(dataset_offset, DATASETS).items():
                    scalars[f"loss_by_dataset_{PAIR_TYPES[0].name}/{name}"] = value
                for group_name, group_norm in zip(GRADIENT_GROUPS, group_norm_sums.tolist()):
                    scalars[f"grad_norm_by_group/{group_name}"] = group_norm
                for name, value in scalars.items():
                    summary_writer.add_scalar(name, value, state.step)
                print(
                    f"trainer: step {state.step} loss {total_loss:.4f} lr {learning_rate:.2e} acc {accumulation} "
                    f"grad {scalars['train/grad_norm_mean']:.3f}/{scalars['train/grad_norm_max']:.3f} "
                    f"{scalars['train/units_per_second']:.1f} units/s, units {state.units_seen:.0f}, "
                    f"mem {scalars['system/gpu_memory_peak_gb']:.1f} GB, {state.training_seconds / 3600:.2f} h",
                    flush=True,
                )
                torch.cuda.reset_peak_memory_stats()
            statistics.zero_()
            gradient_norm_sum.zero_()
            gradient_norm_max.zero_()
            interval_steps = 0
            interval_units = 0.0
            interval_samples = 0
            interval_start_time = time.perf_counter()

        is_smoke_finished = arguments.max_steps > 0 and steps_this_session >= arguments.max_steps
        if is_main and not is_finished and state.training_seconds >= state.next_benchmark_seconds:
            save_snapshot(paths, model, ema_parameters, state, "scheduled")
            state.next_benchmark_seconds = state.training_seconds + arguments.benchmark_interval_seconds
        if is_main and not is_finished and state.training_seconds >= state.next_checkpoint_seconds:
            checkpoint_path = save_checkpoint(paths, model, optimizer, ema_parameters, state)
            state.next_checkpoint_seconds = state.training_seconds + CHECKPOINT_INTERVAL_SECONDS
            if not arguments.no_upload and state.training_seconds >= state.next_upload_seconds:
                if upload_thread is None or not upload_thread.is_alive():
                    upload_thread = start_upload(checkpoint_path, paths, arguments.run_name)
                    state.next_upload_seconds = state.training_seconds + UPLOAD_INTERVAL_SECONDS

        if is_finished or is_smoke_finished:
            if is_main:
                checkpoint_path = save_checkpoint(paths, model, optimizer, ema_parameters, state)
                save_snapshot(paths, model, ema_parameters, state, "final")
                if is_finished:
                    paths.training_done.write_text(json.dumps(state.to_dict()))
                    if not arguments.no_upload:
                        if upload_thread is not None:
                            upload_thread.join()
                        upload_thread = start_upload(checkpoint_path, paths, arguments.run_name)
                print(f"trainer: finished at step {state.step}", flush=True)
            break

    if summary_writer is not None:
        summary_writer.close()
    if upload_thread is not None:
        upload_thread.join()
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
