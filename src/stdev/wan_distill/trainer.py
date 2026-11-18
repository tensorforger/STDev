"""
DDP trainer of the single step rectified flow distillation of the Wan 2.1 latent upscaler. The student
(stdev.wan_distill.student, initialized from the teacher EMA weights) maps the teacher's initial noise at sigma 1
directly to the teacher's sample: velocity target = noise_scale * noise - residual(teacher latents), conditioned
on the hff (NaN = null) and the guidance scale the teacher used. Epochs over the pair dataset
(stdev.wan_distill.data), constant LR after a short warmup, EMA on all ranks. After every epoch (and before the
first) a checkpoint is written and the 1000-image benchmark runs on all ranks with single step sampling.
`control/decay` in the run directory starts a linear LR decay to zero over DECAY_EPOCHS more epochs, after which
the run ends (final checkpoint, benchmark, R2 upload). `control/stop` (or SIGTERM) checkpoints and exits;
relaunching resumes (mid epoch from the same deterministic schedule).

Usage: torchrun --standalone --nproc-per-node 8 -m stdev.wan_distill.trainer --run-name NAME
           [--data-root /workspace/STDev] [--learning-rate LR] [--max-epochs N] [--max-steps N] [--no-upload]
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")

import argparse
import json
import math
import signal
import threading
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as distributed
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from stdev.pretrain.config import PretrainPaths
from stdev.wan_distill.config import (
    ACCUMULATION,
    ADAM_BETAS,
    CONTROL_CHECK_INTERVAL_STEPS,
    DECAY_EPOCHS,
    DEFAULT_DATA_ROOT,
    DYNAMO_CACHE_SIZE_LIMIT,
    EMA_DECAY,
    EVALUATION_DATA_RUN,
    EVALUATION_VARIANTS,
    FULL_VAE_DIRECTORY,
    GRADIENT_CLIP_NORM,
    KEPT_CHECKPOINTS,
    KEPT_UPLOADS,
    LEARNING_RATE,
    LOG_INTERVAL_STEPS,
    PAIRS_DIRECTORY,
    RUNS_DIRECTORY,
    SEED,
    TEACHER_CHECKPOINT,
    UPLOAD_BUCKET,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from stdev.wan_distill.data import RankPairs, epoch_schedule
from stdev.wan_distill.evaluation import DistributedEvaluation
from stdev.wan_distill.student import WanStudentTransformer, load_teacher_into_student
from stdev.wan_pretrain.config import MODEL, PAIR_TYPES, TOKENS_PER_UNIT
from stdev.wan_pretrain.model import to_residual

TRAINING_DTYPE = torch.bfloat16
LOSS_GROUPS = ("null_hff", "unguided", "guided")


class TrainingState:
    """
    Resumable counters and schedule state.
    """

    def __init__(self) -> None:
        self.step = 0
        self.epoch = 0
        self.step_in_epoch = 0
        self.samples_seen = 0
        self.units_seen = 0.0
        self.training_seconds = 0.0
        self.decay_start_step = -1
        self.decay_steps = 0
        self.ema_updates = 0
        self.evaluated_epochs: list[int] = []

    def learning_rate(self, base_learning_rate: float) -> float:
        learning_rate = base_learning_rate * min(1.0, (self.step + 1) / WARMUP_STEPS)
        if self.decay_start_step < 0:
            return learning_rate
        return learning_rate * max(0.0, 1.0 - (self.step - self.decay_start_step) / max(self.decay_steps, 1))

    def decay_finished(self) -> bool:
        return self.decay_start_step >= 0 and self.step >= self.decay_start_step + self.decay_steps

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def load_dict(self, values: dict) -> None:
        self.__dict__.update(values)


class RunDirectory(PretrainPaths):
    @property
    def done_marker(self) -> Path:
        return self.root / "training_done.json"


def save_checkpoint(paths: RunDirectory, model: torch.nn.Module, optimizer: torch.optim.Optimizer, ema: list[torch.Tensor], state: TrainingState) -> Path:
    path = paths.checkpoints / f"checkpoint_{state.step:07d}.pt"
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": [parameter.cpu() for parameter in ema],
            "optimizer": optimizer.state_dict(),
            "state": state.to_dict(),
            "model_config": MODEL.__dict__,
            "architecture": "WanStudentTransformer",
        },
        temporary_path,
    )
    temporary_path.replace(path)
    for old_path in sorted(paths.checkpoints.glob("checkpoint_*.pt"))[:-KEPT_CHECKPOINTS]:
        old_path.unlink()
    print(f"trainer: checkpoint {path.name}", flush=True)
    return path


def upload_checkpoint(local_path: Path, run_name: str, env_path: Path) -> None:
    """
    Uploads a hard linked checkpoint to R2 (UPLOAD_BUCKET/run_name/), keeps the newest KEPT_UPLOADS remote
    checkpoints. Failures are logged, never raised.
    """
    try:
        from stdev.s3_tools.client import get_s3_client
        from stdev.video_downloader.stream_videos import load_env_file

        load_env_file(env_path)
        client = get_s3_client()
        prefix = f"{run_name}/"
        client.upload_file(str(local_path), UPLOAD_BUCKET, prefix + local_path.name)
        listed = client.list_objects_v2(Bucket=UPLOAD_BUCKET, Prefix=prefix).get("Contents", [])
        for key in sorted(item["Key"] for item in listed if item["Key"].endswith(".pt"))[:-KEPT_UPLOADS]:
            client.delete_object(Bucket=UPLOAD_BUCKET, Key=key)
        print(f"trainer: uploaded {local_path.name} to s3://{UPLOAD_BUCKET}/{prefix}", flush=True)
    except Exception as error:
        print(f"trainer: upload of {local_path.name} failed: {error!r}", flush=True)
    finally:
        local_path.unlink(missing_ok=True)


def start_upload(checkpoint_path: Path, run_name: str, env_path: Path) -> threading.Thread:
    upload_directory = checkpoint_path.parent / "upload"
    upload_directory.mkdir(exist_ok=True)
    linked_path = upload_directory / checkpoint_path.name
    linked_path.unlink(missing_ok=True)
    os.link(checkpoint_path, linked_path)
    thread = threading.Thread(target=upload_checkpoint, args=(linked_path, run_name, env_path), daemon=True)
    thread.start()
    return thread


def warm_compile(model: WanStudentTransformer, distributed_model: DistributedDataParallel, device: torch.device) -> None:
    """
    Compiles forward and backward of every pair type shape before the clock starts (no gradient sync).
    """
    for pair_type in PAIR_TYPES:
        batch_size = pair_type.micro_batch_size
        model.checkpointed_blocks = pair_type.checkpointed_blocks
        with distributed_model.no_sync():
            with torch.autocast("cuda", dtype=TRAINING_DTYPE):
                velocity = distributed_model(
                    torch.randn(batch_size, 16, *pair_type.target_latent_size, device=device),
                    torch.randn(batch_size, 16, *pair_type.source_latent_size, device=device),
                    torch.rand(batch_size, device=device),
                    torch.ones(batch_size, dtype=torch.bool, device=device),
                    torch.ones(batch_size, device=device),
                )
            velocity.float().square().mean().backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


def make_inference_copy(model: WanStudentTransformer, weights: list[torch.Tensor] | None) -> WanStudentTransformer:
    """
    bf16 eval copy of the student with its own parameters or the given (EMA) parameters.
    """
    inference_model = WanStudentTransformer(model.config).to(next(model.parameters()).device).eval().requires_grad_(False)
    with torch.no_grad():
        source_parameters = weights if weights is not None else [parameter.detach() for parameter in model.parameters()]
        for parameter, weight in zip(inference_model.parameters(), source_parameters, strict=True):
            parameter.copy_(weight)
    return inference_model.to(torch.bfloat16)


def run_evaluation(
    evaluation: DistributedEvaluation,
    model: WanStudentTransformer,
    ema: list[torch.Tensor],
    state: TrainingState,
    paths: RunDirectory,
    summary_writer: SummaryWriter | None,
    tag: str,
) -> None:
    start_time = time.perf_counter()
    output_directory = paths.benchmarks / tag
    output_directory.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {"tag": tag, "step": state.step, "epoch": state.epoch, "samples_seen": state.samples_seen}
    copies: dict[str, WanStudentTransformer] = {}
    for variant in EVALUATION_VARIANTS:
        if variant.weights not in copies:
            copies[variant.weights] = make_inference_copy(model, ema if variant.weights == "ema" else None)
        metrics = evaluation.evaluate(copies[variant.weights], variant.guidance, output_directory, variant.name)
        for name, value in metrics.items():
            results[f"{variant.name}_{name}"] = value
            if summary_writer is not None:
                summary_writer.add_scalar(f"benchmark_{variant.name}/{name}", value, state.step)
    del copies
    torch.cuda.empty_cache()
    results["benchmark_seconds"] = time.perf_counter() - start_time
    if summary_writer is not None:
        summary_writer.add_scalar("benchmark/epoch", state.epoch, state.step)
        summary_writer.flush()
        with (paths.benchmarks / "results.jsonl").open("a") as results_file:
            results_file.write(json.dumps(results) + "\n")
        print(f"benchmark: {json.dumps(results)}", flush=True)


def read_control(paths: RunDirectory, state: TrainingState, steps_per_epoch: int) -> tuple[int, int]:
    """
    returns:
        decay_steps (0 when no new decay request), stop_requested (0 or 1)
    """
    decay_steps = 0
    if state.decay_start_step < 0 and paths.decay_request.exists():
        content = paths.decay_request.read_text().strip()
        decay_steps = int(content) if content else DECAY_EPOCHS * steps_per_epoch
        paths.decay_request.rename(paths.control / "decay.accepted")
    return decay_steps, int(paths.stop_request.exists())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-epochs", type=int, default=0, help="end (without decay) after this many epochs")
    parser.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps of this session")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--skip-initial-benchmark", action="store_true")
    parser.add_argument("--evaluation-count", type=int, default=0, help="smoke test: benchmark only the first N images")
    arguments = parser.parse_args()
    data_root = arguments.data_root

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    is_main = rank == 0
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = DYNAMO_CACHE_SIZE_LIMIT

    paths = RunDirectory(data_root / RUNS_DIRECTORY / arguments.run_name)
    if is_main:
        paths.create()
    distributed.barrier()

    model = WanStudentTransformer(MODEL)
    optimizer_parameters = list(model.parameters())
    state = TrainingState()
    checkpoints = sorted(paths.checkpoints.glob("checkpoint_*.pt"))
    checkpoint = torch.load(checkpoints[-1], map_location="cpu", weights_only=False) if checkpoints else None
    if checkpoint is None:
        load_teacher_into_student(model, data_root / TEACHER_CHECKPOINT)
    else:
        model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY, fused=True)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        state.load_dict(checkpoint["state"])
        ema = [parameter.to(device) for parameter in checkpoint["ema"]]
        if is_main:
            print(f"trainer: resumed from {checkpoints[-1].name} at step {state.step}, epoch {state.epoch}", flush=True)
        del checkpoint
    else:
        ema = [parameter.detach().float().clone() for parameter in model.parameters()]
    del optimizer_parameters

    model.compile_blocks()
    distributed_model = DistributedDataParallel(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    distributed_model.register_comm_hook(None, default_hooks.bf16_compress_hook)
    parameters = list(model.parameters())
    compile_start = time.perf_counter()
    warm_compile(model, distributed_model, device)

    load_start = time.perf_counter()
    pairs = RankPairs(data_root / PAIRS_DIRECTORY, rank, world_size)
    counts = torch.tensor([pairs.count(pair_type) // pair_type.micro_batch_size for pair_type in PAIR_TYPES], device=device)
    distributed.all_reduce(counts, op=distributed.ReduceOp.MIN)
    micro_batches_per_type = {pair_type.name: int(count) for pair_type, count in zip(PAIR_TYPES, counts.tolist())}
    steps_per_epoch = sum(micro_batches_per_type.values()) // ACCUMULATION
    evaluation = DistributedEvaluation(PretrainPaths(data_root / EVALUATION_DATA_RUN), data_root / FULL_VAE_DIRECTORY, rank, world_size, arguments.evaluation_count)
    summary_writer = SummaryWriter(str(paths.tensorboard / "train")) if is_main else None
    if is_main:
        print(
            f"trainer: {sum(parameter.numel() for parameter in parameters) / 1e6:.1f}M parameters, {world_size} ranks, "
            f"compiled in {load_start - compile_start:.0f}s, data loaded in {time.perf_counter() - load_start:.0f}s, "
            f"micro-batches per epoch {json.dumps(micro_batches_per_type)}, {steps_per_epoch} steps per epoch, "
            f"lr {arguments.learning_rate:.1e}, accumulation {ACCUMULATION}",
            flush=True,
        )
        summary_writer.add_text("config", json.dumps({**{key: str(value) for key, value in vars(arguments).items()}, "steps_per_epoch": steps_per_epoch}), state.step)

    if 0 not in state.evaluated_epochs and not arguments.skip_initial_benchmark:
        run_evaluation(evaluation, model, ema, state, paths, summary_writer, "epoch_000")
        state.evaluated_epochs.append(0)

    termination_requested = False

    def request_termination(signal_number: int, frame: object) -> None:
        nonlocal termination_requested
        termination_requested = True

    signal.signal(signal.SIGTERM, request_termination)

    type_names = [pair_type.name for pair_type in PAIR_TYPES]
    statistics = torch.zeros(2 * len(PAIR_TYPES) + 2 * len(LOSS_GROUPS), device=device)
    gradient_norm_sum = torch.zeros((), device=device)
    gradient_norm_max = torch.zeros((), device=device)
    interval_start = time.perf_counter()
    interval_samples = 0
    steps_this_session = 0
    upload_thread: threading.Thread | None = None
    env_path = data_root / ".env"
    finished = False
    stopped = False

    while not finished and not stopped:
        schedule = epoch_schedule(micro_batches_per_type, state.epoch, SEED)
        while state.step_in_epoch < steps_per_epoch:
            step_start = time.perf_counter()
            learning_rate = state.learning_rate(arguments.learning_rate)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            step_samples = 0
            step_units = 0.0
            for micro_step in range(ACCUMULATION):
                pair_type, batch_index = schedule[state.step_in_epoch * ACCUMULATION + micro_step]
                batch = {name: tensor.to(device, non_blocking=True) for name, tensor in pairs.batch(pair_type, state.epoch, rank, batch_index).items()}
                batch_size = batch["hff"].shape[0]
                noise = batch["noise_scale"][:, None, None, None, None] * batch["noise"].float()
                residual = to_residual(batch["latents_generated"].float(), batch["latents_source"].float())
                hff_mask = ~torch.isnan(batch["hff"])
                model.checkpointed_blocks = pair_type.checkpointed_blocks
                synchronize = micro_step == ACCUMULATION - 1
                with torch.autocast("cuda", dtype=TRAINING_DTYPE), (nullcontext() if synchronize else distributed_model.no_sync()):
                    velocity = distributed_model(
                        noise, batch["latents_source"], torch.nan_to_num(batch["hff"], nan=0.0), hff_mask, batch["guidance"]
                    )
                    sample_losses = (velocity.float() - (noise - residual)).square().mean(dim=(1, 2, 3, 4))  # [B]
                    loss = sample_losses.mean()
                    (loss / ACCUMULATION).backward()
                type_index = type_names.index(pair_type.name)
                statistics[2 * type_index] += loss.detach()
                statistics[2 * type_index + 1] += 1
                groups = torch.where(~hff_mask, 0, torch.where(batch["guidance"] != 1.0, 2, 1))
                offset = 2 * len(PAIR_TYPES)
                statistics[offset : offset + len(LOSS_GROUPS)] += torch.zeros(len(LOSS_GROUPS), device=device).index_add_(0, groups, sample_losses.detach())
                statistics[offset + len(LOSS_GROUPS) :] += torch.bincount(groups, minlength=len(LOSS_GROUPS)).float()
                step_samples += world_size * batch_size
                step_units += world_size * batch_size * pair_type.tokens_per_sample / TOKENS_PER_UNIT

            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            gradient_norm_sum += gradient_norm.detach()
            gradient_norm_max = torch.maximum(gradient_norm_max, gradient_norm.detach())
            state.ema_updates += 1
            decay = min(EMA_DECAY, (1.0 + state.ema_updates) / (10.0 + state.ema_updates))
            torch._foreach_lerp_(ema, [parameter.detach() for parameter in parameters], 1.0 - decay)

            state.step += 1
            state.step_in_epoch += 1
            state.samples_seen += step_samples
            state.units_seen += step_units
            steps_this_session += 1
            interval_samples += step_samples
            if steps_this_session > 1:
                state.training_seconds += time.perf_counter() - step_start

            if state.step % CONTROL_CHECK_INTERVAL_STEPS == 0 or termination_requested:
                control = torch.zeros(2, device=device)
                if is_main:
                    decay_steps, stop_requested = read_control(paths, state, steps_per_epoch)
                    control[0] = decay_steps
                    control[1] = stop_requested
                control[1] += float(termination_requested)
                distributed.all_reduce(control, op=distributed.ReduceOp.MAX)
                if control[0] > 0 and state.decay_start_step < 0:
                    state.decay_start_step = state.step
                    state.decay_steps = int(control[0].item())
                    if is_main:
                        print(f"trainer: decay over {state.decay_steps} steps from step {state.step}", flush=True)
                if control[1] > 0:
                    stopped = True

            if state.step % LOG_INTERVAL_STEPS == 0:
                distributed.all_reduce(statistics)
                if is_main:
                    elapsed = time.perf_counter() - interval_start
                    type_losses = statistics[: 2 * len(PAIR_TYPES)].view(-1, 2).tolist()
                    offset = 2 * len(PAIR_TYPES)
                    group_sums = statistics[offset : offset + len(LOSS_GROUPS)].tolist()
                    group_counts = statistics[offset + len(LOSS_GROUPS) :].tolist()
                    total_loss = sum(group_sums) / max(1.0, sum(group_counts))
                    scalars = {
                        "train/loss": total_loss,
                        "train/learning_rate": learning_rate,
                        "train/epoch": state.epoch + state.step_in_epoch / steps_per_epoch,
                        "train/samples_per_second": interval_samples / max(elapsed, 1e-6),
                        "train/grad_norm_mean": gradient_norm_sum.item() / LOG_INTERVAL_STEPS,
                        "train/grad_norm_max": gradient_norm_max.item(),
                        "train/training_hours": state.training_seconds / 3600,
                        "train/units_seen": state.units_seen,
                        "system/gpu_memory_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                    }
                    for name, (loss_sum, count) in zip(type_names, type_losses):
                        if count > 0:
                            scalars[f"loss_by_type/{name}"] = loss_sum / count
                    for name, loss_sum, count in zip(LOSS_GROUPS, group_sums, group_counts):
                        if count > 0:
                            scalars[f"loss_by_condition/{name}"] = loss_sum / count
                    for name, value in scalars.items():
                        summary_writer.add_scalar(name, value, state.step)
                    print(
                        f"trainer: step {state.step} epoch {scalars['train/epoch']:.2f} loss {total_loss:.4f} lr {learning_rate:.2e} "
                        f"grad {scalars['train/grad_norm_mean']:.3f}/{scalars['train/grad_norm_max']:.3f} "
                        f"{scalars['train/samples_per_second']:.1f} samples/s, {state.training_seconds / 3600:.2f} h",
                        flush=True,
                    )
                    torch.cuda.reset_peak_memory_stats()
                statistics.zero_()
                gradient_norm_sum.zero_()
                gradient_norm_max.zero_()
                interval_samples = 0
                interval_start = time.perf_counter()

            if state.decay_finished():
                finished = True
            if arguments.max_steps > 0 and steps_this_session >= arguments.max_steps:
                stopped = True
            if finished or stopped:
                break

        if state.step_in_epoch >= steps_per_epoch:
            state.epoch += 1
            state.step_in_epoch = 0
            if arguments.max_epochs > 0 and state.epoch >= arguments.max_epochs:
                finished = True
        if stopped and not finished:
            if is_main:
                save_checkpoint(paths, model, optimizer, ema, state)
                paths.stop_request.unlink(missing_ok=True)
                print(f"trainer: stopped at step {state.step}", flush=True)
            break
        tag = "final" if finished else f"epoch_{state.epoch:03d}"
        state.evaluated_epochs.append(state.epoch)
        checkpoint_path = save_checkpoint(paths, model, optimizer, ema, state) if is_main else None
        run_evaluation(evaluation, model, ema, state, paths, summary_writer, tag)
        if is_main:
            if finished:
                paths.done_marker.write_text(json.dumps(state.to_dict()))
            if not arguments.no_upload and (upload_thread is None or not upload_thread.is_alive()):
                upload_thread = start_upload(checkpoint_path, arguments.run_name, env_path)
        interval_start = time.perf_counter()

    if summary_writer is not None:
        summary_writer.close()
    if upload_thread is not None:
        upload_thread.join()
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
