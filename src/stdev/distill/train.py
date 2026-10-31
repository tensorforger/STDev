"""
Student training of one distillation round (DDP over all GPUs). Rectified flow on the teacher
pairs: x_t = (1 - t) * teacher_output + t * noise with t from the fixed noise levels of the 1 to
STUDENT_MAX_STEPS step uniform schedules (training_timesteps), velocity target noise - teacher_output, conditioned exactly as the teacher was (same dropout masks). Trains
TRAINING_REPEATS * pairs / batch steps, then saves a full checkpoint and exports the raw and EMA
weights. The first round starts from the teacher weights. The final linear LR decay starts when
`control/decay` appears (optional content: decay steps, default DEFAULT_DECAY_FRACTION of the steps
done); `training_done` is written when it is finished.

Usage: torchrun --nproc-per-node 8 -m stdev.distill.train --run-name NAME --round R
"""

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from stdev.distill.config import (
    ADAM_BETAS,
    CONTROL_CHECK_INTERVAL_STEPS,
    DATA_WORKERS_PER_RANK,
    DEFAULT_DECAY_FRACTION,
    EMA_DECAY,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CLIP_NORM,
    KEPT_CHECKPOINTS,
    LEARNING_RATE,
    LOG_INTERVAL_STEPS,
    MICRO_BATCH_SIZE,
    SEED,
    STUDENT_MAX_STEPS,
    TEACHER_WEIGHTS_PATH,
    TRAINING_REPEATS,
    WARMUP_STEPS,
    DistillPaths,
    run_paths,
)
from stdev.distill.data import RoundPairDataset, count_pairs
from stdev.upscalers.pretrain_large import UpscalerUNet

TRAINING_DTYPE = torch.bfloat16


def training_timesteps() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Noise levels of the uniform 1 to STUDENT_MAX_STEPS step schedules; each schedule has the same
    total weight, split equally over its steps.

    returns:
        timesteps: [T], values: (0...1], 1 is pure noise; probabilities: [T], sum 1
    """
    weights: dict[float, float] = {}
    for steps in range(1, STUDENT_MAX_STEPS + 1):
        for step in range(steps):
            timestep = round(1.0 - step / steps, 6)
            weights[timestep] = weights.get(timestep, 0.0) + 1.0 / (steps * STUDENT_MAX_STEPS)
    timesteps = sorted(weights, reverse=True)
    return torch.tensor(timesteps), torch.tensor([weights[timestep] for timestep in timesteps])


class DistillState:
    """
    Resumable counters and learning rate schedule state.
    """

    def __init__(self) -> None:
        self.round = -1
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


def latest_checkpoint(paths: DistillPaths) -> Path | None:
    checkpoints = sorted(paths.checkpoints.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(
    paths: DistillPaths,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_parameters: list[torch.Tensor],
    state: DistillState,
) -> None:
    path = paths.checkpoints / f"checkpoint_r{state.round:04d}_{state.step:09d}.pt"
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
    print(f"train: checkpoint {path.name}", flush=True)


def export_weights(
    paths: DistillPaths, model: torch.nn.Module, ema_parameters: list[torch.Tensor]
) -> None:
    """
    Latest student weights (fp32) in the PretrainLargeUpscaler format: export/student_{raw,ema}.
    """
    raw = {name: tensor.detach().float().cpu().contiguous() for name, tensor in model.state_dict().items()}
    ema = dict(raw)
    for (name, _), ema_parameter in zip(model.named_parameters(), ema_parameters):
        ema[name] = ema_parameter.float().cpu().contiguous()
    for variant, tensors in (("raw", raw), ("ema", ema)):
        temporary_path = paths.export / f".student_{variant}.tmp"
        save_file(tensors, str(temporary_path))
        temporary_path.replace(paths.export / f"student_{variant}.safetensors")


def read_decay_request(paths: DistillPaths, state: DistillState) -> int:
    """
    returns:
        decay steps of a new decay request, 0 when there is none
    """
    if state.decay_start_step >= 0 or not paths.decay_request.exists():
        return 0
    content = paths.decay_request.read_text().strip()
    decay_steps = int(content) if content else max(1, round(DEFAULT_DECAY_FRACTION * state.step))
    paths.decay_request.rename(paths.control / "decay.accepted")
    return decay_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--round", type=int, required=True)
    arguments = parser.parse_args()

    distributed.init_process_group("nccl")
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(SEED + 1000 * arguments.round + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    paths = run_paths(arguments.run_name)
    is_main = rank == 0

    model = UpscalerUNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=0.0, fused=True
    )
    state = DistillState()
    ema_parameters: list[torch.Tensor] = []
    checkpoint_path = latest_checkpoint(paths)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        state.load_dict(checkpoint["state"])
        if is_main:
            ema_parameters = [parameter.to(device) for parameter in checkpoint["ema"]]
        del checkpoint
        if state.round >= arguments.round:
            raise RuntimeError(f"round {arguments.round} already trained ({checkpoint_path.name})")
    else:
        model.load_state_dict(load_file(str(TEACHER_WEIGHTS_PATH)))
    if is_main and not ema_parameters:
        ema_parameters = [parameter.detach().float().clone() for parameter in model.parameters()]
    state.round = arguments.round

    distributed_model = DistributedDataParallel(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    distributed_model.register_comm_hook(None, default_hooks.bf16_compress_hook)
    compiled_model = torch.compile(distributed_model)
    parameters = list(model.parameters())

    pairs_directory = paths.pairs(arguments.round)
    pair_count = count_pairs(pairs_directory)
    samples_per_step = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size
    round_steps = max(1, TRAINING_REPEATS * pair_count // samples_per_step)
    data_loader = DataLoader(
        RoundPairDataset(pairs_directory, rank, world_size, seed=SEED * 1000 + arguments.round),
        batch_size=MICRO_BATCH_SIZE,
        num_workers=DATA_WORKERS_PER_RANK,
        pin_memory=True,
        prefetch_factor=4,
    )
    batches = iter(data_loader)
    summary_writer = SummaryWriter(str(paths.tensorboard / "train")) if is_main else None
    if is_main:
        print(
            f"train: round {arguments.round}, {pair_count} pairs, {round_steps} steps of "
            f"{samples_per_step} samples, from step {state.step}",
            flush=True,
        )

    timesteps, timestep_probabilities = (tensor.to(device) for tensor in training_timesteps())
    if is_main:
        levels = ", ".join(f"{level:.3f}: {probability:.3f}" for level, probability in zip(timesteps.tolist(), timestep_probabilities.tolist()))
        print(f"train: timesteps {levels}", flush=True)
    loss_sum = torch.zeros((), device=device)
    loss_count = 0
    round_loss_sum = 0.0
    round_loss_count = 0
    interval_start_time = time.perf_counter()
    interval_start_step = state.step
    step_start_time = time.perf_counter()
    round_start_seconds = time.perf_counter()
    for round_step in range(round_steps):
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = state.learning_rate()
        for micro_step in range(GRADIENT_ACCUMULATION_STEPS):
            batch = {name: value.to(device, non_blocking=True) for name, value in next(batches).items()}
            latents_teacher = batch["latents_teacher"]
            noise = batch["noise"]
            sigmas = timesteps[torch.multinomial(timestep_probabilities, latents_teacher.shape[0], replacement=True)]
            blend = sigmas[:, None, None, None]
            noisy_latents = (1.0 - blend) * latents_teacher + blend * noise
            is_last_micro_step = micro_step == GRADIENT_ACCUMULATION_STEPS - 1
            synchronization = nullcontext() if is_last_micro_step else distributed_model.no_sync()
            with synchronization:
                with torch.autocast("cuda", dtype=TRAINING_DTYPE):
                    velocity = compiled_model(
                        noisy_latents,
                        sigmas,
                        batch["latents_low"],
                        batch["text_embedding"],
                        batch["text_mask"],
                        batch["high_frequency_features"],
                        batch["high_frequency_mask"],
                    )
                loss = functional.mse_loss(velocity.float(), noise - latents_teacher)
                (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            loss_sum += loss.detach()
            loss_count += 1
        torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if is_main:
            state.ema_updates += 1
            decay = min(EMA_DECAY, (1.0 + state.ema_updates) / (10.0 + state.ema_updates))
            torch._foreach_lerp_(ema_parameters, [parameter.detach() for parameter in parameters], 1.0 - decay)

        now = time.perf_counter()
        if round_step > 0:
            state.training_seconds += now - step_start_time
        else:
            interval_start_time = now
            interval_start_step = state.step + 1
            if is_main:
                print(f"train: first step after {now - round_start_seconds:.0f} s", flush=True)
        step_start_time = now
        state.step += 1
        state.samples_seen += samples_per_step

        if state.step % CONTROL_CHECK_INTERVAL_STEPS == 0 or round_step == 0:
            control = torch.zeros(1, device=device)
            if is_main:
                control[0] = read_decay_request(paths, state)
            distributed.broadcast(control, src=0)
            if control[0] > 0 and state.decay_start_step < 0:
                state.decay_start_step = state.step
                state.decay_steps = int(control[0].item())
                if is_main:
                    print(f"train: decay over {state.decay_steps} steps from step {state.step}", flush=True)

        if is_main and state.step % LOG_INTERVAL_STEPS == 0:
            elapsed = time.perf_counter() - interval_start_time
            steps = state.step - interval_start_step
            mean_loss = loss_sum.item() / max(loss_count, 1)
            round_loss_sum += loss_sum.item()
            round_loss_count += loss_count
            samples_per_second = steps * samples_per_step / max(elapsed, 1e-6)
            summary_writer.add_scalar("train/loss", mean_loss, state.step)
            summary_writer.add_scalar("train/learning_rate", state.learning_rate(), state.step)
            summary_writer.add_scalar("train/samples_per_second", samples_per_second, state.step)
            summary_writer.add_scalar("train/round", arguments.round, state.step)
            print(
                f"train: step {state.step} loss {mean_loss:.4f} lr {state.learning_rate():.2e} "
                f"{samples_per_second:.0f} samples/s",
                flush=True,
            )
            loss_sum.zero_()
            loss_count = 0
            interval_start_time = time.perf_counter()
            interval_start_step = state.step
        if state.decay_finished():
            break

    if is_main:
        save_checkpoint(paths, model, optimizer, ema_parameters, state)
        export_weights(paths, model, ema_parameters)
        result = {
            "round": arguments.round,
            "pairs": pair_count,
            "steps": round_step + 1,
            "step": state.step,
            "loss": round_loss_sum / max(round_loss_count, 1),
            "learning_rate": state.learning_rate(),
            "seconds": time.perf_counter() - round_start_seconds,
        }
        (paths.round_results(arguments.round) / "train.json").write_text(json.dumps(result))
        if state.decay_finished():
            paths.training_done.write_text(json.dumps(state.to_dict()))
        print(f"train: {json.dumps(result)}", flush=True)
        summary_writer.close()
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
