"""
Single-GPU training of the FlowUpscaler UNet (random initialization, no dropout) as a plain multi
step rectified flow model on the FLUX.2 SFT dataset (scripts/generate_flux_dataset.py): the target
is the original FLUX.2 output (latents_orig), noise and logit-normal sigmas are drawn on the fly,
teacher outputs are not used. Sampling uses 4 uniform Euler steps.

After every epoch over all blocks the EMA weights are benchmarked with scripts/run_benchmark.py
on flux_1024 (images and metrics in raw_results/NAME_epoch_NNN, scalars in tensorboard).
Control files and resuming work as in scripts/run_small_distill.py (`control/decay`, `control/stop`).

Usage: python scripts/run_flow_sft.py --run-name NAME [--dataset-directory PATH] [--smoke]
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import run_small_distill as shared
from stdev.distill.config import ADAM_BETAS, GRADIENT_CLIP_NORM, LOG_INTERVAL_STEPS, SEED, run_paths
from stdev.sft.config import DATASET_DIRECTORY, TRAINING_SIGMA_SHIFT
from stdev.sft.data import EpochBucketBatchSampler, SFTBlockDataset, collate_blocks
from stdev.upscalers import flow_upscaler_sft
from stdev.upscalers.pretrain_large import shift_sigmas

ARCHITECTURE = "flow_sft"
shared.ARCHITECTURE_MODULES[ARCHITECTURE] = flow_upscaler_sft
DEVICE = shared.DEVICE


def sample_training_sigmas(batch_size: int) -> torch.Tensor:
    """
    returns:
        sigmas: [B], values: [0...1], logit-normal distributed and shifted
    """
    return shift_sigmas(torch.sigmoid(torch.randn(batch_size, device=DEVICE)), TRAINING_SIGMA_SHIFT)


def run_epoch(
    compiled_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_parameters: list[torch.Tensor],
    state: shared.TrainingState,
    dataset: SFTBlockDataset,
    steps_per_epoch: int,
    summary_writer: SummaryWriter,
) -> float:
    """
    returns:
        mean training loss over the epoch
    """
    parameters = list(compiled_model.parameters())
    accumulation_steps = shared.BATCH_SIZE // shared.MICRO_BATCH_SIZE
    batch_sampler = EpochBucketBatchSampler(
        dataset.bucket_sizes(), shared.MICRO_BATCH_SIZE, seed=SEED
    )
    batch_sampler.epoch = state.epoch
    data_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=shared.DATA_WORKERS_PER_RANK,
        collate_fn=collate_blocks,
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
            latents_high = batch["latents_orig"].to(DEVICE, non_blocking=True)
            latents_low = batch["latents_div_2"].to(DEVICE, non_blocking=True)
            batch_size = latents_high.shape[0]
            noise = torch.randn_like(latents_high)
            sigmas = sample_training_sigmas(batch_size)
            blend = sigmas[:, None, None, None]
            noisy_latents = (1.0 - blend) * latents_high + blend * noise
            with torch.autocast("cuda", dtype=shared.TRAINING_DTYPE):
                velocity = compiled_model(noisy_latents, sigmas, latents_low)
            loss = functional.mse_loss(velocity.float(), noise - latents_high)
            (loss / accumulation_steps).backward()
            loss_sum += loss.detach()
            loss_count += 1
            epoch_loss_sum += loss.detach()
            epoch_loss_count += 1
            state.samples_seen += batch_size
        torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        state.ema_updates += 1
        decay = min(shared.EMA_DECAY, (1.0 + state.ema_updates) / (10.0 + state.ema_updates))
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
                f"run_flow_sft: epoch {state.epoch + 1} step {state.step} "
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
    parser.add_argument("--dataset-directory", type=Path, default=DATASET_DIRECTORY)
    parser.add_argument(
        "--max-epochs", type=int, default=0, help="stop after this many epochs in total"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"{shared.SMOKE_STEPS_PER_EPOCH} steps per epoch and a smoke benchmark without results",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = shared.COMPILE_CACHE_SIZE_LIMIT

    paths = run_paths(arguments.run_name)
    paths.create()

    dataset = SFTBlockDataset(arguments.dataset_directory)
    if len(dataset) == 0:
        raise FileNotFoundError(f"no blocks in {arguments.dataset_directory}")
    accumulation_steps = shared.BATCH_SIZE // shared.MICRO_BATCH_SIZE
    steps_per_epoch = (
        sum(size // shared.MICRO_BATCH_SIZE for size in dataset.bucket_sizes().values())
        // accumulation_steps
    )
    if arguments.smoke:
        steps_per_epoch = min(steps_per_epoch, shared.SMOKE_STEPS_PER_EPOCH)

    model = flow_upscaler_sft.FlowUpscalerSFTUNet().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=shared.LEARNING_RATE,
        betas=ADAM_BETAS,
        weight_decay=shared.WEIGHT_DECAY,
        fused=True,
    )
    ema_parameters = [parameter.detach().float().clone() for parameter in model.parameters()]
    state = shared.TrainingState()
    checkpoint_path = shared.latest_checkpoint(paths.checkpoints)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema_parameters = [parameter.to(DEVICE) for parameter in checkpoint["ema"]]
        state.load_dict(checkpoint["state"])
        del checkpoint
        print(f"run_flow_sft: resumed from {checkpoint_path.name} at step {state.step}", flush=True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"run_flow_sft: {parameter_count / 1e6:.1f}M parameters, {len(dataset)} blocks "
        f"{dataset.bucket_sizes()}, {steps_per_epoch} steps of {shared.BATCH_SIZE} samples per epoch",
        flush=True,
    )
    compiled_model = torch.compile(model, dynamic=False)
    summary_writer = SummaryWriter(str(paths.tensorboard / "flow_sft"))

    last_metrics_path = (
        shared.RAW_RESULTS_DIRECTORY
        / f"{arguments.run_name}_epoch_{state.epoch:03d}"
        / "metrics.json"
    )
    if state.epoch > 0 and not arguments.smoke and not last_metrics_path.exists():
        metrics = shared.run_epoch_benchmark(
            paths, arguments.run_name, state.epoch, arguments.smoke, ARCHITECTURE
        )
        shared.log_benchmark(summary_writer, metrics, state)

    while not state.decay_finished() and not paths.stop_request.exists():
        if arguments.max_epochs > 0 and state.epoch >= arguments.max_epochs:
            break
        shared.read_control(paths, state, steps_per_epoch)
        epoch_start_time = time.perf_counter()
        epoch_loss = run_epoch(
            compiled_model,
            optimizer,
            ema_parameters,
            state,
            dataset,
            steps_per_epoch,
            summary_writer,
        )
        state.epoch += 1
        state.training_seconds += time.perf_counter() - epoch_start_time
        summary_writer.add_scalar("epoch/train_loss", epoch_loss, state.step)
        summary_writer.add_scalar("epoch/training_hours", state.training_seconds / 3600, state.step)
        print(f"run_flow_sft: epoch {state.epoch} finished, mean loss {epoch_loss:.4f}", flush=True)

        shared.save_checkpoint(paths.checkpoints, model, optimizer, ema_parameters, state)
        shared.export_weights(paths, model, ema_parameters)
        metrics = shared.run_epoch_benchmark(
            paths, arguments.run_name, state.epoch, arguments.smoke, ARCHITECTURE
        )
        if metrics is not None:
            shared.log_benchmark(summary_writer, metrics, state)
    summary_writer.close()


if __name__ == "__main__":
    main()
