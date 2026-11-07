"""
Single-GPU supervised fine-tuning of the pretrain_large latent upscaler on a pre-generated FLUX.2
teacher dataset (scripts/generate_flux_dataset.py): one resolution bucket per training batch,
rectified flow on the high resolution latents conditioned on the low resolution (bilinear 2x
downscale, re-encoded) latents, the pooled SDXL text embedding and the quantile normalized high
frequency features of the decoded high resolution image. Much simpler than the online pretraining
pipeline: the dataset is fully generated beforehand, so there is no parallel downloading or
encoding, just reading blocks from disk and training on them.

Usage: python scripts/run_sft.py --run-name NAME [--init-checkpoint PATH] [--max-steps N]
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from stdev.pretrain.high_frequency import QuantileNormalizer
from stdev.sft.config import (
    ADAM_BETAS,
    CHECKPOINT_INTERVAL_STEPS,
    DATA_WORKERS,
    DATASET_DIRECTORY as DEFAULT_DATASET_DIRECTORY,
    EMA_DECAY,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CHECKPOINTING,
    GRADIENT_CLIP_NORM,
    HIGH_FREQUENCY_DROPOUT_PROBABILITY,
    HIGH_FREQUENCY_QUANTILE_COUNT,
    HIGH_FREQUENCY_STATISTICS_SAMPLES,
    KEPT_CHECKPOINTS,
    LEARNING_RATE,
    LOG_INTERVAL_STEPS,
    MICRO_BATCH_SIZE,
    SEED,
    TEXT_DROPOUT_PROBABILITY,
    TRAINING_SIGMA_SHIFT,
    TRAINING_STEPS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    high_frequency_quantiles_path,
    run_directory,
)
from stdev.sft.data import (
    BucketBatchSampler,
    SFTBlockDataset,
    collate_blocks,
    collect_raw_high_frequency_features,
)
from stdev.upscalers.pretrain_large import UpscalerUNet, shift_sigmas

TRAINING_DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")


def sample_training_sigmas(batch_size: int) -> torch.Tensor:
    """
    returns:
        sigmas: [B], values: [0...1], logit-normal distributed and shifted
    """
    return shift_sigmas(torch.sigmoid(torch.randn(batch_size, device=DEVICE)), TRAINING_SIGMA_SHIFT)


def ensure_high_frequency_quantiles(dataset_directory: Path) -> QuantileNormalizer:
    quantiles_path = high_frequency_quantiles_path(dataset_directory)
    if not quantiles_path.exists():
        raw_features = collect_raw_high_frequency_features(
            dataset_directory, HIGH_FREQUENCY_STATISTICS_SAMPLES
        )
        normalizer = QuantileNormalizer.fit(raw_features, HIGH_FREQUENCY_QUANTILE_COUNT)
        quantiles_path.parent.mkdir(parents=True, exist_ok=True)
        normalizer.save(quantiles_path, len(raw_features))
        print(
            f"run_sft: fitted high frequency quantiles from {len(raw_features)} samples", flush=True
        )
    return QuantileNormalizer.load(quantiles_path)


def latest_checkpoint(checkpoints_directory: Path) -> Path | None:
    checkpoints = sorted(checkpoints_directory.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(
    checkpoints_directory: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_parameters: list[torch.Tensor],
    step: int,
) -> None:
    path = checkpoints_directory / f"checkpoint_{step:09d}.pt"
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": [parameter.cpu() for parameter in ema_parameters],
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        temporary_path,
    )
    temporary_path.replace(path)
    for old_path in sorted(checkpoints_directory.glob("checkpoint_*.pt"))[:-KEPT_CHECKPOINTS]:
        old_path.unlink()
    print(f"run_sft: checkpoint {path.name}", flush=True)


def export_weights(
    export_directory: Path, model: torch.nn.Module, ema_parameters: list[torch.Tensor]
) -> None:
    """
    Latest weights (fp32) in the PretrainLargeUpscaler format: export/sft_{raw,ema}.
    """
    raw = {
        name: tensor.detach().float().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    ema = dict(raw)
    for (name, _), ema_parameter in zip(model.named_parameters(), ema_parameters):
        ema[name] = ema_parameter.float().cpu().contiguous()
    for variant, tensors in (("raw", raw), ("ema", ema)):
        temporary_path = export_directory / f".sft_{variant}.tmp"
        save_file(tensors, str(temporary_path))
        temporary_path.replace(export_directory / f"sft_{variant}.safetensors")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--dataset-directory",
        type=Path,
        default=None,
        help="Overrides the dataset directory (default: stdev.sft.config.DATASET_DIRECTORY).",
    )
    parser.add_argument("--init-checkpoint", type=Path, help="pretrained weights to start from")
    parser.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    dataset_directory = arguments.dataset_directory or DEFAULT_DATASET_DIRECTORY

    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_root = run_directory(arguments.run_name)
    checkpoints_directory = run_root / "checkpoints"
    export_directory = run_root / "export"
    tensorboard_directory = run_root / "tensorboard"
    for directory in (checkpoints_directory, export_directory, tensorboard_directory):
        directory.mkdir(parents=True, exist_ok=True)

    normalizer = ensure_high_frequency_quantiles(dataset_directory)
    dataset = SFTBlockDataset(dataset_directory)
    bucket_sizes = dataset.bucket_sizes()
    print(f"run_sft: buckets {bucket_sizes}", flush=True)
    batch_sampler = BucketBatchSampler(bucket_sizes, MICRO_BATCH_SIZE, seed=SEED)
    data_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=DATA_WORKERS,
        collate_fn=collate_blocks,
        pin_memory=True,
        persistent_workers=DATA_WORKERS > 0,
    )
    batches = iter(data_loader)

    model = UpscalerUNet(gradient_checkpointing=GRADIENT_CHECKPOINTING).to(DEVICE)
    if arguments.init_checkpoint is not None:
        model.load_state_dict(load_file(str(arguments.init_checkpoint)))
        print(f"run_sft: initialized from {arguments.init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    ema_parameters = [parameter.detach().float().clone() for parameter in model.parameters()]
    step = 0

    checkpoint_path = latest_checkpoint(checkpoints_directory)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema_parameters = [parameter.to(DEVICE) for parameter in checkpoint["ema"]]
        step = checkpoint["step"]
        print(f"run_sft: resumed from {checkpoint_path.name} at step {step}", flush=True)

    compiled_model = torch.compile(model)
    parameters = list(model.parameters())
    summary_writer = SummaryWriter(str(tensorboard_directory))

    max_steps = arguments.max_steps if arguments.max_steps > 0 else TRAINING_STEPS
    loss_sum = torch.zeros((), device=DEVICE)
    loss_count = 0
    interval_start_step = step
    interval_start_time = time.perf_counter()

    while step < max_steps:
        warmup_factor = min(1.0, (step + 1) / WARMUP_STEPS)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = LEARNING_RATE * warmup_factor
        for _ in range(GRADIENT_ACCUMULATION_STEPS):
            batch = next(batches)
            latents_high = batch["latents_orig"].to(DEVICE, non_blocking=True)
            latents_low = batch["latents_div_2"].to(DEVICE, non_blocking=True)
            text_embedding = batch["text_embedding"].to(DEVICE, non_blocking=True)
            features = (
                torch.from_numpy(normalizer.normalize(batch["high_frequency_raw"].numpy()))
                .float()
                .to(DEVICE)
            )
            batch_size = latents_high.shape[0]
            text_mask = torch.rand(batch_size, device=DEVICE) >= TEXT_DROPOUT_PROBABILITY
            feature_mask = (
                torch.rand(features.shape, device=DEVICE) >= HIGH_FREQUENCY_DROPOUT_PROBABILITY
            ) & ~torch.isnan(features)
            noise = torch.randn_like(latents_high)
            sigmas = sample_training_sigmas(batch_size)
            blend = sigmas[:, None, None, None]
            noisy_latents = (1.0 - blend) * latents_high + blend * noise
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
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            loss_sum += loss.detach()
            loss_count += 1
        torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        decay = min(EMA_DECAY, (1.0 + step) / (10.0 + step))
        torch._foreach_lerp_(
            ema_parameters, [parameter.detach() for parameter in parameters], 1.0 - decay
        )
        step += 1

        if step % LOG_INTERVAL_STEPS == 0:
            elapsed = time.perf_counter() - interval_start_time
            mean_loss = loss_sum.item() / max(loss_count, 1)
            summary_writer.add_scalar("train/loss", mean_loss, step)
            summary_writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)
            summary_writer.add_scalar(
                "train/steps_per_second", (step - interval_start_step) / max(elapsed, 1e-6), step
            )
            print(f"run_sft: step {step} loss {mean_loss:.4f}", flush=True)
            loss_sum.zero_()
            loss_count = 0
            interval_start_time = time.perf_counter()
            interval_start_step = step

        if step % CHECKPOINT_INTERVAL_STEPS == 0 or step >= max_steps:
            save_checkpoint(checkpoints_directory, model, optimizer, ema_parameters, step)
            export_weights(export_directory, model, ema_parameters)

    print(f"run_sft: finished at step {step}", flush=True)


if __name__ == "__main__":
    main()
