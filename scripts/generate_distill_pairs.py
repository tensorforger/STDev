"""
Generates the rectified flow distillation pairs from the SFT dataset (scripts/generate_flux_dataset.py):
for every block the SFT-trained (EMA) upscaler is sampled from seeded random noise with TEACHER_STEPS
Euler steps at TEACHER_SIGMA_SHIFT, conditioned on the block's low resolution latents, pooled text
embedding and quantile normalized high frequency features, each randomly dropped out as in SFT.
Pairs are written as shards in the pair format of stdev.distill.data, one directory per resolution
bucket, `shard_XXXXX` covering SFT_PAIRS_SHARD_SAMPLES consecutive blocks of the bucket. Shards that
already exist are skipped, so a stopped run resumes by rerunning the same command.

Usage: torchrun --standalone --nproc_per_node=8 scripts/generate_distill_pairs.py
           python scripts/generate_distill_pairs.py --smoke   (few blocks per bucket, separate directory)
"""

import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from stdev.distill.config import (
    SFT_PAIRS_BATCH_SIZE,
    SFT_PAIRS_DIRECTORY,
    SFT_PAIRS_SHARD_SAMPLES,
    SFT_TEACHER_WEIGHTS_PATH,
    TEACHER_SIGMA_SHIFT,
    TEACHER_STEPS,
)
from stdev.distill.data import PAIR_SUFFIX, PairShard, make_noise, write_pair_shard
from stdev.pretrain.high_frequency import QuantileNormalizer
from stdev.sft.config import (
    BUCKET_SEED_STRIDE,
    DATASET_DIRECTORY as DEFAULT_DATASET_DIRECTORY,
    HIGH_FREQUENCY_DROPOUT_PROBABILITY,
    RESOLUTION_BUCKETS,
    TEXT_DROPOUT_PROBABILITY,
    high_frequency_quantiles_path,
)
from stdev.sft.data import collate_blocks, list_block_paths, read_block
from stdev.upscalers.pretrain_large import (
    INFERENCE_DTYPE,
    UpscalerUNet,
    sample_high_resolution_latents,
)

SMOKE_BLOCKS_PER_BUCKET = 8
COMPILE_CACHE_SIZE_LIMIT = 64

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-directory",
        type=Path,
        default=DEFAULT_DATASET_DIRECTORY,
        help="SFT blocks directory (default: stdev.sft.config.DATASET_DIRECTORY).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Pairs directory (default: stdev.distill.config.SFT_PAIRS_DIRECTORY).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=SFT_TEACHER_WEIGHTS_PATH,
        help="Teacher weights (default: stdev.distill.config.SFT_TEACHER_WEIGHTS_PATH).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Use only {SMOKE_BLOCKS_PER_BUCKET} blocks per bucket, written to a separate directory.",
    )
    return parser.parse_args()


def load_teacher(weights_path: Path) -> torch.nn.Module:
    model = UpscalerUNet()
    model.load_state_dict(load_file(str(weights_path)))
    return torch.compile(model.to(DEVICE, INFERENCE_DTYPE).eval().requires_grad_(False))


@torch.no_grad()
def generate_shard(
    teacher: torch.nn.Module,
    normalizer: QuantileNormalizer,
    block_paths: list[Path],
    first_seed: int,
    dropout_seed: int,
) -> PairShard:
    """
    Teacher pairs of consecutive blocks of one bucket; pair i uses make_noise(first_seed + i).
    """
    dropout_generator = torch.Generator(device=DEVICE).manual_seed(dropout_seed)
    parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("low", "teacher", "text", "features", "text_mask", "feature_mask")
    }
    seeds: list[int] = []
    for start in range(0, len(block_paths), SFT_PAIRS_BATCH_SIZE):
        batch_paths = block_paths[start : start + SFT_PAIRS_BATCH_SIZE]
        blocks = collate_blocks([read_block(path) for path in batch_paths])
        latents_low = blocks["latents_div_2"].to(DEVICE)  # [B, 32, h, w]
        text_embedding = blocks["text_embedding"].to(DEVICE)  # [B, 768]
        features = (
            torch.from_numpy(normalizer.normalize(blocks["high_frequency_raw"].numpy()))
            .float()
            .to(DEVICE)
        )  # [B, 4]
        batch_size = len(batch_paths)
        text_mask = (
            torch.rand(batch_size, generator=dropout_generator, device=DEVICE)
            >= TEXT_DROPOUT_PROBABILITY
        )
        feature_mask = (
            torch.rand(features.shape, generator=dropout_generator, device=DEVICE)
            >= HIGH_FREQUENCY_DROPOUT_PROBABILITY
        ) & ~torch.isnan(features)
        batch_seeds = [first_seed + start + offset for offset in range(batch_size)]
        height, width = blocks["latents_orig"].shape[-2:]
        noise = torch.stack([make_noise(seed, height, width) for seed in batch_seeds]).to(DEVICE)
        latents_teacher = sample_high_resolution_latents(
            teacher,
            latents_low,
            text_embedding,
            text_mask,
            features,
            feature_mask,
            num_steps=TEACHER_STEPS,
            sigma_shift=TEACHER_SIGMA_SHIFT,
            noise=noise,
        )
        parts["low"].append(latents_low.cpu())
        parts["teacher"].append(latents_teacher.cpu())
        parts["text"].append(text_embedding.cpu())
        parts["features"].append(features.cpu())
        parts["text_mask"].append(text_mask.cpu())
        parts["feature_mask"].append(feature_mask.cpu())
        seeds.extend(batch_seeds)
    return PairShard(*(torch.cat(values) for values in parts.values()), seeds)


def main() -> None:
    arguments = parse_arguments()
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = COMPILE_CACHE_SIZE_LIMIT

    output_directory = arguments.output_directory or (
        SFT_PAIRS_DIRECTORY.with_name(SFT_PAIRS_DIRECTORY.name + "_smoke")
        if arguments.smoke
        else SFT_PAIRS_DIRECTORY
    )
    block_paths_per_bucket = {
        bucket.name: list_block_paths(arguments.dataset_directory, bucket)[
            : SMOKE_BLOCKS_PER_BUCKET if arguments.smoke else None
        ]
        for bucket in RESOLUTION_BUCKETS
    }
    for bucket in RESOLUTION_BUCKETS:
        (output_directory / bucket.name).mkdir(parents=True, exist_ok=True)

    all_shards = [
        (bucket_index, shard_index)
        for bucket_index, bucket in enumerate(RESOLUTION_BUCKETS)
        for shard_index in range(
            -(-len(block_paths_per_bucket[bucket.name]) // SFT_PAIRS_SHARD_SAMPLES)
        )
    ]
    pending_shards = [
        (bucket_index, shard_index)
        for bucket_index, shard_index in all_shards[RANK::WORLD_SIZE]
        if not (
            output_directory
            / RESOLUTION_BUCKETS[bucket_index].name
            / f"shard_{shard_index:05d}{PAIR_SUFFIX}"
        ).exists()
    ]
    if not pending_shards:
        print(f"[rank {RANK}] Everything already exists", flush=True)
        return

    teacher = load_teacher(arguments.weights)
    normalizer = QuantileNormalizer.load(high_frequency_quantiles_path(arguments.dataset_directory))

    with tqdm(total=len(pending_shards), desc=f"Shards [rank {RANK}]", position=RANK) as progress:
        for bucket_index, shard_index in pending_shards:
            bucket = RESOLUTION_BUCKETS[bucket_index]
            first_block = shard_index * SFT_PAIRS_SHARD_SAMPLES
            shard_block_paths = block_paths_per_bucket[bucket.name][
                first_block : first_block + SFT_PAIRS_SHARD_SAMPLES
            ]
            first_seed = bucket_index * BUCKET_SEED_STRIDE + first_block
            shard = generate_shard(
                teacher, normalizer, shard_block_paths, first_seed, dropout_seed=first_seed
            )
            write_pair_shard(
                output_directory / bucket.name,
                f"shard_{shard_index:05d}",
                shard,
                {"bucket": bucket.name, "blocks": [path.name for path in shard_block_paths]},
            )
            progress.update(1)


if __name__ == "__main__":
    main()
