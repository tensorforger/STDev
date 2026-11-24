"""
Generalization check of a FLUX.2 pretrain checkpoint: flow matching loss with fixed noise / sigmas on cache shards
grouped by age (old shards were reused for training several times, the newest ones barely or not at all). A falling
loss towards old shards means memorization of the reused cache.

Usage: CUDA_VISIBLE_DEVICES=7 python scripts/flux_tools/flux_generalization_gap.py DATA_RUN CHECKPOINT [--weights ema|raw]
           [--groups 6] [--shards-per-group 3]
"""

import argparse
import time
from pathlib import Path

import torch

from stdev.flux_pretrain.cache import list_shards, read_shard
from stdev.flux_pretrain.config import PAIR_TYPES, run_paths
from stdev.flux_pretrain.model import FluxUpscalerTransformer, ModelConfig, to_residual

BATCH_SIZE = 4
SAMPLES_PER_SHARD = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_run")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--shards-per-group", type=int, default=3)
    arguments = parser.parse_args()
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**{**checkpoint["model_config"], "residual_std": tuple(checkpoint["model_config"]["residual_std"])})
    model = FluxUpscalerTransformer(config)
    model.load_state_dict(checkpoint["model"])
    if arguments.weights == "ema":
        with torch.no_grad():
            for parameter, ema_parameter in zip(model.parameters(), checkpoint["ema"]):
                parameter.copy_(ema_parameter)
    model = model.cuda().eval().requires_grad_(False)
    checkpoint_time = arguments.checkpoint.stat().st_mtime
    print(f"{arguments.checkpoint.name}, {checkpoint['state']['training_seconds'] / 3600:.2f} h, {arguments.weights}")
    shards = list_shards(run_paths(arguments.data_run).cache / PAIR_TYPES[0].name)
    group_size = len(shards) // arguments.groups
    print(f"{len(shards)} shards of {PAIR_TYPES[0].name}; group = shard age range, loss mean over sigma in (0.1 .. 0.9)")
    for group in range(arguments.groups):
        group_shards = shards[group * group_size : (group + 1) * group_size]
        picked = [group_shards[index * len(group_shards) // arguments.shards_per_group] for index in range(arguments.shards_per_group)]
        losses = []
        for shard_path in picked:
            shard = read_shard(shard_path)
            generator = torch.Generator(device="cuda").manual_seed(0)
            for start in range(0, SAMPLES_PER_SHARD, BATCH_SIZE):
                latents_source = shard.latents_source[start : start + BATCH_SIZE].cuda()
                latents_target = shard.latents_target[start : start + BATCH_SIZE].cuda()
                hff = shard.hff[start : start + BATCH_SIZE].cuda()
                residual = to_residual(latents_target, latents_source, config.residual_std)
                noise = torch.randn(residual.shape, generator=generator, device="cuda")
                sigmas = torch.linspace(0.1, 0.9, BATCH_SIZE, device="cuda")
                noisy = (1 - sigmas[:, None, None, None]) * residual + sigmas[:, None, None, None] * noise
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    velocity = model(noisy, sigmas, latents_source, hff, torch.ones_like(hff, dtype=torch.bool))
                losses.append((velocity.float() - (noise - residual)).square().mean().item())
        oldest_age = (checkpoint_time - int(picked[0].name.split("_")[0]) / 1e9) / 3600
        newest_age = (checkpoint_time - int(picked[-1].name.split("_")[0]) / 1e9) / 3600
        print(f"  group {group}: shard age at checkpoint {oldest_age:5.2f} .. {newest_age:5.2f} h, loss {sum(losses) / len(losses):.4f}")


if __name__ == "__main__":
    main()
