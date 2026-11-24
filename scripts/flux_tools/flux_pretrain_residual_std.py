"""
Per channel std of latents_target - bilinear 2x(latents_source) in the FLUX.2 pretrain cache, per pair type and on
the training mix (samples weighted by count_share), plus the bilinear-only and corrupted-only values.
Usage: python scripts/flux_tools/flux_pretrain_residual_std.py DATA_RUN_NAME
"""

import sys

import torch

from stdev.flux_pretrain.cache import list_shards, read_shard
from stdev.flux_pretrain.config import PAIR_TYPES, SOURCE_VARIANTS, run_paths
from stdev.flux_pretrain.model import align_source


def channel_moments(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    args:
        residual: [N, C, H, W]
    returns:
        mean of squares: [C], mean: [C]
    """
    return residual.square().mean(dim=(0, 2, 3)), residual.mean(dim=(0, 2, 3))


def main() -> None:
    paths = run_paths(sys.argv[1])
    mixed_square = torch.zeros(32, dtype=torch.float64)
    mixed_mean = torch.zeros(32, dtype=torch.float64)
    bilinear_index = SOURCE_VARIANTS.index("bilinear")
    for pair_type in PAIR_TYPES:
        squares, means, groups = [], [], {"bilinear": [], "corrupted": []}
        for shard_path in list_shards(paths.cache / pair_type.name):
            shard = read_shard(shard_path)
            residual = shard.latents_target - align_source(shard.latents_source)
            square, mean = channel_moments(residual)
            squares.append(square.double())
            means.append(mean.double())
            is_bilinear = shard.variant_index == bilinear_index
            for name, mask in (("bilinear", is_bilinear), ("corrupted", ~is_bilinear)):
                if mask.any():
                    groups[name].append(residual[mask].square().mean().item())
        square = torch.stack(squares).mean(0)
        mean = torch.stack(means).mean(0)
        std = (square - mean.square()).sqrt()
        print(f"{pair_type.name}: {len(squares)} shards, mean std {std.mean():.3f}, rms by source: "
              + ", ".join(f"{name} {sum(values) / len(values):.3f}" for name, values in groups.items() if values))
        print("  " + ", ".join(f"{value:.3f}" for value in std.tolist()))
        mixed_square += pair_type.count_share * square
        mixed_mean += pair_type.count_share * mean
    mixed_std = (mixed_square - mixed_mean.square()).sqrt()
    print(f"mix: mean std {mixed_std.mean():.3f}")
    print("RESIDUAL_STD = (" + ", ".join(f"{value:.3f}" for value in mixed_std.tolist()) + ")")


if __name__ == "__main__":
    main()
