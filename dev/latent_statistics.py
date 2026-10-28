"""
Statistics of high resolution latents, the residual over interpolated low resolution latents, and
the teacher noise, to choose the parameterization of the flow.
"""

from pathlib import Path

import torch
import torch.nn.functional as functional
import vaezip

DATASET_DIRECTORY = Path(__file__).resolve().parents[1] / "train_datasets" / "flux_1024_20k"
NUM_SAMPLES = 256


def main() -> None:
    block_paths = sorted(DATASET_DIRECTORY.glob("block_*.vz.safetensors"))[:NUM_SAMPLES]
    high, low, noise = [], [], []
    for path in block_paths:
        block = vaezip.load(path)
        high.append(block["latents_orig"].decode(dtype=torch.float32))
        low.append(block["latents_div_2"].decode(dtype=torch.float32))
        noise.append(block["init_latents"].decode(dtype=torch.float32))
    high, low, noise = torch.cat(high), torch.cat(low), torch.cat(noise)
    print("high mean/std", high.mean().item(), high.std().item())
    print("high per-channel std", high.std(dim=(0, 2, 3)).numpy().round(2))
    print("low mean/std", low.mean().item(), low.std().item())
    print("noise mean/std", noise.mean().item(), noise.std().item())
    for mode in ("nearest", "bilinear", "bicubic"):
        upsampled = functional.interpolate(low, scale_factor=2, mode=mode)
        residual = high - upsampled
        print(
            f"{mode}: residual std {residual.std().item():.4f} "
            f"mse {residual.pow(2).mean().item():.4f} "
            f"per-sample-mean-shift {residual.mean(dim=(2, 3)).abs().mean().item():.4f}"
        )
    downsampled = functional.avg_pool2d(high, 2)
    print("avgpool(high) vs low mse", (downsampled - low).pow(2).mean().item())
    correlation = (noise * (high - high.mean())).mean() / (noise.std() * high.std())
    print("noise-high correlation", correlation.item())


if __name__ == "__main__":
    main()
