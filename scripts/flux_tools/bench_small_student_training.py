"""
Training throughput of the small FLUX.2 student (SMALL_MODEL) on one GPU with synthetic batches: compiled forward +
backward + AdamW step, including the seeded noise generation, for several micro-batch sizes at 1024 and 512.
Usage: CUDA_VISIBLE_DEVICES=1 python scripts/flux_tools/bench_small_student_training.py [--seconds 20] [--sizes 8 16 32]
"""

import argparse
import time

import torch

from stdev.flux_distill.small_config import SMALL_MODEL
from stdev.flux_distill.student import FluxStudentTransformer, seeded_noise
from stdev.flux_pretrain.model import LATENT_CHANNELS, to_residual


def measure(model: FluxStudentTransformer, optimizer: torch.optim.Optimizer, batch_size: int, side: int, seconds: float) -> tuple[float, float]:
    device = torch.device("cuda")
    target = side // 8
    latents_source = torch.randn(batch_size, LATENT_CHANNELS, target // 2, target // 2, device=device)
    latents_generated = torch.randn(batch_size, LATENT_CHANNELS, target, target, device=device)
    hff = torch.rand(batch_size, device=device)
    mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    guidance = torch.ones(batch_size, device=device)
    seeds = torch.arange(batch_size, dtype=torch.int64)

    def step() -> None:
        noise = seeded_noise(seeds, (LATENT_CHANNELS, target, target), device)
        residual = to_residual(latents_generated, latents_source, SMALL_MODEL.residual_std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            velocity = model(noise, latents_source, hff, mask, guidance)
        (velocity.float() - (noise - residual)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        step()
        count += 1
    torch.cuda.synchronize()
    return count * batch_size / (time.perf_counter() - start), torch.cuda.max_memory_allocated() / 2**30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32])
    arguments = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 64
    model = FluxStudentTransformer(SMALL_MODEL).cuda().train()
    print(f"{sum(parameter.numel() for parameter in model.parameters()) / 1e6:.2f}M parameters", flush=True)
    model.compile_blocks()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    for side in (1024, 512):
        for batch_size in arguments.sizes:
            size = batch_size if side == 1024 else 4 * batch_size
            try:
                rate, memory = measure(model, optimizer, size, side, arguments.seconds)
                print(f"{side} px, batch {size}: {rate:.1f} samples/s ({rate * (side / 1024) ** 2:.1f} units/s), peak {memory:.1f} GB", flush=True)
            except torch.OutOfMemoryError:
                print(f"{side} px, batch {size}: OOM", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
