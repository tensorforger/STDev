"""
Model-only 1-step latency (bf16, compiled, batch 1, 512 -> 1024: 64 -> 128 latents) and peak memory of the large and
the small FLUX.2 students.
Usage: CUDA_VISIBLE_DEVICES=0 python scripts/flux_tools/measure_flux_students.py
"""

import time

import torch

from stdev.flux_distill.small_config import SMALL_MODEL
from stdev.flux_distill.student import FluxStudentTransformer
from stdev.flux_pretrain.config import MODEL
from stdev.flux_pretrain.model import LATENT_CHANNELS

ITERATIONS = 200


def measure(name: str, model: FluxStudentTransformer) -> None:
    model = model.cuda().to(torch.bfloat16).eval().requires_grad_(False)
    model.compile_blocks()
    noise = torch.randn(1, LATENT_CHANNELS, 128, 128, device="cuda", dtype=torch.bfloat16)
    source = torch.randn(1, LATENT_CHANNELS, 64, 64, device="cuda", dtype=torch.bfloat16)
    hff = torch.full((1,), 0.5, device="cuda")
    mask = torch.ones(1, dtype=torch.bool, device="cuda")
    guidance = torch.ones(1, device="cuda")
    with torch.no_grad():
        for _ in range(20):
            model(noise, source, hff, mask, guidance)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            model(noise, source, hff, mask, guidance)
            torch.cuda.synchronize()
            timings.append(1000 * (time.perf_counter() - start))
    timings.sort()
    parameters = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    print(
        f"{name}: {parameters:.1f}M, median {timings[len(timings) // 2]:.2f} ms, p95 {timings[int(0.95 * len(timings))]:.2f} ms, "
        f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB",
        flush=True,
    )


def main() -> None:
    torch.backends.cudnn.benchmark = True
    measure("large", FluxStudentTransformer(MODEL))
    measure("small", FluxStudentTransformer(SMALL_MODEL))


if __name__ == "__main__":
    main()
