"""
Throughput of the FLUX.2 pretrain encoder step (target 1024 + source 512 encode + hff) eager vs compiled.
Usage: CUDA_VISIBLE_DEVICES=7 python scripts/flux_tools/bench_flux_encoder.py
"""

import time

import torch

from stdev.flux_pretrain.config import VAE_DIRECTORY
from stdev.hff.immerkaer import estimate_immerkaer
from stdev.vaes.flux_2_vae import Flux2VAE


def measure(vae: Flux2VAE, batch_size: int, repeats: int = 6) -> float:
    targets = torch.rand(batch_size, 3, 1024, 1024, device="cuda")
    sources = torch.rand(batch_size, 3, 512, 512, device="cuda")
    with torch.no_grad():
        for _ in range(2):
            vae.encode(targets)
            vae.encode(sources)
            estimate_immerkaer(targets)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            vae.encode(targets)
            vae.encode(sources)
            estimate_immerkaer(targets)
        torch.cuda.synchronize()
    return repeats * batch_size / (time.perf_counter() - start)


def main() -> None:
    torch.backends.cudnn.benchmark = True
    vae = Flux2VAE(str(VAE_DIRECTORY), device="cuda", dtype=torch.bfloat16)
    for batch_size in (4, 8):
        print(f"eager batch {batch_size}: {measure(vae, batch_size):.1f} samples/s", flush=True)
    vae.model.encoder.compile(dynamic=False)
    for batch_size in (4, 8):
        print(f"compiled batch {batch_size}: {measure(vae, batch_size):.1f} samples/s", flush=True)
    vae.model.encoder.to(memory_format=torch.channels_last)
    print(f"compiled channels_last batch 8: {measure(vae, 8):.1f} samples/s", flush=True)
    print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
