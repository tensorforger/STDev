"""
Splits the training step time of a run into GPU compute (synthetic batches already on the GPU)
and data loading (iterating the run's DataLoader alone).

Usage: python dev/training_step_timing.py RUN [--steps 60] [--override NAME=EXPR ...]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tools import load_run_module, resolve_run_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--override", action="append", default=[])
    arguments = parser.parse_args()
    module = load_run_module(resolve_run_file(arguments.run), arguments.override)

    batch_size = module.BATCH_SIZE
    model = module.UpscalerUNet().cuda()
    compiled_model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    latents_full = torch.randn(batch_size, 32, 128, 128, device="cuda")
    latents_low = torch.randn(batch_size, 32, 64, 64, device="cuda")
    prompt_embeds = torch.randn(batch_size, 77, 768, device="cuda")
    uses_text = "prompt_embeds" in module.UpscalerUNet.forward.__code__.co_varnames

    def step() -> None:
        sigmas = torch.rand(batch_size, device="cuda")
        noise = torch.randn_like(latents_full)
        noisy = (1 - sigmas[:, None, None, None]) * latents_full + sigmas[:, None, None, None] * noise
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs = (noisy, sigmas, latents_low) + ((prompt_embeds,) if uses_text else ())
            velocity = compiled_model(*inputs)
        loss = functional.mse_loss(velocity.float(), noise - latents_full)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(arguments.steps):
        step()
    torch.cuda.synchronize()
    print(f"GPU step {(time.perf_counter() - start) / arguments.steps * 1000:.1f} ms")

    data_loader = DataLoader(
        module.LatentPairDataset(module.DATASET_DIRECTORY),
        batch_size=batch_size,
        shuffle=True,
        num_workers=module.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    iterator = iter(data_loader)
    for _ in range(10):
        next(iterator)
    start = time.perf_counter()
    for _ in range(arguments.steps):
        next(iterator)
    print(f"data batch {(time.perf_counter() - start) / arguments.steps * 1000:.1f} ms")


if __name__ == "__main__":
    main()
