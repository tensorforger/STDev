"""
Single GPU training throughput (forward, backward, AdamW step, synthetic batches on the GPU) of a
run_0078 based architecture at several high resolution latent sizes, batch sizes and with or
without gradient checkpointing.

Usage: python dev/pretrain_training_throughput.py --name NAME [--override NAME=EXPR ...]
           [--latent-sizes 64 128 256] [--batch-sizes 64 16 4] [--steps 20] [--checkpointing 0 1]
Results are appended to dev/data/pretrain_training_throughput.jsonl.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tools import REPOSITORY_ROOT, load_run_module, resolve_run_file

RESULTS_PATH = REPOSITORY_ROOT / "dev" / "data" / "pretrain_training_throughput.jsonl"


def measure(
    overrides: list[str], latent_size: int, batch_size: int, gradient_checkpointing: bool, steps: int
) -> dict[str, float]:
    """
    returns:
        parameters_m, step_ms, samples_per_second, peak_memory_gb
    """
    module = load_run_module(
        resolve_run_file("0078"),
        overrides + [f"GRADIENT_CHECKPOINTING={gradient_checkpointing}"],
    )
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = module.UpscalerUNet().cuda().train()
    compiled_model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)
    latents_high = torch.randn(batch_size, 32, latent_size, latent_size, device="cuda")
    latents_low = torch.randn(batch_size, 32, latent_size // 2, latent_size // 2, device="cuda")
    prompt_embeds = torch.randn(batch_size, 77, 768, device="cuda")

    def step() -> None:
        sigmas = torch.rand(batch_size, device="cuda")
        noise = torch.randn_like(latents_high)
        noisy = (1 - sigmas[:, None, None, None]) * latents_high + sigmas[:, None, None, None] * noise
        with torch.autocast("cuda", dtype=torch.bfloat16):
            velocity = compiled_model(noisy, sigmas, latents_low, prompt_embeds)
        loss = functional.mse_loss(velocity.float(), noise - latents_high)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for _ in range(4):
        step()
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    step_seconds = (time.perf_counter() - start_time) / steps
    return {
        "parameters_m": sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        "step_ms": 1000 * step_seconds,
        "samples_per_second": batch_size / step_seconds,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--latent-sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 16, 4])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--checkpointing", type=int, nargs="+", default=[0, 1])
    arguments = parser.parse_args()
    torch.backends.cudnn.benchmark = True

    for latent_size, batch_size in zip(arguments.latent_sizes, arguments.batch_sizes):
        for gradient_checkpointing in map(bool, arguments.checkpointing):
            record: dict[str, object] = {
                "name": arguments.name,
                "latent_size": latent_size,
                "image_resolution": latent_size * 8,
                "batch_size": batch_size,
                "gradient_checkpointing": gradient_checkpointing,
            }
            try:
                record.update(
                    measure(
                        arguments.override,
                        latent_size,
                        batch_size,
                        gradient_checkpointing,
                        arguments.steps,
                    )
                )
            except torch.OutOfMemoryError:
                record["error"] = "oom"
            print(json.dumps(record), flush=True)
            with RESULTS_PATH.open("a") as results_file:
                results_file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
