import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional

from stdev.upscalers.loading import load_upscaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_FILE = REPOSITORY_ROOT / "src" / "stdev" / "upscalers" / "run_0002.py"
RESULTS_FILE = REPOSITORY_ROOT / "dev" / "round_2" / "training_throughput_256.json"
CONFIGURATIONS = ((256, 16, True, True),)
WARMUP_STEPS = 5
MEASURED_STEPS = 20


def measure_configuration(
    run_module: object,
    base_channels: int,
    batch_size: int,
    use_bfloat16: bool,
    use_compile: bool,
) -> dict[str, float | int | bool | str]:
    """Measure synchronous optimizer-step throughput for a synthetic fixed-shape batch."""
    run_module.BASE_CHANNELS = base_channels
    model = run_module.ConditionalUNet().to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    latents_full = torch.randn(batch_size, 32, 128, 128, device="cuda")
    condition = torch.randn_like(latents_full)
    time_values = torch.sigmoid(torch.randn(batch_size, device="cuda"))
    noise = torch.randn_like(latents_full)
    blend = time_values[:, None, None, None]
    noisy_latents = (1.0 - blend) * latents_full + blend * noise
    velocity_target = noise - latents_full
    training_model = torch.compile(model) if use_compile else model

    def training_step() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            velocity = training_model(noisy_latents, condition, time_values)
            loss = functional.mse_loss(velocity, velocity_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.detach()

    result: dict[str, float | int | bool | str] = {
        "base_channels": base_channels,
        "batch_size": batch_size,
        "bfloat16": use_bfloat16,
        "compile": use_compile,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    try:
        for _ in range(WARMUP_STEPS):
            training_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        for _ in range(MEASURED_STEPS):
            loss = training_step()
        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - start_time
        result["milliseconds_per_step"] = elapsed_seconds * 1000.0 / MEASURED_STEPS
        result["samples_per_second"] = batch_size * MEASURED_STEPS / elapsed_seconds
        result["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        result["loss"] = loss.item()
    except torch.cuda.OutOfMemoryError:
        result["error"] = "out_of_memory"
    finally:
        del training_model, model, optimizer
        del latents_full, condition, time_values, noise, blend, noisy_latents, velocity_target
        gc.collect()
        torch.cuda.empty_cache()
    return result


def main() -> None:
    torch.manual_seed(0)
    upscaler = load_upscaler(RUN_FILE)
    del upscaler
    run_module = sys.modules["run_0002"]
    results = []
    for base_channels, batch_size, use_bfloat16, use_compile in CONFIGURATIONS:
        result = measure_configuration(
            run_module, base_channels, batch_size, use_bfloat16, use_compile
        )
        results.append(result)
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
