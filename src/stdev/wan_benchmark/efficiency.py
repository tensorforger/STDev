import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity
from tqdm import tqdm

from stdev.benchmark.efficiency import (
    _extract_memory_statistics,
    _save_memory_timeline_image,
    _save_time_profile_artifacts,
)
from stdev.wan_upscalers.base import Upscaler


@dataclass(frozen=True)
class EfficiencyScenario:
    """
    Latent sizes (T, H, W) of the input and of the target.
    """

    name: str
    input_size: tuple[int, int, int]
    output_size: tuple[int, int, int]


EFFICIENCY_SCENARIOS = (
    EfficiencyScenario("image_1024", (1, 64, 64), (1, 128, 128)),
    EfficiencyScenario("video_512", (5, 64, 64), (9, 64, 64)),
)


def _make_inputs(
    scenario: EfficiencyScenario, latent_channels: int, dtype: torch.dtype, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    returns:
        latents: [1, C, T_lr, H_lr, W_lr], random, hff: [1], dtype: torch.float32
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    latents = torch.randn(
        1, latent_channels, *scenario.input_size, generator=generator, device="cuda", dtype=dtype
    )
    return latents, torch.full((1,), 0.5, device="cuda")


def measure_latency(
    upscaler: Upscaler,
    scenario: EfficiencyScenario,
    latent_channels: int,
    dtype: torch.dtype,
    warmup_iterations: int,
    benchmark_iterations: int,
    profile_directory: Path,
    seed: int,
) -> dict[str, float]:
    """
    Measures the latency of a single model forward pass over timed passes, then saves the CPU and CUDA
    time profile of one extra pass.
    """
    latents, hff = _make_inputs(scenario, latent_channels, dtype, seed)

    with torch.no_grad():
        for _ in tqdm(range(warmup_iterations), desc=f"Warmup {scenario.name}", leave=False):
            upscaler.single_forward(latents, hff, scenario.output_size)
        torch.cuda.synchronize()

        latencies_ms: list[float] = []
        for _ in tqdm(range(benchmark_iterations), desc=f"Latency {scenario.name}", leave=False):
            start_time = time.perf_counter()
            upscaler.single_forward(latents, hff, scenario.output_size)
            torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start_time) * 1000)

        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as profiler:
            upscaler.single_forward(latents, hff, scenario.output_size)
            torch.cuda.synchronize()

    _save_time_profile_artifacts(profiler, profile_directory)
    return {
        f"latency_ms_avg_{scenario.name}": float(np.mean(latencies_ms)),
        f"latency_ms_peak_{scenario.name}": float(np.max(latencies_ms)),
    }


def measure_memory(
    upscaler: Upscaler,
    scenario: EfficiencyScenario,
    latent_channels: int,
    dtype: torch.dtype,
    warmup_iterations: int,
    profile_directory: Path,
    seed: int,
) -> dict[str, float]:
    """
    Measures allocated CUDA memory over one profiled `upscale`, including the model weights.
    Must be called on the not compiled upscaler: the profiler cannot attribute memory of compiled graphs.
    """
    latents, hff = _make_inputs(scenario, latent_channels, dtype, seed)

    with torch.no_grad():
        for _ in range(warmup_iterations):
            upscaler.upscale(latents, hff, scenario.output_size)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        ) as profiler:
            upscaler.upscale(latents, hff, scenario.output_size)
            torch.cuda.synchronize()

    profile_directory.mkdir(parents=True, exist_ok=True)
    _save_memory_timeline_image(profiler, profile_directory / "memory_timeline.png")
    peak_memory_mb, average_memory_mb = _extract_memory_statistics(profiler)
    return {
        f"cuda_mem_avg_{scenario.name}": average_memory_mb,
        f"cuda_mem_peak_{scenario.name}": peak_memory_mb,
    }
