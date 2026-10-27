import base64
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity
from tqdm import tqdm
from torch.profiler._memory_profiler import MemoryProfileTimeline

from stdev.upscalers.base import Upscaler

BYTES_PER_MEGABYTE = 1024**2
PROFILE_TABLE_ROW_LIMIT = 20


def _extract_memory_statistics(
    profiler: torch.profiler.profile,
) -> tuple[float, float]:
    """
    returns:
        peak and time-weighted average of allocated CUDA memory during the profiled pass, in MB
    """
    times, sizes = MemoryProfileTimeline(profiler._memory_profile())._coalesce_timeline("cuda:0")
    times = np.array(times, dtype=np.float64)  # [N]
    total_bytes = np.array(sizes, dtype=np.float64).sum(axis=1)  # [N]
    durations = np.diff(times)  # [N - 1]
    if durations.sum() > 0:
        average_bytes = float((total_bytes[:-1] * durations).sum() / durations.sum())
    else:
        average_bytes = float(total_bytes.mean())
    return float(total_bytes.max()) / BYTES_PER_MEGABYTE, average_bytes / BYTES_PER_MEGABYTE


def _save_memory_timeline_image(profiler: torch.profiler.profile, image_path: Path) -> None:
    html_path = image_path.with_suffix(".html")
    profiler.export_memory_timeline(str(html_path), device="cuda:0")
    html = html_path.read_text(encoding="utf-8")
    match = re.search(r'<img[^>]+src=["\']data:image/png;base64,([^"\']+)["\']', html)
    image_path.write_bytes(base64.b64decode(match.group(1)))
    html_path.unlink()


def _save_profile_artifacts(profiler: torch.profiler.profile, profile_directory: Path) -> None:
    profile_directory.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(profile_directory / "trace.json"))
    for sort_key in ("self_cpu_time_total", "self_cuda_time_total"):
        table = profiler.key_averages().table(sort_by=sort_key, row_limit=PROFILE_TABLE_ROW_LIMIT)
        (profile_directory / f"table_{sort_key}.txt").write_text(table)
    _save_memory_timeline_image(profiler, profile_directory / "memory_timeline.png")


def measure_efficiency(
    upscaler: Upscaler,
    resolution: int,
    latent_channels: int,
    latent_size: int,
    scale_factor: int,
    dtype: torch.dtype,
    prompt_embeds: torch.Tensor,
    warmup_iterations: int,
    benchmark_iterations: int,
    profile_directory: Path,
    seed: int,
) -> dict[str, float]:
    """
    Measures latency over timed passes, then CUDA memory over one extra profiled pass.
    The upscaler input is random latents of the low resolution matching the output `resolution`.
    """
    low_resolution_size = latent_size // scale_factor
    generator = torch.Generator(device="cuda").manual_seed(seed)
    latents = torch.randn(
        1,
        latent_channels,
        low_resolution_size,
        low_resolution_size,
        generator=generator,
        device="cuda",
        dtype=dtype,
    )  # [1, C, H_lr, W_lr]

    with torch.no_grad():
        for _ in tqdm(range(warmup_iterations), desc=f"Warmup {resolution}", leave=False):
            upscaler.upscale(latents, prompt_embeds)
        torch.cuda.synchronize()

        latencies_ms: list[float] = []
        for _ in tqdm(range(benchmark_iterations), desc=f"Latency {resolution}", leave=False):
            start_time = time.perf_counter()
            upscaler.upscale(latents, prompt_embeds)
            torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start_time) * 1000)

        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        ) as profiler:
            upscaler.upscale(latents, prompt_embeds)
            torch.cuda.synchronize()

    _save_profile_artifacts(profiler, profile_directory)
    peak_memory_mb, average_memory_mb = _extract_memory_statistics(profiler)

    return {
        f"latency_ms_avg_{resolution}": float(np.mean(latencies_ms)),
        f"latency_ms_peak_{resolution}": float(np.max(latencies_ms)),
        f"cuda_mem_avg_{resolution}": average_memory_mb,
        f"cuda_mem_peak_{resolution}": peak_memory_mb,
    }
