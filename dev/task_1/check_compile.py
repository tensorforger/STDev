import sys
import tempfile
from pathlib import Path

import torch

from stdev.benchmark.efficiency import measure_efficiency
from stdev.upscalers.baseline import BaselineUpscaler

upscaler = BaselineUpscaler()
upscaler.prepare_to_benchmark()
upscaler.compile()

for resolution in (512, 1024):
    torch._dynamo.reset()
    results = measure_efficiency(
        upscaler=upscaler,
        resolution=resolution,
        latent_channels=32,
        latent_size=resolution // 8,
        scale_factor=2,
        dtype=torch.bfloat16,
        prompt="test",
        warmup_iterations=3,
        benchmark_iterations=10,
        profile_directory=Path(tempfile.mkdtemp()),
        seed=0,
    )
    print(results, file=sys.stderr)
