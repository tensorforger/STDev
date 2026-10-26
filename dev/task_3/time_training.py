import sys
from pathlib import Path

import torch

from stdev.upscalers.loading import load_upscaler

run_file = Path("src/stdev/upscalers/run_0000.py")
upscaler = load_upscaler(run_file)
module = sys.modules["run_0000"]
module.TRAINING_TIME_SECONDS = 60
upscaler.train(smoke_mode=False)
print("peak VRAM GB", torch.cuda.max_memory_allocated() / 1024**3)
