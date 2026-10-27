from pathlib import Path

import torch

from stdev.upscalers import run_0000

dev_directory = Path(__file__).resolve().parent / "throughput_artifacts"
run_0000.TRAINING_TIME_SECONDS = 60
run_0000.LOG_INTERVAL_STEPS = 20
run_0000.OUTPUT_DIRECTORY = dev_directory / "model"
run_0000.TENSORBOARD_DIRECTORY = dev_directory / "tensorboard"
run_0000.Run0000Upscaler().train(smoke_mode=False)
print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
