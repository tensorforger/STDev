import time
from pathlib import Path

import torch
import vaezip

block_paths = sorted(Path("train_datasets/flux_1024_5k").glob("*.vz.safetensors"))[100:140]

start_time = time.perf_counter()
for block_path in block_paths:
    encoded = vaezip.load(block_path, device="cpu")
    latents_orig = encoded["latents_orig"].decode(dtype=torch.float32, device="cpu")
    latents_div_2 = encoded["latents_div_2"].decode(dtype=torch.float32, device="cpu")
print("cpu per block seconds", (time.perf_counter() - start_time) / len(block_paths))
print(latents_orig.shape, latents_orig.dtype)
