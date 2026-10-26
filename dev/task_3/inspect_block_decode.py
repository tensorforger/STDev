import time
from pathlib import Path

import torch
import vaezip

block_paths = sorted(Path("train_datasets/flux_1024_5k").glob("*.vz.safetensors"))[:20]
print("file size MB", block_paths[0].stat().st_size / 1e6)

for repeat in range(2):
    start_time = time.perf_counter()
    for block_path in block_paths:
        encoded = vaezip.load(block_path, device="cpu")
        latents_orig = encoded["latents_orig"].decode(dtype=torch.bfloat16, device="cuda")
        latents_div_2 = encoded["latents_div_2"].decode(dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    print("per block seconds", (time.perf_counter() - start_time) / len(block_paths))

print(latents_orig.shape, latents_orig.dtype, latents_div_2.shape)
print("orig mean/std", latents_orig.float().mean().item(), latents_orig.float().std().item())
print("div2 mean/std", latents_div_2.float().mean().item(), latents_div_2.float().std().item())
