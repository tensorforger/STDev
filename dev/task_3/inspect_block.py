import inspect
import time
from pathlib import Path

import torch
import vaezip

print([name for name in dir(vaezip) if not name.startswith("_")])
for function_name in ("load", "load_tensor", "decode", "save_tensor"):
    function = getattr(vaezip, function_name, None)
    if function is not None:
        print(function_name, inspect.signature(function))
        print(inspect.getdoc(function))
        print()

block_path = Path("train_datasets/flux_1024_5k/block_000001.vz.safetensors")
start_time = time.perf_counter()
loaded = vaezip.load(block_path, device="cuda")
torch.cuda.synchronize()
print("load seconds", time.perf_counter() - start_time)
for name, value in loaded.items():
    print(name, type(value), getattr(value, "shape", None), getattr(value, "dtype", None))
    if isinstance(value, torch.Tensor):
        print("  mean", value.float().mean().item(), "std", value.float().std().item())
