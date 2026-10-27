import glob, torch, vaezip
for path in sorted(glob.glob("train_datasets/flux_1024_1k/block_*.vz.safetensors"))[:8]:
    block = vaezip.load(path)
    for name in ("init_latents", "latents_orig", "latents_div_2"):
        tensor = block[name].decode(dtype=torch.float32)[0]
        print(name, tuple(tensor.shape), "mean %.3f std %.3f" % (tensor.mean().item(), tensor.std().item()), "per-channel std range %.2f..%.2f" % (tensor.std(dim=(1, 2)).min().item(), tensor.std(dim=(1, 2)).max().item()))
    break
