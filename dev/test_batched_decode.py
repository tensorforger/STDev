"""
Checks that per-sample encoded activations stacked into one EncodedTensor decode on GPU to the same
values as per-sample decoding, and times it.
"""

import time
from pathlib import Path

import torch
import vaezip
from safetensors import safe_open

DATASET_DIRECTORY = Path(__file__).resolve().parents[1] / "train_datasets" / "flux_1024_20k"
KEY = "activations_grid"
BATCH_SIZE = 16


def read_raw(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(str(path), framework="pt") as handle:
        names = [name for name in handle.keys() if name.startswith(KEY + ".")]
        tensors = {name[len(KEY) + 1 :]: handle.get_tensor(name) for name in names}
        metadata = handle.metadata()
    return tensors, {"codec_config": metadata[f"{KEY}.codec_config"]}


def main() -> None:
    paths = sorted(DATASET_DIRECTORY.glob("block_*.vz.safetensors"))[:BATCH_SIZE]
    start = time.perf_counter()
    raws = [read_raw(path) for path in paths]
    print(f"read {time.perf_counter() - start:.3f}s")
    metadata = raws[0][1]
    stacked = {
        name: torch.cat([raw[0][name] for raw in raws], dim=0).cuda()
        for name in raws[0][0]
        if name != "shape"
    }
    shape = raws[0][0]["shape"].clone()
    shape[0] = BATCH_SIZE
    stacked["shape"] = shape
    encoded = vaezip.EncodedTensor(tensors=stacked, metadata=metadata)
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        batched = encoded.decode(dtype=torch.bfloat16)
        torch.cuda.synchronize()
        print(f"batched decode {time.perf_counter() - start:.4f}s shape {tuple(batched.shape)}")
    reference = torch.cat(
        [vaezip.load(path)[KEY].to("cuda").decode(dtype=torch.bfloat16) for path in paths[:2]]
    )
    print("max abs diff", (batched[:2].float() - reference.float()).abs().max().item())
    print("activation std", batched.float().std().item(), "mean", batched.float().mean().item())
    token_norm = batched.float().norm(dim=1)
    print("token norm mean/std", token_norm.mean().item(), token_norm.std().item())


if __name__ == "__main__":
    main()
