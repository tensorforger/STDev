"""
Pixel re-encode base for the autoresearch_4 training set: base = E(bicubic 2x(D(latents_512))), the deterministic,
on-manifold part of the 512 -> 1024 Qwen-Image 2.1 latent upscale (decoded with the texture-fix decoder, the
benchmark decoder; encoded with the official encoder, both bf16).

Reads `train_datasets/qwen_4klsdb_40k` and writes `train_datasets/qwen_4klsdb_40k_pixel_base/block_<index>.vz.safetensors`
(same indices) with `latents_1024_base` [1, 64, 64, 64] (fp16).

Run (4 GPUs, resumable): torchrun --standalone --nproc_per_node=4 scripts/encode_qwen_pixel_base.py
"""

import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
import vaezip
from tqdm import tqdm

from stdev.vaes.qwen_image_2_1_vae import QwenImage21VAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "qwen_4klsdb_40k"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "qwen_4klsdb_40k_pixel_base"
DECODER_PATH = REPOSITORY_ROOT / "models" / "texture-fix-vae-for-qwen-image-2.1"
ENCODER_PATH = REPOSITORY_ROOT / "models" / "Qwen-Image-2.1" / "vae"
BATCH_SIZE = 8
LATENT_CODEC = "cast_fp16"

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}"


@torch.no_grad()
def pixel_base(decoder: QwenImage21VAE, encoder: QwenImage21VAE, latents_source: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents_source: [B, 64, h, w], normalized
    returns:
        latents_base: [B, 64, 2h, 2w], normalized latents of the bicubic 2x upscaled decoded source, dtype: torch.float32
    """
    pixels = decoder.decode(latents_source.unsqueeze(2))[:, :3].clamp(0, 1)  # [B, 3, 16h, 16w]
    upscaled = functional.interpolate(pixels, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
    rgba = torch.cat([upscaled, torch.ones_like(upscaled[:, :1])], dim=1)
    return encoder.encode(rgba)[:, :, 0]


def main() -> None:
    torch.set_num_threads(4)
    if WORLD_SIZE > 1:
        distributed.init_process_group("nccl", timeout=timedelta(hours=2), device_id=torch.device(DEVICE))
    torch.cuda.set_device(DEVICE)
    decoder = QwenImage21VAE(str(DECODER_PATH), device=DEVICE, dtype=torch.bfloat16)
    encoder = QwenImage21VAE(str(ENCODER_PATH), device=DEVICE, dtype=torch.bfloat16)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(SOURCE_DIRECTORY.glob("block_*.vz.safetensors"))[RANK::WORLD_SIZE]
    pending = [path for path in source_paths if not (OUTPUT_DIRECTORY / path.name).exists()]
    for start in tqdm(range(0, len(pending), BATCH_SIZE), desc=f"pixel base [rank {RANK}]", position=RANK):
        batch_paths = pending[start : start + BATCH_SIZE]
        latents_source = torch.cat(
            [vaezip.load(path)["latents_512"].decode(dtype=torch.float32) for path in batch_paths]
        ).to(DEVICE)
        bases = pixel_base(decoder, encoder, latents_source).cpu()
        for path, base in zip(batch_paths, bases):
            temporary_path = OUTPUT_DIRECTORY / (path.name + ".tmp")
            vaezip.save({"latents_1024_base": vaezip.encode(base[None].float(), codec=LATENT_CODEC)}, temporary_path)
            temporary_path.rename(OUTPUT_DIRECTORY / path.name)
    if WORLD_SIZE > 1:
        distributed.barrier()
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
