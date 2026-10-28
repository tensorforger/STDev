"""
Checks whether horizontally flipping latents matches encoding the flipped image, i.e. whether flip
augmentation in latent space stays on the data distribution.
"""

from pathlib import Path

import torch
import vaezip

from stdev.vaes.flux_2_vae import Flux2VAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "flux_1024_20k"
NUM_SAMPLES = 16


@torch.no_grad()
def main() -> None:
    vae = Flux2VAE(
        str(REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B" / "vae"), device="cuda", dtype=torch.bfloat16
    )
    paths = sorted(DATASET_DIRECTORY.glob("block_*.vz.safetensors"))[:NUM_SAMPLES]
    flip_errors, reencode_errors, per_channel = [], [], torch.zeros(32, device="cuda")
    sources, targets = [], []
    for path in paths:
        latents = vaezip.load(path)["latents_orig"].decode(dtype=torch.float32).cuda()
        image = vae.decode(latents).float().clamp(0, 1)
        torch.manual_seed(0)
        reencoded = vae.encode(image).float()
        torch.manual_seed(0)
        flipped_encoded = vae.encode(image.flip(-1)).float()
        flipped_latents = reencoded.flip(-1)
        flip_errors.append((flipped_encoded - flipped_latents).pow(2).mean().item())
        reencode_errors.append((reencoded - latents).pow(2).mean().item())
        per_channel += (flipped_encoded - flipped_latents).pow(2).mean(dim=(0, 2, 3)) / len(paths)
        sources.append(flipped_latents[0].flatten(1).T)
        targets.append(flipped_encoded[0].flatten(1).T)
    half = len(paths) // 2
    fit_source, fit_target = torch.cat(sources[:half]), torch.cat(targets[:half])
    channel_map = torch.linalg.lstsq(fit_source, fit_target).solution  # [32, 32]
    test_source, test_target = torch.cat(sources[half:]), torch.cat(targets[half:])
    print("held-out flip mse with linear channel map", (test_source @ channel_map - test_target).pow(2).mean().item())
    print("held-out flip mse without map", (test_source - test_target).pow(2).mean().item())
    print("channel map diagonal", torch.diagonal(channel_map).cpu().numpy().round(2))
    print("latent variance ~1.0")
    print("reencode mse (decode->encode vs stored)", sum(reencode_errors) / len(paths))
    print("flip mse (encode(flip) vs flip(encode))", sum(flip_errors) / len(paths))
    print("per channel flip mse", per_channel.cpu().numpy().round(3))


if __name__ == "__main__":
    main()
