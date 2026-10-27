from pathlib import Path

import torch
import torch.nn.functional as functional

from stdev.upscalers.loading import load_upscaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SECONDS = 150
NUM_CHECK_BLOCKS = 16

upscaler = load_upscaler(REPOSITORY_ROOT / "src" / "stdev" / "upscalers" / "run_0000.py")
module = __import__("sys").modules["run_0000"]
module.TRAINING_TIME_SECONDS = TRAINING_SECONDS
module.OUTPUT_DIRECTORY = REPOSITORY_ROOT / "dev" / "quick_train_output"
upscaler.train(smoke_mode=False)
upscaler.prepare_to_benchmark()

dataset = module.LatentTripletDataset(module.DATASET_DIRECTORY)
upscaled_errors = []
bilinear_errors = []
for index in range(NUM_CHECK_BLOCKS):
    _, latents_full, latents_div_2 = (tensor[None].cuda() for tensor in dataset[index])
    prediction = upscaler.upscale(
        latents_div_2.to(torch.bfloat16), torch.zeros(1, 77, 768, device="cuda")
    )
    bilinear = functional.interpolate(latents_div_2, scale_factor=2, mode="bilinear")
    upscaled_errors.append(functional.mse_loss(prediction.float(), latents_full).item())
    bilinear_errors.append(functional.mse_loss(bilinear, latents_full).item())
print("model latent mse", sum(upscaled_errors) / NUM_CHECK_BLOCKS)
print("bilinear latent mse", sum(bilinear_errors) / NUM_CHECK_BLOCKS)
