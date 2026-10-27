from pathlib import Path

import cv2
import torch
import vaezip

from stdev.text_encoders.sdxl_text_encoder import SDXLTextEncoder
from stdev.utils.image_tools import crop_maximal_rectangle

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = REPOSITORY_ROOT / "datasets" / "axis-v1_1k"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "axis_1024"
SDXL_DIRECTORY = REPOSITORY_ROOT / "models" / "stable-diffusion-xl-base-1.0"
IMAGE_SIZE = 1024
PROMPT_EMBEDS_CODEC = "cast_fp16"


def main() -> None:
    images_directory = OUTPUT_DIRECTORY / "images"
    prompt_embeds_directory = OUTPUT_DIRECTORY / "prompt_embeds"
    images_directory.mkdir(parents=True, exist_ok=True)
    prompt_embeds_directory.mkdir(parents=True, exist_ok=True)
    text_encoder = SDXLTextEncoder(str(SDXL_DIRECTORY), device="cuda", dtype=torch.float32)

    image_paths = sorted(SOURCE_DIRECTORY.glob("*/*.jpg"))
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        image = crop_maximal_rectangle(image, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE)
        cv2.imwrite(str(images_directory / f"{image_path.stem}.png"), image)

        prompt = image_path.with_suffix(".txt").read_text().strip()
        prompt_embeds = text_encoder.encode_text_batch([prompt]).float()  # [1, 77, 768]
        vaezip.save_tensor(
            prompt_embeds,
            prompt_embeds_directory / f"{image_path.stem}.vz.safetensors",
            codec=PROMPT_EMBEDS_CODEC,
        )

    print(f"Saved {len(image_paths)} samples to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
