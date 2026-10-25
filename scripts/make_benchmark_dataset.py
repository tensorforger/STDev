import shutil
from pathlib import Path

import cv2

from stdev.utils.image_tools import crop_maximal_rectangle

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = REPOSITORY_ROOT / "datasets" / "axis-v1_1k"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "axis_1024"
IMAGE_SIZE = 1024


def main() -> None:
    images_directory = OUTPUT_DIRECTORY / "images"
    prompts_directory = OUTPUT_DIRECTORY / "prompts"
    images_directory.mkdir(parents=True, exist_ok=True)
    prompts_directory.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(SOURCE_DIRECTORY.glob("*/*.jpg"))
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        image = crop_maximal_rectangle(image, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE)
        cv2.imwrite(str(images_directory / f"{image_path.stem}.png"), image)
        shutil.copyfile(
            image_path.with_suffix(".txt"), prompts_directory / f"{image_path.stem}.txt"
        )

    print(f"Saved {len(image_paths)} samples to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
