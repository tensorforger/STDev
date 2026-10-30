from pathlib import Path

import cv2
import numpy as np

from high_frequency_metrics import IMAGE_SIZE, compute_metrics
from stdev.utils.image_tools import crop_maximal_rectangle

DATASET_DIRECTORY = Path(__file__).resolve().parents[1] / "datasets" / "axis-v1_1k"


def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    args:
        image: [H, W, 3], pixel_values: [0...255], dtype: uint8
    returns:
        noisy_image: [H, W, 3], pixel_values: [0...255], dtype: uint8
    """
    random_generator = np.random.default_rng(0)
    noise = random_generator.normal(0.0, sigma, image.shape[:2])[..., None]
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def format_metrics(metrics: dict[str, float]) -> str:
    return "  ".join(f"{name}={value:7.3f}" for name, value in metrics.items())


image_paths = sorted(DATASET_DIRECTORY.glob("*/*.jpg"))[::125]
for image_path in image_paths:
    image = crop_maximal_rectangle(cv2.imread(str(image_path)), IMAGE_SIZE, IMAGE_SIZE)
    print(image_path.name)
    print("  original      ", format_metrics(compute_metrics(image)))
    for sigma in (2.0, 5.0, 10.0):
        noisy_image = add_gaussian_noise(image, sigma)
        print(f"  noise {sigma:4.1f}    ", format_metrics(compute_metrics(noisy_image)))
    for sigma in (1.0, 2.0):
        blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
        print(f"  blur {sigma:4.1f}     ", format_metrics(compute_metrics(blurred_image)))
