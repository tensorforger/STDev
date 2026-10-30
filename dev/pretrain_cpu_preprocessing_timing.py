"""
Single thread CPU cost per image of the pretraining preprocessing: JPEG decode, square crop and
area resize to the high and low resolution, and high frequency metrics on the high resolution image.

Usage: python dev/pretrain_cpu_preprocessing_timing.py JPEG [JPEG ...] [--repeats 5]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from high_frequency_metrics import compute_metrics

RESOLUTION_PAIRS = ((512, 256), (1024, 512), (2048, 1024))


def center_square_crop(image: np.ndarray) -> np.ndarray:
    """
    args:
        image: [H, W, 3], dtype: uint8
    returns:
        image: [S, S, 3], S = min(H, W), dtype: uint8
    """
    height, width = image.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return image[top : top + side, left : left + side]


def time_call(function, repeats: int) -> float:
    """
    returns:
        mean milliseconds per call
    """
    function()
    start_time = time.perf_counter()
    for _ in range(repeats):
        function()
    return 1000 * (time.perf_counter() - start_time) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    cv2.setNumThreads(1)

    for image_path in arguments.images:
        encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        print(f"{image_path.name} {decoded.shape[1]}x{decoded.shape[0]} {len(encoded) // 1024} KB")
        print(f"  decode full            {time_call(lambda: cv2.imdecode(encoded, cv2.IMREAD_COLOR), arguments.repeats):7.1f} ms")
        print(f"  decode reduced 2x      {time_call(lambda: cv2.imdecode(encoded, cv2.IMREAD_REDUCED_COLOR_2), arguments.repeats):7.1f} ms")
        square = np.ascontiguousarray(center_square_crop(decoded))
        for high_resolution, low_resolution in RESOLUTION_PAIRS:
            if square.shape[0] < high_resolution:
                continue
            resize_ms = time_call(
                lambda: (
                    cv2.resize(square, (high_resolution, high_resolution), interpolation=cv2.INTER_AREA),
                    cv2.resize(square, (low_resolution, low_resolution), interpolation=cv2.INTER_AREA),
                ),
                arguments.repeats,
            )
            high_image = cv2.resize(square, (high_resolution, high_resolution), interpolation=cv2.INTER_AREA)
            metrics_ms = time_call(lambda: compute_metrics(high_image), arguments.repeats)
            print(f"  {high_resolution}/{low_resolution} resize {resize_ms:7.1f} ms   hf metrics {metrics_ms:7.1f} ms")


if __name__ == "__main__":
    main()
