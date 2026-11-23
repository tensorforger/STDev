"""
Source image construction of the FLUX.2 pretraining: 2x downscale of the target image with the benchmark filter
(cv2 bilinear) or a corrupted variant (other filters, bilinear + JPEG). Same filters as autoresearch_3
(`scripts/encode_flux2_datasets.py`), with a variable JPEG quality.
"""

import io
import random

import cv2
import numpy as np
from PIL import Image

from stdev.flux_pretrain.config import (
    JPEG_QUALITY_MAX,
    JPEG_QUALITY_MIN,
    MIN_CROP_FRACTION,
    SOURCE_VARIANT_WEIGHTS,
    SOURCE_VARIANTS,
)


def draw_variant(generator: random.Random) -> str:
    return generator.choices(SOURCE_VARIANTS, weights=[SOURCE_VARIANT_WEIGHTS[name] for name in SOURCE_VARIANTS])[0]


def draw_jpeg_quality(generator: random.Random) -> int:
    """
    Quality skewed towards JPEG_QUALITY_MAX (almost invisible); JPEG_QUALITY_MIN is the rare extreme.
    """
    return round(JPEG_QUALITY_MAX - (JPEG_QUALITY_MAX - JPEG_QUALITY_MIN) * generator.random() ** 2)


def large_square_crop(image: np.ndarray, side: int, generator: random.Random) -> np.ndarray:
    """
    Random square crop, side = max(side, fraction * short side) with fraction in [MIN_CROP_FRACTION, 1] skewed
    towards 1 (close to the full image), area resized to side.

    args:
        image: [H, W, 3], dtype: uint8, short side >= side
    returns:
        image: [side, side, 3], dtype: uint8
    """
    height, width = image.shape[:2]
    short_side = min(height, width)
    fraction = 1.0 - (1.0 - MIN_CROP_FRACTION) * generator.random() ** 2
    crop_side = min(short_side, max(side, round(fraction * short_side)))
    top = generator.randint(0, height - crop_side)
    left = generator.randint(0, width - crop_side)
    crop = image[top : top + crop_side, left : left + crop_side]
    if crop_side == side:
        return np.ascontiguousarray(crop)
    return cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)


def downscale_variant(image: np.ndarray, variant: str, jpeg_quality: int) -> np.ndarray:
    """
    2x downscale of a uint8 image with one of SOURCE_VARIANTS.

    args:
        image: [S, S, 3], channel_order: RGB, dtype: uint8
    returns:
        image: [S / 2, S / 2, 3], channel_order: RGB, dtype: uint8
    """
    size = (image.shape[1] // 2, image.shape[0] // 2)
    if variant == "bilinear":
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if variant == "nearest":
        return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    if variant == "bicubic":
        return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    if variant == "lanczos":
        return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
    if variant == "antialias":
        return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BILINEAR))
    if variant == "jpeg":
        bilinear = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        buffer = io.BytesIO()
        Image.fromarray(bilinear).save(buffer, format="JPEG", quality=jpeg_quality)
        return np.asarray(Image.open(buffer).convert("RGB"))
    raise ValueError(f"Unknown variant {variant}")
