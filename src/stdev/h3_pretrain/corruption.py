"""
Sample construction of the translator pretraining: a large random square crop, the clean target (INTER_AREA resize to
the bucket side, encoded by the FLUX.2 VAE) and the H3 encoder input, which is either the target itself or a corrupted
variant (another resize filter from the crop, or JPEG of the target).
"""

import io
import random

import cv2
import numpy as np
from PIL import Image

from stdev.h3_pretrain.config import (
    JPEG_QUALITY_MAX,
    JPEG_QUALITY_MIN,
    MIN_CROP_FRACTION,
    SOURCE_VARIANT_WEIGHTS,
    SOURCE_VARIANTS,
)

RESIZE_VARIANTS = ("nearest", "antialias", "bicubic", "lanczos")


def draw_variant(generator: random.Random) -> str:
    return generator.choices(SOURCE_VARIANTS, weights=[SOURCE_VARIANT_WEIGHTS[name] for name in SOURCE_VARIANTS])[0]


def draw_jpeg_quality(generator: random.Random) -> int:
    """
    Quality skewed towards JPEG_QUALITY_MAX (almost invisible); JPEG_QUALITY_MIN is the rare extreme.
    """
    return round(JPEG_QUALITY_MAX - (JPEG_QUALITY_MAX - JPEG_QUALITY_MIN) * generator.random() ** 2)


def large_square_crop_region(image: np.ndarray, side: int, generator: random.Random) -> np.ndarray:
    """
    Random square crop without resizing, crop side = max(side, fraction * short side) with fraction in
    [MIN_CROP_FRACTION, 1] skewed towards 1.

    args:
        image: [H, W, 3], dtype: uint8, short side >= side
    returns:
        crop: [S, S, 3], S >= side, dtype: uint8
    """
    height, width = image.shape[:2]
    short_side = min(height, width)
    fraction = 1.0 - (1.0 - MIN_CROP_FRACTION) * generator.random() ** 2
    crop_side = min(short_side, max(side, round(fraction * short_side)))
    top = generator.randint(0, height - crop_side)
    left = generator.randint(0, width - crop_side)
    return np.ascontiguousarray(image[top : top + crop_side, left : left + crop_side])


def resize_with_variant(crop: np.ndarray, side: int, variant: str) -> np.ndarray:
    """
    args:
        crop: [S, S, 3], channel_order: RGB, dtype: uint8
    returns:
        image: [side, side, 3], channel_order: RGB, dtype: uint8
    """
    size = (side, side)
    if variant == "nearest":
        return cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST)
    if variant == "bicubic":
        return cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
    if variant == "lanczos":
        return np.asarray(Image.fromarray(crop).resize(size, Image.Resampling.LANCZOS))
    if variant == "antialias":
        return np.asarray(Image.fromarray(crop).resize(size, Image.Resampling.BILINEAR))
    raise ValueError(f"Unknown resize variant {variant}")


def compress_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    """
    args:
        image: [S, S, 3], channel_order: RGB, dtype: uint8
    returns:
        image: [S, S, 3], channel_order: RGB, dtype: uint8
    """
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return np.asarray(Image.open(buffer).convert("RGB"))


def make_pair(image: np.ndarray, side: int, generator: random.Random) -> tuple[np.ndarray, np.ndarray, str]:
    """
    args:
        image: [H, W, 3], channel_order: RGB, dtype: uint8, short side >= side
    returns:
        target: [side, side, 3], channel_order: RGB, dtype: uint8, clean
        source: [side, side, 3], channel_order: RGB, dtype: uint8, H3 encoder input
        variant: name in SOURCE_VARIANTS actually applied
    """
    crop = large_square_crop_region(image, side, generator)
    needs_resize = crop.shape[0] != side
    target = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA) if needs_resize else crop
    variant = draw_variant(generator)
    if variant == "jpeg":
        return target, compress_jpeg(target, draw_jpeg_quality(generator)), variant
    if variant in RESIZE_VARIANTS and needs_resize:
        return target, resize_with_variant(crop, side, variant), variant
    return target, target, "clean"
