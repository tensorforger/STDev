"""
High frequency attributes of an image (see notebooks/high_frequency_attributes.ipynb), measured on
luminance, and their quantile normalization over the training distribution.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

IMAGE_SIZE = 1024
EDGE_BLUR_SIGMA = 1.5
CANNY_LOW_THRESHOLD = 40
CANNY_HIGH_THRESHOLD = 100
EDGE_CONTRAST_WINDOW = 15
MINIMAL_EDGE_CONTRAST = 0.15
EDGE_EXCLUSION_RADIUS = 4
FLAT_PIXEL_FRACTION = 0.2
FINE_BAND_SIGMA = 1.0
COARSE_BAND_SIGMA = 2.0
NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
METRIC_NAMES = ("noise_level", "edge_sharpness", "texture_detail", "detail_scale")


def to_luminance(image: np.ndarray) -> np.ndarray:
    """
    args:
        image: [H, W, 3], channel_order: BGR, pixel_values: [0...255], dtype: uint8
    returns:
        luminance: [H, W], pixel_values: [0...1], dtype: float32
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def gaussian_blur(luminance: np.ndarray, sigma: float) -> np.ndarray:
    """
    args:
        luminance: [H, W], dtype: float32
    returns:
        blurred: [H, W], dtype: float32
    """
    return cv2.GaussianBlur(luminance, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def band_pass_decomposition(luminance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Two finest Laplacian-pyramid-like bands at full resolution.

    args:
        luminance: [H, W], dtype: float32
    returns:
        fine_band: [H, W], dtype: float32
        coarse_band: [H, W], dtype: float32
    """
    fine_blurred = gaussian_blur(luminance, FINE_BAND_SIGMA)
    coarse_blurred = gaussian_blur(luminance, COARSE_BAND_SIGMA)
    return luminance - fine_blurred, fine_blurred - coarse_blurred


def compute_white_noise_band_gains() -> tuple[float, float]:
    """
    Variance gains of the band-pass filters for unit-variance white noise.

    returns:
        fine_gain: variance of fine band per unit noise variance
        coarse_gain: variance of coarse band per unit noise variance
    """
    random_generator = np.random.default_rng(0)
    white_noise = random_generator.standard_normal((IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    fine_band, coarse_band = band_pass_decomposition(white_noise)
    return float(fine_band.var()), float(coarse_band.var())


FINE_NOISE_GAIN, COARSE_NOISE_GAIN = compute_white_noise_band_gains()


def detect_edges(luminance: np.ndarray) -> np.ndarray:
    """
    args:
        luminance: [H, W], pixel_values: [0...1], dtype: float32
    returns:
        edges: [H, W], dtype: bool
    """
    blurred = gaussian_blur(luminance, EDGE_BLUR_SIGMA)
    blurred_uint8 = np.clip(blurred * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.Canny(blurred_uint8, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD) > 0


def estimate_noise_level(luminance: np.ndarray) -> float:
    """
    Immerkaer noise sigma estimated on the flattest pixels.

    args:
        luminance: [H, W], pixel_values: [0...1], dtype: float32
    returns:
        noise_sigma: noise standard deviation in 8-bit pixel levels
    """
    noise_response = np.abs(cv2.filter2D(luminance, -1, NOISE_KERNEL))[1:-1, 1:-1]
    blurred = gaussian_blur(luminance, EDGE_BLUR_SIGMA)
    gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    structure_magnitude = np.hypot(gradient_x, gradient_y)[1:-1, 1:-1]
    flatness_threshold = np.quantile(structure_magnitude, FLAT_PIXEL_FRACTION)
    flat_responses = noise_response[structure_magnitude <= flatness_threshold]
    kernel_norm = math.sqrt(float((NOISE_KERNEL**2).sum()))
    return float(flat_responses.mean() * math.sqrt(math.pi / 2.0) / kernel_norm * 255.0)


def estimate_edge_sharpness(luminance: np.ndarray, edges: np.ndarray) -> float:
    """
    Median ratio of peak gradient to local edge contrast on strong edges, inverse edge width.

    args:
        luminance: [H, W], pixel_values: [0...1], dtype: float32
        edges: [H, W], dtype: bool
    returns:
        edge_sharpness: [0...~1], 1 / pixels
    """
    gradient_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gradient_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    peak_gradient = cv2.dilate(np.hypot(gradient_x, gradient_y), np.ones((3, 3), np.uint8))
    contrast_kernel = np.ones((EDGE_CONTRAST_WINDOW, EDGE_CONTRAST_WINDOW), np.uint8)
    smoothed = gaussian_blur(luminance, 0.7)
    local_contrast = cv2.dilate(smoothed, contrast_kernel) - cv2.erode(smoothed, contrast_kernel)
    strong_edges = edges & (local_contrast >= MINIMAL_EDGE_CONTRAST)
    if strong_edges.sum() < 100:
        return float("nan")
    return float(np.median(peak_gradient[strong_edges] / local_contrast[strong_edges]))


def estimate_texture(
    luminance: np.ndarray, edges: np.ndarray, noise_sigma: float
) -> tuple[float, float]:
    """
    Noise-compensated band-pass energy outside of edges and its fine-band fraction.

    args:
        luminance: [H, W], pixel_values: [0...1], dtype: float32
        edges: [H, W], dtype: bool
        noise_sigma: noise standard deviation in 8-bit pixel levels
    returns:
        texture_detail: RMS of noise-free band-pass signal in 8-bit pixel levels
        detail_scale: [0...1], fraction of texture energy in the finest band
    """
    exclusion_size = 2 * EDGE_EXCLUSION_RADIUS + 1
    exclusion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (exclusion_size, exclusion_size)
    )
    non_edge = cv2.dilate(edges.astype(np.uint8), exclusion_kernel) == 0
    fine_band, coarse_band = band_pass_decomposition(luminance * 255.0)
    noise_variance = noise_sigma**2
    fine_energy = max(0.0, float((fine_band[non_edge] ** 2).mean()) - FINE_NOISE_GAIN * noise_variance)
    coarse_energy = max(
        0.0, float((coarse_band[non_edge] ** 2).mean()) - COARSE_NOISE_GAIN * noise_variance
    )
    total_energy = fine_energy + coarse_energy
    detail_scale = fine_energy / total_energy if total_energy > 1e-8 else float("nan")
    return math.sqrt(total_energy), detail_scale


def compute_metrics(image: np.ndarray) -> dict[str, float]:
    """
    args:
        image: [H, W, 3], channel_order: BGR, pixel_values: [0...255], dtype: uint8
    returns:
        metrics: raw values of METRIC_NAMES
    """
    luminance = to_luminance(image)
    edges = detect_edges(luminance)
    noise_level = estimate_noise_level(luminance)
    edge_sharpness = estimate_edge_sharpness(luminance, edges)
    texture_detail, detail_scale = estimate_texture(luminance, edges, noise_level)
    return {
        "noise_level": noise_level,
        "edge_sharpness": edge_sharpness,
        "texture_detail": texture_detail,
        "detail_scale": detail_scale,
    }


def compute_feature_vector(image: np.ndarray) -> np.ndarray:
    """
    args:
        image: [H, W, 3], channel_order: BGR, pixel_values: [0...255], dtype: uint8
    returns:
        features: [4], raw values in METRIC_NAMES order, NaN where undefined, dtype: float32
    """
    metrics = compute_metrics(image)
    return np.array([metrics[name] for name in METRIC_NAMES], dtype=np.float32)


class QuantileNormalizer:
    """
    Maps raw feature values to empirical CDF quantiles [0...1] of a reference sample; 0.5 is the
    median image. NaN stays NaN.
    """

    def __init__(self, quantiles: np.ndarray) -> None:
        """
        args:
            quantiles: [4, Q], increasing reference values per feature at levels linspace(0, 1, Q)
        """
        self.quantiles = quantiles.astype(np.float64)
        self.levels = np.linspace(0.0, 1.0, quantiles.shape[1])

    @classmethod
    def fit(cls, raw_features: np.ndarray, quantile_count: int) -> "QuantileNormalizer":
        """
        args:
            raw_features: [N, 4], may contain NaN
        """
        levels = np.linspace(0.0, 1.0, quantile_count)
        quantiles = np.stack(
            [
                np.quantile(column[~np.isnan(column)], levels)
                for column in raw_features.astype(np.float64).T
            ]
        )
        return cls(quantiles)

    def normalize(self, raw_features: np.ndarray) -> np.ndarray:
        """
        args:
            raw_features: [N, 4], may contain NaN
        returns:
            normalized: [N, 4], values: [0...1] or NaN, dtype: float32
        """
        normalized = np.full(raw_features.shape, np.nan, dtype=np.float32)
        for index in range(raw_features.shape[1]):
            column = raw_features[:, index]
            valid = ~np.isnan(column)
            reference = self.quantiles[index]
            lower = np.interp(column[valid], reference, self.levels, left=0.0, right=1.0)
            upper = np.interp(
                -column[valid], -reference[::-1], self.levels[::-1], left=1.0, right=0.0
            )
            normalized[valid, index] = 0.5 * (lower + upper)
        return normalized

    def save(self, path: Path, sample_count: int) -> None:
        path.write_text(
            json.dumps(
                {
                    "metric_names": list(METRIC_NAMES),
                    "sample_count": sample_count,
                    "quantiles": self.quantiles.tolist(),
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "QuantileNormalizer":
        return cls(np.array(json.loads(path.read_text())["quantiles"]))
