"""
Batched Immerkaer fine-scale noise estimator used as the standard high frequency feature (HFF)
conditioning. The raw estimate is mapped to [0...1] through a fixed quantile table calibrated once
on ReLAION images (see dev/calibrate_immerkaer.py), so values are absolute and deterministic.
"""

import math

import torch
import torch.nn.functional as functional

LUMINANCE_WEIGHTS = (0.299, 0.587, 0.114)
IMMERKAER_KERNEL = ((1.0, -2.0, 1.0), (-2.0, 4.0, -2.0), (1.0, -2.0, 1.0))
IMMERKAER_SCALE = math.sqrt(math.pi / 2.0) / 6.0

# Raw Immerkaer values (luminance in [0...1]) at evenly spaced quantile levels 0, 1/(N-1), ..., 1.
IMMERKAER_QUANTILES: tuple[float, ...] = (
    0.00000000,
    0.00020116,
    0.00036722,
    0.00045936,
    0.00058467,
    0.00066617,
    0.00073944,
    0.00081956,
    0.00088596,
    0.00095097,
    0.00101954,
    0.00107751,
    0.00112964,
    0.00118689,
    0.00124444,
    0.00129973,
    0.00135998,
    0.00142074,
    0.00148595,
    0.00155106,
    0.00161290,
    0.00168215,
    0.00174499,
    0.00181354,
    0.00186737,
    0.00193704,
    0.00199897,
    0.00207116,
    0.00212763,
    0.00219403,
    0.00225236,
    0.00232901,
    0.00238967,
    0.00244775,
    0.00252111,
    0.00260088,
    0.00267553,
    0.00276829,
    0.00283433,
    0.00291129,
    0.00298126,
    0.00305011,
    0.00313832,
    0.00321171,
    0.00330140,
    0.00337240,
    0.00345847,
    0.00355506,
    0.00365330,
    0.00375433,
    0.00386084,
    0.00395091,
    0.00405266,
    0.00416922,
    0.00428589,
    0.00439411,
    0.00451155,
    0.00462194,
    0.00477484,
    0.00490455,
    0.00505853,
    0.00519100,
    0.00533937,
    0.00548513,
    0.00564388,
    0.00578612,
    0.00592169,
    0.00607432,
    0.00626565,
    0.00647446,
    0.00669298,
    0.00689549,
    0.00709790,
    0.00729320,
    0.00755613,
    0.00778564,
    0.00800775,
    0.00827138,
    0.00862279,
    0.00891582,
    0.00919540,
    0.00957094,
    0.00995562,
    0.01035687,
    0.01084383,
    0.01131693,
    0.01175147,
    0.01235478,
    0.01285607,
    0.01353223,
    0.01422602,
    0.01504661,
    0.01593244,
    0.01703135,
    0.01859967,
    0.02029203,
    0.02264602,
    0.02552253,
    0.03004623,
    0.03913622,
    0.13334534,
)


def to_luminance(images: torch.Tensor) -> torch.Tensor:
    """
    args:
        images: [N, C, H, W], C in {3, 4}, channel_order: RGB or RGBA (alpha ignored), pixel_values: [0...1]
    returns:
        luminance: [N, 1, H, W], pixel_values: [0...1]
    """
    weights = torch.tensor(LUMINANCE_WEIGHTS, dtype=images.dtype, device=images.device)
    return (images[:, :3] * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def compute_raw_immerkaer(images: torch.Tensor) -> torch.Tensor:
    """
    Raw Immerkaer noise sigma estimate, averaged over frames for videos.

    args:
        images: [B, C, H, W] or [B, C, T, H, W], C in {3, 4}, channel_order: RGB or RGBA (alpha ignored),
            pixel_values: [0...1], dtype: float
    returns:
        raw_values: [B], non-negative, same units as pixel values
    """
    if images.ndim == 5:
        batch_size, _, frame_count = images.shape[:3]
        frames = images.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [B*T, C, H, W]
    elif images.ndim == 4:
        batch_size, frame_count = images.shape[0], 1
        frames = images
    else:
        raise ValueError(f"Expected [B, C, H, W] or [B, C, T, H, W], got {tuple(images.shape)}")
    if images.shape[1] not in (3, 4):
        raise ValueError(f"Expected 3 or 4 channels, got {images.shape[1]}")

    kernel = torch.tensor(IMMERKAER_KERNEL, dtype=frames.dtype, device=frames.device)
    response = functional.conv2d(
        to_luminance(frames), kernel.view(1, 1, 3, 3)
    )  # [B*T, 1, H-2, W-2]
    frame_values = IMMERKAER_SCALE * response.abs().mean(dim=(1, 2, 3))  # [B*T]
    return frame_values.view(batch_size, frame_count).mean(dim=1)


def quantile_normalize(raw_values: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    """
    args:
        raw_values: [B]
        quantiles: sorted raw values at evenly spaced quantile levels from 0 to 1
    returns:
        normalized: [B], pixel_values: [0...1], dtype: same as raw_values
    """
    table = torch.tensor(quantiles, dtype=raw_values.dtype, device=raw_values.device)
    lower_index = (torch.searchsorted(table, raw_values, right=True) - 1).clamp(
        0, table.numel() - 2
    )
    lower = table[lower_index]
    upper = table[lower_index + 1]
    fraction = (raw_values - lower) / (upper - lower).clamp_min(1e-12)
    return ((lower_index + fraction) / (table.numel() - 1)).clamp(0.0, 1.0)


def estimate_immerkaer(images: torch.Tensor) -> torch.Tensor:
    """
    Quantile-normalized Immerkaer high frequency feature. Higher means more fine-scale detail/noise.

    args:
        images: [B, 3, H, W] or [B, 4, H, W] or [B, 3, T, H, W], channel_order: RGB or RGBA,
            pixel_values: must be already normalized to [0...1], dtype: float
    returns:
        values: [B], range: [0...1]
    """
    if not IMMERKAER_QUANTILES:
        raise RuntimeError("IMMERKAER_QUANTILES is not calibrated, run dev/calibrate_immerkaer.py")
    return quantile_normalize(compute_raw_immerkaer(images), IMMERKAER_QUANTILES)
