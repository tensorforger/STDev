import torch
import torch.nn.functional as functional

D65_WHITE_POINT = (0.95047, 1.0, 1.08883)
SRGB_TO_XYZ_MATRIX = (
    (0.412453, 0.357580, 0.180423),
    (0.212671, 0.715160, 0.072169),
    (0.019334, 0.119193, 0.950227),
)
LAB_LINEAR_THRESHOLD = 0.008856
LAB_LINEAR_SLOPE = 7.787
LAB_LINEAR_OFFSET = 16.0 / 116.0
CIEDE2000_POWER_BASE = 25.0**7


def downsample_by_two(pixels: torch.Tensor) -> torch.Tensor:
    """
    Bilinear 2x downsampling, the operation that produces the low resolution inputs of the benchmark.

    args:
        pixels: [B, 3, H, W], pixel_values: [0...1]
    returns:
        pixels: [B, 3, H / 2, W / 2], pixel_values: [0...1]
    """
    return functional.interpolate(pixels, scale_factor=0.5, mode="bilinear", align_corners=False)


def rgb_to_lab(pixels: torch.Tensor) -> torch.Tensor:
    """
    sRGB to CIELAB (D65, 2 degree observer), same constants as `skimage.color.rgb2lab`.

    args:
        pixels: [B, 3, H, W], channel_order: RGB, pixel_values: [0...1], dtype: torch.float64
    returns:
        lab: [B, 3, H, W], channels: L [0...100], a, b, dtype: torch.float64
    """
    linear = torch.where(pixels > 0.04045, ((pixels + 0.055) / 1.055) ** 2.4, pixels / 12.92)
    matrix = torch.tensor(SRGB_TO_XYZ_MATRIX, dtype=pixels.dtype, device=pixels.device)
    white_point = torch.tensor(D65_WHITE_POINT, dtype=pixels.dtype, device=pixels.device)
    xyz = torch.einsum("xc,bchw->bxhw", matrix, linear) / white_point.view(1, 3, 1, 1)
    transformed = torch.where(
        xyz > LAB_LINEAR_THRESHOLD,
        xyz.clamp(min=LAB_LINEAR_THRESHOLD) ** (1.0 / 3.0),
        LAB_LINEAR_SLOPE * xyz + LAB_LINEAR_OFFSET,
    )
    lightness = 116.0 * transformed[:, 1] - 16.0
    channel_a = 500.0 * (transformed[:, 0] - transformed[:, 1])
    channel_b = 200.0 * (transformed[:, 1] - transformed[:, 2])
    return torch.stack([lightness, channel_a, channel_b], dim=1)


def ciede2000(lab_first: torch.Tensor, lab_second: torch.Tensor) -> torch.Tensor:
    """
    CIEDE2000 color difference (Sharma et al. 2005, unit weighting factors).

    args:
        lab_first, lab_second: [B, 3, H, W], channels: L, a, b, dtype: torch.float64
    returns:
        difference: [B, H, W], dtype: torch.float64
    """
    lightness_first, a_first, b_first = lab_first.unbind(dim=1)
    lightness_second, a_second, b_second = lab_second.unbind(dim=1)

    chroma_mean = (torch.hypot(a_first, b_first) + torch.hypot(a_second, b_second)) / 2
    chroma_mean_power = chroma_mean**7
    a_scale = 1 + 0.5 * (
        1 - torch.sqrt(chroma_mean_power / (chroma_mean_power + CIEDE2000_POWER_BASE))
    )
    a_first_adjusted = a_first * a_scale
    a_second_adjusted = a_second * a_scale

    chroma_first = torch.hypot(a_first_adjusted, b_first)
    chroma_second = torch.hypot(a_second_adjusted, b_second)
    hue_first = torch.remainder(torch.atan2(b_first, a_first_adjusted), 2 * torch.pi)
    hue_second = torch.remainder(torch.atan2(b_second, a_second_adjusted), 2 * torch.pi)

    chroma_product_is_zero = chroma_first * chroma_second == 0
    hue_difference = hue_second - hue_first
    hue_difference = torch.where(
        hue_difference.abs() <= torch.pi,
        hue_difference,
        hue_difference - 2 * torch.pi * torch.sign(hue_difference),
    )
    hue_difference = torch.where(
        chroma_product_is_zero, torch.zeros_like(hue_difference), hue_difference
    )

    lightness_difference = lightness_second - lightness_first
    chroma_difference = chroma_second - chroma_first
    hue_term_difference = (
        2 * torch.sqrt(chroma_first * chroma_second) * torch.sin(hue_difference / 2)
    )

    lightness_mean = (lightness_first + lightness_second) / 2
    chroma_mean_adjusted = (chroma_first + chroma_second) / 2
    hue_sum = hue_first + hue_second
    hue_mean = torch.where(
        (hue_first - hue_second).abs() <= torch.pi,
        hue_sum / 2,
        torch.where(
            hue_sum < 2 * torch.pi, (hue_sum + 2 * torch.pi) / 2, (hue_sum - 2 * torch.pi) / 2
        ),
    )
    hue_mean = torch.where(chroma_product_is_zero, hue_sum, hue_mean)

    degrees_30 = torch.pi / 6
    hue_weight = (
        1
        - 0.17 * torch.cos(hue_mean - degrees_30)
        + 0.24 * torch.cos(2 * hue_mean)
        + 0.32 * torch.cos(3 * hue_mean + torch.pi / 30)
        - 0.20 * torch.cos(4 * hue_mean - 63 * torch.pi / 180)
    )
    hue_mean_degrees = torch.rad2deg(hue_mean)
    rotation_angle = degrees_30 * torch.exp(-(((hue_mean_degrees - 275) / 25) ** 2))
    chroma_mean_adjusted_power = chroma_mean_adjusted**7
    rotation_scale = 2 * torch.sqrt(
        chroma_mean_adjusted_power / (chroma_mean_adjusted_power + CIEDE2000_POWER_BASE)
    )

    lightness_offset = (lightness_mean - 50) ** 2
    lightness_scale = 1 + 0.015 * lightness_offset / torch.sqrt(20 + lightness_offset)
    chroma_scale = 1 + 0.045 * chroma_mean_adjusted
    hue_scale = 1 + 0.015 * chroma_mean_adjusted * hue_weight

    lightness_ratio = lightness_difference / lightness_scale
    chroma_ratio = chroma_difference / chroma_scale
    hue_ratio = hue_term_difference / hue_scale
    squared_difference = (
        lightness_ratio**2
        + chroma_ratio**2
        + hue_ratio**2
        - torch.sin(2 * rotation_angle) * rotation_scale * chroma_ratio * hue_ratio
    )
    return torch.sqrt(squared_difference.clamp(min=0))


def roundtrip_errors(
    predicted_pixels: torch.Tensor, initial_pixels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Downsamples the upscaled prediction back to the input resolution and compares it with the initial image.

    args:
        predicted_pixels: [B, 3, 2 * H, 2 * W], channel_order: RGB, pixel_values: [0...1]
        initial_pixels: [B, 3, H, W], channel_order: RGB, pixel_values: [0...1]
    returns:
        mse: [B], mean squared error per image, pixel_values: [0...1]
        ciede2000_difference: [B], mean CIEDE2000 per image
    """
    roundtrip_pixels = downsample_by_two(predicted_pixels).double()
    initial_pixels = initial_pixels.double()
    mse = (roundtrip_pixels - initial_pixels).square().mean(dim=(1, 2, 3))
    color_difference = ciede2000(rgb_to_lab(roundtrip_pixels), rgb_to_lab(initial_pixels))
    return mse, color_difference.mean(dim=(1, 2))
