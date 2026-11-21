"""
Shared visual style of the Steppe cards: palette, fonts, a supersampled drawing layer, background grids, the Steppe
mark and image patch helpers.

A card has three layers: background (grids, supersampled), images (pasted at 1x, the shown pixels are not resampled
again) and overlay (lines, boxes, text, supersampled with alpha). Supersampled layers are downscaled with
premultiplied alpha, which gives anti-aliased thin lines with plain PIL.
"""

import argparse
import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

Color = tuple[int, int, int]
Point = tuple[float, float]
Box = tuple[float, float, float, float]

CARDS_DIRECTORY = Path(__file__).resolve().parent
FONTS_DIRECTORY = CARDS_DIRECTORY / "fonts"
STDEV_ROOT = CARDS_DIRECTORY.parents[1]
ML_ROOT = CARDS_DIRECTORY.parents[2]
DEFAULT_INPUT_DIRECTORY = ML_ROOT / "comfyui_results"
DEFAULT_OUTPUT_DIRECTORY = ML_ROOT / "tensorforger" / "assets" / "steppe"

SUPERSAMPLING = 3

BACKGROUND: Color = (11, 12, 14)
PANEL: Color = (17, 19, 23)
GRID_FAINT: Color = (23, 26, 30)
GRID: Color = (40, 44, 50)
BORDER: Color = (62, 67, 75)
DASH: Color = (86, 92, 101)
MUTED: Color = (128, 134, 144)
TEXT_DIM: Color = (180, 182, 186)
TEXT: Color = (234, 232, 227)
ACCENT: Color = (233, 169, 75)
ACCENT_DIM: Color = (150, 110, 52)
OTHER: Color = (112, 118, 128)

CARD_SIZE = (2400, 1350)
MARGIN = 72

SANS = "Inter-Regular"
SANS_MEDIUM = "Inter-Medium"
SANS_SEMIBOLD = "Inter-SemiBold"
SANS_LIGHT = "Inter-Light"
DISPLAY = "InterDisplay-SemiBold"
DISPLAY_MEDIUM = "InterDisplay-Medium"
DISPLAY_LIGHT = "InterDisplay-Light"
MONO = "JetBrainsMono-Regular"
MONO_MEDIUM = "JetBrainsMono-Medium"
MONO_LIGHT = "JetBrainsMono-Light"


def with_alpha(color: Color, alpha: float = 1.0) -> tuple[int, int, int, int]:
    return (*color, round(255 * alpha))


@cache
def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """
    args:
        name: file stem in fonts/, e.g. "Inter-Medium"
        size: pixel size of the supersampled layer
    """
    return ImageFont.truetype(str(FONTS_DIRECTORY / f"{name}.ttf"), size)


def text_width(text: str, font_name: str, size: float, tracking: float = 0.0) -> float:
    """returns: width in 1x pixels; tracking = extra letter spacing in 1x pixels"""
    return font(font_name, round(size * SUPERSAMPLING)).getlength(
        text
    ) / SUPERSAMPLING + tracking * len(text)


@dataclass
class Layer:
    """RGBA drawing surface at SUPERSAMPLING x the card size; all coordinates are 1x card pixels."""

    width: int
    height: int
    image: Image.Image = field(init=False)
    draw: ImageDraw.ImageDraw = field(init=False)

    def __post_init__(self) -> None:
        self.image = Image.new(
            "RGBA", (self.width * SUPERSAMPLING, self.height * SUPERSAMPLING), (0, 0, 0, 0)
        )
        self.draw = ImageDraw.Draw(self.image)

    @staticmethod
    def scaled(points: Iterable[Point]) -> list[Point]:
        return [(x * SUPERSAMPLING, y * SUPERSAMPLING) for x, y in points]

    @staticmethod
    def stroke(width: float) -> int:
        return max(1, round(width * SUPERSAMPLING))

    def line(
        self,
        points: Sequence[Point],
        color: Color,
        width: float = 1.0,
        alpha: float = 1.0,
        dash: tuple[float, float] | None = None,
    ) -> None:
        """dash: (on, off) lengths in 1x pixels, None = solid"""
        if dash is None:
            self.draw.line(
                self.scaled(points),
                fill=with_alpha(color, alpha),
                width=self.stroke(width),
                joint="curve",
            )
            return
        on_length, off_length = dash
        for (start_x, start_y), (end_x, end_y) in itertools.pairwise(points):
            length = math.hypot(end_x - start_x, end_y - start_y)
            if length == 0:
                continue
            direction_x, direction_y = (end_x - start_x) / length, (end_y - start_y) / length
            position = 0.0
            while position < length:
                dash_end = min(position + on_length, length)
                segment = [
                    (start_x + direction_x * position, start_y + direction_y * position),
                    (start_x + direction_x * dash_end, start_y + direction_y * dash_end),
                ]
                self.draw.line(
                    self.scaled(segment), fill=with_alpha(color, alpha), width=self.stroke(width)
                )
                position = dash_end + off_length

    def rectangle(
        self,
        box: Box,
        outline: Color | None = None,
        fill: Color | None = None,
        width: float = 1.0,
        alpha: float = 1.0,
        fill_alpha: float = 1.0,
        radius: float = 0.0,
        dash: tuple[float, float] | None = None,
    ) -> None:
        left, top, right, bottom = box
        if dash is not None and outline is not None:
            corners = [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]
            self.line(corners, outline, width, alpha, dash)
            return
        self.draw.rounded_rectangle(
            (
                left * SUPERSAMPLING,
                top * SUPERSAMPLING,
                right * SUPERSAMPLING - 1,
                bottom * SUPERSAMPLING - 1,
            ),
            radius=radius * SUPERSAMPLING,
            fill=with_alpha(fill, fill_alpha) if fill is not None else None,
            outline=with_alpha(outline, alpha) if outline is not None else None,
            width=self.stroke(width) if outline is not None else 0,
        )

    def circle(
        self,
        center: Point,
        radius: float,
        fill: Color | None = None,
        outline: Color | None = None,
        width: float = 1.0,
        alpha: float = 1.0,
    ) -> None:
        center_x, center_y = center
        self.draw.ellipse(
            (
                (center_x - radius) * SUPERSAMPLING,
                (center_y - radius) * SUPERSAMPLING,
                (center_x + radius) * SUPERSAMPLING,
                (center_y + radius) * SUPERSAMPLING,
            ),
            fill=with_alpha(fill, alpha) if fill is not None else None,
            outline=with_alpha(outline, alpha) if outline is not None else None,
            width=self.stroke(width),
        )

    def polygon(self, points: Sequence[Point], fill: Color, alpha: float = 1.0) -> None:
        self.draw.polygon(self.scaled(points), fill=with_alpha(fill, alpha))

    def text(
        self,
        position: Point,
        text: str,
        font_name: str,
        size: float,
        color: Color,
        anchor: str = "la",
        alpha: float = 1.0,
        tracking: float = 0.0,
    ) -> None:
        """anchor: PIL text anchor; tracking: extra letter spacing in 1x pixels"""
        typeface = font(font_name, round(size * SUPERSAMPLING))
        x, y = position
        if tracking == 0:
            self.draw.text(
                (x * SUPERSAMPLING, y * SUPERSAMPLING),
                text,
                font=typeface,
                fill=with_alpha(color, alpha),
                anchor=anchor,
            )
            return
        total_width = text_width(text, font_name, size, tracking)
        if anchor[0] == "m":
            x -= total_width / 2
        elif anchor[0] == "r":
            x -= total_width
        for character in text:
            self.draw.text(
                (x * SUPERSAMPLING, y * SUPERSAMPLING),
                character,
                font=typeface,
                fill=with_alpha(color, alpha),
                anchor="l" + anchor[1],
            )
            x += typeface.getlength(character) / SUPERSAMPLING + tracking

    def flatten(self) -> Image.Image:
        """returns: the layer downscaled to 1x, RGBA"""
        premultiplied = self.image.convert("RGBa").resize(
            (self.width, self.height), Image.Resampling.LANCZOS
        )
        return premultiplied.convert("RGBA")


class Card:
    """A card: background layer, pasted images, overlay layer."""

    def __init__(self, size: tuple[int, int] = CARD_SIZE, background: Color = BACKGROUND) -> None:
        self.width, self.height = size
        self.base = Image.new("RGB", size, background)
        self.background = Layer(*size)
        self.overlay = Layer(*size)
        self.images: list[tuple[Image.Image, tuple[int, int]]] = []

    def paste(self, image: Image.Image, position: Point) -> None:
        self.images.append((image, (round(position[0]), round(position[1]))))

    def render(self) -> Image.Image:
        result = self.base.convert("RGBA")
        result.alpha_composite(self.background.flatten())
        for image, position in self.images:
            if image.mode == "RGBA":
                result.alpha_composite(image, position)
            else:
                result.paste(image, position)
        result.alpha_composite(self.overlay.flatten())
        return result.convert("RGB")


def draw_grid(
    layer: Layer,
    box: Box,
    step: float,
    color: Color = GRID_FAINT,
    width: float = 1.0,
    alpha: float = 1.0,
) -> None:
    left, top, right, bottom = box
    x = left
    while x <= right:
        layer.line([(x, top), (x, bottom)], color, width, alpha)
        x += step
    y = top
    while y <= bottom:
        layer.line([(left, y), (right, y)], color, width, alpha)
        y += step


def draw_crosshairs(
    layer: Layer,
    points: Iterable[Point],
    size: float = 6,
    color: Color = DASH,
    width: float = 1.0,
    alpha: float = 1.0,
) -> None:
    for x, y in points:
        layer.line([(x - size, y), (x + size, y)], color, width, alpha)
        layer.line([(x, y - size), (x, y + size)], color, width, alpha)


def draw_perspective_grid(
    layer: Layer,
    box: Box,
    horizon_y: float,
    vanishing_x: float,
    columns: int = 24,
    rows: int = 14,
    spread: float = 3.0,
    color: Color = GRID,
    width: float = 1.0,
    alpha: float = 1.0,
) -> None:
    """
    Ground plane grid between the horizon and the bottom of the box, converging to (vanishing_x, horizon_y); rows get
    denser towards the horizon (1 / depth spacing) and fade out. spread: plane width at the bottom / box width.
    """
    left, _, right, bottom = box
    half_width = (right - left) * spread / 2
    for column in range(columns + 1):
        bottom_x = vanishing_x - half_width + 2 * half_width * column / columns
        layer.line([(vanishing_x, horizon_y), (bottom_x, bottom)], color, width, alpha * 0.6)
    nearest_depth, farthest_depth = 1.0, 14.0
    for row in range(rows + 1):
        fraction = row / rows
        depth = nearest_depth + (farthest_depth - nearest_depth) * fraction**1.6
        y = horizon_y + (bottom - horizon_y) * nearest_depth / depth
        layer.line([(left, y), (right, y)], color, width, alpha * (1.0 - 0.8 * fraction))


def draw_background(card: Card, step: float = 60) -> None:
    """faint square grid with a stronger grid every 6 cells"""
    full = (0, 0, card.width, card.height)
    draw_grid(card.background, full, step, GRID_FAINT)
    draw_grid(card.background, full, step * 6, GRID, alpha=0.8)


def draw_card_frame(
    card: Card, inset: float = 24, color: Color = BORDER, alpha: float = 0.7
) -> None:
    layer = card.overlay
    right, bottom = card.width - inset, card.height - inset
    layer.rectangle((inset, inset, right, bottom), outline=color, alpha=alpha)
    draw_crosshairs(
        layer, [(inset, inset), (right, inset), (inset, bottom), (right, bottom)], 10, MUTED, 1, 0.8
    )


def draw_mark(
    layer: Layer,
    position: Point,
    size: float,
    color: Color = MUTED,
    accent: Color = ACCENT,
    width: float = 1.0,
    alpha: float = 1.0,
    levels: int = 3,
) -> None:
    """
    The Steppe mark: a square with a 2x2 grid; its top right cell holds the next 2x2 grid (chained 2x -> 2x), the
    finest cell is filled with the accent (the generated detail).
    """
    x, y = position
    side = size
    for level in range(levels):
        layer.rectangle((x, y, x + side, y + side), outline=color, width=width, alpha=alpha)
        layer.line([(x + side / 2, y), (x + side / 2, y + side)], color, width, alpha)
        layer.line([(x, y + side / 2), (x + side, y + side / 2)], color, width, alpha)
        x, side = x + side / 2, side / 2
        if level == levels - 1:
            layer.rectangle((x, y, x + side, y + side), fill=accent, alpha=alpha)


def draw_brand(layer: Layer, position: Point, size: float = 22, alpha: float = 0.9) -> float:
    """
    Small, muted mark and wordmark, position = top left.
    returns: right edge x
    """
    x, y = position
    draw_mark(layer, (x, y), size, MUTED, ACCENT, 1.0, alpha)
    text_x = x + size * 1.55
    word_size = size * 0.95
    layer.text(
        (text_x, y + size / 2),
        "steppe",
        DISPLAY_MEDIUM,
        word_size,
        TEXT_DIM,
        "lm",
        alpha,
        tracking=0.3,
    )
    return text_x + text_width("steppe", DISPLAY_MEDIUM, word_size, 0.3)


def draw_badge(layer: Layer, position: Point, text: str, color: Color, size: float = 15) -> float:
    """
    Label on a translucent dark plate, used on top of images. position = top left.
    returns: badge width
    """
    x, y = position
    width = text_width(text, MONO_MEDIUM, size, 1.2) + size * 1.4
    height = size * 1.9
    layer.rectangle((x, y, x + width, y + height), fill=BACKGROUND, fill_alpha=0.78)
    layer.text((x + size * 0.7, y + height / 2), text, MONO_MEDIUM, size, color, "lm", tracking=1.2)
    return width


@cache
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def fit(image: Image.Image, width: int | None = None, height: int | None = None) -> Image.Image:
    """Lanczos resize to the given width or height, keeping the aspect ratio."""
    if width is None and height is not None:
        width = round(image.width * height / image.height)
    if height is None and width is not None:
        height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


@dataclass(frozen=True)
class Patch:
    """A square region of the low resolution image, in low resolution pixels."""

    center: Point
    size: float

    def box(self) -> Box:
        center_x, center_y = self.center
        half = self.size / 2
        return (center_x - half, center_y - half, center_x + half, center_y + half)


def patch_pair(
    low: Image.Image,
    high: Image.Image,
    patch: Patch,
    tile: int,
    display: Image.Resampling = Image.Resampling.LANCZOS,
) -> tuple[Image.Image, Image.Image]:
    """
    The bicubic tile is made at the same upscale factor as the high resolution image, then both tiles get the same
    display resize to tile x tile (none if the region is natively that size in the high resolution image).
    display: NEAREST shows the pixels of deep zooms exactly
    returns:
        (bicubic upscale of the region, the same region of the high resolution image)
    """
    factor = high.width / low.width
    left, top, right, bottom = patch.box()
    high_box = tuple(round(value * factor) for value in (left, top, right, bottom))
    native_size = (high_box[2] - high_box[0], high_box[3] - high_box[1])
    bicubic = low.resize(native_size, Image.Resampling.BICUBIC, box=(left, top, right, bottom))
    upscaled = high.crop(high_box)
    if native_size != (tile, tile):
        bicubic = bicubic.resize((tile, tile), display)
        upscaled = upscaled.resize((tile, tile), display)
    return bicubic, upscaled


def draw_pair(
    card: "Card",
    position: Point,
    tile: int,
    bicubic: Image.Image,
    upscaled: Image.Image,
    gap: float = 12,
    badge_size: float = 15,
    labels: tuple[str, str] = ("BICUBIC 4×", "STEPPE 4×"),
) -> Box:
    """
    Bicubic and upscaled tiles side by side with borders and badges, position = top left.
    returns: box of the pair
    """
    x, y = position
    upscaled_x = x + tile + gap
    card.paste(bicubic, (x, y))
    card.paste(upscaled, (upscaled_x, y))
    card.overlay.rectangle((x - 1, y - 1, x + tile + 1, y + tile + 1), outline=BORDER)
    card.overlay.rectangle(
        (upscaled_x - 1, y - 1, upscaled_x + tile + 1, y + tile + 1), outline=ACCENT, alpha=0.85
    )
    inset = badge_size * 0.65
    draw_badge(card.overlay, (x + inset, y + inset), labels[0], TEXT_DIM, badge_size)
    draw_badge(card.overlay, (upscaled_x + inset, y + inset), labels[1], ACCENT, badge_size)
    return (x, y, upscaled_x + tile, y + tile)


def draw_region(layer: Layer, box: Box, width: float = 1.5) -> None:
    """accent region box with a dark under-stroke, visible on busy images"""
    layer.rectangle(box, outline=BACKGROUND, width=width + 2.5, alpha=0.75)
    layer.rectangle(box, outline=ACCENT, width=width)


def region_on_image(patch: Patch, image_position: Point, display_scale: float) -> Box:
    """returns: the patch box in card coordinates for an image shown at image_position with display_scale"""
    left, top, right, bottom = patch.box()
    image_x, image_y = image_position
    return (
        image_x + left * display_scale,
        image_y + top * display_scale,
        image_x + right * display_scale,
        image_y + bottom * display_scale,
    )


def save_card(
    image: Image.Image, output_directory: Path, name: str, jpeg_quality: int = 92
) -> None:
    """Writes <name>.png (master) and <name>.jpg (web)."""
    output_directory.mkdir(parents=True, exist_ok=True)
    png_path = output_directory / f"{name}.png"
    jpeg_path = output_directory / f"{name}.jpg"
    image.save(png_path, optimize=True)
    image.save(jpeg_path, quality=jpeg_quality, subsampling=0, optimize=True, progressive=True)
    print(
        f"{name}: {image.width}x{image.height}, png {png_path.stat().st_size / 1e6:.1f} MB, jpg {jpeg_path.stat().st_size / 1e6:.1f} MB"
    )


def standard_arguments(description: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser
