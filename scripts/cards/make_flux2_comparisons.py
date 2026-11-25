"""
Steppe-Flux.2 comparison cards: the full 4x image on the left, 2-3 large bicubic | Steppe pairs on the right (each
row is one zoomed region, its shape follows the row), one small caption line at the bottom.

Run: python scripts/cards/make_flux2_comparisons.py [--only A C] [--input-directory ../comfy_results_flux_2]
"""

from dataclasses import dataclass
from pathlib import Path

import style
from PIL import Image
from style import Card

DEFAULT_INPUT_DIRECTORY = style.ML_ROOT / "comfy_results_flux_2"

PADDING = 32
COLUMN_GAP = 20
ROW_GAP = 14
TILE_GAP = 10
FOOTER_HEIGHT = 64
BADGE_SIZE = 14
INDEX_SIZE = 22


@dataclass(frozen=True)
class Zoom:
    """
    A zoomed region of the low resolution image: center and width in low resolution pixels; the height follows the
    aspect ratio of its row. weight: relative row height.
    """

    center: style.Point
    width: float
    weight: float = 1.0


@dataclass(frozen=True)
class ComparisonCard:
    name: str
    letter: str
    zooms: tuple[Zoom, ...]
    source: str = "MOSS-Image"


CARDS = (
    ComparisonCard(
        "flux2_compare_cat",
        "A",
        (Zoom((668, 268), 135, 0.5), Zoom((312, 520), 165, 1.5)),
    ),
    ComparisonCard(
        "flux2_compare_neon",
        "B",
        (Zoom((388, 482), 230, 0.8), Zoom((395, 650), 150, 1.0), Zoom((640, 990), 170, 1.0)),
    ),
    ComparisonCard(
        "flux2_compare_steppe",
        "C",
        (Zoom((628, 280), 240, 1.0), Zoom((292, 330), 150, 1.0), Zoom((365, 615), 150, 1.0)),
    ),
    ComparisonCard(
        "flux2_compare_coffee",
        "D",
        (Zoom((480, 610), 200, 1.0), Zoom((360, 770), 150, 1.0), Zoom((470, 290), 170, 1.0)),
    ),
    ComparisonCard(
        "flux2_compare_gulls",
        "E",
        (Zoom((395, 650), 160, 1.15), Zoom((388, 180), 220, 0.85)),
    ),
    ComparisonCard(
        "flux2_compare_food",
        "F",
        (Zoom((310, 760), 200, 1.0), Zoom((540, 700), 180, 1.0)),
    ),
    ComparisonCard(
        "flux2_compare_newspaper",
        "G",
        (Zoom((350, 520), 300, 0.85), Zoom((520, 640), 260, 0.95), Zoom((300, 800), 170, 1.2)),
    ),
)


def zoom_box(zoom: Zoom, aspect: float, image_size: tuple[int, int]) -> style.Box:
    """
    aspect: height / width of the shown tile
    returns: the region in low resolution pixels, shifted inside the image
    """
    half_width, half_height = zoom.width / 2, zoom.width * aspect / 2
    center_x = min(max(zoom.center[0], half_width), image_size[0] - half_width)
    center_y = min(max(zoom.center[1], half_height), image_size[1] - half_height)
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def zoom_pair(
    low: Image.Image, high: Image.Image, box: style.Box, size: tuple[int, int]
) -> tuple[Image.Image, Image.Image]:
    """
    Bicubic upscale of the region at the same factor as the high resolution image and the same region of the high
    resolution image, both with the same display resize to size.
    returns:
        (bicubic, upscaled)
    """
    factor = high.width / low.width
    high_box = tuple(round(value * factor) for value in box)
    native_size = (high_box[2] - high_box[0], high_box[3] - high_box[1])
    bicubic = low.resize(native_size, Image.Resampling.BICUBIC, box=box)
    upscaled = high.crop(high_box)
    return (
        bicubic.resize(size, Image.Resampling.LANCZOS),
        upscaled.resize(size, Image.Resampling.LANCZOS),
    )


def draw_index(layer: style.Layer, position: style.Point, index: int) -> None:
    """small accent square with the zoom number, position = top left"""
    x, y = position
    layer.rectangle((x, y, x + INDEX_SIZE, y + INDEX_SIZE), fill=style.ACCENT)
    layer.text(
        (x + INDEX_SIZE / 2, y + INDEX_SIZE / 2),
        str(index),
        style.MONO_MEDIUM,
        14,
        style.BACKGROUND,
        "mm",
    )


def build(card_spec: ComparisonCard, input_directory: Path) -> Image.Image:
    low = style.load_image(str(input_directory / f"{card_spec.letter}_low.png"))
    high = style.load_image(str(input_directory / f"{card_spec.letter}_high.png"))
    card = Card()
    style.draw_background(card)
    overlay = card.overlay

    image_height = card.height - PADDING - FOOTER_HEIGHT
    shown = style.fit(high, height=image_height)
    image_x, image_y = PADDING, PADDING
    display_scale = shown.width / low.width
    card.paste(shown, (image_x, image_y))
    overlay.rectangle(
        (image_x - 1, image_y - 1, image_x + shown.width + 1, image_y + shown.height + 1),
        outline=style.BORDER,
    )

    area_left = image_x + shown.width + COLUMN_GAP
    tile_width = (card.width - PADDING - area_left - TILE_GAP) // 2
    upscaled_x = area_left + tile_width + TILE_GAP
    total_weight = sum(zoom.weight for zoom in card_spec.zooms)
    rows_height = image_height - ROW_GAP * (len(card_spec.zooms) - 1)
    top = image_y
    for index, zoom in enumerate(card_spec.zooms, start=1):
        if index == len(card_spec.zooms):
            row_height = image_y + image_height - top
        else:
            row_height = round(rows_height * zoom.weight / total_weight)
        box = zoom_box(zoom, row_height / tile_width, low.size)
        bicubic, upscaled = zoom_pair(low, high, box, (tile_width, row_height))
        card.paste(bicubic, (area_left, top))
        card.paste(upscaled, (upscaled_x, top))
        overlay.rectangle(
            (area_left - 1, top - 1, area_left + tile_width + 1, top + row_height + 1),
            outline=style.BORDER,
        )
        overlay.rectangle(
            (upscaled_x - 1, top - 1, upscaled_x + tile_width + 1, top + row_height + 1),
            outline=style.ACCENT,
            alpha=0.85,
        )
        inset = 10
        draw_index(overlay, (area_left + inset, top + inset), index)
        badge_y = top + inset + (INDEX_SIZE - BADGE_SIZE * 1.9) / 2
        style.draw_badge(
            overlay,
            (area_left + inset + INDEX_SIZE + 6, badge_y),
            "BICUBIC 4×",
            style.TEXT_DIM,
            BADGE_SIZE,
        )
        style.draw_badge(
            overlay, (upscaled_x + inset, badge_y), "STEPPE 4×", style.ACCENT, BADGE_SIZE
        )

        region = (
            image_x + box[0] * display_scale,
            image_y + box[1] * display_scale,
            image_x + box[2] * display_scale,
            image_y + box[3] * display_scale,
        )
        style.draw_region(overlay, region)
        draw_index(overlay, (region[0], region[1] - INDEX_SIZE), index)
        top += row_height + ROW_GAP

    footer_y = card.height - FOOTER_HEIGHT / 2
    brand_right = style.draw_brand(overlay, (PADDING, footer_y - 10), 20)
    caption = (
        f"{card_spec.source} {low.width}×{low.height}  →  Steppe-Flux.2 200M 1-step, 2× → 2×  →  "
        f"{high.width}×{high.height}   ·   bicubic at the same 4× factor   ·   no prompt, no refinement pass"
    )
    overlay.text((brand_right + 28, footer_y), caption, style.MONO, 15, style.MUTED, "lm")
    overlay.text(
        (card.width - PADDING, footer_y),
        "github.com/TensorForger/ComfyUI-Steppe",
        style.MONO,
        15,
        style.MUTED,
        "rm",
    )
    return card.render()


def main() -> None:
    parser = style.standard_arguments(__doc__)
    parser.set_defaults(input_directory=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--only", nargs="*", default=None, help="image letters, e.g. A C")
    arguments = parser.parse_args()
    for card_spec in CARDS:
        if arguments.only and card_spec.letter not in arguments.only:
            continue
        style.save_card(
            build(card_spec, arguments.input_directory), arguments.output_directory, card_spec.name
        )


if __name__ == "__main__":
    main()
