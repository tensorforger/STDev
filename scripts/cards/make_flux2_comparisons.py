"""
Steppe-Flux.2 comparison cards: the full 4x image on the left, 2-3 large bicubic | Steppe pairs on the right (each
row is one zoomed region, its shape follows the row), one small caption line at the bottom.

Zoom regions live in flux2_zooms.json (edit them visually with edit_flux2_zooms.py).

Run: python scripts/cards/make_flux2_comparisons.py [--only A C] [--input-directory ../comfy_results_flux_2]
"""

import json
from dataclasses import dataclass
from pathlib import Path

import style
from PIL import Image
from style import Card

DEFAULT_INPUT_DIRECTORY = style.ML_ROOT / "comfy_results_flux_2"
ZOOMS_FILE = style.CARDS_DIRECTORY / "flux2_zooms.json"

PADDING = 32
COLUMN_GAP = 20
ROW_GAP = 14
TILE_GAP = 10
FOOTER_HEIGHT = 64
BADGE_SIZE = 14
INDEX_SIZE = 22


@dataclass
class Zoom:
    """
    A zoomed region of the low resolution image: center and width in low resolution pixels; the height follows the
    aspect ratio of its row. weight: relative row height.
    """

    center: tuple[float, float]
    width: float
    weight: float = 1.0


@dataclass
class ComparisonCard:
    name: str
    letter: str
    zooms: list[Zoom]
    source: str = "MOSS-Image"


@dataclass(frozen=True)
class RowLayout:
    """One zoom row in card pixels, and its region in low resolution pixels."""

    top: int
    height: int
    box: style.Box


@dataclass(frozen=True)
class CardLayout:
    image_height: int
    image_width: int
    area_left: int
    tile_width: int
    rows: tuple[RowLayout, ...]


def load_cards(path: Path = ZOOMS_FILE) -> list[ComparisonCard]:
    cards = []
    for entry in json.loads(path.read_text()):
        zooms = [
            Zoom(tuple(zoom["center"]), zoom["width"], zoom.get("weight", 1.0))
            for zoom in entry["zooms"]
        ]
        cards.append(
            ComparisonCard(entry["name"], entry["letter"], zooms, entry.get("source", "MOSS-Image"))
        )
    return cards


def save_cards(cards: list[ComparisonCard], path: Path = ZOOMS_FILE) -> None:
    entries = [
        {
            "name": card_spec.name,
            "letter": card_spec.letter,
            "source": card_spec.source,
            "zooms": [
                {
                    "center": [round(zoom.center[0], 1), round(zoom.center[1], 1)],
                    "width": round(zoom.width, 1),
                    "weight": round(zoom.weight, 3),
                }
                for zoom in card_spec.zooms
            ],
        }
        for card_spec in cards
    ]
    path.write_text(json.dumps(entries, indent=2) + "\n")


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


def card_layout(
    zooms: list[Zoom], low_size: tuple[int, int], high_size: tuple[int, int]
) -> CardLayout:
    """Positions of the image and the zoom rows on the card, shared by the generator and the editor."""
    card_width, card_height = style.CARD_SIZE
    image_height = card_height - PADDING - FOOTER_HEIGHT
    image_width = round(high_size[0] * image_height / high_size[1])
    area_left = PADDING + image_width + COLUMN_GAP
    tile_width = (card_width - PADDING - area_left - TILE_GAP) // 2
    total_weight = sum(zoom.weight for zoom in zooms)
    rows_height = image_height - ROW_GAP * (len(zooms) - 1)
    rows = []
    top = PADDING
    for index, zoom in enumerate(zooms, start=1):
        if index == len(zooms):
            row_height = PADDING + image_height - top
        else:
            row_height = round(rows_height * zoom.weight / total_weight)
        rows.append(RowLayout(top, row_height, zoom_box(zoom, row_height / tile_width, low_size)))
        top += row_height + ROW_GAP
    return CardLayout(image_height, image_width, area_left, tile_width, tuple(rows))


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


def build(card_spec: ComparisonCard, low: Image.Image, high: Image.Image) -> Image.Image:
    layout = card_layout(card_spec.zooms, low.size, high.size)
    card = Card()
    style.draw_background(card)
    overlay = card.overlay

    shown = high.resize((layout.image_width, layout.image_height), Image.Resampling.LANCZOS)
    image_x, image_y = PADDING, PADDING
    display_scale = shown.width / low.width
    card.paste(shown, (image_x, image_y))
    overlay.rectangle(
        (image_x - 1, image_y - 1, image_x + shown.width + 1, image_y + shown.height + 1),
        outline=style.BORDER,
    )

    area_left, tile_width = layout.area_left, layout.tile_width
    upscaled_x = area_left + tile_width + TILE_GAP
    for index, row in enumerate(layout.rows, start=1):
        top, row_height, box = row.top, row.height, row.box
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


def load_pair(input_directory: Path, letter: str) -> tuple[Image.Image, Image.Image]:
    """returns: (low resolution image, high resolution image)"""
    low = Image.open(input_directory / f"{letter}_low.png").convert("RGB")
    high = Image.open(input_directory / f"{letter}_high.png").convert("RGB")
    return low, high


def main() -> None:
    parser = style.standard_arguments(__doc__)
    parser.set_defaults(input_directory=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--zooms-file", type=Path, default=ZOOMS_FILE)
    parser.add_argument("--only", nargs="*", default=None, help="image letters, e.g. A C")
    arguments = parser.parse_args()
    for card_spec in load_cards(arguments.zooms_file):
        if arguments.only and card_spec.letter not in arguments.only:
            continue
        low, high = load_pair(arguments.input_directory, card_spec.letter)
        style.save_card(build(card_spec, low, high), arguments.output_directory, card_spec.name)


if __name__ == "__main__":
    main()
