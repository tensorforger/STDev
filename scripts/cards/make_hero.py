"""
Hero card: the full cat image on the left, three bicubic | Steppe patch pairs right next to it (connected to their
regions by short leader lines), the message and a feature checklist in the right column.

Run: python scripts/cards/make_hero.py
"""

from pathlib import Path

import style
from PIL import Image
from style import Card, Layer, Patch

LOW_RESOLUTION_FILE = "main_low_res.png"
HIGH_RESOLUTION_FILE = "main_high_res_.png"

TILE = 372
TILE_GAP = 12
PATCHES = (
    Patch((806, 330), 64),
    Patch((380, 650), 120),
    Patch((890, 1078), 120),
)
HEADLINE = ("4× detail,", "decoded directly.")
SUBLINE = ("No second denoising pass", "of the base model.")
FEATURES = (
    "Image upscaling, 2× and 4×",
    "Video spatial upscaling",
    "Video temporal upscaling, 2× fps",
    "Sharpness control (hff)",
    "One step, ~45 ms per 2×",
    "Any aspect ratio",
    "Wan 2.1, Wan 2.2, Qwen-Image, Krea 2",
    "ComfyUI nodes and Python package",
)
CAPTION = ("Krea 2 Turbo 960×1280", "Steppe 200M 1-step, 2× → 2×")


def draw_check(layer: Layer, position: style.Point, size: float) -> None:
    """accent check mark in a thin square, position = top left"""
    x, y = position
    layer.rectangle((x, y, x + size, y + size), outline=style.BORDER, radius=3)
    layer.line(
        [
            (x + size * 0.24, y + size * 0.52),
            (x + size * 0.43, y + size * 0.72),
            (x + size * 0.78, y + size * 0.3),
        ],
        style.ACCENT,
        2.2,
    )


def build(input_directory: Path) -> Image.Image:
    low = style.load_image(str(input_directory / LOW_RESOLUTION_FILE))
    high = style.load_image(str(input_directory / HIGH_RESOLUTION_FILE))
    card = Card()
    style.draw_background(card)
    overlay = card.overlay
    margin = style.MARGIN

    shown = style.fit(high, height=card.height - 2 * margin)
    image_x, image_y = margin, margin
    image_right = image_x + shown.width
    display_scale = shown.width / low.width
    card.paste(shown, (image_x, image_y))
    overlay.rectangle(
        (image_x - 1, image_y - 1, image_right + 1, image_y + shown.height + 1),
        outline=style.BORDER,
    )
    style.draw_badge(
        overlay, (image_x + 14, image_y + shown.height - 42), "STEPPE 4×  3840×5120", style.TEXT
    )

    pair_x = image_right + 48
    pair_right = pair_x + 2 * TILE + TILE_GAP
    row_gap = (card.height - 2 * margin - 3 * TILE) / 2
    for index, patch in enumerate(PATCHES):
        tile_y = margin + index * (TILE + row_gap)
        bicubic, upscaled = style.patch_pair(low, high, patch, TILE)
        style.draw_pair(card, (pair_x, tile_y), TILE, bicubic, upscaled, TILE_GAP)

        region = style.region_on_image(patch, (image_x, image_y), display_scale)
        start_y = (region[1] + region[3]) / 2
        end_y = tile_y + TILE / 2
        path = [
            (region[2], start_y),
            (image_right + 8, start_y),
            (pair_x - 14, end_y),
            (pair_x, end_y),
        ]
        overlay.line(path, style.BACKGROUND, 3.5, 0.6)
        overlay.line(path, style.ACCENT, 1.2, 0.9)
        style.draw_region(overlay, region)

    x = pair_right + 72
    style.draw_brand(overlay, (x, margin + 6), 24)
    overlay.text((x, margin + 64), HEADLINE[0], style.DISPLAY, 52, style.TEXT)
    overlay.text((x, margin + 124), HEADLINE[1], style.DISPLAY, 52, style.TEXT)
    overlay.text((x, margin + 200), SUBLINE[0], style.DISPLAY_LIGHT, 31, style.ACCENT)
    overlay.text((x, margin + 240), SUBLINE[1], style.DISPLAY_LIGHT, 31, style.ACCENT)

    features_top = margin + 340
    row_height = 80
    check_size = 26
    overlay.line(
        [(x, features_top - 30), (card.width - margin - 8, features_top - 30)],
        style.GRID,
        dash=(4, 4),
    )
    for row, feature in enumerate(FEATURES):
        y = features_top + row * row_height
        draw_check(overlay, (x, y), check_size)
        overlay.text(
            (x + check_size + 18, y + check_size / 2), feature, style.SANS, 23, style.TEXT, "lm"
        )
    for line_index, text in enumerate(CAPTION):
        overlay.text(
            (x, card.height - margin - 26 + line_index * 26),
            text,
            style.MONO,
            15,
            style.MUTED,
            "ld",
            0.85,
        )

    style.draw_card_frame(card)
    return card.render()


def main() -> None:
    arguments = style.standard_arguments(__doc__).parse_args()
    style.save_card(build(arguments.input_directory), arguments.output_directory, "hero_cat")


if __name__ == "__main__":
    main()
