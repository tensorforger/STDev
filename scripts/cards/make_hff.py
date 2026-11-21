"""
HFF card: the portrait on the left, one small sunlit skin region in a deep, pixel exact zoom on the right: bicubic and
three hff values in a 2x2 grid, with an hff scale marking the three values.

Run: python scripts/cards/make_hff.py
"""

from pathlib import Path

import style
from PIL import Image
from style import Card, Patch

LOW_RESOLUTION_FILE = "hff_orig_00001_.png"
HFF_FILES = (
    ("0.2", "hff_0.2_00001_.png"),
    ("0.6", "hff_0.6_00001_.png"),
    ("0.95", "hff_0.95_00001_.png"),
)
PATCH = Patch((625, 680), 38.5)
TILE = 462
TILE_GAP = 14
HFF_NOTES = {"0.2": "smooth, soft texture", "0.6": "natural", "0.95": "crisp, fine texture"}


def draw_scale(
    layer: style.Layer, left: float, right: float, y: float, values: tuple[float, ...]
) -> None:
    """hff ruler from 0 to 1 with markers at the shown values"""
    layer.line([(left, y), (right, y)], style.BORDER)
    for step in range(11):
        x = left + (right - left) * step / 10
        height = 10 if step % 5 == 0 else 5
        layer.line([(x, y - height), (x, y)], style.DASH)
    layer.text((left, y + 14), "0", style.MONO, 15, style.MUTED, "mt")
    layer.text((right, y + 14), "1", style.MONO, 15, style.MUTED, "mt")
    layer.text((left - 24, y), "hff", style.MONO_MEDIUM, 16, style.TEXT_DIM, "rm")
    for value in values:
        x = left + (right - left) * value
        layer.circle((x, y), 8, fill=style.BACKGROUND, outline=style.ACCENT, width=2)
        layer.circle((x, y), 3.5, fill=style.ACCENT)
        layer.text((x, y - 18), f"{value:g}", style.MONO_MEDIUM, 17, style.ACCENT, "md")


def build(input_directory: Path) -> Image.Image:
    low = style.load_image(str(input_directory / LOW_RESOLUTION_FILE))
    highs = {value: style.load_image(str(input_directory / name)) for value, name in HFF_FILES}
    card = Card()
    style.draw_background(card)
    overlay = card.overlay
    margin = style.MARGIN

    reference = highs["0.95"]
    shown = style.fit(reference, height=card.height - 2 * margin)
    image_x, image_y = margin, margin
    image_right = image_x + shown.width
    display_scale = shown.width / low.width
    card.paste(shown, (image_x, image_y))
    overlay.rectangle(
        (image_x - 1, image_y - 1, image_right + 1, image_y + shown.height + 1),
        outline=style.BORDER,
    )
    style.draw_badge(
        overlay, (image_x + 14, image_y + shown.height - 42), "STEPPE 4×  HFF 0.95", style.TEXT
    )

    column_left = image_right + 72
    style.draw_brand(overlay, (column_left, margin + 6), 24)
    overlay.text((column_left, margin + 60), "Dial in the detail.", style.DISPLAY, 56, style.TEXT)
    overlay.text(
        (column_left, margin + 136),
        "One number from 0 to 1 sets the amount of fine detail. Absolute and measurable:",
        style.SANS,
        23,
        style.TEXT_DIM,
    )
    overlay.text(
        (column_left, margin + 168),
        "Steppe Measure HFF reads it from any reference photo.",
        style.SANS,
        23,
        style.TEXT_DIM,
    )

    grid_top = margin + 290
    grid_left = column_left
    tiles = [("BICUBIC 4×", None)] + [(f"HFF {value}", value) for value, _ in HFF_FILES]
    for index, (title, value) in enumerate(tiles):
        row, column = divmod(index, 2)
        tile_x = grid_left + column * (TILE + TILE_GAP)
        tile_y = grid_top + row * (TILE + TILE_GAP)
        high = highs[value] if value is not None else reference
        bicubic, upscaled = style.patch_pair(low, high, PATCH, TILE, Image.Resampling.NEAREST)
        card.paste(upscaled if value is not None else bicubic, (tile_x, tile_y))
        color = style.ACCENT if value is not None else style.TEXT_DIM
        overlay.rectangle(
            (tile_x - 1, tile_y - 1, tile_x + TILE + 1, tile_y + TILE + 1),
            outline=color if value else style.BORDER,
            alpha=0.85,
        )
        style.draw_badge(overlay, (tile_x + 10, tile_y + 10), title, color)

    grid_right = grid_left + 2 * TILE + TILE_GAP
    grid_bottom = grid_top + 2 * TILE + TILE_GAP
    notes_left = grid_right + 56
    notes_right = card.width - margin - 8
    draw_scale(
        overlay,
        notes_left + 40,
        notes_right - 10,
        grid_top + 40,
        tuple(float(value) for value, _ in HFF_FILES),
    )
    notes_y = grid_top + 130
    for value, _ in HFF_FILES:
        overlay.line([(notes_left, notes_y), (notes_right, notes_y)], style.GRID, dash=(4, 4))
        overlay.text(
            (notes_left, notes_y + 20), f"hff {value}", style.MONO_MEDIUM, 18, style.ACCENT
        )
        overlay.text((notes_left, notes_y + 52), HFF_NOTES[value], style.SANS, 19, style.TEXT_DIM)
        notes_y += 110
    overlay.line([(notes_left, notes_y), (notes_right, notes_y)], style.GRID, dash=(4, 4))
    notes = (
        "Pixel exact 12× zoom of a",
        f"{PATCH.size:g} px region of the 960×1280",
        "Krea 2 Turbo input. Steppe 200M",
        "1-step 2× → 2×, guidance 2.",
    )
    for line_index, text in enumerate(notes):
        overlay.text(
            (notes_left, grid_bottom - 78 + line_index * 26),
            text,
            style.MONO,
            15,
            style.MUTED,
            "ld",
        )

    left, top, right, bottom = PATCH.box()
    region = (
        image_x + left * display_scale,
        image_y + top * display_scale,
        image_x + right * display_scale,
        image_y + bottom * display_scale,
    )
    style.draw_region(overlay, region)
    overlay.line([(region[2], region[1]), (grid_left, grid_top)], style.ACCENT, 1, 0.5)
    overlay.line([(region[2], region[3]), (grid_left, grid_bottom)], style.ACCENT, 1, 0.5)

    style.draw_card_frame(card)
    return card.render()


def main() -> None:
    arguments = style.standard_arguments(__doc__).parse_args()
    style.save_card(build(arguments.input_directory), arguments.output_directory, "hff_portrait")


if __name__ == "__main__":
    main()
