"""
Comparison cards with different layouts:
- compare_anime: image on the right, two large bicubic | Steppe pairs on the left
- compare_cloth: split view (bicubic | Steppe) of one zoomed region, thumbnail with the region on the right
- compare_leaves: the wide image on top, a strip of three pairs below (any aspect ratio)

Run: python scripts/cards/make_comparisons.py
"""

from pathlib import Path

import style
from PIL import Image
from style import Card, Patch

ANIME_FILES = ("steppe_demo_low_res_anime.png", "steppe_demo_high_res_anime.png")
ANIME_PATCHES = (Patch((165, 640), 200), Patch((440, 505), 120))
ANIME_LABELS = ("PHOTOREALISTIC", "ANIME")
CLOTH_FILES = ("steppe_demo_low_res_cloth.png", "steppe_demo_high_res_cloth.png")
CLOTH_PATCH = Patch((690, 590), 240)
LEAVES_FILES = ("steppe_demo_low_res_leave.png", "steppe_demo_high_res_leave.png")
LEAVES_PATCHES = (Patch((208, 192), 70), Patch((416, 229), 70), Patch((939, 320), 70))
CAPTION = "Krea 2 Turbo · Steppe 200M 1-step, chained 2× → 2× · no prompt, no refinement pass"


def load_pair(input_directory: Path, files: tuple[str, str]) -> tuple[Image.Image, Image.Image]:
    return style.load_image(str(input_directory / files[0])), style.load_image(
        str(input_directory / files[1])
    )


def draw_header(
    layer: style.Layer, position: style.Point, title: str, lines: tuple[str, ...]
) -> float:
    """returns: y below the header"""
    x, y = position
    style.draw_brand(layer, (x, y + 6), 24)
    layer.text((x, y + 60), title, style.DISPLAY, 52, style.TEXT)
    for index, line in enumerate(lines):
        layer.text((x, y + 134 + index * 32), line, style.SANS, 23, style.TEXT_DIM)
    return y + 134 + len(lines) * 32


def build_anime(input_directory: Path) -> Image.Image:
    low, high = load_pair(input_directory, ANIME_FILES)
    card = Card()
    style.draw_background(card)
    overlay = card.overlay
    margin = style.MARGIN

    shown = style.fit(high, height=card.height - 2 * margin)
    image_x = card.width - margin - shown.width
    display_scale = shown.width / low.width
    card.paste(shown, (image_x, margin))
    overlay.rectangle(
        (image_x - 1, margin - 1, image_x + shown.width + 1, margin + shown.height + 1),
        outline=style.BORDER,
    )
    style.draw_badge(
        overlay, (image_x + 14, margin + shown.height - 42), "STEPPE 4×  3840×5120", style.TEXT
    )

    draw_header(
        overlay,
        (margin + 8, margin),
        "Any style, one model.",
        ("Anime character, photorealistic street: both upscaled in one pass.",),
    )
    overlay.text((margin + 8, margin + 206), CAPTION, style.MONO, 14, style.MUTED, "lt", 0.8)
    tile = 466
    gap = 12
    pair_x = margin + 8
    pair_right = pair_x + 2 * tile + gap
    row_tops = (card.height - margin - 2 * tile - 24, card.height - margin - tile)
    for patch, label, row_top in zip(ANIME_PATCHES, ANIME_LABELS, row_tops, strict=True):
        bicubic, upscaled = style.patch_pair(low, high, patch, tile)
        style.draw_pair(card, (pair_x, row_top), tile, bicubic, upscaled, gap)
        region = style.region_on_image(patch, (image_x, margin), display_scale)
        style.draw_region(overlay, region)
        region_middle_y = (region[1] + region[3]) / 2
        anchor_x = pair_right + 22
        middle_y = row_top + tile / 2
        overlay.line(
            [
                (region[0], region_middle_y),
                (anchor_x + 120, region_middle_y),
                (anchor_x, middle_y),
                (pair_right, middle_y),
            ],
            style.ACCENT,
            1,
            0.55,
        )
        overlay.circle((anchor_x, middle_y), 3, fill=style.ACCENT)
        overlay.text(
            (pair_right + 16, middle_y + (16 if region_middle_y < middle_y else -16)),
            label,
            style.MONO_MEDIUM,
            15,
            style.ACCENT,
            "lt" if region_middle_y < middle_y else "ld",
            tracking=1.2,
        )
    style.draw_card_frame(card)
    return card.render()


def build_cloth(input_directory: Path) -> Image.Image:
    low, high = load_pair(input_directory, CLOTH_FILES)
    card = Card()
    style.draw_background(card)
    overlay = card.overlay
    margin = style.MARGIN

    side = card.height - 2 * margin
    bicubic, upscaled = style.patch_pair(low, high, CLOTH_PATCH, side)
    split = side // 2
    view = upscaled.copy()
    view.paste(bicubic.crop((0, 0, split, side)), (0, 0))
    card.paste(view, (margin, margin))
    overlay.rectangle(
        (margin - 1, margin - 1, margin + side + 1, margin + side + 1), outline=style.BORDER
    )
    divider_x = margin + split
    overlay.line([(divider_x, margin), (divider_x, margin + side)], style.ACCENT, 2)
    overlay.circle(
        (divider_x, margin + side / 2),
        18,
        fill=style.BACKGROUND,
        outline=style.ACCENT,
        width=2,
        alpha=0.95,
    )
    overlay.text((divider_x, margin + side / 2), "‹ ›", style.MONO_MEDIUM, 18, style.ACCENT, "mm")
    style.draw_badge(overlay, (margin + 16, margin + 16), "BICUBIC 4×", style.TEXT_DIM, 17)
    upscaled_label_width = style.text_width("STEPPE 4×", style.MONO_MEDIUM, 17, 1.2) + 17 * 1.4
    style.draw_badge(
        overlay,
        (margin + side - 16 - upscaled_label_width, margin + 16),
        "STEPPE 4×",
        style.ACCENT,
        17,
    )

    column_left = margin + side + 90
    header_bottom = draw_header(
        overlay,
        (column_left, margin),
        "Texture, not blur.",
        ("Threads, fibers and weave are generated", "in latent space and decoded directly."),
    )
    thumbnail_height = card.height - margin - 60 - (header_bottom + 70)
    thumbnail = style.fit(high, height=int(thumbnail_height))
    thumbnail_x, thumbnail_y = column_left, header_bottom + 70
    card.paste(thumbnail, (thumbnail_x, thumbnail_y))
    overlay.rectangle(
        (
            thumbnail_x - 1,
            thumbnail_y - 1,
            thumbnail_x + thumbnail.width + 1,
            thumbnail_y + thumbnail.height + 1,
        ),
        outline=style.BORDER,
    )
    region = style.region_on_image(
        CLOTH_PATCH, (thumbnail_x, thumbnail_y), thumbnail.width / low.width
    )
    style.draw_region(overlay, region)
    notes = (
        "960×1280 → 3840×5120",
        f"zoom: {CLOTH_PATCH.size:g} px input region",
        "Krea 2 Turbo · Steppe 200M",
        "1-step, chained 2× → 2×",
    )
    notes_x = thumbnail_x + thumbnail.width + 40
    for index, text in enumerate(notes):
        overlay.text(
            (notes_x, thumbnail_y + thumbnail.height - (len(notes) - 1 - index) * 28),
            text,
            style.MONO,
            15,
            style.MUTED,
            "ld",
        )
    style.draw_card_frame(card)
    return card.render()


def build_leaves(input_directory: Path) -> Image.Image:
    low, high = load_pair(input_directory, LEAVES_FILES)
    card = Card()
    style.draw_background(card)
    overlay = card.overlay
    margin = style.MARGIN

    shown = style.fit(high, width=card.width - 2 * margin)
    display_scale = shown.width / low.width
    card.paste(shown, (margin, margin))
    overlay.rectangle(
        (margin - 1, margin - 1, margin + shown.width + 1, margin + shown.height + 1),
        outline=style.BORDER,
    )
    style.draw_badge(
        overlay, (margin + 14, margin + shown.height - 42), "STEPPE 4×  5120×2048", style.TEXT
    )

    strip_top = margin + shown.height + 40
    tile = card.height - margin - strip_top
    gap, pair_gap = 10, 36
    pair_width = 2 * tile + gap
    strip_left = card.width - margin - 3 * pair_width - 2 * pair_gap
    for index, patch in enumerate(LEAVES_PATCHES):
        pair_x = strip_left + index * (pair_width + pair_gap)
        bicubic, upscaled = style.patch_pair(low, high, patch, tile)
        style.draw_pair(card, (pair_x, strip_top), tile, bicubic, upscaled, gap, badge_size=13)
        region = style.region_on_image(patch, (margin, margin), display_scale)
        style.draw_region(overlay, region)
        overlay.line(
            [((region[0] + region[2]) / 2, region[3]), (pair_x + pair_width / 2, strip_top - 1)],
            style.ACCENT,
            1,
            0.55,
        )
        overlay.circle((pair_x + pair_width / 2, strip_top - 1), 3, fill=style.ACCENT)

    x = margin + 8
    style.draw_brand(overlay, (x, strip_top + 2), 22)
    overlay.text((x, strip_top + 46), "Wide, tall, any size.", style.DISPLAY, 40, style.TEXT)
    overlay.text(
        (x, strip_top + 104),
        "1280×512 → 5120×2048 in two 2× steps.",
        style.SANS,
        21,
        style.TEXT_DIM,
    )
    overlay.text(
        (x, strip_top + 134), "Any aspect ratio from 512×512 up.", style.SANS, 21, style.TEXT_DIM
    )
    overlay.text(
        (x, card.height - margin + 2),
        "Krea 2 Turbo · Steppe 200M 1-step 2× → 2×",
        style.MONO,
        14,
        style.MUTED,
        "ld",
        0.8,
    )
    style.draw_card_frame(card)
    return card.render()


def main() -> None:
    arguments = style.standard_arguments(__doc__).parse_args()
    builders = {
        "compare_anime": build_anime,
        "compare_cloth": build_cloth,
        "compare_leaves": build_leaves,
    }
    for name, builder in builders.items():
        style.save_card(builder(arguments.input_directory), arguments.output_directory, name)


if __name__ == "__main__":
    main()
