"""
Covers and icon:
- cover_banner_21x9 (2520x1080, ComfyUI registry / Manager banner), cover_16x9 (1920x1080, posts),
  cover_social_2x1 (1280x640, GitHub social preview): wordmark and claim on the left, a split input | Steppe view on
  the right, a perspective ground grid towards the horizon behind both
- icon_400 (400x400, ComfyUI registry icon): the Steppe mark

Run: python scripts/cards/make_covers.py
"""

from pathlib import Path

import style
from PIL import Image
from style import Card, Patch

LOW_RESOLUTION_FILE = "main_low_res.png"
HIGH_RESOLUTION_FILE = "main_high_res_.png"
COVER_PATCH = Patch((383, 652), 190)
COVERS = (
    ("cover_banner_21x9", (2520, 1080)),
    ("cover_16x9", (1920, 1080)),
    ("cover_social_2x1", (1280, 640)),
)


def split_view(low: Image.Image, high: Image.Image, side: int) -> Image.Image:
    """left half: bicubic upscale of the input, right half: Steppe 4x"""
    bicubic, upscaled = style.patch_pair(low, high, COVER_PATCH, side)
    view = upscaled.copy()
    view.paste(bicubic.crop((0, 0, side // 2, side)), (0, 0))
    return view


def build_cover(size: tuple[int, int], low: Image.Image, high: Image.Image) -> Image.Image:
    width, height = size
    unit = height / 1080
    card = Card(size)
    background = card.background
    overlay = card.overlay
    horizon_y = height * 0.58
    style.draw_grid(background, (0, 0, width, horizon_y), 60 * unit, style.GRID_FAINT)
    vanishing_x = width - height * 0.12 - height * 0.36
    style.draw_perspective_grid(
        background,
        (0, 0, width, height),
        horizon_y,
        vanishing_x,
        columns=40,
        rows=16,
        spread=3.4,
        color=style.BORDER,
        alpha=1.0,
    )
    background.line([(0, horizon_y), (width, horizon_y)], style.BORDER, 1, 0.9)
    background.line(
        [(vanishing_x, 0), (vanishing_x, horizon_y)], style.GRID, 1, 0.6, dash=(6 * unit, 8 * unit)
    )

    side = round(height * 0.72)
    view_x = width - round(height * 0.12) - side
    view_y = round((height - side) / 2)
    card.paste(split_view(low, high, side), (view_x, view_y))
    overlay.rectangle(
        (view_x - 1, view_y - 1, view_x + side + 1, view_y + side + 1), outline=style.BORDER
    )
    divider_x = view_x + side // 2
    overlay.line([(divider_x, view_y), (divider_x, view_y + side)], style.ACCENT, 2 * unit)
    overlay.circle(
        (divider_x, view_y + side / 2),
        16 * unit,
        fill=style.BACKGROUND,
        outline=style.ACCENT,
        width=2 * unit,
    )
    badge_size = 15 * unit
    style.draw_badge(
        overlay, (view_x + 14 * unit, view_y + 14 * unit), "INPUT", style.TEXT_DIM, badge_size
    )
    upscaled_width = (
        style.text_width("STEPPE 4×", style.MONO_MEDIUM, badge_size, 1.2) + badge_size * 1.4
    )
    style.draw_badge(
        overlay,
        (view_x + side - 14 * unit - upscaled_width, view_y + 14 * unit),
        "STEPPE 4×",
        style.ACCENT,
        badge_size,
    )
    style.draw_crosshairs(
        overlay,
        [
            (view_x, view_y),
            (view_x + side, view_y),
            (view_x, view_y + side),
            (view_x + side, view_y + side),
        ],
        12 * unit,
        style.MUTED,
    )

    text_x = round(height * 0.11)
    mark_size = 92 * unit
    style.draw_mark(
        overlay, (text_x, height * 0.24), mark_size, style.TEXT_DIM, style.ACCENT, 2.2 * unit
    )
    overlay.text(
        (text_x + mark_size + 34 * unit, height * 0.24 + mark_size / 2),
        "steppe",
        style.DISPLAY_MEDIUM,
        120 * unit,
        style.TEXT,
        "lm",
        tracking=1.0 * unit,
    )
    overlay.text(
        (text_x, height * 0.24 + mark_size + 70 * unit),
        "Generative latent upscaler",
        style.DISPLAY_LIGHT,
        54 * unit,
        style.TEXT,
        "la",
    )
    overlay.text(
        (text_x, height * 0.24 + mark_size + 146 * unit),
        "4× detail in latent space, decoded directly.",
        style.SANS,
        30 * unit,
        style.TEXT_DIM,
        "la",
    )
    overlay.text(
        (text_x, height * 0.24 + mark_size + 190 * unit),
        "No second denoising pass of the base model.",
        style.SANS,
        30 * unit,
        style.ACCENT,
        "la",
    )
    tags = ("Wan 2.1", "Wan 2.2", "Qwen-Image", "Krea 2", "ComfyUI")
    tag_x = text_x
    tag_y = height * 0.24 + mark_size + 268 * unit
    for tag in tags:
        tag_width = style.text_width(tag, style.MONO, 20 * unit) + 28 * unit
        overlay.rectangle(
            (tag_x, tag_y, tag_x + tag_width, tag_y + 40 * unit),
            fill=style.BACKGROUND,
            fill_alpha=0.85,
            outline=style.BORDER,
            radius=4 * unit,
        )
        overlay.text(
            (tag_x + tag_width / 2, tag_y + 20 * unit),
            tag,
            style.MONO,
            20 * unit,
            style.TEXT_DIM,
            "mm",
        )
        tag_x += tag_width + 12 * unit
    style.draw_card_frame(card, inset=22 * unit)
    return card.render()


def build_icon() -> Image.Image:
    size = 400
    card = Card((size, size))
    style.draw_perspective_grid(
        card.background,
        (0, 0, size, size),
        size * 0.62,
        size / 2,
        columns=12,
        rows=7,
        spread=3.0,
        color=style.GRID,
        alpha=0.8,
    )
    mark_size = 236
    position = ((size - mark_size) / 2, (size - mark_size) / 2 - 8)
    card.overlay.rectangle(
        (
            position[0] - 6,
            position[1] - 6,
            position[0] + mark_size + 6,
            position[1] + mark_size + 6,
        ),
        fill=style.BACKGROUND,
        fill_alpha=0.9,
    )
    style.draw_mark(card.overlay, position, mark_size, style.TEXT, style.ACCENT, 5.0)
    return card.render()


def main() -> None:
    arguments = style.standard_arguments(__doc__).parse_args()
    low = style.load_image(str(arguments.input_directory / LOW_RESOLUTION_FILE))
    high = style.load_image(str(arguments.input_directory / HIGH_RESOLUTION_FILE))
    for name, size in COVERS:
        style.save_card(build_cover(size, low, high), arguments.output_directory, name)
    icon = build_icon()
    icon_path = Path(arguments.output_directory) / "icon_400.png"
    icon.save(icon_path, optimize=True)
    print(f"icon_400: {icon.width}x{icon.height}, {icon_path.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
