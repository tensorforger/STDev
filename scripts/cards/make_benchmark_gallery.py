"""
Benchmark gallery card: the same region of three Axis-V1 benchmark images (512 -> 1024 from the Wan 2.1 latent) for
the main solutions and the ground truth; the full ground truth image with the region on the left of each row.

Run: python scripts/cards/make_benchmark_gallery.py
"""

import style
from PIL import Image
from style import Card

DATASET_DIRECTORY = style.STDEV_ROOT / "benchmark_datasets" / "axis_v1_1k_wan_2x"
RESULTS_DIRECTORY = style.STDEV_ROOT / "raw_results" / "upscale_benchmark_axis_v1_1k"
SAMPLES = ((46, (480, 640)), (606, (720, 80)), (844, (800, 640)))
REGION = 160
COLUMNS = (
    ("decode_bicubic", "BICUBIC"),
    ("decode_span", "SPAN"),
    ("decode_realesrgan", "REAL-ESRGAN"),
    ("spacepxl_wan21_vae_upscale2x", "VAE-UPSCALE2X"),
    ("ours_distilled_1_step_200m", "STEPPE 1-STEP"),
    (None, "GROUND TRUTH"),
)
TILE = 296
GAP = 12
ROW_GAP = 26


def load_output(key: str | None, index: int) -> Image.Image:
    name = f"{index:04d}.png"
    if key is None:
        return style.load_image(str(DATASET_DIRECTORY / "ground_truth" / name))
    return style.load_image(str(RESULTS_DIRECTORY / key / "images" / name))


def build() -> Image.Image:
    card = Card()
    style.draw_background(card)
    overlay = card.overlay
    margin = style.MARGIN

    style.draw_brand(overlay, (margin + 8, margin + 6), 24)
    overlay.text(
        (margin + 8, margin + 62), "Closest to the ground truth", style.DISPLAY, 52, style.TEXT
    )
    overlay.text(
        (margin + 8, margin + 136),
        "Same input for all: the Wan 2.1 latent of the 512 px image. Output 1024 px, 160 px regions shown.",
        style.SANS,
        23,
        style.TEXT_DIM,
    )

    grid_width = 7 * TILE + 6 * GAP
    grid_left = (card.width - grid_width) / 2
    grid_top = card.height - margin - 40 - 3 * TILE - 2 * ROW_GAP
    for column, (key, title) in enumerate(COLUMNS):
        x = grid_left + (column + 1) * (TILE + GAP)
        color = (
            style.ACCENT
            if key and key.startswith("ours")
            else style.TEXT
            if key is None
            else style.MUTED
        )
        overlay.text((x, grid_top - 16), title, style.MONO_MEDIUM, 16, color, "ld", tracking=1.2)
    overlay.text(
        (grid_left, grid_top - 16), "IMAGE", style.MONO_MEDIUM, 16, style.MUTED, "ld", tracking=1.2
    )

    for row, (index, (left, top)) in enumerate(SAMPLES):
        y = grid_top + row * (TILE + ROW_GAP)
        truth = load_output(None, index)
        thumbnail = style.fit(truth, width=TILE)
        card.paste(thumbnail, (grid_left, y))
        overlay.rectangle(
            (grid_left - 1, y - 1, grid_left + TILE + 1, y + TILE + 1), outline=style.BORDER
        )
        scale = TILE / truth.width
        region = (
            grid_left + left * scale,
            y + top * scale,
            grid_left + (left + REGION) * scale,
            y + (top + REGION) * scale,
        )
        overlay.rectangle(region, outline=style.BACKGROUND, width=4, alpha=0.8)
        overlay.rectangle(region, outline=style.ACCENT, width=2)
        for column, (key, _) in enumerate(COLUMNS):
            x = grid_left + (column + 1) * (TILE + GAP)
            crop = (
                load_output(key, index)
                .crop((left, top, left + REGION, top + REGION))
                .resize((TILE, TILE), Image.Resampling.LANCZOS)
            )
            card.paste(crop, (x, y))
            ours = key is not None and key.startswith("ours")
            overlay.rectangle(
                (x - 1, y - 1, x + TILE + 1, y + TILE + 1),
                outline=style.ACCENT if ours else style.BORDER,
                alpha=0.9 if ours else 1.0,
            )

    steppe_column = next(
        column for column, (key, _) in enumerate(COLUMNS) if key and key.startswith("ours")
    )
    steppe_x = grid_left + (steppe_column + 1) * (TILE + GAP)
    overlay.rectangle(
        (steppe_x - 7, grid_top - 44, steppe_x + TILE + 7, grid_top + 3 * TILE + 2 * ROW_GAP + 7),
        outline=style.ACCENT,
        alpha=0.5,
        dash=(6, 5),
    )

    overlay.text(
        (card.width - margin, card.height - 44),
        "Images: Axis-V1 by refine-axis (CC BY 4.0), cropped · VAE-UPSCALE2X: spacepxl Wan2.1-VAE-upscale2x decoder",
        style.MONO,
        14,
        style.MUTED,
        "rm",
        0.8,
    )
    style.draw_card_frame(card)
    return card.render()


def main() -> None:
    arguments = style.standard_arguments(__doc__).parse_args()
    style.save_card(build(), arguments.output_directory, "benchmark_gallery")


if __name__ == "__main__":
    main()
