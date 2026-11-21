"""
Benchmark table card: all 2x solutions of the unified Wan 2.1 benchmark without the privileged ground truth hff rows,
sorted by patch FID, Steppe rows highlighted, patch FID with an inline log scale bar.

Run: python scripts/cards/make_benchmark_table.py
"""

import json
import math
from pathlib import Path

import style
from make_benchmark_plot import RESULTS_FILE
from PIL import Image
from style import Card

ROWS = (
    ("ours_base_20_steps", "Steppe 200M base · 20 steps", "latent · generative", True),
    ("ours_distilled_1_step_200m", "Steppe 200M · 1 step", "latent · generative", True),
    ("ours_distilled_1_step_20m", "Steppe 20M · 1 step", "latent · generative", True),
    (
        "spacepxl_wan21_vae_upscale2x",
        "Wan2.1-VAE-upscale2x (spacepxl)",
        "decoder with built-in 2×",
        False,
    ),
    (
        "ours_distilled_1_step_20m_tiny_vae",
        "Steppe 20M · 1 step + tiny VAE",
        "latent · generative",
        True,
    ),
    ("decode_span", "SPAN 2× (Nomos)", "pixel, after decode", False),
    (
        "ours_distilled_1_step_2m_tiny_vae",
        "Steppe 2M · 1 step + tiny VAE",
        "latent · generative",
        True,
    ),
    ("decode_realesrgan", "Real-ESRGAN x2plus", "pixel, after decode", False),
    ("decode_bicubic", "bicubic", "pixel, after decode", False),
    ("sesquilsr", "SesquiLSR", "latent · for hires fix", False),
    ("wan_nn_latent_upscaler", "NN latent upscaler (VAE-Utils)", "latent · for hires fix", False),
    ("latent_bicubic_decode", "bicubic in latent space", "latent", False),
)
COLUMNS = (
    ("SOLUTION", 0),
    ("PATCH FID ↓", 640),
    ("FID ↓", 1150),
    ("COLOR SHIFT ΔE00 ↓", 1330),
    ("LATENCY", 1640),
    ("PEAK VRAM", 1880),
)
BAR_RANGE = (3.0, 150.0)
BAR_WIDTH = 330


def build(results_file: Path) -> Image.Image:
    results = json.loads(results_file.read_text())
    card = Card()
    style.draw_background(card)
    overlay = card.overlay
    margin = style.MARGIN

    style.draw_brand(overlay, (margin + 8, margin + 6), 24)
    overlay.text(
        (margin + 8, margin + 62),
        "2× upscaling from the Wan 2.1 latent",
        style.DISPLAY,
        52,
        style.TEXT,
    )
    overlay.text(
        (margin + 8, margin + 136),
        "1000 Axis-V1 photos, 512 → 1024. Only the latent goes in (no prompt), the decoded image comes out.",
        style.SANS,
        23,
        style.TEXT_DIM,
    )

    table_left = margin + 8
    table_right = card.width - margin - 8
    header_y = margin + 236
    row_top = header_y + 34
    row_height = (card.height - margin - 70 - row_top) / len(ROWS)
    for title, offset in COLUMNS:
        overlay.text(
            (table_left + offset, header_y),
            title,
            style.MONO_MEDIUM,
            16,
            style.MUTED,
            "ld",
            tracking=1.2,
        )
    overlay.line([(table_left, row_top - 6), (table_right, row_top - 6)], style.BORDER)

    low_bar, high_bar = (math.log10(value) for value in BAR_RANGE)
    for index, (key, name, kind, ours) in enumerate(ROWS):
        metrics = results[key]["metrics"]
        top = row_top + index * row_height
        middle = top + row_height / 2
        if ours:
            overlay.rectangle(
                (table_left, top + 3, table_right, top + row_height - 3),
                fill=style.ACCENT,
                fill_alpha=0.07,
            )
            overlay.rectangle(
                (table_left, top + 3, table_left + 3, top + row_height - 3), fill=style.ACCENT
            )
        overlay.line(
            [(table_left, top + row_height), (table_right, top + row_height)],
            style.GRID,
            dash=(3, 5),
        )
        name_color = style.TEXT if ours else style.TEXT_DIM
        overlay.text(
            (table_left + 22, middle - 3),
            name,
            style.SANS_MEDIUM if ours else style.SANS,
            23,
            name_color,
            "ld",
        )
        overlay.text(
            (table_left + 22, middle + 3),
            kind,
            style.MONO,
            14,
            style.ACCENT_DIM if ours else style.MUTED,
            "lt",
        )

        patch_fid = metrics["patch_fid"]
        bar_left = table_left + COLUMNS[1][1]
        fraction = (math.log10(patch_fid) - low_bar) / (high_bar - low_bar)
        overlay.rectangle(
            (bar_left, middle - 4, bar_left + BAR_WIDTH, middle + 4), fill=style.GRID_FAINT
        )
        overlay.rectangle(
            (bar_left, middle - 4, bar_left + BAR_WIDTH * fraction, middle + 4),
            fill=style.ACCENT if ours else style.OTHER,
            fill_alpha=0.9 if ours else 0.6,
        )
        overlay.text(
            (bar_left + BAR_WIDTH + 24, middle),
            f"{patch_fid:.2f}",
            style.MONO_MEDIUM,
            22,
            style.ACCENT if ours else style.TEXT_DIM,
            "lm",
        )

        values = (
            (COLUMNS[2][1], f"{metrics['fid']:.2f}"),
            (COLUMNS[3][1], f"{metrics['chroma_shift_ciede2000']:.2f}"),
            (COLUMNS[4][1], f"{metrics['latency_ms']:.0f} ms"),
            (COLUMNS[5][1], f"{metrics['peak_vram_mb'] / 1000:.1f} GB"),
        )
        for offset, text in values:
            overlay.text(
                (table_left + offset, middle),
                text,
                style.MONO,
                21,
                style.TEXT if ours else style.TEXT_DIM,
                "lm",
            )

    overlay.text(
        (card.width - margin, card.height - 44),
        "patch FID: native 256 px tiles, log bar · latency: batch 1, VAE decode included, eager bf16, RTX 5090 · Axis-V1 (CC BY 4.0)",
        style.MONO,
        14,
        style.MUTED,
        "rm",
        0.8,
    )
    style.draw_card_frame(card)
    return card.render()


def main() -> None:
    parser = style.standard_arguments(__doc__)
    parser.add_argument("--results-file", type=Path, default=RESULTS_FILE)
    arguments = parser.parse_args()
    style.save_card(build(arguments.results_file), arguments.output_directory, "benchmark_table")


if __name__ == "__main__":
    main()
