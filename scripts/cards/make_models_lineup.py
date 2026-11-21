"""
Models lineup card: the three released Steppe Wan 2.1 models side by side (size, steps, benchmark patch FID and
latency, intended use); the default model is highlighted.

Run: python scripts/cards/make_models_lineup.py
"""

import json
from dataclasses import dataclass
from pathlib import Path

import style
from make_benchmark_plot import RESULTS_FILE
from PIL import Image
from style import Card


@dataclass(frozen=True)
class Model:
    title: str
    file_name: str
    benchmark_key: str
    parameters: str
    steps: str
    file_size: str
    uses: tuple[str, ...]
    default: bool = False
    tag: str = ""


STAT_ROW_HEIGHT = 84

MODELS = (
    Model(
        "200M · 1 step",
        "steppe_wan_2_1_200M_1step",
        "ours_distilled_1_step_200m",
        "222M",
        "1",
        "445 MB",
        (
            "the quality of the base model in one step",
            "images, videos, chained 2× → 2×",
            "90% of use cases",
        ),
        default=True,
        tag="DEFAULT",
    ),
    Model(
        "200M base",
        "steppe_wan_2_1_200M_base",
        "ours_base_20_steps",
        "222M",
        "1-200 (20)",
        "444 MB",
        (
            "most control: steps, CFG on hff,",
            "negative hff, shift, noise scale",
            "research and experiments",
        ),
        tag="CONTROL",
    ),
    Model(
        "20M · 1 step",
        "steppe_wan_2_1_20M_1step",
        "ours_distilled_1_step_20m",
        "19.5M",
        "1",
        "39 MB",
        ("very large outputs (2048 → 4096)", "real time and streaming use", "39 MB download"),
        tag="LITE",
    ),
)


def build(results_file: Path) -> Image.Image:
    results = json.loads(results_file.read_text())
    card = Card()
    style.draw_background(card)
    overlay = card.overlay
    margin = style.MARGIN

    style.draw_brand(overlay, (margin + 8, margin + 6), 24)
    overlay.text((margin + 8, margin + 62), "Three models, one node", style.DISPLAY, 52, style.TEXT)
    overlay.text(
        (margin + 8, margin + 136),
        "For Wan 2.1 VAE latents: Wan 2.1, Wan 2.2 14B, Qwen-Image, Krea 2. MIT license, downloaded by the node on first use.",
        style.SANS,
        23,
        style.TEXT_DIM,
    )

    gap = 40
    panel_top = margin + 240
    panel_bottom = card.height - margin - 40
    panel_width = (card.width - 2 * margin - 16 - 2 * gap) / 3
    for index, model in enumerate(MODELS):
        metrics = results[model.benchmark_key]["metrics"]
        left = margin + 8 + index * (panel_width + gap)
        right = left + panel_width
        color = style.ACCENT if model.default else style.TEXT_DIM
        overlay.rectangle(
            (left, panel_top, right, panel_bottom),
            fill=style.PANEL,
            fill_alpha=0.85,
            outline=style.ACCENT if model.default else style.BORDER,
            width=1.5 if model.default else 1,
        )
        style.draw_crosshairs(
            overlay,
            [(left, panel_top), (right, panel_top), (left, panel_bottom), (right, panel_bottom)],
            8,
            color,
        )
        x = left + 44
        overlay.text((x, panel_top + 44), model.tag, style.MONO_MEDIUM, 17, color, "lt", tracking=2)
        style.draw_mark(
            overlay,
            (right - 44 - 46, panel_top + 40),
            46,
            style.BORDER,
            style.ACCENT if model.default else style.DASH,
            1.2,
        )
        overlay.text((x, panel_top + 92), model.title, style.DISPLAY, 50, style.TEXT, "lt")
        overlay.text((x, panel_top + 162), model.file_name, style.MONO, 16, style.MUTED, "lt")

        stats = (
            ("parameters", model.parameters),
            ("steps", model.steps),
            ("file", model.file_size),
            ("patch FID", f"{metrics['patch_fid']:.2f}"),
            ("latency 1024 px", f"{metrics['latency_ms']:.0f} ms"),
        )
        stats_top = panel_top + 220
        for row, (label, value) in enumerate(stats):
            y = stats_top + row * STAT_ROW_HEIGHT
            overlay.line([(x, y), (right - 44, y)], style.GRID, dash=(4, 4))
            overlay.text((x, y + STAT_ROW_HEIGHT / 2), label, style.SANS, 23, style.TEXT_DIM, "lm")
            value_color = (
                style.ACCENT
                if model.default and label in ("patch FID", "latency 1024 px")
                else style.TEXT
            )
            overlay.text(
                (right - 44, y + STAT_ROW_HEIGHT / 2),
                value,
                style.MONO_MEDIUM,
                26,
                value_color,
                "rm",
            )
        overlay.line(
            [
                (x, stats_top + len(stats) * STAT_ROW_HEIGHT),
                (right - 44, stats_top + len(stats) * STAT_ROW_HEIGHT),
            ],
            style.GRID,
            dash=(4, 4),
        )
        uses_top = panel_bottom - 44 - len(model.uses) * 38 - 40
        overlay.line([(x, uses_top), (right - 44, uses_top)], style.GRID, dash=(4, 4))
        for row, use in enumerate(model.uses):
            overlay.text(
                (x, uses_top + 40 + row * 38),
                ("→ " if row == 0 else "   ") + use,
                style.SANS,
                23,
                style.TEXT if row == 0 else style.TEXT_DIM,
                "lt",
            )

    overlay.text(
        (card.width - margin, card.height - 44),
        "patch FID and latency: 512 → 1024 benchmark, VAE decode included, batch 1, RTX 5090",
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
    style.save_card(build(arguments.results_file), arguments.output_directory, "models_lineup")


if __name__ == "__main__":
    main()
