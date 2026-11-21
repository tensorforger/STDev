"""
Benchmark card: latency (log) against patch FID (log, lower is better, drawn upwards) of the 2x upscaling solutions
from the unified Wan 2.1 benchmark. Steppe points in the accent color with their Pareto curve, other solutions grey.

Run: python scripts/cards/make_benchmark_plot.py
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import style
from PIL import Image
from style import Card

RESULTS_FILE = style.STDEV_ROOT / "raw_results" / "upscale_benchmark_axis_v1_1k" / "results.json"


@dataclass(frozen=True)
class Entry:
    key: str
    label: str
    ours: bool
    offset: tuple[float, float]
    horizontal: str
    vertical: str


LABEL_SIZE = 21
VALUE_SIZE = 15
LABEL_BLOCK_HEIGHT = 48

ENTRIES = (
    Entry("ours_base_20_steps", "Steppe 200M base · 20 steps", True, (0, -22), "m", "above"),
    Entry("ours_distilled_1_step_200m", "Steppe 200M · 1 step", True, (0, -22), "m", "above"),
    Entry("ours_distilled_1_step_20m", "Steppe 20M · 1 step", True, (18, 14), "l", "below"),
    Entry(
        "ours_distilled_1_step_20m_tiny_vae",
        "Steppe 20M + tiny VAE",
        True,
        (-10, -20),
        "m",
        "above",
    ),
    Entry(
        "ours_distilled_1_step_2m_tiny_vae", "Steppe 2M + tiny VAE", True, (-20, 0), "r", "middle"
    ),
    Entry(
        "spacepxl_wan21_vae_upscale2x",
        "Wan2.1-VAE-upscale2x decoder",
        False,
        (20, 0),
        "l",
        "middle",
    ),
    Entry("decode_span", "SPAN", False, (20, 0), "l", "middle"),
    Entry("decode_bicubic", "bicubic", False, (0, 18), "m", "below"),
    Entry("decode_realesrgan", "Real-ESRGAN", False, (20, 0), "l", "middle"),
    Entry("sesquilsr", "SesquiLSR (latent)", False, (20, 0), "l", "middle"),
    Entry("wan_nn_latent_upscaler", "NN latent upscaler", False, (20, 0), "l", "middle"),
)
PARETO_KEYS = (
    "ours_distilled_1_step_2m_tiny_vae",
    "ours_distilled_1_step_20m_tiny_vae",
    "ours_distilled_1_step_20m",
    "ours_distilled_1_step_200m",
    "ours_base_20_steps",
)
LATENCY_RANGE = (10.0, 2000.0)
FID_RANGE = (3.0, 70.0)
LATENCY_TICKS = (10, 20, 50, 100, 200, 500, 1000)
FID_TICKS = (4, 6, 10, 20, 40)


class LogAxes:
    """log-log mapping of (latency, patch FID) into a plot box; lower FID is drawn higher"""

    def __init__(self, box: style.Box) -> None:
        self.box = box

    def x(self, latency: float) -> float:
        left, _, right, _ = self.box
        low, high = (math.log10(value) for value in LATENCY_RANGE)
        return left + (math.log10(latency) - low) / (high - low) * (right - left)

    def y(self, fid: float) -> float:
        _, top, _, bottom = self.box
        low, high = (math.log10(value) for value in FID_RANGE)
        return top + (math.log10(fid) - low) / (high - low) * (bottom - top)

    def point(self, latency: float, fid: float) -> style.Point:
        return (self.x(latency), self.y(fid))


def build(results_file: Path) -> Image.Image:
    results = json.loads(results_file.read_text())
    metrics = {entry.key: results[entry.key]["metrics"] for entry in ENTRIES}
    card = Card()
    style.draw_background(card)
    overlay = card.overlay
    margin = style.MARGIN

    style.draw_brand(overlay, (margin + 8, margin + 6), 24)
    overlay.text(
        (margin + 8, margin + 62),
        "The best quality of all 2× upscalers we tested",
        style.DISPLAY,
        52,
        style.TEXT,
    )
    overlay.text(
        (margin + 8, margin + 136),
        "1000 photos, 512 → 1024 from the Wan 2.1 latent only (no prompt). Latency: batch 1, VAE decode included, RTX 5090.",
        style.SANS,
        23,
        style.TEXT_DIM,
    )

    plot_box = (margin + 120, margin + 250, card.width - margin - 40, card.height - margin - 90)
    axes = LogAxes(plot_box)
    left, top, right, bottom = plot_box
    overlay.rectangle((left, top, right, bottom), fill=style.BACKGROUND, fill_alpha=0.65)
    for tick in LATENCY_TICKS:
        x = axes.x(tick)
        overlay.line([(x, top), (x, bottom)], style.GRID, dash=(5, 6))
        overlay.text(
            (x, bottom + 16),
            f"{tick} ms" if tick < 1000 else "1 s",
            style.MONO,
            17,
            style.MUTED,
            "mt",
        )
    for tick in FID_TICKS:
        y = axes.y(tick)
        overlay.line([(left, y), (right, y)], style.GRID, dash=(5, 6))
        overlay.text((left - 16, y), str(tick), style.MONO, 17, style.MUTED, "rm")
    overlay.rectangle((left, top, right, bottom), outline=style.BORDER)
    style.draw_crosshairs(
        overlay, [(left, top), (right, top), (left, bottom), (right, bottom)], 8, style.MUTED
    )
    overlay.text(
        (right, bottom + 52),
        "LATENCY  →  slower",
        style.MONO_MEDIUM,
        16,
        style.MUTED,
        "rt",
        tracking=1.5,
    )
    overlay.text(
        (left - 16, top - 22),
        "PATCH FID  ·  lower is better  ↑",
        style.MONO_MEDIUM,
        16,
        style.MUTED,
        "ld",
        tracking=1.5,
    )

    bicubic_fid = metrics["decode_bicubic"]["patch_fid"]
    overlay.line([(left, axes.y(bicubic_fid)), (right, axes.y(bicubic_fid))], style.OTHER, 1, 0.5)

    pareto_points = [
        axes.point(metrics[key]["latency_ms"], metrics[key]["patch_fid"]) for key in PARETO_KEYS
    ]
    overlay.line(pareto_points, style.ACCENT, 2.0, 0.8)

    for entry in ENTRIES:
        values = metrics[entry.key]
        x, y = axes.point(values["latency_ms"], values["patch_fid"])
        if entry.ours:
            overlay.circle((x, y), 12, fill=style.ACCENT, alpha=0.18)
            overlay.circle((x, y), 6.5, fill=style.ACCENT)
        else:
            overlay.rectangle((x - 6, y - 6, x + 6, y + 6), fill=style.BACKGROUND)
            overlay.rectangle((x - 6, y - 6, x + 6, y + 6), outline=style.OTHER, width=1.5)
        block_x = x + entry.offset[0]
        block_top = {
            "above": y + entry.offset[1] - LABEL_BLOCK_HEIGHT,
            "below": y + entry.offset[1],
            "middle": y + entry.offset[1] - LABEL_BLOCK_HEIGHT / 2,
        }[entry.vertical]
        color = style.TEXT if entry.ours else style.TEXT_DIM
        label_font = style.SANS_MEDIUM if entry.ours else style.SANS
        overlay.text(
            (block_x, block_top), entry.label, label_font, LABEL_SIZE, color, entry.horizontal + "t"
        )
        value_text = f"{values['patch_fid']:.2f}  ·  {values['latency_ms']:.0f} ms"
        value_color = style.ACCENT if entry.ours else style.MUTED
        overlay.text(
            (block_x, block_top + LABEL_BLOCK_HEIGHT),
            value_text,
            style.MONO,
            VALUE_SIZE,
            value_color,
            entry.horizontal + "d",
        )

    overlay.text(
        (right - 14, axes.y(bicubic_fid) - 8), "bicubic level", style.MONO, 15, style.MUTED, "rd"
    )

    reduction = 1 - metrics["ours_distilled_1_step_200m"]["patch_fid"] / bicubic_fid
    callout_x, callout_y = right - 500, axes.y(22)
    overlay.rectangle(
        (callout_x, callout_y, right - 30, callout_y + 132), fill=style.PANEL, outline=style.BORDER
    )
    overlay.text(
        (callout_x + 24, callout_y + 20),
        f"−{reduction * 100:.0f}%",
        style.DISPLAY_MEDIUM,
        54,
        style.ACCENT,
    )
    overlay.text(
        (callout_x + 24, callout_y + 92),
        "patch FID vs bicubic, 1 step",
        style.SANS,
        20,
        style.TEXT_DIM,
    )

    legend_x, legend_y = left + 30, top + 34
    overlay.circle((legend_x, legend_y), 6.5, fill=style.ACCENT)
    overlay.text((legend_x + 20, legend_y), "Steppe", style.SANS_MEDIUM, 20, style.TEXT, "lm")
    overlay.rectangle(
        (legend_x + 130, legend_y - 6, legend_x + 142, legend_y + 6), outline=style.OTHER, width=1.5
    )
    overlay.text(
        (legend_x + 156, legend_y), "other 2× solutions", style.SANS, 20, style.TEXT_DIM, "lm"
    )

    overlay.text(
        (card.width - margin, card.height - 44),
        "Axis-V1 photos (CC BY 4.0) · patch FID on native 256 px tiles · eager bf16 · latent bicubic (patch FID 145) not shown",
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
    style.save_card(build(arguments.results_file), arguments.output_directory, "benchmark_plot")


if __name__ == "__main__":
    main()
