"""
Steppe-Flux.2 benchmark card: latency (log) against patch FID (log, lower is better, drawn upwards) of the 2x upscaling
solutions from the unified FLUX.2 benchmark. The plot fills the card, one small caption line at the bottom. Steppe
points in the accent color with their Pareto curve, other solutions grey. VOSR (a 4x restorer, unfair at 2x) and
Hildegard (no FID) are not shown.

Run: python scripts/cards/make_flux2_benchmark_plot.py [--results-file raw_results/flux_upscale_benchmark_axis_v1_1k/results.json]
"""

import json
import math
from pathlib import Path

import style
from make_benchmark_plot import Entry
from PIL import Image
from style import Card

RESULTS_FILE = style.STDEV_ROOT / "raw_results" / "flux_upscale_benchmark_axis_v1_1k" / "results.json"

PADDING = 32
FOOTER_HEIGHT = 64
LABEL_SIZE = 21
VALUE_SIZE = 15
LABEL_BLOCK_HEIGHT = 48

ENTRIES = (
    Entry("ours_base_30_steps_200m", "Steppe 200M base · 30 steps", True, (0, 22), "m", "below"),
    Entry("ours_distilled_1_step_200m", "Steppe 200M · 1 step", True, (0, -22), "m", "above"),
    Entry("ours_distilled_1_step_20m", "Steppe 20M · 1 step", True, (20, 8), "l", "below"),
    Entry(
        "ours_distilled_1_step_200m_tiny_vae",
        "Steppe 200M + TAEF2",
        True,
        (20, 8),
        "l",
        "below",
    ),
    Entry(
        "ours_distilled_1_step_20m_tiny_vae", "Steppe 20M + TAEF2", True, (-22, 0), "r", "middle"
    ),
    Entry("flow_upscaler", "FlowUpscaler", False, (22, 0), "l", "middle"),
    Entry("flow_upscaler_tiny_vae", "FlowUpscaler + TAEF2", False, (22, 4), "l", "below"),
    Entry("decode_span", "SPAN", False, (22, 0), "l", "middle"),
    Entry("sesquilsr", "SesquiLSR (latent)", False, (22, 0), "l", "middle"),
    Entry("decode_bicubic", "bicubic", False, (0, 16), "m", "below"),
    Entry("decode_realesrgan", "Real-ESRGAN", False, (22, 0), "l", "middle"),
)
PARETO_KEYS = (
    "ours_distilled_1_step_20m_tiny_vae",
    "ours_distilled_1_step_200m_tiny_vae",
    "ours_distilled_1_step_20m",
    "ours_distilled_1_step_200m",
    "ours_base_30_steps_200m",
)
LATENCY_RANGE = (10.0, 3000.0)
FID_RANGE = (1.4, 12.0)
LATENCY_TICKS = (10, 20, 50, 100, 200, 500, 1000, 2000)
FID_TICKS = (1.5, 2, 3, 4, 6, 8, 10)


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

    plot_box = (
        PADDING + 70,
        PADDING + 44,
        card.width - PADDING - 24,
        card.height - FOOTER_HEIGHT - 48,
    )
    axes = LogAxes(plot_box)
    left, top, right, bottom = plot_box
    overlay.rectangle((left, top, right, bottom), fill=style.BACKGROUND, fill_alpha=0.65)
    for tick in LATENCY_TICKS:
        x = axes.x(tick)
        overlay.line([(x, top), (x, bottom)], style.GRID, dash=(5, 6))
        overlay.text(
            (x, bottom + 12),
            f"{tick} ms" if tick < 1000 else f"{tick // 1000} s",
            style.MONO,
            16,
            style.MUTED,
            "mt",
        )
    for tick in FID_TICKS:
        y = axes.y(tick)
        overlay.line([(left, y), (right, y)], style.GRID, dash=(5, 6))
        overlay.text((left - 14, y), f"{tick:g}", style.MONO, 16, style.MUTED, "rm")
    overlay.rectangle((left, top, right, bottom), outline=style.BORDER)
    overlay.text(
        (right, top - 12),
        "LATENCY  →  slower",
        style.MONO_MEDIUM,
        15,
        style.MUTED,
        "rd",
        tracking=1.5,
    )
    overlay.text(
        (left, top - 12),
        "PATCH FID  ·  lower is better  ↑",
        style.MONO_MEDIUM,
        15,
        style.MUTED,
        "ld",
        tracking=1.5,
    )

    bicubic_fid = metrics["decode_bicubic"]["patch_fid"]
    overlay.line([(left, axes.y(bicubic_fid)), (right, axes.y(bicubic_fid))], style.OTHER, 1, 0.5)
    overlay.text(
        (right - 14, axes.y(bicubic_fid) - 8), "bicubic level", style.MONO, 15, style.MUTED, "rd"
    )

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

    reduction = 1 - metrics["ours_distilled_1_step_200m"]["patch_fid"] / bicubic_fid
    callout_x, callout_y = axes.x(400), axes.y(3.4)
    overlay.text(
        (callout_x, callout_y),
        f"−{reduction * 100:.0f}%",
        style.DISPLAY_MEDIUM,
        44,
        style.ACCENT,
        "ls",
    )
    overlay.text(
        (callout_x, callout_y + 14),
        "patch FID vs bicubic, 200M 1 step",
        style.SANS,
        18,
        style.TEXT_DIM,
        "lt",
    )

    legend_x, legend_y = left + 30, bottom - 34
    overlay.circle((legend_x, legend_y), 6.5, fill=style.ACCENT)
    overlay.text((legend_x + 20, legend_y), "Steppe-Flux.2", style.SANS_MEDIUM, 19, style.TEXT, "lm")
    other_x = legend_x + 20 + style.text_width("Steppe-Flux.2", style.SANS_MEDIUM, 19) + 36
    overlay.rectangle(
        (other_x - 6, legend_y - 6, other_x + 6, legend_y + 6), outline=style.OTHER, width=1.5
    )
    overlay.text(
        (other_x + 18, legend_y), "other 2× solutions", style.SANS, 19, style.TEXT_DIM, "lm"
    )

    footer_y = card.height - FOOTER_HEIGHT / 2
    brand_right = style.draw_brand(overlay, (PADDING, footer_y - 10), 20)
    overlay.text(
        (brand_right + 28, footer_y),
        "2× upscaling of the FLUX.2 latent only (no prompt), 512 → 1024, 1000 Axis-V1 photos (CC BY 4.0)"
        "   ·   patch FID on native 256 px tiles   ·   batch 1, VAE decode included, eager bf16, RTX 5090"
        "   ·   latent bicubic (114) not shown",
        style.MONO,
        15,
        style.MUTED,
        "lm",
    )
    return card.render()


def main() -> None:
    parser = style.standard_arguments(__doc__)
    parser.add_argument("--results-file", type=Path, default=RESULTS_FILE)
    arguments = parser.parse_args()
    style.save_card(
        build(arguments.results_file), arguments.output_directory, "flux2_benchmark_plot"
    )


if __name__ == "__main__":
    main()
