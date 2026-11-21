"""
Autoresearch card: patch FID against experiment number for both agent-driven research phases (FLUX.2 latents, Wan 2.1
latents), every run as a dot and the best result so far as a step line. Phase 2 ends with round 10 (the later rounds
tune batch size, stability and memory, not quality).

Run: python scripts/cards/make_autoresearch.py
"""

import itertools
import math
from dataclasses import dataclass

import style
from autoresearch_data import Run, load_phase
from PIL import Image
from style import Card

PHASE_2_LAST_ROUND = 10


@dataclass(frozen=True)
class Panel:
    phase: int
    title: str
    subtitle: str
    fid_range: tuple[float, float]
    fid_ticks: tuple[float, ...]
    annotations: dict[int, str]


PANELS = (
    Panel(
        1,
        "PHASE 1 · FLUX.2 LATENTS",
        "512 → 1024 images, 30 min runs + one 2 h round",
        (1.2, 8.0),
        (1.5, 2, 3, 4, 6),
        {
            9: "logit-normal training sigmas",
            16: "LR decay recipe",
            26: "capacity at full resolution",
            63: "window attention",
            77: "2 h: deep full-res decoder",
        },
    ),
    Panel(
        2,
        "PHASE 2 · WAN 2.1 LATENTS",
        "512 → 1024 images + 2× frame rate videos, 30 min + 2 h runs",
        (3.0, 60.0),
        (4, 6, 10, 20, 40),
        {
            2: "learned source encoder",
            7: "transformer on the residual flow",
            12: "causal temporal attention",
            32: "2 h: width 384",
        },
    ),
)


ANNOTATION_SIZE = 19
ANNOTATION_HEIGHT = 26
CANDIDATE_DIRECTIONS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (1, 1), (0, 1), (-1, 1))
CANDIDATE_DISTANCES = (16, 40, 80, 130, 190)


def overlaps(
    box: style.Box, obstacles: list[tuple[float, float, float]], placed: list[style.Box]
) -> bool:
    left, top, right, bottom = box
    for x, y, radius in obstacles:
        if left - radius < x < right + radius and top - radius < y < bottom + radius:
            return True
    return any(
        left < other[2] and other[0] < right and top < other[3] and other[1] < bottom
        for other in placed
    )


def place_label(
    point: style.Point,
    width: float,
    height: float,
    bounds: style.Box,
    obstacles: list[tuple[float, float, float]],
    placed: list[style.Box],
) -> style.Box:
    """returns: the first label box around the point (nearest first, above-left preferred) free of obstacles"""
    x, y = point
    for distance in CANDIDATE_DISTANCES:
        for direction_x, direction_y in CANDIDATE_DIRECTIONS:
            box_left = (
                x
                + direction_x * distance
                - (width if direction_x < 0 else width / 2 if direction_x == 0 else 0)
            )
            box_top = (
                y
                + direction_y * distance
                - (height if direction_y < 0 else height / 2 if direction_y == 0 else 0)
            )
            box = (box_left, box_top, box_left + width, box_top + height)
            inside = (
                bounds[0] <= box[0]
                and box[2] <= bounds[2]
                and bounds[1] <= box[1]
                and box[3] <= bounds[3]
            )
            if inside and not overlaps(box, obstacles, placed):
                return box
    return (x + 16, y - height - 16, x + 16 + width, y - 16)


def draw_panel(card: Card, box: style.Box, panel: Panel, runs: list[Run]) -> None:
    overlay = card.overlay
    left, top, right, bottom = box
    overlay.rectangle(box, fill=style.BACKGROUND, fill_alpha=0.7)
    first_id, last_id = runs[0].run_id, runs[-1].run_id
    low_fid, high_fid = (math.log10(value) for value in panel.fid_range)
    inner_left, inner_right = left + 40, right - 40

    def x_of(run_id: float) -> float:
        return inner_left + (run_id - first_id) / (last_id - first_id) * (inner_right - inner_left)

    def y_of(fid: float) -> float:
        return top + (math.log10(fid) - low_fid) / (high_fid - low_fid) * (bottom - top)

    for tick in panel.fid_ticks:
        overlay.line([(left, y_of(tick)), (right, y_of(tick))], style.GRID, dash=(5, 6))
        overlay.text((left - 14, y_of(tick)), f"{tick:g}", style.MONO, 16, style.MUTED, "rm")
    for run_id in range(0, last_id + 1, 10):
        overlay.line([(x_of(run_id), top), (x_of(run_id), bottom)], style.GRID_FAINT)
        overlay.text((x_of(run_id), bottom + 14), str(run_id), style.MONO, 16, style.MUTED, "mt")
    overlay.rectangle(box, outline=style.BORDER)
    style.draw_crosshairs(
        overlay, [(left, top), (right, top), (left, bottom), (right, bottom)], 8, style.MUTED
    )

    best = float("inf")
    steps: list[style.Point] = []
    records: list[Run] = []
    for run in runs:
        if run.fid_patch < best:
            best = run.fid_patch
            if steps:
                steps.append((x_of(run.run_id), steps[-1][1]))
            steps.append((x_of(run.run_id), y_of(best)))
            records.append(run)
    steps.append((x_of(last_id), steps[-1][1]))
    overlay.line(steps, style.ACCENT, 2.0, 0.85)

    record_ids = {run.run_id for run in records}
    for run in runs:
        point = (x_of(run.run_id), y_of(run.fid_patch))
        if run.run_id in record_ids:
            overlay.circle(point, 10, fill=style.ACCENT, alpha=0.18)
            overlay.circle(point, 5.5, fill=style.ACCENT)
        elif run.long_budget:
            overlay.circle(point, 5.5, outline=style.ACCENT, width=1.5, alpha=0.7)
        else:
            overlay.circle(point, 4.5, fill=style.OTHER, alpha=0.75)
        if run.long_budget and run.run_id in record_ids:
            overlay.circle(point, 9, outline=style.ACCENT, width=1.5)

    obstacles = [(x_of(run.run_id), y_of(run.fid_patch), 9.0) for run in runs]
    for (start_x, start_y), (end_x, end_y) in itertools.pairwise(steps):
        samples = max(1, int(math.hypot(end_x - start_x, end_y - start_y) / 6))
        for sample in range(samples + 1):
            fraction = sample / samples
            obstacles.append(
                (
                    start_x + (end_x - start_x) * fraction,
                    start_y + (end_y - start_y) * fraction,
                    4.0,
                )
            )
    placed: list[style.Box] = []
    by_id = {run.run_id: run for run in runs}
    for run_id, label in panel.annotations.items():
        run = by_id[run_id]
        point = (x_of(run_id), y_of(run.fid_patch))
        value = f"{run.fid_patch:g}"
        label_width = style.text_width(label, style.SANS, ANNOTATION_SIZE)
        width = label_width + 12 + style.text_width(value, style.MONO, ANNOTATION_SIZE - 3)
        label_box = place_label(
            point,
            width,
            ANNOTATION_HEIGHT,
            (left + 8, top + 8, right - 8, bottom - 8),
            obstacles,
            placed,
        )
        placed.append(label_box)
        box_left, box_top, box_right, box_bottom = label_box
        nearest_x = min(max(point[0], box_left), box_right)
        nearest_y = min(max(point[1], box_top), box_bottom)
        if math.hypot(nearest_x - point[0], nearest_y - point[1]) > 24:
            overlay.line([point, (nearest_x, nearest_y)], style.ACCENT, 1, 0.45, dash=(3, 3))
        middle_y = (box_top + box_bottom) / 2
        overlay.text((box_left, middle_y), label, style.SANS, ANNOTATION_SIZE, style.TEXT, "lm")
        overlay.text(
            (box_left + label_width + 12, middle_y),
            value,
            style.MONO,
            ANNOTATION_SIZE - 3,
            style.ACCENT,
            "lm",
        )
    overlay.text((left + 14, top + 14), "↑ better", style.MONO, 15, style.MUTED, "lt")

    overlay.text(
        (left, top - 58), panel.title, style.MONO_MEDIUM, 18, style.TEXT, "ld", tracking=1.5
    )
    overlay.text((left, top - 28), panel.subtitle, style.SANS, 19, style.TEXT_DIM, "ld")
    summary = f"{len(runs)} runs   {runs[0].fid_patch:g} → {best:g}"
    overlay.text((right, top - 28), summary, style.MONO_MEDIUM, 18, style.ACCENT, "rd")


def build() -> Image.Image:
    card = Card()
    style.draw_background(card)
    overlay = card.overlay
    margin = style.MARGIN
    phases = {
        panel.phase: load_phase(panel.phase, PHASE_2_LAST_ROUND if panel.phase == 2 else None)
        for panel in PANELS
    }
    total_runs = sum(len(runs) for runs in phases.values())

    style.draw_brand(overlay, (margin + 8, margin + 6), 24)
    overlay.text(
        (margin + 8, margin + 62),
        f"Designed by {total_runs} agent-driven experiments",
        style.DISPLAY,
        52,
        style.TEXT,
    )
    overlay.text(
        (margin + 8, margin + 136),
        "AI agents proposed, ran and judged rounds of parallel training runs. Patch FID on held-out benchmark images (lower is better).",
        style.SANS,
        23,
        style.TEXT_DIM,
    )

    panel_top = margin + 300
    panel_bottom = card.height - margin - 90
    gap = 150
    panel_width = (card.width - 2 * margin - 80 - gap) / 2
    for index, panel in enumerate(PANELS):
        left = margin + 80 + index * (panel_width + gap)
        draw_panel(
            card, (left, panel_top, left + panel_width, panel_bottom), panel, phases[panel.phase]
        )

    legend_y = card.height - 52
    legend_x = margin + 80
    overlay.circle((legend_x, legend_y), 4.5, fill=style.OTHER)
    overlay.text((legend_x + 16, legend_y), "run", style.SANS, 18, style.TEXT_DIM, "lm")
    overlay.circle((legend_x + 90, legend_y), 5.5, fill=style.ACCENT)
    overlay.text((legend_x + 106, legend_y), "new best", style.SANS, 18, style.TEXT_DIM, "lm")
    overlay.circle((legend_x + 230, legend_y), 5.5, outline=style.ACCENT, width=1.5)
    overlay.text((legend_x + 246, legend_y), "2 h run", style.SANS, 18, style.TEXT_DIM, "lm")
    overlay.line([(legend_x + 350, legend_y), (legend_x + 390, legend_y)], style.ACCENT, 2.0)
    overlay.text((legend_x + 402, legend_y), "best so far", style.SANS, 18, style.TEXT_DIM, "lm")
    overlay.text(
        (card.width - margin, legend_y),
        "8 / 4 parallel RTX 5090 runs per round · patch FID axes are log scale and differ between phases",
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
    style.save_card(build(), arguments.output_directory, "autoresearch")


if __name__ == "__main__":
    main()
