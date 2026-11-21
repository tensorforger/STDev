"""
How it works card: the usual latent upscale + hires fix pipeline (second sampling pass of the base model) against the
Steppe pipeline (one step, decoded directly), latents drawn as grids on tilted planes, plus three model facts.

Run: python scripts/cards/make_how_it_works.py
"""

from dataclasses import dataclass

import style
from PIL import Image
from style import Card, Layer

NODE_HEIGHT = 76
NODE_FONT_SIZE = 24
PLANE_SKEW = 0.32


def plane_point(origin: style.Point, size: float, u: float, v: float) -> style.Point:
    """unit square (u right, v down) on a tilted plane: the right edge rises, the depth is shortened"""
    x0, y0 = origin
    return (x0 + u * size + v * size * PLANE_SKEW, y0 + v * size * 0.62 - u * size * 0.18)


def draw_latent(
    layer: Layer,
    origin: style.Point,
    size: float,
    cells: int,
    color: style.Color,
    label: str,
    blurred: bool = False,
) -> float:
    """
    A latent as a grid on a tilted plane. blurred: dashed, faded inner lines (an interpolated latent).
    returns: right edge x
    """
    corners = [plane_point(origin, size, u, v) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))]
    layer.polygon(corners, style.PANEL, 0.9)
    for line_index in range(1, cells):
        fraction = line_index / cells
        start, end = plane_point(origin, size, fraction, 0), plane_point(origin, size, fraction, 1)
        layer.line(
            [start, end], color, 1, 0.5 if blurred else 0.55, dash=(3, 4) if blurred else None
        )
        start, end = plane_point(origin, size, 0, fraction), plane_point(origin, size, 1, fraction)
        layer.line(
            [start, end], color, 1, 0.5 if blurred else 0.55, dash=(3, 4) if blurred else None
        )
    layer.line([*corners, corners[0]], color, 1.5, 0.9)
    bottom = max(point[1] for point in corners)
    center_x = (corners[0][0] + corners[2][0]) / 2
    layer.text((center_x, bottom + 22), label, style.MONO, 15, style.MUTED, "mt")
    return max(point[0] for point in corners)


def node_width(text: str) -> float:
    return style.text_width(text, style.MONO_MEDIUM, NODE_FONT_SIZE) + 56


def draw_node(
    layer: Layer,
    x: float,
    y: float,
    text: str,
    color: style.Color,
    note: str | None = None,
    filled: bool = False,
) -> float:
    """returns: right edge x"""
    width = node_width(text)
    if filled:
        layer.rectangle(
            (x, y - NODE_HEIGHT / 2, x + width, y + NODE_HEIGHT / 2),
            fill=color,
            fill_alpha=0.12,
            radius=6,
        )
    layer.rectangle(
        (x, y - NODE_HEIGHT / 2, x + width, y + NODE_HEIGHT / 2), outline=color, width=1.5, radius=6
    )
    layer.text((x + width / 2, y), text, style.MONO_MEDIUM, NODE_FONT_SIZE, color, "mm")
    if note:
        layer.text(
            (x + width / 2, y + NODE_HEIGHT / 2 + 18), note, style.SANS, 19, style.MUTED, "mt"
        )
    return x + width


@dataclass(frozen=True)
class Node:
    text: str
    color: style.Color
    note: str | None = None
    filled: bool = False

    def width(self) -> float:
        return node_width(self.text)

    def draw(self, layer: Layer, x: float, y: float) -> float:
        return draw_node(layer, x, y, self.text, self.color, self.note, self.filled)


@dataclass(frozen=True)
class Latent:
    size: float
    cells: int
    color: style.Color
    label: str
    blurred: bool = False

    def width(self) -> float:
        return self.size * (1 + PLANE_SKEW)

    def draw(self, layer: Layer, x: float, y: float) -> float:
        return draw_latent(
            layer,
            (x, y - self.size * 0.22),
            self.size,
            self.cells,
            self.color,
            self.label,
            self.blurred,
        )


def draw_row(
    layer: Layer,
    left: float,
    right: float,
    y: float,
    elements: list[Node | Latent],
    arrow_colors: list[style.Color],
) -> None:
    """elements spread over the full width, arrows between them"""
    spacing = (right - left - sum(element.width() for element in elements)) / (len(elements) - 1)
    x = left
    for index, element in enumerate(elements):
        end = element.draw(layer, x, y)
        if index < len(elements) - 1:
            draw_arrow(layer, end + 10, end + spacing - 10, y, arrow_colors[index])
        x += element.width() + spacing


def draw_arrow(layer: Layer, start_x: float, end_x: float, y: float, color: style.Color) -> None:
    layer.line([(start_x + 8, y), (end_x - 10, y)], color, 1.5, 0.8)
    layer.polygon([(end_x - 4, y), (end_x - 14, y - 6), (end_x - 14, y + 6)], color, 0.9)


def build() -> Image.Image:
    card = Card()
    style.draw_background(card)
    overlay = card.overlay
    margin = style.MARGIN

    style.draw_brand(overlay, (margin + 8, margin + 6), 24)
    overlay.text(
        (margin + 8, margin + 62),
        "A generative upscaler in latent space",
        style.DISPLAY,
        52,
        style.TEXT,
    )
    overlay.text(
        (margin + 8, margin + 136),
        "Steppe is a small generative model trained only for upscaling. Its output latent goes straight to the VAE decoder.",
        style.SANS,
        23,
        style.TEXT_DIM,
    )

    row_left, row_right = margin + 8, card.width - margin - 8
    usual_y, steppe_y = margin + 420, margin + 790
    for row_y, title, color in (
        (usual_y, "USUAL: LATENT UPSCALE + HIRES FIX", style.TEXT_DIM),
        (steppe_y, "STEPPE", style.ACCENT),
    ):
        overlay.text(
            (row_left, row_y - 150), title, style.MONO_MEDIUM, 18, color, "ld", tracking=1.5
        )
        overlay.line([(row_left, row_y - 132), (row_right, row_y - 132)], style.GRID, dash=(4, 5))

    usual = [
        Node("KSampler", style.TEXT_DIM, "base model"),
        Latent(140, 6, style.TEXT_DIM, "latent 64×64"),
        Node("Latent Upscale", style.OTHER, "interpolation"),
        Latent(200, 12, style.OTHER, "latent 128×128, blurry", blurred=True),
        Node("KSampler again", style.TEXT_DIM, "base model, high resolution", filled=True),
        Node("VAE Decode", style.TEXT_DIM),
    ]
    draw_row(overlay, row_left, row_right, usual_y, usual, [style.DASH] * 5)

    steppe = [
        Node("KSampler", style.TEXT_DIM, "base model"),
        Latent(140, 6, style.TEXT_DIM, "latent 64×64"),
        Node("Steppe 2×", style.ACCENT, "1 step · ~45 ms at 1024 px", filled=True),
        Latent(200, 12, style.ACCENT, "latent 128×128, detailed"),
        Node("VAE Decode", style.TEXT_DIM, "no second pass"),
    ]
    steppe_right = row_right - (node_width("KSampler again") + 120)
    draw_row(
        overlay,
        row_left,
        steppe_right,
        steppe_y,
        steppe,
        [style.DASH, style.DASH, style.ACCENT, style.ACCENT],
    )
    overlay.text(
        (row_right, steppe_y), "decoded directly", style.DISPLAY_LIGHT, 34, style.ACCENT, "rm"
    )

    facts = (
        (
            "WINDOW TRANSFORMER",
            "on the full resolution latent grid,",
            "shifted 8×8 windows, 200M or 20M",
        ),
        ("RESIDUAL FLOW", "generates the detail over the upsampled", "latent with rectified flow"),
        (
            "ONE STEP",
            "distilled from the 20 step base model,",
            "images and videos, 2× frame rate too",
        ),
    )
    facts_top = card.height - margin - 190
    column_width = (card.width - 2 * margin - 16) / 3
    for index, (title, line_one, line_two) in enumerate(facts):
        x = margin + 8 + index * column_width
        overlay.rectangle(
            (x, facts_top, x + column_width - 24, card.height - margin - 16),
            fill=style.PANEL,
            outline=style.BORDER,
            fill_alpha=0.8,
        )
        overlay.text(
            (x + 28, facts_top + 30), title, style.MONO_MEDIUM, 18, style.ACCENT, "lt", tracking=1.5
        )
        overlay.text((x + 28, facts_top + 76), line_one, style.SANS, 22, style.TEXT_DIM, "lt")
        overlay.text((x + 28, facts_top + 110), line_two, style.SANS, 22, style.TEXT_DIM, "lt")
    style.draw_card_frame(card)
    return card.render()


def main() -> None:
    arguments = style.standard_arguments(__doc__).parse_args()
    style.save_card(build(), arguments.output_directory, "how_it_works")


if __name__ == "__main__":
    main()
