"""
Workflow card: the ComfyUI workflow screenshot at native resolution in the card frame, with the basic graph and the
example workflow names. The card size follows the screenshot (no resampling of the UI).

Run: python scripts/cards/make_workflow.py
"""

from pathlib import Path

import style
from PIL import Image
from style import Card

SCREENSHOT_FILE = "Screenshot_workflow.png"
HEADER_HEIGHT = 250
FOOTER_HEIGHT = 70


def build(input_directory: Path) -> Image.Image:
    screenshot = style.load_image(str(input_directory / SCREENSHOT_FILE))
    margin = style.MARGIN
    width = max(1920, screenshot.width + 2 * margin + 32)
    height = max(1080, screenshot.height + 2 * margin + HEADER_HEIGHT + FOOTER_HEIGHT + 40)
    card = Card((width, height))
    style.draw_background(card)
    overlay = card.overlay

    style.draw_brand(overlay, (margin + 8, margin + 6), 24)
    overlay.text(
        (margin + 8, margin + 58),
        "Two nodes between KSampler and VAE Decode",
        style.DISPLAY,
        46,
        style.TEXT,
    )
    overlay.text(
        (margin + 8, margin + 126),
        "Install ComfyUI-Steppe from the Manager, the models download on first use. Example: Krea 2 Turbo, 4× in two steps.",
        style.SANS,
        21,
        style.TEXT_DIM,
    )
    overlay.text(
        (margin + 8, margin + 166),
        "KSampler  →  Load Steppe Upscaler + Steppe Upscale Latent (× 2)  →  VAE Decode",
        style.MONO,
        18,
        style.ACCENT,
    )

    shot_x = (width - screenshot.width) / 2
    shot_y = margin + HEADER_HEIGHT + 20
    window = (
        shot_x - 12,
        shot_y - 40,
        shot_x + screenshot.width + 12,
        shot_y + screenshot.height + 12,
    )
    card.background.rectangle(window, fill=style.PANEL, outline=style.BORDER, radius=8)
    for index in range(3):
        overlay.circle(
            (window[0] + 22 + index * 20, window[1] + 20), 5, outline=style.DASH, width=1.2
        )
    overlay.text(
        ((window[0] + window[2]) / 2, window[1] + 20),
        "krea_2_steppe_image_upscale_4x.json",
        style.MONO,
        15,
        style.MUTED,
        "mm",
    )
    card.paste(screenshot, (shot_x, shot_y))
    card.overlay.rectangle(
        (shot_x - 1, shot_y - 1, shot_x + screenshot.width + 1, shot_y + screenshot.height + 1),
        outline=style.GRID,
    )

    overlay.text(
        (margin + 8, height - margin + 4),
        "example_workflows/  ·  krea_2_steppe_image_upscale_4x.json  ·  wan_2.1_steppe_video_upscale_spatiotemporal.json",
        style.MONO,
        16,
        style.MUTED,
        "ld",
    )
    style.draw_card_frame(card)
    return card.render()


def main() -> None:
    arguments = style.standard_arguments(__doc__).parse_args()
    style.save_card(build(arguments.input_directory), arguments.output_directory, "workflow")


if __name__ == "__main__":
    main()
