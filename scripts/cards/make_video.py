"""
Side by side comparison videos (H.264 MP4): the input video (bicubic 2x, each frame repeated to the output frame
rate) next to the Steppe 2x resolution + 2x frame rate result, with labels in the card style.
- video_compare.mp4: the full frames
- video_compare_zoom.mp4: the same center region of both, zoomed so the detail is visible at README width

Run: python scripts/cards/make_video.py
"""

import subprocess
import tempfile
from pathlib import Path

import style
from style import Layer

LOW_RESOLUTION_FILE = "steppe_demo_low_res.mp4"
HIGH_RESOLUTION_FILE = "steppe_demo_high_res.mp4"
OUTPUT_SIZE = (1280, 1024)
DIVIDER = 8
ZOOM_CROP = (640, 512)
ZOOM_CENTER = (720, 560)


def probe(path: Path) -> tuple[int, int, str]:
    """returns: width, height, frame rate (ffprobe fraction)"""
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    width, height, rate = output.split(",")
    return int(width), int(height), rate


def make_labels(width: int, height: int, left_label: str, right_label: str, path: Path) -> None:
    """Transparent overlay: badges, divider, brand."""
    layer = Layer(width, height)
    half = (width - DIVIDER) // 2
    layer.rectangle((half, 0, half + DIVIDER, height), fill=style.BACKGROUND)
    layer.line([(half + DIVIDER / 2, 0), (half + DIVIDER / 2, height)], style.ACCENT, 1.5, 0.8)
    style.draw_badge(layer, (20, 20), left_label, style.TEXT_DIM, 20)
    style.draw_badge(layer, (half + DIVIDER + 20, 20), right_label, style.ACCENT, 20)
    brand_width = 34 * 1.55 + style.text_width("steppe", style.DISPLAY_MEDIUM, 34 * 0.95, 0.3)
    layer.rectangle(
        (width - brand_width - 44, height - 66, width - 12, height - 12),
        fill=style.BACKGROUND,
        fill_alpha=0.6,
        radius=4,
    )
    style.draw_brand(layer, (width - brand_width - 28, height - 56), 34)
    layer.flatten().save(path)


def encode(arguments: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *arguments], check=True)


def build_video(low_path: Path, high_path: Path, output_path: Path, zoom: bool) -> None:
    high_width, high_height, high_rate = probe(high_path)
    low_width, _, _ = probe(low_path)
    factor = high_width / low_width
    width, height = OUTPUT_SIZE
    if zoom:
        crop_width, crop_height = ZOOM_CROP
        left = ZOOM_CENTER[0] - crop_width // 2
        top = ZOOM_CENTER[1] - crop_height // 2
        high_filter = (
            f"crop={crop_width}:{crop_height}:{left}:{top},scale={width}:{height}:flags=lanczos"
        )
        low_filter = (
            f"crop={crop_width / factor:.0f}:{crop_height / factor:.0f}:{left / factor:.0f}:{top / factor:.0f},"
            f"scale={width}:{height}:flags=bicubic"
        )
        labels = (
            f"INPUT · BICUBIC · {low_width}×{high_height / factor:.0f} · 16 FPS · ZOOM",
            f"STEPPE · {high_width}×{high_height} · 32 FPS · ZOOM",
        )
    else:
        high_filter = f"scale={width}:{height}:flags=lanczos"
        low_filter = f"scale={width}:{height}:flags=bicubic"
        labels = (
            f"INPUT · {low_width}×{high_height / factor:.0f} · 16 FPS",
            f"STEPPE 2× · {high_width}×{high_height} · 32 FPS",
        )
    total_width = 2 * width + DIVIDER
    with tempfile.TemporaryDirectory() as temporary:
        labels_path = Path(temporary) / "labels.png"
        make_labels(total_width, height, labels[0], labels[1], labels_path)
        graph = (
            f"[0:v]fps={high_rate},{low_filter},pad={width + DIVIDER}:{height}:0:0:color=black[left];"
            f"[1:v]{high_filter}[right];"
            f"[left][right]hstack=inputs=2[stacked];"
            f"[stacked][2:v]overlay=0:0:format=auto:shortest=1,format=yuv420p[out]"
        )
        encode(
            [
                "-i",
                str(low_path),
                "-i",
                str(high_path),
                "-loop",
                "1",
                "-i",
                str(labels_path),
                "-filter_complex",
                graph,
                "-map",
                "[out]",
                "-shortest",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
    print(f"{output_path.name}: {total_width}x{height}, {output_path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    arguments = style.standard_arguments(__doc__).parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    low_path = arguments.input_directory / LOW_RESOLUTION_FILE
    high_path = arguments.input_directory / HIGH_RESOLUTION_FILE
    build_video(low_path, high_path, arguments.output_directory / "video_compare.mp4", zoom=False)
    build_video(
        low_path, high_path, arguments.output_directory / "video_compare_zoom.mp4", zoom=True
    )


if __name__ == "__main__":
    main()
