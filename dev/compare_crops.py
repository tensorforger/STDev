"""
Native resolution crops of the teacher image next to the outputs of several runs.

Usage: python dev/compare_crops.py OUTPUT.png RUN_ID [RUN_ID ...] [--samples N] [--crop 256]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "flux_1024" / "images"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--offset", type=int, default=0)
    arguments = parser.parse_args()

    first_directory = REPOSITORY_ROOT / "raw_results" / f"run_{arguments.runs[0]}" / "images_flux_1024"
    sample_names = sorted(path.name[: -len("_out.png")] for path in first_directory.glob("*_out.png"))
    sample_names = sample_names[arguments.offset : arguments.offset + arguments.samples]
    rows = []
    for sample_name in sample_names:
        teacher = cv2.imread(str(IMAGES_DIRECTORY / f"{sample_name}.png"))
        low_resolution = cv2.imread(str(first_directory / f"{sample_name}_in.png"))
        height, width = teacher.shape[:2]
        top, left = height // 2 - arguments.crop // 2, width // 2 - arguments.crop // 2
        crops = [low_resolution, teacher] + [
            cv2.imread(
                str(REPOSITORY_ROOT / "raw_results" / f"run_{run}" / "images_flux_1024" / f"{sample_name}_out.png")
            )
            for run in arguments.runs
        ]
        crops = [crop[top : top + arguments.crop, left : left + arguments.crop] for crop in crops]
        rows.append(np.concatenate(crops, axis=1))
    montage = np.concatenate(rows, axis=0)
    labels = ["LR bilinear", "teacher"] + arguments.runs
    for index, label in enumerate(labels):
        cv2.putText(montage, label, (index * arguments.crop + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(arguments.output), montage)


if __name__ == "__main__":
    main()
