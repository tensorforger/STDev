"""
One line per dev/evaluations result: run, overrides, weights, patch FID, FID, mse.

Usage: python dev/summarize_evaluations.py [SUBSTRING]
"""

import json
import sys
from pathlib import Path

EVALUATIONS_DIRECTORY = Path(__file__).resolve().parent / "evaluations"


def main() -> None:
    substring = sys.argv[1] if len(sys.argv) > 1 else ""
    for path in sorted(EVALUATIONS_DIRECTORY.glob("*.json")):
        if substring not in path.name:
            continue
        result = json.loads(path.read_text())
        quality = result["quality"]
        print(
            f"{result['run']} samples={result['samples']} {' '.join(result['overrides'])} "
            f"{Path(result['weights_directory']).name if result['weights_directory'] else ''} "
            f"fid_patch={quality.get('fid_patch', float('nan')):.3f} "
            f"fid={quality.get('fid', float('nan')):.3f} "
            f"mse_lr={quality['mse_lr_image'] * 1e3:.3f} mse_hr={quality['mse_hr_image'] * 1e3:.3f}"
        )


if __name__ == "__main__":
    main()
