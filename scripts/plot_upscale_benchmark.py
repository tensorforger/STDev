"""
Plots the unified upscale benchmark: latency (log) against patch_fid (log, inverted axis, higher is better).
Latent upscalers are circles (ours colored, others grey), the other solutions (image upscalers, decoder
fine-tune, bicubic) are grey squares. A blue curve connects our solutions. Ground truth hff variants
(`*_gt_hff`) and solutions evaluated on a subset without patch_fid are not plotted.

Run:
    python scripts/plot_upscale_benchmark.py
    python scripts/plot_upscale_benchmark.py --results-directory raw_results/flux_upscale_benchmark_axis_v1_1k
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIRECTORY = REPOSITORY_ROOT / "raw_results" / "upscale_benchmark_axis_v1_1k"
OUR_PREFIX = "ours_"
GROUND_TRUTH_HFF_SUFFIX = "_gt_hff"
OTHER_LATENT_UPSCALERS = (
    "latent_bicubic_decode",
    "wan_nn_latent_upscaler",
    "sesquilsr",
    "flow_upscaler",
    "flow_upscaler_tiny_vae",
)
OUR_COLOR = "tab:blue"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-directory", type=Path, default=DEFAULT_RESULTS_DIRECTORY)
    parser.add_argument(
        "--output", type=Path, help="Defaults to <results directory>/latency_vs_patch_fid.png"
    )
    return parser.parse_args()


def plot_results(results: dict[str, dict], output_path: Path) -> None:
    results = {
        name: result
        for name, result in results.items()
        if not name.endswith(GROUND_TRUTH_HFF_SUFFIX) and result["metrics"]["patch_fid"] is not None
    }
    latencies = {name: result["metrics"]["latency_ms"] for name, result in results.items()}
    patch_fids = {name: result["metrics"]["patch_fid"] for name, result in results.items()}
    groups = {
        "Ours": [name for name in results if name.startswith(OUR_PREFIX)],
        "Other latent upscalers": [name for name in results if name in OTHER_LATENT_UPSCALERS],
        "Image upscalers, decoder fine-tune, bicubic": [
            name
            for name in results
            if not name.startswith(OUR_PREFIX) and name not in OTHER_LATENT_UPSCALERS
        ],
    }
    styles = {
        "Ours": (OUR_COLOR, "o"),
        "Other latent upscalers": ("grey", "o"),
        "Image upscalers, decoder fine-tune, bicubic": ("grey", "s"),
    }

    figure, axes = plt.subplots(figsize=(10, 6.5))
    for label, names in groups.items():
        color, marker = styles[label]
        names = sorted(names, key=latencies.get)
        is_ours = label == "Ours"
        axes.scatter(
            [latencies[name] for name in names],
            [patch_fids[name] for name in names],
            color=color,
            marker=marker,
            s=45,
            label=label,
            zorder=3,
        )
        if is_ours:
            axes.plot(
                [latencies[name] for name in names],
                [patch_fids[name] for name in names],
                color=color,
                linewidth=1.2,
                zorder=2,
            )
        for name in names:
            axes.annotate(
                name.removeprefix(OUR_PREFIX),
                (latencies[name], patch_fids[name]),
                xytext=(5, -10) if is_ours else (5, 4),
                textcoords="offset points",
                fontsize=7,
                color=color if is_ours else "dimgrey",
            )

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.invert_yaxis()
    axes.set_xlabel("Latency, ms (batch size 1, decode included, log scale)")
    axes.set_ylabel("Patch FID (inverted axis, higher is better, log scale)")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)


def main() -> None:
    arguments = parse_arguments()
    results = json.loads((arguments.results_directory / "results.json").read_text())
    output_path = arguments.output or arguments.results_directory / "latency_vs_patch_fid.png"
    plot_results(results, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
