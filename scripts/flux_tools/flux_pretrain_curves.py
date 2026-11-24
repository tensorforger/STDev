"""
Hour window averages of the FLUX.2 pretrain training scalars (tensorboard) and the visual benchmark trend.
Usage: python scripts/flux_tools/flux_pretrain_curves.py RUN_NAME
"""

import json
import sys
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from stdev.flux_pretrain.config import run_paths

WINDOW_EDGES = (0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24)


def main() -> None:
    paths = run_paths(sys.argv[1])
    accumulator = EventAccumulator(str(paths.tensorboard / "train"), size_guidance={"scalars": 0})
    accumulator.Reload()
    hours_by_step = {event.step: event.value for event in accumulator.Scalars("train/training_hours")}
    last_hours = max(hours_by_step.values())
    edges = [edge for edge in WINDOW_EDGES if edge <= last_hours + 0.5]
    windows = list(zip([0.0] + edges[:-1], edges))
    print("window_end_hours", " ".join(f"{end:>8}" for _, end in windows))
    for tag in sorted(accumulator.Tags()["scalars"]):
        if tag.startswith(("data_produced", "train/units_seen", "train/samples_seen", "data/produced_total")):
            continue
        sums: dict[int, list[float]] = defaultdict(list)
        maxima: dict[int, float] = defaultdict(float)
        for event in accumulator.Scalars(tag):
            hours = hours_by_step.get(event.step)
            if hours is None:
                continue
            for index, (start, end) in enumerate(windows):
                if start <= hours < end:
                    sums[index].append(event.value)
                    maxima[index] = max(maxima[index], event.value)
        reduce_max = tag.endswith("_max")
        values = [
            (maxima[index] if reduce_max else sum(sums[index]) / len(sums[index])) if sums[index] else float("nan")
            for index in range(len(windows))
        ]
        print(f"{tag:<52}", " ".join(f"{value:8.4g}" for value in values))
    results_path = paths.benchmarks / "results.jsonl"
    if results_path.exists():
        print("visual benchmark (hours: raw / ema low_res_mse 1e-4, chroma dE00, latent_mse_hr)")
        for line in results_path.read_text().splitlines():
            result = json.loads(line)
            print(
                f"  {result['training_seconds'] / 3600:5.2f} h  "
                f"lr_mse {1e4 * result['raw_low_res_mse']:.2f} / {1e4 * result['ema_low_res_mse']:.2f}  "
                f"chroma {result['raw_chroma_ciede2000']:.3f} / {result['ema_chroma_ciede2000']:.3f}  "
                f"latent {result['raw_latent_mse_hr']:.4f} / {result['ema_latent_mse_hr']:.4f}  "
                f"({result['benchmark_seconds']:.0f} s)"
            )
    references_path = paths.benchmarks / "references.json"
    if references_path.exists():
        print("references", references_path.read_text().replace("\n", " "))


if __name__ == "__main__":
    main()
