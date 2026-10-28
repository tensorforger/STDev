"""
Table of all benchmarked runs: patch FID, FID, mse, latency, memory, training steps, final loss.

Usage: python dev/summarize_results.py [RUN_ID ...]
"""

import json
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def read_training_log(run_name: str) -> tuple[str, str, str]:
    log_path = REPOSITORY_ROOT / "train_logs" / run_name / "train.log"
    if not log_path.exists():
        return "-", "-", "-"
    text = log_path.read_text(errors="ignore")
    steps = re.findall(r"saved weights after (\d+) steps", text)
    losses = re.findall(r"step \d+ loss ([0-9.]+)", text)
    parameters = re.findall(r"([0-9.]+)M parameters", text)
    return (
        steps[-1] if steps else "-",
        losses[-1] if losses else "-",
        parameters[-1] if parameters else "-",
    )


def main() -> None:
    selected = set(sys.argv[1:])
    rows = []
    for metrics_path in sorted((REPOSITORY_ROOT / "raw_results").glob("run_*/metrics.json")):
        run_name = metrics_path.parent.name
        if selected and run_name[4:] not in selected:
            continue
        metrics = json.loads(metrics_path.read_text())
        quality = metrics["quality"]["flux_1024"]
        efficiency = metrics["efficiency"]
        steps, loss, parameters = read_training_log(run_name)
        rows.append(
            [
                run_name[4:],
                f"{quality.get('fid_patch', float('nan')):.2f}",
                f"{quality['fid']:.2f}",
                f"{quality['mse_hr_image'] * 1000:.2f}",
                f"{quality['mse_lr_image'] * 1000:.2f}",
                f"{efficiency['latency_ms_avg_512']:.2f}",
                f"{efficiency['latency_ms_avg_1024']:.2f}",
                f"{efficiency['latency_ms_avg_2048']:.2f}",
                f"{efficiency['cuda_mem_peak_1024']:.0f}",
                f"{efficiency['cuda_mem_peak_2048']:.0f}",
                parameters,
                steps,
                loss,
            ]
        )
    header = [
        "run", "fid_patch", "fid", "mse_hr*1e3", "mse_lr*1e3", "lat512", "lat1024", "lat2048",
        "mem1024", "mem2048", "params_M", "steps", "loss",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
