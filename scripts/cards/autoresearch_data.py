"""
Parses the autoresearch experiment logs (markdown) into per run records: round, run name and the primary patch FID.

Run (prints the parsed records): python scripts/cards/autoresearch_data.py
"""

import re
from dataclasses import dataclass
from pathlib import Path

import style

PHASE_LOGS = {
    1: (style.STDEV_ROOT / "autoresearch" / "experiment_logs.md", "fid_patch"),
    2: (style.STDEV_ROOT / "autoresearch_2" / "experiment_logs.md", "img fid_patch"),
}
ROUND_HEADING = re.compile(r"^## Round (\d+)")
RUN_ID = re.compile(r"^(\d{4})")
NUMBER = re.compile(r"[\d.]+")


@dataclass
class Run:
    run_id: int
    round_number: int
    name: str
    fid_patch: float
    long_budget: bool


def parse_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_phase(log_path: Path, fid_column: str) -> list[Run]:
    """returns: runs with a primary metric, in run id order; long_budget = run from a 2 hour scaling round"""
    names: dict[int, str] = {}
    fids: dict[int, float] = {}
    rounds: dict[int, int] = {}
    long_rounds: set[int] = set()
    round_number = 0
    header: list[str] | None = None
    for line in log_path.read_text().splitlines():
        heading = ROUND_HEADING.match(line)
        if heading:
            round_number = int(heading.group(1))
            if "2 hour" in line or "2 h " in line:
                long_rounds.add(round_number)
        if not line.startswith("|"):
            header = None
            continue
        cells = parse_cells(line)
        if header is None:
            header = cells
            continue
        run_match = RUN_ID.match(cells[0])
        if not run_match:
            continue
        run_id = int(run_match.group(1))
        if header[0] == "Run ID" and len(header) > 1 and header[1] == "Run name":
            names.setdefault(run_id, cells[1])
            rounds.setdefault(run_id, round_number)
        if fid_column in header and len(cells) > header.index(fid_column):
            value = NUMBER.match(cells[header.index(fid_column)])
            if value:
                fids.setdefault(run_id, float(value.group()))
                rounds.setdefault(run_id, round_number)
    return [
        Run(
            run_id,
            rounds[run_id],
            names.get(run_id, ""),
            fids[run_id],
            rounds[run_id] in long_rounds,
        )
        for run_id in sorted(fids)
    ]


def load_phase(phase: int, last_round: int | None = None) -> list[Run]:
    log_path, fid_column = PHASE_LOGS[phase]
    runs = parse_phase(log_path, fid_column)
    if last_round is not None:
        runs = [run for run in runs if run.round_number <= last_round]
    return runs


def main() -> None:
    for phase in PHASE_LOGS:
        best = float("inf")
        for run in load_phase(phase):
            record = run.fid_patch < best
            best = min(best, run.fid_patch)
            marker = "*" if record else " "
            budget = "2h" if run.long_budget else "  "
            print(
                f"phase {phase} round {run.round_number:2d} {budget} {marker} {run.run_id:04d} {run.fid_patch:6.2f}  {run.name[:90]}"
            )


if __name__ == "__main__":
    main()
