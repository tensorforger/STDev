import json
import sys
from pathlib import Path

CODE_MARKER = "# %%"
MARKDOWN_MARKER = "# %% [markdown]"


def make_cell(cell_type: str, lines: list[str]) -> dict:
    while lines and not lines[-1].strip():
        lines.pop()
    if cell_type == "markdown":
        lines = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in lines]
    source = [line + "\n" for line in lines[:-1]] + lines[-1:]
    cell = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_notebook(source_path: Path, notebook_path: Path) -> None:
    """
    Converts a percent-format script with `# %%` cell markers into an ipynb file.
    """
    cells = []
    cell_type, lines = None, []
    for line in source_path.read_text().splitlines():
        if line.startswith(CODE_MARKER):
            if cell_type is not None:
                cells.append(make_cell(cell_type, lines))
            cell_type = "markdown" if line.startswith(MARKDOWN_MARKER) else "code"
            lines = []
        else:
            lines.append(line)
    cells.append(make_cell(cell_type, lines))
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "STDev (3.12.14)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    build_notebook(Path(sys.argv[1]), Path(sys.argv[2]))
