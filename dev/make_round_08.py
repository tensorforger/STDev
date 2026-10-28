"""
Generates the round 8 run files: window attention (the round 7 winner, step matched) brought into
the regular 30 min time budget on the Pareto front architectures, plus placement and window size.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_round_07 import LINEAR_ATTENTION, WINDOW_ATTENTION

UPSCALERS_DIRECTORY = Path(__file__).resolve().parents[1] / "src" / "stdev" / "upscalers"
DOCSTRING_PREFIX = """    Stacked recipe (training shift 1, LR 4e-4, linear decay over the last 30%, pooled text
    condition)"""

BASE_DOCSTRINGS = {
    "0041": """    condition) with levels (256, 192, 256), blocks (4, 1, 1), 1 middle block.""",
    "0043": """    condition) with levels (256, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1).""",
    "0044": """    condition) with only 2 levels (256, 256), blocks (2, 1), 1 middle block.""",
}


def with_levels(replacements: list[tuple[str, str]], encoder_condition: str, decoder_condition: str):
    """
    Changes which levels get a context module (default in round 7: levels >= 1).
    """
    output = []
    encoder_done = False
    for old, new in replacements:
        if "self.encoder_contexts = nn.ModuleList(" in new:
            first = new.index("if level >= 1")
            second = new.index("if level >= 1", first + 1)
            new = (
                new[:first]
                + f"if {encoder_condition}"
                + new[first + len("if level >= 1") : second]
                + f"if {decoder_condition}"
                + new[second + len("if level >= 1") :]
            )
            encoder_done = True
        output.append((old, new))
    if not encoder_done:
        raise RuntimeError("context module list not found")
    return output


RUNS = {
    "0056": ("0043", WINDOW_ATTENTION, "with levels (256, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1), 8x8 window attention after the level 1, 2 and middle stages."),
    "0057": ("0043", with_levels(WINDOW_ATTENTION, "level >= 2", "level >= 2"), "with levels (256, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1), 8x8 window attention after the level 2 and middle stages only."),
    "0058": ("0043", with_levels(WINDOW_ATTENTION, "level >= 1", "True"), "with levels (256, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1), 8x8 window attention after the level 1, 2 and middle stages and the full resolution decoder stage."),
    "0059": ("0041", WINDOW_ATTENTION, "with levels (256, 192, 256), blocks (4, 1, 1), 1 middle block, 8x8 window attention after the level 1, 2 and middle stages."),
    "0060": ("0044", WINDOW_ATTENTION, "with only 2 levels (256, 256), blocks (2, 1), 1 middle block, 8x8 window attention after the level 1 and middle stages."),
    "0061": ("0043", LINEAR_ATTENTION, "with levels (256, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1), global linear attention after the level 1, 2 and middle stages."),
    "0062": ("0043", WINDOW_ATTENTION + [("WINDOW_SIZE = 8\n", "WINDOW_SIZE = 16\n")], "with levels (256, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1), 16x16 window attention after the level 1, 2 and middle stages."),
    "0063": ("0043", WINDOW_ATTENTION + [("LEVEL_CHANNELS = (256, 256, 512)\n", "LEVEL_CHANNELS = (384, 256, 512)\n")], "with levels (384, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1), 8x8 window attention after the level 1, 2 and middle stages (scale candidate)."),
}


def wrap(text: str) -> str:
    lines, current = [], "   "
    for word in text.split():
        if len(current) + 1 + len(word) > 100:
            lines.append(current)
            current = "   "
        current += " " + word
    return "\n".join(lines + [current])


def main() -> None:
    for run_id, (base_run, replacements, docstring_suffix) in RUNS.items():
        path = UPSCALERS_DIRECTORY / f"run_{run_id}.py"
        if path.exists():
            continue
        text = (UPSCALERS_DIRECTORY / f"run_{base_run}.py").read_text()
        text = text.replace(f"run_{base_run}", f"run_{run_id}").replace(
            f"Run{base_run}Upscaler", f"Run{run_id}Upscaler"
        )
        docstring = wrap("condition) " + docstring_suffix)
        for old, new in replacements + [(BASE_DOCSTRINGS[base_run], docstring)]:
            if text.count(old) != 1:
                raise RuntimeError(f"run_{run_id}: expected one occurrence of {old[:70]!r}")
            text = text.replace(old, new)
        long_lines = [line for line in text.split("\n") if len(line) > 100]
        if long_lines:
            raise RuntimeError(f"run_{run_id}: long lines {long_lines}")
        path.write_text(text)
        print("wrote", path.name)


if __name__ == "__main__":
    main()
