"""
Generates the round 5 run files from run_0027 (stack + text, levels (256, 256, 512)) by exact
text substitutions: an architecture scan of level 0 capacity and cheap lower levels.
"""

from pathlib import Path

UPSCALERS_DIRECTORY = Path(__file__).resolve().parents[1] / "src" / "stdev" / "upscalers"
BASE_RUN = "0027"
BASE_DOCSTRING = """    Stacked recipe (training shift 1, LR 4e-4, linear decay over the last 30%, pooled text
    condition) with level channels (256, 384, 768), lower levels narrowed to (256, 256, 512)."""
DOCSTRING_PREFIX = """    Stacked recipe (training shift 1, LR 4e-4, linear decay over the last 30%, pooled text
    condition)"""
BASE_CHANNELS = "LEVEL_CHANNELS = (256, 256, 512)\n"
BASE_BLOCKS = "LEVEL_BLOCKS = (2, 2, 2)\n"


def channels(value: str) -> tuple[str, str]:
    return (BASE_CHANNELS, f"LEVEL_CHANNELS = {value}\n")


RUNS = {
    "0032": ([(BASE_BLOCKS, "LEVEL_BLOCKS = (3, 2, 2)\n")], "with levels (256, 256, 512), 3 blocks at level 0."),
    "0033": ([channels("(320, 256, 512)")], "with levels (320, 256, 512)."),
    "0034": ([channels("(256, 192, 384)")], "with levels (256, 192, 384)."),
    "0035": ([channels("(192, 192, 384)")], "with levels (192, 192, 384)."),
    "0036": (
        [(BASE_BLOCKS, "LEVEL_BLOCKS = (2, 1, 1)\n"), ("MIDDLE_BLOCKS = 2\n", "MIDDLE_BLOCKS = 1\n")],
        "with levels (256, 256, 512), blocks (2, 1, 1) and 1 middle block.",
    ),
    "0037": ([channels("(256, 256, 256)")], "with levels (256, 256, 256)."),
    "0038": ([channels("(384, 256, 512)")], "with levels (384, 256, 512)."),
    "0039": (
        [("\nSEED = 0\n", "\nSEED = 1\n")],
        "with levels (256, 256, 512), training seed 1 (replicate of run_0027).",
    ),
}


def main() -> None:
    base_text = (UPSCALERS_DIRECTORY / f"run_{BASE_RUN}.py").read_text()
    for run_id, (replacements, docstring_suffix) in RUNS.items():
        text = base_text.replace(f"run_{BASE_RUN}", f"run_{run_id}").replace(
            f"Run{BASE_RUN}Upscaler", f"Run{run_id}Upscaler"
        )
        docstring = f"{DOCSTRING_PREFIX}\n    {docstring_suffix}"
        for old, new in replacements + [(BASE_DOCSTRING, docstring)]:
            if text.count(old) != 1:
                raise RuntimeError(f"run_{run_id}: expected one occurrence of {old[:70]!r}")
            text = text.replace(old, new)
        long_lines = [line for line in text.split("\n") if len(line) > 100]
        if long_lines:
            raise RuntimeError(f"run_{run_id}: long lines {long_lines}")
        path = UPSCALERS_DIRECTORY / f"run_{run_id}.py"
        if path.exists():
            raise RuntimeError(f"{path} exists")
        path.write_text(text)
        print("wrote", path.name)


if __name__ == "__main__":
    main()
