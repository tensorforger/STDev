"""
Generates the round 9 run files from run_0063 (levels (384, 256, 512), encoder blocks (1, 1, 1),
decoder blocks (3, 1, 1), 8x8 window attention at levels 1, 2 and middle) by exact substitutions:
both ends of the Pareto front with window attention, EMA, training shift, full resolution attention.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_round_08 import wrap

UPSCALERS_DIRECTORY = Path(__file__).resolve().parents[1] / "src" / "stdev" / "upscalers"
BASE_RUN = "0063"
BASE_DOCSTRING = """    condition) with levels (384, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1), 8x8
    window attention after the level 1, 2 and middle stages (scale candidate)."""
CHANNELS = "LEVEL_CHANNELS = (384, 256, 512)\n"
ENCODER_BLOCKS = "LEVEL_BLOCKS = (1, 1, 1)\n"
DECODER_BLOCKS = "DECODER_LEVEL_BLOCKS = (3, 1, 1)\n"
STEPS = "NUM_SAMPLING_STEPS = 10\n"
QUALITY_END_STEPS = (STEPS, "NUM_SAMPLING_STEPS = 14\n")

EMA = [
    ("GRADIENT_CLIP_NORM = 1.0\n", "GRADIENT_CLIP_NORM = 1.0\nEMA_MAX_DECAY = 0.9999\n"),
    ('MODEL_FILE_NAME = "model.safetensors"\n', 'MODEL_FILE_NAME = "model.safetensors"\nRAW_MODEL_FILE_NAME = "model_raw.safetensors"\n'),
    (
        "        parameter_count = sum(parameter.numel() for parameter in model.parameters())",
        """        parameters = list(model.parameters())
        ema_parameters = [parameter.detach().clone() for parameter in parameters]
        parameter_count = sum(parameter.numel() for parameter in parameters)""",
    ),
    (
        """            optimizer.step()
            warmup_scheduler.step()
""",
        """            optimizer.step()
            warmup_scheduler.step()
            ema_decay = min(EMA_MAX_DECAY, (1.0 + step) / (10.0 + step))
            torch._foreach_lerp_(
                ema_parameters, [parameter.detach() for parameter in parameters], 1.0 - ema_decay
            )
""",
    ),
    (
        """        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        save_file(model.state_dict(), str(OUTPUT_DIRECTORY / MODEL_FILE_NAME))""",
        """        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        save_file(model.state_dict(), str(OUTPUT_DIRECTORY / RAW_MODEL_FILE_NAME))
        with torch.no_grad():
            for parameter, ema_parameter in zip(parameters, ema_parameters):
                parameter.copy_(ema_parameter)
        save_file(model.state_dict(), str(OUTPUT_DIRECTORY / MODEL_FILE_NAME))""",
    ),
]

FULL_RESOLUTION_ATTENTION = [
    (
        """        self.decoder_contexts = nn.ModuleList(
            WindowAttention(LEVEL_CHANNELS[level]) if level >= 1 else nn.Identity()
            for level in range(len(LEVEL_CHANNELS))
        )""",
        """        self.decoder_contexts = nn.ModuleList(
            WindowAttention(LEVEL_CHANNELS[level]) for level in range(len(LEVEL_CHANNELS))
        )""",
    ),
]

RUNS = {
    "0064": (
        [(CHANNELS, "LEVEL_CHANNELS = (256, 256, 512)\n"), (DECODER_BLOCKS, "DECODER_LEVEL_BLOCKS = (5, 1, 1)\n")],
        "with levels (256, 256, 512), encoder blocks (1, 1, 1), decoder blocks (5, 1, 1), 8x8 window attention after the level 1, 2 and middle stages.",
    ),
    "0065": (
        [(DECODER_BLOCKS, "DECODER_LEVEL_BLOCKS = (4, 1, 1)\n"), QUALITY_END_STEPS],
        "with levels (384, 256, 512), encoder blocks (1, 1, 1), decoder blocks (4, 1, 1), 8x8 window attention after the level 1, 2 and middle stages, 14 sampling steps.",
    ),
    "0066": (
        [(CHANNELS, "LEVEL_CHANNELS = (512, 256, 512)\n"), QUALITY_END_STEPS],
        "with levels (512, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1), 8x8 window attention after the level 1, 2 and middle stages, 14 sampling steps.",
    ),
    "0067": (
        [
            (CHANNELS, "LEVEL_CHANNELS = (256, 256)\n"),
            (ENCODER_BLOCKS, "LEVEL_BLOCKS = (1, 1)\n"),
            (DECODER_BLOCKS, "DECODER_LEVEL_BLOCKS = (3, 1)\n"),
        ],
        "with 2 levels (256, 256), encoder blocks (1, 1), decoder blocks (3, 1), 8x8 window attention after the level 1 and middle stages.",
    ),
    "0068": (
        EMA + [QUALITY_END_STEPS],
        "with levels (384, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1), 8x8 window attention after the level 1, 2 and middle stages, EMA of the weights, 14 sampling steps.",
    ),
    "0069": (
        [
            (CHANNELS, "LEVEL_CHANNELS = (256, 256)\n"),
            (ENCODER_BLOCKS, "LEVEL_BLOCKS = (1, 1)\n"),
            (DECODER_BLOCKS, "DECODER_LEVEL_BLOCKS = (2, 1)\n"),
        ],
        "with 2 levels (256, 256), encoder blocks (1, 1), decoder blocks (2, 1), 8x8 window attention after the level 1 and middle stages.",
    ),
    "0070": (
        [(CHANNELS, "LEVEL_CHANNELS = (256, 256, 512)\n"), ("TRAINING_SIGMA_SHIFT = 1.0\n", "TRAINING_SIGMA_SHIFT = 1.5\n")],
        "with levels (256, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1), 8x8 window attention after the level 1, 2 and middle stages, training shift 1.5.",
    ),
    "0071": (
        FULL_RESOLUTION_ATTENTION + [QUALITY_END_STEPS],
        "with levels (384, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1), 8x8 window attention after every decoder stage incl. full resolution and after the level 1, 2 encoder and middle stages, 14 sampling steps.",
    ),
}


def main() -> None:
    base_text = (UPSCALERS_DIRECTORY / f"run_{BASE_RUN}.py").read_text()
    for run_id, (replacements, docstring_suffix) in RUNS.items():
        path = UPSCALERS_DIRECTORY / f"run_{run_id}.py"
        if path.exists():
            continue
        text = base_text.replace(f"run_{BASE_RUN}", f"run_{run_id}").replace(
            f"Run{BASE_RUN}Upscaler", f"Run{run_id}Upscaler"
        )
        docstring = wrap("condition) " + docstring_suffix)
        for old, new in replacements + [(BASE_DOCSTRING, docstring)]:
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
