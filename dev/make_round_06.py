"""
Generates the round 6 run files from run_0036 (levels (256, 256, 512), blocks (2, 1, 1), 1 middle
block) by exact text substitutions.
"""

from pathlib import Path

UPSCALERS_DIRECTORY = Path(__file__).resolve().parents[1] / "src" / "stdev" / "upscalers"
BASE_RUN = "0036"
BASE_DOCSTRING = """    condition) with levels (256, 256, 512), blocks (2, 1, 1) and 1 middle block."""
BASE_CHANNELS = "LEVEL_CHANNELS = (256, 256, 512)\n"
BASE_BLOCKS = "LEVEL_BLOCKS = (2, 1, 1)\n"
DECODER_BLOCKS_LINE = """                LEVEL_BLOCKS[level],
            )"""

CONVNEXT = [
    ("TIMESTEP_SCALE = 1000.0\n", "TIMESTEP_SCALE = 1000.0\nCONVNEXT_KERNEL_SIZE = 7\nCONVNEXT_EXPANSION = 4\n"),
    (
        '''class ResidualBlockSequence(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ConditionedResidualBlock(in_channels if index == 0 else out_channels, out_channels)
            for index in range(num_blocks)
        )''',
        '''class ConditionedConvNextBlock(nn.Module):
    """
    ConvNeXt block: depthwise convolution, normalization modulated by the condition, pointwise MLP.
    A 1x1 projection adapts the input channels. The last projection is zero initialized, so each
    block starts as the identity (after the input projection).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.input_projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.depthwise_conv = nn.Conv2d(
            out_channels,
            out_channels,
            CONVNEXT_KERNEL_SIZE,
            padding=CONVNEXT_KERNEL_SIZE // 2,
            groups=out_channels,
        )
        self.norm = make_group_norm(out_channels)
        self.condition_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(CONDITION_DIM, 2 * out_channels)
        )
        self.pointwise_first = nn.Conv2d(out_channels, CONVNEXT_EXPANSION * out_channels, 1)
        self.pointwise_second = nn.Conv2d(CONVNEXT_EXPANSION * out_channels, out_channels, 1)
        nn.init.zeros_(self.pointwise_second.weight)
        nn.init.zeros_(self.pointwise_second.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C_in, H, W]
            condition: [B, CONDITION_DIM]
        returns:
            features: [B, C_out, H, W]
        """
        features = self.input_projection(features)
        scale, shift = self.condition_projection(condition).chunk(2, dim=1)  # [B, C_out]
        hidden = self.norm(self.depthwise_conv(features))
        hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.pointwise_second(functional.silu(self.pointwise_first(hidden)))
        return features + hidden


class ResidualBlockSequence(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        block_type: type[nn.Module] = ConditionedResidualBlock,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            block_type(in_channels if index == 0 else out_channels, out_channels)
            for index in range(num_blocks)
        )''',
    ),
    (
        "            ResidualBlockSequence(LEVEL_CHANNELS[level], LEVEL_CHANNELS[level], LEVEL_BLOCKS[level])",
        """            ResidualBlockSequence(
                LEVEL_CHANNELS[level],
                LEVEL_CHANNELS[level],
                LEVEL_BLOCKS[level],
                ConditionedConvNextBlock if level == 0 else ConditionedResidualBlock,
            )""",
    ),
    (
        DECODER_BLOCKS_LINE,
        """                LEVEL_BLOCKS[level],
                ConditionedConvNextBlock if level == 0 else ConditionedResidualBlock,
            )""",
    ),
    (BASE_BLOCKS, "LEVEL_BLOCKS = (4, 1, 1)\n"),
]

RUNS = {
    "0040": ([(BASE_BLOCKS, "LEVEL_BLOCKS = (3, 1, 1)\n")], "with levels (256, 256, 512), blocks (3, 1, 1), 1 middle block."),
    "0041": (
        [(BASE_BLOCKS, "LEVEL_BLOCKS = (4, 1, 1)\n"), (BASE_CHANNELS, "LEVEL_CHANNELS = (256, 192, 256)\n")],
        "with levels (256, 192, 256), blocks (4, 1, 1), 1 middle block.",
    ),
    "0042": (CONVNEXT, "with levels (256, 256, 512), 4 ConvNeXt blocks at level 0, blocks (1, 1) below."),
    "0043": (
        [
            ("LEVEL_BLOCKS = (2, 1, 1)\n", "LEVEL_BLOCKS = (1, 1, 1)\nDECODER_LEVEL_BLOCKS = (3, 1, 1)\n"),
            (DECODER_BLOCKS_LINE, """                DECODER_LEVEL_BLOCKS[level],
            )"""),
        ],
        "with levels (256, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1).",
    ),
    "0044": (
        [(BASE_CHANNELS, "LEVEL_CHANNELS = (256, 256)\n"), (BASE_BLOCKS, "LEVEL_BLOCKS = (2, 1)\n")],
        "with only 2 levels (256, 256), blocks (2, 1), 1 middle block.",
    ),
    "0045": ([(BASE_CHANNELS, "LEVEL_CHANNELS = (256, 128, 256)\n")], "with levels (256, 128, 256), blocks (2, 1, 1)."),
    "0046": ([("BATCH_SIZE = 16\n", "BATCH_SIZE = 8\n")], "with levels (256, 256, 512), blocks (2, 1, 1), batch size 8."),
    "0047": ([("LEARNING_RATE = 4e-4\n", "LEARNING_RATE = 6e-4\n")], "with levels (256, 256, 512), blocks (2, 1, 1), LR 6e-4."),
}


def main() -> None:
    base_text = (UPSCALERS_DIRECTORY / f"run_{BASE_RUN}.py").read_text()
    for run_id, (replacements, docstring_suffix) in RUNS.items():
        if (UPSCALERS_DIRECTORY / f"run_{run_id}.py").exists():
            continue
        text = base_text.replace(f"run_{BASE_RUN}", f"run_{run_id}").replace(
            f"Run{BASE_RUN}Upscaler", f"Run{run_id}Upscaler"
        )
        docstring = f"    condition) {docstring_suffix}"
        if len(docstring) > 100:
            words, lines, current = docstring_suffix.split(), [], "    condition)"
            for word in words:
                if len(current) + 1 + len(word) > 100:
                    lines.append(current)
                    current = "   "
                current += " " + word
            docstring = "\n".join(lines + [current])
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
