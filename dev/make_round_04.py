"""
Generates the round 4 run files from run_0021 (stacked recipe + pooled text condition) by exact
text substitutions. Every run gets level 0 width 256 and per level block counts.
"""

from pathlib import Path

UPSCALERS_DIRECTORY = Path(__file__).resolve().parents[1] / "src" / "stdev" / "upscalers"
BASE_RUN = "0021"
BASE_DOCSTRING = """    Stacked recipe (training shift 1, LR 4e-4, linear decay over the last 30%) with mean pooled
    prompt embeddings added to the condition."""

BASE = [
    ("LEVEL_CHANNELS = (192, 384, 768)\n", "LEVEL_CHANNELS = (256, 384, 768)\n"),
    ("BLOCKS_PER_LEVEL = 2\n", "LEVEL_BLOCKS = (2, 2, 2)\n"),
    (
        "            ResidualBlockSequence(LEVEL_CHANNELS[level], LEVEL_CHANNELS[level], BLOCKS_PER_LEVEL)",
        "            ResidualBlockSequence(LEVEL_CHANNELS[level], LEVEL_CHANNELS[level], LEVEL_BLOCKS[level])",
    ),
    (
        """                LEVEL_CHANNELS[level],
                BLOCKS_PER_LEVEL,
            )""",
        """                LEVEL_CHANNELS[level],
                LEVEL_BLOCKS[level],
            )""",
    ),
]

CROSS_ATTENTION = [
    ("TEXT_EMBEDDING_DIM = 768\n", "TEXT_EMBEDDING_DIM = 768\nATTENTION_HEAD_DIM = 64\n"),
    (
        '''class UpscalerUNet(nn.Module):''',
        '''class TextCrossAttention(nn.Module):
    """
    Image features attend to the prompt tokens. The output projection is zero initialized, so the
    module starts as the identity.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = make_group_norm(channels)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key_value = nn.Linear(TEXT_EMBEDDING_DIM, 2 * channels)
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C, H, W]
            prompt_embeds: [B, T, TEXT_EMBEDDING_DIM]
        returns:
            features: [B, C, H, W]
        """
        batch_size, channels, height, width = features.shape
        heads = channels // ATTENTION_HEAD_DIM
        query = self.query(self.norm(features)).reshape(
            batch_size, heads, ATTENTION_HEAD_DIM, height * width
        ).transpose(2, 3)  # [B, heads, HW, D]
        key, value = (
            tensor.reshape(batch_size, -1, heads, ATTENTION_HEAD_DIM).transpose(1, 2)
            for tensor in self.key_value(prompt_embeds).chunk(2, dim=-1)
        )  # [B, heads, T, D]
        attended = functional.scaled_dot_product_attention(query, key, value)  # [B, heads, HW, D]
        attended = attended.transpose(2, 3).reshape(batch_size, channels, height, width)
        return features + self.output(attended)


class UpscalerUNet(nn.Module):''',
    ),
    (
        """        self.output_norm = make_group_norm(LEVEL_CHANNELS[0])""",
        """        self.text_attentions = nn.ModuleList(
            TextCrossAttention(LEVEL_CHANNELS[-1]) for _ in range(3)
        )
        self.output_norm = make_group_norm(LEVEL_CHANNELS[0])""",
    ),
    (
        """            features = encoder_stage(features, condition)
            skips.append(features)""",
        """            features = encoder_stage(features, condition)
            if level == len(LEVEL_CHANNELS) - 1:
                features = self.text_attentions[0](features, prompt_embeds)
            skips.append(features)""",
    ),
    (
        """        features = self.middle_blocks(features, condition)
""",
        """        features = self.text_attentions[1](self.middle_blocks(features, condition), prompt_embeds)
""",
    ),
    (
        """            features = self.decoder_stages[level](torch.cat([features, skip], dim=1), condition)
""",
        """            features = self.decoder_stages[level](torch.cat([features, skip], dim=1), condition)
            if level == len(LEVEL_CHANNELS) - 1:
                features = self.text_attentions[2](features, prompt_embeds)
""",
    ),
]

STACK_DOCSTRING = """    Stacked recipe (training shift 1, LR 4e-4, linear decay over the last 30%, pooled text
    condition) with level channels (256, 384, 768)"""
RUNS = {
    "0024": ([], f"{STACK_DOCSTRING}."),
    "0025": (
        [("LEVEL_CHANNELS = (256, 384, 768)\n", "LEVEL_CHANNELS = (384, 384, 768)\n")],
        f"{STACK_DOCSTRING}, level 0 widened to 384.",
    ),
    "0026": (
        [("LEVEL_BLOCKS = (2, 2, 2)\n", "LEVEL_BLOCKS = (3, 2, 2)\n")],
        f"{STACK_DOCSTRING}, 3 blocks at level 0.",
    ),
    "0027": (
        [("LEVEL_CHANNELS = (256, 384, 768)\n", "LEVEL_CHANNELS = (256, 256, 512)\n")],
        f"{STACK_DOCSTRING}, lower levels narrowed to (256, 256, 512).",
    ),
    "0028": (
        [("LEVEL_CHANNELS = (256, 384, 768)\n", "LEVEL_CHANNELS = (160, 256, 512)\n")],
        f"{STACK_DOCSTRING}, narrowed to (160, 256, 512) for low latency.",
    ),
    "0029": (
        [("DECAY_FRACTION = 0.3\n", "DECAY_FRACTION = 0.6\n")],
        f"{STACK_DOCSTRING}, LR decay over the last 60%.",
    ),
    "0030": (CROSS_ATTENTION, f"{STACK_DOCSTRING}, cross-attention to the prompt tokens at the lowest level."),
    "0031": (
        [
            ("LEVEL_BLOCKS = (2, 2, 2)\n", "LEVEL_BLOCKS = (2, 2, 3)\n"),
            ("MIDDLE_BLOCKS = 2\n", "MIDDLE_BLOCKS = 4\n"),
        ],
        f"{STACK_DOCSTRING}, deeper lowest level (3 blocks) and 4 middle blocks.",
    ),
}


def main() -> None:
    base_text = (UPSCALERS_DIRECTORY / f"run_{BASE_RUN}.py").read_text()
    for run_id, (replacements, docstring) in RUNS.items():
        text = base_text.replace(f"run_{BASE_RUN}", f"run_{run_id}").replace(
            f"Run{BASE_RUN}Upscaler", f"Run{run_id}Upscaler"
        )
        for old, new in BASE + replacements + [(BASE_DOCSTRING, docstring)]:
            if text.count(old) != 1:
                raise RuntimeError(f"run_{run_id}: expected one occurrence of {old[:70]!r}")
            text = text.replace(old, new)
        long_lines = [line for line in text.split("\n") if len(line) > 100]
        if long_lines:
            print(f"run_{run_id}: long lines {long_lines}")
        path = UPSCALERS_DIRECTORY / f"run_{run_id}.py"
        if path.exists():
            raise RuntimeError(f"{path} exists")
        path.write_text(text)
        print("wrote", path.name)


if __name__ == "__main__":
    main()
