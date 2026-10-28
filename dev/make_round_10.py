"""
Generates the round 10 run files (2 hour scaling round) from run_0066 by exact substitutions.
The template generalizes run_0066: per level channels and block counts, window attention after
every residual block on the levels in ATTENTION_LEVELS (the middle uses the last flag), an optional
pointwise MLP after each attention (SDXL style transformer block with window attention), optional
gradient checkpointing, 2 hour training with checkpoints at 30, 60 and 90 minutes.
With ATTENTION_LEVELS = (False, True, True) and one block per stage on levels 1 and 2, the model is
identical to run_0066.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_round_08 import wrap

UPSCALERS_DIRECTORY = Path(__file__).resolve().parents[1] / "src" / "stdev" / "upscalers"
BASE_RUN = "0066"
BASE_DOCSTRING = """    Stacked recipe (training shift 1, LR 4e-4, linear decay over the last 30%, pooled text
    condition) with levels (512, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1), 8x8
    window attention after the level 1, 2 and middle stages, 14 sampling steps."""

ARCHITECTURE_CONSTANTS = """LEVEL_CHANNELS = (512, 256, 512)
LEVEL_BLOCKS = (1, 1, 1)
DECODER_LEVEL_BLOCKS = (3, 1, 1)
MIDDLE_BLOCKS = 1
"""

TEMPLATE = [
    ("import math\n", "import math\n"),
    ("import torch.nn.functional as functional\n", "import torch.nn.functional as functional\nimport torch.utils.checkpoint\n"),
    (
        ARCHITECTURE_CONSTANTS,
        ARCHITECTURE_CONSTANTS
        + "ATTENTION_LEVELS = (False, True, True)\nTRANSFORMER_MLP = False\nMLP_EXPANSION = 4\n"
        "GRADIENT_CHECKPOINTING = False\n",
    ),
    (
        "TRAINING_TIME_SECONDS = 30 * 60\n",
        "TRAINING_TIME_SECONDS = 120 * 60\nCHECKPOINT_TIMES_SECONDS = (30 * 60, 60 * 60, 90 * 60)\n",
    ),
    (
        '''class ResidualBlockSequence(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ConditionedResidualBlock(in_channels if index == 0 else out_channels, out_channels)
            for index in range(num_blocks)
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            features = block(features, condition)
        return features


''',
        "",
    ),
    (
        '''class UpscalerUNet(nn.Module):''',
        '''class PointwiseFeedForward(nn.Module):
    """
    Per position MLP of a transformer block. The last projection is zero initialized.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = make_group_norm(channels)
        self.expand = nn.Conv2d(channels, MLP_EXPANSION * channels, 1)
        self.contract = nn.Conv2d(MLP_EXPANSION * channels, channels, 1)
        nn.init.zeros_(self.contract.weight)
        nn.init.zeros_(self.contract.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C, H, W]
        returns:
            features: [B, C, H, W]
        """
        return features + self.contract(functional.gelu(self.expand(self.norm(features))))


class WindowTransformerBlock(nn.Module):
    """
    Window attention, optionally followed by a pointwise MLP (SDXL style transformer block).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = WindowAttention(channels)
        self.feed_forward = PointwiseFeedForward(channels) if TRANSFORMER_MLP else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C, H, W]
        returns:
            features: [B, C, H, W]
        """
        return self.feed_forward(self.attention(features))


class ResidualBlockSequence(nn.Module):
    """
    Residual blocks, each optionally followed by a window transformer block.
    """

    def __init__(
        self, in_channels: int, out_channels: int, num_blocks: int, use_attention: bool
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ConditionedResidualBlock(in_channels if index == 0 else out_channels, out_channels)
            for index in range(num_blocks)
        )
        self.attentions = nn.ModuleList(
            WindowTransformerBlock(out_channels) if use_attention else nn.Identity()
            for _ in range(num_blocks)
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C_in, H, W]
            condition: [B, CONDITION_DIM]
        returns:
            features: [B, C_out, H, W]
        """
        for block, attention in zip(self.blocks, self.attentions):
            if GRADIENT_CHECKPOINTING and self.training:
                features = torch.utils.checkpoint.checkpoint(
                    block, features, condition, use_reentrant=False
                )
                features = torch.utils.checkpoint.checkpoint(
                    attention, features, use_reentrant=False
                )
            else:
                features = attention(block(features, condition))
        return features


class UpscalerUNet(nn.Module):''',
    ),
    (
        """    Convolutional UNet without attention predicting the flow velocity of high resolution latents.
    Bilinearly upsampled low resolution latents are concatenated to the noisy latents.""",
        """    Convolutional UNet with window attention predicting the flow velocity of high resolution
    latents. Bilinearly upsampled low resolution latents are concatenated to the noisy latents.""",
    ),
    (
        """            ResidualBlockSequence(LEVEL_CHANNELS[level], LEVEL_CHANNELS[level], LEVEL_BLOCKS[level])
            for level in range(len(LEVEL_CHANNELS))""",
        """            ResidualBlockSequence(
                LEVEL_CHANNELS[level],
                LEVEL_CHANNELS[level],
                LEVEL_BLOCKS[level],
                ATTENTION_LEVELS[level],
            )
            for level in range(len(LEVEL_CHANNELS))""",
    ),
    (
        """        self.middle_blocks = ResidualBlockSequence(
            LEVEL_CHANNELS[-1], LEVEL_CHANNELS[-1], MIDDLE_BLOCKS
        )""",
        """        self.middle_blocks = ResidualBlockSequence(
            LEVEL_CHANNELS[-1], LEVEL_CHANNELS[-1], MIDDLE_BLOCKS, ATTENTION_LEVELS[-1]
        )""",
    ),
    (
        """                DECODER_LEVEL_BLOCKS[level],
            )
            for level in range(len(LEVEL_CHANNELS))
        )
        self.encoder_contexts = nn.ModuleList(
            WindowAttention(LEVEL_CHANNELS[level]) if level >= 1 else nn.Identity()
            for level in range(len(LEVEL_CHANNELS))
        )
        self.middle_context = WindowAttention(LEVEL_CHANNELS[-1])
        self.decoder_contexts = nn.ModuleList(
            WindowAttention(LEVEL_CHANNELS[level]) if level >= 1 else nn.Identity()
            for level in range(len(LEVEL_CHANNELS))
        )
""",
        """                DECODER_LEVEL_BLOCKS[level],
                ATTENTION_LEVELS[level],
            )
            for level in range(len(LEVEL_CHANNELS))
        )
""",
    ),
    (
        "            features = self.encoder_contexts[level](encoder_stage(features, condition))\n",
        "            features = encoder_stage(features, condition)\n",
    ),
    (
        "        features = self.middle_context(self.middle_blocks(features, condition))\n",
        "        features = self.middle_blocks(features, condition)\n",
    ),
    (
        """            features = self.decoder_contexts[level](
                self.decoder_stages[level](torch.cat([features, skip], dim=1), condition)
            )
""",
        """            features = self.decoder_stages[level](torch.cat([features, skip], dim=1), condition)
""",
    ),
    (
        """        step = 0
        loss_sum = torch.zeros((), device=DEVICE)""",
        """        step = 0
        next_checkpoint_index = 0
        loss_sum = torch.zeros((), device=DEVICE)""",
    ),
    (
        """            if smoke_mode:
                is_finished = step >= SMOKE_STEPS""",
        """            if (
                not smoke_mode
                and next_checkpoint_index < len(CHECKPOINT_TIMES_SECONDS)
                and elapsed_seconds >= CHECKPOINT_TIMES_SECONDS[next_checkpoint_index]
            ):
                minutes = CHECKPOINT_TIMES_SECONDS[next_checkpoint_index] // 60
                OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
                checkpoint_path = OUTPUT_DIRECTORY / f"checkpoint_{minutes:03d}min.safetensors"
                save_file(model.state_dict(), str(checkpoint_path))
                print(f"{RUN_NAME}: checkpoint {checkpoint_path.name} at step {step}", flush=True)
                next_checkpoint_index += 1

            if smoke_mode:
                is_finished = step >= SMOKE_STEPS""",
    ),
]


def architecture(
    channels: str,
    encoder_blocks: str,
    decoder_blocks: str,
    middle_blocks: int = 1,
    attention_levels: str = "(False, True, True)",
    transformer_mlp: bool = False,
    gradient_checkpointing: bool = False,
    sampling_steps: int = 14,
) -> list[tuple[str, str]]:
    return [
        (
            ARCHITECTURE_CONSTANTS
            + "ATTENTION_LEVELS = (False, True, True)\nTRANSFORMER_MLP = False\nMLP_EXPANSION = 4\n"
            "GRADIENT_CHECKPOINTING = False\n",
            f"LEVEL_CHANNELS = {channels}\nLEVEL_BLOCKS = {encoder_blocks}\n"
            f"DECODER_LEVEL_BLOCKS = {decoder_blocks}\nMIDDLE_BLOCKS = {middle_blocks}\n"
            f"ATTENTION_LEVELS = {attention_levels}\nTRANSFORMER_MLP = {transformer_mlp}\n"
            f"MLP_EXPANSION = 4\nGRADIENT_CHECKPOINTING = {gradient_checkpointing}\n",
        ),
        ("NUM_SAMPLING_STEPS = 14\n", f"NUM_SAMPLING_STEPS = {sampling_steps}\n"),
    ]


DOCSTRING_PREFIX = (
    "2 hour scaling run (checkpoints at 30/60/90 min), stacked recipe (training shift 1, LR 4e-4, "
    "linear decay over the last 30%, pooled text condition, window attention after every residual "
    "block on ATTENTION_LEVELS):"
)
RUNS = {
    "0072": (
        architecture("(256, 256)", "(1, 1)", "(3, 1)", attention_levels="(False, True)", sampling_steps=10),
        "small, 2 levels (256, 256), encoder blocks (1, 1), decoder blocks (3, 1) (= run_0067).",
    ),
    "0073": (
        architecture("(256, 256, 512)", "(1, 1, 1)", "(5, 1, 1)"),
        "medium, levels (256, 256, 512), decoder blocks (5, 1, 1) (= run_0064).",
    ),
    "0074": (
        architecture("(512, 256, 512)", "(1, 1, 1)", "(3, 1, 1)"),
        "large, levels (512, 256, 512), decoder blocks (3, 1, 1) (= run_0066).",
    ),
    "0075": (
        architecture("(768, 384, 768)", "(1, 1, 1)", "(4, 1, 1)"),
        "wide full resolution, levels (768, 384, 768), decoder blocks (4, 1, 1).",
    ),
    "0076": (
        architecture("(1024, 512, 1024)", "(1, 1, 1)", "(4, 1, 1)", gradient_checkpointing=True),
        "very wide full resolution, levels (1024, 512, 1024), decoder blocks (4, 1, 1).",
    ),
    "0077": (
        architecture("(512, 256, 512)", "(1, 1, 1)", "(8, 1, 1)"),
        "deep full resolution decoder, levels (512, 256, 512), decoder blocks (8, 1, 1).",
    ),
    "0078": (
        architecture(
            "(512, 512, 1024)", "(1, 2, 2)", "(3, 2, 2)", middle_blocks=2,
            attention_levels="(True, True, True)", transformer_mlp=True,
            gradient_checkpointing=True,
        ),
        "SDXL style: levels (512, 512, 1024), encoder (1, 2, 2), decoder (3, 2, 2), 2 middle "
        "blocks, window transformer block (attention + MLP) after every residual block on all "
        "levels incl. full resolution.",
    ),
    "0079": (
        architecture(
            "(512, 768, 1280)", "(1, 2, 2)", "(3, 2, 2)", middle_blocks=2,
            transformer_mlp=True, gradient_checkpointing=True,
        ),
        "large low resolution stages: levels (512, 768, 1280), encoder (1, 2, 2), decoder "
        "(3, 2, 2), 2 middle blocks, window transformer blocks on levels 1, 2 and middle.",
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
        docstring = wrap(f"{DOCSTRING_PREFIX} {docstring_suffix}")
        for old, new in TEMPLATE + replacements + [(BASE_DOCSTRING, docstring)]:
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
