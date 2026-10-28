"""
Generates the round 3 run files from run_0009 (training shift 1) by exact text substitutions.
Every run gets the stacked recipe: LR 4e-4 with linear decay to zero over the last 30% of time.
"""

from pathlib import Path

UPSCALERS_DIRECTORY = Path(__file__).resolve().parents[1] / "src" / "stdev" / "upscalers"
BASE_RUN = "0009"
BASE_DOCSTRING = (
    "    Baseline trained with unshifted logit-normal noise levels (more low noise training)."
)

STACK = [
    ("LEARNING_RATE = 2e-4\n", "LEARNING_RATE = 4e-4\n"),
    ("WARMUP_STEPS = 100\n", "WARMUP_STEPS = 100\nDECAY_FRACTION = 0.3\n"),
    (
        """            optimizer.step()
            warmup_scheduler.step()
""",
        """            optimizer.step()
            warmup_scheduler.step()
            if training_start_time is not None:
                remaining_fraction = 1.0 - (time.perf_counter() - training_start_time) / (
                    TRAINING_TIME_SECONDS
                )
                decay_factor = min(1.0, max(0.0, remaining_fraction) / DECAY_FRACTION)
                warmup_factor = min(1.0, (step + 1) / WARMUP_STEPS)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = LEARNING_RATE * warmup_factor * decay_factor
""",
    ),
]

TEXT_CONDITIONING = [
    (
        '''    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
            latents_full: [C, H, W], dtype: torch.float32
            latents_div_2: [C, H / 2, W / 2], dtype: torch.float32
        """
        encoded_block = vaezip.load(self.block_paths[index])
        latents_full, latents_div_2 = (
            encoded_block[name].decode(dtype=torch.float32)[0]
            for name in ("latents_orig", "latents_div_2")
        )
        return latents_full, latents_div_2''',
        '''    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
            latents_full: [C, H, W], dtype: torch.float32
            latents_div_2: [C, H / 2, W / 2], dtype: torch.float32
            prompt_embeds: [77, 768], dtype: torch.float32
        """
        encoded_block = vaezip.load(self.block_paths[index])
        latents_full, latents_div_2, prompt_embeds = (
            encoded_block[name].decode(dtype=torch.float32)[0]
            for name in ("latents_orig", "latents_div_2", "prompt_embeds")
        )
        return latents_full, latents_div_2, prompt_embeds''',
    ),
    (
        "    Full resolution latents paired with the 2x downscaled latents.\n",
        "    Full resolution latents paired with the 2x downscaled latents and the prompt embeddings.\n",
    ),
    (
        "def iterate_batches(data_loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:",
        "def iterate_batches(data_loader: DataLoader) -> Iterator[tuple[torch.Tensor, ...]]:",
    ),
    (
        '''class ConditioningEncoder(nn.Module):
    """
    Sinusoidal embedding of the noise level followed by an MLP.
    """

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(TIME_FREQUENCY_DIM, CONDITION_DIM),
            nn.SiLU(),
            nn.Linear(CONDITION_DIM, CONDITION_DIM),
        )

    def forward(self, sigmas: torch.Tensor) -> torch.Tensor:
        """
        args:
            sigmas: [B], values: [0...1], 1 is pure noise
        returns:
            condition: [B, CONDITION_DIM]
        """''',
        '''class ConditioningEncoder(nn.Module):
    """
    Sinusoidal embedding of the noise level followed by an MLP, plus a projection of the mean
    pooled prompt embeddings.
    """

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(TIME_FREQUENCY_DIM, CONDITION_DIM),
            nn.SiLU(),
            nn.Linear(CONDITION_DIM, CONDITION_DIM),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(TEXT_EMBEDDING_DIM, CONDITION_DIM),
            nn.SiLU(),
            nn.Linear(CONDITION_DIM, CONDITION_DIM),
        )

    def forward(self, sigmas: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        args:
            sigmas: [B], values: [0...1], 1 is pure noise
            prompt_embeds: [B, 77, TEXT_EMBEDDING_DIM]
        returns:
            condition: [B, CONDITION_DIM]
        """
        weight_dtype = self.projection[0].weight.dtype
        text_condition = self.text_projection(prompt_embeds.mean(dim=1).to(weight_dtype))''',
    ),
    (
        "        return self.projection(embedding.to(self.projection[0].weight.dtype))",
        "        return self.projection(embedding.to(weight_dtype)) + text_condition",
    ),
    ("TIMESTEP_SCALE = 1000.0\n", "TIMESTEP_SCALE = 1000.0\nTEXT_EMBEDDING_DIM = 768\n"),
    (
        """        latents_low_resolution: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"
        args:
            noisy_latents: [B, C, H, W]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_low_resolution: [B, C, H / 2, W / 2]""",
        """        latents_low_resolution: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"
        args:
            noisy_latents: [B, C, H, W]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_low_resolution: [B, C, H / 2, W / 2]
            prompt_embeds: [B, 77, TEXT_EMBEDDING_DIM]""",
    ),
    ("        condition = self.conditioning(sigmas)\n", "        condition = self.conditioning(sigmas, prompt_embeds)\n"),
    (
        """            latents_full, latents_div_2 = (
                tensor.to(DEVICE, non_blocking=True) for tensor in batch
            )  # [B, C, H, W], [B, C, H / 2, W / 2]""",
        """            latents_full, latents_div_2, prompt_embeds = (
                tensor.to(DEVICE, non_blocking=True) for tensor in batch
            )  # [B, C, H, W], [B, C, H / 2, W / 2], [B, 77, 768]""",
    ),
    (
        "                velocity = compiled_model(noisy_latents, sigmas, latents_div_2)",
        "                velocity = compiled_model(noisy_latents, sigmas, latents_div_2, prompt_embeds)",
    ),
    (
        "            prompt_embeds: [B, 77, 768], dtype = float32, SDXL text encoder embeddings, unused\n",
        "            prompt_embeds: [B, 77, 768], dtype = float32, SDXL text encoder embeddings\n",
    ),
    (
        """        latents_low_resolution = tensor.to(INFERENCE_DTYPE)
""",
        """        latents_low_resolution = tensor.to(INFERENCE_DTYPE)
        prompt_embeds = prompt_embeds.to(INFERENCE_DTYPE)
""",
    ),
    (
        """                latents_low_resolution,
            )
            latents = latents + (next_sigma - current_sigma) * velocity.float()""",
        """                latents_low_resolution,
                prompt_embeds,
            )
            latents = latents + (next_sigma - current_sigma) * velocity.float()""",
    ),
    (
        "            prompt_embeds: [B, 77, 768], dtype = float32, unused\n",
        "            prompt_embeds: [B, 77, 768], dtype = float32\n",
    ),
    (
        "        return self.model(noisy_latents, sigmas, tensor.to(INFERENCE_DTYPE))",
        """        return self.model(
            noisy_latents,
            sigmas,
            tensor.to(INFERENCE_DTYPE),
            prompt_embeds.to(INFERENCE_DTYPE),
        )""",
    ),
]

STACK_DOCSTRING = "Stacked recipe (training shift 1, LR 4e-4, linear decay over the last 30%)"
RUNS = {
    "0016": ([], f"{STACK_DOCSTRING}."),
    "0017": (
        [("LEARNING_RATE = 4e-4\n", "LEARNING_RATE = 8e-4\n")],
        f"{STACK_DOCSTRING} with LR 8e-4.",
    ),
    "0018": (
        [("LEVEL_CHANNELS = (192, 384, 768)\n", "LEVEL_CHANNELS = (256, 512, 1024)\n")],
        f"{STACK_DOCSTRING} with wide levels (256, 512, 1024).",
    ),
    "0019": (
        [("LEVEL_CHANNELS = (192, 384, 768)\n", "LEVEL_CHANNELS = (256, 384, 768)\n")],
        f"{STACK_DOCSTRING} with a wider full resolution level (256, 384, 768).",
    ),
    "0020": (
        [("BLOCKS_PER_LEVEL = 2\n", "BLOCKS_PER_LEVEL = 3\n")],
        f"{STACK_DOCSTRING} with 3 blocks per level.",
    ),
    "0021": (TEXT_CONDITIONING, f"{STACK_DOCSTRING} with mean pooled prompt embeddings added to the condition."),
    "0022": (
        [("BATCH_SIZE = 16\n", "BATCH_SIZE = 32\n")],
        f"{STACK_DOCSTRING} with batch size 32.",
    ),
    "0023": (
        [("LEVEL_CHANNELS = (192, 384, 768)\n", "LEVEL_CHANNELS = (128, 256, 512)\n")],
        f"{STACK_DOCSTRING} with narrow levels (128, 256, 512).",
    ),
}


def main() -> None:
    base_text = (UPSCALERS_DIRECTORY / f"run_{BASE_RUN}.py").read_text()
    for run_id, (replacements, docstring) in RUNS.items():
        text = base_text.replace(f"run_{BASE_RUN}", f"run_{run_id}").replace(
            f"Run{BASE_RUN}Upscaler", f"Run{run_id}Upscaler"
        )
        for old, new in STACK + replacements + [(BASE_DOCSTRING, "    " + docstring)]:
            if text.count(old) != 1:
                raise RuntimeError(f"run_{run_id}: expected one occurrence of {old[:70]!r}")
            text = text.replace(old, new)
        path = UPSCALERS_DIRECTORY / f"run_{run_id}.py"
        if path.exists():
            raise RuntimeError(f"{path} exists")
        path.write_text(text)
        print("wrote", path.name)


if __name__ == "__main__":
    main()
