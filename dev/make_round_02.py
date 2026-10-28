"""
Generates the round 2 run files from run_0000 by exact text substitutions.
"""

from pathlib import Path

UPSCALERS_DIRECTORY = Path(__file__).resolve().parents[1] / "src" / "stdev" / "upscalers"
BASE_DOCSTRING = """    Baseline: multi-step flow matching convolutional UNet conditioned on concatenated low
    resolution latents, with logit-normal shifted timesteps and Euler sampling."""

SPLIT_SHIFT = [
    ("SIGMA_SHIFT = 3.0\n", "TRAINING_SIGMA_SHIFT = 3.0\nSAMPLING_SIGMA_SHIFT = 6.0\n"),
    ("NUM_SAMPLING_STEPS = 20\n", "NUM_SAMPLING_STEPS = 10\n"),
    (
        '''def shift_sigmas(sigmas: torch.Tensor) -> torch.Tensor:
    """
    Moves noise levels towards pure noise, as in SD3.

    args:
        sigmas: [...], values: [0...1], 1 is pure noise
    returns:
        shifted_sigmas: [...], values: [0...1], 1 is pure noise
    """
    return SIGMA_SHIFT * sigmas / (1.0 + (SIGMA_SHIFT - 1.0) * sigmas)''',
        '''def shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """
    Moves noise levels towards pure noise for shift > 1, as in SD3.

    args:
        sigmas: [...], values: [0...1], 1 is pure noise
    returns:
        shifted_sigmas: [...], values: [0...1], 1 is pure noise
    """
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)''',
    ),
    (
        "    return shift_sigmas(torch.sigmoid(torch.randn(batch_size, device=device)))",
        """    return shift_sigmas(
        torch.sigmoid(torch.randn(batch_size, device=device)), TRAINING_SIGMA_SHIFT
    )""",
    ),
    (
        """        sampling_sigmas = shift_sigmas(
            torch.linspace(1.0, 0.0, NUM_SAMPLING_STEPS + 1, device=tensor.device)
        )""",
        """        sampling_sigmas = shift_sigmas(
            torch.linspace(1.0, 0.0, NUM_SAMPLING_STEPS + 1, device=tensor.device),
            SAMPLING_SIGMA_SHIFT,
        )""",
    ),
    (
        "        self.model = torch.compile(self.model)",
        '        self.model = torch.compile(self.model, mode="reduce-overhead")',
    ),
]

UNIFORM_SIGMAS = [
    (
        '''    """
    returns:
        sigmas: [B], values: [0...1], logit-normal distributed and shifted
    """
    return shift_sigmas(
        torch.sigmoid(torch.randn(batch_size, device=device)), TRAINING_SIGMA_SHIFT
    )''',
        '''    """
    returns:
        sigmas: [B], values: [0...1], uniformly distributed
    """
    return torch.rand(batch_size, device=device)''',
    ),
    ("TRAINING_SIGMA_SHIFT = 3.0\n", ""),
    ("SAMPLING_SIGMA_SHIFT = 6.0\n", "SAMPLING_SIGMA_SHIFT = 3.0\n"),
]

LEARNING_RATE_DECAY = [
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

RUNS = {
    "0008": (
        UNIFORM_SIGMAS,
        "Baseline trained with uniformly distributed noise levels instead of shifted logit-normal.",
    ),
    "0009": (
        [("TRAINING_SIGMA_SHIFT = 3.0\n", "TRAINING_SIGMA_SHIFT = 1.0\n"),
         ("SAMPLING_SIGMA_SHIFT = 6.0\n", "SAMPLING_SIGMA_SHIFT = 3.0\n")],
        "Baseline trained with unshifted logit-normal noise levels (more low noise training).",
    ),
    "0010": (
        [("TRAINING_SIGMA_SHIFT = 3.0\n", "TRAINING_SIGMA_SHIFT = 1.0 / 3.0\n"),
         ("SAMPLING_SIGMA_SHIFT = 6.0\n", "SAMPLING_SIGMA_SHIFT = 3.0\n")],
        "Baseline trained with logit-normal noise levels shifted towards low noise (shift 1/3).",
    ),
    "0011": (
        [("\nSEED = 0\n", "\nSEED = 1\n")],
        "Baseline with training seed 1, a replicate measuring training seed noise.",
    ),
    "0012": (
        LEARNING_RATE_DECAY,
        "Baseline with the learning rate linearly decayed to zero over the last 30% of training time.",
    ),
    "0013": (
        [("LEARNING_RATE = 2e-4\n", "LEARNING_RATE = 4e-4\n")],
        "Baseline with doubled learning rate 4e-4.",
    ),
    "0014": (
        [("LEVEL_CHANNELS = (192, 384, 768)\n", "LEVEL_CHANNELS = (128, 256, 512, 768)\n")],
        "Baseline with a 4 level UNet (128, 256, 512, 768): cheaper high resolution, deeper context.",
    ),
    "0015": (
        [("LEVEL_CHANNELS = (192, 384, 768)\n", "LEVEL_CHANNELS = (64, 384, 768)\n")],
        "Baseline with a narrow full resolution level (64, 384, 768).",
    ),
}


def main() -> None:
    base_text = (UPSCALERS_DIRECTORY / "run_0000.py").read_text()
    for run_id, (replacements, docstring) in RUNS.items():
        text = base_text.replace("run_0000", f"run_{run_id}").replace(
            "Run0000Upscaler", f"Run{run_id}Upscaler"
        )
        for old, new in SPLIT_SHIFT + replacements + [(BASE_DOCSTRING, "    " + docstring)]:
            if text.count(old) != 1:
                raise RuntimeError(f"run_{run_id}: expected one occurrence of {old[:60]!r}")
            text = text.replace(old, new)
        path = UPSCALERS_DIRECTORY / f"run_{run_id}.py"
        if path.exists():
            raise RuntimeError(f"{path} exists")
        path.write_text(text)
        print("wrote", path.name)


if __name__ == "__main__":
    main()
