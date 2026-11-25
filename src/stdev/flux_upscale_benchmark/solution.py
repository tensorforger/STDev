from abc import ABC, abstractmethod

import torch

LATENT_CHANNELS = 32
INPUT_LATENT_SIZE = 64
OUTPUT_PIXEL_SIZE = 1024
INPUT_SHAPE = (LATENT_CHANNELS, INPUT_LATENT_SIZE, INPUT_LATENT_SIZE)


class FluxUpscaleSolution(ABC):
    """
    One 2x upscaling solution for the FLUX.2 VAE, benchmarked end to end: the only input is the normalized FLUX.2
    latent of a 512x512 image (no prompt, no hff, nothing from the ground truth), the output is the decoded 1024x1024
    image. Whether the solution upscales before or after decoding is an implementation detail. The benchmark calls
    `upscale` under `torch.no_grad()`, and constructors must not allocate GPU memory (everything is loaded in `load`).
    """

    uses_hff: bool = False

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def load(self) -> None:
        """
        Loads all weights (upscaler and decoder) to the GPU.
        """

    @abstractmethod
    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        args:
            latents: [B, 32, 64, 64], normalized FLUX.2 VAE latents (batch norm applied, unpatchified),
                dtype: torch.float32, device: cuda
            seed: noise seed of generative solutions, ignored by deterministic ones
            hff: unused (interface of the shared runner)
        returns:
            pixels: [B, 3, 1024, 1024], channel_order: RGB, pixel_values: [0...1] (clamped or not), dtype: torch.float32
        """

    def get_settings(self) -> dict[str, object]:
        """
        returns:
            solution specific settings (checkpoint, steps, ...) written to the benchmark log
        """
        return {}
