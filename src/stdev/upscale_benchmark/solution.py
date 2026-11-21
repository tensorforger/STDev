from abc import ABC, abstractmethod

import torch

LATENT_CHANNELS = 16
INPUT_LATENT_SIZE = 64
OUTPUT_PIXEL_SIZE = 1024


class UpscaleSolution(ABC):
    """
    One 2x upscaling solution for the Wan 2.1 VAE, benchmarked end to end: the only input is the latent of a
    512x512 image (no prompt, no conditioning), the output is the decoded 1024x1024 image. Whether the solution
    upscales before or after decoding is an implementation detail. The benchmark calls `upscale` under
    `torch.no_grad()`, and constructors must not allocate GPU memory (everything is loaded in `load`).
    Solutions with `uses_hff = True` additionally receive the hff of the ground truth image (privileged variant).
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
            latents: [B, 16, 1, 64, 64], normalized Wan 2.1 VAE latents, dtype: torch.float32, device: cuda
            seed: noise seed of generative solutions, ignored by deterministic ones
            hff: [B], values: [0...1], dtype: torch.float32, device: cuda, passed only if `uses_hff`
        returns:
            pixels: [B, 3, 1024, 1024], channel_order: RGB, pixel_values: [0...1] (clamped or not), dtype: torch.float32
        """

    def get_settings(self) -> dict[str, object]:
        """
        returns:
            solution specific settings (checkpoint, steps, ...) written to the benchmark log
        """
        return {}
