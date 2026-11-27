from abc import ABC, abstractmethod

import torch


class Upscaler(ABC):
    """
    Self-contained high-level wrapper of a Ming-Image VAE latent upscaler (autoresearch_5).
    Each upscaler definition file should contain the implementation of this interface.

    Latents are normalized Ming-Image VAE latents [B, 16, H, W] (8x spatial compression). The task is exactly
    2x spatial upscaling: [B, 16, H, W] -> [B, 16, 2H, 2W]. There is no conditioning besides the latents.
    """

    @abstractmethod
    def train(self, smoke_mode: bool = False) -> None:
        """
        Starts the training process and saves the weights.
        In smoke_mode should return after several training steps.
        """

    @abstractmethod
    def prepare_to_benchmark(self) -> None:
        """
        Loads the saved weights in eager mode. May be called several times by the benchmark.
        """

    @abstractmethod
    def compile(self) -> None:
        """
        Compiles the model with torch.compile. Called after prepare_to_benchmark().
        """

    @abstractmethod
    def upscale(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, 16, H, W], dtype = float32 (or float16, bfloat16), device = "cuda"
        returns:
            latents: [B, 16, 2H, 2W], same dtype as the input
        """

    @abstractmethod
    def single_forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Runs the underlying model exactly once on dummy internal inputs (e.g. random noise and a
        fixed noise level). Used only by the latency and memory benchmark, not for producing samples.
        args:
            latents: [B, 16, H, W], dtype = float32 (or float16, bfloat16), device = "cuda"
        returns:
            tensor: model output for the [2H, 2W] grid, any dtype
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the upscaler, equal to its file name without extension.
        """
