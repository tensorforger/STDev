from abc import ABC, abstractmethod

import torch


class Translator(ABC):
    """
    Self-contained high-level wrapper of an H3-Image -> FLUX.2 latent translator (autoresearch_6).
    Each run file should contain the implementation of this interface.

    Input: normalized MiniMax-H3-Image latents [B, 24, H, W] (16x spatial compression).
    Output: normalized FLUX.2 latents [B, 32, 2H, 2W] (8x) of the same image: the pixel resolution does not change.
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
    def translate(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, 24, H, W], normalized H3 latents, dtype = float32 (or float16, bfloat16), device = "cuda"
        returns:
            latents: [B, 32, 2H, 2W], normalized FLUX.2 latents, dtype: torch.float32
        """

    @abstractmethod
    def single_forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Runs the underlying model exactly once (everything a one-step distilled student would run: model forward and
        any fixed pre / post processing). Used only by the latency and memory benchmark.
        args:
            latents: [B, 24, H, W], dtype = float32 (or float16, bfloat16), device = "cuda"
        returns:
            tensor: model output for the [2H, 2W] grid, any dtype
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the run, equal to its file name without extension.
        """
