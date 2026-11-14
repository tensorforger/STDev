from abc import ABC, abstractmethod

import torch


class Upscaler(ABC):
    """
    Self-contained high-level wrapper of a Wan latent upscaler.
    Each upscaler definition file should contain the implementation of this interface.

    Latents are always [B, C, T, H, W] (Wan 2.1 VAE: C = 16, T = 1 for images, 1 + 4k pixel frames give
    1 + k latent frames). One call can upscale space, time, or both: the target is given as `output_size`.
    The only conditioning is `hff`, the high frequency feature (Immerkaer estimator, see
    `stdev.hff.immerkaer`) of the target sample.
    """

    @abstractmethod
    def train(self, smoke_mode: bool = False) -> None:
        """
        Starts the training process and saves the weights.
        In smoke_mode should return after several training steps, covering all training sample kinds.
        """

    @abstractmethod
    def prepare_to_benchmark(self) -> None:
        """
        Should be called before benchmarking the upscaler. Loads the saved weights.
        """

    @abstractmethod
    def compile(self) -> None:
        """
        Compiles the model with torch.compile. Called after prepare_to_benchmark().
        """

    @abstractmethod
    def upscale(
        self,
        latents: torch.Tensor,
        hff: torch.Tensor,
        output_size: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        args:
            latents: [B, C, T_lr, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            hff: [B], values: [0...1], dtype = float32, high frequency feature of the target sample
            output_size: (T_hr, H_hr, W_hr), target latent size, each entry >= the input entry
        returns:
            latents: [B, C, T_hr, H_hr, W_hr], same dtype as the input
        """

    @abstractmethod
    def single_forward(
        self,
        latents: torch.Tensor,
        hff: torch.Tensor,
        output_size: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Runs the underlying model exactly once on dummy internal inputs (e.g. random noise and a
        fixed noise level). Used only by the latency benchmark, not for producing samples.
        args:
            latents: [B, C, T_lr, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            hff: [B], values: [0...1], dtype = float32
            output_size: (T_hr, H_hr, W_hr), target latent size
        returns:
            tensor: model output for the high resolution grid, any dtype
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the upscaler, equal to its file name without extension.
        """
