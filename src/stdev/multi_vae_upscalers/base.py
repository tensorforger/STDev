from abc import ABC, abstractmethod

import torch


class MultiVAEUpscaler(ABC):
    """
    Self-contained high-level wrapper of a latent upscaler that serves many VAE latent spaces with
    one set of weights. Latents use the layout of stdev.vaes.vae.VAE: [B, C, T, H, W], normalized,
    T = 1 for images. Each definition file should contain the implementation of this interface.
    """

    @abstractmethod
    def supported_vae_names(self) -> tuple[str, ...]:
        """
        Names (VAE.NAME) of the latent spaces the model was built for.
        """

    @abstractmethod
    def prepare_to_inference(self, weights_path: str) -> None:
        """
        Loads the weights and moves the model to the device. Must be called before upscale().
        """

    @abstractmethod
    def compile(self) -> None:
        """
        Compiles the model with torch.compile. Called after prepare_to_inference().
        """

    @abstractmethod
    def upscale(
        self,
        latents: torch.Tensor,
        vae_name: str,
        high_frequency_feature: float | None = None,
    ) -> torch.Tensor:
        """
        args:
            latents: [B, C, T, H_lr, W_lr], normalized latents of the VAE `vae_name`
            vae_name: one of supported_vae_names()
            high_frequency_feature: quantile normalized HFF of the target, [0...1], None = unconditional
        returns:
            latents: [B, C, T, H_lr * 2, W_lr * 2], dtype: same as the input
        """

    @abstractmethod
    def single_forward(self, latents: torch.Tensor, vae_name: str) -> torch.Tensor:
        """
        Runs the underlying model exactly once on random noise and a fixed noise level. Used only
        for latency benchmarks.
        args:
            latents: [B, C, T, H_lr, W_lr]
        returns:
            velocity: [B * T, C, H_lr * 2, W_lr * 2]
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the upscaler.
        """
