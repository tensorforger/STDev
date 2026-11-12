from abc import ABC, abstractmethod

import torch


class VAE(torch.nn.Module, ABC):
    """
    Common interface for all image and video VAE wrappers. Intended for inference only.

    Latent space is always [B, C, T, H, W] (T = 1 for images), normalized to roughly mean=0, std=1.
    Pixel space is always normalized to [0...1]; the wrapper handles the model-specific range itself.
    Image VAEs (is_video=False) use pixels [B, C, H, W], video VAEs (is_video=True) use [B, C, T, H, W].
    Inputs and outputs are float32 regardless of weights dtype. Decoded pixels are not clamped.
    Subclasses must define the class constant NAME equal to the config entry.
    """

    NAME: str

    def __init__(
        self,
        pixel_channels: int,
        latent_channels: int,
        spatial_compression: int,
        temporal_compression: int,
        is_video: bool,
        size_multiple: int | None = None,
    ) -> None:
        super().__init__()
        self.size_multiple = size_multiple or spatial_compression
        self.pixel_channels = pixel_channels
        self.latent_channels = latent_channels
        self.spatial_compression = spatial_compression
        self.temporal_compression = temporal_compression
        self.is_video = is_video

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def is_valid_num_frames(self, num_frames: int) -> bool:
        """
        returns:
            whether encode accepts this number of pixel frames
        """
        return (num_frames - 1) % self.temporal_compression == 0

    def num_latent_frames(self, num_frames: int) -> int:
        """
        args:
            num_frames: pixel frames accepted by encode
        returns:
            latent frames produced by encode
        """
        return 1 + (num_frames - 1) // self.temporal_compression

    def num_pixel_frames(self, num_latent_frames: int) -> int:
        """
        args:
            num_latent_frames: latent frames passed to decode
        returns:
            pixel frames produced by decode
        """
        return 1 + (num_latent_frames - 1) * self.temporal_compression

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        args:
            pixels: image VAE: [B, C, H, W], video VAE: [B, C, T, H, W], pixel_values: [0...1]
        returns:
            latents: [B, C_latent, T_latent, H_latent, W_latent], dtype: torch.float32
        """
        expected_dimensions = 5 if self.is_video else 4
        if pixels.ndim != expected_dimensions:
            raise ValueError(
                f"{self.NAME}: expected {expected_dimensions}D pixels, got {tuple(pixels.shape)}"
            )
        if pixels.shape[1] != self.pixel_channels:
            raise ValueError(
                f"{self.NAME}: expected {self.pixel_channels} pixel channels, got {pixels.shape[1]}"
            )
        if pixels.shape[-2] % self.size_multiple or pixels.shape[-1] % self.size_multiple:
            raise ValueError(f"{self.NAME}: H and W must be divisible by {self.size_multiple}")
        if self.is_video and not self.is_valid_num_frames(pixels.shape[2]):
            raise ValueError(f"{self.NAME}: invalid number of frames {pixels.shape[2]}")

        if not self.is_video:
            pixels = pixels.unsqueeze(2)
        return self._encode(pixels.to(self.device).float()).float()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C_latent, T_latent, H_latent, W_latent]
        returns:
            pixels: image VAE: [B, C, H, W], video VAE: [B, C, T, H, W], pixel_values: [0...1], dtype: torch.float32
        """
        if latents.ndim != 5 or latents.shape[1] != self.latent_channels:
            raise ValueError(
                f"{self.NAME}: expected [B, {self.latent_channels}, T, H, W], got {tuple(latents.shape)}"
            )
        if not self.is_video and latents.shape[2] != 1:
            raise ValueError(f"{self.NAME}: image VAE expects T_latent = 1, got {latents.shape[2]}")

        pixels = self._decode(latents.to(self.device).float()).float()
        return pixels if self.is_video else pixels.squeeze(2)

    @abstractmethod
    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        args:
            pixels: [B, C, T, H, W], pixel_values: [0...1], dtype: torch.float32
        returns:
            latents: [B, C_latent, T_latent, H_latent, W_latent], normalized
        """

    @abstractmethod
    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C_latent, T_latent, H_latent, W_latent], normalized, dtype: torch.float32
        returns:
            pixels: [B, C, T, H, W], pixel_values: [0...1]
        """

    def get_latent_channels(self) -> int:
        return self.latent_channels

    def get_spatial_compression(self) -> int:
        return self.spatial_compression

    def get_temporal_compression(self) -> int:
        return self.temporal_compression

    def get_vae_name(self) -> str:
        return self.NAME
