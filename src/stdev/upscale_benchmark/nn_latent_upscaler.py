"""
Architecture of the Wan 2.1 neural latent upscaler (2x) from https://github.com/spacepxl/ComfyUI-VAE-Utils
(`latent_upscale/model.py`, MIT license), re-typed with the original module names so that the released weights
`wan21_latent_upscale_2x.safetensors` load unchanged. It works on raw (unnormalized) Wan 2.1 latents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class LayerNorm3d(nn.LayerNorm):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C, T, H, W]
        returns:
            features: [B, C, T, H, W], normalized over C
        """
        features = features.permute(0, 2, 3, 4, 1)
        features = functional.layer_norm(
            features, self.normalized_shape, self.weight, self.bias, self.eps
        )
        return features.permute(0, 4, 1, 2, 3)


class DiCoBlock3d(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float, kernel_size: int) -> None:
        super().__init__()
        self.conv2 = nn.Conv3d(
            hidden_size,
            hidden_size,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, kernel_size // 2, kernel_size // 2),
            padding_mode="replicate",
        )
        self.conv3 = nn.Conv3d(hidden_size, hidden_size, kernel_size=1)
        self.cca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(hidden_size, hidden_size, kernel_size=1),
            nn.Sigmoid(),
        )
        feedforward_channels = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv3d(hidden_size, feedforward_channels, kernel_size=1)
        self.conv5 = nn.Conv3d(feedforward_channels, hidden_size, kernel_size=1)
        self.norm1 = LayerNorm3d(hidden_size, eps=1e-6, elementwise_affine=False)
        self.norm2 = LayerNorm3d(hidden_size, eps=1e-6, elementwise_affine=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        args:
            inputs: [B, C, T, H, W]
        returns:
            outputs: [B, C, T, H, W]
        """
        features = functional.gelu(self.conv2(self.norm1(inputs)))
        features = inputs + self.conv3(features * self.cca(features))
        return features + self.conv5(functional.gelu(self.conv4(self.norm2(features))))


class WanNNLatentUpscaler(nn.Module):
    """
    Convolutional residual upscaler: pixel shuffle of the network output added to the nearest neighbor upscaled input.
    """

    def __init__(
        self,
        channels: int = 16,
        hidden_size: int = 128,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        depth: int = 8,
        upscale: int = 2,
    ) -> None:
        super().__init__()
        self.upscale = upscale
        self.conv_in = nn.Conv3d(
            channels,
            hidden_size,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            padding_mode="replicate",
        )
        self.blocks = nn.ModuleList(
            DiCoBlock3d(hidden_size, mlp_ratio, kernel_size) for _ in range(depth)
        )
        self.conv_out = nn.Conv3d(
            hidden_size,
            channels * upscale**2,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            padding_mode="replicate",
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C, T, H, W], raw Wan 2.1 latents
        returns:
            latents: [B, C, T, 2 * H, 2 * W], raw Wan 2.1 latents
        """
        features = self.conv_in(latents)
        for block in self.blocks:
            features = block(features)
        residual = functional.pixel_shuffle(
            self.conv_out(features).movedim(1, 2), self.upscale
        ).movedim(2, 1)
        stretched = functional.interpolate(
            latents, scale_factor=(1, self.upscale, self.upscale), mode="nearest-exact"
        )
        return residual + stretched
