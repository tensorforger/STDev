"""
Tiny FLUX.2 VAE: TAEF2 of https://huggingface.co/madebyollin/taef2 (architecture of https://github.com/madebyollin/taesd,
`flux_2` variant with group norm pooling in the 8x downscaled blocks). TAEF2 works directly in the normalized FLUX.2
latent space (batch norm statistics already applied, unpatchified 32 channels), the same as `Flux2VAE`.
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file

from stdev.vaes.vae import VAE

LATENT_CHANNELS = 32
HIDDEN_CHANNELS = 64
CLAMP_SCALE = 3.0
POOL_GROUPS = 4
POOL_EXPANSION = 4


def make_convolution(input_channels: int, output_channels: int, **kwargs: object) -> nn.Conv2d:
    return nn.Conv2d(input_channels, output_channels, 3, padding=1, **kwargs)


class SoftClamp(nn.Module):
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.tanh(latents / CLAMP_SCALE) * CLAMP_SCALE


class TinyBlock(nn.Module):
    """
    Residual block of TAESD with optional global pooling branch (group norm over a 1x1 expansion).
    """

    def __init__(self, input_channels: int, output_channels: int, use_pool: bool = False) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            make_convolution(input_channels, output_channels),
            nn.ReLU(),
            make_convolution(output_channels, output_channels),
            nn.ReLU(),
            make_convolution(output_channels, output_channels),
        )
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1, bias=False)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.fuse = nn.ReLU()
        self.pool = None
        if use_pool:
            pooled_channels = input_channels * POOL_EXPANSION
            self.pool = nn.Sequential(
                nn.Conv2d(input_channels, pooled_channels, 1, bias=False),
                nn.GroupNorm(POOL_GROUPS, pooled_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(pooled_channels, input_channels, 1, bias=False),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            features = features + self.pool(features)
        return self.fuse(self.conv(features) + self.skip(features))


def build_encoder() -> nn.Sequential:
    channels = HIDDEN_CHANNELS
    return nn.Sequential(
        make_convolution(3, channels),
        TinyBlock(channels, channels),
        make_convolution(channels, channels, stride=2, bias=False),
        *[TinyBlock(channels, channels) for _ in range(3)],
        make_convolution(channels, channels, stride=2, bias=False),
        *[TinyBlock(channels, channels) for _ in range(3)],
        make_convolution(channels, channels, stride=2, bias=False),
        *[TinyBlock(channels, channels, use_pool=True) for _ in range(3)],
        make_convolution(channels, LATENT_CHANNELS),
    )


def build_decoder() -> nn.Sequential:
    channels = HIDDEN_CHANNELS
    return nn.Sequential(
        SoftClamp(),
        make_convolution(LATENT_CHANNELS, channels),
        nn.ReLU(),
        *[TinyBlock(channels, channels, use_pool=True) for _ in range(3)],
        nn.Upsample(scale_factor=2),
        make_convolution(channels, channels, bias=False),
        *[TinyBlock(channels, channels) for _ in range(3)],
        nn.Upsample(scale_factor=2),
        make_convolution(channels, channels, bias=False),
        *[TinyBlock(channels, channels) for _ in range(3)],
        nn.Upsample(scale_factor=2),
        make_convolution(channels, channels, bias=False),
        TinyBlock(channels, channels),
        make_convolution(channels, 3),
    )


def convert_diffusers_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    `encoder.layers.<i>.*` / `decoder.layers.<i>.*` (diffusers AutoencoderTiny naming) to the indices of
    `build_encoder` / `build_decoder` (the decoder has the soft clamp as an extra first module).
    """
    converted = {}
    for key, tensor in state_dict.items():
        part, _, index, *suffix = key.split(".")
        offset = 1 if part == "decoder" else 0
        converted[".".join([part, str(int(index) + offset), *suffix])] = tensor
    return converted


class TinyFluxAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = build_encoder()
        self.decoder = build_decoder()


class TinyFlux2VAE(VAE):
    """
    Tiny (TAEF2) counterpart of the FLUX.2 VAE, same normalized latent space.

    Usage rules:
    - path_to_model is the taef2.safetensors file. Pixels [B, 3, H, W] RGB in [0...1].
    - H and W must be divisible by 16. Latents [B, 32, 1, H/8, W/8].
    """

    NAME = "tiny_flux_2"

    def __init__(
        self,
        path_to_model: str,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            pixel_channels=3,
            latent_channels=LATENT_CHANNELS,
            spatial_compression=8,
            temporal_compression=1,
            is_video=False,
            size_multiple=16,
        )
        model = TinyFluxAutoencoder()
        model.load_state_dict(convert_diffusers_state_dict(load_file(path_to_model)))
        self.model = model.to(device, dtype).eval()

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.model.encoder(pixels[:, :, 0].to(self.dtype)).float().unsqueeze(2)

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model.decoder(latents[:, :, 0].to(self.dtype)).float().unsqueeze(2)
