import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from spandrel import ImageModelDescriptor, ModelLoader

from stdev.flux_upscale_benchmark.solution import LATENT_CHANNELS, FluxUpscaleSolution
from stdev.vaes.flux_2_vae import Flux2VAE

VAE_DTYPE = torch.bfloat16
SESQUI_WEIGHTS_NAME = "upscaler_Flux2.safetensors"


def to_raw_latents(vae: Flux2VAE, latents: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents: [B, 32, H, W], normalized FLUX.2 latents (batch norm applied on the 2x2 patchified latents)
    returns:
        latents: [B, 32, H, W], raw FLUX.2 VAE encoder latents
    """
    return vae.unpatchify_latents(vae.patchify_latents(latents) * vae.std + vae.mean)


def to_normalized_latents(vae: Flux2VAE, latents: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents: [B, 32, H, W], raw FLUX.2 VAE encoder latents
    returns:
        latents: [B, 32, H, W], normalized FLUX.2 latents
    """
    return vae.unpatchify_latents((vae.patchify_latents(latents) - vae.mean) / vae.std)


class FullVAESolution(FluxUpscaleSolution):
    """
    Base of the solutions that use the full FLUX.2 VAE decoder at 512 or on upscaled latents.
    """

    def __init__(self, name: str, description: str, vae_path: Path) -> None:
        super().__init__(name, description)
        self.vae_path = vae_path

    def load(self) -> None:
        self.vae = Flux2VAE(str(self.vae_path), device="cuda", dtype=VAE_DTYPE)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, 32, h, w], normalized FLUX.2 latents
        returns:
            pixels: [B, 3, 8h, 8w], pixel_values: [0...1] (not clamped)
        """
        return self.vae.decode(latents.unsqueeze(2))


class DecodeBicubicSolution(FullVAESolution):
    def __init__(self, name: str, vae_path: Path) -> None:
        super().__init__(name, "Full VAE decode, bicubic 2x upsampling in pixel space", vae_path)

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        pixels = self.decode(latents).clamp(0, 1)  # [B, 3, 512, 512]
        return functional.interpolate(pixels, scale_factor=2, mode="bicubic", align_corners=False)


class LatentBicubicSolution(FullVAESolution):
    def __init__(self, name: str, vae_path: Path) -> None:
        super().__init__(
            name, "Bicubic 2x stretch of the normalized latents, full VAE decode", vae_path
        )

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        stretched = functional.interpolate(
            latents, scale_factor=2, mode="bicubic", align_corners=False
        )  # [B, 32, 128, 128]
        return self.decode(stretched)


class SesquiLSRSolution(FullVAESolution):
    """
    SesquiLSR (https://github.com/LoganBooker/SesquiLSR) FLUX.2 latent upscaler at scale 2.0, bf16 as in its node.
    It works on raw unpatchified 32 channel latents: the normalized benchmark input is denormalized with the batch
    norm statistics of the VAE (what its `make_flux2` adaptor does), upscaled, normalized again and decoded.
    """

    def __init__(self, name: str, vae_path: Path, repository_path: Path) -> None:
        super().__init__(name, "SesquiLSR latent upscaler (FLUX.2), full VAE decode", vae_path)
        self.repository_path = repository_path

    def load(self) -> None:
        super().load()
        sys.path.insert(0, str(self.repository_path))
        from sesqui_lsr import load_upscaler

        self.upscaler = load_upscaler(
            load_file(str(self.repository_path / "models" / SESQUI_WEIGHTS_NAME))
        )
        if self.upscaler.head_conv[0].weight.shape[1] != LATENT_CHANNELS:
            raise ValueError(f"{SESQUI_WEIGHTS_NAME} is not a {LATENT_CHANNELS} channel upscaler")
        self.upscaler = self.upscaler.to("cuda", torch.bfloat16)

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        raw_latents = to_raw_latents(self.vae, latents)  # [B, 32, 64, 64]
        target_size = (2 * latents.shape[2], 2 * latents.shape[3])
        upscaled = self.upscaler(raw_latents.to(torch.bfloat16), target_size).float()
        return self.decode(to_normalized_latents(self.vae, upscaled))

    def get_settings(self) -> dict[str, object]:
        return {"weights": SESQUI_WEIGHTS_NAME, "scale": 2.0, "dtype": "bfloat16"}


class DecodeSuperResolutionSolution(FullVAESolution):
    """
    Full VAE decode at 512, then a 2x pixel space super resolution network loaded with spandrel.
    """

    def __init__(self, name: str, description: str, vae_path: Path, weights_path: Path) -> None:
        super().__init__(name, description, vae_path)
        self.weights_path = weights_path

    def load(self) -> None:
        super().load()
        descriptor = ModelLoader().load_from_file(str(self.weights_path))
        if not isinstance(descriptor, ImageModelDescriptor) or descriptor.scale != 2:
            raise ValueError(f"{self.weights_path} is not a 2x image super resolution model")
        descriptor = descriptor.to("cuda").eval()
        self.network_dtype = torch.bfloat16 if descriptor.supports_bfloat16 else torch.float32
        self.descriptor = (
            descriptor.bfloat16() if descriptor.supports_bfloat16 else descriptor.float()
        )

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        pixels = self.decode(latents).clamp(0, 1)  # [B, 3, 512, 512]
        return self.descriptor(pixels.to(self.network_dtype)).float()

    def get_settings(self) -> dict[str, object]:
        return {"weights": self.weights_path.name, "dtype": str(self.network_dtype)}
