import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from spandrel import ImageModelDescriptor, ModelLoader

from stdev.upscale_benchmark.nn_latent_upscaler import WanNNLatentUpscaler
from stdev.upscale_benchmark.solution import LATENT_CHANNELS, UpscaleSolution
from stdev.vaes.wan_2_1_vae import Wan21VAE

VAE_DTYPE = torch.bfloat16
SESQUI_WEIGHTS_NAME = "upscaler_Wan21.safetensors"


class Wan21UpscaleDecoderVAE(Wan21VAE):
    """
    Wan 2.1 VAE with the 2x upscaling decoder of https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x: the
    decoder predicts 12 channels that are pixel shuffled into a 2x larger image. The encoder is unchanged.
    """

    NAME = "wan_2_1_upscale2x"

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = (latents * self.std + self.mean).to(self.dtype)
        decoded = self.model.decode(latents, return_dict=False)[0].float()  # [B, 12, T, H, W]
        pixels = functional.pixel_shuffle(decoded.movedim(2, 1), upscale_factor=2).movedim(1, 2)
        return (pixels + 1.0) / 2.0


class FullVAESolution(UpscaleSolution):
    """
    Base of the solutions that use the full Wan 2.1 VAE decoder at the original 512 resolution or on latents.
    """

    def __init__(self, name: str, description: str, vae_path: Path) -> None:
        super().__init__(name, description)
        self.vae_path = vae_path

    def load(self) -> None:
        self.vae = Wan21VAE(str(self.vae_path), device="cuda", dtype=VAE_DTYPE, as_image=True)


class DecodeBicubicSolution(FullVAESolution):
    def __init__(self, name: str, vae_path: Path) -> None:
        super().__init__(name, "Full VAE decode, bicubic 2x upsampling in pixel space", vae_path)

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        pixels = self.vae.decode(latents).clamp(0, 1)  # [B, 3, 512, 512]
        return functional.interpolate(pixels, scale_factor=2, mode="bicubic", align_corners=False)


class LatentBicubicSolution(FullVAESolution):
    def __init__(self, name: str, vae_path: Path) -> None:
        super().__init__(name, "Bicubic 2x latent stretch, full VAE decode", vae_path)

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        stretched = functional.interpolate(
            latents.squeeze(2), scale_factor=2, mode="bicubic", align_corners=False
        )  # [B, 16, 128, 128]
        return self.vae.decode(stretched.unsqueeze(2))


class WanNNLatentUpscalerSolution(FullVAESolution):
    """
    Neural latent upscaler of ComfyUI-VAE-Utils on raw latents (as in ComfyUI), full VAE decode.
    """

    def __init__(self, name: str, vae_path: Path, weights_path: Path) -> None:
        super().__init__(
            name, "Neural latent upscaler (WanNNLatentUpscaler), full VAE decode", vae_path
        )
        self.weights_path = weights_path

    def load(self) -> None:
        super().load()
        self.upscaler = WanNNLatentUpscaler()
        self.upscaler.load_state_dict(load_file(str(self.weights_path)))
        self.upscaler = self.upscaler.to("cuda").eval().requires_grad_(False)

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        raw_latents = latents * self.vae.std + self.vae.mean
        upscaled = self.upscaler(raw_latents)
        return self.vae.decode((upscaled - self.vae.mean) / self.vae.std)

    def get_settings(self) -> dict[str, object]:
        return {"weights": self.weights_path.name, "dtype": "float32"}


class SesquiLSRSolution(FullVAESolution):
    """
    SesquiLSR (https://github.com/LoganBooker/SesquiLSR) Wan 2.1 latent upscaler at scale 2.0 on raw latents (identity
    adaptor in ComfyUI), bf16 as in its node, full VAE decode. The repository is cloned to `repository_path`.
    """

    def __init__(self, name: str, vae_path: Path, repository_path: Path) -> None:
        super().__init__(name, "SesquiLSR latent upscaler (Wan 2.1), full VAE decode", vae_path)
        self.repository_path = repository_path

    def load(self) -> None:
        super().load()
        sys.path.insert(0, str(self.repository_path))
        from sesqui_lsr import LatentUpscaler

        self.upscaler = LatentUpscaler(in_channels=LATENT_CHANNELS)
        self.upscaler.load_state_dict(
            load_file(str(self.repository_path / "models" / SESQUI_WEIGHTS_NAME))
        )
        self.upscaler = self.upscaler.to("cuda", torch.bfloat16).eval().requires_grad_(False)

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        raw_latents = (latents * self.vae.std + self.vae.mean).squeeze(2)  # [B, 16, 64, 64]
        upscaled = self.upscaler(raw_latents.to(torch.bfloat16), (128, 128)).float()
        return self.vae.decode(((upscaled.unsqueeze(2)) - self.vae.mean) / self.vae.std)

    def get_settings(self) -> dict[str, object]:
        return {"weights": SESQUI_WEIGHTS_NAME, "scale": 2.0, "dtype": "bfloat16"}


class UpscaleVAESolution(UpscaleSolution):
    def __init__(self, name: str, upscale_vae_path: Path) -> None:
        super().__init__(
            name, "Wan2.1-VAE-upscale2x decoder (2x upscaling integrated into the decoder)"
        )
        self.upscale_vae_path = upscale_vae_path

    def load(self) -> None:
        self.vae = Wan21UpscaleDecoderVAE(
            str(self.upscale_vae_path), device="cuda", dtype=VAE_DTYPE, as_image=True
        )

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.vae.decode(latents)

    def get_settings(self) -> dict[str, object]:
        return {"weights": self.upscale_vae_path.name}


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
        pixels = self.vae.decode(latents).clamp(0, 1)  # [B, 3, 512, 512]
        return self.descriptor(pixels.to(self.network_dtype)).float()

    def get_settings(self) -> dict[str, object]:
        return {"weights": self.weights_path.name, "dtype": str(self.network_dtype)}
