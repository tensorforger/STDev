from pathlib import Path

import torch
import torch.nn.functional as functional
from diffusers import AutoencoderKLQwenImage21

from stdev.qwen_upscalers.base import Upscaler

RUN_NAME = "run_0016"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEVICE = "cuda"
LATENT_CHANNELS = 64
DECODER_PATH = REPOSITORY_ROOT / "models" / "texture-fix-vae-for-qwen-image-2.1"
ENCODER_PATH = REPOSITORY_ROOT / "models" / "Qwen-Image-2.1" / "vae"
PIXEL_BASE_CHUNK = 2


class PixelBase:
    """
    Deterministic on-manifold part of the upscale: E(bicubic 2x(D(latents_512))) with the texture-fix decoder and the
    official encoder (bf16), exactly as scripts/encode_qwen_pixel_base.py builds the training bases.
    """

    def __init__(self) -> None:
        self.decoder = AutoencoderKLQwenImage21.from_pretrained(str(DECODER_PATH), torch_dtype=torch.bfloat16)
        self.decoder = self.decoder.to(DEVICE).eval().requires_grad_(False)
        self.encoder = AutoencoderKLQwenImage21.from_pretrained(str(ENCODER_PATH), torch_dtype=torch.bfloat16)
        self.encoder = self.encoder.to(DEVICE).eval().requires_grad_(False)
        shape = (1, LATENT_CHANNELS, 1, 1, 1)
        self.mean = torch.tensor(self.decoder.config.latents_mean, device=DEVICE).view(shape)
        self.std = torch.tensor(self.decoder.config.latents_std, device=DEVICE).view(shape)

    @torch.no_grad()
    def __call__(self, latents_source: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents_source: [B, C, H, W], normalized
        returns:
            latents_base: [B, C, 2H, 2W], normalized, dtype: torch.float32
        """
        bases = []
        for start in range(0, latents_source.shape[0], PIXEL_BASE_CHUNK):
            chunk = latents_source[start : start + PIXEL_BASE_CHUNK].float().unsqueeze(2)
            denormalized = (chunk * self.std + self.mean).to(torch.bfloat16)
            pixels = self.decoder.decode(denormalized, return_dict=False)[0].float()[:, :3, 0]
            pixels = ((pixels + 1.0) / 2.0).clamp(0, 1)
            upscaled = functional.interpolate(pixels, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
            rgba = torch.cat([upscaled, torch.ones_like(upscaled[:, :1])], dim=1) * 2.0 - 1.0
            encoded = self.encoder.encode(rgba.unsqueeze(2).to(torch.bfloat16)).latent_dist.mode().float()
            bases.append(((encoded - self.mean) / self.std)[:, :, 0])
        return torch.cat(bases)


class Run0016Upscaler(Upscaler):
    """
    Reference (no learning): the pixel re-encode base E(bicubic 2x(D(source))), texture-fix decoder, official encoder.
    single_forward measures the whole base computation (decode 512, bicubic, encode 1024).
    """

    def __init__(self) -> None:
        self.pixel_base: PixelBase | None = None

    def train(self, smoke_mode: bool = False) -> None:
        print(f"{RUN_NAME}: no training (reference)", flush=True)

    def prepare_to_benchmark(self) -> None:
        if self.pixel_base is None:
            self.pixel_base = PixelBase()

    def compile(self) -> None:
        pass

    @torch.no_grad()
    def upscale(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C, H, W], dtype = float32 (or float16, bfloat16), device = "cuda"
        returns:
            latents: [B, C, 2H, 2W], same dtype as the input
        """
        return self.pixel_base(latents).to(latents.dtype)

    @torch.no_grad()
    def single_forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.pixel_base(latents)

    def get_name(self) -> str:
        return RUN_NAME
