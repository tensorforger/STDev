import torch

from diffusers import AutoencoderKL

from stdev.vaes.vae import VAE


class SDXLVAE(VAE):
    """
    SDXL VAE, 4 latent channels, scaling_factor normalization.

    Usage rules:
    - Type: image, non-causal. Pixels [B, 3, H, W] RGB in [0...1].
    - H and W must be divisible by 8. Latents [B, 4, 1, H/8, W/8].
    - Decode: latents [B, 4, 1, h, w] -> pixels [B, 3, 8h, 8w].
    - Original weights overflow in fp16; use fp32 or bf16.
    """

    NAME = "sdxl"

    def __init__(
        self,
        path_to_model: str,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        model = AutoencoderKL.from_pretrained(path_to_model).to(device, dtype)
        super().__init__(
            pixel_channels=3,
            latent_channels=model.config.latent_channels,
            spatial_compression=2 ** (len(model.config.block_out_channels) - 1),
            temporal_compression=1,
            is_video=False,
        )
        self.model = model.eval()
        self.scaling_factor = model.config.scaling_factor

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        image = (pixels[:, :, 0] * 2.0 - 1.0).to(self.dtype)
        latents = self.model.encode(image).latent_dist.mode().float()
        return (latents * self.scaling_factor).unsqueeze(2)

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = (latents[:, :, 0] / self.scaling_factor).to(self.dtype)
        image = self.model.decode(latents, return_dict=False)[0].float()
        return ((image + 1.0) / 2.0).unsqueeze(2)
