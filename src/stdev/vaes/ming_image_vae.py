import torch

from diffusers import AutoencoderKLQwenImage

from stdev.vaes.vae import VAE

LATENTS_MEAN = (-0.0586, -0.0151, -0.0631, 0.0459, -0.0352, -0.0133, -0.0111, 0.0419, -0.0014, -0.0881, 0.0092, 0.0833, 0.0004, -0.0075, 0.0421, 0.0190)
LATENTS_STD = (0.1275, 0.0958, 0.1072, 0.1255, 0.0866, 0.1087, 0.1125, 0.0955, 0.1485, 0.0929, 0.0989, 0.0901, 0.1011, 0.0909, 0.1086, 0.1081)


class MingImageVAE(VAE):
    """
    RGBA VAE of Ming-Image 0.1 / 0.2 Design, byte-identical to the Qwen-Image-Layered VAE (Wan 2.1 / Qwen-Image
    architecture with 4 pixel channels, 16 latent channels, 8x spatial compression).

    Usage rules:
    - Type: image, single frame. Pixels [B, 4, H, W] RGBA in [0...1]; pass opaque images with an alpha channel of ones.
    - H and W must be divisible by 8. Latents [B, 16, 1, H/8, W/8].
    - Normalization: per channel mean / std of the raw posterior modes measured on 512 4KLSDB 1024 crops (opaque RGBA).
      Raw latents have std ~0.116 (= 1 / scaling_factor 8.0064 of the Ming config); the Qwen-Image-Layered config's Wan
      statistics do not fit these weights.
    """

    NAME = "ming_image"

    def __init__(
        self,
        path_to_model: str,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        model = AutoencoderKLQwenImage.from_pretrained(path_to_model).to(device, dtype)
        super().__init__(
            pixel_channels=model.config.input_channels,
            latent_channels=model.config.z_dim,
            spatial_compression=8,
            temporal_compression=1,
            is_video=False,
        )
        self.model = model.eval()
        shape = (1, self.latent_channels, 1, 1, 1)
        self.register_buffer("mean", torch.tensor(LATENTS_MEAN, device=device).view(shape), persistent=False)
        self.register_buffer("std", torch.tensor(LATENTS_STD, device=device).view(shape), persistent=False)

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        latents = self.model.encode((pixels * 2.0 - 1.0).to(self.dtype)).latent_dist.mode().float()
        return (latents - self.mean) / self.std

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = (latents * self.std + self.mean).to(self.dtype)
        pixels = self.model.decode(latents, return_dict=False)[0].float()
        return (pixels + 1.0) / 2.0
