import torch

from diffusers import AutoencoderKLQwenImage21

from stdev.vaes.vae import VAE


class QwenImage21VAE(VAE):
    """
    Qwen-Image 2.1 RGBA VAE, per-channel mean/std latent normalization.

    Usage rules:
    - Type: image, non-causal (single frame). Pixels [B, 4, H, W] RGBA in [0...1]; alpha 1 is opaque.
    - H and W must be divisible by the spatial compression (16). Latents [B, 64, 1, H/16, W/16].
    - Decode: latents [B, 64, 1, h, w] -> pixels [B, 4, 16h, 16w] including alpha.
    - Pass opaque images with an explicit alpha channel of ones.
    """

    NAME = "qwen_image_2_1"

    def __init__(
        self,
        path_to_model: str,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        model = AutoencoderKLQwenImage21.from_pretrained(path_to_model).to(device, dtype)
        super().__init__(
            pixel_channels=model.config.in_channels,
            latent_channels=model.config.z_dim,
            spatial_compression=model.spatial_compression_ratio,
            temporal_compression=1,
            is_video=False,
        )
        self.model = model.eval()

        shape = (1, self.latent_channels, 1, 1, 1)
        self.register_buffer(
            "mean",
            torch.tensor(model.config.latents_mean, device=device).view(shape),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(model.config.latents_std, device=device).view(shape),
            persistent=False,
        )

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        latents = self.model.encode((pixels * 2.0 - 1.0).to(self.dtype)).latent_dist.mode().float()
        return (latents - self.mean) / self.std

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = (latents * self.std + self.mean).to(self.dtype)
        pixels = self.model.decode(latents, return_dict=False)[0].float()
        return (pixels + 1.0) / 2.0
