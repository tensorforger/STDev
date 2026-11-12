import torch

from diffusers import AutoencoderKLWan

from stdev.vaes.vae import VAE


class Wan21VAE(VAE):
    """
    Wan 2.1 VAE, 16 latent channels, per-channel mean/std latent normalization.

    Usage rules:
    - Type: video (as_image=False) or image (as_image=True), causal in time.
    - Video pixels [B, 3, T, H, W] RGB in [0...1]; image pixels [B, 3, H, W].
    - H and W must be divisible by 8; T must be 1 + 4k. Latents [B, 16, 1 + k, H/8, W/8].
    - Decode: latents [B, 16, L, h, w] -> pixels [B, 3, 1 + 4(L - 1), 8h, 8w].
    - First frame is encoded alone (T = 1 gives L = 1), so image mode equals a one-frame video.
    """

    NAME = "wan_2_1"

    def __init__(
        self,
        path_to_model: str,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        as_image: bool = False,
    ) -> None:
        model = AutoencoderKLWan.from_pretrained(path_to_model).to(device, dtype)
        super().__init__(
            pixel_channels=3,
            latent_channels=model.config.z_dim,
            spatial_compression=model.config.scale_factor_spatial,
            temporal_compression=1 if as_image else model.config.scale_factor_temporal,
            is_video=not as_image,
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
        video = self.model.decode(latents, return_dict=False)[0].float()
        return (video + 1.0) / 2.0
