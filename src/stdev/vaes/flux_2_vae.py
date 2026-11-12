import torch

from diffusers.models import AutoencoderKLFlux2

from stdev.vaes.vae import VAE


class Flux2VAE(VAE):
    """
    FLUX.2 VAE, 32 latent channels, per-channel batch-norm statistics normalization.

    Usage rules:
    - Type: image, non-causal. Pixels [B, 3, H, W] RGB in [0...1].
    - H and W must be divisible by 8. Latents [B, 32, 1, H/8, W/8].
    - Decode: latents [B, 32, 1, h, w] -> pixels [B, 3, 8h, 8w].
    - Latent normalization uses 2x2-patchified statistics, so latent H and W must be even: pixel H and W divisible by 16.
    """

    NAME = "flux_2"

    def __init__(
        self,
        path_to_model: str,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        model = AutoencoderKLFlux2.from_pretrained(path_to_model)
        mean = model.bn.running_mean.view(1, -1, 1, 1).float()
        std = torch.sqrt(
            model.bn.running_var.view(1, -1, 1, 1).float() + model.config.batch_norm_eps
        )
        super().__init__(
            pixel_channels=3,
            latent_channels=model.config.latent_channels,
            spatial_compression=8,
            temporal_compression=1,
            is_video=False,
            size_multiple=16,
        )
        self.model = model.to(device, dtype).eval()
        self.register_buffer("mean", mean.to(device), persistent=False)
        self.register_buffer("std", std.to(device), persistent=False)

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        image = (pixels[:, :, 0] * 2.0 - 1.0).to(self.dtype)
        latents = self.model.encode(image).latent_dist.mean.float()
        latents = self.unpatchify_latents((self.patchify_latents(latents) - self.mean) / self.std)
        return latents.unsqueeze(2)

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self.unpatchify_latents(
            self.patchify_latents(latents[:, :, 0]) * self.std + self.mean
        )
        image = self.model.decode(latents.to(self.dtype), return_dict=False)[0].float()
        return ((image + 1.0) / 2.0).unsqueeze(2)

    @staticmethod
    def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C, H, W]
        returns:
            latents: [B, 4C, H/2, W/2]
        """
        batch_size, channels, height, width = latents.shape
        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(batch_size, channels * 4, height // 2, width // 2)

    @staticmethod
    def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, 4C, H, W]
        returns:
            latents: [B, C, 2H, 2W]
        """
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(batch_size, channels // 4, height * 2, width * 2)
