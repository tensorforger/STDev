import torch

from diffusers import AutoencoderKLLTX2Video

from stdev.vaes.vae import VAE


class LTX25VAE(VAE):
    """
    LTX 2.5 video VAE, 128 latent channels, per-channel mean/std and scaling_factor latent normalization.

    Usage rules:
    - Type: video, causal encoder (non-causal decoder). Pixels [B, 3, T, H, W] RGB in [0...1].
    - H and W must be divisible by 32; T must be 1 + 8k. Latents [B, 128, 1 + k, H/32, W/32].
    - Decode: latents [B, 128, L, h, w] -> pixels [B, 3, 1 + 8(L - 1), 32h, 32w].
    - T = 1 encodes a single image into one latent frame.
    """

    NAME = "ltx_2_5"

    def __init__(
        self,
        path_to_model: str,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        model = AutoencoderKLLTX2Video.from_pretrained(path_to_model).to(device, dtype)
        super().__init__(
            pixel_channels=model.config.in_channels,
            latent_channels=model.config.latent_channels,
            spatial_compression=model.spatial_compression_ratio,
            temporal_compression=model.temporal_compression_ratio,
            is_video=True,
        )
        self.model = model.eval()
        self.scaling_factor = model.config.scaling_factor

        shape = (1, self.latent_channels, 1, 1, 1)
        self.register_buffer(
            "mean", model.latents_mean.detach().float().view(shape).to(device), persistent=False
        )
        self.register_buffer(
            "std", model.latents_std.detach().float().view(shape).to(device), persistent=False
        )

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        latents = self.model.encode((pixels * 2.0 - 1.0).to(self.dtype)).latent_dist.mode().float()
        return (latents - self.mean) * self.scaling_factor / self.std

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = (latents * self.std / self.scaling_factor + self.mean).to(self.dtype)
        video = self.model.decode(latents, return_dict=False)[0].float()
        return (video + 1.0) / 2.0
