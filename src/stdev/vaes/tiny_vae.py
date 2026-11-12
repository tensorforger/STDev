import torch

from stdev.vaes.taehv import TAEHV
from stdev.vaes.vae import VAE


class TinyVAE(VAE):
    """
    Shared base of TAEHV-based tiny video VAEs. Latents match the normalized latent space of the full VAE,
    so no mean/std handling is needed. Subclasses set NAME, the weights file name and the compression factors.
    """

    def __init__(
        self,
        path_to_model: str,
        latent_channels: int,
        spatial_compression: int,
        temporal_compression: int,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            pixel_channels=3,
            latent_channels=latent_channels,
            spatial_compression=spatial_compression,
            temporal_compression=temporal_compression,
            is_video=True,
        )
        self.model = TAEHV(checkpoint_path=path_to_model).to(device, dtype).eval()

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        video = pixels.permute(0, 2, 1, 3, 4).to(self.dtype)  # [B, T, C, H, W]
        latents = self.model.encode_video(video, parallel=True, show_progress_bar=False)
        return latents.permute(0, 2, 1, 3, 4)

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4).to(self.dtype)  # [B, T, C, H, W]
        video = self.model.decode_video(latents, parallel=True, show_progress_bar=False)
        return video.permute(0, 2, 1, 3, 4)
