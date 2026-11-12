import torch

from stdev.vaes.tiny_vae import TinyVAE


class TinyWan21VAE(TinyVAE):
    """
    Tiny (TAEHV) counterpart of the Wan 2.1 VAE, same normalized latent space.

    Usage rules:
    - path_to_model is the taew2_1.safetensors file. Pixels [B, 3, T, H, W] RGB in [0...1].
    - H and W must be divisible by 8; T must be 1 + 4k. Latents [B, 16, 1 + k, H/8, W/8].
    """

    NAME = "tiny_wan_2_1"

    def __init__(
        self,
        path_to_model: str,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            path_to_model,
            latent_channels=16,
            spatial_compression=8,
            temporal_compression=4,
            device=device,
            dtype=dtype,
        )
