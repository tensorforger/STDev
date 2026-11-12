import torch

from stdev.vaes.tiny_vae import TinyVAE


class TinyLTX25VAE(TinyVAE):
    """
    Tiny (TAEHV) counterpart of the LTX 2.5 VAE, same normalized latent space, wide decoder.

    Usage rules:
    - path_to_model is the taeltx2_3_wide.safetensors file. Pixels [B, 3, T, H, W] RGB in [0...1].
    - H and W must be divisible by 32; T must be 1 + 8k. Latents [B, 128, 1 + k, H/32, W/32].
    """

    NAME = "tiny_ltx_2_5"

    def __init__(
        self,
        path_to_model: str,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            path_to_model,
            latent_channels=128,
            spatial_compression=32,
            temporal_compression=8,
            device=device,
            dtype=dtype,
        )
