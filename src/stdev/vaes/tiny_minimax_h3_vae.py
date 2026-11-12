import torch

from stdev.vaes.tiny_vae import TinyVAE


class TinyMiniMaxH3VAE(TinyVAE):
    """
    Tiny (TAEHV) counterpart of the MiniMax-H3 VAE, same normalized latent space and 17-frame chunking.

    Usage rules:
    - path_to_model is the taeh3.safetensors file. Pixels [B, 3, T, H, W] RGB in [0...1].
    - H and W must be divisible by 16; T must be 5 + 17n with n >= 1 (22, 39, 56, ...).
    - Latent frames: T = 5 + 17n gives 2 + 5n. Latents [B, 24, L, H/16, W/16].
    - Decode: L = 2 + 5n gives exactly 5 + 17n frames.
    """

    NAME = "tiny_minimax_h3"

    def __init__(
        self,
        path_to_model: str,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            path_to_model,
            latent_channels=24,
            spatial_compression=16,
            temporal_compression=17,
            device=device,
            dtype=dtype,
        )

    def is_valid_num_frames(self, num_frames: int) -> bool:
        return (num_frames - 5) % 17 == 0 and num_frames >= 22

    def num_latent_frames(self, num_frames: int) -> int:
        return 2 + 5 * ((num_frames - 5) // 17)

    def num_pixel_frames(self, num_latent_frames: int) -> int:
        return 5 + 17 * ((num_latent_frames - 2) // 5)
