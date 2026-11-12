import torch

from diffusers import AutoencoderKLMiniMaxH3

from stdev.vaes.vae import VAE


class MiniMaxH3VAE(VAE):
    """
    MiniMax-H3 video VAE, 24 latent channels, per-channel mean/std latent normalization,
    pixels normalized with ImageNet mean/std inside the wrapper.

    Usage rules:
    - Type: video, chunked in 17-frame clips (not strictly causal). Pixels [B, 3, T, H, W] RGB in [0...1].
    - H and W must be divisible by 16; T must be 5 + 17n with n >= 1 (22, 39, 56, ...), or 1.
    - Latent frames: T = 5 + 17n gives 2 + 5n; T = 1 gives 1. Latents [B, 24, L, H/16, W/16].
    - Decode: L = 2 + 5n gives exactly 5 + 17n frames; other L >= 3 are padded internally.
    - Decode does not support L < 3, so a single-frame latent (T = 1) cannot be decoded.
    - Encoder is kept in fp32 even with bf16 weights, as required by the model.
    """

    NAME = "minimax_h3"

    def __init__(
        self,
        path_to_model: str,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        model = AutoencoderKLMiniMaxH3.from_pretrained(path_to_model, torch_dtype=dtype).to(device)
        super().__init__(
            pixel_channels=model.config.in_channels,
            latent_channels=model.config.latent_channels,
            spatial_compression=model.spatial_compression_ratio,
            temporal_compression=model.temporal_compression_ratio,
            is_video=True,
        )
        self.model = model.eval()
        self.encoder_dtype = next(model.encoder.parameters()).dtype
        self.decoder_dtype = next(model.decoder.parameters()).dtype

        latent_shape = (1, self.latent_channels, 1, 1, 1)
        pixel_shape = (1, 3, 1, 1, 1)
        self.register_buffer(
            "mean",
            torch.tensor(model.config.latents_mean, device=device).view(latent_shape),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(model.config.latents_std, device=device).view(latent_shape),
            persistent=False,
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(pixel_shape),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(pixel_shape),
            persistent=False,
        )

    def is_valid_num_frames(self, num_frames: int) -> bool:
        return num_frames == 1 or ((num_frames - 5) % 17 == 0 and num_frames >= 22)

    def num_latent_frames(self, num_frames: int) -> int:
        return 1 if num_frames == 1 else 2 + 5 * ((num_frames - 5) // 17)

    def num_pixel_frames(self, num_latent_frames: int) -> int:
        return 1 if num_latent_frames == 1 else 5 + 17 * ((num_latent_frames - 2) // 5)

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = ((pixels - self.pixel_mean) / self.pixel_std).to(self.encoder_dtype)
        latents = self.model.encode(pixels).latent_dist.mode().float()
        return (latents - self.mean) / self.std

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.shape[2] < 3:
            raise ValueError(
                f"{self.NAME}: decode requires at least 3 latent frames, got {latents.shape[2]}"
            )
        latents = (latents * self.std + self.mean).to(self.decoder_dtype)
        video = self.model.decode(latents, return_dict=False)[0].float()
        return video * self.pixel_std + self.pixel_mean
