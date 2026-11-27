import torch

from diffusers import AutoencoderKLMiniMaxH3Image

from stdev.vaes.vae import VAE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MiniMaxH3ImageVAE(VAE):
    """
    MiniMax-H3-Image VAE: causal 3D CNN encoder run on a single frame, 2D ViT decoder (24 blocks on the latent grid,
    one linear 16x16 patch projection, no progressive upsampling). 24 latent channels, 16x spatial compression,
    per-channel mean / std latent normalization, ImageNet normalized pixels inside the wrapper.
    Also loads decoder fine-tunes with the same encoder (`ostris/ostris_h3_image_vae`).

    Usage rules:
    - Type: image. Pixels [B, 3, H, W] RGB in [0...1]; H and W divisible by 16. Latents [B, 24, 1, H/16, W/16].
    - Tiling (256 px tiles, >= 64 px linear blend overlap) is on by default for encode and decode, as released;
      `use_tiling=False` encodes / decodes the whole image at once (full attention over all tokens in the decoder).
    - The encoder stays fp32 (pinned by diffusers) even with bf16 weights; `dtype` reaches the decoder blocks.
    """

    NAME = "minimax_h3_image"

    def __init__(
        self,
        path_to_model: str,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        use_tiling: bool = True,
    ) -> None:
        model = AutoencoderKLMiniMaxH3Image.from_pretrained(path_to_model, torch_dtype=dtype).to(device)
        super().__init__(
            pixel_channels=model.config.in_channels,
            latent_channels=model.config.latent_channels,
            spatial_compression=model.spatial_compression_ratio,
            temporal_compression=1,
            is_video=False,
        )
        if use_tiling:
            model.enable_tiling()
        else:
            model.disable_tiling()
        self.model = model.eval()
        latent_shape = (1, self.latent_channels, 1, 1, 1)
        pixel_shape = (1, 3, 1, 1)
        self.register_buffer(
            "mean", torch.tensor(model.config.latents_mean, device=device).view(latent_shape), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(model.config.latents_std, device=device).view(latent_shape), persistent=False
        )
        self.register_buffer(
            "pixel_mean", torch.tensor(IMAGENET_MEAN, device=device).view(pixel_shape), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(IMAGENET_STD, device=device).view(pixel_shape), persistent=False
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.decoder.transformer_blocks.parameters()).dtype

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        image = (pixels[:, :, 0] - self.pixel_mean) / self.pixel_std
        latents = self.model.encode(image).latent_dist.mode().float().unsqueeze(2)
        return (latents - self.mean) / self.std

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        raw_latents = (latents * self.std + self.mean)[:, :, 0].to(self.dtype)
        image = self.model.decode(raw_latents, return_dict=False)[0].float()
        return (image * self.pixel_std + self.pixel_mean).unsqueeze(2)
