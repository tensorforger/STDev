import json
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.transforms.functional as vision_functional
from safetensors.torch import load_file

from stdev.flux_upscale_benchmark.baselines import FullVAESolution

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
COLOR_FIX_SIGMA = 5.0
COLOR_FIX_KERNEL_SIZE = 41
DINOV2_NAMES = {"dinov2b": "dinov2_vitb14", "dinov2l": "dinov2_vitl14"}


def wavelet_color_fix(pixels: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Low frequencies (gaussian blur, sigma 5) from the reference, high frequencies from the prediction, as the
    `wavelet` alignment of the VOSR inference script (cv2 GaussianBlur, reflect 101 border).

    args:
        pixels: [B, 3, H, W], pixel_values: [0...1]
        reference: [B, 3, H, W], pixel_values: [0...1]
    returns:
        pixels: [B, 3, H, W], pixel_values: [0...1]
    """
    blur = lambda image: vision_functional.gaussian_blur(
        image, [COLOR_FIX_KERNEL_SIZE] * 2, [COLOR_FIX_SIGMA] * 2
    )
    return (blur(reference) + pixels - blur(pixels)).clamp(0, 1)


class VOSRSolution(FullVAESolution):
    """
    VOSR one step super resolution (https://github.com/cswry/VOSR, CVPR 2026) in "decode, then upscale" mode: full
    FLUX.2 decode at 512, bicubic 2x to 1024, VOSR one step (own VAE latent space: SD 2.1 for 0.5B, Qwen-Image 2D for
    1.4B, DINOv2 condition of the bicubic image, no prompt), wavelet color fix against the bicubic image. Official
    defaults (fp32, posterior mode encode, no tiling). The repository is cloned to `repository_path`, the HF weights
    of CSWRY/VOSR are in `weights_path`.
    """

    def __init__(
        self,
        name: str,
        vae_path: Path,
        repository_path: Path,
        weights_path: Path,
        model_name: str,
        input_size: int = 512,
    ) -> None:
        factor = 1024 // input_size
        resize = "" if input_size == 512 else f", area downscale to {input_size}"
        super().__init__(
            name,
            f"Full VAE decode{resize}, then VOSR {model_name} one step {factor}x (bicubic input, wavelet color fix)",
            vae_path,
        )
        self.input_size = input_size
        self.repository_path = repository_path
        self.weights_path = weights_path
        self.model_name = model_name

    def load(self) -> None:
        super().load()
        if str(self.repository_path) not in sys.path:
            sys.path.insert(0, str(self.repository_path))
        from models.lightningdit import LightningDiT
        from tiled_vae import decode_latent, encode_latent

        self.encode_latent, self.decode_latent = encode_latent, decode_latent
        model_directory = self.weights_path / self.model_name
        arguments = types.SimpleNamespace(**json.loads((model_directory / "args.json").read_text()))
        arguments.auxiliary_time_cond = bool(getattr(arguments, "auxiliary_time_cond", True))
        self.arguments = arguments
        self.light_decoder = None
        if arguments.ae_type == "qwen":
            from models.qwenimage_vae2d import AutoencoderKLQwenImage2D

            self.sr_vae = AutoencoderKLQwenImage2D.from_pretrained(
                str(self.weights_path / "Qwen-Image-vae-2d")
            )
            latent_channels = 16
        else:
            from diffusers import AutoencoderKL

            from models.light_decoder import LightDecoder

            self.sr_vae = AutoencoderKL.from_pretrained(
                str(self.weights_path / "stable-diffusion-2-1-base"), subfolder="vae"
            )
            checkpoint = torch.load(self.weights_path / "sd21_lwdecoder.pth", map_location="cpu")
            config = checkpoint["config"]
            self.light_decoder = LightDecoder(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                block_out_channels=tuple(config["block_out_channels"]),
                layers_per_block=config["layers_per_block"],
            )
            self.light_decoder.load_state_dict(checkpoint["model_state_dict"])
            self.light_decoder = self.light_decoder.to("cuda").eval().requires_grad_(False)
            latent_channels = 4
        self.sr_vae = self.sr_vae.to("cuda").eval().requires_grad_(False)
        self.image_encoder = self._load_dinov2(arguments.enc_type)

        model = LightningDiT(
            input_size=arguments.resolution // 8,
            patch_size=arguments.patch_size,
            in_channels=2 * latent_channels,
            out_channels=latent_channels,
            hidden_size=arguments.dim,
            depth=arguments.depth,
            num_heads=arguments.num_heads,
            mlp_ratio=arguments.mlp_ratio,
            z_dims=arguments.enc_dim,
            encdim_ratio=arguments.encdim_ratio,
            auxiliary_time_cond=arguments.auxiliary_time_cond,
            use_qknorm=arguments.use_qknorm,
            use_swiglu=arguments.use_swiglu,
            use_rope=arguments.use_rope,
            use_rmsnorm=arguments.use_rmsnorm,
            wo_shift=arguments.wo_shift,
            num_fused_layers=len(arguments.layer_dinov2b_list),
        )
        model.load_state_dict(
            load_file(str(model_directory / "checkpoints" / "ema_model.safetensors")), strict=False
        )
        self.model = model.to("cuda").eval().requires_grad_(False)

    def _load_dinov2(self, encoder_type: str) -> nn.Module:
        cache_directory = self.weights_path / "torch_cache"
        torch.hub.set_dir(str(cache_directory))
        encoder = torch.hub.load(
            str(cache_directory / "facebookresearch_dinov2_main"),
            DINOV2_NAMES[encoder_type],
            source="local",
        )
        encoder.head = nn.Identity()
        return encoder.to("cuda").eval().requires_grad_(False)

    def _image_features(self, bicubic_pixels: torch.Tensor) -> list[torch.Tensor]:
        """
        DINOv2 tokens of the selected layers (the last one normalized), as `get_venc_features` of VOSR.

        args:
            bicubic_pixels: [B, 3, H, W], pixel_values: [0...1]
        returns:
            features: list of [B, N, D]
        """
        size = self.arguments.dinov2_size
        pixels = functional.interpolate(bicubic_pixels, size, mode="bicubic").clip(0, 1)
        pixels = vision_functional.normalize(pixels, IMAGENET_MEAN, IMAGENET_STD)
        tokens = self.image_encoder.prepare_tokens_with_masks(pixels, None)
        layer_tokens = []
        for block in self.image_encoder.blocks:
            tokens = block(tokens)
            layer_tokens.append(tokens[:, 1:])
        layer_tokens[-1] = self.image_encoder.norm(tokens)[:, 1:]
        return [layer_tokens[index] for index in self.arguments.layer_dinov2b_list]

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        pixels = self.decode(latents).clamp(0, 1)  # [B, 3, 512, 512]
        if self.input_size != pixels.shape[-1]:
            pixels = functional.interpolate(pixels, size=self.input_size, mode="area")
        bicubic = functional.interpolate(
            pixels, size=2 * latents.shape[-1] * 8, mode="bicubic", align_corners=False
        ).clamp(0, 1)  # [B, 3, 1024, 1024]
        low_quality = bicubic * 2.0 - 1.0
        latents_low_quality, latents_mean, latents_std = self.encode_latent(
            self.sr_vae, low_quality, self.arguments, "cuda", True
        )
        generator = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(
            latents_low_quality.shape, generator=generator, device="cuda"
        )
        batch_size = latents.shape[0]
        velocity = self.model.forward_flexible(
            torch.cat([latents_low_quality, noise], dim=1),
            torch.ones(batch_size, device="cuda"),
            torch.zeros(batch_size, device="cuda"),
            self._image_features(bicubic),
        )
        latents_high_quality = noise - velocity
        decoded = self.decode_latent(
            self.sr_vae,
            latents_high_quality,
            self.arguments,
            latents_mean,
            latents_std,
            self.light_decoder,
        )
        return wavelet_color_fix(decoded.float() * 0.5 + 0.5, bicubic)

    def get_settings(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "steps": 1,
            "dtype": "float32",
            "input": f"bicubic {1024 // self.input_size}x of the full VAE decode at {self.input_size}",
            "color_fix": "wavelet",
        }
