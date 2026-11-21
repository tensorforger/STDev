import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from stdev.comfy_nodes.wan_2_1.distilled_sampling import NOISE_SCALE_SPATIAL
from stdev.upscale_benchmark.baselines import Wan21UpscaleDecoderVAE
from stdev.upscale_benchmark.metrics import downsample_by_two
from stdev.upscale_benchmark.solution import LATENT_CHANNELS, OUTPUT_PIXEL_SIZE, UpscaleSolution
from stdev.vaes.tiny_wan_2_1_vae import TinyWan21VAE
from stdev.vaes.vae import VAE
from stdev.vaes.wan_2_1_vae import Wan21VAE
from stdev.wan_distill.student import WanStudentTransformer, upscale_single_step
from stdev.wan_pretrain.model import ModelConfig, SamplerConfig, WanUpscalerTransformer, upscale

MODEL_DTYPE = torch.bfloat16
OUTPUT_LATENT_SIZE = OUTPUT_PIXEL_SIZE // 8
METADATA_CONFIG_KEY = "model_config"


def find_latest_checkpoint(directory: Path, run_name: str) -> Path:
    """
    returns:
        EMA checkpoint `wan_2_1_<run_name>_ema_step<N>.safetensors` of the highest step N
    """
    pattern = re.compile(rf"^wan_2_1_{re.escape(run_name)}_ema_step(\d+)\.safetensors$")
    candidates = [
        (int(match.group(1)), path)
        for path in directory.iterdir()
        if (match := pattern.match(path.name))
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint of run {run_name} in {directory}")
    return max(candidates)[1]


def load_checkpoint_weights(path: Path) -> tuple[dict[str, torch.Tensor], ModelConfig]:
    """
    returns:
        weights: parameter name -> tensor on the GPU, config: architecture config stored in the metadata
    """
    with safe_open(str(path), framework="pt") as file:
        config = ModelConfig(**json.loads(file.metadata()[METADATA_CONFIG_KEY]))
    return load_file(str(path), device="cuda"), config


def decode_single_frame(vae: VAE, latents: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents: [B, 16, 1, H, W], normalized Wan 2.1 VAE latents
    returns:
        pixels: [B, 3, 8 * H, 8 * W], pixel_values: [0...1]
    """
    pixels = vae.decode(latents)
    return pixels[:, :, 0] if pixels.ndim == 5 else pixels


def make_null_hff(batch_size: int) -> torch.Tensor:
    """
    returns:
        hff: [B], NaN = null hff embedding, the model decides the sharpness by itself
    """
    return torch.full((batch_size,), float("nan"), device="cuda")


def select_hff(
    use_ground_truth_hff: bool, batch_size: int, hff: torch.Tensor | None
) -> torch.Tensor:
    """
    returns:
        hff: [B], the given ground truth hff if used, otherwise NaN (null hff)
    """
    if not use_ground_truth_hff:
        return make_null_hff(batch_size)
    if hff is None:
        raise ValueError("A ground truth hff variant needs hff")
    return hff


class OursBaseSolution(UpscaleSolution):
    """
    Pretrained residual flow transformer, multi step Euler sampling without guidance, null hff (or the ground truth
    hff), full VAE decoder.
    """

    def __init__(
        self,
        name: str,
        checkpoint_path: Path,
        vae_path: Path,
        steps: int = 20,
        shift: float = 4.0,
        noise_scale: float = 1.0,
        use_ground_truth_hff: bool = False,
    ) -> None:
        hff_description = "ground truth hff" if use_ground_truth_hff else "null hff"
        super().__init__(
            name, f"Pretrained base model, {steps} Euler steps, {hff_description}, full VAE decoder"
        )
        self.uses_hff = use_ground_truth_hff
        self.checkpoint_path = checkpoint_path
        self.vae_path = vae_path
        self.steps = steps
        self.shift = shift
        self.noise_scale = noise_scale

    def load(self) -> None:
        weights, config = load_checkpoint_weights(self.checkpoint_path)
        self.model = WanUpscalerTransformer(config)
        self.model.load_state_dict(weights)
        self.model = self.model.to("cuda", MODEL_DTYPE).eval().requires_grad_(False)
        self.vae = Wan21VAE(str(self.vae_path), device="cuda", dtype=MODEL_DTYPE, as_image=True)

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        sampler = SamplerConfig(
            steps=self.steps, shift=self.shift, noise_scale_spatial=self.noise_scale, seed=seed
        )
        output_size = (1, OUTPUT_LATENT_SIZE, OUTPUT_LATENT_SIZE)
        latents_high_resolution = upscale(
            self.model,
            latents,
            select_hff(self.uses_hff, latents.shape[0], hff),
            output_size,
            sampler,
            MODEL_DTYPE,
        )
        return decode_single_frame(self.vae, latents_high_resolution)

    def get_settings(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint_path.name,
            "steps": self.steps,
            "shift": self.shift,
            "noise_scale": self.noise_scale,
            "guidance": 1.0,
            "hff": "ground_truth" if self.uses_hff else "null",
        }


class OursDistilledSolution(UpscaleSolution):
    """
    Distilled single step student (guidance 1, null hff or the ground truth hff) followed by the full or the tiny
    Wan 2.1 VAE decoder, or by the spacepxl 2x upscaling decoder (4x in total) and a bilinear 2x downscale.
    """

    def __init__(
        self,
        name: str,
        checkpoint_path: Path,
        vae_path: Path,
        tiny_vae_path: Path | None,
        use_ground_truth_hff: bool = False,
        upscale_vae_path: Path | None = None,
    ) -> None:
        if upscale_vae_path is not None:
            decoder = "spacepxl 2x decoder (4x in total), bilinear downscale"
        else:
            decoder = "full VAE" if tiny_vae_path is None else "tiny VAE"
        hff_description = "ground truth hff" if use_ground_truth_hff else "null hff"
        super().__init__(name, f"Distilled single step model, {hff_description}, {decoder} decoder")
        self.uses_hff = use_ground_truth_hff
        self.checkpoint_path = checkpoint_path
        self.vae_path = vae_path
        self.tiny_vae_path = tiny_vae_path
        self.upscale_vae_path = upscale_vae_path

    def load(self) -> None:
        weights, config = load_checkpoint_weights(self.checkpoint_path)
        self.model = WanStudentTransformer(config)
        self.model.load_state_dict(weights)
        self.model = self.model.to("cuda", MODEL_DTYPE).eval().requires_grad_(False)
        if self.upscale_vae_path is not None:
            self.vae = Wan21UpscaleDecoderVAE(
                str(self.upscale_vae_path), device="cuda", dtype=MODEL_DTYPE, as_image=True
            )
        elif self.tiny_vae_path is None:
            self.vae = Wan21VAE(str(self.vae_path), device="cuda", dtype=MODEL_DTYPE, as_image=True)
        else:
            self.vae = TinyWan21VAE(str(self.tiny_vae_path), device="cuda", dtype=MODEL_DTYPE)

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = latents.shape[0]
        generator = torch.Generator(device="cuda").manual_seed(seed)
        noise = NOISE_SCALE_SPATIAL * torch.randn(
            (batch_size, LATENT_CHANNELS, 1, OUTPUT_LATENT_SIZE, OUTPUT_LATENT_SIZE),
            generator=generator,
            device="cuda",
        )
        latents_high_resolution = upscale_single_step(
            self.model,
            latents,
            select_hff(self.uses_hff, batch_size, hff),
            torch.ones(batch_size, device="cuda"),
            noise,
            MODEL_DTYPE,
        )
        pixels = decode_single_frame(self.vae, latents_high_resolution)
        return pixels if self.upscale_vae_path is None else downsample_by_two(pixels.clamp(0, 1))

    def get_settings(self) -> dict[str, object]:
        if self.upscale_vae_path is not None:
            decoder = "spacepxl_2x_downscaled_bilinear"
        else:
            decoder = "full" if self.tiny_vae_path is None else "tiny"
        return {
            "checkpoint": self.checkpoint_path.name,
            "steps": 1,
            "noise_scale": NOISE_SCALE_SPATIAL,
            "guidance": 1.0,
            "hff": "ground_truth" if self.uses_hff else "null",
            "decoder": decoder,
        }
