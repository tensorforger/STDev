import importlib.util
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file

from stdev.flux_distill.student import FluxStudentTransformer, seeded_noise, upscale_single_step
from stdev.flux_pretrain.model import FluxUpscalerTransformer, ModelConfig, SamplerConfig, upscale
from stdev.flux_upscale_benchmark.solution import LATENT_CHANNELS, FluxUpscaleSolution
from stdev.vaes.flux_2_vae import Flux2VAE
from stdev.vaes.tiny_flux_2_vae import TinyFlux2VAE
from stdev.vaes.vae import VAE

MODEL_DTYPE = torch.bfloat16
METADATA_CONFIG_KEY = "model_config"
FLOW_UPSCALER_TIMESTEP = 1000.0


def find_latest_checkpoint(directory: Path, run_name: str, weights: str = "raw") -> Path:
    """
    returns:
        checkpoint `flux_2_<run_name>_<weights>_step<N>.safetensors` of the highest step N
    """
    pattern = re.compile(rf"^flux_2_{re.escape(run_name)}_{weights}_step(\d+)\.safetensors$")
    candidates = [
        (int(match.group(1)), path)
        for path in directory.iterdir()
        if (match := pattern.match(path.name))
    ]
    if not candidates:
        raise FileNotFoundError(f"No {weights} checkpoint of run {run_name} in {directory}")
    return max(candidates)[1]


def load_checkpoint_weights(path: Path) -> tuple[dict[str, torch.Tensor], ModelConfig]:
    """
    returns:
        weights: parameter name -> tensor on the GPU, config: architecture config stored in the metadata
    """
    with safe_open(str(path), framework="pt") as file:
        values = json.loads(file.metadata()[METADATA_CONFIG_KEY])
    config = ModelConfig(**{**values, "residual_std": tuple(values["residual_std"])})
    return load_file(str(path), device="cuda"), config


def load_decoder(vae_path: Path, tiny_vae_path: Path | None) -> VAE:
    if tiny_vae_path is None:
        return Flux2VAE(str(vae_path), device="cuda", dtype=MODEL_DTYPE)
    return TinyFlux2VAE(str(tiny_vae_path), device="cuda", dtype=MODEL_DTYPE)


def decoder_name(tiny_vae_path: Path | None) -> str:
    return "full" if tiny_vae_path is None else "tiny (TAEF2)"


def null_hff(batch_size: int) -> torch.Tensor:
    """
    returns:
        hff: [B], NaN = null hff embedding, the model decides the sharpness by itself
    """
    return torch.full((batch_size,), float("nan"), device="cuda")


class OursFluxBaseSolution(FluxUpscaleSolution):
    """
    Pretrained residual flow transformer (raw weights), multi step Euler sampling without guidance, null hff, full
    FLUX.2 VAE decoder.
    """

    def __init__(
        self,
        name: str,
        checkpoint_path: Path,
        vae_path: Path,
        steps: int = 30,
        shift: float = 2.0,
        noise_scale: float = 1.0,
    ) -> None:
        super().__init__(
            name, f"Pretrained base model, {steps} Euler steps, null hff, full VAE decoder"
        )
        self.checkpoint_path = checkpoint_path
        self.vae_path = vae_path
        self.steps = steps
        self.shift = shift
        self.noise_scale = noise_scale

    def load(self) -> None:
        weights, config = load_checkpoint_weights(self.checkpoint_path)
        self.model = FluxUpscalerTransformer(config)
        self.model.load_state_dict(weights)
        self.model = self.model.to("cuda", MODEL_DTYPE).eval().requires_grad_(False)
        self.vae = load_decoder(self.vae_path, None)

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        sampler = SamplerConfig(
            steps=self.steps, shift=self.shift, noise_scale=self.noise_scale, seed=seed
        )
        latents_high_resolution = upscale(
            self.model, latents, null_hff(latents.shape[0]), sampler, MODEL_DTYPE
        )
        return self.vae.decode(latents_high_resolution.unsqueeze(2))

    def get_settings(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint_path.name,
            "steps": self.steps,
            "shift": self.shift,
            "noise_scale": self.noise_scale,
            "guidance": 1.0,
            "hff": "null",
            "decoder": "full",
        }


class OursFluxDistilledSolution(FluxUpscaleSolution):
    """
    Distilled single step student (guidance 1, null hff) followed by the full or the tiny FLUX.2 VAE decoder.
    """

    def __init__(
        self, name: str, checkpoint_path: Path, vae_path: Path, tiny_vae_path: Path | None
    ) -> None:
        super().__init__(
            name,
            f"Distilled single step model, null hff, {decoder_name(tiny_vae_path)} VAE decoder",
        )
        self.checkpoint_path = checkpoint_path
        self.vae_path = vae_path
        self.tiny_vae_path = tiny_vae_path

    def load(self) -> None:
        weights, config = load_checkpoint_weights(self.checkpoint_path)
        self.model = FluxStudentTransformer(config)
        self.model.load_state_dict(weights)
        self.model = self.model.to("cuda", MODEL_DTYPE).eval().requires_grad_(False)
        self.vae = load_decoder(self.vae_path, self.tiny_vae_path)

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, _, height, width = latents.shape
        noise = seeded_noise(
            torch.full((batch_size,), seed, dtype=torch.int64),
            (LATENT_CHANNELS, 2 * height, 2 * width),
            latents.device,
        )
        latents_high_resolution = upscale_single_step(
            self.model,
            latents,
            null_hff(batch_size),
            torch.ones(batch_size, device="cuda"),
            noise,
            MODEL_DTYPE,
        )
        return self.vae.decode(latents_high_resolution.unsqueeze(2))

    def get_settings(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint_path.name,
            "steps": 1,
            "guidance": 1.0,
            "hff": "null",
            "decoder": decoder_name(self.tiny_vae_path),
        }


def load_flow_upscaler_class(source_path: Path) -> type[nn.Module]:
    """
    returns:
        `UpscalerUNet` of the released FlowUpscaler (`upscaler_unet.py` of the HF repository)
    """
    specification = importlib.util.spec_from_file_location("flow_upscaler_unet", source_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.UpscalerUNet


class FlowUpscalerSolution(FluxUpscaleSolution):
    """
    Released FlowUpscaler (https://huggingface.co/TensorForger/FlowUpscaler, 59M attention free UNet, the ancestor
    of this project) as in its ComfyUI node: one Euler step from pure noise at timestep 1000 on normalized FLUX.2
    latents (sample = noise - prediction), then the full or the tiny FLUX.2 VAE decoder.
    """

    def __init__(
        self, name: str, repository_path: Path, vae_path: Path, tiny_vae_path: Path | None
    ) -> None:
        super().__init__(
            name, f"FlowUpscaler (released), 1 step, {decoder_name(tiny_vae_path)} VAE decoder"
        )
        self.repository_path = repository_path
        self.vae_path = vae_path
        self.tiny_vae_path = tiny_vae_path

    def load(self) -> None:
        model = load_flow_upscaler_class(self.repository_path / "upscaler_unet.py")()
        model.load_state_dict(load_file(str(self.repository_path / "flow_upscaler.safetensors")))
        self.model = model.to("cuda", MODEL_DTYPE).eval().requires_grad_(False)
        self.vae = load_decoder(self.vae_path, self.tiny_vae_path)

    def upscale(
        self, latents: torch.Tensor, seed: int, hff: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        generator = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(
            (batch_size, channels, 2 * height, 2 * width), generator=generator, device="cuda"
        )
        timestep = torch.full((batch_size,), FLOW_UPSCALER_TIMESTEP, device="cuda", dtype=MODEL_DTYPE)
        prediction = self.model(
            sample=noise.to(MODEL_DTYPE), timestep=timestep, latents_small=latents.to(MODEL_DTYPE)
        )
        latents_high_resolution = noise - prediction.float()
        return self.vae.decode(latents_high_resolution.unsqueeze(2))

    def get_settings(self) -> dict[str, object]:
        return {
            "weights": "flow_upscaler.safetensors",
            "steps": 1,
            "timestep": FLOW_UPSCALER_TIMESTEP,
            "dtype": "bfloat16",
            "decoder": decoder_name(self.tiny_vae_path),
        }
