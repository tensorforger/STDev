"""
ComfyUI nodes of the FLUX.2 VAE latent upscaler (images, 2x).
"""

import torch

from stdev.comfy_nodes import common
from stdev.comfy_nodes.flux_2.sampling import UpscaleSettings, upscale_latents
from stdev.flux_pretrain.model import FluxUpscalerTransformer, ModelConfig

VAE_NAME = "flux_2"
ARCHITECTURE = "FluxUpscalerTransformer"
PATCHIFIED_CHANNELS = 128
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
CATEGORY = "STDev/FLUX.2"


class Flux2UpscalerHandle:
    def __init__(self, model: FluxUpscalerTransformer) -> None:
        self.model = model


class LoadFlux2Upscaler:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("FLUX2_UPSCALER",)
    RETURN_NAMES = ("upscaler",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model_name": (common.list_checkpoints(VAE_NAME, ARCHITECTURE),),
                "dtype": (list(DTYPES), {"default": "bfloat16"}),
            }
        }

    def load(self, model_name: str, dtype: str) -> tuple[Flux2UpscalerHandle]:
        weights, config = common.load_checkpoint(model_name)
        config["residual_std"] = tuple(config["residual_std"])
        model = FluxUpscalerTransformer(ModelConfig(**config))
        model.load_state_dict(weights)
        model = model.to(DTYPES[dtype]).eval().requires_grad_(False)
        return (Flux2UpscalerHandle(model),)


class UpscaleFlux2Image:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "upscaler": ("FLUX2_UPSCALER",),
                "latent": ("LATENT",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 1.0,
                        "max": 30.0,
                        "step": 0.1,
                        "tooltip": "1 = no guidance (single pass).",
                    },
                ),
                "hff": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "High frequency feature (Immerkaer, [0...1]) of the positive branch.",
                    },
                ),
                "negative_hff": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "High frequency feature of the negative branch. -1 = no hff conditioning (null embedding).",
                    },
                ),
                "shift": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "noise_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Initial noise temperature.",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**63 - 1}),
            }
        }

    def upscale(
        self,
        upscaler: Flux2UpscalerHandle,
        latent: dict,
        steps: int,
        cfg: float,
        hff: float,
        negative_hff: float,
        shift: float,
        noise_scale: float,
        seed: int,
    ) -> tuple[dict]:
        samples = latent["samples"]
        assert samples.ndim == 4 and samples.shape[1] == PATCHIFIED_CHANNELS, (
            f"expected FLUX.2 latents [B, 128, H, W], got {tuple(samples.shape)}"
        )
        settings = UpscaleSettings(
            steps=steps,
            cfg=cfg,
            hff=hff,
            negative_hff=None if negative_hff < 0 else negative_hff,
            shift=shift,
            noise_scale=noise_scale,
            seed=seed,
        )
        progress_bar = common.make_progress_bar(samples.shape[0] * steps)

        def on_step() -> None:
            common.check_interrupted()
            progress_bar.update(1)

        with common.on_device(upscaler.model) as device:
            upscaled = upscale_latents(upscaler.model, samples, settings, device, on_step)
        return ({**latent, "samples": upscaled},)


NODE_CLASS_MAPPINGS = {
    "STDevLoadFlux2Upscaler": LoadFlux2Upscaler,
    "STDevUpscaleFlux2Image": UpscaleFlux2Image,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "STDevLoadFlux2Upscaler": "Load FLUX.2 Upscaler",
    "STDevUpscaleFlux2Image": "Upscale FLUX.2 Image",
}
