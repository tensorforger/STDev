"""
ComfyUI nodes of the single step distilled Wan 2.1 latent upscalers (large student and small students): the
teacher architecture plus a guidance condition, always one forward pass.
"""

import torch

from stdev.comfy_nodes import common
from stdev.comfy_nodes.wan_2_1.distilled_sampling import (
    DistilledSettings,
    upscale_latents_single_step,
)
from stdev.comfy_nodes.wan_2_1.nodes import CATEGORY, DTYPES, LATENT_CHANNELS, VAE_NAME
from stdev.wan_distill.student import WanStudentTransformer
from stdev.wan_pretrain.model import ModelConfig

ARCHITECTURE = "WanStudentTransformer"


class Wan21DistilledUpscalerHandle:
    def __init__(self, model: WanStudentTransformer) -> None:
        self.model = model


class LoadWan21DistilledUpscaler:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("WAN21_DISTILLED_UPSCALER",)
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

    def load(self, model_name: str, dtype: str) -> tuple[Wan21DistilledUpscalerHandle]:
        weights, config = common.load_checkpoint(model_name)
        model = WanStudentTransformer(ModelConfig(**config))
        model.load_state_dict(weights)
        model = model.to(DTYPES[dtype]).eval().requires_grad_(False)
        return (Wan21DistilledUpscalerHandle(model),)


def run_upscale(
    upscaler: Wan21DistilledUpscalerHandle, latent: dict, settings: DistilledSettings
) -> tuple[dict]:
    samples = latent["samples"]
    assert samples.ndim == 5 and samples.shape[1] == LATENT_CHANNELS, (
        f"expected Wan 2.1 latents [B, 16, T, H, W], got {tuple(samples.shape)}"
    )
    progress_bar = common.make_progress_bar(samples.shape[0])

    def on_sample() -> None:
        common.check_interrupted()
        progress_bar.update(1)

    with common.on_device(upscaler.model) as device:
        upscaled = upscale_latents_single_step(upscaler.model, samples, settings, device, on_sample)
    return ({**latent, "samples": upscaled},)


def make_input_types(video: bool) -> dict:
    required = {
        "upscaler": ("WAN21_DISTILLED_UPSCALER",),
        "latent": ("LATENT",),
    }
    if video:
        required["spatial"] = ("BOOLEAN", {"default": True, "tooltip": "2x height and width."})
        required["temporal"] = (
            "BOOLEAN",
            {"default": True, "tooltip": "2x frames: T -> 1 + 2 * (T - 1) latent frames."},
        )
    required.update(
        {
            "guidance": (
                "FLOAT",
                {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Distilled classifier free guidance scale, trained on 1-3. 1 = no guidance.",
                },
            ),
            "hff": (
                "FLOAT",
                {
                    "default": 0.9,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "High frequency feature (Immerkaer, [0...1]). -1 = no hff conditioning (null embedding).",
                },
            ),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**63 - 1}),
        }
    )
    return {"required": required}


def make_settings(
    spatial: bool, temporal: bool, guidance: float, hff: float, seed: int
) -> DistilledSettings:
    assert spatial or temporal, "enable spatial and / or temporal upscaling"
    return DistilledSettings(
        spatial=spatial,
        temporal=temporal,
        guidance=guidance,
        hff=None if hff < 0 else hff,
        seed=seed,
    )


class UpscaleWan21DistilledImage:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return make_input_types(video=False)

    def upscale(
        self,
        upscaler: Wan21DistilledUpscalerHandle,
        latent: dict,
        guidance: float,
        hff: float,
        seed: int,
    ) -> tuple[dict]:
        assert latent["samples"].shape[2] == 1, (
            "image node expects single frame latents, use the video node"
        )
        return run_upscale(upscaler, latent, make_settings(True, False, guidance, hff, seed))


class UpscaleWan21DistilledVideo:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return make_input_types(video=True)

    def upscale(
        self,
        upscaler: Wan21DistilledUpscalerHandle,
        latent: dict,
        spatial: bool,
        temporal: bool,
        guidance: float,
        hff: float,
        seed: int,
    ) -> tuple[dict]:
        settings = make_settings(spatial, temporal, guidance, hff, seed)
        return run_upscale(upscaler, latent, settings)


NODE_CLASS_MAPPINGS = {
    "STDevLoadWan21DistilledUpscaler": LoadWan21DistilledUpscaler,
    "STDevUpscaleWan21DistilledImage": UpscaleWan21DistilledImage,
    "STDevUpscaleWan21DistilledVideo": UpscaleWan21DistilledVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "STDevLoadWan21DistilledUpscaler": "Load Wan 2.1 Distilled Upscaler",
    "STDevUpscaleWan21DistilledImage": "Upscale Wan 2.1 Image (1 step)",
    "STDevUpscaleWan21DistilledVideo": "Upscale Wan 2.1 Video (1 step)",
}
