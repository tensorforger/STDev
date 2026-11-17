"""
ComfyUI nodes of the Wan 2.1 VAE latent upscaler.
"""

import torch

from stdev.comfy_nodes import common
from stdev.comfy_nodes.wan_2_1.sampling import UpscaleSettings, upscale_latents
from stdev.wan_pretrain.model import ModelConfig, WanUpscalerTransformer

VAE_NAME = "wan_2_1"
LATENT_CHANNELS = 16
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
CATEGORY = "STDev/Wan 2.1"
NEGATIVE_HFF_TOOLTIP = (
    "High frequency feature of the negative branch. -1 = no hff conditioning (null embedding)."
)


class Wan21UpscalerHandle:
    def __init__(self, model: WanUpscalerTransformer) -> None:
        self.model = model


class LoadWan21Upscaler:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("WAN21_UPSCALER",)
    RETURN_NAMES = ("upscaler",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model_name": (common.list_checkpoints(VAE_NAME),),
                "dtype": (list(DTYPES), {"default": "bfloat16"}),
            }
        }

    def load(self, model_name: str, dtype: str) -> tuple[Wan21UpscalerHandle]:
        weights, config = common.load_checkpoint(model_name)
        model = WanUpscalerTransformer(ModelConfig(**config))
        model.load_state_dict(weights)
        model = model.to(DTYPES[dtype]).eval().requires_grad_(False)
        return (Wan21UpscalerHandle(model),)


def run_upscale(
    upscaler: Wan21UpscalerHandle, latent: dict, settings: UpscaleSettings
) -> tuple[dict]:
    samples = latent["samples"]
    assert samples.ndim == 5 and samples.shape[1] == LATENT_CHANNELS, (
        f"expected Wan 2.1 latents [B, 16, T, H, W], got {tuple(samples.shape)}"
    )
    steps_total = samples.shape[0] * settings.steps
    progress_bar = common.make_progress_bar(steps_total)

    def on_step() -> None:
        common.check_interrupted()
        progress_bar.update(1)

    with common.on_device(upscaler.model) as device:
        upscaled = upscale_latents(upscaler.model, samples, settings, device, on_step)
    return ({**latent, "samples": upscaled},)


def make_input_types(video: bool) -> dict:
    required = {
        "upscaler": ("WAN21_UPSCALER",),
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
            "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
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
                    "tooltip": NEGATIVE_HFF_TOOLTIP,
                },
            ),
            "shift": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0, "step": 0.1}),
            "noise_scale": (
                "FLOAT",
                {
                    "default": 0.8 if video else 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Initial noise temperature.",
                },
            ),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**63 - 1}),
        }
    )
    return {"required": required}


def make_settings(
    spatial: bool,
    temporal: bool,
    steps: int,
    cfg: float,
    hff: float,
    negative_hff: float,
    shift: float,
    noise_scale: float,
    seed: int,
) -> UpscaleSettings:
    assert spatial or temporal, "enable spatial and / or temporal upscaling"
    return UpscaleSettings(
        spatial=spatial,
        temporal=temporal,
        steps=steps,
        cfg=cfg,
        hff=hff,
        negative_hff=None if negative_hff < 0 else negative_hff,
        shift=shift,
        noise_scale=noise_scale,
        seed=seed,
    )


class UpscaleWan21Image:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return make_input_types(video=False)

    def upscale(
        self,
        upscaler: Wan21UpscalerHandle,
        latent: dict,
        steps: int,
        cfg: float,
        hff: float,
        negative_hff: float,
        shift: float,
        noise_scale: float,
        seed: int,
    ) -> tuple[dict]:
        assert latent["samples"].shape[2] == 1, (
            "image node expects single frame latents, use the video node"
        )
        settings = make_settings(
            True, False, steps, cfg, hff, negative_hff, shift, noise_scale, seed
        )
        return run_upscale(upscaler, latent, settings)


class UpscaleWan21Video:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return make_input_types(video=True)

    def upscale(
        self,
        upscaler: Wan21UpscalerHandle,
        latent: dict,
        spatial: bool,
        temporal: bool,
        steps: int,
        cfg: float,
        hff: float,
        negative_hff: float,
        shift: float,
        noise_scale: float,
        seed: int,
    ) -> tuple[dict]:
        settings = make_settings(
            spatial, temporal, steps, cfg, hff, negative_hff, shift, noise_scale, seed
        )
        return run_upscale(upscaler, latent, settings)


NODE_CLASS_MAPPINGS = {
    "STDevLoadWan21Upscaler": LoadWan21Upscaler,
    "STDevUpscaleWan21Image": UpscaleWan21Image,
    "STDevUpscaleWan21Video": UpscaleWan21Video,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "STDevLoadWan21Upscaler": "Load Wan 2.1 Upscaler",
    "STDevUpscaleWan21Image": "Upscale Wan 2.1 Image",
    "STDevUpscaleWan21Video": "Upscale Wan 2.1 Video",
}
