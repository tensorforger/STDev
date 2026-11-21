from pathlib import Path
from typing import Callable

from stdev.upscale_benchmark.baselines import (
    DecodeBicubicSolution,
    DecodeSuperResolutionSolution,
    LatentBicubicSolution,
    SesquiLSRSolution,
    UpscaleVAESolution,
    WanNNLatentUpscalerSolution,
)
from stdev.upscale_benchmark.solution import UpscaleSolution
from stdev.upscale_benchmark.solutions_ours import (
    OursBaseSolution,
    OursDistilledSolution,
    find_latest_checkpoint,
)


def build_solution_factories(models_directory: Path) -> dict[str, Callable[[], UpscaleSolution]]:
    """
    returns:
        solution name -> factory, in benchmark order; factories only read paths, weights are loaded by `load`
    """
    vae_path = models_directory / "Wan2.1-T2V-1.3B-Diffusers" / "vae"
    tiny_vae_path = models_directory / "taew2_1.safetensors"
    upscale_vae_path = (
        models_directory
        / "Wan2.1-VAE-upscale2x"
        / "diffusers"
        / "Wan2.1_VAE_upscale2x_imageonly_real_v1"
    )
    checkpoints_directory = models_directory / "latent_upscale_models"
    base_checkpoint = find_latest_checkpoint(checkpoints_directory, "wan_pretrain_v1")
    large_checkpoint = find_latest_checkpoint(checkpoints_directory, "wan_distill_v1")
    small_20m_checkpoint = find_latest_checkpoint(checkpoints_directory, "wan_small_s20m")
    small_2m_checkpoint = find_latest_checkpoint(checkpoints_directory, "wan_small_s2m")

    factories: dict[str, Callable[[], UpscaleSolution]] = {}
    for name, build_solution in {
        "ours_base_20_steps": lambda name, use_hff: OursBaseSolution(
            name, base_checkpoint, vae_path, use_ground_truth_hff=use_hff
        ),
        "ours_distilled_1_step_200m": lambda name, use_hff: OursDistilledSolution(
            name, large_checkpoint, vae_path, None, use_hff
        ),
        "ours_distilled_1_step_20m": lambda name, use_hff: OursDistilledSolution(
            name, small_20m_checkpoint, vae_path, None, use_hff
        ),
        "ours_distilled_1_step_20m_tiny_vae": lambda name, use_hff: OursDistilledSolution(
            name, small_20m_checkpoint, vae_path, tiny_vae_path, use_hff
        ),
        "ours_distilled_1_step_2m_tiny_vae": lambda name, use_hff: OursDistilledSolution(
            name, small_2m_checkpoint, vae_path, tiny_vae_path, use_hff
        ),
    }.items():
        factories[name] = lambda name=name, build=build_solution: build(name, False)
        hff_name = f"{name}_gt_hff"
        factories[hff_name] = lambda name=hff_name, build=build_solution: build(name, True)

    factories |= {
        "ours_distilled_1_step_200m_spacepxl_decoder_downscaled": lambda: OursDistilledSolution(
            "ours_distilled_1_step_200m_spacepxl_decoder_downscaled",
            large_checkpoint,
            vae_path,
            None,
            upscale_vae_path=upscale_vae_path,
        ),
        "decode_bicubic": lambda: DecodeBicubicSolution("decode_bicubic", vae_path),
        "spacepxl_wan21_vae_upscale2x": lambda: UpscaleVAESolution(
            "spacepxl_wan21_vae_upscale2x",
            upscale_vae_path,
        ),
        "latent_bicubic_decode": lambda: LatentBicubicSolution("latent_bicubic_decode", vae_path),
        "wan_nn_latent_upscaler": lambda: WanNNLatentUpscalerSolution(
            "wan_nn_latent_upscaler",
            vae_path,
            models_directory / "wan21_latent_upscale_2x.safetensors",
        ),
        "sesquilsr": lambda: SesquiLSRSolution(
            "sesquilsr", vae_path, models_directory / "SesquiLSR"
        ),
        "decode_span": lambda: DecodeSuperResolutionSolution(
            "decode_span",
            "Full VAE decode, then SPAN 2x (2xNomosUni_span_multijpg)",
            vae_path,
            models_directory / "2xNomosUni_span_multijpg" / "2xNomosUni_span_multijpg.safetensors",
        ),
        "decode_realesrgan": lambda: DecodeSuperResolutionSolution(
            "decode_realesrgan",
            "Full VAE decode, then Real-ESRGAN x2plus",
            vae_path,
            models_directory / "RealESRGAN_x2plus.pth",
        ),
    }
    return factories
