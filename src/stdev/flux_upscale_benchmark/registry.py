from pathlib import Path
from typing import Callable

from stdev.flux_upscale_benchmark.baselines import (
    DecodeBicubicSolution,
    DecodeSuperResolutionSolution,
    LatentBicubicSolution,
    SesquiLSRSolution,
)
from stdev.flux_upscale_benchmark.solution import FluxUpscaleSolution
from stdev.flux_upscale_benchmark.solutions_ours import (
    FlowUpscalerSolution,
    OursFluxBaseSolution,
    OursFluxDistilledSolution,
    find_latest_checkpoint,
)
from stdev.flux_upscale_benchmark.vosr import VOSRSolution


def build_solution_factories(
    models_directory: Path,
) -> dict[str, Callable[[], FluxUpscaleSolution]]:
    """
    returns:
        solution name -> factory, in benchmark order; factories only read paths, weights are loaded by `load`
    """
    vae_path = models_directory / "FLUX.2-klein-4B" / "vae"
    tiny_vae_path = models_directory / "taef2" / "taef2.safetensors"
    checkpoints_directory = models_directory / "latent_upscale_models"
    flow_upscaler_path = models_directory / "FlowUpscaler"
    vosr_repository = models_directory / "VOSR"
    vosr_weights = models_directory / "VOSR_weights"

    def checkpoint(run_name: str) -> Path:
        return find_latest_checkpoint(checkpoints_directory, run_name, "raw")

    return {
        "ours_base_30_steps_200m": lambda: OursFluxBaseSolution(
            "ours_base_30_steps_200m", checkpoint("flux_pretrain_v1"), vae_path
        ),
        "ours_distilled_1_step_200m": lambda: OursFluxDistilledSolution(
            "ours_distilled_1_step_200m", checkpoint("flux_distill_v1"), vae_path, None
        ),
        "ours_distilled_1_step_200m_tiny_vae": lambda: OursFluxDistilledSolution(
            "ours_distilled_1_step_200m_tiny_vae",
            checkpoint("flux_distill_v1"),
            vae_path,
            tiny_vae_path,
        ),
        "ours_distilled_1_step_20m": lambda: OursFluxDistilledSolution(
            "ours_distilled_1_step_20m", checkpoint("flux_small_s20m"), vae_path, None
        ),
        "ours_distilled_1_step_20m_tiny_vae": lambda: OursFluxDistilledSolution(
            "ours_distilled_1_step_20m_tiny_vae",
            checkpoint("flux_small_s20m"),
            vae_path,
            tiny_vae_path,
        ),
        "flow_upscaler": lambda: FlowUpscalerSolution(
            "flow_upscaler", flow_upscaler_path, vae_path, None
        ),
        "flow_upscaler_tiny_vae": lambda: FlowUpscalerSolution(
            "flow_upscaler_tiny_vae", flow_upscaler_path, vae_path, tiny_vae_path
        ),
        "decode_bicubic": lambda: DecodeBicubicSolution("decode_bicubic", vae_path),
        "latent_bicubic_decode": lambda: LatentBicubicSolution("latent_bicubic_decode", vae_path),
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
        "decode_vosr_0_5b": lambda: VOSRSolution(
            "decode_vosr_0_5b", vae_path, vosr_repository, vosr_weights, "VOSR_0.5B_os"
        ),
        "decode_vosr_1_4b": lambda: VOSRSolution(
            "decode_vosr_1_4b", vae_path, vosr_repository, vosr_weights, "VOSR_1.4B_os"
        ),
        "decode_256_vosr_0_5b_x4": lambda: VOSRSolution(
            "decode_256_vosr_0_5b_x4", vae_path, vosr_repository, vosr_weights, "VOSR_0.5B_os", 256
        ),
        "decode_256_vosr_1_4b_x4": lambda: VOSRSolution(
            "decode_256_vosr_1_4b_x4", vae_path, vosr_repository, vosr_weights, "VOSR_1.4B_os", 256
        ),
    }
