"""
Hildegard (FLUX.2 klein 9B + Hildegard refiner LoRA, https://huggingface.co/42lux/hildegard) on the FLUX.2 upscale
benchmark, run in-process in ComfyUI with the nodes of its example workflow. Runs with the ComfyUI venv (Hildegard
only exists as ComfyUI nodes), writes predictions and efficiency like `stdev.upscale_benchmark.runner.run_inference`;
quality is evaluated afterwards in the project venv by `scripts/run_flux_upscale_benchmark.py --external hildegard`.

Pipeline per image (example workflow `42lux-hildegard-refiner.json`, one 1024 tile for 512 -> 1024):
    normalized FLUX.2 latent [32, 64, 64] -> 2x2 patchify (ComfyUI FLUX.2 latent format) -> full VAE decode (512)
    -> Hildegard Plan (tile 1024, min scale 2.0, lanczos) -> references: tile (weight 0.9), 3x3 position map
    (cell 512, weight 0.8), global thumbnail (max 768, weight 0.8), all VAE encoded -> klein 9B fp8 + LoRA 1.0,
    empty latent, Flux2Scheduler 4 steps, euler, BasicGuider (no CFG), constant baseline prompt of the README
    -> full VAE decode (1024).
The prompt is the same for every image: it is encoded once at load (text encoder time and memory are not counted).

Prerequisites (ComfyUI at COMFYUI_ROOT): custom nodes ComfyUI-42lux-Hildegard-Refiner and ComfyUI-Flux2Klein-Enhancer;
models diffusion_models/flux-2-klein-9b-fp8.safetensors (black-forest-labs/FLUX.2-klein-9b-fp8, gated; fallback:
flux-2-klein-9b.safetensors of black-forest-labs/FLUX.2-klein-9B cast to fp8_e4m3fn at load),
loras/42lux_hildegard-refiner.safetensors, text_encoders/qwen_3_8b.safetensors (Comfy-Org/flux2-klein-9B),
vae/flux2-vae.safetensors.

Run:
    ~/comfyui/.venv/bin/python scripts/run_hildegard_benchmark.py [--max-samples 8]
    .venv/bin/python scripts/run_flux_upscale_benchmark.py --external hildegard [--max-samples 8]
"""

import argparse
import asyncio
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

COMFYUI_ROOT = Path.home() / "comfyui"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "axis_v1_1k_flux2_2x"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "raw_results" / "flux_upscale_benchmark_axis_v1_1k"
SMOKE_RESULTS_DIRECTORY = REPOSITORY_ROOT / "dev" / "flux_upscale_benchmark_smoke"
SOLUTION_NAME = "hildegard"

UNET_NAME = "flux-2-klein-9b-fp8.safetensors"
FALLBACK_UNET_NAME = "flux-2-klein-9b.safetensors"
FALLBACK_WEIGHT_DTYPE = "fp8_e4m3fn"
TEXT_ENCODER_NAME = "qwen_3_8b.safetensors"
VAE_NAME = "flux2-vae.safetensors"
LORA_NAME = "42lux_hildegard-refiner.safetensors"
LORA_STRENGTH = 1.0
TILE_SIZE = 1024
MIN_SCALE_FACTOR = 2.0
SCALING_METHOD = "lanczos"
CELL_SIZE = 512
GLOBAL_MAX_SIZE = 768
REFERENCE_WEIGHTS = (0.9, 0.8, 0.8)
STEPS = 4
SAMPLER_NAME = "euler"
BASELINE_PROMPT = (
    "RFNTILE. refine and add detail to this upscaled tile. Restore the image quality and resolve it to a sharp, "
    "high-resolution result. Remove compression artifacts, banding, and noise, and clarify soft or blurred areas into "
    "crisp, clean edges and definition. Enrich existing textures and surfaces with fine, intricate, physically "
    "accurate detail, matching the existing grain, focus, and material properties of each surface. Keep in-focus "
    "areas crisp and sharp, keep softly blurred areas soft, and leave flat or evenly-toned areas clean and smooth. "
    "Recover detail only where detail is already present, and add no new objects, elements, or content — refine only "
    "what is already in the tile. Preserve the original lighting, colour, contrast, and composition exactly as shown. "
    "Produce a clean, photorealistic result faithful to the source."
)

SEED = 0
NUM_WARMUP_SAMPLES = 3
BYTES_PER_MEGABYTE = 2**20
LATENT_CHANNELS = 32
INPUT_LATENT_SIZE = 64
OUTPUT_PIXEL_SIZE = 1024


def load_module(path: Path, name: str) -> object:
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents: [B, 32, H, W], normalized FLUX.2 latents (unpatchified)
    returns:
        latents: [B, 128, H / 2, W / 2], ComfyUI FLUX.2 latent format
    """
    batch_size, channels, height, width = latents.shape
    latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
    return latents.permute(0, 1, 3, 5, 2, 4).reshape(batch_size, channels * 4, height // 2, width // 2)


class HildegardSolution:
    def __init__(self, reference_weights: tuple[float, float, float] = REFERENCE_WEIGHTS) -> None:
        self.reference_weights = reference_weights
        sys.path.insert(0, str(COMFYUI_ROOT))
        import nodes

        self.nodes = nodes
        asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
        self.refiner = load_module(
            COMFYUI_ROOT / "custom_nodes" / "ComfyUI-42lux-Hildegard-Refiner" / "hildegard_refiner.py",
            "hildegard_refiner",
        )
        self.enhancer = load_module(
            COMFYUI_ROOT / "custom_nodes" / "ComfyUI-Flux2Klein-Enhancer" / "flux2_klein_ref_controller.py",
            "flux2_klein_ref_controller",
        )

    def load(self) -> None:
        from comfy_extras.nodes_custom_sampler import KSamplerSelect
        from comfy_extras.nodes_flux import Flux2Scheduler

        mappings = self.nodes.NODE_CLASS_MAPPINGS
        clip = mappings["CLIPLoader"]().load_clip(TEXT_ENCODER_NAME, "flux2", "default")[0]
        self.conditioning = mappings["CLIPTextEncode"]().encode(clip, BASELINE_PROMPT)[0]
        del clip
        gc.collect()
        import comfy.model_management

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        self.vae = mappings["VAELoader"]().load_vae(VAE_NAME)[0]
        if (COMFYUI_ROOT / "models" / "diffusion_models" / UNET_NAME).exists():
            self.unet = (UNET_NAME, "default")
        else:
            self.unet = (FALLBACK_UNET_NAME, FALLBACK_WEIGHT_DTYPE)
        model = mappings["UNETLoader"]().load_unet(*self.unet)[0]
        model = mappings["LoraLoaderModelOnly"]().load_lora_model_only(model, LORA_NAME, LORA_STRENGTH)[0]
        for reference_index, weight in enumerate(self.reference_weights):
            model = self.enhancer.Flux2KleinRefLatentWeight().execute(model, reference_index, weight)[0]
        self.model = model
        self.sampler = KSamplerSelect.execute(SAMPLER_NAME)[0]
        self.sigmas = Flux2Scheduler.execute(STEPS, TILE_SIZE, TILE_SIZE)[0]

    def upscale(self, latents: torch.Tensor, seed: int) -> torch.Tensor:
        """
        args:
            latents: [1, 32, 64, 64], normalized FLUX.2 latents
        returns:
            pixels: [1, 3, 1024, 1024], pixel_values: [0...1]
        """
        from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
        from comfy_extras.nodes_edit_model import ReferenceLatent

        refiner = self.refiner
        image = self.vae.decode(patchify_latents(latents)).clamp(0, 1)  # [1, 512, 512, 3]
        image = image.reshape(-1, *image.shape[-3:])[:1]
        upscaled, plan, _ = refiner._hildegard_compute_upscale(
            image=image.cpu(),
            tile_width=TILE_SIZE,
            tile_height=TILE_SIZE,
            overlap=refiner.OVERLAP_DICT["1/4 Tile"],
            min_scale_factor=MIN_SCALE_FACTOR,
            tile_order_v=refiner.TILE_ORDER_DICT["spiral"],
            scaling_method=SCALING_METHOD,
        )
        if plan["grid_x"] * plan["grid_y"] != 1 or upscaled.shape[1:3] != (TILE_SIZE, TILE_SIZE):
            raise ValueError(f"Expected one {TILE_SIZE} tile, got plan {plan}")
        source = refiner._tensor_to_pil(upscaled[0])
        tile_latent = self.vae.encode(upscaled)
        position_latent = self.vae.encode(
            refiner._pil_to_tensor(refiner._build_position_map(source, 0, 0, TILE_SIZE, TILE_SIZE, CELL_SIZE))
        )
        global_latent = self.vae.encode(refiner._pil_to_tensor(refiner._build_global_thumb(source, GLOBAL_MAX_SIZE)))

        conditioning = self.conditioning
        for reference in (tile_latent, position_latent, global_latent):
            conditioning = ReferenceLatent.execute(conditioning, {"samples": reference})[0]
        guider = BasicGuider.execute(self.model, conditioning)[0]
        noise = RandomNoise.execute(seed)[0]
        empty = self.nodes.NODE_CLASS_MAPPINGS["EmptyLatentImage"]().generate(TILE_SIZE, TILE_SIZE, 1)[0]
        sampled = SamplerCustomAdvanced.execute(noise, guider, self.sampler, self.sigmas, empty)[0]
        pixels = self.vae.decode(sampled["samples"])
        pixels = pixels.reshape(-1, *pixels.shape[-3:])[:1]
        return pixels.movedim(-1, 1).float()

    def get_settings(self) -> dict[str, object]:
        return {
            "model": self.unet[0],
            "weight_dtype": self.unet[1],
            "lora": LORA_NAME,
            "lora_strength": LORA_STRENGTH,
            "text_encoder": TEXT_ENCODER_NAME,
            "prompt": "README baseline prompt (constant, encoded once)",
            "tile": TILE_SIZE,
            "pre_upscale": SCALING_METHOD,
            "cell_size": CELL_SIZE,
            "global_max_size": GLOBAL_MAX_SIZE,
            "reference_weights": list(self.reference_weights),
            "steps": STEPS,
            "sampler": SAMPLER_NAME,
            "guider": "basic (no CFG)",
            "warmup_samples": NUM_WARMUP_SAMPLES,
        }


def save_prediction(pixels: torch.Tensor, path: Path) -> None:
    """
    args:
        pixels: [1, 3, H, W], channel_order: RGB, pixel_values: [0...1]
    """
    image = pixels.clamp(0, 1).mul(255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray(image).save(path)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, help="Quick check, writes to the smoke directory.")
    parser.add_argument(
        "--subset",
        type=int,
        help="Only the first N samples, written to the main results directory (FID metrics are not measured).",
    )
    parser.add_argument("--name", default=SOLUTION_NAME)
    parser.add_argument(
        "--reference-weights", type=float, nargs=3, default=REFERENCE_WEIGHTS, help="tile, position, global"
    )
    arguments = parser.parse_args()
    results_directory = SMOKE_RESULTS_DIRECTORY if arguments.max_samples else RESULTS_DIRECTORY
    predictions_directory = results_directory / arguments.name / "images"
    predictions_directory.mkdir(parents=True, exist_ok=True)
    sample_names = json.loads((DATASET_DIRECTORY / "manifest.json").read_text())["sample_names"]
    sample_names = sample_names[: arguments.max_samples or arguments.subset]
    all_latents = torch.load(DATASET_DIRECTORY / "latents.pt")

    solution = HildegardSolution(tuple(arguments.reference_weights))
    gc.collect()
    torch.cuda.empty_cache()
    baseline_bytes = torch.cuda.memory_allocated()
    solution.load()
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    for _ in range(NUM_WARMUP_SAMPLES):
        fake_latents = torch.randn(
            1, LATENT_CHANNELS, INPUT_LATENT_SIZE, INPUT_LATENT_SIZE, generator=generator, device="cuda"
        )
        solution.upscale(fake_latents, SEED)
    torch.cuda.synchronize()
    weights_bytes = torch.cuda.memory_allocated() - baseline_bytes

    latencies_ms: list[float] = []
    peak_bytes = 0
    for index, name in enumerate(sample_names):
        latents = all_latents[index : index + 1].float().cuda()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        pixels = solution.upscale(latents, SEED + index)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start_time) * 1000)
        peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated())
        if tuple(pixels.shape) != (1, 3, OUTPUT_PIXEL_SIZE, OUTPUT_PIXEL_SIZE):
            raise ValueError(f"Hildegard returned {tuple(pixels.shape)}")
        save_prediction(pixels, predictions_directory / f"{name}.png")
        print(f"{index + 1}/{len(sample_names)} {latencies_ms[-1]:.0f} ms", flush=True)

    efficiency = {
        "latency_ms": float(np.mean(latencies_ms)),
        "latency_ms_median": float(np.median(latencies_ms)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
        "peak_vram_mb": (peak_bytes - baseline_bytes) / BYTES_PER_MEGABYTE,
        "weights_vram_mb": weights_bytes / BYTES_PER_MEGABYTE,
    }
    external = {
        "description": "Hildegard refiner (FLUX.2 klein 9B fp8 + LoRA, ComfyUI), full VAE decode, 4 steps, "
        "constant baseline prompt",
        "settings": solution.get_settings(),
        "efficiency": efficiency,
    }
    (results_directory / arguments.name / "external.json").write_text(json.dumps(external, indent=2))
    print(json.dumps(external, indent=2))


if __name__ == "__main__":
    main()
