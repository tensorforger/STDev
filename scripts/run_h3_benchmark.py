"""
Reference rows of the autoresearch_6 benchmark (`stdev.h3_benchmark`): VAE decoders applied to the ground truth latents
of `benchmark_datasets/axis_v1_1k_h3_flux2`.

References:
    h3            MiniMax-H3-Image decoder (released tiling) on the H3 latents: the baseline to beat
    h3_untiled    same decoder, whole image at once
    ostris        ostris/ostris_h3_image_vae decoder fine-tune (tiled) on the H3 latents
    flux2         FLUX.2 VAE decoder on the FLUX.2 latents of the ground truth: oracle (upper bound of the stack)
    flux2_small   black-forest-labs/FLUX.2-small-decoder (distilled 28M decoder, same encoder / latent space) on the FLUX.2
                  latents: oracle of the cheaper full-quality stack
    taef2         TAEF2 tiny decoder on the FLUX.2 latents: oracle of the fast stack

Quality on 1000 images -> `raw_results/h3/reference_<name>/metrics.json`; `--speed` adds batch 1 decode latency and
total peak VRAM (weights + activations) at 1024 / 2048 / 4096 px.

Trained translators (`--translator src/stdev/h3_translators/run_<ID>.py`): speed of `single_forward` (one model pass,
batch 1, at the H3 grids of 1024 / 2048 / 4096 px: eager peak VRAM with weights, then compiled latency), then the full
`translate` (eager) on all 1000 images, decoded with the FLUX.2 small decoder (main, `quality`) and TAEF2
(`quality_taef2`) -> `raw_results/h3/<run name>/metrics.json`.

Run:
    python scripts/run_h3_benchmark.py --reference h3 --speed
    python scripts/run_h3_benchmark.py --reference flux2 --smoke
    python scripts/run_h3_benchmark.py --translator src/stdev/h3_translators/run_0000.py [--smoke]
    python scripts/run_h3_benchmark.py --prepare-cache
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm

from stdev.h3_benchmark.benchmark import (
    BYTES_PER_MEGABYTE,
    FLUX2_LATENT_CHANNELS,
    H3_LATENT_CHANNELS,
    SPEED_RESOLUTIONS,
    H3BenchmarkDataset,
    build_fid_evaluator,
    evaluate_quality,
    measure_latency,
    measure_peak_memory,
)
from stdev.h3_translators.base import Translator
from stdev.h3_translators.loading import load_translator
from stdev.vaes.flux_2_vae import Flux2VAE
from stdev.vaes.minimax_h3_image_vae import MiniMaxH3ImageVAE
from stdev.vaes.tiny_flux_2_vae import TinyFlux2VAE
from stdev.vaes.vae import VAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "axis_v1_1k_h3_flux2"
STATISTICS_DIRECTORY = REPOSITORY_ROOT / "benchmark_cache" / "axis_v1_1k_h3_flux2"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "raw_results" / "h3"
MODELS_DIRECTORY = REPOSITORY_ROOT / "models"
VAE_DTYPE = torch.bfloat16
REFERENCES = ("h3", "h3_untiled", "ostris", "flux2", "flux2_small", "taef2")
DECODE_BATCH_SIZE = 4
SMOKE_NUM_SAMPLES = 16
NUM_SAVED_IMAGES = 32
WARMUP_ITERATIONS = 3
LATENCY_ITERATIONS = 10
TRANSLATOR_WARMUP_ITERATIONS = 5
TRANSLATOR_LATENCY_ITERATIONS = 20
TRANSLATE_BATCH_SIZE = 8
TRANSLATOR_DECODERS = {"quality": "flux2_small", "quality_taef2": "taef2"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference", choices=REFERENCES)
    group.add_argument("--translator", type=Path)
    group.add_argument("--prepare-cache", action="store_true", help="Only fill the FID statistics cache.")
    parser.add_argument("--speed", action="store_true", help="Also measure latency / VRAM at 1024-4096 px.")
    parser.add_argument("--speed-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_decoder(reference: str) -> tuple[VAE, str]:
    """
    returns:
        VAE wrapper and the latent space it decodes ("h3" or "flux2")
    """
    if reference in ("h3", "h3_untiled"):
        path = MODELS_DIRECTORY / "MiniMax-H3-Image" / "vae_diffusers"
        return MiniMaxH3ImageVAE(str(path), dtype=VAE_DTYPE, use_tiling=reference == "h3"), "h3"
    if reference == "ostris":
        return MiniMaxH3ImageVAE(str(MODELS_DIRECTORY / "ostris_h3_image_vae"), dtype=VAE_DTYPE), "h3"
    if reference == "flux2":
        return Flux2VAE(str(MODELS_DIRECTORY / "FLUX.2-klein-4B" / "vae"), dtype=VAE_DTYPE), "flux2"
    if reference == "flux2_small":
        return Flux2VAE(str(MODELS_DIRECTORY / "FLUX.2-small-decoder"), dtype=VAE_DTYPE), "flux2"
    if reference == "taef2":
        return TinyFlux2VAE(str(MODELS_DIRECTORY / "taef2" / "taef2.safetensors"), dtype=VAE_DTYPE), "flux2"
    raise ValueError(reference)


def decoder_only_bytes(vae: VAE) -> int:
    decoder = getattr(vae.model, "decoder")
    modules = [decoder] + [getattr(vae.model, name) for name in ("post_quant_conv",) if hasattr(vae.model, name)]
    return sum(parameter.numel() * parameter.element_size() for module in modules for parameter in module.parameters())


def measure_speed(vae: VAE, latent_space: str) -> dict[str, dict[str, float | str]]:
    """
    Decode of random normalized latents at every SPEED_RESOLUTIONS size, batch 1, eager.
    """
    channels, compression = (H3_LATENT_CHANNELS, 16) if latent_space == "h3" else (FLUX2_LATENT_CHANNELS, 8)
    decoder_megabytes = decoder_only_bytes(vae) / BYTES_PER_MEGABYTE
    results: dict[str, dict[str, float | str]] = {}
    for resolution in SPEED_RESOLUTIONS:
        size = resolution // compression
        latents = torch.randn(1, channels, 1, size, size, device="cuda")

        def decode() -> torch.Tensor:
            return vae.decode(latents)

        try:
            activation_megabytes = measure_peak_memory(decode)
            latency = measure_latency(decode, WARMUP_ITERATIONS, LATENCY_ITERATIONS)
            results[str(resolution)] = {
                "latency_ms_mean": latency["mean"],
                "latency_ms_p95": latency["p95"],
                "peak_vram_mb": decoder_megabytes + activation_megabytes,
                "decoder_weights_mb": decoder_megabytes,
            }
        except torch.OutOfMemoryError:
            results[str(resolution)] = {"error": "out of memory"}
        torch.cuda.empty_cache()
        print(resolution, results[str(resolution)], flush=True)
    return results


def make_predictor(vae: VAE, latent_space: str, dataset: H3BenchmarkDataset) -> Callable[[torch.Tensor], torch.Tensor]:
    latents = dataset.latents_h3 if latent_space == "h3" else dataset.latents_flux2

    def predict(indices: torch.Tensor) -> torch.Tensor:
        batch = latents[indices].cuda().float()
        return torch.cat(
            [
                vae.decode(batch[start : start + DECODE_BATCH_SIZE].unsqueeze(2))
                for start in range(0, batch.shape[0], DECODE_BATCH_SIZE)
            ]
        )

    return predict


@torch.no_grad()
def measure_translator_speed(translator: Translator, smoke: bool) -> dict[str, dict[str, float | str]]:
    """
    Eager peak VRAM (weights + activations) of one `single_forward` per resolution, then compiled latency.
    The weights must not be loaded yet: memory is counted from the level before `prepare_to_benchmark`.
    """
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline_bytes = torch.cuda.memory_allocated()
    translator.prepare_to_benchmark()
    torch.cuda.synchronize()
    weights_megabytes = (torch.cuda.memory_allocated() - baseline_bytes) / BYTES_PER_MEGABYTE
    resolutions = SPEED_RESOLUTIONS[:1] if smoke else SPEED_RESOLUTIONS
    results: dict[str, dict[str, float | str]] = {}
    inputs = {}
    for resolution in resolutions:
        inputs[resolution] = torch.randn(1, H3_LATENT_CHANNELS, resolution // 16, resolution // 16, device="cuda")
        try:
            activation_megabytes = measure_peak_memory(lambda: translator.single_forward(inputs[resolution]))
            results[str(resolution)] = {"peak_vram_mb": weights_megabytes + activation_megabytes, "weights_mb": weights_megabytes}
        except torch.OutOfMemoryError:
            results[str(resolution)] = {"error": "out of memory"}
        torch.cuda.empty_cache()
    if not smoke:
        translator.compile()
    for resolution in resolutions:
        if "error" in results[str(resolution)]:
            continue
        latency = measure_latency(
            lambda: translator.single_forward(inputs[resolution]),
            1 if smoke else TRANSLATOR_WARMUP_ITERATIONS,
            2 if smoke else TRANSLATOR_LATENCY_ITERATIONS,
        )
        results[str(resolution)].update({"latency_ms_mean": latency["mean"], "latency_ms_p95": latency["p95"]})
        print(resolution, results[str(resolution)], flush=True)
    if not smoke:
        torch._dynamo.reset()
    return results


@torch.no_grad()
def translate_dataset(translator: Translator, dataset: H3BenchmarkDataset) -> torch.Tensor:
    """
    returns:
        latents: [N, 32, 128, 128], predicted normalized FLUX.2 latents, dtype: torch.float32, on cpu
    """
    predictions = []
    for start in tqdm(range(0, len(dataset), TRANSLATE_BATCH_SIZE), desc="Translate"):
        latents = dataset.latents_h3[start : start + TRANSLATE_BATCH_SIZE].cuda().float()
        predicted = translator.translate(latents).float()
        expected_shape = (latents.shape[0], FLUX2_LATENT_CHANNELS, 2 * latents.shape[2], 2 * latents.shape[3])
        if tuple(predicted.shape) != expected_shape:
            raise ValueError(f"translate returned {tuple(predicted.shape)}, expected {expected_shape}")
        predictions.append(predicted.cpu())
    return torch.cat(predictions)


def make_latent_predictor(vae: VAE, latents: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    def predict(indices: torch.Tensor) -> torch.Tensor:
        batch = latents[indices].cuda().float()
        return torch.cat(
            [
                vae.decode(batch[start : start + DECODE_BATCH_SIZE].unsqueeze(2))
                for start in range(0, batch.shape[0], DECODE_BATCH_SIZE)
            ]
        )

    return predict


def benchmark_translator(translator_path: Path, smoke: bool) -> None:
    translator = load_translator(translator_path)
    name = translator.get_name()
    metrics: dict = {"settings": {"name": name, "smoke": smoke, "decoder_dtype": str(VAE_DTYPE)}}
    metrics["speed"] = measure_translator_speed(translator, smoke)
    translator.prepare_to_benchmark()
    dataset = H3BenchmarkDataset(DATASET_DIRECTORY, max_samples=SMOKE_NUM_SAMPLES if smoke else None)
    predicted_latents = translate_dataset(translator, dataset)
    metrics["latent_mse"] = float((predicted_latents - dataset.latents_flux2.float()).square().mean())
    translator.model = None
    torch.cuda.empty_cache()
    results_directory = RESULTS_DIRECTORY / name
    with tempfile.TemporaryDirectory() as temporary_directory:
        statistics_directory = Path(temporary_directory) if smoke else STATISTICS_DIRECTORY
        for key, reference in TRANSLATOR_DECODERS.items():
            vae, _ = load_decoder(reference)
            images_directory = None if smoke else results_directory / f"images_{reference}"
            if images_directory is not None and images_directory.exists():
                shutil.rmtree(images_directory)
            metrics[key] = evaluate_quality(
                make_latent_predictor(vae, predicted_latents),
                dataset,
                build_fid_evaluator(dataset, statistics_directory),
                images_directory,
                NUM_SAVED_IMAGES,
            )
            del vae
            torch.cuda.empty_cache()
    metrics["settings"]["num_samples"] = len(dataset)
    if not smoke:
        results_directory.mkdir(parents=True, exist_ok=True)
        (results_directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def main() -> None:
    arguments = parse_arguments()
    if arguments.prepare_cache:
        build_fid_evaluator(H3BenchmarkDataset(DATASET_DIRECTORY), STATISTICS_DIRECTORY)
        return
    if arguments.translator is not None:
        benchmark_translator(arguments.translator, arguments.smoke)
        return
    vae, latent_space = load_decoder(arguments.reference)
    name = f"reference_{arguments.reference}"
    results_directory = RESULTS_DIRECTORY / name
    metrics_path = results_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() and not arguments.smoke else {}
    metrics["settings"] = {"name": name, "smoke": arguments.smoke, "decoder_dtype": str(VAE_DTYPE)}

    if arguments.speed or arguments.speed_only:
        metrics["speed"] = measure_speed(vae, latent_space)
    if not arguments.speed_only:
        dataset = H3BenchmarkDataset(DATASET_DIRECTORY, max_samples=SMOKE_NUM_SAMPLES if arguments.smoke else None)
        with tempfile.TemporaryDirectory() as temporary_directory:
            statistics_directory = Path(temporary_directory) if arguments.smoke else STATISTICS_DIRECTORY
            images_directory = None if arguments.smoke else results_directory / "images"
            if images_directory is not None and images_directory.exists():
                shutil.rmtree(images_directory)
            metrics["quality"] = evaluate_quality(
                make_predictor(vae, latent_space, dataset),
                dataset,
                build_fid_evaluator(dataset, statistics_directory),
                images_directory,
                NUM_SAVED_IMAGES,
            )
        metrics["settings"]["num_samples"] = len(dataset)
    if not arguments.smoke:
        results_directory.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
