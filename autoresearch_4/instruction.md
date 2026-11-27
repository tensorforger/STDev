# Autoresearch 4: Qwen-Image 2.1 latent upscaler

Phase 4. Goal: calibrate the pretrain recipe (autoresearch_3, `./autoresearch_3/experiment_logs.md`) to the
Qwen-Image 2.1 VAE and test new ideas: low precision training and quantized inference. Log:
`./autoresearch_4/experiment_logs.md` (long-term memory, update every round). Earlier phases are read-only references.

Research setup differs from the pretrain on purpose: single dataset, no hff conditioning, no input corruption, smaller
model (~50M), single resolution 512 -> 1024.

## Dataset

`./train_datasets/qwen_4klsdb_40k`: 40k 4KLSDB photos (`train_x4` files 0-64, 4K images center cropped and
antialias-resized to 1024, `crop_maximal_rectangle`), encoded with the Qwen-Image 2.1 VAE (bf16, posterior mode, opaque
RGBA). `block_<index>.vz.safetensors`, `vaezip.load(path)[key].decode(dtype=torch.float32)`, latents stored as fp16:

| Key | Shape | Content |
|---|---|---|
| `latents_1024` | [1, 64, 64, 64] | target (normalized latents, 16x compression) |
| `latents_512` | [1, 64, 32, 32] | input: bilinear 2x downscale (cv2 INTER_LINEAR, uint8), exactly as the benchmark |
| `hff_1024`, `hff_512` | [1] | Immerkaer hff (unused in research) |

Generation: `scripts/download_flux2_sources.py 4klsdb --num-files 65`, `scripts/encode_qwen_datasets.py`.

## Objective

Task: normalized latents `[B, 64, 32, 32] -> [B, 64, 64, 64]` (512 -> 1024 px), no conditioning besides the latents.
Metrics in priority order: patch_fid (main), then fid / low_res_mse / chroma ΔE00, then latency / peak VRAM (a Pareto
front with quality: move the whole front). Training loss only for same-data, same-loss comparisons (same SEED = same
batches).

Key questions:
1. What Qwen-Image 2.1 latents are and how they differ from FLUX.2 / Wan 2.1 latents (smoothness, noise robustness).
2. Normalized residual vs direct latent generation (one direct run is enough).
3. Low precision training: FP8 / MXFP8 / NVFP4 (torchao), mixed precision per layer type. Same 30 min budget:
   more samples per second must turn into better quality.
4. Low precision inference and quantization of trained models (PTQ, quantization aware training): latency / VRAM vs
   quality.

Final deliverable: "RECOMMENDED PRETRAIN RECIPE" section in the log (with the precision recipe for training and
inference).

## Benchmark

Fixed set `./benchmark_datasets/axis_v1_1k_qwen_2x`: 1000 Axis-V1 photos, ground truth 1024, input = bilinear 512
downscale encoded with the Qwen-Image 2.1 VAE. The benchmark upscales all 1000 latents (batch 8, `upscale`), decodes
with the **texture-fix** Qwen-Image 2.1 decoder (bf16, RGB of the RGBA output) and reports:

| Metric | Key in `metrics.json` |
|---|---|
| patch FID (native 256 px tiles) | `quality.patch_fid` |
| FID (whole image, Inception 299) | `quality.fid` |
| low res MSE (report x 1e-4) | `quality.low_res_mse` |
| chroma ΔE00 | `quality.chroma_ciede2000` |
| latency ms (p95), single forward, after `compile()` | `efficiency.latency_ms_p95` |
| peak VRAM MB, eager single forward, weights included | `efficiency.peak_vram_mb` |

References: `python scripts/run_qwen_benchmark.py --reference oracle|pixel_bicubic|latent_bilinear` (add
`--decoder official` for the official decoder, info only). Results: `./raw_results/qwen/<name>/metrics.json`.

## Rules

- One self-contained run file per run: `./src/stdev/qwen_upscalers/run_<ID>.py` implementing
  `stdev.qwen_upscalers.base.Upscaler` (4 digit ID, no other project imports; earlier run files are read-only).
  Weights -> `./trained_models/qwen/run_<ID>/`, tensorboard -> `./tensorboard/qwen/run_<ID>/`.
  Quantization runs may load the weights of an earlier run instead of training from scratch.
- 4 parallel runs per round (one RTX 5090 each), 30 min training measured by the run (compile before the clock),
  ~5 rounds, ~20 runs. `./qwen_infra/start_round.sh 0000 0001 0002 0003` (`--smoke` for a quick check),
  launch asynchronously and wait for the notification. Logs: `./train_logs/qwen/run_<ID>/`.
- Limits: window / factorized attention only, <= 30 sampling steps, <= 1B parameters.
- Do not modify the benchmark, infrastructure or dataset. Scratch work in `./dev/autoresearch_4/`.
