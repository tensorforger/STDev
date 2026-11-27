# Autoresearch 5: Ming-Image VAE latent upscaler

Phase 5 (one exploratory round so far). Goal: check how the FLUX.2 recipe (autoresearch_3) and the transferable
autoresearch_4 findings work on the Ming-Image 0.1 / 0.2 Design VAE (= Qwen-Image-Layered RGBA VAE; Wan 2.1 / Qwen-Image
1 architecture, 8x, 16 channels; check: `reports/ming_image_vae_check.md`). Log: `./autoresearch_5/experiment_logs.md`.
Research setup as in autoresearch_4: single dataset, no hff, no input corruption, ~50M model, 512 -> 1024.

## Dataset

`./train_datasets/ming_4klsdb_40k`: the same 40k 4KLSDB 1024 crops as autoresearch_4, encoded with the Ming-Image VAE
(bf16, posterior mode, opaque RGBA, per channel normalized with `src/stdev/vaes/ming_image_vae.py` statistics).
`block_<index>.vz.safetensors`, latents stored as fp16:

| Key | Shape | Content |
|---|---|---|
| `latents_1024` | [1, 16, 128, 128] | target |
| `latents_512` | [1, 16, 64, 64] | input: bilinear 2x downscale (cv2 INTER_LINEAR, uint8), exactly as the benchmark |
| `hff_1024`, `hff_512` | [1] | Immerkaer hff (unused in research) |

Generation: `scripts/encode_ming_datasets.py`.

## Objective

Task: normalized latents `[B, 16, 64, 64] -> [B, 16, 128, 128]` (512 -> 1024 px), no conditioning besides the latents.
Metrics in priority order: patch_fid (main), then fid / low_res_mse / chroma ΔE00, then latency / peak VRAM.

## Benchmark

Fixed set `./benchmark_datasets/axis_v1_1k_ming_2x`: the 1000 Axis-V1 crops of autoresearch_3 / 4 (same ground truth,
same FID statistics), input = bilinear 512 downscale encoded with the Ming-Image VAE; predictions decoded with the same
VAE (bf16, RGB of the RGBA output). Metrics and keys as in autoresearch_4 (`quality.patch_fid`, `quality.fid`,
`quality.low_res_mse`, `quality.chroma_ciede2000`, `efficiency.latency_ms_p95`, `efficiency.peak_vram_mb`).
References: `python scripts/run_ming_benchmark.py --reference oracle|pixel_bicubic|latent_bilinear`.
Results: `./raw_results/ming/<name>/metrics.json`.

## Rules

- Run files `./src/stdev/ming_upscalers/run_<ID>.py` implementing `stdev.ming_upscalers.base.Upscaler`; weights ->
  `./trained_models/ming/run_<ID>/`; `./ming_infra/start_round.sh 0000 0001 0002 0003` (4 GPUs, 30 min each).
- Check compiled training with `dev/autoresearch_4/compiled_check.py`-style runs before a round (`--smoke` skips
  compilation). Scratch work in `./dev/autoresearch_5/`.
