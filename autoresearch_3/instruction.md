# Autoresearch 3: FLUX.2 latent upscaler

Phase 3 of autoresearch. Phase 1 (`./autoresearch`, FLUX.2, direct HR latent generation, UNet + window transformer
blocks) and phase 2 (`./autoresearch_2`, Wan 2.1, normalized residual generation, plain window transformer) are
read-only references. Goal: the best FLUX.2 recipe, test whether the findings of phases 1 and 2 compound, small
ablations, and the recipe for the new FLUX.2 upscaler pretrain. Log: `./autoresearch_3/experiment_logs.md` (long-term
memory, keep it updated every round).

## Task

512 -> 1024 px image upscaling in FLUX.2 VAE latent space only: normalized latents `[B, 32, 64, 64] -> [B, 32, 128, 128]`.
No conditioning besides the latents (no text; hff is not used in research, it is added at pretrain).

## Dataset

Two sources, 20k samples each, every image center cropped and resized to 1024x1024 (`crop_maximal_rectangle`,
antialiased), encoded with the full FLUX.2 VAE (bf16, posterior mean):

- `./train_datasets/flux2_relaion_20k` (A): ReLAION high resolution photos (min side >= 1024).
- `./train_datasets/flux2_4klsdb_20k` (B): 4KLSDB 4K photos (~4x downscale: dense, sharp detail).

`block_<index>.vz.safetensors` (vaezip, `vaezip.load(path)[key].decode(dtype=torch.float32)`), codec `lbrq_8bit`:

| Key | Shape | Content |
|---|---|---|
| `latents_1024` | [1, 32, 128, 128] | target |
| `latents_512` | [1, 32, 64, 64] | input: bilinear 2x downscale (cv2 INTER_LINEAR, uint8), exactly as the benchmark |
| `latents_512_nearest` / `_bicubic` / `_lanczos` | [1, 32, 64, 64] | corrupted inputs: sharper downscaling filters |
| `latents_512_antialias` | [1, 32, 64, 64] | corrupted input: antialiased (blurrier) bilinear |
| `latents_512_jpeg` | [1, 32, 64, 64] | corrupted input: bilinear + JPEG quality 20-50 |
| `hff_1024`, `hff_512` | [1] | Immerkaer hff (unused in research) |

Generation: `scripts/download_flux2_sources.py`, `scripts/encode_flux2_datasets.py`.

## Objective (priority order)

1. **patch_fid** (lower is better): main metric.
2. fid, low_res_mse, chroma ΔE00 (roundtrip consistency: prediction downscaled to 512 vs the input image).
3. latency_ms_p95 and peak_vram_mb of a single model forward (batch 1, 64 -> 128 latents). Quality and efficiency form
   a Pareto front: understand the parameter relationships, move the whole front.
4. Training loss, only for architecture comparisons with identical data / loss setups (same SEED = same batches).

## Benchmark

Fixed set `./benchmark_datasets/axis_v1_1k_flux2_2x`: 1000 Axis-V1 photos, ground truth 1024, input = bilinear 512
downscale encoded with the FLUX.2 VAE. The benchmark upscales all 1000 latents (batch 8, `upscale`), decodes with the
full FLUX.2 VAE (bf16) and computes:

| Metric | Key in `metrics.json` |
|---|---|
| patch FID (native 256 px tiles) | `quality.patch_fid` |
| FID (whole image, Inception 299) | `quality.fid` |
| low res MSE (report x 1e-4) | `quality.low_res_mse` |
| chroma ΔE00 | `quality.chroma_ciede2000` |
| latency ms (p95), single forward, compiled | `efficiency.latency_ms_p95` |
| peak VRAM MB, eager single forward, weights included | `efficiency.peak_vram_mb` |

Extra info: `quality.latent_mse_hr`, `efficiency.end_to_end_latency_ms_p95` (full `upscale` + decode, batch 1).
References (`python scripts/run_flux_benchmark.py --reference oracle|pixel_bicubic|latent_bilinear`) are in
`./raw_results/flux/reference_*`. Results: `./raw_results/flux/run_<ID>/metrics.json`, first 32 predictions in `images/`.

## Rules

- One self-contained run file per run: `./src/stdev/flux_upscalers/run_<ID>.py` (4 digit ID, continue from the highest),
  implementing `stdev.flux_upscalers.base.Upscaler`; no other project imports. Previous run files are read-only.
  Weights -> `./trained_models/flux/run_<ID>/`, tensorboard `train/loss` -> `./tensorboard/flux/run_<ID>/` (not in smoke mode).
- Budget: N = 4 parallel runs per round (one RTX 5090 each, 32 GB), T = 30 min training measured by the run itself
  (compile before the clock; guard kills at T + 5 min). About 5 rounds, ~20 runs total.
- Limits: no O(n^2) attention over the image (window / factorized only), <= 30 sampling steps, <= 1B parameters.
- Launch a round asynchronously, then wait for the completion notification (do not poll):
  `./flux_infra/start_round.sh 0000 0001 0002 0003` (`--smoke` for a quick check). Logs: `./train_logs/flux/run_<ID>/`.
- Do not modify the benchmark, infrastructure or datasets. Small experiments and scratch files go to `./dev`.
- Training seed noise in phase 2 was 0.1-0.3 patch FID at 30 min: use same-seed paired loss and replicates for decisions.

## Key questions

1. Direct HR latent generation (phase 1) vs normalized residual generation (phase 2).
2. UNet with window transformer blocks (phase 1 recipe) vs plain window transformer (phase 2 recipe), ~50M each.
3. Other phase 2 shifts on FLUX.2: learned source path, condition RMS norm + QK-norm, few steps / high shift / noise
   temperature < 1, small batch + higher LR.
4. Data: A only, B only, A + B.
5. Input corruption during training (other downscaling filters, JPEG): does it harm the bilinear benchmark? If not, it
   adds robustness for image-to-image upscaling and generative model artifacts.

Final deliverable: a "RECOMMENDED PRETRAIN RECIPE" section in the log (architecture, size, data mix, corruption,
training and sampling settings; hff conditioning is added at pretrain).
