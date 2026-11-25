# FLUX.2 upscaler: unified benchmark against other 2x upscaling solutions (2026-11-25, RTX 5090)

Same pattern as the Wan 2.1 benchmark (`reports/wan_upscaling_benchmark.md`). Code: `scripts/run_flux_upscale_benchmark.py`
(prerequisites and download commands at the top), `src/stdev/flux_upscale_benchmark/` (dataset, solution wrappers,
registry; the runner and metrics of `src/stdev/upscale_benchmark/` are shared), Hildegard driver
`scripts/run_hildegard_benchmark.py` (runs in the ComfyUI venv), tiny VAE `src/stdev/vaes/tiny_flux_2_vae.py`, plot
`scripts/plot_upscale_benchmark.py --results-directory raw_results/flux_upscale_benchmark_axis_v1_1k`.
Results: `raw_results/flux_upscale_benchmark_axis_v1_1k/` (`results.json`, `<solution>/results.json`, all 1024x1024
predictions as PNG, `latency_vs_patch_fid.png`); logs `raw_results/flux_upscale_benchmark_*.log`.

## Setup

- Task for every solution: the normalized FLUX.2 latent of a 512x512 image (`[1, 32, 64, 64]`, batch norm applied,
  unpatchified; nothing else: no prompt, no hff, no ground truth information) in, the decoded 1024x1024 image out.
  Upscaling before or after decoding is an implementation detail.
- Data: the autoresearch_3 set `benchmark_datasets/axis_v1_1k_flux2_2x` (first 1000 Axis-V1 images, antialiased center
  crop to 1024 = ground truth, cv2 bilinear 2x downscale, full FLUX.2 VAE encode = input). Rebuilt locally with the same
  builder; references reproduce the training reports (`decode_bicubic` 8.12 = autoresearch_3 pixel bicubic 8.12;
  20M student 2.20 vs 2.21). VAE oracle (privileged, not a solution): 0.71 (autoresearch_3).
- Metrics as in the Wan benchmark: `patch_fid` (native 256 px tiles, primary), `fid` (whole image at 299), roundtrip
  `low_res_mse` / `chroma ΔE00` (prediction bilinear downsampled to 512 vs the 512 input image), latency (batch 1,
  warmup on random latents, synchronized wall time per real sample, decode included, eager), peak VRAM (peak allocated
  CUDA memory above the level before loading, weights included). One noise seed per image (`seed + index`).
- Ours: raw weights (not EMA), null hff, guidance 1, bf16, eager. Base = `flux_pretrain_v1` step 129590 (30 Euler steps,
  shift 2, noise 1.0); 200M student = `flux_distill_v1` step 1083; 20M student = `flux_small_s20m` step 109780; 1 step.
  Decoder: full FLUX.2 VAE or TAEF2 (`madebyollin/taef2`, works in the normalized latent space; verified: 33.1 dB PSNR
  on ground truth latents vs 36.2 dB full VAE, 18.2 dB if fed raw latents).
- Third party (Wan-specific solutions removed):
  - `flow_upscaler` (+ `_tiny_vae`): the released FlowUpscaler (`TensorForger/FlowUpscaler`, 59M UNet, our ancestor
    project), one step at timestep 1000 as in its ComfyUI node, normalized latents, bf16.
  - `sesquilsr`: SesquiLSR FLUX.2 weights (`upscaler_Flux2.safetensors`, repository at 3872801, 2026-11-21, V1
    architecture, loaded with its `load_upscaler`), scale 2.0, bf16, on raw unpatchified latents (denormalized with the
    VAE batch norm statistics as its `make_flux2` adaptor does), full decode.
  - `latent_bicubic_decode`, `decode_bicubic`, `decode_span` (2xNomosUni_span_multijpg), `decode_realesrgan`
    (RealESRGAN_x2plus): as in the Wan benchmark, with the FLUX.2 VAE.
  - VOSR (CVPR 2026, `CSWRY/VOSR`, one step `_os` checkpoints, official defaults: fp32, posterior mode encode, DINOv2
    condition, wavelet color fix), "decode, then upscale": `decode_vosr_{0_5b,1_4b}` = full decode at 512, bicubic 2x,
    VOSR; `decode_256_vosr_{0_5b,1_4b}_x4` = decode area downscaled to 256, bicubic 4x, VOSR (the factor it is trained
    for). The wrapper reproduces the official `inference_vosr_onestep.py` (same quality on the same inputs).
  - Hildegard (`42lux/hildegard`, rank 32 LoRA for FLUX.2 klein 9B, ComfyUI only): run in-process in ComfyUI with the
    nodes of its example workflow: full decode at 512, Hildegard Plan (one 1024 tile, lanczos 2x), references tile /
    3x3 position map (cell 512) / global thumbnail (768) VAE encoded, klein 9B fp8 (`FLUX.2-klein-9b-fp8`) + LoRA 1.0,
    empty latent, Flux2Scheduler 4 steps, euler, no CFG, full decode. The README "baseline prompt" is used as a
    constant (same for every image, encoded once at load; Qwen3-8B encoder time and memory not counted).
    `hildegard` = official reference weights (tile 0.9, position 0.8, global 0.8, `Flux2KleinRefLatentWeight` of
    ComfyUI-Flux2Klein-Enhancer); `hildegard_ref_weights_1` = all 1.0 (see conclusion 8). 3 warmup samples instead of 10.
    Evaluated on the first 30 images only (`--subset 30`), per image metrics only.

## Results (sorted by patch_fid)

Full set (1000 images):

| solution | patch_fid | fid | low_res_mse 1e-4 | chroma ΔE00 | latency ms (p95) | peak VRAM MB |
|---|---|---|---|---|---|---|
| `ours_base_30_steps_200m` | **1.72** | 1.95 | 6.4 | 1.81 | 1671.1 (1694.0) | 3027 |
| `ours_distilled_1_step_200m` | 1.75 | 2.02 | 6.5 | 1.82 | 172.1 (172.7) | 3030 |
| `ours_distilled_1_step_20m` | 2.20 | 2.63 | 7.4 | 1.93 | 131.1 (131.4) | 2648 |
| `ours_distilled_1_step_200m_tiny_vae` | 3.22 | 4.55 | 8.0 | 2.30 | 59.8 (60.3) | 1035 |
| `ours_distilled_1_step_20m_tiny_vae` | 3.81 | 5.17 | 8.8 | 2.38 | 18.8 (19.3) | 690 |
| `flow_upscaler` | 4.28 | 4.50 | 12.5 | 2.41 | 128.2 (128.5) | 2725 |
| `flow_upscaler_tiny_vae` | 4.45 | 7.54 | 13.9 | 2.77 | 16.1 (16.5) | 768 |
| `decode_span` | 6.18 | 3.67 | 9.1 | 2.62 | 29.6 (29.9) | 784 |
| `sesquilsr` | 7.71 | 7.26 | 11.6 | 2.30 | 123.5 (123.8) | 2617 |
| `decode_bicubic` | 8.12 | 1.77 | 5.0 | 1.74 | 24.6 (24.9) | 779 |
| `decode_realesrgan` | 9.03 | 7.99 | 10.1 | 2.34 | 52.8 (53.4) | 811 |
| `decode_vosr_1_4b` | 25.88 | 15.83 | 27.3 | 2.94 | 595.3 (596.8) | 9207 |
| `decode_256_vosr_1_4b_x4` | 28.34 | 15.89 | 25.8 | 2.89 | 595.7 (597.1) | 9204 |
| `decode_vosr_0_5b` | 29.67 | 24.08 | 31.9 | 3.38 | 405.5 (406.2) | 5336 |
| `decode_256_vosr_0_5b_x4` | 33.58 | 23.23 | 30.1 | 3.26 | 405.3 (405.8) | 5334 |
| `latent_bicubic_decode` | 114.32 | 88.41 | 95.5 | 5.93 | 120.9 (121.0) | 2609 |

Hildegard (~7 s per image) was run only on the first 30 images: fid / patch_fid **not measured** (FID on 30 images is
meaningless, the full set would take ~2-3 h per variant). Per image metrics of all solutions on the same 30 images
(`raw_results/flux_upscale_benchmark_axis_v1_1k/subset_30_roundtrip.json`):

| solution (first 30 images) | low_res_mse 1e-4 | chroma ΔE00 | latency ms | peak VRAM MB |
|---|---|---|---|---|
| `decode_bicubic` | 5.1 | 2.04 | 24.6 | 779 |
| `ours_distilled_1_step_200m` | 7.0 | 2.18 | 172.1 | 3030 |
| `ours_base_30_steps_200m` | 6.9 | 2.19 | 1671.1 | 3027 |
| `ours_distilled_1_step_20m` | 7.6 | 2.27 | 131.1 | 2648 |
| `ours_distilled_1_step_200m_tiny_vae` | 8.2 | 2.56 | 59.8 | 1035 |
| `sesquilsr` | 11.3 | 2.60 | 123.5 | 2617 |
| `ours_distilled_1_step_20m_tiny_vae` | 8.9 | 2.64 | 18.8 | 690 |
| `decode_realesrgan` | 10.3 | 2.65 | 52.8 | 811 |
| `flow_upscaler` | 13.4 | 2.83 | 128.2 | 2725 |
| `decode_span` | 10.0 | 2.91 | 29.6 | 784 |
| `flow_upscaler_tiny_vae` | 14.5 | 3.12 | 16.1 | 768 |
| `decode_256_vosr_1_4b_x4` | 23.8 | 3.28 | 595.7 | 9204 |
| `decode_vosr_1_4b` | 26.3 | 3.38 | 595.3 | 9207 |
| `decode_256_vosr_0_5b_x4` | 27.6 | 3.71 | 405.3 | 5334 |
| `hildegard_ref_weights_1` | 21.7 | 3.80 | 6994 | 11920 |
| `decode_vosr_0_5b` | 29.8 | 3.89 | 405.5 | 5336 |
| `latent_bicubic_decode` | 82.4 | 6.50 | 120.9 | 2609 |
| `hildegard` (official weights) | 71.1 | 6.59 | 7046 | 11920 |

Plot (full set only): `raw_results/flux_upscale_benchmark_axis_v1_1k/latency_vs_patch_fid.png`. Pareto front of latency
vs patch_fid: `flow_upscaler_tiny_vae` (16.1 ms, 4.45), `ours_distilled_1_step_20m_tiny_vae` (18.8, 3.81),
`..._200m_tiny_vae` (59.8, 3.22), `..._20m` (131.1, 2.20), `..._200m` (172.1, 1.75), `ours_base_30_steps_200m`
(1671, 1.72).

## Conclusions

1. **Ours is the best at every latency above 17 ms.** The 200M student reaches patch FID 1.75 (base 1.72), -78% vs
   bicubic (8.12) and -59% vs the best competitor, the released FlowUpscaler (4.28). Only FlowUpscaler with the tiny VAE
   is faster (16.1 ms vs 18.8 ms) and it is clearly worse (4.45 vs 3.81). Compared with Wan 2.1 (base 4.31, -70% vs
   bicubic) the FLUX.2 models are much closer to the VAE oracle (1.72 vs 0.71; Wan 4.31 vs 1.30 on another set).
2. **Distillation is free again**: 1 step 200M is +2% patch FID / +4% fid vs the 30 step base at 9.7x lower latency,
   equal roundtrip errors. The 20M student costs +26% patch FID (Wan: +34%).
3. **The decoder dominates latency.** The full FLUX.2 decode at 1024 is ~110 ms (20M: 131 ms full vs 19 ms tiny); one
   200M forward is ~40 ms eager. TAEF2 is a much better tiny decoder than the Wan one: +1.5 / +1.6 patch FID (+84% /
   +73%) for the 200M / 20M students, and with it the model size still matters (3.22 vs 3.81; on Wan 10.4 vs 11.6 was
   noise). The 20M + TAEF2 pipeline (18.8 ms, 690 MB, 3.81) beats every third party solution at any latency.
4. **FlowUpscaler (our previous project) is the strongest third party solution** (4.28, 4.45 with TAEF2): the new
   architecture and recipe halve its patch FID and roundtrip error (MSE 12.5 vs 6.5e-4, ΔE00 2.41 vs 1.82).
5. **SesquiLSR (FLUX.2) is usable this time** (7.71, better than bicubic 8.12 and Real-ESRGAN 9.03; on Wan 29), but it is
   built as a pre-step for a refinement pass and stays far behind generative latent upscaling. Plain latent bicubic is
   unusable (114).
6. **Pixel space SR after the 512 decode** helps little: SPAN 6.18 (best non-latent, 30 ms), Real-ESRGAN 9.03 (worse than
   bicubic on patch FID). Bicubic has the best whole image fid (1.77) and roundtrip MSE (5.0e-4): `fid` does not see
   missing detail, use `patch_fid`.
7. **VOSR does not work on this task** (25.9-33.6 patch FID, 3-4x worse than bicubic, 0.4-0.6 s, 5-9 GB). It is trained
   only for 4x real world restoration at 512 output. With a clean 2x input it treats the remaining fine detail as
   degradation and repaints it (noise, painterly smearing). Verified with the official script on 6 images (PSNR vs
   ground truth: 2x 22.2 / 21.9, 2x tiled at its 512 training size 21.2 / 21.7, 256 -> 4x 22.7 / 22.1, bicubic 26.4,
   1.4B / 0.5B): tiling makes it worse, 4x looks cleaner but invents more texture from 4x fewer pixels (full set patch
   FID 28.3 / 33.6 vs 25.9 / 29.7 at 2x). Not a wrapper bug: our wrapper and the official script give the same quality.
8. **Hildegard redraws instead of upscaling.** With the official workflow weights (tile 0.9, position 0.8, global 0.8)
   subjects move and change (a lioness gets a tiger face, animals shift and rescale): ΔE00 6.59, MSE 71e-4 on 30
   images, as bad as latent bicubic. With all reference weights at 1.0 it stays aligned and adds sharp, plausible
   detail, but the roundtrip errors are still 1.7x / 3x ours (ΔE00 3.80 vs 2.18, MSE 21.7 vs 7.0e-4) at 40x the latency
   (7.0 s vs 172 ms) and 4x the VRAM (11.9 GB without the text encoder). Its realism might be competitive on patch FID,
   which was not measured. Its strength (prompted, tiled refinement of multi-megapixel images) is outside this task.

## Caveats

- One run, one noise seed per image; differences below ~0.1 patch FID should not be trusted.
- Eager bf16 on one RTX 5090, batch 1, not compiled; VOSR in fp32 (official default), Hildegard fp8 in ComfyUI (3
  warmup samples). Hildegard latency excludes the one-off prompt encoding with Qwen3-8B (constant prompt).
- Hildegard and VOSR need a prompt / were trained for another factor: they are used at their documented settings plus
  one ablation each, not tuned further. Hildegard numbers are from 30 images only.
- SesquiLSR was updated on 2026-11-21 (new FLUX.2 V1 architecture and new Wan 2.1 weights): the Wan benchmark row
  (2026-11-20) used the previous weights.
- No human judgement; small text and faces were not inspected separately.
