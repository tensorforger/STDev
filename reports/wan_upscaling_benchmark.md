# Wan 2.1 upscaler: unified benchmark against other 2x upscaling solutions (2026-11-20, RTX 5090)

Code: `scripts/run_upscale_benchmark.py` (prerequisites and download commands at the top of the file),
`src/stdev/upscale_benchmark/` (dataset, metrics, solution wrappers, runner), plot `scripts/plot_upscale_benchmark.py`.
Results: `raw_results/upscale_benchmark_axis_v1_1k/` (`results.json`, `<solution>/results.json`, all 1024x1024
predictions as PNG, `latency_vs_patch_fid.png`).

## Setup

- Task for every solution: the normalized Wan 2.1 latent of a 512x512 image (`[1, 16, 1, 64, 64]`, nothing else: no
  prompt, no hff) in, the decoded 1024x1024 image out. Upscaling before or after decoding is an implementation detail.
- Data: 1000 images of Axis-V1 (`refine-axis/axis-v1_1k`, high resolution photos, 8 categories), center cropped to
  1024x1024 (`crop_maximal_rectangle`) = ground truth; bilinear downsampled to 512x512 and encoded with the full Wan 2.1
  VAE = input. A different and much sharper image domain than the ReLAION `wan_eval_1000` of the earlier reports:
  patch FID values are not comparable between the two.
- Metrics: `fid` (whole image, resized to 299), `patch_fid` (native 256 px tiles, the primary metric), `low_res_mse` and
  `chroma ΔE00` (roundtrip: prediction bilinear downsampled to 512 and compared with the initial 512 image, MSE on
  [0...1] pixels and mean CIEDE2000), latency (batch 1, 10 warmup passes on fake latents, then synchronized wall time of
  every real sample, decode included), peak VRAM (peak allocated CUDA memory above the level before loading, weights
  included).
- Our models: null hff (the model decides the sharpness itself), guidance / CFG 1, bf16, eager (no `torch.compile`).
  Base = `wan_pretrain_v1` step 161590, 20 Euler steps, shift 4, noise 1.0. Students: `wan_distill_v1` (222M),
  `wan_small_s20m` (19.5M), `wan_small_s2m` (2.3M), 1 step. `_gt_hff`: the same with the Immerkaer hff of the ground
  truth image as input (privileged), same noise seed per image.
- Third party: `decode_bicubic` (full VAE decode at 512, bicubic 2x in pixel space), `spacepxl_wan21_vae_upscale2x`
  (decoder with integrated 2x, pixel shuffle), `latent_bicubic_decode` (bicubic latent stretch, full decode),
  `wan_nn_latent_upscaler` (ComfyUI-VAE-Utils neural latent upscaler, raw latents), `sesquilsr` (SesquiLSR Wan 2.1, 2.0x,
  bf16), `decode_span` (SPAN 2xNomosUni_span_multijpg after the full decode), `decode_realesrgan` (RealESRGAN_x2plus after
  the full decode). Last row of ours: 200M student, decode with the spacepxl decoder (2048 px), bilinear 2x downscale.
- Single run, one noise seed per image (`seed + index`).

## Results (sorted by patch_fid)

| solution | patch_fid | fid | low_res_mse 1e-4 | chroma ΔE00 | latency ms (p95) | peak VRAM MB |
|---|---|---|---|---|---|---|
| `ours_base_20_steps_gt_hff` | 4.24 | 3.11 | 16.5 | 2.37 | 1050.4 (1056.7) | 4970 |
| `ours_distilled_1_step_200m_gt_hff` | 4.29 | 3.17 | 16.5 | 2.38 | 180.8 (192.8) | 4941 |
| `ours_base_20_steps` | 4.31 | 3.01 | 16.4 | 2.35 | 1051.3 (1057.9) | 4970 |
| `ours_distilled_1_step_200m` | 4.36 | 3.07 | 16.4 | 2.36 | 178.9 (182.6) | 4941 |
| `ours_distilled_1_step_200m_spacepxl_decoder_downscaled` | 4.51 | 4.10 | 18.9 | 2.58 | 198.9 (204.7) | 4971 |
| `ours_distilled_1_step_20m` | 5.85 | 3.85 | 17.3 | 2.45 | 140.8 (144.7) | 4514 |
| `ours_distilled_1_step_20m_gt_hff` | 5.97 | 3.95 | 17.4 | 2.47 | 148.5 (159.2) | 4514 |
| `spacepxl_wan21_vae_upscale2x` | 6.33 | 5.80 | 19.9 | 2.75 | 29.3 (30.7) | 1304 |
| `ours_distilled_1_step_20m_tiny_vae_gt_hff` | 10.40 | 7.03 | 20.6 | 2.75 | 27.0 (29.2) | 2271 |
| `ours_distilled_1_step_20m_tiny_vae` | 10.43 | 6.43 | 20.3 | 2.71 | 26.5 (28.0) | 2271 |
| `decode_span` | 11.19 | 4.83 | 18.5 | 3.16 | 34.6 (36.8) | 1309 |
| `ours_distilled_1_step_2m_tiny_vae` | 11.60 | 7.45 | 21.2 | 2.77 | 21.9 (23.1) | 2238 |
| `ours_distilled_1_step_2m_tiny_vae_gt_hff` | 11.62 | 8.02 | 21.5 | 2.81 | 22.9 (24.1) | 2238 |
| `decode_realesrgan` | 13.83 | 8.83 | 18.0 | 2.71 | 57.4 (60.2) | 1336 |
| `decode_bicubic` | 14.13 | 3.33 | 13.7 | 2.37 | 29.8 (32.1) | 1304 |
| `sesquilsr` | 29.23 | 7.65 | 19.3 | 2.74 | 136.1 (143.2) | 4482 |
| `wan_nn_latent_upscaler` | 51.07 | 31.20 | 33.7 | 3.68 | 134.4 (138.1) | 4486 |
| `latent_bicubic_decode` | 144.99 | 88.54 | 74.5 | 4.85 | 133.1 (137.9) | 4475 |

Pareto front of latency vs patch_fid (without the privileged `_gt_hff` rows): `ours_distilled_1_step_2m_tiny_vae`
(21.9 ms, 11.60), `..._20m_tiny_vae` (26.5, 10.43), `spacepxl_wan21_vae_upscale2x` (29.3, 6.33),
`..._20m` (140.8, 5.85), `..._200m` (178.9, 4.36), `ours_base_20_steps` (1051, 4.31).

## Conclusions

1. **Quality: ours is clearly the best at 140 ms and above.** Base 4.31, 200M student 4.36, 20M student 5.85, versus
   14.13 for bicubic, 11.19 SPAN, 13.83 Real-ESRGAN, 6.33 spacepxl, 29 SesquiLSR. Relative to bicubic our patch FID is
   -70% (base, 200M) and -59% (20M); on the ReLAION set it was -50% / -31%: the sharper the images, the more bicubic
   loses and the larger the gain. Whole image `fid` is much less discriminative (bicubic 3.33 beats the 20M student at
   3.85 and is close to the base at 3.01): it does not see missing detail, use `patch_fid`.
2. **Distillation is almost free**: 1 step (200M) is +1% patch FID and +2% fid over the 20 step base, same roundtrip
   errors, 5.9x faster. The 20M student costs +34% patch FID versus the 200M (full VAE decode in both).
3. **Latency is dominated by the decoder, not by the upscaler.** The full VAE decode of a 128x128 latent is ~115 ms
   slower than the tiny decoder (20M student: 140.8 ms with the full VAE vs 26.5 ms with the tiny one, same model), i.e.
   ~80% of the 20M pipeline. One 222M forward pass is ~45 ms in eager mode (200M: 179 ms, base: 20 passes = 1051 ms).
   Peak VRAM follows the decoder as well: ~4.5-5.0 GB with the full decode at 1024, ~2.2 GB with the tiny one, 1.3 GB for
   everything that decodes at 512. Compiling the model (30 ms / 4.6 ms for 200M / 20M in the earlier reports) would not
   change this.
4. **The tiny VAE decoder limits quality below ~140 ms.** With it the 20M student's patch FID rises from 5.85 to 10.43
   (+78%) and the model size hardly matters any more (2M: 11.60, 20M: 10.43), and the roundtrip errors are the
   worst of our models (ΔE00 2.7-2.8, MSE 20-21e-4 vs 16.4e-4). The tiny VAE rows still beat bicubic (14.13) and
   Real-ESRGAN (13.83); the 20M one also beats SPAN (11.19, 34.6 ms), the 2M one does not.
5. **spacepxl's decoder is the strongest competitor at low latency**: patch FID 6.33 at 29 ms and 1.3 GB, 4.8x faster than
   our 20M student for +8% patch FID. Its integrated 2x decoder runs at the cost of the original 512 decode. Its fid
   (5.80), MSE (19.9e-4) and ΔE00 (2.75) are worse than ours and the bicubic floor, in line with the author's note
   that it is tuned for perceived sharpness and may shift colors. Patch FID rewards that sharpness; these reference based
   metrics do not measure the "wan speckle" fix it was made for. There is no point of ours between 30 and 140 ms: that
   gap (5.9-6.3 patch FID at ~30-60 ms) is where a fast decoder for our latents would pay off.
6. **Chaining ours with the spacepxl decoder does not help** (decode to 2048 px, bilinear 2x downscale): patch FID 4.51 vs
   4.36, fid 4.10 vs 3.07, MSE 18.9e-4 vs 16.4e-4, ΔE00 2.58 vs 2.36, +20 ms. The decoder's own sharpening and color shift
   survives the downscale and adds error to our latents. spacepxl's claim (downscaled output beats the original decoder)
   was not reproduced for our latents with these metrics. The spacepxl decoder runs at the original latent resolution
   and outputs 12 channels pixel shuffled to 2x, so decoding a 128x128 latent costs the same as the full decode at 1024.
7. **Ground truth hff does not help at guidance 1.** Patch FID changes by -0.07 (base), -0.07 (200M), +0.12 (20M),
   -0.03 (20M tiny), +0.02 (2M tiny); whole image fid gets worse by 0.1-0.6 and ΔE00 by 0.02-0.04 everywhere. Differences
   of this size are within what one seed can produce. With the null embedding the models already pick the sharpness
   themselves, so no hff is needed at inference. Consistent with the known small hff effect on the velocity (1-4%); CFG
   with large scales was not tested here.
8. **Third party latent upscalers are not usable without a refinement pass**: latent bicubic 145, NN latent upscaler 51,
   SesquiLSR 29 patch FID, all worse than bicubic in pixel space, with roundtrip MSE 19-75e-4. They are built as a
   pre-step for a hires-fix denoise pass (SesquiLSR's own README), which the benchmark forbids (no prompt, single
   tensor). Our generative one step upscalers produce directly decodable latents, which is the point of the project.
9. **Roundtrip floor**: `decode_bicubic` (MSE 13.7e-4, ΔE00 2.37) is the VAE reconstruction error itself. Ours reaches
   ΔE00 2.35-2.36 (equal) and MSE 16.4e-4 (+20%): the added detail changes the low resolution content only slightly.
   Pixel space SPAN has the worst color shift of the image upscalers (ΔE00 3.16).

## Caveats

- One run and one seed per image, no confidence intervals; differences below ~0.1-0.2 patch FID should not be trusted
  (training seed noise of earlier runs was 0.1-0.3).
- Latency is eager bf16 on an RTX 5090, not compiled, batch 1. Peak VRAM includes the weights, and the Wan VAE wrapper
  keeps the (unused) encoder too (every full VAE solution carries ~240 MB of VAE weights).
- SPAN is a community x2 model (jpg degradation + multiscale training), Real-ESRGAN x2plus a real world restoration model:
  neither was trained for VAE decoder artifacts. spacepxl's model is image-only v1.
- No perceptual or human judgement; small text and faces were not inspected separately for this benchmark.

## Next steps suggested by the results

1. A fast decoder for our latents (decoder fine-tune on 128x128 latents generated by our students, tiny or with an
   integrated 2x like spacepxl's) is worth more than a better upscaler: it attacks the dominant latency term and the tiny
   VAE quality gap (10.4 vs 5.9).
2. Fill the 30-140 ms gap of the Pareto front with a student + faster decoder instead of further upscaler distillation.
3. Drop hff from the inference interface of the 1-step students (null hff is as good).
