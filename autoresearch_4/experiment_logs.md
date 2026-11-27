# Experiment logs (autoresearch 4: Qwen-Image 2.1, 512 -> 1024)

## Status

- DONE (2026-11-27): rounds 0-7, 31 runs (0000-0030, incl. PTQ 0024-0026 and the reference 0016). The Qwen-Image 2.1
  VAE is marked PROBLEMATIC for a latent upscaler: decoded quality plateaus at patch_fid ~5.8-6.0 (bicubic 8.69, but fid
  4.4 vs bicubic 2.21 and worse colour) and does not respond to model size or training time. No pretrain recipe; the
  next pretrain uses another VAE. See FINAL SUMMARY at the end.

## Key findings (keep updated)

- (round 7) Scaling does not break the latent only plateau: 1 h runs at 48M / 97M / 170M / 273M give 6.01 / 5.75 /
  5.96 / 6.95 (sampler 30 / 4) vs 6.01 at 30 min / 48M. Flow loss and latent mse keep improving (0.141 -> 0.121), the
  decoded quality does not: the decoder's off-manifold fragility, not capacity or training time, limits this VAE.
- (round 0) Q1: Qwen-Image 2.1 latents are smooth per token like Wan 2.1 (lag-1 autocorrelation 0.89, high frequency
  energy 7%), but the bilinear source explains only 61% of the target variance (Wan 83%, FLUX.2 41%): each 32x32
  source token covers 32x32 px, so the missing detail is a large, smooth residual (std 0.67 vs target 1.07, residual
  lag-1 0.80 vs FLUX.2 0.39). The decoder is the most noise sensitive of the three: 10% relative white latent noise
  costs 29.7 dB PSNR (Wan 33.3, FLUX.2 40.2), but smooth (4x low-pass) noise only 35.6 dB. The sensitivity is to
  token level (high latent frequency) errors, not to off-manifold channel directions (top / bottom PCA halves equal).
- (round 0) The 512 px latents are offset from the 1024 px latents of the same image: per channel residual mean
  -0.57 ... +0.76 at residual std 0.45-0.78. The AR3 residual parameterization only rescales (run_0002 tests centering).
- (round 0) Q3 profiling (w384 d12 B16, compiled): bf16 126 samples/s and compute bound (B32 same samples/s, 30 GB).
  GEMMs are only 36% of the step (fused elementwise / norms 35%, attention 12%, depthwise conv 8%). Raw GEMMs are
  2-2.5x faster in FP8 / MXFP8 (sm_120 native `_scaled_mm`), but torchao Float8Linear gives only +1% (tensorwise; rowwise
  -14%) end to end, at width 768 +5%; memory -20%. MXFP8 / NVFP4 training recipes of torchao 0.18 fail to import with
  this Triton build; torchao NVFP4 linear falls back to a slow path.
- (round 1) ALL round 1 models are only at the pixel bicubic level (patch_fid 8.2-8.7 vs 8.69; fid 5.0-5.4 vs 2.21;
  low_res_mse 16e-4 vs 8e-4), unlike FLUX.2 (AR3: 2.45 vs bicubic 8.12 at the same budget). Cause (error
  decomposition, run_0002, 48 images): the high frequency part (above half Nyquist) of the latent error holds only 24%
  of its energy but makes 92% of the low res error and most of the colour error after decoding (gt + high band error:
  low_res_mse 16.9e-4, dE00 2.87; gt + low band error: 8.8e-4, 2.19; prediction 18.2e-4, 2.97). The Qwen decoder turns
  token level latent mismatches into colour blotches; the flat flow MSE spends capacity on the harmless low band.
- (after round 6) Q4 PTQ of 0008 (block linears only, same session latency): FP8 dynamic w+a (0024) 6.07 / 4.49 at
  2.27 ms (-22%), int8 dynamic (0025) 6.05 at 3.56 ms (slower), int8 weight-only (0026) 6.02 at 104 ms (slow fallback
  path). Quantization is lossless within noise even on this decoder sensitive latent space; FP8 is the only speedup.
  Peak VRAM rises (188 vs 137 MB, activation buffers); weights 95 -> 72 MB (the model is small).
- (after round 6) The decoder loss implementation is correct (dev/autoresearch_4/decoder_loss_gradient_check.py):
  bf16 vs fp32 gradient cosine 0.997, a small step along -grad lowers the loss by 9-11%, the gradient covers exactly the
  4 crops. But 10x that step raises the loss by 50%: the decoder loss landscape is very sharp around the data (the same
  off-manifold fragility), a poor training signal. Rounds 4 / 6 are genuine negatives.
- (round 6) Decoder aware losses stay negative in every form tried: restricted to sigma < 0.3 on the delta flow
  (0020 weight 300: 7.22, 0021 weight 1000: 9.44 vs 0017 4.71 at the same sampler; colour not improved, low_res_mse
  13.1 / 13.8 vs 12.6) and even on the deterministic base regression (0022: 42.8 / low_res_mse 22.5 vs 0019 13.2 / 9.9),
  where it should be valid. The decoder costs ~35% of the steps, but the losses are worse, not just slower. The 0022
  result is suspicious enough to verify the implementation (gradient check) before concluding.
- (round 6) VAE free factorization over the frozen learned base (0023): 6.96 / fid 4.78 / low_res_mse 14.9, worse than
  the exact base (4.71) and no better than the unfactorized 0008 (6.01 at 30 / 2); the base forward costs ~45% of the
  steps (9.1k vs 16.8k). The 0019 base is not yet good enough to replace the VAE round trip.
- (round 5) FACTORIZATION BREAKS THE PLATEAU: flow on the delta over the pixel re-encode base E(bicubic(D(source)))
  (0017) 4.93 / fid 3.65 vs 6.01 / 4.48 (0008); sampler 30 / 4 / 1.0 4.71. Most of it is information: the base as an
  extra input to the old residual model (0018) 5.22. The base alone (0016) 9.10 / fid 2.46 / low_res_mse 7.9 / dE
  1.83 behaves like pixel bicubic. Remaining problem: adding the generated delta degrades colour (low_res_mse 7.9 ->
  13.2): the generated detail is still off-manifold. Lower temperature trades detail for colour (T 0.6: 7.39 / 10.2).
- (round 5) The deterministic factor is learnable: a single forward regression of the base (0019) reaches low_res_mse
  9.9 / dE 2.06 (exact base 7.9 / 1.83), patch_fid 13.2 (blurry by construction). Inference without the VAE round trip
  is possible.
- (round 4) A decoder-in-the-loop loss on x0 predictions FAILS badly (patch_fid 57-101 vs 6.0): at mid / high sigma
  x0 = x_t - sigma v is the posterior mean, and forcing its decode to match one ground truth biases the velocity field;
  the sampled outputs drift off the data distribution. A decoder term is only valid at very low sigma or on samples.
- (after round 4) Latent geometry (dev/autoresearch_4/decoder_gain_analysis.py, error_by_patch_pca.py,
  manifold_completion.py; 2x2 token patches = 256 dims): the data covariance spans 4 orders of magnitude (eigenvalues
  30 ... 0.004, half the directions below 0.07), while the decoder's linear gain is nearly flat (2-10x, mildly higher
  for low variance directions; corr(log gain, log variance) -0.86 full res, +0.24 at 8x pooling). The model is worse
  than the mean in the low variance directions (error / variance 1.2 in the bottom 96, 0.18 in the top 32), and decoder
  damage is non-additive: the bottom 96 error (7% of the error energy) alone gives low_res_mse 10.8e-4 vs 12.7e-4 for
  the whole error. Shrinking those coefficients to the mean hurts (bottom 128 zeroed: 20.8e-4): they hold real,
  context dependent (non-linear) structure. The local manifold is curved: no linear / post-hoc projection fixes it.
- (round 3) PLATEAU at patch_fid ~6.0, independent of compute and data seen: 0008 (compound) 6.01, internal 128 x 128
  sub-token grid (0009, 4x tokens) 6.07, width 768 / 159M (0010) 6.00, high frequency weight x64 (0011) 5.97.
  Capacity is not the limit.
- (round 3) Why (dev/autoresearch_4/high_band_swap.py, linear_base.py, pixel_reencode_base.py): a Qwen token covers
  16x16 px, so latent frequencies above half Nyquist are 32-64 px pixel wavelengths: coarse content fully determined by
  the 512 input, not generative detail. Ground truth low band alone decodes to low_res_mse 81e-4; with the high band of
  another image 134e-4, with its own high band shifted by one token 153e-4. And latent L2 is the wrong yardstick:
  encode(bicubic(decode(source))) is FURTHER from the target (latent mse 0.217) than the models (0.14) or a closed form
  linear 3x3 unpatchify base (0.17), yet decodes best (low_res_mse 9.6e-4 vs 15e-4 / 34e-4). The decoder punishes
  off-manifold latents with colour errors; on-manifold latents decode well even far away in L2. Consequence: the loss
  must see the decoder (round 4).
- (round 2) Frequency weighted latent loss works: velocity error above half Nyquist weighted x4 (0004) 6.63, x16 (0005) 6.37 vs
  8.18 (fid 4.52 vs 5.04): the biggest gain so far, but low_res_mse barely moves (14.9 vs 15.8e-4); all models are
  still far from FLUX.2 level. Hypothesis: compute starved, a Qwen token covers 16x16 px, so the 50M model at 64x64
  tokens spends 4x less compute per image than the AR3 FLUX.2 model (round 3: finer internal grid vs width).
- (round 2) RoPE + mask-free shifted windows (0006) 7.95 vs 8.18, +5% steps: adopted (better and faster). Depthwise
  conv on the block width (0007) 8.99: worse by 0.8 despite +9% steps, rejected (the conv on the 4D hidden matters).
- (round 1) Mean centered residual (0002) 8.18 vs 8.63: the 512 / 1024 latent offset matters. Q2: direct 8.71 vs
  residual 8.63 (fid 5.41 vs 5.19): residual slightly better, small as for FLUX.2. Q3: FP8 tensorwise training 8.55 =
  bf16 quality (+2.5% steps).
- (round 1) Sampler (0002, 200 images, relative only): temperature < 1 lowers latent / roundtrip error but raises
  patch_fid (T 0 = regression: 28.2 vs 19.2); shift 4 / 10 steps slightly better than 30 / 2; 60 steps worse.
- (round 1) Q3 / speed profiling: whole model torch.compile +7%, with max-autotune-no-cudagraphs +14% (144 vs 126
  samples/s); RoPE + mask-free shifted windows (flash path) +5%; depthwise conv on the block width (not the 4D hidden)
  +10% and -11% memory. Stable torch 2.14.1 (cu130) is 21% slower than the nightly on this GPU (bf16 100 samples/s).
  torchao MXFP8 / NVFP4 training needs bf16 2D inputs (NVFP4: M, K, N divisible by 128): usage, not version issues.
- (round 0) Texture-fix decoder oracle patch_fid 1.07 vs official 2.44 (fid 0.55 / 0.60): the benchmark decoder matters.

## External knowledge: the Qwen-Image 2.1 VAE is Qwen-Image-VAE-2.0 f16c64

The diffusers config (encoder base 96, decoder base 144, dim_mult (1, 2, 4, 8, 8), residual AvgDown3D / DupUp3D
shortcuts) matches the f16c64 variant of the Qwen-Image-VAE-2.0 technical report (arXiv 2605.13565, 76M encoder / 248M
decoder). Report facts that explain this phase's measurements:
- No KL term and no latent noise augmentation: the decoder only saw clean encoder outputs (-> extreme off-manifold
  fragility, round 0 / 3 / 4 measurements).
- Latents aligned to a middle layer of DINOv2-L (learnable linear projection; margin cosine + pairwise distance-matrix
  losses; strict margins early, loosened later) for diffusability: tokens are partly semantic, non-linear functions of
  their context (-> bilinear / linear bases fail, one-token shifts are catastrophic, re-encoding works).
- Global skip connections: space-to-channel folding of the input (pixel unshuffle + channel averaging) bypassing the
  first downsampling: part of each token is a folded pixel code of its 16x16 block (-> coarse colour lives in the
  channels and in the "high" latent band).
- No GAN (L1 + LPIPS only): the texture-fix adversarial decoder finetune matters (oracle 1.07 vs 2.44 patch_fid).
- Qwen-Image 2.1 itself: 7B DiT, 32 single-stream layers, patch size 1 on the latent grid, flow matching with dynamic
  shift, 40 steps (Turbo 8), native 2K. Sources: github.com/QwenLM/Qwen-Image-2.1, arxiv.org/abs/2605.13565.
Consequence: the deterministic part of a 2x upscale includes the encoder's semantic code; a small upscaler should not
learn it inside the flow. Factorize: deterministic on-manifold base (re-encode, or a distilled network) + generated delta.

## Prior knowledge (carried over, see autoresearch_3 for details)

- Recipe (AR3 run_0016 / 0018, FLUX.2): plain window transformer w384 d12 (~48M) on the full resolution latent grid,
  shifted 8x8 windows + relative bias, adaLN-zero, QK-norm, condition RMS norm, GELU MLP + depthwise conv, learned
  source path (2 res blocks + pixel shuffle); flow on the per channel normalized residual over the bilinear source;
  AdamW (0.9, 0.95), wd 0.01, LR 1.7e-3 at B4 (128x128 grid), warmup 1000, linear decay over the last 50%; logit-normal
  sigmas, shift 1; sampler Euler 30 steps / shift 2 / temperature 1.0.
- Residual vs direct: -3 patch FID on Wan (smooth latents), -0.3 on FLUX.2. Temperature < 1 hurt FLUX.2, helped Wan.
- Seed noise at 30 min ~0.03 patch FID (AR3), 0.1-0.3 (AR2). Latency differences < ~20% between rounds are noise.

## Round 0: preparation (no training)

### Data and benchmark

- 4KLSDB `train_x4` files 0-64 (40885 images, first 40000 sorted used), 1024 crops (`crop_maximal_rectangle`), bilinear
  512 inputs, Qwen-Image 2.1 VAE bf16, opaque RGBA. Latents stored as fp16 (`cast_fp16`, 640 KB per sample, 26 GB).
- Benchmark: Axis-V1 1k, same crops as AR3 (`axis_v1_1k_qwen_2x`), decoded with the texture-fix decoder.
- The texture-fix VAE (madebyollin) has rescaled weights in both encoder and decoder, but its encoder is functionally
  identical to the official one (latent difference exactly 0 on 8 Axis images): the latent space is shared. Its decoder
  roundtrip PSNR is slightly lower (32.3 vs 32.8 dB on 8 Axis images) as expected for an adversarial finetune.
- The Qwen decoder at 1024 needs ~6.7 GB per image in bf16: the benchmark decodes one image at a time.

### Latent analysis (dev/autoresearch_4/latent_analysis.py, 256 4KLSDB images, 24 decoded; dev/autoresearch_4/latent_analysis.json)

| | Qwen-Image 2.1 | FLUX.2 | Wan 2.1 |
|---|---|---|---|
| latent grid at 1024 px | 64 x 64 x 64 ch (16x) | 128 x 128 x 32 ch (8x) | 128 x 128 x 16 ch (8x) |
| target std / residual std | 1.07 / 0.67 | 0.96 / 0.74 | 0.50 / 0.20 |
| corr(target, bilinear source) | 0.77 | 0.67 | 0.88 |
| variance explained by source | 61% | 41% | 83% |
| lag-1 autocorrelation target / residual | 0.89 / 0.80 | 0.62 / 0.39 | 0.89 / 0.57 |
| high frequency energy target / residual | 7% / 14% | 29% / 48% | 9% / 35% |
| PCA components for 90 / 99% (participation ratio) | 34 / 58 (19.3) | 24 / 31 (19.4) | 6 / 13 (3.8) |
| roundtrip PSNR vs image | 33.9 | 35.4 | 32.7 |
| decode(bilinear 2x latents) PSNR vs clean decode | 19.7 | 21.3 | 22.3 |
| decoder PSNR, white noise 2 / 5 / 10 / 20% | 40.8 / 34.3 / 29.7 / 24.7 | 53.1 / 45.9 / 40.2 / 34.4 | 45.5 / 38.3 / 33.3 / 28.6 |
| decoder PSNR, 10% noise in PCA top / bottom half | 29.8 / 29.6 | 40.2 / 40.3 | 34.5 / 32.6 |
| decoder PSNR, 10% smooth noise (4x low-pass) | 35.6 | 36.1 | 35.3 |
| storage PSNR fp16 / bf16 / lbrq_8bit | 61.3 / 55.3 / 52.4 | 63.1 / 62.2 / 60.1 | 60.7 / 57.2 / 56.6 |

Noise is relative to the mean channel std of each VAE. Notes:
- Qwen latents are smooth per token, but a token covers 16x16 px: the residual is smooth too (lag-1 0.80) and large.
  Expectation: the residual parameterization helps (61% of the variance is free), less than on Wan.
- Decoder noise sensitivity is a token-frequency effect: white noise hurts Qwen 10 dB more than FLUX.2, smooth noise
  the same. Upscaler errors at the token level (the last sampling steps, grid artifacts) will be amplified. Ideas:
  sampler steps / shift for the final low noise steps, loss weighting at low sigma, frequency weighted loss.
- Storage: even the 8 bit codec would have been fine (52 dB); fp16 is used anyway.

### References (raw_results/qwen/reference_*, texture-fix decoder unless noted)

| reference | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 |
|---|---|---|---|---|
| oracle (VAE roundtrip of the GT) | 1.07 | 0.55 | 2.0 | 1.07 |
| oracle, official decoder | 2.44 | 0.60 | 2.0 | 1.05 |
| pixel bicubic | 8.69 | 2.21 | 8.0 | 1.85 |
| latent bilinear | 226.3 | 192.7 | 131 | 6.84 |

### Precision profiling (dev/autoresearch_4/profile_training.py, profile_breakdown.py, gemm_microbenchmark.py)

| model / precision | samples/s | peak GB |
|---|---|---|
| w384 d12 bf16 B16 | 126 | 15.4 |
| w384 d12 bf16 B32 | 128 | 30.0 |
| w384 d12 fp8 tensorwise B16 | 128 | 12.0 |
| w384 d12 fp8 rowwise B16 | 109 | 12.0 |
| w768 d16 (197M) bf16 B4 | 35.6 | 12.9 |
| w768 d16 (197M) fp8 tensorwise B4 | 37.5 | 10.5 |

Raw GEMM throughput (TFLOPS, [65536 x 384] x [384 x 1536]): bf16 204, fp8 tensorwise 430, fp8 rowwise 381, mxfp8 472.

### Infrastructure notes

- Training block loading costs 0.5 ms single threaded but ~110 ms with 192 torch threads (oversubscription): scripts that
  read blocks in the main process must call torch.set_num_threads(1); DataLoader workers already do.
- 30 min at B16 / 7.9 steps/s is ~14k steps, ~5.7 epochs of the 40k samples.
- Tools: dev/autoresearch_4/sampler_sweep.py (re-evaluate weights with other sampler settings),
  dev/autoresearch_4/compiled_check.py (20 s compiled training + compile / upscale: `--smoke` skips compilation and
  missed a wrong compile shape in the first round 1 launch; run this check before every round).

## Round 1: calibration of the AR3 recipe (key questions 2, 3)

Base run_0000: AR3 recipe ported (64 channels, 64 x 64 grid), B16 LR 1.7e-3, sampler 30 / 2 / 1.0, SEED 0, clean data.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0000 | AR3 recipe, normalized residual | AR3 0018 | baseline |
| 0001 | 0000 with direct flow | 0000 | Q2 residual vs direct |
| 0002 | 0000 with mean centered residual | 0000 | calibration: the 512 / 1024 latent offset |
| 0003 | 0000 trained with FP8 tensorwise block linears (bf16 inference) | 0000 | Q3 quality at FP8 |

Launched: ./qwen_infra/start_round.sh 0000 0001 0002 0003

### Round 1 results (sampler 30 / 2 / 1.0)

| run | change | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | lat p95 ms | VRAM MB | steps | final loss |
|---|---|---|---|---|---|---|---|---|---|
| 0000 | AR3 recipe, residual | 8.63 | 5.19 | 16.0 | 2.53 | 3.0 | 137 | 14.0k | 0.378 |
| 0001 | direct | 8.71 | 5.41 | 15.6 | 2.51 | 4.5 | 137 | 14.1k | 0.209 (other space) |
| 0002 | mean centered residual | **8.18** | **5.04** | 15.8 | 2.51 | 5.7 | 137 | 14.0k | 0.384 |
| 0003 | FP8 tensorwise training | 8.55 | 5.10 | 16.0 | 2.52 | 47 (outlier) | 137 | 14.3k | 0.379 |

(First launch failed: wrong compile shape, see infrastructure notes.) A ground truth vs 0002 crop shows plausible
detail with colour shifts and blotchy texture (dev/autoresearch_4/r1_compare.png).

Sampler sweep of 0002 on 200 images (FID values are not comparable to the 1000 image benchmark; relative only):

| setting | patch_fid | fid | low_res_mse 1e-4 | dE00 | latent mse |
|---|---|---|---|---|---|
| 30/2/1.0 | 19.24 | 7.10 | 18.0 | 2.90 | 0.144 |
| 30/2/0.7 | 19.05 | 7.87 | 14.8 | 2.66 | 0.111 |
| 30/2/0.4 | 22.84 | 9.08 | 13.5 | 2.55 | 0.092 |
| 30/2/0.0 | 28.17 | 9.97 | 13.2 | 2.53 | 0.083 |
| 30/1/1.0 | 20.17 | 7.29 | 18.7 | 2.95 | 0.146 |
| 30/4/1.0 | 18.44 | 7.14 | 16.9 | 2.82 | 0.140 |
| 60/2/1.0 | 20.31 | 7.35 | 18.8 | 2.97 | 0.147 |
| 10/2/1.0 | 18.69 | 7.63 | 15.9 | 2.74 | 0.132 |

Error decomposition (dev/autoresearch_4/error_decomposition.py, 0002, 48 images):

| decoded latents | low_res_mse 1e-4 | dE00 | latent error energy |
|---|---|---|---|
| ground truth | 2.0 | 1.31 | 0 |
| prediction | 18.2 | 2.97 | 0.157 |
| gt + low band error (<= half Nyquist) | 8.8 | 2.19 | 0.119 |
| gt + high band error | 16.9 | 2.87 | 0.037 |

Speed profiling after round 1 (dev/autoresearch_4/profile_training.py, B16, samples/s): blocks compiled (run default)
126, whole model compiled 135, whole model max-autotune-no-cudagraphs 144, 0006 133, 0007 140 (13.7 vs 15.4 GB);
torch 2.14.1 venv (/workspace/venv_torch214): bf16 100, fp8 tensorwise 110.

## Round 2: decoder aware loss and kernel friendly architecture (base 0002)

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0004 | 0002 + velocity error above half Nyquist weighted x4 | 0002 | fix the token level errors the decoder amplifies |
| 0005 | 0002 + high band weight x16 | 0002 | strength of the weighting |
| 0006 | 0002 + axial RoPE, shifted windows without the mask (flash SDPA) | 0002 | kernel friendly attention (+5% speed) |
| 0007 | 0002 + depthwise conv on the block width instead of the MLP hidden | 0002 | kernel friendly MLP (+10% speed, -11% memory) |

Launched: ./qwen_infra/start_round.sh 0004 0005 0006 0007 (compile setup unchanged for comparability with 0002).

### Round 2 results (sampler 30 / 2 / 1.0)

| run | change vs 0002 | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | lat p95 ms | VRAM MB | steps |
|---|---|---|---|---|---|---|---|---|
| 0002 | reference | 8.18 | 5.04 | 15.8 | 2.51 | 5.7 | 137 | 14.0k |
| 0004 | high frequency loss x4 | 6.63 | 4.62 | 15.1 | 2.45 | 3.4 | 137 | 13.9k |
| 0005 | high frequency loss x16 | **6.37** | **4.52** | 14.9 | 2.43 | 5.7 | 137 | 13.9k |
| 0006 | RoPE + mask-free shifted windows | 7.95 | 4.97 | 15.7 | 2.50 | 2.9 | 137 | 14.8k |
| 0007 | depthwise conv on the block width | 8.99 | 5.27 | 16.1 | 2.54 | 43 (outlier) | 140 | 15.2k |

## Round 3: compound, compute per image (base 0008)

All runs: whole model torch.compile (max-autotune-no-cudagraphs) for training.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0008 | 0002 + high frequency loss x16 + RoPE / mask-free windows | 0005, 0006 | new reference, compounding |
| 0009 | 0008 on an internal 128 x 128 sub-token grid (pixel shuffle in, unshuffle out), ~50M, B4 | 0008 | compute per image via a finer grid (FLUX.2 equivalent tokens) |
| 0010 | 0008 at width 768 (159M), B8 (B16 OOM), LR 1.0e-3 | 0008 | compute per image via parameters |
| 0011 | 0008 with high frequency weight x64 | 0008 | saturation of the loss weighting |

Launched: ./qwen_infra/start_round.sh 0008 0009 0010 0011

### Round 3 results (sampler 30 / 2 / 1.0)

| run | change vs 0008 | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | lat p95 ms | VRAM MB | steps x batch |
|---|---|---|---|---|---|---|---|---|
| 0008 | compound: 0002 + loss x16 + RoPE / mask-free + full compile | 6.01 | 4.48 | 14.8 | 2.42 | 2.9 | 137 | 16.8k x 16 |
| 0009 | internal 128 x 128 sub-token grid | 6.07 | 4.93 | 15.3 | 2.47 | 51 (outlier) | 255 | 17.3k x 4 |
| 0010 | width 768, 159M, B8, LR 1e-3 | 6.00 | 4.43 | 14.8 | 2.41 | 50 (outlier) | 392 | 11.7k x 8 |
| 0011 | high frequency weight x64 | 5.97 | 4.49 | 14.8 | 2.43 | 2.9 | 137 | 16.7k x 16 |

High band swap (0008, 48 images; low / high band split at half Nyquist of the 64 x 64 grid):

| decoded latents | low_res_mse 1e-4 | dE00 |
|---|---|---|
| ground truth | 2.0 | 1.31 |
| gt low band only | 80.8 | 6.07 |
| gt low + prediction high | 16.1 | 2.80 |
| gt low + 0.5 x prediction high | 33.5 | 4.03 |
| gt low + other image high | 134.3 | 8.05 |
| gt low + gt high shifted by one token | 153.5 | 7.97 |
| prediction | 17.3 | 2.88 |

The latent spectrum decays smoothly (no carrier peak at Nyquist).

Deterministic bases (48 images):

| base | latent mse to target | low_res_mse 1e-4 | dE00 |
|---|---|---|---|
| bilinear latents | 0.430 | 102.7 | 6.89 |
| closed form linear 3x3 -> 2x2 unpatchify (4000 blocks, ridge) | 0.171 | 33.7 | 4.23 |
| best model 0008 | 0.141 | ~15 | 2.42 |
| encode(bicubic(decode(source))) | 0.217 | 9.6 | 2.22 |

Decoder crop feasibility (dev/autoresearch_4/decoder_crop_check.py): a crop with a 2 token margin reproduces the full
decode in its centre (mse 2e-4; margin 4: 4e-5). 4 crops of 12 x 12 tokens: forward + backward 60 ms, 3.9 GB.

## Round 4: decoder aware loss (base 0008)

Each step decodes 4 random 12 x 12 token crops of the x0 prediction (x_t - sigma v, mapped to latents) with the frozen
bf16 texture-fix decoder; MSE on the central 8 x 8 tokens (128 px) vs the decoded target, added to the flow loss.
Steps are ~2x slower (4.5 vs 9.3 steps/s).

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0012 | 0008 + decoder loss on 2x downscaled pixels, weight 1000 | 0008 | does a decoder aware loss break the plateau |
| 0013 | 0012 with weight 3000 | 0012 | strength |
| 0014 | 0012 on full resolution pixels | 0012 | supervise detail too |
| 0015 | 0012 with plain velocity MSE (no frequency weighting) | 0012, 0002 | is the latent weighting still needed |

Launched: ./qwen_infra/start_round.sh 0012 0013 0014 0015

### Round 4 results (sampler 30 / 2 / 1.0)

| run | change vs 0008 | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | steps | final decoder loss |
|---|---|---|---|---|---|---|---|
| 0008 | reference | 6.01 | 4.48 | 14.8 | 2.42 | 16.8k | - |
| 0012 | decoder loss, 2x pooled pixels, weight 1000 | 61.2 | 29.3 | 36.1 | 3.95 | 9.6k | 6.5e-3 |
| 0013 | weight 3000 | 88.9 | 50.2 | 45.4 | 4.57 | 9.6k | 7.8e-3 |
| 0014 | full resolution pixels | 56.9 | 30.2 | 35.8 | 3.95 | 9.7k | 9.8e-3 |
| 0015 | 0012 without frequency weighting | 101.3 | 59.1 | 49.4 | 4.85 | 9.6k | 8.2e-3 |

### Latent geometry after round 4

Decoder gain along the 2x2 patch PCA directions (12 Axis crops, eps 0.05, fp32 texture-fix decoder):

| direction (by data variance) | variance | gain full res | gain 2x pooled | gain 8x pooled |
|---|---|---|---|---|
| 0 | 30.2 | 4.9 | 4.1 | 3.1 |
| 15 | 3.3 | 2.8 | 1.8 | 0.80 |
| 63 | 0.37 | 2.8 | 1.5 | 0.54 |
| 127 | 0.071 | 3.7 | 1.8 | 0.48 |
| 191 | 0.030 | 5.3 | 2.1 | 0.45 |
| 255 | 0.0037 | 10.7 | 3.7 | 0.60 |

0008 error by direction group (24 images): data variance / error energy / error per variance: top 32 171.5 / 19.1 /
0.18; 32-96 28.9 / 14.3 / 0.53; 96-160 4.8 / 5.1 / 1.09; bottom 96 2.3 / 2.8 / 1.22. Decoding gt + the error of one
group only: low_res_mse 12.3 / 15.5 / 12.1 / 10.8e-4 (whole error 12.7e-4).

## Round 5: factorization into a deterministic on-manifold base and a generated delta

Deterministic part: base = E(bicubic 2x(D(source))) (texture-fix decoder, official encoder), on the encoder manifold
by construction, decodes to the bicubic image. Generated part: delta = target - base. Training bases precomputed
(scripts/encode_qwen_pixel_base.py, `train_datasets/qwen_4klsdb_40k_pixel_base`, ~16 min on 4 GPUs); inference
recomputes the base with the VAE. Statistics (dev/autoresearch_4/pixel_base_statistics.py, 4000 blocks): delta rms std
0.386 (residual over bilinear 0.613): the base explains ~82% of the target variance (bilinear + mean ~54%); delta
still has a per channel mean (-0.46 ... 0.57), base - bilinear rms std 0.658. Generator: dev/autoresearch_4/make_round_5.py.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0016 | the base alone (no learning) | - | reference: what the deterministic factor gives |
| 0017 | flow on the normalized delta over the base, stem on [noisy, base] + source path | 0008 | factorized generation (upper bound, VAE at inference) |
| 0018 | 0008 (residual over bilinear) + the base as an extra stem input | 0008 | parameterization vs information |
| 0019 | deterministic single forward regression of the base from the source (no noise) | 0008 | can a network learn the deterministic factor |

### Round 5 results (sampler 30 / 2 / 1.0 unless noted)

| run | what | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | lat p95 ms | VRAM MB | latent mse |
|---|---|---|---|---|---|---|---|---|
| 0008 | previous best (plateau) | 6.01 | 4.48 | 14.8 | 2.42 | 2.9 | 137 | 0.141 |
| 0016 | base E(bicubic(D(source))) alone | 9.10 | 2.46 | 7.9 | 1.83 | 97 (whole base) | 3654 | 0.188 |
| 0017 | delta flow over the base | 4.93 | 3.65 | 13.2 | 2.29 | 45 (outlier) | 1438 (VAEs) | 0.126 |
| 0018 | base as extra input, residual over bilinear | 5.22 | 3.69 | 13.3 | 2.30 | 2.9 | 1441 | 0.121 |
| 0019 | regression of the base (single forward) | 13.16 | 5.46 | 9.9 | 2.06 | 2.9 | 137 | 0.171 |

VRAM of 0016-0018 includes the decoder and encoder used for the base. End to end (base + 30 steps + decode) 381-400 ms
vs 300 ms for 0008.

Sampler sweep of 0017 (1000 images; steps / shift / temperature -> patch_fid / fid / low_res_mse 1e-4 / dE00):
30/2/1.0 4.93 / 3.65 / 13.2 / 2.29; 30/3/1.0 4.77 / 3.57 / 12.9 / 2.26; **30/4/1.0 4.71 / 3.54 / 12.6 / 2.23**;
20/2/1.0 4.74 / 3.57 / 12.7 / 2.25; 10/2/1.0 4.91 / 3.58 / 11.7 / 2.17; 30/2/0.9 4.83 / 3.60 / 12.1 / 2.20;
30/2/0.8 5.35 / 3.68 / 11.3 / 2.14; 30/3/0.8 5.53 / 3.69 / 11.1 / 2.12; 30/2/0.6 7.39 / 3.98 / 10.2 / 2.05.

## Round 6: on-manifold delta, VAE free base (generator dev/autoresearch_4/make_round_6.py)

Delta statistics over the learned base (dev/autoresearch_4/learned_base_statistics.py): rms std 0.358 (exact base 0.386).

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0020 | 0017 + decoder loss only for sigma < 0.3 (x0 ~ sample), weight 300, 2x pooled; sampler 30 / 4 | 0017, 0012 | make the generated delta on-manifold without biasing the velocity |
| 0021 | 0020 with weight 1000 | 0020 | strength |
| 0022 | 0019 + decoder loss (deterministic target: valid at all times), weight 1000 | 0019 | tighter learned base |
| 0023 | delta flow over the frozen 0019 learned base (no VAE at inference); sampler 30 / 4 | 0017, 0019 | VAE free factorized pipeline |

Launched: ./qwen_infra/start_round.sh 0020 0021 0022 0023

### Round 6 results (sampler 30 / 4 / 1.0 for the flows; 0017 reference resampled at 30 / 4)

| run | change | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | steps | final decoder loss |
|---|---|---|---|---|---|---|---|
| 0017 | reference (exact base, delta flow) | 4.71 | 3.54 | 12.6 | 2.23 | 16.8k | - |
| 0020 | + decoder loss for sigma < 0.3, weight 300 | 7.22 | 4.03 | 13.1 | 2.35 | 10.8k | 1.26e-3 |
| 0021 | + decoder loss for sigma < 0.3, weight 1000 | 9.44 | 4.41 | 13.8 | 2.42 | 10.7k | 1.24e-3 |
| 0019 | base regression (reference, single forward) | 13.16 | 5.46 | 9.9 | 2.06 | 16.7k | - |
| 0022 | base regression + decoder loss, weight 1000 | 42.83 | 24.15 | 22.5 | 3.15 | 9.7k | 5.82e-3 |
| 0023 | delta flow over the frozen 0019 base (VAE free) | 6.96 | 4.78 | 14.9 | 2.46 | 9.1k | - |

### Post training quantization of 0008 (benchmark only, 0008 re-benchmarked in the same session)

| run | block linears | patch_fid | fid | low_res_mse 1e-4 | dE00 | lat p95 ms | peak VRAM MB | weights MB |
|---|---|---|---|---|---|---|---|---|
| 0008 | bf16 | 6.01 | 4.48 | 14.8 | 2.42 | 2.90 | 137 | 95 |
| 0024 | FP8 dynamic activations + weights (per row) | 6.07 | 4.49 | 14.8 | 2.43 | 2.27 | 188 | 72 |
| 0025 | int8 dynamic activations + weights | 6.05 | 4.50 | 14.8 | 2.42 | 3.56 | 158 | 72 |
| 0026 | int8 weight-only | 6.02 | 4.48 | 14.8 | 2.42 | 104 | 146 | 72 |

## Round 7: 1 h scaling ladder, latent only (generator dev/autoresearch_4/make_round_7.py)

Base: AR3 FLUX.2 recipe (relative bias, masked shifted windows) + per channel shift / scale residual normalization +
frequency weighted loss x16, training and sampling shift 3 (Qwen-like noise emphasis), 30 steps, whole model compile.

| Run ID | width x depth | params | batch / LR | note |
|---|---|---|---|---|
| 0027 | 384 x 12 | 48M | B16 / 1.7e-3 | |
| 0028 | 512 x 16 | 97M | B16 / 1.4e-3 | |
| 0029 | 640 x 20 | 170M | B8 / 1.1e-3 | |
| 0030 | 768 x 24 | 273M | B8 / 0.9e-3 | gradient checkpointing (B8 OOM without) |

### Round 7 results (1 h; benchmark sampler 30 / 3 / 1.0, resampled at 30 / 2 and 30 / 4)

| run | width x depth | params | steps x batch | patch_fid 30/3 | 30/2 | 30/4 | fid 30/4 | low_res_mse 30/4 | dE00 30/4 | latent mse | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0027 | 384 x 12 | 48M | 31.6k x 16 | 6.14 | 6.55 | 6.01 | 4.49 | 13.6 | 2.34 | 0.122 | 137 |
| 0028 | 512 x 16 | 97M | 16.2k x 16 | 5.83 | 6.11 | **5.75** | **4.39** | 13.5 | 2.33 | 0.121 | 242 |
| 0029 | 640 x 20 | 170M | 18.7k x 8 | 6.08 | 6.41 | 5.96 | 4.53 | 13.6 | 2.34 | 0.121 | 401 |
| 0030 | 768 x 24 | 273M | 9.1k x 8 (checkpointing) | 7.18 | 7.70 | 6.95 | 5.01 | 14.3 | 2.41 | 0.126 | 615 |

Final training loss (frequency weighted flow loss) keeps falling with time and size (0027 2.05 -> 1.78 over 31k steps;
0029 1.71), and latent mse drops from 0.141 (30 min) to 0.121, but the decoded metrics barely move: twice the time and
3.5x the parameters give at most -0.25 patch_fid. Sampling shift 4 is best for every size (-0.35 to -0.75 vs shift 2).

## FINAL SUMMARY (Qwen-Image 2.1 VAE: problematic, parked)

Best latent only results (latent in, latent out; texture-fix decoder; bicubic 8.69 / fid 2.21 / low_res_mse 8.0e-4 /
dE00 1.85; oracle 1.07 / 0.55):
- 30 min, 48M: run_0008 6.01 / fid 4.48 (0011 5.97). 1 h, 97M: run_0028 5.75 / fid 4.39 at sampler 30 / 4 / 1.0.
- Diagnostic only (decode -> bicubic -> encode base at inference, rejected as a recipe): run_0017 4.71 / fid 3.54.
- Compare: FLUX.2 30 min 2.45 vs bicubic 8.12 (AR3); Wan 3.77 vs 5.49 (AR2, 2 h).

### Answers to the key questions

1. What the latents are: Qwen-Image-VAE-2.0 f16c64 (64 channels, 16x, RGBA; no KL, no latent noise augmentation,
   DINOv2-L middle layer alignment, space-to-channel skip connections, decoder without GAN). Smooth per token like Wan
   (lag-1 0.89) but a token covers 16x16 px, so the source explains only 61% of the target variance and the "high" latent
   band (32-64 px wavelengths) holds coarse, input-determined content. The decoder is the most noise sensitive of the
   three (10% relative noise: 29.7 dB vs FLUX.2 40.2 / Wan 33.3), damage is non-additive and non-linear: any off
   manifold error turns into colour blotches, even when latent L2 is small (a re-encoded bicubic image is further away
   in L2 than the models, yet decodes far better). Local data covariance spans 4 orders of magnitude while the decoder
   gain is nearly flat; the loss landscape of a decoder loss is very sharp around the data.
2. Residual vs direct: residual slightly better (8.63 vs 8.71, fid 5.19 vs 5.41); the per channel shift (mean) of the
   residual matters on this VAE (8.18 vs 8.63): the 512 px latents are offset from the 1024 px latents.
3. Low precision training: FP8 tensorwise training = bf16 quality (8.55 vs 8.63). Speed: GEMMs are only 36% of a step
   (elementwise / norms 35%, attention 12%, depthwise conv 8%), so FP8 gives +1% at w384 and +5% at w768 (raw FP8 GEMMs
   2-2.5x), memory -20%. torchao MXFP8 / NVFP4 training does not run on this stack (Triton build; on torch 2.14.1 needs
   2D bf16 inputs) and stable torch 2.14.1 is 21% slower than the nightly. Free speed instead: whole model compile with
   max-autotune +14%, RoPE + mask-free shifted windows +5% (and slightly better: 7.95 vs 8.18). Depthwise conv on the
   block width instead of the 4D hidden: +10% speed but -0.8 patch_fid (rejected).
4. Low precision inference: PTQ of the block linears is lossless within noise; FP8 dynamic w+a -22% latency
   (2.90 -> 2.27 ms) at +51 MB peak; int8 dynamic slower, int8 weight-only falls back to a 104 ms path.

### What worked, what did not (latent only)

- Worked (high confidence): per channel shift + scale residual normalization; frequency weighted flow loss (velocity
  error above half Nyquist x16: 8.18 -> 6.37, saturated by x16); sampling shift 3-4; RoPE / mask-free windows (small).
- Did not work: more capacity at 30 min (finer 128 x 128 sub-token grid, width 768), more time + size at 1 h (round 7),
  decoder in the loop losses (x0 at all sigma: catastrophic; low sigma only: worse; on a deterministic regression:
  worse; implementation verified), lower noise temperature (trades detail for colour 1:1), post-hoc shrinkage of low
  variance patch directions.
- The factorization diagnosis (rounds 5-6): the deterministic part of the task is "be the encoder" (DINO aligned code
  of the bicubic image); given it, generation works (4.71). A learned single forward base reaches low_res_mse 9.9 vs
  7.9 exact, but a delta flow over the learned base (0023) only gives 6.96.

### If this VAE is revisited

- Train much longer / larger only with a different objective: the flow loss keeps improving while decoded quality
  stays flat, so scaling the current recipe is not the lever.
- Candidate directions: a learned deterministic "encoder emulator" base trained to near exactness (bigger / longer
  0019, distillation from the VAE), then a delta flow; an upscaler aware decoder (like texture-fix, but trained on
  imperfect latents; breaks drop-in compatibility); a refinement pass of the Qwen-Image 2.1 DiT itself (what the
  community node does).
- Data and benchmark (`train_datasets/qwen_4klsdb_40k`, `_pixel_base`, `benchmark_datasets/axis_v1_1k_qwen_2x`) and all
  dev tools (dev/autoresearch_4/) are kept locally; the generators make_round_5/6/7.py and make_ptq_runs.py rebuild the
  run families.

## All runs

| run | summary | patch_fid (benchmark sampler) |
|---|---|---|
| 0000 | AR3 recipe, residual | 8.63 |
| 0001 | direct | 8.71 |
| 0002 | mean centered residual | 8.18 |
| 0003 | FP8 tensorwise training | 8.55 |
| 0004 | high frequency loss x4 | 6.63 |
| 0005 | high frequency loss x16 | 6.37 |
| 0006 | RoPE + mask-free shifted windows | 7.95 |
| 0007 | depthwise conv on the block width | 8.99 |
| 0008 | compound (0005 + 0006 + full compile) | 6.01 |
| 0009 | internal 128 x 128 sub-token grid | 6.07 |
| 0010 | width 768, 159M | 6.00 |
| 0011 | high frequency loss x64 | 5.97 |
| 0012 | decoder loss (x0, all sigma), 2x pooled, w1000 | 61.17 |
| 0013 | decoder loss w3000 | 88.93 |
| 0014 | decoder loss full resolution | 56.93 |
| 0015 | decoder loss without frequency weighting | 101.33 |
| 0016 | reference: E(bicubic(D(source))) | 9.10 |
| 0017 | delta flow over the re-encode base (diagnostic) | 4.93 (4.71 at 30/4) |
| 0018 | re-encode base as extra input | 5.22 |
| 0019 | regression of the re-encode base | 13.16 |
| 0020 | 0017 + low sigma decoder loss w300 | 7.22 |
| 0021 | 0017 + low sigma decoder loss w1000 | 9.44 |
| 0022 | 0019 + decoder loss | 42.83 |
| 0023 | delta flow over the learned base (VAE free) | 6.96 |
| 0024 | PTQ 0008: FP8 dynamic w+a | 6.07 |
| 0025 | PTQ 0008: int8 dynamic w+a | 6.05 |
| 0026 | PTQ 0008: int8 weight-only | 6.02 |
| 0027 | 1 h, 384 x 12 (48M) | 6.14 (6.01 at 30/4) |
| 0028 | 1 h, 512 x 16 (97M) | 5.83 (5.75 at 30/4) |
| 0029 | 1 h, 640 x 20 (170M) | 6.08 (5.96 at 30/4) |
| 0030 | 1 h, 768 x 24 (273M) | 7.18 (6.95 at 30/4) |
