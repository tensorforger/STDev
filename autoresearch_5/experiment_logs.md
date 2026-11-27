# Experiment logs (autoresearch 5: Ming-Image VAE, 512 -> 1024)

## Status

- Rounds 0-1 done (exploratory single round, 4 runs). Best: run_0003 at sampler 30 / 3 / 0.9: patch_fid 6.34 / fid 3.25
  (bicubic 10.58, oracle 3.57): 60% of the bicubic -> oracle gap at 30 min (FLUX.2 AR3 77%, Qwen-Image 2.1 AR4 36%).

## Key findings (keep updated)

- (round 0, reports/ming_image_vae_check.md) Ming-Image VAE = Qwen-Image-Layered RGBA VAE (Wan 2.1 / Qwen-Image 1
  architecture, 8x, 16 channels), but its own latent space (raw std ~0.116, per channel linear R2 0.80 to Wan). Robust
  decoder (10% latent noise 39.5 dB, FLUX.2 40.6, Qwen-Image 2.1 29.8). Detail heavy 2x task: the bilinear source
  explains 21% of the target variance (FLUX.2 41%, Wan 85%), 30% of the energy above half Nyquist.
- (round 0) The VAE round trip is lossy on photos: oracle patch_fid 3.57 (FLUX.2 0.71, Qwen-Image 2.1 texture-fix 1.07,
  Wan 1.30), pixel bicubic 10.58: a high floor for any upscaler on this benchmark (the VAE targets design / RGBA content).

- (round 1) The FLUX.2 recipe works as is: 6.92 / fid 3.39 at 30 min (bicubic 10.58 / 3.18). Residual vs direct: no
  difference (6.92 vs 6.96), as expected for a detail heavy latent (FLUX.2: -0.3, Wan: -3).
- (round 1) AR4 package (mean centered residual, RoPE / mask-free windows, whole model compile: +20% steps) 6.68 (-0.24);
  the frequency weighted loss adds nothing here (6.68 with and without, fid 3.39 / 3.34): it was a Qwen-Image 2.1
  specific fix.
- (round 1) Sampler: initial noise temperature 0.9 improves every metric (0003: 6.34 / fid 3.25 / low_res_mse 13.2 /
  dE00 2.32 vs 6.63 / 3.29 / 14.5 / 2.40 at T 1.0, shift 3), unlike FLUX.2 / Qwen-Image 2.1 where it traded detail for
  colour (Wan-like). Shift 3-4 slightly better than 2 (-0.05 ... -0.1); 20 steps close to 30.
- (round 1) Roundtrip metrics stay worse than pixel bicubic (low_res_mse 13-15 vs 10e-4; fid ~bicubic), the same
  pattern as FLUX.2 in AR3; the decoder does not produce Qwen-Image 2.1 style colour blotches.

## Prior knowledge

- AR3 (FLUX.2) recipe: window transformer w384 d12, residual over the bilinear source (per channel std), learned source
  path, B4, LR 1.7e-3, sampler 30 / 2 / 1.0; residual gain small for detail heavy latents (-0.3 on FLUX.2).
- AR4 (Qwen-Image 2.1, problematic VAE): transferable: per channel mean centering of the residual, frequency weighted
  flow loss (x16 above half Nyquist), sampling shift 3-4, RoPE + mask-free shifted windows (+5% speed), whole model
  compile with max-autotune (+14%).

## Round 0: preparation

- Qwen-Image 2.1 latent caches deleted (no more tests on this machine); Axis ground truth / low resolution PNGs and the
  FID statistics reused (identical crops).
- Latent normalization: own per channel statistics of the raw posterior modes (512 crops): raw std 0.087-0.149.
- Residual statistics (dev/autoresearch_5/residual_statistics.py, 6000 blocks): target rms 0.985, residual rms std
  0.896 (per channel 0.60-1.13), residual mean |max| ~0.08.

### References (raw_results/ming/reference_*)

| reference | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 |
|---|---|---|---|---|
| oracle (VAE roundtrip of the GT) | 3.57 | 0.86 | 3 | 1.37 |
| pixel bicubic | 10.58 | 3.18 | 10 | 2.21 |
| latent bilinear | 123.9 | 84.6 | 65 | 4.60 |

## Round 1: FLUX.2 recipe and the autoresearch_4 package (generator dev/autoresearch_5/make_round_1.py)

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0000 | AR3 FLUX.2 recipe (w384 d12, residual / per channel std, source path, B4, LR 1.7e-3, 30 / 2 / 1.0) | flux 0018 | baseline |
| 0001 | 0000 with direct flow | 0000 | residual vs direct on this VAE |
| 0002 | AR4 package: mean centered residual + frequency weighted loss x16 + RoPE / mask-free windows + full compile | qwen 0008 | do the AR4 findings transfer |
| 0003 | 0002 with plain velocity MSE | 0002 | isolate the frequency weighting |

Launched: ./ming_infra/start_round.sh 0000 0001 0002 0003

### Round 1 results (benchmark sampler 30 / 2 / 1.0)

| run | change | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | lat p95 ms | VRAM MB | steps |
|---|---|---|---|---|---|---|---|---|
| 0000 | AR3 FLUX.2 recipe | 6.92 | 3.39 | 14.7 | 2.42 | 10.5 (noise) | 253 | 13.9k |
| 0001 | direct | 6.96 | 3.42 | 14.6 | 2.42 | 7.4 | 253 | 14.0k |
| 0002 | AR4 package | 6.68 | 3.39 | 14.5 | 2.41 | 56 (outlier) | 253 | 16.8k |
| 0003 | AR4 package, plain velocity MSE | 6.68 | 3.34 | 14.6 | 2.41 | 7.4 | 253 | 16.7k |

Sampler sweep (dev/autoresearch_5/sampler_sweep.py, 1000 images; patch_fid / fid / low_res_mse 1e-4 / dE00):

| run | 30/3/1.0 | 30/4/1.0 | 30/3/0.9 | 20/3/1.0 |
|---|---|---|---|---|
| 0000 | 6.87 / 3.34 / 14.5 / 2.41 | 6.83 / 3.30 / 14.4 / 2.40 | - | - |
| 0002 | 6.61 / 3.33 / 14.3 / 2.40 | 6.59 / 3.29 / 14.2 / 2.38 | - | - |
| 0003 | 6.63 / 3.29 / 14.5 / 2.40 | 6.61 / 3.26 / 14.3 / 2.38 | **6.34 / 3.25 / 13.2 / 2.32** | 6.56 / 3.24 / 14.1 / 2.37 |

### Verdict after one round

The Ming-Image VAE is workable: the FLUX.2 recipe transfers without VAE specific fixes and closes ~60% of the bicubic ->
oracle gap in 30 min. Limits: the VAE itself is lossy on photos (oracle 3.57), so absolute numbers stay above the
FLUX.2 / Wan ones. Next if pursued: temperature 0.8-0.9 and shift sweep on longer runs, the 1 h scaling ladder, RGBA
(alpha) data, and design / typography content closer to the model's domain.
