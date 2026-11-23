# Experiment logs (autoresearch 3: FLUX.2, 512 -> 1024)

## Status

- DONE: 5 rounds, 20 runs (0000-0019). See RECOMMENDED PRETRAIN RECIPE at the end. Recipe run file: run_0016.

## Key findings (keep updated)

- (round 1) Q1: normalized residual flow beats direct on both architectures at equal sampler (14 steps, shift 3,
  T 1.0): window transformer 2.72 vs 3.02, UNet 3.17 vs 3.42 patch_fid; fid also better. Modest gain, as predicted from
  the latent statistics (FLUX.2 residual std 0.68 vs target 0.90; Wan gained -3).
- (round 1) Q2: the phase 2 window transformer beats the phase 1 UNet at equal sampler (2.72 vs 3.17) while seeing 35%
  fewer samples (50k at B4 vs 77k at B16): better per sample, slower per sample (28 vs 41 samples/s). Its roundtrip
  metrics are better too (low_res_mse 7.7-8.6 vs 9.4-9.7 e-4).
- (round 1) Q3 sampler: initial noise temperature < 1 (phase 2) HURTS on FLUX.2 patch_fid (0001: T 0.9 3.27-3.45 vs
  T 1.0 2.72; 0002: 3.52 vs 3.17); it lowers low_res_mse slightly (a smoothness trade). Shift 3 > 4; 20 steps > 14
  (0001 2.59 vs 2.72, 0002 3.13 vs 3.17). New default: 20 steps, shift 3, T 1.0. The round 1 benchmark numbers of
  0001 / 0003 (T 0.9, shift 4) are sampler limited; use the resampled values.
- (round 1) The window transformer is compute bound at ~28 samples/s at any batch (B4 7.2 steps/s 15 GB, B8 3.5
  steps/s 30 GB, B16 OOM; B16 + checkpointing 21 samples/s): larger batches give no extra samples in 30 min.
- (round 2) Seed noise is small for this recipe: 0001 (SEED 0) 2.59 / 3.04 vs 0005 (SEED 1) 2.59 / 3.04 patch_fid /
  fid. Latency of the same model differs 8.2 vs 10.1 ms between rounds: treat latency differences < ~20% as noise.
- (round 2) Q3: the learned source path transfers: without it (0004) 2.82 / 3.44 / lr_mse 9.18 vs 2.59 / 3.04 / 8.83.
- (round 2) Q4 data: 4KLSDB only (0007) 2.60 / fid 2.89 / lr_mse 8.59 = A + B on patch_fid and better on fid and
  roundtrip; ReLAION only (0006) 2.70 / 3.21 / 9.06. 30 min runs see ~50k samples (1.3 epochs of A + B), so data volume
  is not binding yet: the dense, sharp 4KLSDB crops are the more valuable source per sample.
- (round 3) Q5: input corruption COSTS on the bilinear benchmark, growing with the corrupted share: 50% filters +0.29,
  50% JPEG +0.30 (fid +0.5), 50% all +0.38, 83% all +0.51 patch_fid (seed noise ~0).
- (round 3) Q5 benefit (dev/robustness_eval.py, corrupted axis inputs): the clean model collapses on JPEG inputs
  (patch_fid 49.3: it turns block artifacts into detail) and degrades on blurry (antialias 6.33) and nearest (4.82)
  inputs. 50% all-corruption training: JPEG 10.1, antialias 3.69, nearest 3.74. Each family protects only against
  itself (filters do not help JPEG, JPEG does not help nearest). Verdict: a large robustness gain for a small benchmark
  cost; for production image-to-image / generated inputs, corruption training is worth it.
- (round 4) 20% corruption (0012, 4% per family) halves the benchmark cost (2.76, +0.17) and keeps most robustness:
  JPEG 11.3 (clean 49.3, 50% mix 10.1), antialias 4.14 (6.33 / 3.69), nearest 4.05 (4.82 / 3.74).
- (round 4) Data weighted ReLAION : 4KLSDB = 1 : 2 (0013): 2.55 / 3.00 / 8.69, small gain on all metrics (to be
  confirmed by the round 5 replicate). LR 2.4e-3 (0014) = 1.7e-3 (2.59): not LR limited. Deep / narrow w320 d16
  (0015) 2.66, -14% VRAM, same latency: no gain; keep w384 d12.
- (round 4) Sampler optimum moves with model quality (as in phase 1): 0013 20/3 2.55, 20/2 2.49, 30/3 2.47, 30/2 2.45.
- Seed replicates (0005, 0017, 0018) change SEED and INFERENCE_SEED (training and sampling noise): full replicates.
- (round 5) Replicates are tight: clean weighted data 0013 / 0018 2.45 / 2.46; candidate 0016 / 0017 2.58 / 2.55.
  Data weighting confirmed but small: unweighted 0001 / 0005 at 30 / 2 / 1.0 2.48 / 2.49 (fid 3.04 / 3.06) vs weighted
  2.45 / 2.46 (fid 3.01 / 2.95).
- (round 5) JPEG weighted 20% corruption (0016) costs +0.11 patch_fid on the bilinear benchmark and gives JPEG 10.4
  (clean 0018: 47.5), antialias 3.82 (4.81), nearest 4.05.
- (round 5) Q2 final: the UNet with every transferred shift (0019) 2.95 vs window transformer 2.45. The transformer wins.
- Latency p95 of the compiled single forward has rare large outliers when 4 benchmarks run in parallel (0010 27 ms,
  0017 48 ms for the 8 ms model): a measurement artifact, not a model property.

## Prior knowledge (carried over, see the phase 1 / 2 logs for details)

- Phase 1 (FLUX.2, 20k FLUX-generated samples, direct flow on HR latents): capacity at full latent resolution
  (decoder heavy level 0), depth > width for latency, 8x8 window attention, SDXL style window transformer blocks on all
  levels best per step. LR 4e-4, batch 16, linear decay over the last 30% (essential), training shift 1, sampling
  14-20 steps at shift 2-3. Negative: REPA, global tokens, text cross-attention, coupled noise, 2x2 patchify, flips.
- Phase 2 (Wan 2.1, smooth latents: HR vs upsampled LR correlation 0.95, residual std 0.22): flow on the per channel
  normalized residual over the bilinear aligned source (-3 patch FID), learned source path (pixel shuffle; halved patch
  FID vs bilinear only), plain window transformer (adaLN-zero, shifted windows, relative bias, GELU MLP + depthwise conv)
  beats the UNet, condition RMS norm + QK-norm for stability, small batch (2-4) + high LR (1.2e-3 - 1.7e-3) at 30 min,
  decay over the last 50%, sampling 14 steps / shift 4 / initial noise temperature 0.9. Seed noise 0.1-0.3 patch FID.

## Round 0: preparation (no training)

### Data and benchmark

- Sources: ReLAION shard 0 (seed 0 shuffle, same urls as the Wan dataset), 20000 images; 4KLSDB `train_x4` files
  0-32, 20757 images (first 20000 sorted used). Axis-V1 1k for the benchmark.
- `crop_maximal_rectangle` fixed before encoding: it passed `cv2.INTER_AREA` positionally (as `dst`), so it silently
  used INTER_LINEAR (aliased 4x downscales of 4K images). Now: INTER_AREA when downscaling, Lanczos when upscaling.
  The axis ground truth of this benchmark therefore differs slightly from the Wan axis benchmark (not comparable 1:1).
- Input corruption variants on one 1024 image (dev/check_flux2_smoke.py, mean |gradient| of the 512 image / mean
  abs diff to bilinear): bilinear 7.40 / 0, nearest 8.77 / 2.09, bicubic 8.59 / 0.65, lanczos 7.59 / 0.55,
  antialias 6.32 / 0.75, jpeg q20-50 6.71 / 3.86. Note: cv2 INTER_AREA at exactly 2x equals bilinear, so the blurry
  variant is PIL antialiased bilinear (triangle filter over 4x4 source pixels).

### Latent statistics (dev/flux2_latent_statistics.py, 1500 blocks per set)

| set | target std | residual std (latents_1024 - bilinear(latents_512)) | corr(target, aligned source) |
|---|---|---|---|
| ReLAION (A) | 0.846 | 0.629 | 0.695 |
| 4KLSDB (B) | 0.946 | 0.726 | 0.677 |
| A + B | 0.902 | 0.680 | 0.688 |

FLUX.2 latents are NOT smooth (Wan: target 0.72 / residual 0.22 / corr 0.95). The residual normalization only rescales
by ~1/0.68 and removes a source term that explains ~45% of the variance. Prediction: the residual flow helps much less
than on Wan (phase 2: -3 patch FID); the gain, if any, comes from the free low frequency base (fid, low_res_mse, chroma).
RESIDUAL_STD table (A + B, per channel) is in run_0001.

### References (raw_results/flux/reference_*)

| reference | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 |
|---|---|---|---|---|
| oracle (VAE roundtrip of the GT) | 0.71 | 0.72 | 1.0 | 1.08 |
| pixel bicubic | 8.12 | 1.77 | 5.0 | 1.74 |
| latent bilinear | 124.6 | 91.7 | 81.1 | 5.47 |

The FLUX.2 VAE ceiling (0.71) is much lower than Wan's (1.30): more room for the upscaler.

### Infrastructure notes

- Training throughput / memory (dev/profile_training.py, compiled, random data): run_0000 UNet B16 2.58 steps/s
  (41 samples/s), peak 25.3 GB; with gradient checkpointing 1.96 steps/s, 7.7 GB. Eager (smoke mode) B16 UNet OOMs, so
  `--smoke` only works for runs that fit eagerly; real runs compile.
- Smoke benchmark (3 step weights): single forward p95 ~14 ms, peak VRAM 256 MB for the 47.6M transformer; end to end
  (14 steps + FLUX.2 decode at 1024) ~300 ms, dominated by the decode.
- Tools: dev/sampler_sweep.py (sampling settings on trained weights / checkpoints, no training),
  dev/profile_training.py (compiled throughput and memory with constant overrides).

## Round 1: phase 1 vs phase 2 recipe, 2x2 factorial (key questions 1 and 2)

Each architecture keeps its own phase recipe (training batch / LR / schedule, sampler); only the parameterization is
crossed. Data A + B, bilinear inputs only, 30 min, SEED 0.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0000 | Phase 1 recipe: UNet (384, 192, 384) + window transformer blocks, direct, 46.6M | phase 1 0077 / 0078 | baseline A |
| 0001 | Phase 2 recipe: window transformer w384 d12, learned source, normalized residual, 47.6M | phase 2 0072 | baseline B |
| 0002 | 0000 + normalized residual | 0000 | parameterization effect on the UNet |
| 0003 | 0001 with direct flow | 0001 | parameterization effect on the transformer |

Launched: ./flux_infra/start_round.sh 0000 0001 0002 0003

### Plan for the next rounds (to be revised with results)

- Round 2: data (A only / B only vs A + B from round 1) on the round 1 winner + a seed replicate of the winner.
- Round 3: input corruption mixes (all variants, sharp-only, jpeg-only) vs bilinear-only.
- Round 4: remaining phase 2 shifts not covered by the factorial on the winner (source path, condition / QK norm,
  batch / LR), sampler sweeps between rounds without training.
- Round 5: compound the winners, replicate, write the RECOMMENDED PRETRAIN RECIPE.

### Round 1 results (benchmark sampler of each recipe)

| run | arch | param | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | lat p95 ms | VRAM MB | steps x batch | final loss |
|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | UNet | direct | 3.42 | 3.83 | 9.71 | 2.17 | 7.8 | 275 | 4.8k x 16 | 0.540 |
| 0001 | WT | residual | 3.45 | 3.38 | 7.74 | 1.97 | 8.2 | 256 | 12.6k x 4 | 0.795 |
| 0002 | UNet | residual | 3.17 | 3.33 | 9.37 | 2.13 | 8.0 | 275 | 4.8k x 16 | 0.838 |
| 0003 | WT | direct | 3.91 | 3.76 | 7.78 | 2.00 | 8.3 | 256 | 12.4k x 4 | 0.498 |

Sampler sweep (dev/sampler_sweep.py, all 1000 images; steps / shift / temperature -> patch_fid, fid, low_res_mse 1e-4):

| run | 14/3/1.0 | 14/4/1.0 | 14/3/0.9 | 14/4/0.9 | 20/3/1.0 |
|---|---|---|---|---|---|
| 0001 WT residual | 2.72 / 3.07 / 8.56 | 2.85 / 3.13 / 8.36 | 3.27 / 3.31 / 7.88 | 3.45 (bench) | **2.59** / 3.04 / 8.83 |
| 0002 UNet residual | 3.17 (bench) | 3.23 / 3.37 / 9.15 | 3.52 / 3.52 / 8.61 | 3.66 / 3.59 / 8.46 | 3.13 / 3.31 / 9.69 |
| 0003 WT direct | 3.02 / 3.37 / 8.51 | 3.24 / 3.49 / 8.28 | - | - | - |
| 0000 UNet direct | 3.42 (bench) | - | - | - | - |

Visual check (dev/round1_crops.png): all four produce plausible fur / texture; differences are subtle at 30 min.
Latency / VRAM are equal by construction (~8 ms, 256-275 MB): the size match worked.

## Round 2: source path ablation, seed noise, data (key questions 3 and 4)

Base: run_0001 (window transformer, residual, B4, LR 1.7e-3) with the new sampler (20 / 3 / 1.0). Reference: 0001
resampled 2.59 (same SEED 0 batches). A planned B16 run was dropped (no throughput gain, see findings).

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0004 | 0001 without the learned source path (bilinear concat only) | 0001 | does the phase 2 source path transfer to FLUX.2 |
| 0005 | 0001, SEED 1 | 0001 | training seed noise |
| 0006 | 0001 on ReLAION only (A) | 0001 | data |
| 0007 | 0001 on 4KLSDB only (B) | 0001 | data |

Launched: ./flux_infra/start_round.sh 0004 0005 0006 0007

### Round 2 results (sampler 20 / 3 / 1.0)

| run | change vs 0001 | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | lat p95 ms | VRAM MB | steps | final loss |
|---|---|---|---|---|---|---|---|---|---|
| 0001 | reference (resampled) | 2.59 | 3.04 | 8.83 | - | 8.2 | 256 | 12.6k | 0.795 |
| 0004 | no learned source path | 2.82 | 3.44 | 9.18 | 2.11 | 7.9 | 233 | 13.4k | 0.832 |
| 0005 | SEED 1 | 2.59 | 3.04 | 8.81 | 2.07 | 10.1 | 256 | 12.8k | 0.791 |
| 0006 | ReLAION only | 2.70 | 3.21 | 9.06 | 2.12 | 8.8 | 256 | 12.7k | 0.654 |
| 0007 | 4KLSDB only | 2.60 | 2.89 | 8.59 | 2.04 | 8.8 | 256 | 12.4k | 0.910 |

(Losses of 0006 / 0007 are on their own data: not comparable.)

## Round 3: input corruption during training (key question 5)

Base: 0001 recipe (window transformer, residual, A + B, SEED 0, sampler 20 / 3 / 1.0); only SOURCE_KEY_WEIGHTS differ.
Reference without corruption: 0001 / 0005 2.59.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0008 | 50% bilinear + 50% all five corruptions | 0001 | general robustness mix |
| 0009 | 50% bilinear + 50% filters only (nearest, bicubic, lanczos, antialias) | 0001 | filter part alone |
| 0010 | 50% bilinear + 50% JPEG q20-50 | 0001 | JPEG part alone |
| 0011 | uniform over all six (bilinear 1/6) | 0001 | strongest corruption |

Launched: ./flux_infra/start_round.sh 0008 0009 0010 0011. Note: dev/robustness_eval.py build (Flux VAE encode,
~3 min) shared GPU 0 with 0008 early in its training: 0008 may have slightly fewer steps.

Robustness eval (dev/robustness_eval.py, dev experiment): the 1000 axis images with nearest / antialias / JPEG
q20-50 512 inputs, patch_fid against the same ground truth. Measures the benefit side of question 5.

### Round 3 results (bilinear benchmark)

| run | corruption mix | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | lat p95 ms | steps |
|---|---|---|---|---|---|---|---|
| 0001 | none (reference) | 2.59 | 3.04 | 8.83 | - | 8.2 | 12.6k |
| 0008 | 50% bilinear + 50% all five | 2.97 | 3.37 | 9.15 | 2.11 | 8.3 | 12.5k |
| 0009 | 50% bilinear + 50% filters | 2.88 | 3.32 | 8.91 | 2.09 | 8.5 | 12.7k |
| 0010 | 50% bilinear + 50% JPEG | 2.89 | 3.55 | 9.80 | 2.17 | 27.3 (noise) | 12.6k |
| 0011 | uniform over all six | 3.10 | 3.58 | 9.37 | 2.13 | 8.6 | 12.4k |

### Robustness eval (corrupted axis inputs; patch_fid / fid / low_res_mse 1e-4 against the corrupted input)

| run | train mix | nearest | antialias | jpeg q20-50 |
|---|---|---|---|---|
| 0001 | clean | 4.82 / 5.54 / 15.2 | 6.33 / 3.10 / 3.4 | 49.27 / 19.05 / 5.2 |
| 0008 | 50% all | 3.74 / 5.08 / 16.3 | 3.69 / 3.00 / 5.4 | 10.08 / 12.96 / 14.0 |
| 0009 | 50% filters | 3.47 / 5.04 / 16.3 | 3.68 / 2.92 / 5.2 | 47.80 / 18.85 / 6.7 |
| 0010 | 50% JPEG | 5.08 / 5.81 / 18.3 | 4.75 / 3.38 / 4.7 | 8.50 / 11.74 / 14.3 |
| 0011 | 83% all | 3.54 / 5.16 / 16.6 | 3.62 / 3.20 / 6.1 | 9.74 / 13.04 / 14.2 |

The higher low_res_mse of corruption-trained models on corrupted inputs is intended: they remove the artifacts instead
of reproducing them in the roundtrip.

## Round 4: corruption rate, data weighting, LR, shape

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0012 | 80% bilinear + 20% all five corruptions | 0008 | robustness at a lower benchmark cost |
| 0013 | ReLAION : 4KLSDB = 1 : 2 per sample, clean | 0001, 0007 | compound the data finding |
| 0014 | LR 2.4e-3, clean | 0001 | still LR limited? |
| 0015 | width 320, depth 16 (44.2M), clean | 0001 | depth vs width at equal size (source path shrinks to 7.6M) |

Launched: ./flux_infra/start_round.sh 0012 0013 0014 0015

### Round 4 results (sampler 20 / 3 / 1.0)

| run | change vs 0001 | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | lat p95 ms | VRAM MB | steps |
|---|---|---|---|---|---|---|---|---|
| 0012 | 20% corruption (all five) | 2.76 | 3.23 | 9.06 | 2.10 | 8.0 | 256 | 12.8k |
| 0013 | ReLAION : 4KLSDB = 1 : 2 | 2.55 | 3.00 | 8.69 | 2.06 | 8.3 | 256 | 12.8k |
| 0014 | LR 2.4e-3 | 2.59 | 3.04 | 8.89 | 2.08 | 8.2 | 256 | 12.8k |
| 0015 | width 320, depth 16 (44.2M) | 2.66 | 3.13 | 8.92 | 2.09 | 8.2 | 220 | 11.9k |

Robustness of 0012 (patch_fid / fid): nearest 4.05 / 5.18, antialias 4.14 / 2.88, jpeg 11.26 / 14.53.
Sampler sweep of 0013 (patch_fid / fid): 20/3 2.55 / 3.00, 20/2 2.49 / 2.99, 30/3 2.47 / 2.99, 30/2 2.45 / 3.01.

## Round 5: compound and confirm (sampler 30 / 2 / 1.0)

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0016 | PRODUCTION CANDIDATE: 0013 + 20% corruption (jpeg 6, nearest 4, antialias 4, bicubic 3, lanczos 3 %) | 0012, 0013 | final recipe |
| 0017 | 0016, SEED 1 | 0016 | replicate |
| 0018 | 0013 (clean, weighted data), SEED 1 | 0013, 0005 | confirm the data weighting gain vs 0005 |
| 0019 | UNet + residual + learned source (w192) + weighted data + new sampler | 0002 | Q2 with all transferable shifts |

References at 30 / 2 / 1.0: 0013 2.45 (SEED 0). Launched: ./flux_infra/start_round.sh 0016 0017 0018 0019

### Round 5 results (sampler 30 / 2 / 1.0)

| run | what | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | lat p95 ms | VRAM MB | end to end ms |
|---|---|---|---|---|---|---|---|---|
| 0016 | candidate: weighted data + 20% corruption, SEED 0 | 2.58 | 3.17 | 9.32 | 2.12 | 8.0 | 256 | 338 |
| 0017 | candidate, SEED 1 | 2.55 | 3.21 | 9.27 | 2.13 | 48 (outlier) | 256 | 344 |
| 0018 | clean weighted data, SEED 1 | 2.46 | 2.95 | 9.05 | 2.10 | 8.2 | 256 | 346 |
| 0019 | UNet + residual + source path + weighted data | 2.95 | 3.34 | 9.92 | 2.17 | 8.3 | 280 | 346 |

Resampled at 30 / 2 / 1.0: 0001 2.48 / 3.04 / 9.23, 0005 2.49 / 3.06 / 9.21, 0013 2.45 / 3.01.
Robustness (patch_fid / fid): 0016 nearest 4.05 / 5.25, antialias 3.82 / 2.88, jpeg 10.42 / 13.17;
0018 antialias 4.81 / 2.84, jpeg 47.52 / 19.29. The 30 step sampler raises low_res_mse (9.0-9.3 vs 8.6-8.9 at 20 / 3):
more generated detail, slightly less roundtrip fidelity.

## ANSWERS TO THE KEY QUESTIONS

1. Direct vs normalized residual: residual wins on both architectures (-0.25 to -0.30 patch_fid, better fid).
   Much smaller than on Wan (-3) because FLUX.2 latents are not smooth (residual std 0.68 vs target 0.90, correlation
   0.69). Residual is the universal choice for both VAE types: it never hurts and helps more the smoother the latents.
2. UNet + window transformer blocks vs plain window transformer (~50M, equal latency / VRAM): plain window transformer,
   by 0.45 at equal sampler and 0.5 with all shifts applied to both (2.45 vs 2.95). It is better per sample (fewer
   samples seen in 30 min) and has better roundtrip metrics.
3. Phase 2 shifts on FLUX.2: transfer = residual flow, learned source path (+0.23 without), window transformer, small
   batch / high LR. Do NOT transfer: initial noise temperature < 1 (hurts by 0.4-0.7) and sampling shift 4 (2-3 better;
   the optimum moved from 20 / 3 to 30 / 2 as models improved). Not tested here (kept as stabilizers for long runs):
   condition RMS norm, QK-norm.
4. Data: 4KLSDB alone = A + B on patch_fid and better on fid; ReLAION alone worse (+0.11). ReLAION : 4KLSDB = 1 : 2 is
   best (small, replicated gain). 30 min runs see ~1.3 epochs, so data volume is untested at this scale.
5. Input corruption: costs on the bilinear benchmark (+0.11 at 20% JPEG weighted, +0.38 at 50%, +0.51 at 83%) but the
   clean model collapses on JPEG inputs (47-49 patch_fid) and degrades on blurry inputs. 20% corruption keeps most of
   the robustness (JPEG 10.4, antialias 3.8). Each family only protects against itself: mix JPEG and filters.

## RECOMMENDED PRETRAIN RECIPE (FLUX.2 512 -> 1024; reference implementation: run_0016)

Architecture:
- Plain window transformer on the full resolution latent grid (no UNet): 3x3 conv stem on [noisy residual sample,
  bilinear 2x source] + learned source path (2 residual blocks at the source resolution, pixel shuffle 2x, 1x1 out)
  added to the stem; DiT style adaLN-zero blocks, LayerNorm without affine, shifted 8x8 windows (alternating) with a
  learned relative position bias, QK-norm, GELU MLP 4D with a depthwise 3x3 conv on the hidden; adaLN final layer, zero
  initialized linear head; windows folded into the SDPA batch (fused kernel).
- Condition: sigma embedding + hff embedding (Immerkaer, added at pretrain; 10% dropout to a null embedding for CFG as
  in the Wan pretrain), RMS normalized.
- Size: scale width / depth like the Wan pretrain (width 768, depth 16, ~220M); deep / narrow gave no gain at 50M.
  Note: the source path scales with width^2 (11M of 47.6M at width 384): consider a narrower source encoder (untested).
Parameterization: flow on r = (x - bilinear 2x(source)) / RESIDUAL_STD (per channel, table in run_0016; recompute on
  the pretrain data mix).
Data:
- ReLAION : 4KLSDB = 1 : 2 per sample (center crop, antialiased resize, `crop_maximal_rectangle`).
- Input corruption on 20% of samples: JPEG q20-50 6%, nearest 4%, antialias 4%, bicubic 3%, lanczos 3% (the rest
  bilinear, as in the benchmark). Corruption applies to the input only; the target and its hff stay clean.
- Online pipeline caveat: 4KLSDB train_x4 holds ~130k unique 4K images, used up in ~3.5 h at the expected pretrain
  rate; take several crops per 4K image or cache the 1024 PNGs (~200 GB). Download is not the bottleneck (~65 img/s).
- Mixed resolution buckets (CONTEXT.md: sequential resolution phases forget) are not covered by this research.
Training: flow matching, velocity target, logit-normal sigmas, training shift 1, AdamW (0.9, 0.95), weight decay 0.01,
  grad clip 1, bf16, fused AdamW. LR 1.7e-3 at batch 4 for 48M (2.4e-3 equal: not LR limited); for the larger model use
  the phase 2 square root batch rule and retune the base LR lower for width 768. Warmup >= 1000 steps, linear decay to
  0 over the last 50%. The transformer is compute bound (~28 samples/s per 5090 at 48M at any batch), so batch is a
  communication / stability choice, not a throughput one.
Sampling: Euler, 30 steps, shift 2, initial noise temperature 1.0 (re-sweep on the final model; the optimum moves to
  more steps / lower shift as quality grows). Then 1-step rectified flow distillation as for Wan.
Expected: at 30 min / 48M the recipe gives patch_fid 2.55-2.58 (clean data variant 2.45) vs pixel bicubic 8.12 and the
  VAE oracle 0.71.

## All runs (patch_fid at the run's benchmark sampler; resampled values in the round sections)

| run | summary | patch_fid |
|---|---|---|
| 0000 | UNet, direct (phase 1 recipe) | 3.42 |
| 0001 | WT, residual (phase 2 recipe; 2.48 at 30/2/1.0) | 3.45 |
| 0002 | UNet, residual | 3.17 |
| 0003 | WT, direct (3.02 at 14/3/1.0) | 3.91 |
| 0004 | WT residual, no learned source | 2.82 |
| 0005 | 0001 SEED 1 | 2.59 |
| 0006 | ReLAION only | 2.70 |
| 0007 | 4KLSDB only | 2.60 |
| 0008 | 50% all corruption | 2.97 |
| 0009 | 50% filter corruption | 2.88 |
| 0010 | 50% JPEG corruption | 2.89 |
| 0011 | 83% all corruption | 3.10 |
| 0012 | 20% all corruption | 2.76 |
| 0013 | ReLAION : 4KLSDB = 1 : 2 (2.45 at 30/2/1.0) | 2.55 |
| 0014 | LR 2.4e-3 | 2.59 |
| 0015 | width 320, depth 16 | 2.66 |
| 0016 | candidate (weighted data + 20% JPEG weighted corruption) | 2.58 |
| 0017 | 0016 SEED 1 | 2.55 |
| 0018 | 0013 SEED 1 | 2.46 |
| 0019 | UNet with all shifts | 2.95 |
