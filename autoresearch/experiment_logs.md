# Experiment logs

## Key findings (keep updated)

- USER DECISIONS (prep, 2026-10-28): primary quality metric is `quality.flux_1024.fid_patch`
  (native 256 px tiles); `fid` is a helper/diagnostic metric like the mse_* metrics (now stated in
  instruction.md). CUDA graphs / any compile mode inside a run's compile() are allowed. Round 1
  retrains run_0000 in one slot for its fid_patch and as a training-seed noise reference.

- Baseline run_0000: FID 5.09, latency 2.12 / 4.14 / 12.5 ms (512/1024/2048), 106.6M params,
  peak mem 233 / 299 / 580 MB. mse_lr_image 0.0022 is 3.4x worse than pixel bicubic (0.00064):
  the baseline does not preserve the low-frequency content of its input (color / structure drift).

- Status after round 5: fid_patch 4.54 (baseline, 20 steps) -> 2.19-2.27 at lower latency.
  Recipe: training sigmas logit-normal shift 1, LR 4e-4 with linear decay over the last 30%, 10
  Euler steps at sampling shift 3, pooled text embedding in the condition, reduce-overhead compile.
  Architecture: level 0 (full res) >= 256 channels carries the quality, lower levels can be small
  and shallow. Pareto (lat1024 ms, fid_patch): 0023 (1.97, 3.26), 0035 (2.33, 2.58),
  0036 (2.43, 2.27), 0037 (2.61, 2.24), 0032 (3.72, 2.19).
- Noise: training seed ~0.03 fid_patch (0027 vs 0039; 0000 vs 0011 equal), inference seed ~0.05.
  Whole image fid is much noisier (0.7 between seeds).
- More sampling steps hurt (grainy decoded detail), less so for stronger models: likely an
  undertraining effect, keep the step count as a tunable at scale.
- See the Scale assessment section: 30 min results are biased towards fast models.

## Scale assessment (keep updated; user request after round 5)

The 30 min budget is compute bound (training GPU bound, 116 ms/step for run_0036), so fixed time
favors fast models. The goal is a recipe for training at scale, so each conclusion is also judged
per step: same SEED means identical batches and sigmas, so train/loss at equal steps is a paired
comparison. Use steps before the time based LR decay starts (decay starts at 70% of the run's own
steps); 3k and 5k steps are safe for all runs so far. From round 7, runs save checkpoints at
25/50/75% for learning curves.

Table notes legend: SCALE+ = likely better at scale than its 30 min result (better per step, lost
on speed); SCALE = conclusion holds per step too (budget-robust); SCALE? = budget-dependent, untested.

Scale candidates (lose or tie at 30 min, win per step):
- Level 0 width 384 (0025, 0038): lowest paired loss of all (0.655 @5k vs 0.679 for 0027, 0.666
  for 3 blocks at 256). Main candidate for the high quality end at scale.
- Separate LR encoder with FiLM (0004): paired loss 0.499 vs 0.510 @5k on the old recipe. Its cost
  could be removed at inference by computing the LR features once per upscale.
- Larger batch (0022), EMA (0001), LR decay length (0029): budget-specific, re-tune at scale.

Budget-robust negatives (no gain per step either): wide lower levels (0018 vs 0019), depth at low
resolution (0031), coupled noise (0002), patchify 2x2 (0005, worse per step), REPA on the old
recipe (0003, flow loss 0.5109 vs 0.5102 @5k), text cross-attention (0030, no gain at all).

Round 7 (step matched) answers: REPA (2 depths) and control stream global tokens give zero gain
per step (budget-robust negatives). Window / linear attention at the low resolution levels give a
constant per-step gain (SCALE+), window attention is the cheaper one.

## Round 0: preparation (no training)

### Reference upscalers on flux_1024 (dev/reference_fids.py, dev/patch_fid_check.py)

| Reference          | FID (benchmark, 299 resize) | Patch FID (256 tiles, native) | mse_hr_image | mse_lr_image |
|--------------------|-----------------------------|-------------------------------|--------------|--------------|
| oracle_hr_gt       | 0.37                        | 0.25                          | 0.00013      | 0.00005      |
| pixel_bicubic      | 1.85                        | 8.45                          | 0.00190      | 0.00064      |
| latent_bicubic     | 72.1                        | -                             | 0.01577      | 0.01186      |
| latent_bilinear    | 77.9                        | 114.2                         | 0.01411      | 0.01032      |
| run_0000 baseline  | 5.09                        | (weights lost)                | 0.00422      | 0.00216      |

- torchmetrics FID resizes 1024 images to 299 before Inception, so it barely sees the detail a 2x
  upscaler must generate: blurry pixel bicubic scores 1.85, better than the trained baseline.
  Native resolution patch FID separates oracle and bicubic by 34x (vs 5x).
- Baseline is worse than bicubic on full FID: it loses low-frequency fidelity. Strong lead:
  predict on top of a good LR-consistent base (pixel-decoded/bicubic init, residual
  parameterization, or a stronger LR conditioning path).

### Infrastructure notes

- Training data `latents_div_2` = encode(bilinear(decode(latents_orig))), `latent_dist.mean`; the benchmark
  LR input is built the same way from the 8-bit PNG, so train/benchmark inputs match.
- Dataloading: latents decode ~7 ms/sample on one CPU core (fine). `activations_grid` CPU decode
  is ~700 ms/sample (unusable in workers); `encoded.to("cuda").decode(dtype=torch.bfloat16)`
  takes ~3.6 ms/sample. For distillation, return EncodedTensor/raw from workers and decode on GPU.
- Latency = wall-clock of `single_forward` incl. Python launch overhead. At 512 it is launch-bound
  (2.17 ms wall vs 1.24 ms CUDA). `compile()` is part of the run file; CUDA graphs
  (mode="reduce-overhead") gave 1.46 ms at 512 for the baseline (dev/profiles).
- 35 min guard covers startup + compile + 30 min training + save; keep compile short.
- Dataset 146 GB, RAM 282 GB: it stays in the page cache across runs.
- torch 2.16 nightly (cu132), triton 3.9: SDPA and flex_attention available; no flash-attn/natten.

### Additional prep notes (session workspace-43)

- Visual check of baseline outputs (dev/compare_full.png, dev/compare_crops.png): a global
  desaturation / darkening relative to the teacher is clearly visible. This is the low frequency
  drift behind the bad FID, not missing detail.
- Inference seed noise: a 4 min baseline checkpoint (dev/checkpoints/run_0000_240s) gives FID
  14.22 / 14.41 / 14.19 for INFERENCE_SEED 0/1/2, so seed noise is about +-0.1 FID; training seed
  noise is not measured yet (consider a seed replicate of the best run in an early round).
  10 Euler steps gave 13.48 vs 14.22 at 20 steps on that checkpoint, so the step count is worth
  sweeping on real checkpoints with dev/evaluate_quality.py.
- Baseline compile modes (dev/profiles/run_0000_*.json), latency ms at 512 / 1024 / 2048:
  default 2.17 / 3.57 / 11.6; reduce-overhead 1.46 / 3.46 / 11.7; max-autotune 1.47 / 3.37 / 11.5
  (~6 min autotune per shape, so not worth it); max-autotune-no-cudagraphs 1.81 / 3.54 / 11.4.
  reduce-overhead matches eager outputs exactly (dev/check_compile_consistency.py) and survives the
  benchmark sequence (memory, compile, dynamo.reset, latency, quality).
- Latency noise: the profiler and the round differ by up to ~15% at 1024 (3.57 vs 4.14 ms) for
  the same model. Treat latency differences under ~10% as noise unless measured back to back.
- Batched GPU decode: concatenate per-sample EncodedTensor tensors along dim 0, set shape[0] = B,
  then decode once. 16 activations_grid take 14 ms, 16 latents take 1 ms (dev/test_batched_decode.py).
  Reading only the needed keys with safetensors.safe_open is about 2x faster than vaezip.load,
  which reads the whole 7 MB file.
- The infra --smoke benchmark skips compile() and latency warmup, so a compile failure would only
  show up in the real round. Run dev/profile_upscaler.py RUN_FILE --quality-samples 4 before a round.
- The benchmark now also reports quality.flux_1024.fid_patch (256 px native tiles, added in prep by
  session stdev-51). As relayed by stdev-51, the user chose fid_patch as the PRIMARY quality metric,
  whole image fid and mse_* are helper metrics only (instruction.md updated). The original
  run_0000 metrics.json has no fid_patch; it and the original tensorboard events are saved in
  dev/archive/run_0000_original/. trained_models/run_0000 and tensorboard/run_0000 were cleared,
  so the round 1 rerun starts clean. Rerun the baseline in round 1 (approved).

### Dev tools (dev/)

| Tool | Purpose |
|------|---------|
| summarize_results.py | Table of all runs: FID, mse, latency, memory, params, steps, final loss, Pareto mark |
| profile_upscaler.py RUN_FILE | Smoke train into a temp dir, then params + real compiled latency + memory as the benchmark measures them. --train-seconds N gives throughput, --override NAME=EXPR, --compile-mode, --quality-samples N for a full-flow check, --weights-output DIR keeps the weights |
| evaluate_quality.py ID | Benchmark quality of a trained run with constant overrides (steps, shift, seed), --weights-directory for other checkpoints, results in dev/evaluations/ |
| reference_fids.py | Trivial reference upscalers (oracle, latent and pixel interpolation) |
| check_compile_consistency.py RUN_FILE | Compiled vs eager upscale() output difference |

The tools rely on run files keeping the module constants OUTPUT_DIRECTORY, TENSORBOARD_DIRECTORY,
TRAINING_TIME_SECONDS and a `self.model` nn.Module, as run_0000 does. They need a free GPU, so use
them between rounds or on GPUs a round leaves free (set CUDA_VISIBLE_DEVICES).
patch_fid_check.py --reference NAME computes full and patch FID of a reference upscaler on its own.

## Round 1

### Setup notes (2026-10-28)

- The dev/ tools listed above were LOST (dev/ was cleaned). Rebuilt minimal versions:
  dev/run_tools.py (load run file with NAME=EXPR overrides), dev/profile_run.py RUN
  (--train-seconds, --override, --quality-samples, --resolutions; params, throughput, memory,
  compiled latency), dev/evaluate_quality.py RUN (--samples, --override, --weights-directory,
  --tag, --compile; results in dev/evaluations/), dev/summarize_results.py (markdown table),
  dev/latent_statistics.py, dev/test_batched_decode.py.
- Latent statistics (dev/latent_statistics.py, 256 samples): latents_orig std 1.0 per channel,
  latents_div_2 std 0.98. Residual over 2x upsampled LR latents has std 0.79 (bilinear),
  0.83 (bicubic), 0.87 (nearest): LR latents in latent space explain little of the HR latents, so a
  residual-over-upsampled-LR parameterization buys little. Teacher init_latents correlate 0.39 with
  latents_orig (R^2 ~0.15): the 4-step teacher keeps a lot of its noise structure.
- Batched GPU decode of activations_grid works and is exact (16 samples: 14 ms); activations
  std 15, token norm ~836.
- Throughput in 60 s parallel profiles (steps/s at batch 16): baseline ~6.7 (12k steps / 30 min),
  REPA 4.8, LR encoder 5.3, patchify 10.5, narrow 13.6, wide 4.2. Training memory 4-16.5 GB.
- All new runs use compile(mode="reduce-overhead") (CUDA graphs): an inference-only change,
  512 latency 2.2 -> ~1.55 ms for the baseline architecture, no change at 1024/2048.

### Hypotheses

1. EMA should reduce the low frequency drift and improve both FIDs for free.
2. Coupled noise (teacher init_latents) straightens the flow; the strong noise/output correlation
   suggests the model can learn a near-deterministic map. Risk: 10 epochs of fixed pairs overfit.
3. REPA on teacher activations (block 15) speeds up representation learning in a 30 min budget.
4. A separate LR encoder modulating every block uses the LR signal better than input concat.
5. Patchify 2x2 (UNet on the 64x64 LR grid) should cut latency a lot at similar quality: every
   level of the baseline costs about the same FLOPs and the 128x128 level is the most memory bound.
6./7. Width scaling (narrow / wide) gives the slope of quality vs. latency for a fixed 30 min.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0000 | Baseline rerun | - | fid_patch anchor, training seed noise vs archived 5.09 fid |
| 0001 | EMA (ramped decay) | 0000 | EMA effect |
| 0002 | Coupled noise (teacher init_latents) | 0000 | noise coupling effect |
| 0003 | REPA alignment to teacher activations (64x64 level, weight 0.5) | 0000 | distillation of representations |
| 0004 | Separate LR encoder + zero-init spatial FiLM (no input concat) | 0000 | LR conditioning path |
| 0005 | Patchify 2x2 UNet on LR grid, channels (384,768,768) | 0000 | latency/architecture |
| 0006 | Narrow (128,256,512) | 0000 | width scaling, low latency |
| 0007 | Wide (256,512,1024) | 0000 | width scaling, high capacity |

Launched: ./infra/start_round.sh 0000 0001 0002 0003 0004 0005 0006 0007

### Round 1 results (benchmark: 20 steps, shift 3; latency ms at 1024)

| run | fid_patch | fid | mse_lr*1e3 | lat1024 | params | steps | note |
|---|---|---|---|---|---|---|---|
| 0000 baseline | 4.54 | 5.09 | 2.05 | 4.17 (default compile) | 107M | 11.6k | fid identical to archived 5.09: training is deterministic with SEED 0 |
| 0001 EMA | 4.70 | 4.89 | 2.06 | 4.13 | 107M | 11.0k | raw weights of the same run: fid_patch 4.58, fid 5.23 -> EMA ~neutral; SCALE?: EMA usually matters more for long runs; untested with the current recipe |
| 0002 coupled noise | 7.13 | 6.55 | 2.79 | 4.00 | 107M | 11.7k | loss 0.157 (deterministic map) but much worse samples. RETIRED |
| 0003 REPA w0.5 | 6.10 | 6.29 | 2.40 | 4.05 | 107M | 8.4k | no per-step loss gain, 30% slower. RETIRED for now; SCALE?: paired flow loss 0.5109 vs 0.5102 @5k (no gain per step either); retested step matched in round 7 |
| 0004 LR encoder FiLM | 4.66 | 5.02 | 2.21 | 4.84 | 115M | 9.4k | lower loss per step (0.464 vs 0.486 @8k), equal per time, slower; SCALE+: paired loss 0.4986 vs 0.5102 @5k, clearly more sample efficient; lost only on speed. Its LR features could be computed once per upscale (not per step) |
| 0005 patchify 2x2 | 6.99 | 5.19 | 1.95 | 3.13 | 162M | 17.6k | good fid/mse but bad detail: full resolution processing matters |
| 0006 narrow 128 | 5.70 | 6.68 | 2.98 | 1.96 | 51M | 23.3k | |
| 0007 wide 256 | 4.40 | 4.63 | 1.96 | 5.43 | 182M | 7.3k | lower loss per step and per time; SCALE+: paired loss 0.4850 vs 0.5102 @5k |

Data order and sigmas are identical across runs with the same SEED, so per-step losses are paired.

### Inference sweep on trained checkpoints (dev/evaluate_quality.py, 1000 samples)

run_0000 fid_patch by sampling setting (shift 3 unless noted): 5 steps 3.73, 7 3.76, 10 3.93,
14 4.16, 20 4.54, 30 4.78; 20 steps shift 1: 4.84, shift 6: 3.98, shift 12: 3.71;
10 steps shift 1.5: 4.29, shift 6: 3.71. INFERENCE_SEED 1 at 20 steps: 4.50 (seed noise ~0.05).
run_0007 10 steps 3.62 (vs 4.40 at 20, 4.89 at 30); run_0006 10 steps 4.77; run_0005 30 steps 7.44.
mse_lr also improves with fewer low noise steps (2.05 -> 1.62 at 5 steps).

KEY FINDING: every step spent at low noise hurts. Cause: logit-normal + shift 3 training almost
never samples sigma < 0.1 (P ~ 0.05%) and only ~1.3% in [0.1, 0.25], so the velocity field at low
noise is untrained, and the sampler integrates garbage there. Fix candidates: training sigma
distribution with more low noise mass (round 2), and/or few-step / high-shift sampling.
Current best inference for shift 3 trained models: 10 steps, sampling shift 6 (3.71 on 0000).

Width/latency (10 steps, shift 3): narrow 4.77 @1.96 ms, base 3.93 @~3.5 ms (reduce-overhead),
wide 3.62 @5.43 ms.

Latency profiles of candidates (reduce-overhead, 1024): (96,384,768) 2.95 ms; (64,384,768) 2.79;
(128,384,1024) 3.78; (192,384,768,768) 3.67; (128,256,512,768) 2.46; (160,320,640) 2.59.

Lost tool incident: `rm -rf dev/profile_*` deleted dev/profile_run.py; it was recreated, temp
directories now live in dev/temporary/.

## Round 2

Base for all: run_0000 + reduce-overhead compile, separate TRAINING_SIGMA_SHIFT and
SAMPLING_SIGMA_SHIFT, 10 sampling steps. Runs with shift 3 training sample with shift 6 (compare
with run_0000 at 10 steps / shift 6 = 3.71). Runs with new training distributions sample with
shift 3, and every run gets a post-hoc steps/shift sweep. Generator: dev/make_round_02.py.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0008 | Uniform training sigmas | 0000 | low noise training mass |
| 0009 | Logit-normal, training shift 1 | 0000 | low noise training mass |
| 0010 | Logit-normal, training shift 1/3 | 0000 | strong low noise emphasis |
| 0011 | Training SEED 1 | 0000 | training seed noise level |
| 0012 | Linear LR decay to 0 over last 30% of time | 0000 | LR schedule |
| 0013 | LR 4e-4 | 0000 | LR magnitude |
| 0014 | 4 levels (128,256,512,768) | 0000, 0006 | cheaper high res + deeper context (2.46 ms) |
| 0015 | Narrow level 0 (64,384,768) | 0000 | cost of full resolution width (2.79 ms) |

### Round 2 results (benchmark: 10 steps; shift 3 trained runs sample with shift 6)

| run | fid_patch | fid | mse_lr*1e3 | lat1024 | steps | note |
|---|---|---|---|---|---|---|
| 0000 @10/shift6 (eval) | 3.71 | 4.81 | 1.67 | ~3.5 | 11.6k | reference |
| 0008 uniform sigma | 3.89 | 3.33 | 1.78 | 4.10 | 11.7k | shift 6 sampling: 3.72 |
| 0009 train shift 1 | 3.55 | 3.44 | 1.76 | 4.12 | 11.1k | best fid tradeoff; sampling shift 3 best |
| 0010 train shift 1/3 | 3.83 | 3.54 | 1.91 | 4.03 | 11.8k | |
| 0011 seed 1 | 3.71 | 4.09 | 1.48 | 4.07 | 12.0k | fid_patch identical to 0000 (3.71); fid differs 0.7 -> fid is noisy, fid_patch is stable |
| 0012 LR decay 30% | 3.60 | 4.55 | 1.62 | 4.02 | 12.1k | |
| 0013 LR 4e-4 | 3.53 | 4.73 | 1.39 | 4.12 | 11.3k | big mse gain |
| 0014 4 levels (128,..,768) | 4.85 | 7.10 | 2.04 | 2.50 | 20.2k | ~ narrow 3-level: level 0 width decides |
| 0015 (64,384,768) | 6.77 | 8.02 | 2.98 | 3.16 | 16.4k | narrow level 0 kills detail |

Post-hoc sampling sweeps (dev/sweep.py, dev/evaluations/r2_sweep): more steps still hurt for every
training distribution: 0009 5/10/20/30 steps (shift 3) = 3.69/3.55/3.94/4.22, 20 steps shift 6 3.58;
0010 (heavy low noise training) 10/20/30 = 3.83/4.21/4.46; 0008 10/20/30 = 3.89/4.42/4.78.
So the untrained low noise region was NOT the main reason.

Diagnostic (dev/sampling_diagnostics.py, 0009, 32 samples): with more steps the latent std goes
0.89 (3 steps) -> 0.95 (10) -> 0.98 (30, = teacher 0.98), but the decoded image Laplacian energy
goes 13.4 -> 16.3 -> 18.0 vs teacher 15.1. The model generates fine latent detail with the right
energy but the wrong structure, which decodes to grain; fewer steps generate less of it. The
remedy is a model better at fine structure; strongest lever so far is level 0 (full res) width:
fid_patch at 10 steps vs level 0 width 64/128/192/256: 6.77/~4.8/3.7/3.6.

## Round 3

New stacked base: training shift 1 + LR 4e-4 + linear LR decay over the last 30% (run_0016),
10 steps, sampling shift 3. Generator dev/make_round_03.py (from run_0009).

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0016 | Stack: shift 1 + LR 4e-4 + decay | 0009, 0012, 0013 | new base |
| 0017 | Stack + LR 8e-4 | 0016 | LR scan |
| 0018 | Stack + wide (256,512,1024) | 0016, 0007 | quality end of frontier |
| 0019 | Stack + (256,384,768) | 0016 | is level 0 width the key lever |
| 0020 | Stack + 3 blocks per level | 0016 | depth vs width |
| 0021 | Stack + pooled text embedding in the condition | 0016 | text conditioning |
| 0022 | Stack + batch 32 | 0016 | batch size |
| 0023 | Stack + narrow (128,256,512) | 0016, 0006 | low latency end with the new recipe |

### Round 3 results (10 steps, sampling shift 3; lat1024 from the benchmark)

| run | fid_patch | fid | mse_lr*1e3 | lat1024 | params | steps | note |
|---|---|---|---|---|---|---|---|
| 0016 stack | 2.87 | 2.69 | 1.34 | 4.09 | 107M | 11.6k | stack gains 0.66 over best single change (3.53) |
| 0017 stack LR 8e-4 | 3.05 | 2.85 | 1.37 | 4.15 | 107M | 11.0k | 4e-4 is better |
| 0018 stack wide (256,512,1024) | 2.86 | 2.74 | 1.35 | 5.33 | 182M | 7.4k | no gain: lower level width useless; SCALE: paired loss 0.6820 vs 0.6800 for (256,384,768) @5k: the per-step gain is all from level 0, wide lower levels useless even per step (budget-robust) |
| 0019 stack (256,384,768) | 2.57 | 2.63 | 1.26 | 4.41 | 110M | 10.5k | level 0 width is the lever |
| 0020 stack 3 blocks/level | 2.59 | 2.60 | 1.27 | 5.39 | 140M | 9.2k | good but slow; SCALE+: paired loss 0.6816 vs 0.6991 (0016) @5k |
| 0021 stack + pooled text | 2.72 | 2.62 | 1.34 | 4.14 | 108M | 11.2k | -0.15, free |
| 0022 stack batch 32 | 3.28 | 2.91 | 1.51 | 4.13 | 107M | 5.9k | worse: steps matter more; SCALE+: budget-dependent; larger batches are standard with more compute, untested at equal samples |
| 0023 stack narrow (128,256,512) | 3.26 | 2.78 | 1.51 | 1.97 | 51M | 23.2k | 7 steps: 3.17 |

Sampling sweep (dev/evaluations/r3_sweep): 0016 steps 5/10/15/20 = 3.04/2.87/3.01/3.15, shift 2/5 at
10 steps 2.96/2.89, 15 steps shift 5 2.87; 0019 7/10/15 = 2.62/2.57/2.64; 0021 15 steps 2.86;
0023 7/10/15 = 3.17/3.26/3.52. 10 steps / shift 3 stays near optimal; better models tolerate
more steps (the 10 -> 15 penalty shrinks), narrow models prefer fewer.

Round 4 latency profiles (reduce-overhead, 1024 / 2048 ms): 0024 4.11/14.1, 0025 5.37/20.7,
0026 4.66/16.1, 0027 2.80/9.4 (!), 0028 2.29/7.0, 0030 4.32/14.7, 0031 4.71/16.1.
Lower levels cost far more latency than their FLOPs suggest: (256,256,512) is 1.3 ms cheaper than
(256,384,768).

## Round 4

Base: run_0021 + level channels (256,384,768) + per level block counts LEVEL_BLOCKS.
Generator dev/make_round_04.py.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0024 | Stack + text + (256,384,768) | 0019, 0021 | new base |
| 0025 | 0024 with level 0 = 384 | 0024 | level 0 width scan |
| 0026 | 0024 with 3 blocks at level 0 | 0024 | level 0 depth |
| 0027 | 0024 with (256,256,512) | 0024 | cheap lower levels (2.8 ms) |
| 0028 | 0024 with (160,256,512) | 0024, 0023 | low latency point (2.3 ms) |
| 0029 | 0024 with LR decay over 60% | 0024 | decay length |
| 0030 | 0024 + text cross-attention at lowest level (3 modules) | 0024 | token level text conditioning |
| 0031 | 0024 with (2,2,3) blocks + 4 middle | 0024 | depth at low resolution |

### Round 4 results (10 steps, shift 3)

| run | fid_patch | fid | mse_lr*1e3 | lat1024 | params | steps | note |
|---|---|---|---|---|---|---|---|
| 0024 base (256,384,768)+text | 2.50 | 2.55 | 1.25 | 4.55 | 112M | 10.2k | text + level 0 256 stack (2.57, 2.72 -> 2.50) |
| 0025 (384,384,768) | 2.36 | 2.55 | 1.23 | 6.05 | 120M | 6.9k | SCALE+: lowest paired loss of all (0.6551 @5k vs 0.6787 for 0027): level 0 width 384 is the most sample efficient design, lost only on speed |
| 0026 blocks (3,2,2) | 2.29 | 2.47 | 1.21 | 5.13 | 115M | 8.8k | level 0 depth helps; SCALE+: paired loss 0.6661 @5k |
| 0027 (256,256,512) | 2.29 | 2.43 | 1.21 | 2.92 | 59M | 14.7k | BEST: dominates 0024, cheap lower levels -> more steps |
| 0028 (160,256,512) | 2.74 | 2.64 | 1.36 | 2.45 | 54M | 19.5k | |
| 0029 decay over 60% | 2.59 | 2.61 | 1.28 | 4.56 | 112M | 9.8k | worse than 30% (2.50); SCALE?: decay length is budget-specific; re-tune at scale |
| 0030 text cross-attention | 2.50 | 2.59 | 1.26 | 4.72 | 119M | 9.9k | no gain over pooled text. RETIRED |
| 0031 (2,2,3)+4 middle | 2.64 | 2.62 | 1.31 | 5.34 | 160M | 8.8k | low resolution depth is useless; SCALE: paired loss 0.6808 vs 0.6778 (0024) @5k: useless per step too (budget-robust) |

Pareto archive (lat1024 ms, fid_patch): 0023 (1.97, 3.26), 0028 (2.45, 2.74), 0027 (2.92, 2.29).
Rule so far: put capacity at full resolution (level 0), keep lower levels small, maximize steps.

## Round 5: architecture scan around run_0027 (generator dev/make_round_05.py)

Profiled latency (reduce-overhead, 1024 ms) shown; 0027 itself profiles 2.80.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0032 | (256,256,512), blocks (3,2,2) | 0027, 0026 | level 0 depth at 3.52 ms |
| 0033 | (320,256,512) | 0027 | level 0 width at 3.52 ms (pair with 0032) |
| 0034 | (256,192,384) | 0027 | cheaper lower levels, 2.64 ms |
| 0035 | (192,192,384) | 0027, 0028 | low latency, 2.19 ms |
| 0036 | (256,256,512), blocks (2,1,1), 1 middle | 0027 | shallow lower levels, 2.37 ms |
| 0037 | (256,256,256) | 0027 | flat width, 2.48 ms |
| 0038 | (384,256,512) | 0027, 0025 | high end, 4.33 ms |
| 0039 | 0027 with SEED 1 | 0027 | seed noise at the frontier |

### Round 5 results (10 steps, shift 3)

| run | fid_patch | fid | lat1024 | params | steps | note |
|---|---|---|---|---|---|---|
| 0032 (256,256,512) blocks (3,2,2) | 2.19 | 2.42 | 3.72 | 62M | 11.5k | level 0 depth > width (pair 0033); SCALE: per step 0.6662 vs 0.6664 for 0033: depth and width equal per step at this size, depth wins on speed |
| 0033 (320,256,512) | 2.28 | 2.52 | 3.52 | 62M | 10.1k | |
| 0034 (256,192,384) | 2.26 | 2.42 | 2.76 | 39M | 17.1k | |
| 0035 (192,192,384) | 2.58 | 2.60 | 2.33 | 37M | 21.7k | level 0 192 too narrow |
| 0036 (256,256,512) blocks (2,1,1), 1 middle | 2.27 | 2.50 | 2.43 | 38M | 17.3k | frontier |
| 0037 (256,256,256) | 2.24 | 2.36 | 2.61 | 31M | 16.3k | frontier |
| 0038 (384,256,512) | 2.20 | 2.41 | 4.51 | 66M | 9.3k | dominated by 0032; SCALE+: paired loss 0.6552 @5k vs 0.6662 (0032): dominated only at 30 min, best scale candidate |
| 0039 = 0027 seed 1 | 2.31 | 2.49 | 2.94 | 59M | 14.3k | 0027 2.29: seed noise ~0.03 |

Lower levels barely matter (width 256..512, depth 1..2). Level 0 must be >= 256 wide; depth there helps.

Small experiments between rounds 5 and 6:
- Flip augmentation is NOT usable: encode(flip(image)) vs flip(encode(image)) mse 0.44-0.60 (latent var 1,
  re-encode noise 0.09); even the best linear 32x32 channel map leaves 0.29 (dev/flip_equivariance.py).
- Training is GPU compute bound: 0036 116 ms/step on GPU vs 22 ms to load a batch; 0032 172 vs 24 ms
  (dev/training_step_timing.py). Level 0 convolutions at 128x128 dominate.

Pareto archive (lat1024, fid_patch): 0023 (1.97, 3.26), 0035 (2.33, 2.58), 0036 (2.43, 2.27),
0037 (2.61, 2.24), 0032 (3.72, 2.19).

## Round 6 (generator dev/make_round_06.py, base run_0036; profiled latency at 1024)

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0040 | blocks (3,1,1) | 0036, 0032 | level 0 depth with cheap lower levels, 3.03 ms |
| 0041 | (256,192,256) blocks (4,1,1) | 0036 | deep level 0, 3.43 ms |
| 0042 | 4 ConvNeXt blocks (dw7x7, MLP x4) at level 0 | 0036 | block type; 2.78 ms but trains 2x slower |
| 0043 | encoder (1,1,1) / decoder (3,1,1) blocks | 0036 | decoder heavy level 0, 2.39 ms |
| 0044 | 2 levels (256,256), blocks (2,1) | 0036 | is the hierarchy needed, 2.00 ms |
| 0045 | (256,128,256) | 0036 | narrower lower levels, 1.95 ms |
| 0046 | batch 8 | 0036 | more steps |
| 0047 | LR 6e-4 | 0036 | LR for small models |

### Round 6 results (10 steps, shift 3; paired loss = train/loss at 5k steps, before decay)

| run | fid_patch | fid | lat1024 | params | steps | loss@5k | note |
|---|---|---|---|---|---|---|---|
| 0036 base (2,1,1) | 2.27 | 2.50 | 2.43 | 38M | 17.3k | 0.6836 | reference |
| 0040 blocks (3,1,1) | 2.14 | 2.36 | 3.15 | 41M | 12.9k | 0.6690 | frontier; SCALE+ (better per step) |
| 0041 (256,192,256) blocks (4,1,1) | 2.10 | 2.41 | 3.55 | 27M | 11.2k | 0.6605 | BEST quality, frontier; SCALE+ |
| 0042 4 ConvNeXt blocks at level 0 | 2.44 | 2.70 | 2.82 | 39M | 8.3k | 0.6823 | trains 2x slower; equal per step; SCALE?: fewer FLOPs but memory bound, not worth it on this GPU |
| 0043 enc (1,1,1) / dec (3,1,1) | 2.16 | 2.42 | 2.52 | 38M | 17.2k | 0.6768 | same cost as 0036, -0.11: detail capacity belongs in the decoder; SCALE (per step too) |
| 0044 2 levels (256,256) | 2.25 | 2.46 | 2.11 | 18M | 19.5k | 0.6842 | hierarchy barely matters; frontier |
| 0045 (256,128,256) | 2.35 | 2.57 | 2.10 | 19M | 20.7k | 0.6917 | dominated by 0044 |
| 0046 batch 8 | 2.28 | 2.52 | 2.45 | 38M | 32.8k | (0.7107, half samples) | neutral at equal time; per sample equal to batch 16 |
| 0047 LR 6e-4 | 2.29 | 2.47 | 2.45 | 38M | 16.6k | 0.6842 | neutral |

Pareto archive (lat1024, fid_patch): 0023 (1.97, 3.26), 0044 (2.11, 2.25), 0043 (2.52, 2.16),
0040 (3.15, 2.14), 0041 (3.55, 2.10).

## Round 7: step matched test of REPA and global context (user question: do they help at scale?)

Base run_0043. Every run trains exactly 9000 steps (LR decay over the last 30% of steps, the time
limit only as a guard) and saves checkpoints at 2250/4500/6750 steps: comparisons are at equal
steps and samples, so the training slowdown of REPA/attention does not count against them.
Learning curves: dev/sweep.py with MODEL_FILE_NAME override on checkpoints, 500 samples.
Generator dev/make_round_07.py. Profiled training speed (8 concurrent, steps/s) and latency 1024:

| Run ID | Run name | Parent runs | steps/s | lat1024 | Run purpose |
|---|---|---|---|---|---|
| 0048 | control (0043, 9000 steps) | 0043 | 9.4 | 2.41 | reference |
| 0049 | REPA w0.5 on level 1 encoder output (64x64) | 0048, 0003 | 5.7 | 2.38 | REPA at the teacher grid |
| 0050 | REPA w0.5 on middle output (32x32, teacher pooled) | 0048 | 7.3 | 2.35 | REPA at semantic depth |
| 0051 | 8x8 window attention after level 1/2/middle stages | 0048 | 8.8 | 2.65 | local attention |
| 0052 | global linear attention after level 1/2/middle stages | 0048 | 9.0 | 3.07 | cheap global mixing |
| 0053 | control stream (64 tokens, 2 layers) at middle | 0048 | 9.1 | 2.72 | global tokens |
| 0054 | control stream + REPA level 1 | 0053, 0049 | 5.8 | 2.69 | combination |
| 0055 | control streams (128 tokens) at middle and decoder level 1 | 0053 | 8.3 | 3.28 | multi-scale global tokens |

### Round 7 results (step matched, 9000 steps each; benchmark 10 steps, shift 3)

| run | fid_patch | fid | lat1024 | params | train time | loss@5k | loss@9k | note |
|---|---|---|---|---|---|---|---|---|
| 0048 control | 2.49 | 2.64 | 2.52 | 38M | 989 s | 0.6761 | 0.6341 | reference (0043 at 30 min, 17.2k steps: 2.16) |
| 0049 REPA level 1 | 2.51 | 2.70 | 2.47 | 38M | 1635 s | 0.6791 | 0.6339 | features align (cos 0.83) but zero gain. SCALE: robust negative |
| 0050 REPA middle | 2.49 | 2.66 | 2.54 | 38M | 1246 s | 0.6787 | 0.6339 | zero gain. SCALE: robust negative |
| 0051 window attention 8x8 (levels 1, 2, middle) | 2.37 | 2.62 | 2.79 | 42M | 1059 s | 0.6664 | 0.6218 | constant per-step gain 0.010-0.012 from 2k to 9k. SCALE+ |
| 0052 global linear attention (same places) | 2.41 | 2.58 | 3.25 | 42M | 1034 s | 0.6678 | 0.6222 | same gain as window attention, more latency |
| 0053 control stream 64 tokens (middle) | 2.49 | 2.68 | 2.79 | 47M | 1042 s | 0.6786 | 0.6359 | zero gain |
| 0054 control stream + REPA | 2.49 | 2.68 | 2.75 | 47M | 1588 s | 0.6781 | 0.6350 | zero gain |
| 0055 control streams 128 tokens (middle + level 1) | 2.49 | 2.67 | 3.32 | 55M | 1119 s | 0.6762 | 0.6340 | zero gain |

Conclusions (answer to the user question about scale):
- REPA does not help this task even at equal steps: the flow loss is identical from 2k to 9k
  steps and fid_patch is unchanged, at two depths and with a control stream. The upscaler gets
  layout and semantics from the LR input, which is what REPA speeds up in text/class to image
  DiTs. No trend over training either, so no reason to expect it to start helping at scale.
- Global token context (control stream, 64 or 128 tokens, 1 or 2 scales) gives nothing: global
  information is not a bottleneck for 2x upscaling with this conditioning.
- Attention inside the image does help: 8x8 window attention and global linear attention give
  the same per-step gain (the gap stays constant from 2k to 9k steps), so the benefit is
  content adaptive mixing at the low resolution levels, not global range. Window attention is
  the cheaper one (+0.27 ms, 7% slower training). SCALE+.
- Learning curve method: 500 sample checkpoint fid_patch is strongly biased (~6.0 vs 2.49 at 1000)
  and too noisy for gaps of 0.1, and mid-run checkpoints are at peak LR. Use 1000 samples for
  checkpoint evaluations; the paired train loss is the cleanest per-step learning curve.

## Round 8: window attention at the fixed 30 min budget (generator dev/make_round_08.py)

| Run ID | Run name | Parent runs | lat1024 (profile) | Run purpose |
|---|---|---|---|---|
| 0056 | 0043 + window attention levels 1, 2, middle | 0043, 0051 | 2.70 | does the per-step gain survive at fixed time |
| 0057 | 0043 + window attention level 2 + middle only | 0056 | 2.58 | cheaper placement |
| 0058 | 0056 + window attention after the full resolution decoder stage | 0056 | 2.83 | attention at level 0 |
| 0059 | 0041 (deep level 0) + window attention | 0041, 0051 | 3.62 | quality end |
| 0060 | 0044 (2 levels) + window attention | 0044, 0051 | 2.20 | low latency end |
| 0061 | 0043 + linear attention | 0043, 0052 | 3.13 | window vs linear at fixed time |
| 0062 | 0043 + 16x16 window attention | 0056 | 2.76 | window size |
| 0063 | (384,256,512) dec heavy + window attention | 0063 | 4.12 | scale candidate (level 0 width 384) |

### Round 8 results (30 min, 10 steps shift 3; loss@5k paired)

| run | fid_patch | fid | lat1024 | params | steps | loss@5k | note |
|---|---|---|---|---|---|---|---|
| 0043 reference | 2.16 | 2.42 | 2.52 | 38M | 17.2k | 0.6768 | |
| 0056 0043 + window attn (L1, L2, mid) | 2.08 | 2.37 | 2.88 | 42M | 15.1k | 0.6665 | survives at fixed time: -0.08 for +0.36 ms; frontier |
| 0057 window attn L2 + mid only | 2.15 | 2.41 | 2.68 | 41M | 15.2k | 0.6730 | level 1 attention carries most of the gain |
| 0058 0056 + window attn full res decoder | 2.12 | 2.40 | 2.97 | 42M | 13.9k | 0.6636 | better per step, slower; SCALE+ (small) |
| 0059 0041 + window attn | 2.01 | 2.39 | 3.72 | 28M | 11.7k | 0.6566 | frontier |
| 0060 0044 (2 levels) + window attn | 2.18 | 2.41 | 2.35 | 18M | 17.9k | 0.6731 | frontier |
| 0061 0043 + linear attention | 2.12 | 2.37 | 3.32 | 42M | 14.9k | 0.6678 | dominated by window attention |
| 0062 16x16 windows | 2.08 | 2.35 | 2.92 | 42M | 14.6k | 0.6644 | = 8x8 at fixed time, slightly better per step |
| 0063 (384,256,512) dec heavy + window | 1.96 | 2.31 | 4.42 | 49M | 9.7k | 0.6419 | BEST quality; the scale candidate also wins at 30 min |

Sampler sweep (dev/evaluations/r8_sweep): the fewer-steps preference REVERSED for strong models:
0063 8/10/14/20 steps = 2.05/1.96/1.91/1.94, shift 2 at 10: 1.92, 14 steps shift 2: 1.94, 20 steps
shift 2: 1.99; 0059 10/14/20 = 2.01/1.98/2.02; 0056 10/14 = 2.08/2.09; 0060 8/10 = 2.24/2.18.
Confirms the undertraining explanation: the optimal step count grows with model quality.

Pareto archive (lat1024, fid_patch): 0023 (1.97, 3.26), 0044 (2.11, 2.25), 0060 (2.35, 2.18),
0043 (2.52, 2.16), 0057 (2.68, 2.15), 0056 (2.88, 2.08), 0059 (3.72, 2.01; 1.98 at 14 steps),
0063 (4.42, 1.96; 1.91 at 14 steps).

## Round 9: both ends of the front with window attention (generator dev/make_round_09.py, base 0063)

| Run ID | Run name | Parent runs | lat1024 (profile) | Run purpose |
|---|---|---|---|---|
| 0064 | (256,256,512) enc 1 / dec 5 at level 0 + window | 0056, 0059 | 3.31 | decoder heavy deep level 0 (vs 0059) |
| 0065 | (384,256,512) dec (4,1,1) + window, 14 steps | 0063 | 4.70 | quality end, depth |
| 0066 | (512,256,512) dec (3,1,1) + window, 14 steps | 0063 | 5.30 | quality end, width (scale candidate) |
| 0067 | 2 levels (256,256) enc (1,1) dec (3,1) + window | 0060, 0043 | 2.24 | low latency, decoder heavy |
| 0068 | 0063 + EMA, 14 steps | 0063, 0001 | 4.13 | EMA on the strong recipe |
| 0069 | 2 levels (256,256) enc (1,1) dec (2,1) + window | 0067 | 1.90 | lowest latency point |
| 0070 | 0056 with training shift 1.5 | 0056, 0009 | 2.69 | retune training sigmas |
| 0071 | 0063 + window attn at the full res decoder, 14 steps | 0063, 0058 | 4.43 | full res attention at the quality end |

### Round 9 results (30 min; runs marked * sample with 14 steps, others 10; loss@5k paired)

| run | fid_patch | fid | lat1024 | params | steps | loss@5k | note |
|---|---|---|---|---|---|---|---|
| 0063 reference (384, dec 3) | 1.96 (1.91 @14) | 2.31 | 4.42 | 49M | 9.7k | 0.6419 | |
| 0064 (256,256,512) dec (5,1,1) + window | 1.94 | 2.30 | 3.53 | 45M | 11.9k | 0.6525 | frontier; beats 0059 (2.01/1.98@14 at 3.72) |
| 0065* (384,256,512) dec (4,1,1) + window | 1.87 | 2.28 | 5.08 | 53M | 7.9k | 0.6335 | BEST quality (tie), frontier; SCALE+ |
| 0066* (512,256,512) dec (3,1,1) + window | 1.87 | 2.27 | 5.65 | 60M | 6.8k | 0.6226 | ties 0065 at 30 min with 14% fewer steps; best per step of all runs: SCALE+ (strongest scale candidate) |
| 0067 2 levels (256,256) enc (1,1) dec (3,1) + window | 2.08 | 2.40 | 2.37 | 18M | 17.8k | 0.6664 | frontier: = 0056 quality at 0.5 ms less |
| 0068* 0063 + EMA | 1.95 | 2.32 | 4.49 | 49M | 10.1k | 0.6421 | vs 1.91 for 0063 @14 steps: EMA neutral/slightly worse with LR decay. SCALE?: re-check on long runs |
| 0069 2 levels enc (1,1) dec (2,1) + window | 2.23 | 2.48 | 2.06 | 17M | 19.7k | 0.6786 | frontier (marginal over 0044) |
| 0070 0056 with training shift 1.5 | 2.06 | 2.46 | 2.83 | 42M | 15.2k | (other sigmas) | = 0056 (2.08): shift 1-1.5 is flat, keep 1 |
| 0071* 0063 + window attn full res decoder | 1.92 | 2.29 | 4.76 | 50M | 8.8k | 0.6390 | = 0063 @14 (1.91) at fixed time; small per-step gain (SCALE+ small) |

Per-step loss at 5k for the quality end keeps improving with level 0 width: 384/dec3 0.6419,
384/dec4 0.6335, 512/dec3 0.6226. At 30 min the 512 model only gets 6.8k steps (5.4 epochs) and
ties; with more compute it should pull ahead.

Pareto archive after round 9 (lat1024 benchmark, fid_patch benchmark):
0023 (1.97, 3.26), 0069 (2.06, 2.23), 0067 (2.37, 2.08), 0064 (3.53, 1.94), 0065 (5.08, 1.87).
(0063 at 4.42 is 1.96 in the benchmark, 1.91 with 14 steps; 0066 ties 0065 at higher latency.)

Progress summary: baseline fid_patch 4.54 at 4.2 ms -> 2.08 at 2.37 ms (0067, 2x faster) and
1.87 at 5.1 ms (0065). Round 10 on hold: the user is deciding between more 30 min rounds and a
2 hour scaling experiment.
