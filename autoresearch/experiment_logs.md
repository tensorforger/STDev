# Experiment logs

## Key findings (keep updated)

- USER DECISIONS (prep, 2026-10-28): primary quality metric is `quality.flux_1024.fid_patch`
  (native 256 px tiles); `fid` is a helper/diagnostic metric like the mse_* metrics (now stated in
  instruction.md). CUDA graphs / any compile mode inside a run's compile() are allowed. Round 1
  retrains run_0000 in one slot for its fid_patch and as a training-seed noise reference.

- Baseline run_0000: FID 5.09, latency 2.12 / 4.14 / 12.5 ms (512/1024/2048), 106.6M params,
  peak mem 233 / 299 / 580 MB. mse_lr_image 0.0022 is 3.4x worse than pixel bicubic (0.00064):
  the baseline does not preserve the low-frequency content of its input (color / structure drift).

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

## Round 1 plan (prepared, not started)

- Run list: `./infra/start_round.sh "0000" "0001" ... "0007"`. run_0000 is the FIRST run (GPU 0),
  retrained unchanged: its fid_patch anchors every later comparison, and comparing its fid against
  the archived 5.09 gives a first estimate of training-seed noise. New runs are 0001-0007.
- No run files 0001+ exist yet (an earlier attempt's untracked files are gone).
- Leads from prep worth testing early (not binding): an LR-consistent base, i.e. a residual over a
  pixel/bicubic-quality upsample or a stronger LR conditioning path (baseline drifts in color);
  EMA; coupled noise (init_latents); adaLN-Zero LR encoder; sampling step count; CUDA graphs in
  compile(); teacher activation distillation with batched GPU decode.
- Checklist before launching: all 8 GPUs free (nvidia-smi); `./infra/start_round.sh ... --smoke`
  passes; `dev/profile_upscaler.py RUN_FILE --quality-samples 4` on new runs to catch compile()
  failures that the infra smoke skips; launch the real round in the background and wait for the
  notification.
