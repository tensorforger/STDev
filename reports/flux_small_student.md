# FLUX.2 upscaler: small single step student (20.2M, 2026-11-24, 8x RTX 5090)

Code: `src/stdev/flux_distill/small_config.py`, `small_pairs.py` (online pairs of the large student),
`small_trainer.py` (DDP), evaluation `src/stdev/flux_distill/evaluation.py --model small`, launcher
`scripts/run_flux_small_distill.py`; latency `scripts/flux_tools/measure_flux_students.py`, throughput
`scripts/flux_tools/bench_small_student_training.py`. Run: `distill_runs/flux_small_s20m`.

## GPU layout (measured before the launch)

| per GPU, steady state | throughput | peak memory |
|---|---|---|
| large 1-step student generating pairs (192M, batch 16 @1024) | ~30 pairs/s (GPU 100%) | 8 GB |
| small student training (20.2M, batch 8 @1024; batch 16 OOM) | 42.7 samples/s | 22.8 GB |
| both on one GPU (trainer batch 4) | ~21 + ~22 samples/s | ~20 GB |

- DDP scales almost linearly for this model (4 GPUs: 172-183 samples/s, ~44 per GPU).
- 4 generators + 4 trainers balance best: ~120 pairs/s produced vs ~177 samples/s trained (reuse 1.48 in the run).
  Co-locating a generator and a trainer on every GPU gives about the same training throughput with reuse ~1.0, but
  halves the trainer batch and leaves little memory headroom; not used.
- Fixed before the launch: with 4 generators next to the trainer the generators fell to ~16 pairs/s each. Input
  loading decoded 4 full pretraining cache shards (~80 MB, 1.17 s each, target latents included) per 64 pairs on one
  thread. Now: source latents / hff / ids only, one shard per job, two prefetch threads -> ~30 pairs/s again.

## Setup

- **Model:** the Wan small student recipe (student architecture scaled down, MLP 4x, from scratch) at width 256,
  depth 14, head dim 64, condition dim 256, source encoder width 192: 20.2M. The Wan s20m shape (256 x 10) is 15.4M
  without the temporal layers; depth instead of width (phase 1: depth is more latency efficient).
- **Pairs:** the large 1-step student (`flux_distill_v1`, raw weights) on LR latents of the pretraining cache
  (pretraining bucket shares, with corruptions); hff true / random / null 45 / 45 / 10%, CFG U[1, 2] for half of the
  non-null. Rolling 120 GB cache, 657k pairs produced in total.
- **Training:** 4 GPU DDP, global batch 32 @1024 (128 @512), AdamW (0.9, 0.95), wd 0.01, LR 2.1e-3 (Wan s20m 1.5e-3
  at batch 16, sqrt scaled), 1000 warmup steps, linear decay over the last 30% of a fixed 6 h, EMA 0.999.
  109,780 steps, 3.79M samples (Wan s20m: 1.5M in 9.1 h on one GPU), mean pair reuse 1.48, no crash or restart.
- **Evaluation:** autoresearch_3 benchmark (1000 Axis images), 1 step, ground truth hff, every 30 min on a generator
  GPU (~2 min).

## Results (1 step, ground truth hff, cfg 1, EMA unless noted)

| hours | 0.5 | 1 | 2 | 3 | 4 | 4.5 | 5 | 5.5 | 6 (final) |
|---|---|---|---|---|---|---|---|---|---|
| patch_fid | 2.66 | 2.48 | 2.38 | 2.35 | 2.32 | 2.30 | 2.27 | 2.24 | **2.21** |
| fid | 3.14 | 2.93 | 2.83 | 2.79 | 2.75 | 2.73 | 2.69 | 2.65 | 2.62 |
| low_res_mse 1e-4 | 7.97 | 7.81 | 7.67 | 7.62 | 7.57 | 7.55 | 7.52 | 7.50 | 7.48 |

| final | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | 1-step model latency (512 -> 1024) | peak MB |
|---|---|---|---|---|---|---|
| small EMA cfg 1 | 2.21 | 2.62 | 7.48 | 1.931 | 5.9 ms | 130 |
| small raw cfg 1 | 2.21 | 2.64 | 7.48 | 1.932 | | |
| small EMA cfg 2 | 2.24 | 2.64 | 7.44 | 1.935 | | |
| large student (192M) raw cfg 1 | 1.76 | 2.02 | 6.63 | 1.826 | 36.6 ms | 652 |
| teacher, 30 steps | 1.69 | 1.93 | 6.46 | 1.808 | 30 x 36.6 ms | |
| pixel bicubic / VAE oracle | 8.12 / 0.71 | 1.77 / 0.72 | 5.0 / 1.0 | 1.74 / 1.08 | | |

Latency: model only, bf16, compiled, batch 1, RTX 5090 (the FLUX.2 decode at 1024 is not included).

## Findings

1. **The 20M student keeps ~93% of the large student's gain over bicubic** (2.21 vs 1.76 patch_fid; Wan s20m kept
   ~65%: 3.79 vs 2.87, bicubic 5.49) at 6x lower latency (5.9 vs 36.6 ms) and 5x less memory. It beats the best
   48M autoresearch_3 model with 30 steps (2.45).
2. **Not saturated:** patch_fid fell ~0.02 per 30 min at constant LR (3-4.5 h) and ~0.03 per 30 min in the decay.
   A longer run would still help; the decay gave -0.09 in its 1.8 h.
3. **Raw = EMA at the end** (EMA 0.999 is short relative to 110k steps).
4. **CFG input:** cfg 2 costs +0.03 patch_fid (large student +0.02), slightly lower low_res_mse.
5. **Fidelity cost of the small size:** low_res_mse 7.48 vs 6.63 (large), chroma 1.93 vs 1.83.
6. **Throughput layout:** 4 + 4 used all 8 GPUs at ~100%; generator input loading must not decode unused data
   (pretraining cache shards are ~80 MB).

Final checkpoint: `distill_runs/flux_small_s20m/checkpoints/checkpoint_0109780.pt` (local and R2
`checkpoints/flux_small_s20m/`), architecture `FluxStudentTransformer`, `model_config` = `SMALL_MODEL`.
