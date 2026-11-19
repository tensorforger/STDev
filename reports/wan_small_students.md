# Wan 2.1 upscaler: small single step students (size ladder 19.5M -> 0.52M, 2026-11-19, 8x RTX 5090)

Code: `src/stdev/wan_distill/small_config.py` (ladder, schedule), `online_pairs.py` (online pair generator, rolling
cache), `small_trainer.py` (single GPU trainer), launcher `scripts/run_wan_small_distill.py`; latency
`dev/measure_small_students.py`. Runs: `distill_runs/wan_small_<name>`, final checkpoints also on R2
`checkpoints/wan_small_<name>/`.

## Setup

- Architecture: the large student (`WanStudentTransformer`, window transformer + guidance condition) scaled down in
  width and depth, MLP 4x width, trained from scratch, one GPU per model.

| name | width x depth | head dim | params | lr |
|---|---|---|---|---|
| s20m | 256 x 10 | 64 | 19.5M | 1.5e-3 |
| s10m | 192 x 10 | 32 | 10.8M | 1.7e-3 |
| s5m | 128 x 10 | 32 | 5.0M | 2.1e-3 |
| s2m | 96 x 8 | 32 | 2.3M | 2.4e-3 |
| s1m | 64 x 8 | 32 | 1.1M | 3e-3 |
| s05m | 48 x 6 | 16 | 0.52M | 3e-3 |

- Teacher: the large 1-step student `wan_distill_v1` (fid_patch 2.87), run online on GPUs 6-7 (~38 samples/s total) on
  LR latents of the pretraining cache (pretraining type shares; hff true / random / null 45 / 45 / 10%, CFG U[1, 3] for
  half of the non-null), 40 GB rolling cache. Noise stored as a seed and regenerated on the trainer GPU (bit exact
  across GPUs). Pair reuse per model: ~1x (s20m) to ~8x (s05m).
- Objective as the large distillation (sigma 1, velocity = input - residual of the teacher latents). Step batch 8x
  the pretraining micro batch (16 images at 1024), one pair type per step with the pretraining mix, AdamW (0.9, 0.95),
  wd 0.01, 1000 warmup steps, constant lr = 1.5e-3 x sqrt(256 / width) (cap 3e-3), EMA 0.999.
- Benchmark: wan_eval_1000, 1 step, true hff, cfg 1, EMA weights, every 30 min of training (~3 min each on the
  training GPU). The user started the decays (30% more steps) at ~7 h; all runs ended at ~9.1 h of training.

## Results (fid_patch, EMA, 1 step)

| | 1 h | 3 h | 5 h | 7 h (decay start) | final 9.1 h | fid | mse_lr 1e-4 | steps |
|---|---|---|---|---|---|---|---|---|
| s20m | 6.15 | 4.70 | 4.31 | 4.04 | **3.79** | 2.70 | 11.40 | 96k |
| s10m | 6.19 | 4.81 | 4.42 | 4.41 | **3.98** | 2.82 | 11.60 | 124k |
| s5m | 6.08 | 5.21 | 4.99 | 4.85 | **4.46** | 3.00 | 11.98 | 216k |
| s2m | 6.72 | 5.93 | 5.76 | 5.60 | **5.09** | 3.36 | 12.61 | 337k |
| s1m | 7.83 | 7.21 | 6.95 | 6.89 | **6.22** | 3.87 | 13.28 | 530k |
| s05m | 9.97 | 9.33 | 9.05 | 8.90 | **8.00** | 4.62 | 14.36 | 752k |

References: large 1-step student 2.87 / fid 2.06 / mse_lr 9.92, teacher 20 steps 2.76, pixel bicubic 5.49, VAE oracle
1.37. Research best (45M, 14 steps, autoresearch_2): 3.77.

Model-only 1-step latency (bf16, compiled, batch 1, image 512 -> 1024 / video 17 -> 33 @512) and peak memory:

| | large | s20m | s10m | s5m | s2m | s1m | s05m |
|---|---|---|---|---|---|---|---|
| image ms | 30.4 | 4.6 | 8.9 | 9.2 | 7.7 | 7.6 | 6.1 |
| video ms | 89.8 | 9.7 | 14.7 | 11.4 | 9.5 | 9.4 | 7.7 |
| image MB | 716 | 130 | 115 | 114 | 64 | 49 | 46 |

## Findings

1. Quality falls smoothly with size: 19.5M 3.79, 10.8M 3.98, 5.0M 4.46, 2.3M 5.09 (still better than bicubic 5.49),
   1.1M 6.22 and 0.52M 8.00 (worse than bicubic on patch FID). ~2M is the smallest size that beats bicubic; ~5M keeps
   most of the gain. s20m in one step equals the research best 45M model with 14 steps.
2. The gap to the teacher (2.87) is large even at 19.5M (+0.9), and no run had saturated before the decay: the 20M and
   10M curves were still falling ~0.1 per hour at 7 h, and the decay gave a further -0.25 to -0.9 (largest for the
   smallest models). Longer training (and a higher lr for the bigger ones) would likely help s20m / s10m.
3. Below ~20M, smaller is not faster: s10m-s05m take 6-9 ms per 1024 image vs 4.6 ms for s20m. These models are
   launch / overhead bound, and head dim 32 (vs 64 in s20m) is less efficient. For speed the head dim should stay 64 and
   CUDA graphs would remove the overhead; memory still shrinks with size (130 -> 46 MB).
4. Visual (benchmark image 3): all sizes keep the global structure; fine detail and small text degrade with size
   (the large student keeps the small text partly readable, s20m garbles it, s05m smears it).
5. mse_lr grows as models shrink (9.9 large -> 14.4 s05m): small models also lose fidelity to the low resolution input.
