# FLUX.2 upscaler: single step rectified flow distillation (2026-11-24, 8x RTX 5090)

Code: `src/stdev/flux_distill/` (config, student, pairs, trainer, evaluation), launcher `scripts/run_flux_distill.py`.
Teacher: `flux_pretrain_v1` final raw weights (step 129590; 30 Euler steps, shift 2: patch_fid 1.69 on the
autoresearch_3 benchmark). Run: `distill_runs/flux_distill_v1`.

## Setup

- **Online pairs (7 GPUs):** generators take low resolution latents from the pretraining cache (`flux_data`, 90%
  512 -> 1024 / 10% 256 -> 512, including the 20% corrupted inputs), draw hff true 45% / uniform random 45% / null
  10%, CFG scale uniform in [1, 2] for half of the non-null samples (negative = null hff), and a noise seed, then run
  the teacher (30 steps, shift 2; guided samples in a doubled batch). Shards store the LR latents, the teacher latents
  (fp16), the conditions and the seed (the noise is regenerated bit exactly). ~0.75 pairs/s per GPU, ~5.3 pairs/s in
  total; generated residual std ~1 for all condition groups (smoke check).
- **Student (GPU 8, same time):** teacher architecture plus a zero initialized guidance embedding (sinusoidal MLP of
  guidance - 1, added before the condition RMS norm); all other weights from the teacher (raw). At init it equals the
  teacher velocity at sigma 1. Objective at sigma 1: target velocity = noise - residual(teacher latents); inference
  residual = noise - velocity (1 step).
- **Training:** starts at 1,500 pairs (~6 min of generation), 20 min fixed time, micro batch 2 @1024 / 8 @512,
  accumulation 4 (~8.7 samples per step), AdamW lr 3e-4 (50 warmup steps), linear decay over the last 30%,
  wd 0.01, EMA 0.995, ~7.5 samples/s. 1,083 steps, 9.4k samples seen, 7.4k pairs produced (mean reuse 1.26).
  Snapshots every 4 min.
- **Evaluation:** autoresearch_3 benchmark (1000 Axis images), 1 step, ground truth hff, fixed noise seed per image,
  run on all 8 GPUs in parallel after training (~2.7 min per evaluation).

## Results (1 step, ground truth hff)

| snapshot | minutes | samples | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | latent_mse_hr |
|---|---|---|---|---|---|---|---|
| init (teacher weights, 1 step) | 0 | 0 | 6.50 | 3.35 | 5.35 | 1.757 | 0.177 |
| EMA cfg 1 | 4 | 1.9k | 3.75 | 2.66 | 5.46 | 1.711 | 0.223 |
| EMA cfg 1 | 8 | 3.8k | 2.38 | 2.26 | 6.06 | 1.763 | 0.274 |
| EMA cfg 1 | 12 | 5.6k | 1.98 | 2.12 | 6.38 | 1.795 | 0.299 |
| EMA cfg 1 | 16 | 7.5k | 1.85 | 2.07 | 6.52 | 1.808 | 0.309 |
| EMA cfg 1 | 20 (final) | 9.4k | 1.815 | 2.04 | 6.57 | 1.815 | 0.314 |
| EMA cfg 2 | 20 | | 1.823 | 2.06 | 6.52 | 1.815 | 0.312 |
| **raw cfg 1** | 20 | | **1.76** | **2.02** | 6.63 | 1.826 | 0.317 |
| raw cfg 2 | 20 | | 1.78 | 2.05 | 6.58 | 1.828 | 0.315 |
| teacher, 30 steps (raw) | | | 1.69 | 1.93 | 6.46 | 1.808 | 0.314 |
| pixel bicubic / VAE oracle | | | 8.12 / 0.71 | 1.77 / 0.72 | | | |

## Findings

1. **20 minutes on one training GPU are enough for a near lossless 1-step student:** raw 1.76 vs teacher 1.69
   patch_fid, ~99% of the teacher's gain over pixel bicubic (Wan: ~96% with 40k pre-generated pairs and 8 training
   GPUs). Roundtrip metrics are equal to the teacher's (low_res_mse 6.63 vs 6.46, chroma 1.83 vs 1.81).
2. **Still improving at the end:** EMA patch_fid 1.98 -> 1.85 -> 1.815 at 12 / 16 / 20 min, with the decay adding
   to the last step. A longer run (30-40 min) would probably close more of the remaining 0.07.
3. **Raw beats EMA (1.76 vs 1.815):** at 1k steps the EMA (0.995, ~200 steps horizon) still lags behind the decaying
   raw weights. Use raw.
4. **CFG input works but costs a little on FID:** cfg 2 is +0.02 patch_fid vs cfg 1 (Wan: +0.03), as expected for a
   distribution metric; its visual effect (sharpness) has to be judged in ComfyUI. The teacher with cfg <= 2 changes
   the output only moderately (hff changes the velocity by 8-10% for 0.1 vs 0.9).
5. **Online generation removes the separate pair phase:** 7 generator GPUs (5.3 pairs/s) nearly keep up with one
   training GPU (7.5 samples/s, reuse 1.26); total wall time ~35 min incl. pair warmup and evaluation, vs ~4.4 h for
   the Wan distillation (2.1 h pairs + 2.3 h training).

Final checkpoint: `distill_runs/flux_distill_v1/checkpoints/checkpoint_001083.pt` (local and R2
`checkpoints/flux_distill_v1/`), architecture `FluxStudentTransformer`, `model_config` as the teacher (raw and EMA
weights inside; use raw).
