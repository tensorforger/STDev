# Wan 2.1 upscaler: single step rectified flow distillation (2026-11-18, 8x RTX 5090)

Code: `src/stdev/wan_distill/` (pairs, teacher, student, data, evaluation, trainer), pair generator
`scripts/generate_wan_distill_pairs.py`. Teacher: `wan_pretrain_v1` EMA, step 161590 (20 Euler steps, shift 4:
fid_patch 2.76 on `wan_eval_1000`).

## Pair dataset (`pretrain_runs/wan_distill_pairs`, 48 GB)

- 39,999 pairs from the pretraining latent cache in the pretraining production shares: image 512 -> 1024 27.8k,
  video 17 -> 33 @512 8.2k, image 256 -> 512 3.1k, each 1024 video type ~300.
- Per pair: LR latents, teacher initial noise (unit gaussian, 8-bit block quantized before sampling, so the stored
  noise is exactly the teacher input), teacher latents (fp16), hff used (true 45% / uniform random 45% / null 10%),
  true hff, CFG scale (half of the non-null pairs uniform in [1, 3], negative = null hff), noise scale (1.0 spatial,
  0.8 temporal only). The real target stays in the cache (plan json per chunk).
- Teacher throughput (one GPU, s/sample unguided / guided): image 1024 0.63 / 1.27, image 512 0.16 / 0.32,
  video 33@512 1.77 / 3.56, 1024 video ~4.0 / 8.0. Generation took 2.07 h on 8 GPUs (~15% slower than measured).
- Smoke check: stored conditions match the plan, re-sampling from the stored noise reproduces the stored output
  within 1-5% of the teacher-vs-real distance (bf16 batch noise); generated residual std close to real.

## Student and training (`distill_runs/wan_distill_v1`)

- Teacher architecture plus a third scalar condition (guidance scale, sinusoidal MLP of guidance - 1, zero
  initialized last layer, added before the condition RMS norm); all other weights from the teacher EMA. At init the
  student equals the teacher velocity at sigma 1 for any guidance.
- Objective: sigma fixed at 1, input = noise_scale * stored noise, target velocity = input - residual(teacher latents);
  inference residual = input - velocity (1 step).
- 8 GPU DDP, same pair type per micro-step, pretraining micro-batches, accumulation 4 (64 images at 1024 per step),
  716 steps per epoch, AdamW lr 1e-3 (200 warmup steps), wd 0.01, EMA 0.999 on all ranks, ~38 samples/s.
  The decay control started after epoch 5 (one epoch linear decay), end at step 4306 (~2.3 h incl. benchmarks).

## 1000-image benchmark (wan_eval_1000, 1 step, true hff; fid_patch / fid / mse_lr 1e-4)

| | EMA cfg 1 | EMA cfg 2 | raw cfg 1 |
|---|---|---|---|
| init (teacher weights, 1 step) | 5.93 / 2.71 / 7.95 | = | = |
| epoch 1 | 2.89 / 2.09 / 9.89 | 2.90 / 2.11 / 9.93 | 2.65 / 2.17 / 10.21 |
| epoch 2 | 2.86 / 2.08 / 9.89 | 2.88 / 2.12 / 9.91 | 3.26 / 2.17 / 10.03 |
| epoch 3 | 2.85 / 2.06 / 9.89 | 2.87 / 2.10 / 9.94 | 2.48 / 2.09 / 10.11 |
| epoch 4 | 2.88 / 2.07 / 9.89 | 2.89 / 2.10 / 9.92 | 2.59 / 2.12 / 10.17 |
| epoch 5 | 2.85 / 2.07 / 9.89 | 2.88 / 2.10 / 9.94 | 3.48 / 2.22 / 9.94 |
| epoch 6 (in decay) | 2.88 / 2.07 / 9.92 | 2.90 / 2.10 / 9.95 | 2.87 / 2.07 / 9.93 |
| final (decay end) | **2.87 / 2.06 / 9.92** | 2.90 / 2.10 / 9.95 | 2.86 / 2.07 / 9.93 |

References: teacher 20 steps 2.76 / 2.00 / 9.76, pixel bicubic 5.49, VAE oracle 1.37.

## Findings

1. One epoch (~40k pairs, ~15 min) is enough: the EMA student is flat at 2.85-2.89 from epoch 1 on. The 1-step
   student keeps ~96% of the teacher's gain over bicubic (2.87 vs 2.76) with the same mse_lr.
2. Raw weights swing by +-0.5 fid_patch between epochs (2.48 -> 3.48) at constant lr 1e-3, while the EMA is stable:
   the lr is high for this fine-tuning (noise, not progress). The decay ends where the EMA is (raw 2.86 = EMA 2.87).
   The best raw values (2.48, 2.65) are probably lucky points of the oscillation; a lower lr (3e-4) or a shorter run
   with an earlier decay is likely enough.
3. CFG input: cfg 2 is equal or slightly worse than cfg 1 on fid (+0.03 fid_patch). The teacher's CFG <= 3 changes
   the output very little (hff changes the velocity by 1-4%), so the student has little to learn; the condition
   works but its visual effect needs checking in ComfyUI.
4. Visual (benchmark image 3): clean and sharper than the 1-step teacher, no artifacts; small text is still garbled
   (teacher limitation).

Final checkpoint: `distill_runs/wan_distill_v1/checkpoints/checkpoint_0004306.pt` (local and R2
`checkpoints/wan_distill_v1/`), architecture `WanStudentTransformer` (`model_config` as the teacher).
