# FLUX.2 latent upscaler pretraining (8x RTX 5090, 2026-11-23 / 24)

Code: `src/stdev/flux_pretrain/` (model, config, corruption, cache, producer, trainer, benchmark), launcher
`scripts/run_flux_pretrain.py`. Recipe: `autoresearch_3/experiment_logs.md` (RECOMMENDED PRETRAIN RECIPE, run_0016),
scaled. Dev tools: `scripts/flux_tools/flux_pretrain_full_benchmark.py` (autoresearch_3 benchmark of a checkpoint),
`scripts/flux_tools/flux_model_diagnostics.py`, `scripts/flux_tools/flux_hff_sensitivity.py`, `scripts/flux_tools/flux_generalization_gap.py`,
`scripts/flux_tools/flux_pretrain_curves.py`, `scripts/flux_tools/flux_pretrain_residual_std.py`.

## Setup

- **Model (192.2M):** run_0016 window transformer at width 768, depth 18, head dim 64, MLP 3072 with depthwise 3x3
  conv, condition dim 512, source encoder width 512 (2 residual blocks, pixel shuffle). Condition RMS norm, QK-norm,
  normalized residual flow, hff condition with 10% dropout to a learned null embedding (CFG). Width 768 / depth 16
  (the Wan shape) is only 173M without the Wan temporal layers, so depth 18 was used.
- **Residual table:** one per channel table for both buckets, measured on 4.3k produced samples (mean std 0.703).
  The 256 -> 512 bucket differs by only +3.5% (Wan needed per type tables; FLUX.2 does not), and the bucket losses
  are equal during training (0.740 vs 0.744 at 8.5 h).
- **Buckets:** 90% 512 -> 1024, 10% 256 -> 512 (by sample count).
- **Data (online):** ReLAION-high-resolution (urls, min side 1024) : 4KLSDB `train_x4` hr images = 1 : 1.
  4KLSDB (206 parquet files, 879 GB) is streamed from the Hugging Face Hub (~270 MB/s, ~1 h per pass, fresh random
  crop per pass). Square crops of fraction 1 - 0.5 u^2 of the short side (>= 50%, median ~85%), area resized.
- **Corruption (source only):** 20% of samples: JPEG 6% (quality 95 - 75 u^2: mostly almost invisible, q20-50 as
  the tail), nearest 4%, antialias 4%, bicubic 3%, lanczos 3%; the rest cv2 bilinear (= benchmark input).
- **Process layout:** 7 training GPUs (DDP, same bucket per micro-step), 1 encoder GPU (compiled FLUX.2 encoder,
  threaded feeding / shard writing: 12.8 -> 19 samples/s) that also runs the visual benchmark. Rolling 250 GB cache.
- **Schedule (fixed 24 h training time, synchronized across ranks):** warmup 20k units (~7 min), micro batch 2 @1024 /
  8 @512, accumulation x1 -> x2 at 15% (3.6 h) -> x4 at 45% (10.8 h) with sqrt LR (base 1.3e-3, peak 2.6e-3),
  linear decay to 0 over the last 30% (from 16.8 h). AdamW (0.9, 0.95), weight decay 0.05 (Wan: 0.01), grad clip 1,
  bf16, EMA 0.9999.
- **Visual benchmark:** 30 fixed Axis-V1 images (evenly spaced over the first 1000 of the autoresearch_3 order,
  same names) every 30 min, ground truth hff, guidance 1, 30 steps / shift 2, raw and EMA: low_res_mse, chroma dE00,
  latent_mse_hr, images saved. No FID (too few images).

## Run `flux_pretrain_v1` (data run `flux_data`)

24.00 h of training, step 129,590, 4.18M units (4.52M samples), no crash, restart or hang. Throughput 47-49 units/s
(7 GPUs), peak memory 27.0 GB. Producer 1.53M samples (ReLAION / 4KLSDB exactly 1 : 1, corruption shares as
configured), mean reuse 2.96. Final checkpoint: local and R2 `checkpoints/flux_pretrain_v1/checkpoint_000129590.pt`.
The encoder was paused twice (~40 and ~30 min) for full benchmarks on its GPU.

### Full benchmark (autoresearch_3 set, 1000 Axis-V1 images, ground truth hff, 30 steps, shift 2)

| checkpoint | patch_fid | fid | low_res_mse 1e-4 | chroma dE00 | latent_mse_hr |
|---|---|---|---|---|---|
| VAE oracle (reproduced: 0.71 / 0.72 / 1.0 / 1.08 as in autoresearch_3) | 0.71 | 0.72 | 1.0 | 1.08 | 0 |
| 8.5 h EMA (constant LR) | 2.15 | 2.17 | 6.71 | 1.852 | 0.335 |
| 14 h EMA (after the x4 step) | 2.18 | 2.15 | 6.75 | 1.854 | 0.334 |
| **24 h raw (decay end)** | **1.69** | **1.93** | **6.46** | **1.808** | **0.314** |
| 24 h EMA | 1.89 | 1.95 | 6.58 | 1.815 | 0.326 |
| autoresearch_3 run_0016 (same recipe, 48M, 30 min) | 2.58 | 3.17 | 9.32 | 2.12 | - |
| pixel bicubic (autoresearch_3) | 8.12 | 1.77 | 5.0 | 1.74 | - |

The research models had no hff input; with ground truth hff vs null the velocity differs by < 1% (below).

### Training curves (hour window means)

| hours | 1 | 4 | 8 | 10 | 14 | 18 | 22 | 24 |
|---|---|---|---|---|---|---|---|---|
| loss | 0.782 | 0.757 | 0.744 | 0.743 | 0.737 | 0.734 | 0.720 | 0.710 |
| loss ReLAION / 4KLSDB | 0.635 / 0.928 | 0.612 / 0.902 | 0.597 / 0.892 | 0.596 / 0.890 | 0.589 / 0.886 | 0.586 / 0.882 | 0.571 / 0.869 | 0.562 / 0.858 |
| raw weight norm | 951 | 1232 | 1362 | 1368 | 1437 | 1450 | 1254 | 1147 |
| visual EMA low_res_mse 1e-4 | 8.19 | 7.76 | 7.51 | 7.47 | 7.57 | - | - | 7.27 (raw 7.15) |

Loss by source variant at the end: bilinear 0.699, lanczos 0.696, antialias 0.709, bicubic 0.709, nearest 0.740,
JPEG 0.837. 4KLSDB samples are much harder than ReLAION (denser detail).

## Findings

1. **The LR decay was essential.** Patch FID was flat at 2.15-2.18 through the constant LR phase (8.5 -> 14 h) and fell
   by 22% to 1.69 during the 30% decay, with every other metric improving. This differs from the Wan pretrain, where the
   decay improved only the loss and mse_lr; the phase 1 finding ("the decay is essential") holds at this scale.
2. **Raw beats EMA after the decay** (1.69 vs 1.89 patch_fid): EMA 0.9999 still averages weights from the high LR
   phase. Use the raw final weights (teacher for distillation, release). Next run: shorter EMA or no EMA in the decay.
3. **The x4 batch step with sqrt LR (2.6e-3) slightly hurt the fidelity metrics:** the visual EMA low_res_mse had its
   minimum at ~9.5 h and rose by 1.5% after the step (10.8 h), the raw weight norm jumped 1368 -> 1437, while the loss
   kept falling; patch FID stayed flat (2.15 -> 2.18, within noise). Larger LR = higher weight noise floor. Next run:
   keep the LR at the x2 level for the x4 step, and try a longer decay (40-50%), where most of the late gain came from.
4. **Stability fixed (weight decay 0.05 + QK-norm + condition norm):** raw weight norm plateaued at ~1365 (8-10 h,
   before the LR step; Wan: steady growth to 3162). Residual stream RMS at the last block 18.9 / 21.5 / 22.6 / 22.6 at
   3 / 6 / 8.5 / 14 h. Max feed forward adaLN gate ~7.4 (Wan 13-15); the last block's feed forward stays active
   (gate 2-4; Wan: off). Grad norm spikes <= 0.2 after warmup.
5. **Attention heads all healthy (216 heads):** no dead or collapsed head; output share per head >= 2.3%, mean entropy
   64-98% of uniform, max mean peak probability 0.39. Content logits (after QK-norm) flat in blocks 0-11 (max 27.5 in
   block 6), slowly growing in blocks 12-15 (9-11 at 3 h -> 11-14 at 14 h); one block 16 head sharpened (peak
   probability 0.19 -> 0.30). Not a problem at this length.
6. **No memorization at 3x reuse:** paired loss of the 8.5 h and 14 h EMA on the same cache shards: the 14 h model is
   better by 0.012-0.037 on shards it trained on and by 0.018 on the newest shards neither model saw.
7. **hff conditioning is much stronger than in the Wan model:** hff 0.9 vs 0.1 changes the velocity by 7.6-9.9%
   (Wan 1.7-3.6%); ground truth vs null hff by 0.7-0.8% (the null embedding acts as an average hff). CFG on hff should
   need much smaller scales than for Wan.
8. **Data:** 4KLSDB streaming works without a local copy; 4KLSDB stays much harder than ReLAION through training.
   The single encoder GPU (19 samples/s) gave ~3x reuse without measurable harm.

## Infrastructure notes

- The full benchmark needs batch 2 on the encoder GPU (only ~9 GB free next to the encoder and the visual benchmark
  watcher) and takes ~22 min; ~20 min on a free GPU at batch 8 (30 steps, 1000 images).
- Schedule quantities (accumulation, LR, end) use the training time synchronized at control steps: per rank clocks
  differ slightly and an accumulation mismatch between ranks would deadlock DDP.
- `pkill -f <pattern>` from an agent shell kills the agent's own shell when the pattern is in the command line; kill by
  PID.
