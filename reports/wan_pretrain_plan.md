# Wan 2.1 latent upscaler pretraining (8x RTX 5090)

Code: `src/stdev/wan_pretrain/` (model, config, cache, producer, trainer, benchmark), launcher
`scripts/run_wan_pretrain.py`. Recipe: `autoresearch_2/experiment_logs.md` (PRODUCTION RECOMMENDATION), scaled.

## Model

Window transformer of run_0072 scaled to width 768, depth 16, head dim 64, GELU MLP 3072 with depthwise 3x3
conv, condition dim 512, source encoder width 512: **222M parameters** (first plan was 516M at 1024 x 24: needs full
gradient checkpointing, 2.5 units/s/GPU; reduced by user decision, size necessity unknown). All stabilizers kept (condition RMS
norm, QK-norm), causal temporal conv, learned source path, residual parameterization. New: hff dropout (10%)
to a learned null embedding (CFG with negative = no hff).

## Measured (2026-11-16, this machine)

| What | Number |
|---|---|
| Full Wan VAE encode, 1024 / 512 / 256 px | 12.6 / 52 / 215 images/s (no gain from batching at 1024) |
| Tiny VAE encode, 33x512 / 17x512 / 17x1024 / 9x1024 | 32 / 60 / 15 / 25 clips/s |
| Producer, one GPU (mix below) | ~6-10 samples/s (encoder bound) |
| Training without gradient checkpointing | OOM even at 1 image of 1024 px |
| Training, all 24 blocks checkpointed | 44k target tokens/s/GPU (2 images of 1024 px: 743 ms, 8.8 GB) |
| Selective checkpointing (16 / 24 blocks) | 50k tokens/s/GPU, 19.7 GB |
| DDP 2 GPUs, accumulation 4, full mix | ~2.5 units/s/GPU (1 unit = 128 x 128 target latent tokens = one 1024 px image) |
| 222M (768 x 16), no checkpointing, 2 images of 1024 px | 134k tokens/s/GPU, 22.9 GB; video 512 needs 7 / 16 blocks checkpointed (19.9 GB) |
| 222M, DDP 6 GPUs, full mix | ~37 units/s total (6.2 units/s/GPU) |
| NCCL all-reduce 444 MB, 6 GPUs (no P2P on this box, SHM) | 26 ms with NCCL_CUMEM_HOST_ENABLE=0; default cuMem host SHM fails across NUMA nodes (and likely caused a probe hang) |
| 100 image benchmark references | VAE oracle fid_patch 6.43, pixel bicubic 16.4 (small set: biased high) |

## Data

- Images: ReLAION-high-resolution (all shards except 127, min side >= 1024), random square crop with side
  uniform in [target, short side], area resize. 90% 512 -> 1024, 10% 256 -> 512 (by image count).
- Videos: vk-50k (S3, 1024x1024, 129 frames at 24 fps), last 500 keys held out, random window per download.
  By count: 90% 17 -> 33 frames at 512 (random crop side in [512, 1024]), 3.3% each: 9 -> 17 frames at 1024,
  17 frames 512 -> 1024, 9 frames at 512 -> 17 frames at 1024 (temporal + spatial). The 1024 video targets
  are 17 frames (33 frames at 1024 does not fit memory).
- Sources are built from the target pixels (2x box downscale = bilinear 2x, and / or every 2nd frame).
- Image : video target tokens = 5 : 4 (research mix). Micro-step probabilities: image 512 -> 1024 0.60,
  image 256 -> 512 0.02, video 512 0.35, each 1024 video type 0.013.
- Encoder GPUs 6 and 7 (user decision: 6 train + 2 encode) encode everything: it pulls images or videos by deficit against the production shares
  (= consumption shares, so every type is reused equally). Rolling cache of 150 GB, ~2x average reuse expected.

## Training

- 6 GPUs DDP; every micro-step all ranks train the same pair type (equal step time, no straggler); micro
  batch = 2 images @1024, 8 @512, 1 video. Global batch starts at ~13 units, accumulation ramp x2 at 0.5M
  units, x4 at 1.5M units, LR x sqrt(accumulation).
- AdamW (0.9, 0.95), wd 0.01, grad clip 1, bf16, warmup 20k units, constant LR, EMA 0.9999 (rank 0).
- No automatic stop: `touch pretrain_runs/RUN/control/decay` starts a linear decay to 0 over 30% more steps
  (or write the step count into the file). `control/stop` checkpoints and stops; relaunching resumes.
- Checkpoints every 30 min (keep 2), uploaded to R2 `checkpoints/RUN/` every 3 h (keep 2).
- Benchmark every training hour on GPU 6 (shared with an encoder): 100 held-out ReLAION images 512 -> 1024,
  20 Euler steps, shift 4, temperature 1.0, raw and EMA weights, fid_patch / fid / mse_lr; all 100 generated images saved per benchmark
  (`pretrain_runs/RUN/benchmarks/<snapshot>/`, bicubic inputs in `benchmarks/inputs/`, ground truth in the data
  run `benchmark_set/images/`).

## LR

Probe on the 516M model (3 x 2 GPUs, global ~17 units, LR 5e-4 / 1e-3 / 2e-3): loss nearly identical at 4k-8k units
(0.711 / 0.710 / 0.713; 1e-3 slightly best on video), stopped early. The 222M run uses 1.3e-3 (1e-3 x 1024 / 768)
without a new probe (user decision).

## Run

`wan_pretrain_v1` (data run `wan_data`), launched 2026-11-16 ~20:50 UTC:
`python scripts/run_wan_pretrain.py --run-name wan_pretrain_v1 --data-run-name wan_data --learning-rate 1.3e-3`.

## Status at 26.8 h (step 114.7k, 3.54M units, constant LR, before decay)

Health: no crash / restart / hang since launch, no loss spikes (grad norm max <= 0.3 after hour 3), 37 units/s
flat, peak memory 23.6 GB, 8 checkpoint uploads to R2. Producer 18.5 samples/s, 1.89M samples produced (1.32M
image 512 -> 1024), mean reuse 1.45.

### 1000-image evaluation (dev/evaluate_1000.py, EMA weights of step 114.7k, held-out shard 127)

| | fid_patch | fid | mse_lr |
|---|---|---|---|
| VAE oracle | 1.37 | 0.58 | 2.0e-4 |
| pixel bicubic (of the exact 2x box downscale) | 5.49 | 0.26 | 1.0e-4 |
| EMA, 20 steps, shift 4, T 1.0 (benchmark sampler) | **2.78** | 2.06 | 1.02e-3 |
| EMA, 14 steps, shift 4, T 0.9 (research sampler) | 3.32 | 2.09 | 0.93e-3 |

Research reference (autoresearch_2, 1000 images, oracle 1.30): best 45M model after decay 3.77. Gap to the oracle
1.41 here (before decay) vs 2.47 in research. T 1.0 / 20 steps beats the research sampler by 0.54.

### Findings

1. The 100-image benchmark is strongly biased (oracle 6.43, bicubic 16.4) and compresses differences: the model
   looks almost bicubic level there (EMA 15.36) but is halfway between bicubic and oracle at 1000 images. Use it
   only for trends; use dev/evaluate_1000.py (~1 h on a shared encoder GPU) for absolute numbers.
2. EMA 100-image fid_patch improves about log-linearly in training time: 17.59 (3 h), 16.70 (6 h), 16.13 (12 h),
   15.43 (24 h), i.e. -0.6 to -0.7 per doubling, no saturation yet. Image training loss gains shrink per doubling
   (6 -> 12 h -0.011, 12 -> 24 h -0.007). Raw-weight benchmark noise is +-1 (single outlier 18.66 at 11 h): decide
   on EMA only.
3. Video 17 -> 33 at 512 loss is flat since ~16 h (0.183-0.185 -> 0.181-0.182) while image and 1024 temporal
   losses keep falling. vk-50k has 49.5k training clips; each was used ~8x (386k samples, random windows / crops).
   Possible video data limit.
4. Residual normalization is wrong for tiny-VAE spatial video: normalized target std 1.55 (17 frames 512 -> 1024)
   and 2.04 (9 @512 -> 17 @1024) vs 1.01 images, 1.11-1.13 temporal. The spatial table was measured on full-VAE
   images; the combined type also carries temporal residual. Explains their ~1.6-1.8x higher losses. Only ~4% of
   micro-steps. Fix in the next run (per pair type tables); do NOT change the tables for this checkpoint (inference,
   incl. the ComfyUI node, must use the tables it was trained with).
5. Weight norm grows steadily (806 at 0 h, 2119 at 4 h, 3162 at 26 h; all linear layers alike) and the grad norm fell
   from 0.35 to 0.037: AdamW wd 0.01 is weak at this LR. Stable (QK-norm, condition norm), but the relative update
   size shrinks over time; consider wd ~0.05-0.1 next run.
6. Batch ramp events (x2 at 4 h, x4 at 11.5 h, LR x sqrt) caused no instability; EMA kept improving.
7. Visual: sharper than bicubic, but small text is still garbled (benchmark images 17, 42: Chinese sign, MICHELIN),
   improving only slowly between 3 h and 26 h (open problem 4).

## End of pretraining (stopped 2026-11-18 at 45.7 h, step 161.6k, 6.05M units)

LR decay started at step 138.7k (38.5 h, 41.6k decay steps); the run was stopped at 55% of the decay (user decision:
no fid gain). Final checkpoint `checkpoint_000161590.pt` (local + R2 `checkpoints/wan_pretrain_v1/`, with 159955).
The latent cache (`pretrain_runs/wan_data/cache`, 151 GB) is kept for rectified flow distillation.

### Paired 1000-image evaluation (benchmark sampler, final EMA on the set each earlier checkpoint was evaluated on)

| set | earlier checkpoint | fid_patch earlier -> final EMA | final raw | mse_lr (1e-4) earlier -> final EMA | bicubic / oracle |
|---|---|---|---|---|---|
| wan_eval_1000 | 114.7k (26.8 h, constant LR) | 2.779 -> 2.756 | 2.746 | 10.23 -> 9.76 | 5.49 / 1.37 |
| wan_eval_1000_2 | 137.4k (pre decay) | 2.849 -> 2.814 | 2.869 | 9.44 -> 9.17 | 5.24 / 1.38 |
| wan_eval_1000_3 | 158.7k (decay) | 2.820 -> 2.804 | 2.866 | 9.24 -> 9.22 | 5.38 / 1.39 |

- Patch FID saturated at ~2.8 from ~27 h on (-0.02 to -0.04 over the last 47k steps, near inference noise). The
  unpaired series 2.78 -> 2.85 -> 2.82 was set-to-set noise: the three sets differ beyond the first ~99 images
  (non deterministic download) and their bicubic references span 0.25.
- What still improved: mse_lr (fidelity to the low resolution content, -4.6% from 114.7k) and the training loss.
  During the decay the loss fell faster than in any constant LR period (image 0.361 -> 0.349 and video 512
  0.179 -> 0.166 from 36 h to 45 h, vs -0.007 / -0.006 from 12 h to 24 h) and broke the video plateau, but none of
  it shows in patch FID or visually (user: 90k vs 150k nearly identical in ComfyUI). EMA 0.9999 already gives most of
  what the decay gives for samples: pre-decay EMA ~ mid-decay raw (2.75 vs 2.75 on set A).
- Final EMA vs raw: equal fid_patch, EMA better whole-image fid (1.98-2.08 vs 2.09-2.19) and mse_lr: use EMA.
- Weight norm peaked at 3255 (36 h) and fell to 3074 during the decay.

### Model internals (dev/model_diagnostics.py, final EMA, 8 benchmark images, sigma 0.1 / 0.5 / 0.9)

- All 16 blocks are active (relative residual update 0.11-0.29 per block, last block 0.65-0.71); no dead blocks.
  Block 0 rewrites the small stem output (relative update ~8-9).
- adaLN feed forward gates grow with depth: |gate| 0.7-1.7 (blocks 0-1) -> 13-15 (blocks 12-13), then 3.9 (block 14)
  and 0.02 (block 15: its feed forward is practically off; the last block only attends). Research models had adaLN
  scales <= 3 (28 only in a diverging run); here stable thanks to QK-norm, but the residual stream scale is large
  (same picture as the weight norm growth).
- The hff condition has a very small effect on the velocity: hff vs null 1.1-1.9%, hff 0.9 vs 0.1 1.7-3.6% of
  ||v||. CFG on hff therefore needs large guidance scales to have visible effect.
- EMA differs from raw by 13-18% (relative norm) in the blocks, 24% in the output modulation.

### Conclusions for the next runs

1. Under this recipe, 222M saturates on image patch FID after ~1 day (~3.5M units); more steps / decay gave loss
   and mse_lr but not fid_patch or visual change. Next gains need recipe changes, not training time: data with
   small details (open problem 4), larger capacity only if it is not data / metric bound.
2. Fix the residual tables per pair type (tiny-VAE spatial and combined video), consider weight decay 0.05-0.1.
3. Evaluate on one fixed 1000-image set only (wan_eval_1000).
