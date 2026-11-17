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
