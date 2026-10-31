# Stage 2 pretraining report: 512 -> 1024 latent upscaler (run `stage2_v1`)

Run: `stage2_v1`, 2026-10-30 / 31, 8x RTX 5090. Initialized from the stage 1 final checkpoint
(see `reports/stage1_pretrain_report.md`).
Code: commit `43ae0ee` plus the encode / decode chunking fix described in section 4. The fix is
not committed yet.
Final weights: `pretrain_runs/stage2_v1/checkpoints/checkpoint_000054108.pt` and
`pretrain_runs/stage2_v1/export/stage2_final_000054108_{raw,ema}.safetensors`. Both are also
uploaded to R2 under `checkpoints/stage2_v1/`.

## 1. Summary

- Continued training the 451 M parameter `pretrain_large` upscaler at 512 -> 1024 on
  ReLAION-high-resolution. The pipeline is the same online one as stage 1, with a larger minimum
  source side.
- Training: 54,108 steps at global batch 28 = 1.52 M samples seen, in 8.7 h of training.
  - LR: constant 4e-4, then a linear decay to zero from step 47,050 over 7,058 steps (15% of
    the steps done). The decay was triggered manually.
- Final fid_patch on 1,000 held-out 1024 px images: **1.721 (EMA) / 1.736 (raw)**. Full-image
  fid: 1.550 / 1.547.
- **The stage saturated almost immediately.** EMA fid_patch was already 1.75 after 30 min and
  stayed in a 1.74-1.80 band up to 8 h. Only the decay moved it, by -0.05.
- **The run was data starved.** The encoder produced 7.6 samples/s, while training read
  49 samples/s. Each sample was seen ~6.3 times; in stage 1 it was 2.3.

## 2. Dataset

Source and filters are the same as stage 1: `laion/relaion-high-resolution`, training shards
0-126, benchmark shard 127, pwatermark < 0.5, punsafe < 0.5. The difference is the minimum
source side, **>= 1536** (28.8% of rows, measured on shard 0).

| | Value |
|---|---|
| Crop | random square, side uniform in [1024, short side], `INTER_AREA` to 1024 (HR), then to 512 (LR) |
| Latents | FLUX.2 VAE, 64x64 (LR) and 128x128 (HR), 8-bit lbrq |
| Text / HF features | pooled SDXL CLIP-L embedding; 4 HF metrics on the 1024 px HR |
| HF quantiles | refitted on the first 10,000 stage 2 samples (the 1024 px statistics differ from 512 px) |
| Warmup (8 GPUs, 30 min) | 235,072 samples, 461 shards, 129 samples/s |
| Training-time production (1 GPU) | 239,616 samples in 8.7 h, 7.6 samples/s; downloads 249,908 / 402,401 (62%) |
| Unique samples total | 474,688 |
| Position reached | shard 2, row 219,792 of its filtered rows |
| Cache | 296 GB at the end (929 shards after eviction) |

**Benchmark set:** the first 1,000 successfully downloaded images of the filtered, shuffled
shard 127, with a deterministic crop. These are different images and a different resolution
from the stage 1 benchmark, so the fid numbers of the two stages are not comparable.

## 3. Model and training

The model is the same as stage 1 (`src/stdev/upscalers/pretrain_large.py`, 450.9 M
parameters). It was initialized from `stage1_v1/checkpoints/checkpoint_000060892.pt` (raw
weights). The optimizer state was fresh, and the EMA started from the initial weights.

| | Stage 2 | Stage 1 (for reference) |
|---|---|---|
| Resolution | 512 -> 1024 | 256 -> 512 |
| Batch | 4 per GPU (micro-batch 2 x 2), global 28 | 16 per GPU, global 112 |
| LR | 500 step warmup, 4e-4 constant, manual linear decay | same |
| Objective, sigmas, sampler, dropout, optimizer, EMA | same as stage 1 | |
| Training throughput | 49 samples/s (1.75 steps/s) | 197 samples/s |
| Benchmark | 1,000 images, ~1,905 s per run (EMA + raw) | 2,000 images, ~807 s |

The pixels per step are the same as stage 1, so the trainer speed in steps/s is unchanged.

## 4. Pipeline changes for 1024 px

The first launch and the smoke test both hit CUDA out-of-memory errors. The VAE ran full batches
at 4x the pixels of stage 1. Two fixes:

- **Producer:** `encode_in_chunks` encodes the 32-sample batch in chunks of at most
  `ENCODE_CHUNK_PIXELS = 32 * 512 * 512` pixels. At 1024 px that is 8 images per chunk.
- **Benchmark:** `decode_in_chunks` decodes in chunks of at most `DECODE_CHUNK_PIXELS = 16 * 512 * 512`
  pixels, which is 4 images at 1024 px. GPU 7 peaked at 16.6 GB in the smoke test.

The benchmark set was reduced from 2,000 to 1,000 images (`--benchmark-samples 1000`). At
1024 px, 2,000 images would have taken ~55 min, much longer than the 30 min snapshot interval.

## 5. Metrics

### 5.1 Benchmark (1,000 held-out 1024 px images, lower is better)

| Step | Train time | Samples seen | EMA fid_patch | Raw fid_patch | EMA fid | Raw fid | EMA mse_lr_image |
|---|---|---|---|---|---|---|---|
| 3,108 | 0.5 h | 87 k | 1.746 | 2.102 | 1.626 | 1.874 | 0.000387 |
| 6,222 | 1.0 h | 174 k | 1.791 | 1.819 | 1.621 | 1.766 | 0.000385 |
| 9,335 | 1.5 h | 261 k | 1.803 | 1.901 | 1.619 | 1.799 | 0.000384 |
| 12,451 | 2.0 h | 349 k | 1.777 | 2.094 | 1.605 | 1.909 | 0.000384 |
| 15,565 | 2.5 h | 436 k | 1.757 | 2.119 | 1.609 | 1.849 | 0.000383 |
| 18,686 | 3.0 h | 523 k | 1.755 | 1.834 | 1.597 | 1.712 | 0.000383 |
| 21,795 | 3.5 h | 610 k | 1.739 | 2.172 | 1.598 | 1.930 | 0.000382 |
| 24,914 | 4.0 h | 698 k | 1.764 | 2.079 | 1.602 | 1.761 | 0.000381 |
| 37,371 | 6.0 h | 1.05 M | 1.782 | 1.965 | 1.599 | 1.760 | 0.000378 |
| 49,821 | 8.0 h | 1.39 M | 1.768 | 1.739 | 1.584 | 1.568 | 0.000377 |
| **54,108 (final)** | **8.7 h** | **1.52 M** | **1.721** | **1.736** | **1.550** | **1.547** | **0.000375** |

The decay started at step 47,050 (7.6 h). The 8.0 h row is already inside the decay.

**Observations:**
- **fid_patch is flat from the first benchmark.** EMA fid_patch moves within 1.74-1.80 (noise
  level) for 8 h at constant LR. Whole-image fid and mse_lr_image improve slowly but steadily
  (1.626 -> 1.584 and 0.000387 -> 0.000377 before the end). So the model keeps getting better at
  the global structure, but not at the 256 px tile statistics.
- **The decay gives the only clear fid_patch gain.** EMA goes from 1.77 to 1.72. Raw goes from
  1.97-2.17 at constant LR to 1.74, the same as the EMA. The raw weights are noisier at
  constant LR than in stage 1 (batch 28 vs 112).
- **Data starvation may have contributed to the plateau.** The encoder was paused while
  benchmarks ran. Each benchmark took ~32 min, longer than the 30 min early interval, so for
  the first 4 h the encoder was paused more than half the time. Each sample was reused ~6.3x
  overall. A plateau from 0.5 h on, at ~150 k unique samples, is consistent both with a capacity
  or information limit and with too little fresh data. This run cannot tell them apart.

### 5.2 Training loss (mean over 2,000-step windows)

| Steps | 0-2k | 5-7k | 10-12k | 20-22k | 30-32k | 40-42k | 45-47k | 47-49k (decay) | 50-52k (decay) | 52-54k (decay) |
|---|---|---|---|---|---|---|---|---|---|---|
| Loss | 0.407 | 0.409 | 0.405 | 0.403 | 0.405 | 0.405 | 0.398 | 0.396 | 0.391 | 0.392 |

The loss starts at the stage 1 level (0.41 vs 0.40 at the end of stage 1). It is flat at
constant LR and drops ~0.013 during the decay, matching the benchmark.

## 6. Outputs

All paths are under `pretrain_runs/stage2_v1/`:

| Path | Contents |
|---|---|
| `checkpoints/checkpoint_000054108.pt` | final: model (raw), EMA, optimizer, state (7.2 GB) |
| `checkpoints/checkpoint_000049824.pt` | previous hourly checkpoint (mid-decay) |
| `export/stage2_final_000054108_{raw,ema}.safetensors` | final weights, `PretrainLargeUpscaler` format (1.8 GB each) |
| `benchmarks/results.jsonl`, `benchmarks/<snapshot>/` | table 5.1 rows; 100 low / EMA / raw image triples per benchmark |
| `statistics/`, `tensorboard/`, `logs/` | HF quantiles and FID references, tensorboard, logs |
| `cache/` | 296 GB rolling latent cache (disposable) |

## 7. Reproduce

Start from the stage 1 reproduction (`reports/stage1_pretrain_report.md`, section 7) and the
stage 1 final checkpoint. The chunking fix from section 4 is required, so commit it first.

```bash
# requires stage 1: pretrain_runs/stage1_v1/checkpoints/checkpoint_000060892.pt (or from S3)
setsid nohup python scripts/run_pretrain.py --run-name stage2_v1 --stage stage_2 \
  --init-checkpoint pretrain_runs/stage1_v1/checkpoints/checkpoint_000060892.pt \
  --benchmark-samples 1000 > pretrain_runs/stage2_v1.launcher.out 2>&1 &

# LR decay, as in this run: after step 47,050 (~7.6 h of training)
echo 7058 > pretrain_runs/stage2_v1/control/decay
# the run stops by itself after the decay (training_done)
```

**Disk:** the stage 2 cache takes 300 GB. The stage 1 cache was deleted before this run.
