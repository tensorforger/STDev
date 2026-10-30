# Stage 1 pretraining report: 256 -> 512 latent upscaler (run `stage1_v1`)

Run: `stage1_v1`, 2026-10-30, 8x RTX 5090, code at commit `43ae0ee` ("stage 1").
Final weights: `pretrain_runs/stage1_v1/checkpoints/checkpoint_000060892.pt` (also uploaded to S3).

## 1. Summary

- Trained the 451 M parameter `pretrain_large` latent upscaler (FLUX.2 VAE latents, 2x) on
  256 -> 512 crops of ReLAION-high-resolution, fully online: images are downloaded, cropped,
  encoded and cached while training runs.
- 60,892 steps at global batch 112 = 6.82 M samples seen, 9.7 h of training (plus 30 min
  warmup). 2.92 M unique samples produced, so each sample was seen ~2.3 times on average.
- Constant LR 4e-4 for 52,950 steps, then a manual linear decay to zero over 7,942 steps
  (15% of the steps done), triggered when the benchmark curve flattened.
- Final fid_patch on 2,000 held-out images: **2.409 (raw weights) / 2.424 (EMA)**, down from
  3.04 (EMA) after 30 min. Before the decay the EMA gained only ~0.01 per 2 h, so the stage
  is saturated and stage 2 (512 -> 1024) starts from the final checkpoint.

## 2. Dataset

**Source:** `laion/relaion-high-resolution` (Hugging Face, 128 metadata parquet shards of
url + caption + scores). Images are downloaded from the original urls during training; nothing
is stored except the latent cache.

| | Value |
|---|---|
| Training shards | 0-126 in order; rows of each shard shuffled with seed `METADATA_SHUFFLE_SEED = 0` |
| Held-out benchmark shard | 127 |
| Filters | min image side >= 1024 (from metadata), pwatermark < 0.5, punsafe < 0.5 |
| Download attempts / successes (training session) | 3.55 M / 2.19 M (62%) |
| Unique samples produced | 2.92 M (0.78 M during warmup, 2.14 M during training) |
| Position reached | shard 4 (the 5th), row 712,762 of its filtered rows |

**Per sample processing** (`src/stdev/pretrain/producer.py`):

1. Decode the image and take a random square crop with side uniform in [512, short side].
2. Resize the crop to 512 px (HR) with `cv2.INTER_AREA`, then the HR to 256 px (LR), also
   `INTER_AREA`.
3. Compute 4 high frequency (HF) metrics on the HR image: `noise_level`, `edge_sharpness`,
   `texture_detail`, `detail_scale` (`src/stdev/pretrain/high_frequency.py`). NaN when undefined.
4. Encode LR and HR with the FLUX.2 VAE (32 channels, 8x downsampling, bf16). Stored as 8-bit
   `lbrq` latents through vaezip, so the latents are 32x32 (LR) and 64x64 (HR).
5. Encode the caption with the SDXL CLIP-L text encoder, mean pooled over 77 tokens (768-d, fp16).
6. Write shards of 512 samples to the rolling cache. When it exceeds 300 GB, the oldest shards
   are evicted (the cache stayed at 301 GB, ~3,700 shards, for most of the run).

**HF feature normalization:** quantile (empirical CDF) normalization, fitted on the first
10,000 cached samples after warmup (1,001 quantile levels). Values map to [0, 1], where 0.5 is
the median image. Saved in `statistics/high_frequency_quantiles.json`.

**Benchmark set:** the first 2,000 successfully downloaded images of the filtered, shuffled shard 127, with the crop
seeded by the url so it is deterministic. The HR / LR PNGs, latents, text embeddings and HF
features are stored in `benchmark_set/`.

## 3. Model

`src/stdev/upscalers/pretrain_large.py`, `UpscalerUNet`, 450.9 M parameters. It combines the
research runs 0077 (deep full resolution decoder) and 0078 (SDXL-style window transformer
block after every residual block, on all levels).

| | Value |
|---|---|
| Level channels | 768 (full res), 512, 1024 |
| Encoder blocks per level | 1, 2, 2 |
| Decoder blocks per level | 8, 2, 2 |
| Middle blocks | 2 |
| Attention | window attention, window 8, head dim 64, MLP expansion 4, after every residual block |
| Input | noisy HR latents concatenated with the upsampled LR latents |
| Condition vector (dim 1024) | sum of: sigma embedding, text MLP, 4 HF MLPs (sinusoidal embedding, like the timestep) |
| Missing condition | learned null embedding for text and for each HF feature (also used for NaN features) |

## 4. Training

| | Value |
|---|---|
| Objective | flow matching, velocity target `noise - latents_high`, MSE |
| Training sigmas | logit-normal, shift 1.0 |
| Sampling (benchmark) | Euler, 20 steps, sigma shift 2.0 |
| Condition dropout | text 20%, each HF feature independently 20% |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0, grad clip 1.0 |
| LR | 500 step linear warmup, constant 4e-4, manual linear decay to 0 over the last 7,942 steps |
| Batch | 16 per GPU (micro-batch 8 x 2 accumulation), 7 GPUs, global 112 |
| Precision | bf16 autocast, DDP with bf16 gradient compression, `torch.compile` |
| EMA | 0.9999 (warm-started as min(0.9999, (1 + n) / (10 + n))), on rank 0 |
| Seed | 0 (data loader seed `SEED * 1000 + rank`) |

**Process layout** (`scripts/run_pretrain.py`):

1. Build the benchmark set on GPU 7.
2. Warmup for 30 min with all 8 GPUs encoding (0.78 M samples, 1,533 shards). Fit the HF
   quantiles.
3. Train with DDP on GPUs 0-6. GPU 7 runs the producer (encoding) and the benchmark watcher.
   The encoder pauses while a benchmark runs.

**Throughput:**
- **Trainer:** 196-197 samples/s (1.75 steps/s), 22.5 GB per GPU.
- **Encoder:** ~73 samples/s while running, 61 samples/s on average including benchmark pauses.

Training reads the cache faster than it is filled, which is why each sample is reused ~2.3x.

**Benchmark:** at the first scheduled point every 30 min of training for the first 4 h, then
every 2 h, plus the final weights. Both EMA and raw weights are evaluated on the 2,000 held-out
images with text and HF features given. Each benchmark takes ~810 s.
- **Metrics:** `fid_patch` (FID on native 256 px tiles of the 512 px outputs), `fid`
  (whole images), and `mse_lr_image` (MSE between the 2x downscaled output and the LR input).
- **Saved pairs:** 100 LR / generated pairs per benchmark.

## 5. Metrics

### 5.1 Benchmark (2,000 held-out images, lower is better)

| Step | Train time | Samples seen | EMA fid_patch | Raw fid_patch | EMA fid | Raw fid | EMA mse_lr_image |
|---|---|---|---|---|---|---|---|
| 3,156 | 0.5 h | 0.35 M | 3.041 | 3.497 | 3.746 | 4.128 | 0.00080 |
| 6,306 | 1.0 h | 0.71 M | 2.808 | 2.925 | 3.360 | 3.463 | 0.00070 |
| 9,451 | 1.5 h | 1.06 M | 2.700 | 2.691 | 3.203 | 3.336 | 0.00060 |
| 12,599 | 2.0 h | 1.41 M | 2.646 | 3.056 | 3.121 | 3.266 | 0.00060 |
| 15,740 | 2.5 h | 1.76 M | 2.593 | 2.738 | 3.056 | 3.112 | 0.00060 |
| 18,891 | 3.0 h | 2.12 M | 2.552 | 2.731 | 3.012 | 3.053 | 0.00060 |
| 22,032 | 3.5 h | 2.47 M | 2.526 | 2.662 | 2.982 | 3.038 | 0.00060 |
| 25,177 | 4.0 h | 2.82 M | 2.509 | 2.854 | 2.953 | 3.053 | 0.00050 |
| 37,742 | 6.0 h | 4.23 M | 2.447 | 2.690 | 2.869 | 3.041 | 0.00050 |
| 50,309 | 8.0 h | 5.63 M | 2.434 | 2.682 | 2.838 | 2.907 | 0.00050 |
| **60,892 (final)** | **9.7 h** | **6.82 M** | **2.424** | **2.409** | **2.817** | **2.779** | **0.00050** |

The decay started at step 52,950 (8.43 h). `mse_lr_image` is rounded to 4 decimals in
`results.jsonl`, so its changes after 2 h are below resolution.

**Observations:**
- **Most of the gain comes early.** Of the total EMA fid_patch gain (3.04 -> 2.42), 0.53 comes
  in the first 4 h. Hours 4-8 give only 0.075, and the last pre-decay interval (6 -> 8 h)
  gives 0.013.
- **Raw weights are noisy at constant LR.** Their fid_patch ranges from 2.66 to 3.06 at
  constant LR, and the EMA is consistently better.
- **The decay mainly improves the raw weights.** Raw fid_patch drops 2.68 -> 2.41 (fid 2.91 ->
  2.78), ending slightly better than the EMA. The EMA barely changes (2.434 -> 2.424) because,
  with decay 0.9999, it still averages mostly weights from before the decay. For stage 2
  initialization the raw weights are used (`--init-checkpoint` loads `model`).

### 5.2 Training loss (mean over 2,000-step windows, from `logs/trainer.log`)

| Steps | 0-2k | 5-7k | 10-12k | 20-22k | 30-32k | 40-42k | 50-52k | 57-59k (decay) | 60-60.9k (decay) |
|---|---|---|---|---|---|---|---|---|---|
| Loss | 0.656 | 0.449 | 0.432 | 0.418 | 0.412 | 0.408 | 0.405 | 0.400 | 0.396 |

The flow matching loss is dominated by the random sigma per sample, so only the trend is
meaningful.

## 6. Outputs

All paths are under `pretrain_runs/stage1_v1/` (git-ignored):

| Path | Contents |
|---|---|
| `checkpoints/checkpoint_000060892.pt` | final: model (raw), EMA, optimizer, training state (7.2 GB) |
| `checkpoints/checkpoint_000056599.pt` | previous hourly checkpoint (mid-decay) |
| `snapshots/final_000060892.done` | safetensors, bf16 `ema.*` and `raw.*` weights (1.8 GB) |
| `benchmarks/results.jsonl` | all benchmark rows (table 5.1) |
| `benchmarks/<snapshot>/` | 100 pairs per benchmark: `NNNNN_low.png`, `NNNNN_ema.png`, `NNNNN_raw.png` |
| `statistics/` | HF quantiles, FID reference statistics of the benchmark set |
| `tensorboard/` | `train` (loss, LR, throughput), `benchmark` |
| `logs/` | launcher, producer, benchmark, trainer logs |

## 7. Reproduce

The commands below assume commit `43ae0ee` and one 8-GPU machine. They need the Hugging Face
login (to read the ReLAION metadata) and about 330 GB of free disk for the cache and
checkpoints.

```bash
git checkout 43ae0ee
uv venv --python=3.12 && source .venv/bin/activate
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu134
uv pip install -e .
hf auth login

cd models
hf download black-forest-labs/FLUX.2-klein-4B --local-dir FLUX.2-klein-4B
hf download stabilityai/stable-diffusion-xl-base-1.0 --include "text_encoder/*" --include "tokenizer/*" --local-dir stable-diffusion-xl-base-1.0
cd ..

# benchmark set -> 30 min warmup -> training (resumable: rerun the same command after a crash)
setsid nohup python scripts/run_pretrain.py --run-name stage1_v1 --stage stage_1 > /dev/null 2>&1 &
tail -f pretrain_runs/stage1_v1/logs/trainer.log pretrain_runs/stage1_v1/benchmarks/results.jsonl

# when the EMA fid_patch flattens (here: at step ~52,950, 8.4 h): linear decay over 15% of the steps done
touch pretrain_runs/stage1_v1/control/decay
# the run stops by itself when the decay is done (training_done is written)
```

To match this run exactly, trigger the decay with an explicit length after step 52,950:
`echo 7942 > pretrain_runs/stage1_v1/control/decay`.

The result is not bit-exact. Download successes depend on which urls are still alive, and the
order samples arrive in depends on network timing. Both change which images enter the cache
and when.
