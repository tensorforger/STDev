# Small single-step upscaler distilled from the large model: results and comparison

## Summary

A 48.6M parameter upscaler that makes the high resolution latents in **one forward pass** was
trained on the rectified flow distillation pairs of the 450.9M SFT model (teacher noise and low
resolution latents in, teacher output out). It has no noise level, text or high frequency
conditioning.

- Native 256 px patch FID, the primary metric: **1.3514** vs 0.8241 for the 20-step teacher and
  0.8413 for the large 1-step student. It recovers **51%** of the patch FID improvement that SFT
  gave over the stage-2 pretrained model (1.9031 -> 0.8241).
- It is **9.3x smaller** and **98x faster** than the 20-step teacher at 1024 px (8.5 ms vs 838 ms
  for sampling the UNet), and 5.0x faster than the large 1-step student. Peak memory at 1024 px is
  452 MB vs 1,407 MB.
- Compared with the similar sized multi-step models of the autoresearch (2 h, one GPU each, trained
  from scratch on 20k FLUX generations), its patch FID is better than every model up to 60M
  (1.44 for 60M, 1.53 for 45M, 1.78 for 18M) and level with the 88M to 304M models at the
  benchmark's 14 sampling steps (1.38). Those models need 10 to 30 network evaluations per image,
  this one needs one.
- Whole-image FID does not show the same advantage: 1.987, worse than the 45M to 304M autoresearch
  models (1.69 to 1.88) and 2.1x the teacher's (0.9488).
- Training was still improving when the learning rate decay was started (about -0.02 patch FID per
  epoch); the decay epoch gave a further -0.031.

These are in-domain results: the benchmark references are FLUX generations, and the distillation
pairs come from a teacher that was fine-tuned on FLUX samples. The comparison with autoresearch
models is not like for like (see Scope and interpretation).

## Model and training

| Setting | Value |
|---|---|
| Model | `src/stdev/upscalers/distill_small.py`, 48.55M parameters |
| Layout | UNet, channels (512, 256, 256), encoder blocks (1, 1, 1), decoder blocks (3, 2, 1), 1 middle block |
| Blocks | Residual block + 8x8 window attention + pointwise MLP (SDXL style) after every block on all levels |
| Parameter allocation | About 33M in the full resolution encoder and decoder blocks, 25M of them in the 3 decoder blocks (autoresearch: level 0 width >= 512, decoder heavy, lower levels small and shallow) |
| Input and output | Noise and bilinearly upsampled low resolution latents; velocity at t = 1, sample = noise - velocity |
| Data | 13,200 teacher pairs of `train_datasets/flux_sft_distill_pairs` (8 resolution buckets), the same pairs as the large student |
| Recipe | Batch 16, AdamW (0.9, 0.95), LR 4e-4 constant after 100 warmup steps, gradient clip 1, EMA 0.999, bf16 autocast |
| Schedule | 17 epochs at constant LR, then 1 epoch of linear decay to zero (epoch 18, 825 steps per epoch) |
| Budget | 14,850 steps, 237,600 samples seen, 5,710 s (1.59 h) of training on one RTX 5090, plus the benchmark after each epoch |
| Benchmark | Unmodified `scripts/run_benchmark.py` on `benchmark_datasets/flux_1024`, all 1,000 samples, EMA weights, seed 0 |

## Comparison with the teacher

All metrics are lower-is-better. Large model rows are existing results (`reports/sft_vs_stage2_pretrain.md`,
`reports/rectified_flow_distillation_vs_sft.md`).

| Model | Params | Steps | Patch FID | FID | HR image MSE | LR image MSE | HR latent MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT EMA (teacher) | 450.9M | 20 | **0.8241** | **0.9488** | **0.00184074** | **0.00061787** | 0.30440 |
| Large 1-step student | 450.9M | 1 | 0.8413 | 1.0402 | 0.00186533 | 0.00064028 | **0.30388** |
| **Small student (epoch 18)** | **48.6M** | **1** | 1.3514 | 1.9871 | 0.00221600 | 0.00085300 | 0.31999 |
| Stage-2 pretrained EMA | 450.9M | 20 | 1.9031 | 1.4535 | 0.00216347 | 0.00069991 | 0.29354 |

Change of the small student relative to the references:

| Reference | Patch FID | FID | HR image MSE | LR image MSE | HR latent MSE |
|---|---:|---:|---:|---:|---:|
| SFT teacher | +0.527 (+64%) | +1.038 (+109%) | +0.000375 (+20%) | +0.000235 (+38%) | +0.0156 (+5.1%) |
| Large 1-step student | +0.510 (+61%) | +0.947 (+91%) | +0.000351 (+19%) | +0.000213 (+33%) | +0.0161 (+5.3%) |
| Stage-2 pretrained | -0.552 (-29%) | +0.534 (+37%) | +0.000053 (+2.4%) | +0.000153 (+22%) | +0.0265 (+9.0%) |

- The small student is better than the stage-2 pretrained model on patch FID, which measures
  detail statistics, but worse on whole-image FID and on all MSE metrics. It does not preserve the
  low-frequency content of its input as well as the large models (LR image MSE 0.00085 vs 0.00062
  for the teacher; pixel bicubic upscaling has 0.00064).
- The gaps are larger on whole-image FID than on patch FID (+109% vs +64% against the teacher),
  the same ordering as for the large 1-step student (+10% vs +2%), only with larger gaps.

## Efficiency

Latencies are the benchmark's compiled single-forward measurement (ms, average); the sampling
latency of the 20-step teacher is the derived 20 x forward time (UNet only, without VAE or text
encoding).

| Model | Forward 512 | Forward 1024 | Forward 2048 | Sampling 1024 | Peak memory 512 / 1024 / 2048 (MB) |
|---|---:|---:|---:|---:|---|
| SFT EMA (teacher), 20 steps | 11.808 | 41.901 | 216.137 | 838.0 | 997 / 1,408 / 3,049 |
| Large 1-step student | 12.242 | 42.648 | 219.444 | 42.6 | 997 / 1,407 / 3,045 |
| **Small student** | **2.905** | **8.504** | **34.383** | **8.5** | **183 / 452 / 1,528** |

Speedup of the small student: 81x / 98x / 126x over the 20-step teacher and 4.2x / 5.0x / 6.4x over
the large 1-step student at 512 / 1024 / 2048 px.

## Learning curve

Benchmark of the EMA weights after every epoch (epochs 1 to 17 at constant LR 4e-4, epoch 18 is
the linear decay). Training loss is the mean velocity MSE over the epoch.

| Epoch | Step | Train loss | Patch FID | FID | HR image MSE | LR image MSE | HR latent MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 825 | 0.3219 | 10.161 | 13.636 | 0.005117 | 0.002799 | 0.4021 |
| 2 | 1,650 | 0.1576 | 4.438 | 6.054 | 0.003671 | 0.001753 | 0.3679 |
| 3 | 2,475 | 0.1266 | 3.098 | 4.289 | 0.003090 | 0.001375 | 0.3519 |
| 4 | 3,300 | 0.1066 | 2.426 | 3.507 | 0.002731 | 0.001148 | 0.3410 |
| 5 | 4,125 | 0.0949 | 2.095 | 2.907 | 0.002566 | 0.001052 | 0.3355 |
| 6 | 4,950 | 0.0873 | 1.919 | 2.672 | 0.002479 | 0.001003 | 0.3324 |
| 7 | 5,775 | 0.0817 | 1.800 | 2.537 | 0.002428 | 0.000974 | 0.3301 |
| 8 | 6,600 | 0.0775 | 1.705 | 2.430 | 0.002385 | 0.000950 | 0.3282 |
| 9 | 7,425 | 0.0739 | 1.637 | 2.361 | 0.002356 | 0.000932 | 0.3267 |
| 10 | 8,250 | 0.0710 | 1.583 | 2.290 | 0.002326 | 0.000917 | 0.3255 |
| 11 | 9,075 | 0.0686 | 1.539 | 2.240 | 0.002308 | 0.000907 | 0.3246 |
| 12 | 9,900 | 0.0663 | 1.506 | 2.199 | 0.002292 | 0.000897 | 0.3237 |
| 13 | 10,725 | 0.0645 | 1.475 | 2.167 | 0.002276 | 0.000888 | 0.3228 |
| 14 | 11,550 | 0.0628 | 1.446 | 2.132 | 0.002257 | 0.000878 | 0.3221 |
| 15 | 12,375 | 0.0613 | 1.418 | 2.088 | 0.002247 | 0.000871 | 0.3215 |
| 16 | 13,200 | 0.0600 | 1.402 | 2.067 | 0.002238 | 0.000866 | 0.3208 |
| 17 | 14,025 | 0.0586 | 1.383 | 2.039 | 0.002227 | 0.000860 | 0.3203 |
| 18 (decay) | 14,850 | 0.0535 | **1.351** | **1.987** | **0.002216** | **0.000853** | **0.3200** |

- Every metric improved at every epoch; there is no sign of overfitting on the benchmark after 18
  passes over 13,200 pairs.
- The gain per epoch shrinks (patch FID: -0.33 from epoch 4 to 5, -0.054 from 9 to 10, -0.029 from
  13 to 14, -0.020 from 16 to 17) but was still about -0.02 per epoch when the decay was started.
  The run was not saturated; more constant LR epochs would likely have given a lower final value.
- The decay epoch improved patch FID by 0.031 (2.3%) and whole-image FID by 0.052. This is far less
  than the 0.3 to 0.8 patch FID that the autoresearch found for annealing raw weights, because the
  benchmarked weights are an EMA (decay 0.999, about 1,000 steps), which already averages out most
  of the high-LR noise.

## Comparison with similar sized autoresearch models

The autoresearch models are multi-step flow models (text conditioned, noise level conditioned,
trained from scratch on `flux_1024_20k` real FLUX generations) trained for 2 hours on one RTX 5090
(`autoresearch/experiment_logs.md`, round 10; 30 minute results where noted). They are evaluated on
the same `flux_1024` benchmark. Patch FID is at the benchmark setting (14 sampling steps; 10 for the
18M model) and at the best post-hoc setting of the sweep (20 steps). Total latency is forward time
at 1024 px times the number of steps.

| Model | Params | Training | Steps | Patch FID | Patch FID, 20 steps | FID | Forward 1024 (ms) | Sampling 1024 (ms) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **Small student (this report)** | **48.6M** | **1.6 h, distilled** | **1** | **1.351** | - | 1.987 | 8.5 | **8.5** |
| 0072 small, 2 levels (256, 256) | 18M | 2 h | 10 | 1.78 | 1.72 | 2.08 | 2.4 | 24 |
| 0073 (256, 256, 512) dec (5, 1, 1) | 45M | 2 h | 14 | 1.53 | 1.52 | 1.88 | 3.5 | 49 |
| 0074 (512, 256, 512) dec (3, 1, 1) | 60M | 2 h | 14 | 1.44 | 1.40 | 1.78 | 5.7 | 79 |
| 0064 (256, 256, 512) dec (5, 1, 1) | 45M | 30 min | 10 | 1.94 | - | 2.30 | 3.5 | 35 |
| 0066 (512, 256, 512) dec (3, 1, 1) | 60M | 30 min | 14 | 1.87 | - | 2.27 | 5.7 | 79 |
| 0077 deep full resolution decoder | 88M | 2 h | 14 | 1.38 | 1.32 | 1.71 | 10.8 | 151 |
| 0078 SDXL style, all levels | 304M | 2 h | 14 | 1.38 | 1.32 | 1.69 | 14.5 | 203 |

For orientation, the log also has pixel bicubic upscaling at 8.45 patch FID, the baseline at 4.54
(106M, 30 min) and the real image oracle at 0.25.

- **Quality at equal size:** the small student (1.351) is better than the 45M and 60M models
  after 2 h (1.53 and 1.44; 1.52 and 1.40 at their best step count) and much better than the 30
  minute models of the same size (1.94 and 1.87). The 0.05 to 0.09 gap to the 60M model is one to
  two times the measured seed noise of patch FID (about 0.03 for the training seed and 0.05 for the
  inference seed); the gaps to the 45M and 18M models are clearly larger. The difference to the 88M and
  304M models at 14 steps (1.38) is within the noise; their best settings (1.30 to 1.32) are still
  slightly ahead.
- **Cost:** per image the small student needs one evaluation of the UNet. The 45M and 60M
  autoresearch models need 14 to 20 (49 to 113 ms at 1024 px), the 88M model 14 to 30 (151 ms and
  more), so the single-step model is 6x to 13x cheaper than the 45M and 60M models at lower patch
  FID. A single forward is 1.5x to 2.4x more expensive than that of the 60M and 45M models
  (8.5 ms vs 5.7 ms and 3.5 ms); the small student also has window attention and an MLP after
  every block at full resolution, which the 45M and 60M models do not.
- **Whole-image FID is the exception:** 1.987 is worse than the 45M to 304M autoresearch models
  (1.69 to 1.88) and level with the 18M model (2.08), single seed. The log noted that whole-image
  FID varies by up to 0.7 between seeds, so this is a weak signal, but together with the LR image
  MSE (0.00085, vs about 0.0012 to 0.0013 for the 30 minute autoresearch models in rounds 3 and 4;
  2 h MSE values are not in the log) it indicates that global colour and low-frequency fidelity is
  the weaker side of this model, while its fine-detail statistics are good.

## Scope and interpretation

- The two sets of models were trained on different targets. The autoresearch models learned from
  20k FLUX generations from scratch. The small student learned from outputs of a teacher that was
  pretrained on much more data (stage 1 and 2) and SFT-trained on FLUX samples, with a
  deterministic noise to output pairing (reflow). The comparison therefore shows what a 50M model
  reaches when distilled from the large model, not an architecture comparison. The autoresearch
  found that coupled noise from the original 4-step FLUX teacher did not work (0002: patch FID
  7.13 at 30 min); here the pairs come from a straightened teacher flow, so this is not
  contradictory, but the effect of the pair source was not isolated.
- The small student was trained for 237,600 samples; 0073 and 0074 saw about 730k and 430k samples
  (45.6k and 27k steps at batch 16).
- Latency is the benchmark's forward measurement of the compiled model; multi-step sampling
  latencies of the autoresearch models are derived by multiplying with the step count, and the
  autoresearch latencies were measured with the benchmarks of eight runs executing in parallel
  on separate GPUs.
- Single run, single seed, one evaluation per epoch with fixed inference noise (seed 0). The
  EMA weights were benchmarked, raw weights were not. Only the in-domain `flux_1024` benchmark was
  used; `axis_1024` or another real image benchmark is needed for out-of-domain behaviour.
- No visual comparison of the output images is part of this report.

## Artifacts and reproduction

- Model: `src/stdev/upscalers/distill_small.py`; training: `scripts/run_small_distill.py`.
- Run: `distill_runs/small_v1/` (EMA and raw weights in `export/`, tensorboard in `tensorboard/`);
  per-epoch benchmark results and saved samples: `raw_results/small_v1_epoch_NNN/`
  (final result `raw_results/small_v1_epoch_018/`).
- Per-epoch table: `python dev/summarize_small_run.py --run-name small_v1`.

```bash
python scripts/run_small_distill.py --run-name small_v1
touch distill_runs/small_v1/control/decay   # starts the one epoch decay
```
