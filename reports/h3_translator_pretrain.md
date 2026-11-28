# H3-Image -> FLUX.2 latent translator pretraining (8x RTX 5090, 2026-11-28)

Continuation of autoresearch_6 (`autoresearch_6/experiment_logs.md`, run_0004 / run_0006) with the FLUX.2 pretrain
recipe (`reports/flux_pretrain.md`). Code: `src/stdev/h3_pretrain/` (config, model, corruption, h3_encoder, producer,
trainer, benchmark), launcher `scripts/run_h3_pretrain.py`; dev tools `dev/h3_pretrain/` (encoder benchmark, residual
statistics).

Run: `python scripts/run_h3_pretrain.py --run-name h3_translator_v1 --data-run-name h3_data`.
Decay: `echo <hours> > pretrain_runs/h3_translator_v1/control/decay` (linear LR decay to 0 over that many training
hours, then the run ends; an empty file = 40% of the training time so far). Stop / resume: `touch .../control/stop`,
relaunch the same command.

## Setup

- **Model (192.1M):** run_0006 translator = Steppe-Flux.2 window transformer (width 768, depth 18, head dim 64, MLP
  3072 with depthwise 3x3 conv, condition 512, source encoder 512 with pixel shuffle), source path on the H3 grid
  (24 channels, 16x) -> FLUX.2 grid (8x). Stem sees [noisy residual, linear base]. Flow on the normalized residual
  r = (x_flux - linear_base(h3) - mean) / std over the fixed autoresearch_6 least squares 3x3 map
  (`h3_translators/assets/linear_base_k3.pt`).
- **Init:** Steppe-Flux.2 200M base (`steppe_flux_2_200M_base.safetensors`): 308 / 309 tensors, including the hff
  projection; only `source_encoder.input_conv.weight` is fresh (32 -> 24 channels).
- **hff:** Immerkaer hff of the clean target pixels, 10% dropout to the null embedding (CFG); unlike autoresearch_6,
  which had no hff.
- **Residual table:** measured on 4.2k produced samples of the pretrain mix (`dev/h3_pretrain/residual_statistics.py`):
  std mean 0.740 (1024) / 0.761 (512), within 0.94-1.15 per channel of the autoresearch_6 4KLSDB table; means differ by
  up to 0.23 (ReLAION, random crops, corruption); the linear base explains R2 0.31 (autoresearch_6 test: 0.36).
- **Buckets:** 90% 1024 px (H3 64x64 -> FLUX.2 128x128), 10% 512 px (32x32 -> 64x64), by sample count; micro batch
  2 @1024 / 8 @512, same bucket on all ranks per micro step.
- **Data (online):** ReLAION-high-resolution (urls, min side 1024) : 4KLSDB `train_x4` (streamed) = 1 : 1. Large square
  crops (>= 50% of the short side, median ~85%); target = INTER_AREA resize, encoded by the FLUX.2 VAE (bf16).
- **Corruption (H3 encoder input only, 20%):** JPEG 6% (quality 95 - 75 u^2 on the target pixels); nearest 4%,
  antialias 4%, bicubic 3%, lanczos 3% = the crop resized with another filter than the target's INTER_AREA (falls back
  to clean when the crop needs no resize: measured clean share 83%); the rest clean (= benchmark input).
- **H3 encoder speed-up (`h3_pretrain/h3_encoder.py`):** released tiling (256 px tiles, >= 64 px overlap) with all tiles
  of a batch in one compiled call, and the causal 3D convolutions run as exact 2D convolutions (single frame + zero
  causal padding: only the last temporal kernel slice touches data, 1/3 of the work), bf16. 1024: 3.99 -> 18.9 img/s
  (512: 11.0 -> 50.4); relative latent error vs the diffusers fp32 tiled encode 1.7% (bf16; fp32 batched: 0.4%, TF32).
  Untiled encoding is not an option: 30% relative latent difference to the tiled latents.
- **Process layout:** 6 training GPUs (DDP), 2 encoder GPUs (~10 samples/s each, encoder bound), the visual benchmark
  on the first encoder GPU. Rolling 200 GB cache. Measured trainer throughput: 56 units/s on 7 GPUs, 48.3 on 6
  (26.3 GB peak, no checkpointing); 1 encoder would have meant ~5.6x reuse, 2 encoders ~2.4x.
- **Schedule (open ended):** AdamW (0.9, 0.95), peak LR 6e-4 (autoresearch_6: 3e-4 at batch 4, sqrt scaled to the
  global batch 12), warmup 15k units (~5 min), accumulation x1 -> x2 at 1.5 h with the LR held, constant LR until the
  manual decay, weight decay 0.05, grad clip 1, bf16, EMA 0.9999, training sigma shift 2 (autoresearch_6 run_0006).
  Safety stop at 24 h.
- **Visual benchmark:** 30 Axis-V1 1024 crops (same names as the FLUX.2 pretrain benchmark), H3 latents from the
  diffusers fp32 tiled encode, ground truth hff, guidance 1, 30 steps / shift 2 / temperature 1, FLUX.2 small decoder,
  raw and EMA every 30 min of training: psnr / mse vs ground truth, low_res_mse / chroma_ciede2000 (512 roundtrip, as
  autoresearch_6), lowpass_ciede2000 (64 px area: local colour drift), mean_lab_shift (global colour cast),
  latent_mse; images saved in `pretrain_runs/h3_translator_v1/benchmarks/`.

### Visual benchmark references (30 images)

| reference | psnr | mse | low_res_mse | chroma dE00 | lowpass dE00 | mean Lab shift | latent_mse |
|---|---|---|---|---|---|---|---|
| FLUX.2 small decoder oracle | 36.94 | 5.9e-4 | 1.16e-4 | 1.056 | 0.493 | 0.365 | 0 |
| H3 native decoder (tiled, bf16) — baseline | 31.00 | 2.45e-3 | 1.06e-3 | 2.106 | 0.768 | 0.443 | - |
| linear base only | 22.25 | 8.3e-3 | 6.0e-3 | 6.359 | 5.220 | 6.370 | 0.543 |

## Run `h3_translator_v1`

5.66 h of training, step 50,590, 961k units (~1.0M samples), no crash or restart. Decay triggered manually at 3.66 h
(step 36,380), 2.0 h linear. Throughput 46-48 units/s (6 GPUs), 27.0 GB peak after the x2 step. Producer 455k samples
(ReLAION / 4KLSDB exactly 1 : 1, clean 82%, JPEG 6.0%), mean reuse ~2.2. Final checkpoint local and R2
`checkpoints/h3_translator_v1/checkpoint_000050590.pt`.

### Visual benchmark (30 Axis images, 30 steps / shift 2, ground truth hff, FLUX.2 small decoder)

| hours | step | raw psnr | raw chroma dE00 | raw lowpass dE00 | raw mean Lab shift | raw latent_mse | EMA psnr | EMA chroma dE00 | EMA lowpass dE00 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 7.0k | 30.33 | 2.449 | 1.381 | 1.524 | 0.357 | 30.87 | 2.197 | 1.033 |
| 1.0 | 14.0k | 30.72 | 2.324 | 1.222 | 1.129 | 0.353 | 31.07 | 2.137 | 0.960 |
| 2.0 | 24.6k | 30.85 | 2.241 | 1.070 | 0.936 | 0.335 | 31.25 | 2.080 | 0.889 |
| 3.0 | 31.7k | 30.98 | 2.263 | 1.177 | 1.233 | 0.336 | 31.32 | 2.054 | 0.854 |
| 3.5 (decay start 3.66) | 35.3k | 31.18 | 2.125 | 0.980 | 0.912 | 0.334 | 31.33 | 2.054 | 0.851 |
| 4.5 | 42.4k | 31.24 | 2.083 | 0.910 | 0.812 | 0.322 | 31.38 | 2.035 | 0.829 |
| 5.5 | 49.5k | 31.49 | 1.996 | 0.772 | 0.579 | 0.313 | 31.45 | 2.014 | 0.796 |
| **5.66 final** | 50.6k | **31.50** | **2.003** | **0.768** | **0.510** | **0.311** | 31.46 | 2.011 | 0.792 |
| H3 native decoder | - | 31.00 | 2.106 | 0.768 | 0.443 | - | - | - | - |
| FLUX.2 small oracle | - | 36.94 | 1.056 | 0.493 | 0.365 | 0 | - | - | - |

Training loss (window means): 0.62 at 0.3 h, ~0.57-0.59 flat at constant LR (0.6-3.6 h), 0.54 at the decay end.

Findings:
1. The translator beats the native H3 decoder on PSNR (+0.50 dB) and the 512 roundtrip ΔE00 (2.00 vs 2.11) on this
   set, and ties it on low frequency colour drift (0.768 vs 0.768); the global colour cast is still slightly worse
   (0.51 vs 0.44). The colour gap of autoresearch_6 (PSNR / ΔE00 below H3) is closed.
2. As in the FLUX.2 pretrain, the constant LR phase plateaued after ~2 h (raw metrics noisy, EMA slowly improving) and
   the decay gave most of the late gain (raw lowpass ΔE00 0.98 -> 0.77, mean Lab shift 0.91 -> 0.51 in 2 h).
3. Raw ≈ EMA at the end (raw slightly better on colour); use raw weights, as in the FLUX.2 pretrain.
4. Full 1000-image benchmark below; the 1-step distillation is not done yet.

### Full benchmark (autoresearch_6 set, 1000 Axis-V1 1024 crops, 30 steps / shift 2, final checkpoint)

`python scripts/run_h3_pretrain_benchmark.py --checkpoint pretrain_runs/h3_translator_v1/checkpoints/checkpoint_000050590.pt
--settings raw_null raw_truth ema_null` (8 GPUs, ~14 min including the set build cache; set rebuilt with
`build_benchmark` of `scripts/encode_h3_datasets.py`, ~11 min on one GPU). The references reproduce autoresearch_6
exactly. Results: `raw_results/h3/h3_translator_v1_50590_*`.

| row | patch_fid | fid | psnr | low_res_mse | chroma dE00 | TAEF2 patch_fid / fid | latent_mse |
|---|---|---|---|---|---|---|---|
| H3 native decoder (tiled) — baseline | 2.47 | 4.37 | 29.78 | 1.03e-3 | 2.29 | - | - |
| FLUX.2 small decoder oracle | 0.68 | 0.78 | 36.53 | 1.05e-4 | 1.12 | (oracle 2.25 / 3.62) | 0 |
| autoresearch_6 run_0004 (Steppe init, 60 min, 1 GPU) | 1.84 | 4.10 | 29.46 | - | 2.49 | 3.11 / 5.46 | - |
| **raw, null hff** | **1.565** | **2.916** | **30.34** | 9.51e-4 | **2.133** | 2.918 / 5.000 | 0.3138 |
| raw, ground truth hff | 1.541 | 2.899 | 30.34 | 9.51e-4 | 2.134 | 2.901 / 4.981 | 0.3129 |
| EMA, null hff | 1.572 | 2.953 | 30.32 | 9.55e-4 | 2.144 | 2.999 / 5.035 | 0.3188 |

5. **The translator beats the native H3 decoder on every quality metric** with the FLUX.2 small decoder: patch_fid
   -37% (1.57 vs 2.47), fid -33% (2.92 vs 4.37), PSNR +0.56 dB, roundtrip ΔE00 2.13 vs 2.29, low_res_mse -8%. The
   colour / fidelity gap of autoresearch_6 (PSNR and ΔE00 worse than H3) is closed. With the assumed +5% distillation
   cost (1.64 / 3.06) it stays clearly ahead on quality: the "still useful" tier holds on all quality metrics. Speed /
   VRAM are unchanged from autoresearch_6 (same architecture): 1024 end to end ~33 + 70 ms with a 1-step student vs
   40 ms for batched H3 tiles.
6. vs the best autoresearch_6 run (run_0004): patch_fid -15%, fid -29%, PSNR +0.88 dB, ΔE00 -0.36.
7. hff barely matters at guidance 1 (truth vs null: -0.02 patch_fid): null hff, the only option for generated H3
   latents, loses nothing. Raw beats EMA slightly (use raw).
8. TAEF2 still loses to H3 (2.92 / 5.00 vs 2.47 / 4.37) even though the TAEF2 gap shrank (oracle 2.25 / 3.62): the
   fast "100% viable" stack still needs a better tiny decoder or a near-oracle translator.

## Single step rectified flow distillation (`h3_distill_v1`)

Code: `src/stdev/h3_distill/` (config, student, pairs, trainer, evaluation), launcher `scripts/run_h3_distill.py`
(`python scripts/run_h3_distill.py --run-name h3_distill_v1`). FLUX.2 distillation recipe (`reports/flux_rf_distillation.md`):
teacher = `h3_translator_v1` final raw weights (30 steps, shift 2); 7 generator GPUs make online pairs from the
`h3_data` cache inputs (hff true 45% / random 45% / null 10%, CFG U[1, 2] on half of the non-null, negative = null),
1 GPU trains the student (teacher + zero initialized guidance embedding, exactly the teacher at init, sigma 1 only).
25 min training (20 in the FLUX.2 run), lr 3e-4, 50 warmup steps, linear decay over the last 30%, accumulation 4,
EMA 0.995. 1,500 steps, 13.0k samples seen, 9.8k pairs produced (reuse 1.38). ~42 min wall time in total, including
pair warmup (~6 min) and the parallel evaluation. Checkpoint local and R2 `checkpoints/h3_distill_v1/checkpoint_001500.pt`
(`H3StudentTransformer`, raw + EMA).

Evaluation: 1 step on the autoresearch_6 Axis 1k set, FLUX.2 small decoder, fixed noise seed per image, null hff
unless noted (`distill_runs/h3_distill_v1/evaluations/summary.md`).

| evaluation | minutes | patch_fid | fid | psnr | low_res_mse 1e-4 | chroma dE00 | TAEF2 patch_fid / fid | latent_mse |
|---|---|---|---|---|---|---|---|---|
| init (teacher weights, 1 step) | 0 | 7.20 | 4.57 | 31.16 | 7.12 | 1.970 | - | 0.178 |
| raw | 5 | 1.97 | 3.57 | 29.97 | 9.92 | 2.288 | - | 0.325 |
| raw | 10 | 2.18 | 3.81 | 29.89 | 10.21 | 2.387 | - | 0.326 |
| raw | 15 | 1.86 | 3.48 | 30.06 | 9.64 | 2.244 | - | 0.323 |
| raw | 20 | 1.82 | 3.35 | 29.92 | 9.96 | 2.241 | - | 0.324 |
| EMA, final | 25 | 1.72 | 3.18 | 30.33 | 9.48 | 2.140 | - | 0.317 |
| **raw, final** | 25 | **1.665** | **3.171** | **30.32** | 9.49 | **2.141** | 3.058 / 5.172 | 0.319 |
| raw, final, ground truth hff | 25 | 1.574 | 3.091 | 30.28 | 9.61 | 2.151 | - | 0.319 |
| raw, final, ground truth hff, cfg 2 | 25 | 1.626 | 3.109 | 30.24 | 9.62 | 2.155 | - | 0.317 |
| teacher, 30 steps, null hff | | 1.565 | 2.916 | 30.34 | 9.51 | 2.133 | 2.918 / 5.000 | 0.314 |
| H3 native decoder | | 2.47 | 4.37 | 29.78 | 10.3 | 2.29 | - | - |

Findings:
1. **The 1-step student still beats the native H3 decoder on every quality metric:** patch_fid -33% (1.67 vs 2.47),
   fid -27% (3.17 vs 4.37), PSNR +0.54 dB, ΔE00 2.14 vs 2.29. PSNR / ΔE00 / low_res_mse equal the teacher's.
2. Distillation cost vs the 30-step teacher: +6% patch_fid, +9% fid (the FLUX.2 upscaler student: +4% / +5%), i.e.
   ~89% of the teacher's patch_fid gain over H3. Close to the +5% assumed in autoresearch_6.
3. Still improving at the end: most of the gain came in the decay (raw 1.82 -> 1.67 between 20 and 25 min; the
   constant LR phase was noisy, 10 min worse than 5 min). A longer run (40 min, longer decay) should close more of the gap.
4. Raw beats EMA (1.67 vs 1.72), as in the FLUX.2 student. Ground truth hff helps the student more than the teacher
   (1.57 vs 1.67, teacher: 1.54 vs 1.57); cfg 2 costs +0.05 patch_fid (visual effect to be judged in ComfyUI).
5. TAEF2 decoding of the student (3.06 / 5.17) still loses to H3.
6. Speed (autoresearch_6 numbers, same architecture): 1 step 33 ms + FLUX.2 small decoder 70 ms = ~103 ms at 1024 vs
   40 ms batched H3 tiles: "still useful" tier (better quality, slower, more VRAM).

### Longer distillation (`h3_distill_v2`: 60 min, linear decay over the last 80%)

Same setup as v1 except 60 min training and `--decay-fraction 0.8` (constant 3e-4 for 12 min, then 48 min decay).
3,604 steps, 31.2k samples, 21.0k pairs (reuse 1.50). Distillation loss at the end 0.0125 (v1: 0.0201). Checkpoint
R2 `checkpoints/h3_distill_v2/checkpoint_003604.pt`. Snapshots now carry the float32 buffers (`buffers.` prefix).

| evaluation (1 step, null hff) | minutes | patch_fid | fid | psnr | chroma dE00 | latent_mse |
|---|---|---|---|---|---|---|
| raw | 10 | 1.73 | 3.38 | 30.03 | 2.260 | 0.333 |
| raw | 20 | 1.73 | 3.39 | 29.95 | 2.297 | 0.328 |
| raw | 30 | 1.88 | 3.44 | 30.35 | 2.143 | 0.321 |
| raw | 40 | 1.96 | 3.39 | 30.18 | 2.199 | 0.319 |
| raw | 45 | 1.69 | 3.15 | 30.31 | 2.148 | 0.320 |
| raw | 50 | 1.65 | 3.17 | 30.33 | 2.154 | 0.320 |
| raw | 55 | 1.71 | 3.17 | 30.26 | 2.159 | 0.318 |
| **raw, final** | 60 | 1.733 | **3.148** | 30.33 | 2.142 | **0.317** |
| EMA, final | 60 | 1.736 | 3.161 | 30.33 | 2.145 | 0.317 |
| raw, final, ground truth hff | 60 | 1.618 | 3.076 | 30.30 | 2.146 | 0.316 |
| raw, final, ground truth hff, cfg 2 | 60 | 1.629 | 3.074 | 30.29 | 2.147 | 0.316 |
| TAEF2 (raw, final) | 60 | 3.120 | 5.144 | | | |
| v1 raw, final (25 min) | 25 | 1.665 | 3.171 | 30.32 | 2.141 | 0.319 |

Findings:
1. **2.4x longer training did not improve the student:** final 1.73 / 3.15 vs v1 1.67 / 3.17 (patch_fid +0.07, fid
   -0.02, PSNR / ΔE00 equal). The distillation loss kept falling (0.020 -> 0.0125) and latent_mse improved slightly
   (0.319 -> 0.317), but the decoded metrics did not follow.
2. Snapshot to snapshot noise of patch_fid is large (1.65-1.96 within the decay, e.g. 40 min 1.96 vs 45 min 1.69), so
   v1 vs v2 differences of < ~0.1 patch_fid are within noise of single evaluations. fid is smoother: it improved
   until ~45 min and was flat afterwards.
3. Practical conclusion: ~25-30 min with a short decay is enough; the remaining gap to the teacher (~+6-10% patch_fid,
   ~+8% fid) is not a training length problem. Candidates: a better objective (e.g. a few-step student, or
   distribution matching / adversarial post-training), or more diverse pairs (reuse 1.5).
