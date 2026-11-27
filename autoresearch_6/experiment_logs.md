# Autoresearch 6 experiment log: H3-Image latents -> FLUX.2 latents

Setup and rules: `autoresearch_6/instruction.md`.

## Round 0: data, references, speed (2026-11-27)

Data: 40,885 4KLSDB 1024 crops (65 parquet files, no unreadable sources) encoded with the H3-Image encoder (fp32,
tiled) and the FLUX.2 VAE -> `train_datasets/h3_flux2_4klsdb_40k` (48 GB). Benchmark `axis_v1_1k_h3_flux2` (1000
images). Download ~1.5 h (HF ~40 MB/s on this machine), encode ~1 h on 4 GPUs (~9 img/s, the fp32 tiled H3 encoder
dominates).

### Quality references (Axis 1k, 1024 px)

| Decoder | patch_fid | fid | PSNR | ΔE00 |
|---|---|---|---|---|
| H3 native (tiled) — baseline | 2.47 | 4.37 | 29.78 | 2.29 |
| H3 native, untiled | 126.4 | 85.8 | 20.19 | 6.47 |
| ostris fine-tune (tiled) | 2.92 | 2.85 | 31.02 | 1.88 |
| FLUX.2 VAE oracle | 0.71 | 0.72 | 36.57 | 1.08 |
| FLUX.2 small decoder oracle | 0.68 | 0.78 | 36.53 | 1.12 |
| TAEF2 oracle | 2.25 | 3.62 | 32.87 | 1.82 |

ostris trades patch_fid (+18%) for fid (-35%) and fidelity: the smoothing is visible in patch_fid.

### Decode speed (batch 1, eager bf16, RTX 5090; peak VRAM = decoder weights + activations)

| Decoder | 1024 | 2048 | 4096 |
|---|---|---|---|
| H3 native, diffusers (tiles one by one) | 134 ms / 0.81 GB | 637 ms / 0.92 GB | 2363 ms / 1.35 GB |
| H3 native, all tiles batched (chunks of 64) | 40 ms / ~1.04 GB | 207 ms / ~1.46 GB | 763 ms / ~1.46 GB |
| ostris (diffusers) | 129 ms / 0.81 GB | 634 ms / 0.92 GB | 2339 ms / 1.35 GB |
| FLUX.2 VAE (untiled) | 97 ms / 2.53 GB | 475 ms / 9.84 GB | OOM (32 GB) |
| FLUX.2 small decoder (untiled) | 70 ms / 1.88 GB | 346 ms / 7.37 GB | 8901 ms / 29.3 GB (memory bound, barely fits) |
| TAEF2 (untiled) | 8 ms / 0.64 GB | 36 ms / 2.57 GB | 148 ms / 10.26 GB |

The diffusers H3 decode is launch bound (25 / 121 / 441 tiles of 16x16 tokens through 24 blocks, one tile at a time);
batching the tiles gives 3x. The batched numbers exclude the tile blend (cheap) and are the fair baseline for the
"100% viable" criterion. Script: `dev/autoresearch_6/batched_tile_speed.py`.

### What this implies for viability

- Quality headroom is large with the full FLUX.2 decoder (oracle patch_fid 0.71 vs 2.47), but the FLUX.2 decoder alone
  is slower than batched H3 at every size (97 vs 40 ms at 1024) and needs much more VRAM (2.5 vs 0.8-1.0 GB at 1024,
  OOM at 4096 untiled): at best the "still useful" tier unless the decode is tiled.
- With TAEF2 the stack can be fast (8 / 36 / 148 ms decode leaves 30 / 170 / 600 ms for the model vs batched H3), but
  the TAEF2 oracle is only 2.25 patch_fid: with perfect latents and the assumed +5% distillation drop it is ~2.36 vs
  2.47. The 100% viable tier through TAEF2 needs a near-oracle translator (fid has more room: 3.62 vs 4.37).
- VRAM: H3 stays ~1-1.5 GB at all sizes because of tiling. TAEF2 untiled needs 10 GB at 4096; both the translator and
  TAEF2 must run tiled / chunked to win on VRAM.
- Tiling is part of the H3 model: whole-image decode collapses (RoPE on resolution normalized coordinates).

### FLUX.2 small decoder check

`black-forest-labs/FLUX.2-small-decoder` (Apache-2.0) is a distilled drop-in decoder: decoder widths [96, 192, 384, 384]
(27.95M vs 49.62M parameters), retrained `post_quant_conv`. The encoder, quant_conv and batch norm statistics equal our
klein-4B VAE after bf16 rounding (the repository stores fp32, klein stores bf16), so the latent space and the encoded
datasets are shared. Quality equals the full decoder (patch_fid 0.68 vs 0.71, fid 0.78 vs 0.72, PSNR 36.53 vs 36.57) at
-28% latency and -26% VRAM at 1024 / 2048. It is still slower than batched H3 (70 vs 40 ms at 1024) and heavier
(1.9 vs ~1.0 GB). At 4096 it needs tiled decoding. It replaces the full FLUX.2 decoder as the quality decoder of this
project.

## Round 1: does the direct approach learn, residual vs direct, Steppe init, H3 decoder features

Setup: `src/stdev/h3_translators/run_<ID>.py` (interface `base.Translator`), `./h3_infra/start_round.sh`, 30 min per run
on one GPU, batch 4, 30 sampling steps / shift 2 / temperature 1.0 (FLUX.2 recipe). Benchmark: speed of one model pass
at 1024 / 2048 / 4096, quality decoded with the FLUX.2 small decoder (`quality`) and TAEF2 (`quality_taef2`).

Data analysis (`dev/autoresearch_6/linear_base_fit.py`): a least squares map from a 3x3 H3 token neighbourhood to the
2x2 FLUX.2 patch explains R2 0.36 of the target variance (1x1: 0.27, 5x5: 0.37). Residual std 0.36-0.91 per channel
(mean ~0.75). Steppe-Flux.2's 2x task is similar: the bilinear source explained 41%, residual std ~0.7. The linear map
(`assets/linear_base_k3.pt`) is the "base" input of all runs and the residual base of the residual runs.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0000 | direct flow, 47.5M scratch (w384 d12, source path 384 on the H3 grid, stem sees noisy + linear base) | AR5 0003 | does the direct approach learn |
| 0001 | 0000 with the linear residual parameterization | 0000 | residual vs direct |
| 0002 | 0001 at 191.7M (w768 d18), Steppe-Flux.2 base init (304 / 309 tensors; new: H3 source input conv), LR 3e-4, checkpointing (~2.8k steps) | 0001, Steppe-Flux.2 | does Steppe transfer |
| 0003 | 0001 + frozen features of the first 8 H3 decoder blocks (16x16 token tiles) into the source path | 0001 | do H3 decoder features help |

Fixed before launch: the compile warm-up computed the linear base under autocast (bf16) while the loop passes fp32,
which recompiled every run on the first step (~4 min of max-autotune lost); whole-model compile of the 200M model
OOMed (checkpointing ignored), so 0002 compiles per block like the Steppe pretrain.

Launched: ./h3_infra/start_round.sh 0000 0001 0002 0003

### Round 1 results (30 steps / shift 2 / T 1.0; quality with the FLUX.2 small decoder, TAEF2 in the last columns)

| run | change | steps | patch_fid | fid | PSNR | ΔE00 | TAEF2 patch_fid / fid | model ms 1024 / 2048 / 4096 | model VRAM GB 1024 / 4096 |
|---|---|---|---|---|---|---|---|---|---|
| H3 native | baseline | - | 2.47 | 4.37 | 29.78 | 2.29 | - | (decode 40 / 207 / 763 batched) | ~1.0 / ~1.5 |
| 0000 | direct, 47.5M | 14.9k | 2.44 | 5.04 | 29.01 | 2.61 | 4.08 / 6.32 | 7.5 / 36 / 144 | 0.26 / 2.67 |
| 0001 | linear residual | 15.1k | 2.34 | 4.81 | 29.11 | 2.61 | 4.04 / 5.96 | 7.5 / 36 / 144 | 0.26 / 2.67 |
| 0002 | Steppe 200M init | 2.8k | **1.94** | **4.71** | 29.12 | 2.64 | 3.23 / 5.90 | 33 / 139 / 549 | 0.69 / 5.45 |
| 0003 | + H3 decoder features (8 blocks) | 11.8k | 2.19 | 5.27 | 28.79 | 2.76 | 3.89 / 6.38 | 17 / 72 / 301 | 0.53 / 3.15 |

Model latency = one compiled forward (a distilled 1-step student), batch 1, decoder excluded.

- The direct approach learns: after 30 min every run is at or below the H3 patch_fid (2.47) with the FLUX.2 small
  decoder; Steppe init gives -21% (1.94) after only 2.8k fine-tuning steps. Residual over the linear base helps a little
  (2.34 vs 2.44, fid 4.81 vs 5.04).
- fid, PSNR and ΔE00 are worse than H3 in every run. Diagnosis (32 saved images, error by pixel frequency band): the
  detail band error is H3-like (6.5 vs 6.1 RMS, expected for a generative model), but the low-frequency colour error is
  2-2.5x H3's (low-pass chroma RMSE 1.0-1.3 vs 0.52; the FLUX.2 oracle 0.33). Mean chroma is unchanged (ratio 1.00-1.02):
  colours drift locally (e.g. pink skin -> grey), not a global desaturation.
- Sampler sweep on 0001 (`dev/autoresearch_6/sampler_sweep.py`, patch_fid / fid / ΔE00): T 0.9 2.72 / 4.94 / 2.54,
  T 0.8 3.32 / 5.20 / 2.50, shift 3 2.38 / 4.82 / 2.60, shift 3 T 0.9 2.79 / 4.98 / 2.54. A lower temperature blurs
  and barely improves colour: the colour drift is a training-side problem. Keep 30 / 2 / 1.0.
- H3 decoder features: better patch_fid (2.19 vs 2.34) but worse fid / colour and 2x latency (the 8 frozen ViT blocks
  cost ~9 ms at 1024): not worth it in this form.
- TAEF2 costs +1.3 ... +1.7 patch_fid on our latents (oracle +1.5): no TAEF2 run beats H3 yet.
- Training trend (checkpoints at 25 / 50 / 75 / 100 %, patch_fid / fid / ΔE00): 0001 4.32 / 8.43 / 3.95 -> 3.37 / 6.78 /
  3.22 -> 2.61 / 5.50 / 2.86 -> 2.34 / 4.81 / 2.61; 0002 at 50 % 3.02 / 6.30 / 3.17 -> 1.94 / 4.71 / 2.64. Every metric,
  colour included, is still falling fast (most of it during the LR decay): the runs are undertrained, and the colour
  drift is at least partly an undertraining symptom.

## Round 2: time scaling and the low-frequency colour error

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0004 | 0002 for 60 min | 0002 | time scaling of the Steppe-initialized model |
| 0005 | 0002 + velocity error below 1/16 cycles / latent px weighted x8 (30 min) | 0002 | low-frequency loss against the colour drift |
| 0006 | 0002 with training sigma shift 2 (30 min) | 0002 | more training at high noise, where coarse colour is decided |
| 0007 | 0001 (47.5M scratch) for 60 min | 0001 | time scaling of the small scratch model |

Launched: TRAINING_TIME_LIMIT=70m ./h3_infra/start_round.sh 0004 0005 0006 0007

### Round 2 results (30 steps / shift 2 / T 1.0; FLUX.2 small decoder, TAEF2 in the last column)

| run | change | steps | patch_fid | fid | PSNR | ΔE00 | TAEF2 patch_fid / fid | model ms 1024 / 2048 / 4096 |
|---|---|---|---|---|---|---|---|---|
| H3 native | baseline | - | 2.47 | 4.37 | 29.78 | 2.29 | - | (decode 40 / 207 / 763 batched, 134 / 637 / 2363 diffusers) |
| 0002 | Steppe 200M, 30 min (round 1) | 2.8k | 1.94 | 4.71 | 29.12 | 2.64 | 3.23 / 5.90 | 33 / 139 / 549 |
| 0004 | 0002 for 60 min | 5.4k | **1.84** | **4.10** | **29.46** | 2.49 | 3.11 / 5.46 | 33 / 142 / 562 |
| 0005 | 0002 + low-frequency loss x8 | 2.7k | 2.06 | 5.12 | 28.87 | 2.69 | 3.44 / 6.39 | 34 / 143 / 564 |
| 0006 | 0002 + training sigma shift 2 | 2.8k | 1.91 | 4.60 | 29.18 | 2.58 | 3.24 / 5.94 | 33 / 140 / 552 |
| 0007 | 0001 (47.5M scratch) for 60 min | 31.3k | 2.03 | 4.26 | 29.38 | 2.48 | 3.64 / 5.62 | 7.2 / 35 / 140 |

- Time is the lever: 2x training improves every metric, colour included (0004 vs 0002: fid -13%, ΔE -0.15, PSNR
  +0.34; 0007 vs 0001: patch_fid -13%, fid -11%, ΔE -0.12). Both 60 min runs beat H3 on patch_fid and fid; PSNR and
  ΔE00 are still worse than H3 (29.46 vs 29.78, 2.49 vs 2.29), but the gap halves per doubling of training.
- The low-frequency weighted loss hurts every metric (rejected). Training sigma shift 2 is a small gain (-0.03
  patch_fid, -0.11 fid, -0.06 ΔE; within seed noise but consistent across metrics): keep it for the long run.
- With the assumed +5% distillation cost: 0004 -> 1.93 / 4.30 (H3 2.47 / 4.37, both still better), 0007 -> 2.13 / 4.47
  (fid no longer better).
- TAEF2 still loses to H3 on both FIDs (best 3.11 / 5.46).

### Viability after round 2

"Still useful" tier with the FLUX.2 small decoder: better patch_fid (-26%) and fid (-6%) than the native H3 decoder,
but slower and heavier. 1024 px end to end (1-step model + small decoder): 0004 33 + 70 = 103 ms, 0007 7 + 70 = 77 ms vs
H3 40 ms batched / 134 ms diffusers; 2048: 142 + 346 = 488 ms vs 207 / 637 ms; 4096 needs a tiled FLUX.2 decode (the
untiled small decoder uses 29 GB). VRAM at 1024: 0.7 GB model + 1.9 GB decoder vs ~1 GB. Not 100% viable: the FLUX.2
decoder alone is slower than batched H3, and TAEF2 is not good enough. Colour fidelity (PSNR / ΔE00) is still below H3
and is the remaining quality gap.

Handed over to a full 8 GPU training run on another machine (user decision); no round 3 here.
