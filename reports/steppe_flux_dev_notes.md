# Steppe FLUX.2 upscaler: development notes (pretrain, distillation, small student)

Working memory of the FLUX.2 sessions (2026-11-23 / 24, 8x RTX 5090), written for the next VAE (Qwen-Image or
other), so that the next session spends its time on training, not on re-discovering code and pitfalls.
Results and analysis live in the reports; this file is about **how things are built, how to run them, and what went
wrong**.

Reports: `reports/flux_pretrain.md`, `reports/flux_rf_distillation.md`, `reports/flux_small_student.md`;
history in `CONTEXT.md` sections 15-18; recipe source `autoresearch_3/experiment_logs.md`.

## 1. The whole pipeline in one table

| phase | wall time | GPUs | command | output |
|---|---|---|---|---|
| benchmark set (autoresearch_3) | ~5 min | 1 | built by `scripts/flux_tools/flux_pretrain_full_benchmark.py` if missing (calls `build_benchmark` of `scripts/encode_flux2_datasets.py`) | `benchmark_datasets/axis_v1_1k_flux2_2x` |
| smoke tests + residual table | ~1.5 h | all | see section 4 | table into `flux_pretrain/config.py` |
| pretrain | 24 h + ~15 min warmup | 7 train + 1 encode | `python scripts/run_flux_pretrain.py --run-name flux_pretrain_v1 --data-run-name flux_data` | `pretrain_runs/flux_pretrain_v1` |
| large 1-step distillation | ~35 min | 7 generate + 1 train | `python scripts/run_flux_distill.py --run-name flux_distill_v1` | `distill_runs/flux_distill_v1` |
| small 20M student | 6 h + ~10 min | 4 generate + 4 train | `python scripts/run_flux_small_distill.py --run-name flux_small_s20m` | `distill_runs/flux_small_s20m` |

Results (autoresearch_3 benchmark, 1000 Axis images, patch_fid): teacher 30 steps 1.69, large 1-step student 1.76,
20M student 2.21, pixel bicubic 8.12, VAE oracle 0.71. All runs: no crash, no restart.

Always launch long jobs from the agent with `run_in_background: true` (redirect to a log), then wait for the
completion notification. For errors-only monitoring use ONE background command that exits on the first error line:
`tail -n0 -F logs... | grep -E -m1 --line-buffered "CRASHED|HUNG|FAILED|Traceback|OutOfMemory|NCCL error|Killed"`
(the Monitor tool expires every 30 min and wakes the agent; the user did not want periodic status).

## 2. Code map

### Pretrain: `src/stdev/flux_pretrain/`

| file | content |
|---|---|
| `config.py` | everything tunable: `MODEL` (`ModelConfig`, incl. `RESIDUAL_STD`), `BENCHMARK_SAMPLER`, `PairType` buckets (`image_512_1024` 0.9, `image_256_512` 0.1: micro batch, shard size, encode batch), datasets 1:1, crop rule, corruption weights and JPEG range, process layout (`TRAINING_GPUS`, `ENCODER_GPUS`), cache limit, schedule (`TRAINING_HOURS`, `ACCUMULATION_SCHEDULE` as time fractions, LR, decay, wd, EMA), intervals |
| `model.py` | `FluxUpscalerTransformer` (window transformer, adaLN-zero, shifted 8x8 windows, QK-norm, GELU MLP + depthwise conv, source encoder with pixel shuffle, condition = RMS norm(sigma emb + hff emb / learned null)), `to_residual` / `from_residual` (per channel table in `ModelConfig.residual_std`), `upscale` (Euler sampler, NaN hff = null), `forward_with_condition` (reused by the students). **`LATENT_CHANNELS = 32` is a module constant.** |
| `corruption.py` | `large_square_crop` (side >= 50% of short side, skewed to full), `downscale_variant` (bilinear / nearest / bicubic / lanczos / antialias / jpeg), `draw_jpeg_quality` (95 - 75 u^2) |
| `cache.py` | rolling latent cache: shards per pair type (`.vz.safetensors`, vaezip lbrq 8 bit for latents stored as 5D, fp16 scalars), `PairTypeDataset` (endless random shard reader), reuses Wan `list_shards` / eviction |
| `producer.py` | online data: ReLAION url download workers (reused `stdev.pretrain.producer.run_download_worker`), 4KLSDB parquet streaming readers (prefetch next file, delete after use), preprocess workers per dataset (decode, crop, corrupt), encoder process per GPU (feeder thread -> compiled FLUX.2 encoder -> writer thread), deficit balancing of datasets, eviction, `producer_statistics.json` |
| `trainer.py` | DDP trainer: same pair type on all ranks per micro step, time based schedule synchronized across ranks, EMA on rank 0, snapshots for the watcher, checkpoints every 30 min, R2 upload every 3 h, `control/stop`, resume, rich tensorboard (loss by type / sigma / corruption variant / dataset, grad norm by group, weight norm, reuse) |
| `benchmark.py` | visual benchmark: `--build` (30 Axis images, GT / LR PNG, latents, hff) and the watcher (every snapshot, raw + EMA, low_res_mse / chroma / latent_mse_hr, images saved) |

Launcher `scripts/run_flux_pretrain.py`: builds the visual benchmark set, starts producer + watcher, waits for a warm
cache (`WARMUP_MIN_SHARDS_PER_TYPE` shards per type AND `WARMUP_MIN_SAMPLES`), starts torchrun, restarts crashed
services / trainer (resume), kills a trainer whose log is silent > 20 min (stack dump via SIGUSR1 first).

### Distillation: `src/stdev/flux_distill/`

| file | content |
|---|---|
| `config.py` | large distillation: teacher checkpoint + `TEACHER_WEIGHTS="raw"`, sampler 30 / 2, condition mix (null 10%, random hff 45%, true 45%; CFG U[1, 2] for half of non-null), GPUs (7 generate, 1 train), training minutes, LR 3e-4, EMA 0.995 |
| `student.py` | `FluxStudentTransformer` (teacher + zero initialized guidance embedding, sigma fixed 1), `load_teacher_into_student`, `teacher_state` (raw or EMA state dict from a pretrain checkpoint), `model_config_of`, `seeded_noise` (bit exact per sample noise from int64 seeds), `upscale_single_step` |
| `pairs.py` | teacher sampler with CFG (`sample_teacher`, guided samples in a doubled batch), `draw_conditions`, `read_shard_inputs` (source-only decode), `load_job`, shard writer / `PairDataset`, generator `main` |
| `trainer.py` | single GPU large student trainer (fixed minutes, snapshots every 4 min, final checkpoint + R2) |
| `evaluation.py` | 1-step autoresearch_3 benchmark of a snapshot (`--weights`, `--guidance`, `--hff`, `--model large|small`) |
| `small_config.py` | `SMALL_MODEL` (256 x 14, 20.2M), large student checkpoint, GPU split 4 + 4, micro batches, LR sqrt scaling, 6 h |
| `small_pairs.py` | online pair generator of the large student (1 step, 2 prefetch threads, rolling cache with eviction) |
| `small_trainer.py` | DDP trainer of the small student (time schedule synchronized, snapshots / checkpoints every 30 min, resume) |

Launchers: `scripts/run_flux_distill.py` (generators -> trainer when `MIN_PAIRS_TO_START` -> stop generators ->
parallel evaluation of all snapshots on 8 GPUs -> `evaluations/summary.md`), `scripts/run_flux_small_distill.py`
(generators + DDP trainer, evaluation of each new snapshot on the last generator GPU, final evaluations in parallel).

### Analysis and measurement tools (`scripts/flux_tools/`)

| script | purpose |
|---|---|
| `scripts/flux_tools/flux_pretrain_full_benchmark.py` | autoresearch_3 benchmark of a pretrain checkpoint (`--weights`, `--hff`, `--steps`, `--shift`, `--batch`) or `--reference oracle|pixel_bicubic`; builds the set if missing |
| `scripts/flux_tools/flux_model_diagnostics.py` | residual stream RMS per block, adaLN gate / scale magnitudes, per head attention (max logit, entropy, peak probability, output share), QK-norm gains, relative bias; several checkpoints for trends |
| `scripts/flux_tools/flux_hff_sensitivity.py` | relative velocity change hff truth vs null, 0.9 vs 0.1 (CPU) |
| `scripts/flux_tools/flux_generalization_gap.py` | loss on cache shards by age (memorization check); compare two checkpoints on the same shards |
| `scripts/flux_tools/flux_pretrain_curves.py` | hour window means of all tensorboard scalars + visual benchmark trend |
| `scripts/flux_tools/flux_pretrain_residual_std.py` | per channel residual std per bucket and on the mix -> `RESIDUAL_STD` |
| `scripts/flux_tools/bench_flux_encoder.py` | VAE encoder throughput eager vs compiled |
| `scripts/flux_tools/bench_small_student_training.py` | training throughput / memory of a student config with synthetic data |
| `scripts/flux_tools/measure_flux_students.py` | 1-step model latency and memory |

## 3. Pitfalls hit and their fixes (most expensive first)

1. **DDP deadlock risk from per-rank clocks.** Anything that changes the step structure (accumulation factor, end of
   training) must use the training time all-reduced at control steps (`schedule_seconds`), never the rank's own clock.
2. **Generator input loading was the bottleneck, not the GPU.** Pretrain cache shards are ~80 MB (128 samples incl.
   1024 latents) and take 1.2 s to decode; reading 4 shards per 64 pairs on one thread capped a generator at ~14-16
   pairs/s. Decode only the needed keys (`read_shard_inputs`), one shard per job, 2 prefetch threads -> 30 pairs/s.
   Measure each process alone AND in the final layout before a long run.
3. **Encoder throughput:** eager, serial encoder loop gave 12.8 samples/s; compiling `vae.model.encoder`
   (`dynamic=False`, fixed batch shapes per pair type) and moving dequeue/stacking and shard writing into threads
   gave 19 (GPU bound). Pinned memory + `non_blocking`.
4. **GPU memory on the shared encoder GPU:** the visual benchmark watcher keeps ~16 GB of allocator cache and the
   encoder ~6 GB, pausing the encoder (`control/encoder_pause` in the data run) frees compute but not memory: only ~9 GB
   for extra jobs. The full benchmark needs batch 2 there (encode / decode / FID at batch 8 OOMs), with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Diagnostics: loop over heads (a [M, heads, t, D] tensor is 2.4 GB).
5. **vaezip lbrq codec needs 5D tensors** (`block_size [1, 8, 8]`): image latents are stored as [N, C, 1, h, w] and
   squeezed on read.
6. **VAE wrapper shapes:** image VAEs take 4D pixels in `encode` and return 5D latents [B, C, 1, h, w]; `decode` takes
   5D latents and returns 4D pixels (no `[:, :, 0]` after decode).
7. **Eviction helper of the Wan cache only matches `.vz.safetensors`**: pair shards (`.safetensors`) need their own
   (`small_pairs.evict_oldest_pairs`), otherwise the pair cache grows without bound.
8. **`pkill -f <pattern>` killed the agent's own shell** (pattern in its command line, exit 144). Kill by PID:
   `ps -eo pid,args | grep "[p]attern" | awk '{print $1}' | xargs -r kill`.
9. **Leftover processes:** a watcher started for a smoke run kept polling (and holding 16 GB) after its run directory
   was deleted; a smoke generator recreated a deleted directory. Check `nvidia-smi` / `ps` before launching, and wait
   for `--max-seconds` processes to exit before deleting their directory.
10. **Param count of a reused shape changes per VAE:** the Wan 768 x 16 shape is 222M with temporal layers but 173M
    for an image-only VAE (FLUX.2 used depth 18 for 192M); the Wan s20m shape 256 x 10 is 15.4M (FLUX.2 used 256 x 14).
    Always print the param count before deciding.
11. **Benchmark reference definitions differ:** autoresearch_3 `pixel_bicubic` upscales the *decoded* 512 latents;
    the visual benchmark bicubic upscales the 512 PNG. Rebuilt benchmark sets must reproduce the oracle (0.71 / 0.72 /
    1.0 / 1.08 for FLUX.2) before comparing numbers. Axis-V1 has 2000 images; the benchmark uses the first 1000 in
    sorted relative path order.
12. **Use raw weights after a decay:** pretrain EMA 0.9999 was 0.19 patch_fid worse than raw at the end; large
    student EMA 0.995 0.05 worse. Evaluate both; save both in every snapshot.
13. **AGENTS.md rules:** no HEREDOC in shell commands (use small Python `-c` replacements or Write/Edit), full
    variable names, type hints, shape docstrings; scratch code in `./dev`. Downloaded data: run Python with `-I`.
14. **Foreground `sleep` is blocked** in this agent: wait with `timeout N bash -c 'until <condition>; do sleep 10; done'`.
15. `hf download ... "Still waiting to acquire lock"` is harmless (the download continues).

## 4. Checklist for the next VAE (e.g. Qwen-Image 2.1)

What is VAE specific in the FLUX.2 code (port by copying `flux_pretrain` / `flux_distill` to `<vae>_pretrain` /
`<vae>_distill`, or parametrize):

- `LATENT_CHANNELS` (FLUX.2 32; **Qwen-Image 2.1: 64**), the VAE class (`stdev.vaes.qwen_image_2_1_vae.QwenImage21VAE`)
  and its directory.
- Spatial compression: `PairType.target_latent_size` uses `side // 8` (**Qwen-Image 2.1 compresses 16x**: 1024 px ->
  64 x 64 latents, 512 -> 32, 256 -> 16; all multiples of the 8x8 window, but the token count per 1024 image is 4x
  smaller: much faster training, rethink batch sizes / `TOKENS_PER_UNIT`).
- **Qwen-Image 2.1 VAE takes RGBA pixels** ([B, 4, H, W], alpha 1 = opaque): add an alpha channel of ones in the
  producer, the benchmark builders and the Immerkaer hff (it ignores alpha).
- `RESIDUAL_STD`: measure on the first produced shards (`scripts/flux_tools/flux_pretrain_residual_std.py`) per bucket; one table
  was enough for FLUX.2 (+3.5% between buckets), Wan needed per type tables. Also check the latent statistics
  (residual std vs target std, correlation with the upsampled source) as in autoresearch_3 round 0.
- Sampler (steps / shift / temperature) is VAE specific (Wan: 20 / 4 / 1.0; FLUX.2: 30 / 2 / 1.0): sweep on a
  checkpoint early (e.g. at ~8 h) with `scripts/flux_tools/flux_pretrain_full_benchmark.py` on a few hundred images.
- Benchmark set: autoresearch style 1000 Axis images encoded with the new VAE (oracle and pixel bicubic references
  first), plus the 30-image visual set for training.
- Encoder speed of the new VAE decides the GPU split (measure with a copy of `scripts/flux_tools/bench_flux_encoder.py`).

Order of work that worked (~1.5 h before the long run):
1. Download Axis (`hf download refine-axis/axis-v1_1k --repo-type dataset --local-dir datasets/axis-v1_1k`).
2. Producer smoke (`python -m ...producer --run-name <data> --encoder-gpus 7 --max-seconds 300`): throughput, queues
   (full queues = encoder bound), dataset and corruption shares in `producer_statistics.json`.
3. Residual table from the smoke cache; keep the smoke data run as the start of the real data run (rename it).
4. Trainer smoke on all training GPUs (`--max-steps 100 --no-upload`): units/s, memory, then a resume test.
5. Visual benchmark build + watcher on a smoke snapshot; check the references.
6. Launch in the background, errors-only watcher.

## 5. Measured numbers (FLUX.2, RTX 5090) for planning

| what | number |
|---|---|
| FLUX.2 VAE encode, compiled, bf16 | ~18.5 samples/s for target 1024 + source 512 + hff (one GPU) |
| pretrain producer, 1 encoder GPU | 19 samples/s; 1.53M samples in 24 h |
| 4KLSDB `train_x4` | 206 parquet files, 879 GB, hr = JPEG 6000x4000 (~1.2 MB, 0.11 s decode), HF download ~270 MB/s (~1 h per pass) |
| ReLAION-high-resolution online download | enough for 50% of 19 samples/s with 3 download processes |
| pretrain trainer (192M, 7 GPUs, micro batch 2 @1024) | 47-49 units/s, 27 GB peak, no gradient checkpointing |
| teacher sampling (30 steps, bf16, compiled, batch 8 @1024) | ~1 s per image unguided, x2 with CFG |
| full autoresearch_3 benchmark of the 30-step teacher | ~20 min on a free GPU (batch 8), ~22 min at batch 2 on the shared encoder GPU |
| 1-step benchmark of a student | ~2-3 min |
| large distillation trainer (192M, one GPU) | ~7.5 samples/s; 7 teacher generators ~5.3 pairs/s |
| large 1-step student as pair generator | ~30 pairs/s per GPU (batch 16 @1024, 8 GB) |
| small student training (20.2M, batch 8 @1024) | 42.7 samples/s per GPU, 22.8 GB (activation bound; batch 16 OOM), DDP near linear |
| 1-step latency 512 -> 1024 (model only, compiled, batch 1) | large 36.6 ms / 652 MB, small 5.9 ms / 130 MB |
| disk (FLUX.2 runs) | pretrain latent cache 251 GB (limit 250), small student pair cache 123 GB (limit 120), checkpoints ~3 GB each (pretrain, with optimizer) |

## 6. Recipe decisions and what to change next time

- Kept: 24 h fixed time schedule, batch ramp x1 -> x2 (15%) -> x4 (45%), linear decay over the last 30%, weight
  decay 0.05 (fixed the Wan weight norm growth), QK-norm + condition RMS norm, hff with 10% dropout to null, mixed
  buckets 90% 512 -> 1024 / 10% 256 -> 512, large crops, 20% input corruption with a JPEG quality spread.
- Change next time: keep the LR at the x2 level for the x4 batch step (the step raised the weight noise and slightly
  worsened fidelity metrics), consider a longer decay (40-50%; patch_fid was flat at constant LR and fell 22% in the
  decay), shorter or no EMA in the decay.
- hff matters much more for FLUX.2 than for Wan (8-10% velocity change for 0.1 vs 0.9): CFG needs small scales.
- Distillation: online pair generation next to training removes the separate pair phase (large student in ~35 min
  total). 20 min of training was still improving: try 30-40 min. lr 3e-4, warmup 50 steps, decay 30%.
- Small student: 4 + 4 GPU split, 6 h, still improving at the end; a longer run would help.

## 7. Artifacts

- Pretrain checkpoint: R2 `checkpoints/flux_pretrain_v1/checkpoint_000129590.pt` (use raw weights; `model_config`
  inside, incl. `residual_std`).
- Large 1-step student: R2 `checkpoints/flux_distill_v1/checkpoint_001083.pt` (`FluxStudentTransformer`, raw).
- Small 1-step student: R2 `checkpoints/flux_small_s20m/checkpoint_0109780.pt` (`FluxStudentTransformer` with
  `SMALL_MODEL` = width 256, depth 14, head dim 64, MLP 1024, condition 256, source encoder 192, same `residual_std`).
- Local only (not persistent across instances): latent cache `pretrain_runs/flux_data` (264 GB), pair caches,
  benchmark sets `benchmark_datasets/axis_v1_1k_flux2_2x`, FID statistics `benchmark_cache/`, results `raw_results/flux/`.
