# Experiment logs

## Setup notes (before round 1)

- Baseline `run_0000`: 96.5M parameter window-attention UNet on frames folded into the batch (5D latents, 16 channels), temporal attention (identity for single frame inputs), sinusoidal frame index embedding, conditioning = sigma + hff embeddings. Low resolution latents are resized trilinearly to the target size and concatenated to the noisy latents.
- Recipe from phase 1: lr 4e-4, linear decay over the last 30%, logit-normal sigmas with shift 1, Euler 20 steps at sampling shift 2, no EMA, no augmentation.
- Sampling weights of pair types in `run_0000`: image 512->1024 (4), video 17->33 frames at 512 (3), image 256->512 (1), video 17->33 frames at 1024 (1), other pair types have weight 0 (their shapes are supported).
- Batch size comes from a token budget: 16 x 128 x 128 target latent positions, capped at 64 samples.
- Compile: residual and transformer blocks are compiled, all training shapes are warmed up before the clock starts (about 160 s while sharing the GPU with dataset encoding).
- The benchmark set is currently taken from the training samples (1000 images, 100 videos), so metrics are optimistic. To be replaced with held-out data.

## Key findings (keep updated)

- (pre round 1) run_0000 pointed at the missing dataset `wan_1024_i10k_v2k`; fixed the path to
  `wan_1024_i19k_v9k` before its first run (only change to run_0000).
- (pre round 1) Causal VAE alignment, measured on video blocks: target latent frame j matches source
  frame ceil(j/2) best, and target frame 0 == source frame 0 EXACTLY (mse 0.0, same single pixel frame).
  Trilinear 5->9 resize blends neighbouring source frames (wrong). New runs use causal index alignment
  and copy frame 0 from the source for temporal-only upscaling.
- (after round 1) KEY: Wan latents are smooth. Latent std 0.72 (per channel 0.3-0.7, not 1); HR latents
  correlate 0.95 with the bilinear upsampled LR latents; the residual has std 0.22 for images (512->1024,
  per channel 0.13-0.34) and 0.33 for video 17->33 (causal index copy, frame 0 excluded). Flux.2 in phase 1
  had residual std 0.79. With unit noise on x the detail is buried at nearly all training sigmas; the flow
  loss is ~0.07 (vs ~0.6 on Flux). -> Round 2 tests residual parameterization (flow on residual / std).
- (after round 1) References (dev/reference_fids.py): image oracle fid_patch 1.30 / fid 0.58, pixel
  bicubic 8.21 / 2.37 (mse_lr 0.90e-3), latent bilinear 119.6; video oracle (tiny VAE) 3.98 / 2.71, causal
  index copy 5.69 / 4.71. ALL round 1 models are worse than pixel bicubic (images) and index copy (videos).
- (after round 2) Residual parameterization (flow on (x - aligned source) / per channel residual std)
  is the biggest lever on images: UNet 13.8 -> 10.3, transformer 11.6 -> 8.7 fid_patch (benchmark
  sampler); with 10 steps / shift 4 / initial noise temperature 0.8 run_0007 reaches 7.42 (< pixel
  bicubic 8.21). The window transformer beats the UNet at equal input path in both parameterizations.
- (after round 2) Sampling: few steps, high shift and noise temperature < 1 all help (low noise steps add
  grain). Temperature 0.8 on 0007: img 7.85 -> 7.42, vid 7.80 -> 7.05 (0.7: 7.62 / 6.82).
- (after round 2) Video: no model beats causal index copy (5.69). Best: x-space UNet with training shift
  1/3 (0006) 6.89 (6.64 at temperature 0.8); residual runs are grainier on video (visual check).
- (after round 3) Steps are the main lever at 30 min: 0007 -> 0008 (no grad ckpt, half batch, 4.3k -> 11k
  steps) img 7.42 -> 5.94 (same sampler), vid 7.05 -> 5.90. Benchmark pairs only (0009) is slightly better
  on both (5.84 / 5.63, first model to beat index copy on video). Causal temporal attention is free
  (0011 5.90 / 5.83 vs 0008) -> adopted. Training shift 0.5 in residual space is worse (6.39 / 6.16).
- (after round 3) Sampler optimum moves with training: better models want temperature closer to 1 on
  images (0009: T0.9 5.54, T0.8 5.84, T0.7 6.31) but lower on video (T0.7 5.55); 14 steps = 10 steps + .
  From round 4: per task temperature (spatial 0.9, temporal 0.7), 14 steps, shift 4.
- (after round 4) Still step/LR limited at 30 min: quarter token budget (0013, 26.9k steps) img 4.90, LR
  8e-4 (0015) img 5.04 / vid 5.44 vs base 0012 5.43 / 5.58; depth 12 (0014) worse at fixed time (8k steps).
- (after round 5) Width 192 (0018) = width 256 quality at 35% less latency (4.71 vs 4.72 img, 3.75 vs
  5.78 ms): capacity is not binding at 30 min. Token budget 2 x 128^2 (0017) 4.53 / 5.26, LR 1.2e-3 (0019)
  4.64 / 5.34 vs 0016 4.72 / 5.43: still small gains from more steps / higher LR.
- (after round 6) TRAINING SEED NOISE IS LARGE: 0017 (seed 0) 4.53 / 5.26 vs 0022 (seed 1) 4.82 / 5.37 img /
  vid fid_patch. Single-run differences < ~0.3 img are not significant; rounds 4-5 knob gains (0.1-0.2)
  are within noise individually (the cumulative trend 5.43 -> ~4.7 is real). Use paired loss (same seed)
  and seed replicates for decisions.
- (after round 7) Noise: same SEED runs also differ ~0.1 img fid_patch (time based stop + kernel
  nondeterminism: 0020 4.80 vs 0024 raw weights 4.69); different seeds 0.1-0.3. Seed means: 0020 family
  4.84 / 5.33, 0023 (source injection every block) 4.87 / 5.31 -> injection lowers loss per step on both
  seeds (0.358 vs 0.38) but gives no fid gain at 30 min (SCALE+ candidate only).
- (after round 7) EMA is neutral (paired in-run: EMA 4.67 / 5.31 vs raw 4.69 / 5.25). LR decay over 50%
  4.66 / 5.26 (within noise, kept). Fused AdamW +6% steps.
- (after round 8) USER DECISION (2026-11-15): round 9 is a 2 hour scaling round (TRAINING_TIME_LIMIT=135m,
  as in phase 1), candidates chosen to show scaling laws. Question about computing the LR encoder once per
  upscale (outside single_forward) not answered yet: keep it inside the per step model call.
- (after round 8) Paired per-step loss (batch 2, pre-decay @15k): w128 0.451, w192 0.428, w256 0.416; source
  injection 0.429 (= w192, no gain; its lower final loss in round 7 was a logging artifact -> RETIRED);
  U-transformer 0.434 but 10% faster steps (neutral per time) and 15% lower video latency.
- (after round 9, 2 h) 4x time: w192 4.70 -> 4.03, w256 4.53 -> 3.84 img fid_patch. Width at 2 h: 192 / 256 / 384
  = 4.03 / 3.84 / 3.76 img, 5.06 / 4.96 / 4.92 vid (benchmark sampler). Per step w384 < w256 < w192 always;
  per time w192 leads at 15 min, w384 leads from ~30 min on (compute optimal width grows with budget).
  Batch 8 x 128^2 is worse than 2 x 128^2 per time AND per sample (LR not retuned). Sampler optimum moves
  again: 2 h models want temperature 1.0 (spatial) or 20 steps (img -0.15 to -0.2).
- (after round 9) USER: do not start round 10 until the user's questions are answered.
- USER (after round 9): single step distillation of the many-step model works (CONTEXT.md,
  reports/rectified_flow_distillation_vs_sft.md, reports/small_single_step_distillation.md). Plan: a base
  model (max quality, many steps) + a distilled single step model. Caching the LR path outside
  single_forward is low priority (only relevant to the base model, whose goal is quality, not speed).
  User asked for temporal locality tests: can temporal attention be replaced by a small look-back
  (previous frame) convolution without quality loss (inference optimization headroom)?
- USER (after round 9): continue with 30 min rounds; NEW SEED DIRECTIONS: (1) temporal convolutions instead of
  temporal attention (locality, inference headroom), (2) RoPE on window attention, (3) batch size scaling
  rules for an 8 GPU training machine (optimal global batch and LR schedule), extrapolated from 30 min /
  2 h results and the phase 1 full pretraining. Then free exploration.
- (after round 10) TEMPORAL LOCALITY WORKS: causal gated depthwise temporal conv (kernel 2, previous frame) instead
  of temporal attention (0036): img 4.67 / vid 5.23 vs 0028 4.70 / 5.28, paired video loss per step LOWER
  (0.340 vs 0.349 @20k), image loss identical, video latency -16% (9.1 vs 10.8 ms) -> ADOPTED (base 0036).
- (after round 10) RoPE (axial 2D, base 100, window coords) = relative position bias exactly (paired loss 0.433 vs
  0.432 @20k; fid 4.73 / 5.26): neutral. Either is fine.
- (after round 10) Batch: global 16 images is much worse per sample than 2 at the 30 min stage (paired per sample
  loss @48k samples: B2 0.418, B16 LR 1.2e-3 0.474, B16 LR 3.4e-3 0.458); fid 5.41 / 5.09 vs 4.70. sqrt LR
  recovers ~half. Gradient noise scale of checkpoints (dev/gradient_noise_scale.py) is 5-17 images pre-decay,
  but confounded (edge-of-stability bouncing at high LR, Adam preconditioning ignored): magnitude only.
- (after round 11) BATCH: with sqrt LR scaling (LR = 1.2e-3 x sqrt(B / 2)) batch 4 / 8 images are free per sample
  (fid 4.59 / 4.67 vs 4.67 at B2) and B16 with a 1000 step warmup reaches 4.80 (within seed noise of B2;
  100 step warmup: 5.09). The round 10 "large batch is bad" was mostly a too short warmup at high LR.
  B16 at 2x sqrt LR (6.8e-3, short warmup) is too high (5.32). Larger batches lose early (per sample loss)
  and catch up later (critical batch grows as loss falls), annealed fid nearly equal.
- (after round 12) LOSS SPIKES (user question) = ATTENTION LOGIT GROWTH. Runs at LR >= 4.8e-3 (0042, 0045) have
  spikes of 100-1000x (100 step means). dev/activation_stats.py: spiking 0042 has max window attention logits
  842 -> 1424 (block 0) and 574 -> 807 (block 6) with q/k RMS 5-9; stable runs 10-30 with q/k RMS 0.5-1.
  Residual stream 2x larger in the spiking run. adaLN scales moderate everywhere (<= 3). Fix candidate: QK-norm.
- (after round 12) DEAD ATTENTION: in every stable model (incl. the 2 h 0033) the window attention of blocks 2
  and 3 is dead (logits ~0, q/k RMS 0.02-0.07 = uniform averaging).
- (after round 12) Warmup length only matters at large batch / high LR (B2 wu 8000 = wu 100; B8 wu 2000 4.61 vs
  wu 100 4.67). B32 at LR 4.8e-3 (sqrt) is clearly worse at the 30 min sample budget (5.33): critical batch at
  this stage is 8-16 images.
- (after round 13) QK-norm does NOT remove the spikes (0049 13 spikes, grad norm max 5.8e5; 0050 9, 3.7e6) although
  it bounds the attention logits (<= 16) and REVIVES the dead blocks 2-3 (logits 3-9). Quality neutral at
  stable LR (B2 4.66 / 5.20 vs 4.67 / 5.23; B16 3.4e-3 4.89 vs 4.80), +0.2 ms. Spikes hit image and video
  losses in the same window (global optimizer event); the stream entering block 0 (stem + LR encoder, no
  normalization) doubles between checkpoints. -> dev/spike_probe.py: per step per module grad / weight /
  update norms to localize the trigger.
- (after round 13) SPIKE ROOT CAUSE (dev/spike_probe.py, per step per module norms + per sample inputs): weights are
  normal before a spike (update ratios 1-2x); ONE sample of the batch has loss 1e4-6e6 (median 0.7) and its sigma
  lies in a very narrow band (0045: 0.8989 / 0.8993 / 0.8997 / 0.8998; 0050 s2: 0.5174 / 0.5185); hff spread;
  gradient jump largest in `conditioning`. Cause: sinusoidal(sigma * 1000) has a 0.006 period in sigma, the
  unnormalized condition MLP learns a sharp response at high LR that blows up the adaLN modulation in narrow
  sigma windows (attention logit growth was downstream). Fix probes (B16 LR 6.8e-3, 10 min constant LR): RMS
  norm of the condition vector -> 0 spikes (max grad 6.6 vs 1e6) but median loss +4% (0.542 vs 0.522 same
  seed); embedding scale 100 -> 1 / 0 spikes (vs 9 / 6), no loss cost; both -> 0 spikes. Round 14 tests them.
- USER (after round 13): consider training on fewer GPUs; the real pretrain model is ~0.5B (smaller per GPU
  batch). Answer given: outcome depends on global batch / LR / samples only (accumulation is equivalent);
  B_crit ~8-16 images at the 30 min stage and grows during training -> batch ramp; LR must be retuned for
  the 0.5B width (phase 1 451M used 4e-4); PCIe allreduce favours accumulation. Proposed: 2 h batch 4 vs 16
  (B_crit growth) and a 30 min batch ramp run, after the spike fix.
- (after round 14) SPIKE FIX ADOPTED: RMS norm of the condition vector (CONDITION_NORM) removes all spikes (max grad
  norm <= 8.4 at LR 4.8e-3 / 6.8e-3, was up to 3.7e6) at zero quality cost at normal LR (0052 4.66 / 5.21 vs 0036
  4.67 / 5.23). It does NOT raise the useful LR ceiling: B16 at 6.8e-3 is stable but worse (5.18). Embedding scale
  100 only partly fixes (2 spikes, max grad 1.3e4). Base from round 15: 0052.
- USER (after round 14): test Attention Residuals (Kimi Team, arXiv 2603.15031) against the residual stream
  growth before the 2 h batch experiments. Paper: h_l = sum_i softmax_i(w_l . RMSNorm(v_i)) v_i over the embedding
  and earlier sublayer outputs (Full) or block sums (Block), zero init pseudo query per sublayer; 16 layer model
  loss 1.766 -> 1.737 full / 1.746 block, 1.25x compute equivalent, bounded magnitudes.
- (round 15 prep) AttnRes in our model (dev/drafts/attnres_patch.py): sublayers = spatial attn, temporal conv (video
  only), FFN; gated adaLN-zero outputs as values; per token (= per position) weights; output head also aggregated.
  Cost is memory traffic (2 reads of every source per sublayer at D = 192): naive Block -42% steps, Full -61%;
  optimized (normalize each source once, RMSNorm gain folded into the query, no stack of values) Block -32% steps,
  lat 4.31 / 10.4 vs 3.27 / 8.56 ms; Full -59% steps and 372 s cold compile -> Full dropped. Relative overhead
  should shrink with width (AttnRes ~ n D per token vs sublayers ~ D^2).
- USER DECISION (during round 15): DROP AttnRes regardless of a small fid gain. Design constraint for inference on
  low-end hardware: keep as few temporary full resolution buffers as possible (AttnRes keeps up to depth + 1 extra
  stream copies). User observation: AttnRes trains faster per step but no large benefit. Next: find where the
  inference memory peaks of the base come from and cut them (op reorder / fusion).
- (after round 15) AttnRes: paired loss per step -3.5% on both seeds (0.391 / 0.396 vs 0.406 / 0.410 @30k) but
  fixed time fid worse or equal (block-8 4.96 / 5.05 vs 4.66 / 4.71; block-4 4.61) and memory +50-70%. DROPPED
  (user decision: temporary buffers).
- (after round 15) INFERENCE MEMORY (dev/memory_breakdown.py): 75% of the base's forward peak was the window attention
  SDPA call: 5D q/k/v ([N, heads, windows, t, Dh]) with a float bias fall back to the MATH path (materialized
  scores). Folding windows into the batch dim (4D) uses the fused memory-efficient kernel: image peak +172.5 ->
  +89.5 MB, video +339 -> +203 MB, latency 3.33 -> 3.09 / 8.68 -> 8.05 ms, same quality on the 0052 weights
  (4.659 / 5.203 vs 4.658 / 5.205), training throughput equal, train memory 4.2 -> 4.0 GB. -> run_0060 = new base
  (pure implementation change). Remaining peak = FFN hidden (D -> 2F = 1024 channels input projection + depthwise
  conv output live together, 2 x 32 MB at 1024). Exact row tiling of the FFN: peak only -30% (attention buffers
  then dominate) and latency +47% -> not worth it.
- USER PLAN (after round 15): round 16 = 30 min, all memory / latency optimizations that need architecture changes;
  round 17 = 2 h, learning dynamics of batch sizes and LRs + ONE slot for the production solution (its 1 GPU version
  with gradient accumulation at the production effective batch and LR schedule) to check scaling. Final production
  run = round 17 batch / LR schedule + a tested ~0.5B model.
- (after round 16) Architecture memory / latency options (paired loss @20k vs 0052 0.432): RoPE + unshifted windows 0.435
  (fid 4.76 / 5.27, lat -10%), SwiGLU 384 0.436 (4.81 / 5.30, mem -14%), GELU MLP 4D 0.431 (4.81 / 5.33, mem -8%, equal
  per step: candidate for a memory friendly production model), lean combination 0.441 (4.94 / 5.32, lat 3.13 / 7.19
  ms, mem 102 / 199 MB): a real quality-for-latency trade (Pareto point), not free. 4D attention layout (0060) is the
  only free win. Benchmark memory incl. weights, 4D layout: ~120 / 235 MB vs 203 / 373 for the 5D layout.
- (round 17, first launch) LONG HORIZON ATTENTION LOGIT GROWTH: 0065 (B4, LR 1.7e-3, no QK-norm) degraded from ~35k steps
  (loss 0.40 -> 0.5-0.95, mean grad norm 0.15 -> 1.2, max 1681). Checkpoint @42.7k: max window attention logits 1.8e5
  (block 0) and 1.6e7 (block 6), q/k RMS 100-1100, residual stream RMS up to 1084 (healthy ~20-75), adaLN scales up to
  28. Already starting in healthy 30 min models (0052 final: block 7 logits 1865, q/k RMS 7-10). Separate from the
  narrow-sigma spikes (fixed by the condition norm); QK-norm is the fix. PRODUCTION NEEDS BOTH condition RMS norm and
  QK-norm (and possibly stronger weight decay than AdamW's 0.01).
- USER DECISION (round 17): relaunch all runs with every kept improvement (incl. QK-norm) on one identical, larger
  model; only batch and LR differ; record the stopped half runs without benchmark.
- (after round 17) BATCH / LR DYNAMICS (2 h, 45M w384, all stabilizers): all runs stable (QK-norm + condition norm fix both
  instabilities; B4 at LR 1.7e-3 trained 68k steps cleanly). Critical batch GROWS with training: per sample loss B16 vs B4
  +0.016 @30k samples, equal @90k, B16 ahead @120k; B64 gap 0.073 -> 0.010. Final fid: B4 3.77, B16 3.98, ramp 16->64 3.98,
  B64 4.10 (img); video 4.95-5.06. Ramp == constant B16 per sample. Starting large (B64) costs. See PRODUCTION
  RECOMMENDATION at the end.
- (pre round 1) Compile time: run_0000 cold inductor cache 304 s (would TIME OUT at the 35 min guard),
  warm cache 52 s. ALWAYS profile new runs with dev/profile_run.py before a round (warms the cache).

## Dev tools (dev/)

| Tool | Purpose |
|---|---|
| run_tools.py | load a run file with NAME=EXPR overrides and redirected outputs |
| profile_run.py RUN | `--train-seconds N` real compile + training into dev/temporary, params, train peak mem, eager upscale mem, compiled latency (back to back wall). `--train-seconds 0` = smoke. Warms the inductor cache |
| evaluate_quality.py RUN | benchmark quality with overrides (steps, shift), `--weights-directory`, `--model-file` (checkpoints), `--samples`, results appended to dev/evaluations/TAG.jsonl |
| summarize_results.py | markdown table of raw_results + train logs |

## Round 1

Profiles (90 s training on a shared box; steps/s from steps 100-200; latency compiled, wall, ms):

| run | params | steps/s | train mem | lat img 1024 | lat vid 512 | loss@200 (paired) |
|---|---|---|---|---|---|---|
| 0000 | 96.5M | ~1.7 | 8.8 GB (grad ckpt) | 9.47 | 22.2 | - |
| 0001 | 42.3M | ~4.5 | 15.4 GB | 4.03 | 9.92 | 0.965 |
| 0002 | 47.4M | ~4.0 | 16.3 GB | 4.53 | 11.1 | 1.018 (0.46 @300) |
| 0003 | 19.3M | ~2.5 | 10.6 GB (grad ckpt) | 4.85 | 14.3 | 0.276 |

Hypotheses:
1. Baseline 0000 is too heavy for 30 min (~3k steps): phase 1 showed fixed-time budgets favour fast
   conv UNets with capacity at full resolution. 0001 transfers that recipe (and fixes temporal alignment).
2. Learned source path (0002, seed Q3): LR encoder + pixel shuffle + causal temporal phases beats
   bilinear/index upsampling of latents alone.
3. Plain window transformer (0003, seed Q4/Q5): no UNet, SwiGLU, adaLN-zero, shifted windows. Early
   paired loss is much lower; question is whether it holds at fixed time with fewer steps.
Sampler (steps/shift, seed Q1) will be swept post hoc on all checkpoints; training LR / shift next round.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0000 | Baseline (96.5M window-attention UNet, transformer blocks at every level) | - | reference |
| 0001 | Phase 1 fast UNet (256,256,512) dec (4,1,1), window attn L1/L2/mid, temporal attn all levels, causal alignment + frame 0 copy | 0000, phase 1 0064 | recipe transfer, speed vs capacity |
| 0002 | 0001 + learned source path (LR res encoder, pixel shuffle, causal temporal phases) | 0001 | seed Q3 |
| 0003 | Plain window transformer, 8 blocks x 256, SwiGLU + dwconv FFN, adaLN-zero, shifted windows | 0001 | seed Q4/Q5 |

Launched: ./wan_infra/start_round.sh 0000 0001 0002 0003

### Round 1 results (benchmark: 20 steps, sampling shift 2)

| run | img fid_patch | img fid | img mse_lr e3 | vid fid_patch | vid fid | lat img / vid ms | params | steps | loss@3k (paired, pre-decay) |
|---|---|---|---|---|---|---|---|---|---|
| 0000 baseline | 49.1 | 27.5 | 5.10 | 13.9 | 14.1 | 10.3 / 23.5 | 96.5M | 3.2k | - |
| 0001 fast UNet | 29.3 | 15.6 | 3.53 | 9.85 | 9.64 | 4.56 / 10.8 | 42.3M | 7.9k | 0.129 |
| 0002 + learned source path | 15.3 | 6.95 | 1.79 | 8.27 | 7.82 | 5.11 / 11.6 | 47.4M | 7.3k | 0.107 |
| 0003 plain window transformer | 20.4 | 13.2 | 3.13 | 8.63 | 7.99 | 5.17 / 14.9 | 19.3M | 4.5k | 0.098 |

- Baseline is far too slow for 30 min (3.2k steps); the phase 1 style fast UNet is better on every metric.
- Learned source path (seed Q3): biggest single gain, image fid_patch 29.3 -> 15.3, mse_lr halved.
  Directly upsampling latents is NOT ok; LR information handling is the bottleneck.
- Plain transformer (seed Q4): lowest paired loss per step (0.098 vs 0.129 for the UNet with the same
  input path) and 20.4 vs 29.3 fid_patch at fixed time with 43% fewer steps. Strong candidate.
- Sampler sweep on 0002 (dev/evaluations/r1_sweep.jsonl, image fid_patch): 10 steps shift 1/2/3 =
  14.4/13.8/13.8; 20 steps shift 1/2/3/4 = 17.8/15.3/14.3/13.9; 30 steps shift 2 16.9. 0003: 10/2 20.0,
  20/3 19.9; 0001 10/2 28.5. Fewer steps / higher shift is better, but the gain is small (~1.5).
- Visual: all models garble LR content (text) -> detail/structure generation, not color drift.

## Round 2

2x2 factorial {UNet, transformer} x {x-space, residual parameterization}, all with the learned source path,
plus a schedule-only control. Generator: dev/drafts/make_round_02.py. Drafts kept for later:
dev/drafts/multilevel_source_injection.py (0002 + source features at every level, 4 encoder blocks),
dev/drafts/benchmark_pairs_only.py (seed Q2 data mix).

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0004 | Transformer + learned source path (10 steps, shift 3) | 0002, 0003 | combine round 1 winners |
| 0005 | UNet 0002 + residual parameterization (10 steps, shift 2) | 0002 | residual flow, UNet |
| 0006 | UNet 0002 + training shift 1/3 (10 steps, shift 3) | 0002 | schedule-only alternative to residual |
| 0007 | Transformer 0004 + residual parameterization (10 steps, shift 2) | 0004, 0005 | residual flow, transformer |

Launched: ./wan_infra/start_round.sh 0004 0005 0006 0007 (profiles: 0004/0007 24.3M, ~2.4 st/s, lat 5.4/16.1 ms; 0005/0006 47.4M ~4 st/s, 4.55/11.0 ms)

### Round 2 results (benchmark sampler as listed per run; paired losses not comparable across parameterizations)

| run | img fid_patch | img fid | img mse_lr e3 | vid fid_patch | vid fid | lat img / vid ms | steps |
|---|---|---|---|---|---|---|---|
| 0004 transformer + source, x-space (10/3) | 11.56 | 6.42 | 1.44 | 7.29 | 6.74 | 5.80 / 16.6 | 4.3k |
| 0005 UNet + source, residual (10/2) | 10.29 | 4.78 | 1.66 | 8.08 | 7.37 | 5.11 / 11.7 | 7.2k |
| 0006 UNet + source, train shift 1/3 (10/3) | 13.12 | 6.23 | 1.53 | 6.89 | 6.23 | 5.25 / 11.6 | 7.3k |
| 0007 transformer + source, residual (10/2) | 8.68 | 4.58 | 1.60 | 8.43 | 7.62 | 5.80 / 16.7 | 4.3k |

Sampler sweeps (dev/evaluations/r2_sweep.jsonl, r2_temp.jsonl), img / vid fid_patch:
- 0007: 5/2 7.88/7.95, 10/1 9.55/8.97, 10/2 8.68/8.43, 10/4 7.85/7.80, 20/2 9.79/8.91;
  10/4 + temperature 0.9 7.46/7.36, 0.8 7.42/7.05, 0.7 7.62/6.82.
- 0005 5/2 9.12/7.50; 0004 5/3 12.54/7.09; 0006 10/3 temperature 0.8 14.67/6.64 (x-space: temperature
  hurts images).
Conclusions:
- Residual + transformer stack (11.6 -> 8.7 and 13.8 -> 10.3: both ~-3). Factorial is clean.
- Training shift 1/3 in x-space (0006 vs 0002 @10/3 13.8): small image gain (13.1), best video.
- Throughput: transformer 2.4 steps/s at batch 16 with gradient checkpointing; without checkpointing at
  half the token budget 6.0 steps/s (25% more samples/s, 2.5x steps), 21.8 GB.

## Round 3 (generator dev/drafts/make_round_03.py)

Base 0008 = 0007 + no gradient checkpointing + token budget 8 x 128 x 128 + sampler 10 / shift 4 /
temperature 0.8 (INITIAL_NOISE_SCALE). Expected img fid_patch from the sampler alone ~7.4.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0008 | 0007 faster training (no ckpt, half batch) + sampler 10/4/T0.8 | 0007 | new base; more steps |
| 0009 | 0008 benchmark pairs only (img 512->1024 5, vid 17->33@512 4) | 0008 | seed Q2 data mix |
| 0010 | 0008 training shift 0.5 | 0008, 0006 | low noise training in residual space (video) |
| 0011 | 0008 causal temporal attention | 0008 | seed Q6 causality |

Launched: ./wan_infra/start_round.sh 0008 0009 0010 0011

### Round 3 results (sampler 10 / shift 4 / T0.8)

| run | img fid_patch | img fid | img mse_lr e3 | vid fid_patch | vid fid | lat img / vid ms | steps |
|---|---|---|---|---|---|---|---|
| 0008 base (no ckpt, half batch) | 5.94 | 3.69 | 1.22 | 5.90 | 5.10 | 5.83 / 16.6 | 11.0k |
| 0009 benchmark pairs only | 5.84 | 3.65 | 1.20 | 5.63 | 4.78 | 5.78 / 16.6 | 11.5k |
| 0010 training shift 0.5 | 6.39 | 3.75 | 1.23 | 6.16 | 5.32 | 5.75 / 16.4 | 11.1k |
| 0011 causal temporal attention | 5.90 | 3.62 | 1.22 | 5.83 | 5.07 | 5.74 / 16.8 | 11.1k |

Sampler sweep 0009 (dev/evaluations/r3_sweep.jsonl) img / vid: 10/4/T0.9 5.54/5.76, 10/4/T0.7 6.31/5.55,
10/3/T0.8 5.62/5.71, 14/4/T0.8 5.54/5.68.
Seed answers so far: Q2 the auxiliary pairs (256->512 images, 1024 videos) do not help at 30 min (small
negative); Q6 causal temporal attention costs nothing; Q1 training shift 1 stays best (0.5 and 1/3 worse in
residual space), sampler needs few steps, shift 3-4 and temperature < 1.

## Round 4 (generator dev/drafts/make_round_04.py)

Base 0012 = 0009 + causal temporal attention + sampler 14 steps / shift 4 / temperature 0.9 spatial, 0.7
temporal. Profiles: 0012 lat 5.37/16.1 ms, 16.4 GB; 0013 2.3x steps/s, 7.8 GB; 0014 33.3M, 7.69/23.4 ms,
0.70x steps/s.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0012 | 0009 + causal temporal attention + per task temperature, 14 steps | 0009, 0011 | new base |
| 0013 | 0012 with token budget 4 x 128 x 128 | 0012 | more steps (batch size at fixed time) |
| 0014 | 0012 with 12 blocks | 0012 | capacity vs steps |
| 0015 | 0012 with LR 8e-4 | 0012 | seed Q1 LR |

Launched: ./wan_infra/start_round.sh 0012 0013 0014 0015

### Round 4 results (sampler 14 / shift 4 / T 0.9 spatial, 0.7 temporal)

| run | img fid_patch | img fid | img mse_lr e3 | vid fid_patch | vid fid | lat img / vid ms | steps |
|---|---|---|---|---|---|---|---|
| 0012 base | 5.43 | 3.61 | 1.31 | 5.58 | 4.78 | 5.81 / 16.7 | 11.5k |
| 0013 token budget 4 x 128^2 | 4.90 | 3.30 | 1.26 | 5.47 | 4.67 | 5.77 / 16.5 | 26.9k |
| 0014 depth 12 | 5.62 | 3.64 | 1.35 | 5.89 | 5.10 | 8.29 / 23.9 | 8.0k |
| 0015 LR 8e-4 | 5.04 | 3.34 | 1.27 | 5.44 | 4.64 | 5.80 / 16.6 | 11.5k |

0012 vs 0009 (5.84 / 5.63): the new sampler (14 steps, per task temperature) gives -0.41 on images.

## Round 5 (generator dev/drafts/make_round_05.py)

Seed Q5 small experiment: parallel block (fused attention + FFN input projection, dev/drafts/parallel_block_0016.py)
profiles at 5.76 / 16.7 ms vs 5.38 / 16.1 for 0016: NO latency gain on this GPU (window partition / reshape
and the depthwise conv dominate, not the linears), so not trained.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0016 | 0013 + LR 8e-4 (token budget 4 x 128^2) | 0013, 0015 | new base |
| 0017 | 0016 token budget 2 x 128^2 | 0016 | batch size floor |
| 0018 | 0016 width 192 (15.9M, 3.26 / 10.3 ms) | 0016 | low latency end |
| 0019 | 0016 LR 1.2e-3 | 0016 | LR scan |

Launched: ./wan_infra/start_round.sh 0016 0017 0018 0019

### Round 5 results (sampler 14 / shift 4 / T 0.9, 0.7)

| run | img fid_patch | img fid | vid fid_patch | vid fid | lat img / vid ms | params | steps |
|---|---|---|---|---|---|---|---|
| 0016 budget 4 x 128^2 + LR 8e-4 | 4.72 | 3.19 | 5.43 | 4.63 | 5.78 / 16.9 | 24.3M | 26.9k |
| 0017 budget 2 x 128^2 | 4.53 | 3.18 | 5.26 | 4.41 | 5.75 / 16.7 | 24.3M | 38.2k |
| 0018 width 192 | 4.71 | 3.21 | 5.42 | 4.62 | 3.75 / 10.7 | 15.9M | 40.5k |
| 0019 LR 1.2e-3 | 4.64 | 3.14 | 5.34 | 4.50 | 5.77 / 16.6 | 24.3M | 26.9k |

Pareto (lat img ms, img fid_patch): 0018 (3.75, 4.71), 0017 (5.75, 4.53).

## Round 6 (generator dev/drafts/make_round_06.py)

Profiles: at budget 2 x 128^2 training is partly launch bound (width 128 only 1.18x the steps of width 192).

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0020 | width 192 + budget 2 x 128^2 + LR 1.2e-3 (15.9M, 3.31 / 10.4 ms) | 0017, 0018, 0019 | fast stack, new base candidate |
| 0021 | 0020 width 128 (9.2M, 2.32 / 6.18 ms) | 0020 | low latency end |
| 0022 | 0017 SEED 1 | 0017 | training seed noise |
| 0023 | 0020 + source features injected before every block (3.51 / 10.3 ms) | 0020 | seed Q3 stronger source conditioning |

Launched: ./wan_infra/start_round.sh 0020 0021 0022 0023

### Round 6 results (sampler 14 / shift 4 / T 0.9, 0.7)

| run | img fid_patch | img fid | vid fid_patch | vid fid | lat img / vid ms | params | steps | final loss |
|---|---|---|---|---|---|---|---|---|
| 0020 fast stack (w192, 2 x 128^2, LR 1.2e-3) | 4.80 | 3.17 | 5.35 | 4.49 | 3.87 / 10.9 | 15.9M | 48.7k | 0.378 |
| 0021 width 128 | 5.33 | 3.49 | 5.54 | 4.74 | 2.73 / 6.75 | 9.2M | 62.2k | 0.378 |
| 0022 = 0017 SEED 1 | 4.82 | 3.15 | 5.37 | 4.49 | 5.66 / 16.7 | 24.3M | 38.6k | 0.367 |
| 0023 0020 + source injection every block | 4.83 | 3.21 | 5.34 | 4.44 | 3.96 / 10.8 | 16.2M | 49.6k | 0.358 |

- 0023 vs 0020 is paired (same seed): final loss 0.358 vs 0.378 (better per step, SCALE+ candidate) but
  equal fid_patch. Replicated in round 7.
- Pareto (lat img, img fid_patch): 0021 (2.73, 5.33), 0018 (3.75, 4.71), 0017/0022 (5.7, 4.53-4.82).

## Round 7 (generator dev/drafts/make_round_07.py)

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0024 | 0020 + EMA 0.9995 (raw weights kept as model_raw.safetensors) | 0020 | weight noise at batch 2 / high LR |
| 0025 | 0020 LR decay over last 50% | 0020 | annealing length |
| 0026 | 0020 SEED 1 | 0020 | replicate |
| 0027 | 0023 SEED 1 | 0023 | replicate source injection |

Launched: ./wan_infra/start_round.sh 0024 0025 0026 0027

### Round 7 results

| run | img fid_patch | img fid | vid fid_patch | vid fid | lat img / vid ms | steps | final loss |
|---|---|---|---|---|---|---|---|
| 0024 0020 + EMA 0.9995 | 4.67 (raw 4.69) | 3.19 | 5.31 (raw 5.25) | 4.48 | 3.71 / 11.1 | 50.7k | 0.358 |
| 0025 0020 decay 50% | 4.66 | 3.16 | 5.26 | 4.40 | 3.69 / 11.2 | 54.3k | 0.381 |
| 0026 0020 SEED 1 | 4.89 | 3.21 | 5.32 | 4.47 | 3.84 / 10.8 | 51.7k | 0.387 |
| 0027 0023 SEED 1 | 4.92 | 3.19 | 5.29 | 4.48 | 3.92 / 10.8 | 50.1k | 0.357 |

## Round 8 (generator dev/drafts/make_round_08.py)

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0028 | 0020 + decay 50% + fused AdamW (3.33 / 10.4 ms) | 0020, 0025 | base |
| 0029 | 0028 image : video 1 : 1 | 0028 | task trade-off |
| 0030 | 0028 U-shaped transformer: 2 full res (192) / 4 half res (384) / 2 full res, 27.3M (3.35 / 8.81 ms) | 0028 | seed Q4 multi-scale |
| 0031 | 0028 width 256 (5.35 / 16.0 ms) | 0028, 0017 | quality end |

Launched: ./wan_infra/start_round.sh 0028 0029 0030 0031

### Round 8 results

| run | img fid_patch | img fid | vid fid_patch | vid fid | lat img / vid ms | params | steps |
|---|---|---|---|---|---|---|---|
| 0028 base (w192, decay 50%, fused AdamW) | 4.70 | 3.17 | 5.28 | 4.43 | 3.69 / 10.8 | 15.9M | 50.7k |
| 0029 image : video 1 : 1 | 4.81 | 3.25 | 5.27 | 4.39 | 3.67 / 10.9 | 15.9M | 49.3k |
| 0030 U-shaped transformer (2/4/2) | 4.71 | 3.09 | 5.30 | 4.52 | 3.75 / 9.22 | 27.3M | 55.7k |
| 0031 width 256 | 4.53 | 3.08 | 5.21 | 4.34 | 5.68 / 16.5 | 24.3M | 39.0k |

- 1 : 1 task ratio: no video gain, images slightly worse -> keep 5 : 4.
- U-transformer: same quality, lower video latency (Pareto point for video).
- Width 256 again 4.53 (same as 0017): best at 30 min, but within ~2 sigma of w192.

30 min Pareto (lat img ms, img fid_patch / vid fid_patch): 0021 w128 (2.73, 5.33 / 5.54), 0028 w192 (3.69, 4.70 / 5.28),
0030 U-transformer (3.75, 4.71 / 5.30, video 9.2 ms), 0031 w256 (5.68, 4.53 / 5.21).

## Round 9: 2 hour scaling round (generator dev/drafts/make_round_09.py)

All runs = run_0028 recipe, TRAINING_TIME_SECONDS 2 h, LR decay over the last 50% (from 60 min),
checkpoints at 30 / 60 / 90 min (30, 60 pre-decay). LR 1.2e-3 everywhere (clean width axis; w384 is stable
in a 150 s profile). Profiled ms/step -> expected steps in 2 h:

| Run ID | Run name | params | ms/step | ~steps | lat img / vid ms | 30 min reference | Run purpose |
|---|---|---|---|---|---|---|---|
| 0032 | width 384, budget 2 x 128^2 | 46.2M | 76 | 95k | 8.47 / 28.2 | - | expected best (width scaling top) |
| 0033 | width 192 (= 0028) | 15.9M | 36 | 200k | 3.29 / 10.3 | 0028: 4.70 / 5.28 | time scaling of the 30 min winner |
| 0034 | width 256 | 24.3M | 46 | 155k | 5.32 / 15.9 | 0031: 4.53 / 5.21 | width scaling middle |
| 0035 | width 256, budget 8 x 128^2 (4x batch) | 24.3M | 155 | 46k | 5.36 / 16.1 | 0016-like | batch size at larger compute |

Analysis plan: 2 h vs 30 min per architecture; width axis 192 / 256 / 384 at 2 h; batch 2 vs 8 at equal time
(0034 vs 0035) and per sample; checkpoint fid_patch at 30/60/90 min (shape only; pre-decay); sampler
re-sweep on the 2 h models (steps / temperature optimum moves with training).

Launched: TRAINING_TIME_LIMIT=135m ./wan_infra/start_round.sh 0032 0033 0034 0035

### Round 9 results (2 h; benchmark sampler 14 steps / shift 4 / T 0.9, 0.7)

| run | img fid_patch | img fid | img mse_lr e3 | vid fid_patch | vid fid | lat img / vid ms | params | steps | 30 min same arch |
|---|---|---|---|---|---|---|---|---|---|
| 0032 w384 | 3.76 | 2.59 | 1.07 | 4.92 | 3.89 | 9.22 / 28.9 | 46.2M | 96.3k | - |
| 0033 w192 | 4.03 | 2.82 | 1.15 | 5.06 | 4.09 | 3.66 / 11.1 | 15.9M | 195.9k | 0028: 4.70 / 5.28 |
| 0034 w256 | 3.84 | 2.67 | 1.11 | 4.96 | 3.98 | 5.74 / 16.4 | 24.3M | 156.6k | 0031: 4.53 / 5.21 |
| 0035 w256 batch 8 x 128^2 | 4.02 | 2.74 | 1.13 | 5.02 | 4.08 | 5.90 / 16.5 | 24.3M | 46.2k | - |

Paired losses (same seed and recipe; pre-decay):
- per step @10k / 20k / 40k: w192 0.467 / 0.432 / 0.402, w256 0.455 / 0.419 / 0.389, w384 0.439 / 0.405 / 0.374.
- per time @900 / 1800 / 3000 / 3500 s: w192 0.419 / 0.391 / 0.381 / 0.378, w256 0.422 / 0.391 / 0.378 / 0.364,
  w384 0.426 / 0.387 / 0.375 / 0.360, w256 batch 8 0.435 / 0.400 / 0.382 / 0.377.
- equal samples: batch 8 at 5k / 10k steps 0.448 / 0.405 vs batch 2 at 20k / 40k 0.419 / 0.389.
Sampler sweep (dev/evaluations/r9_sweep.jsonl) img / vid fid_patch: 0032 T1.0/0.8 3.55 / 4.95, 20 steps 3.56 / 4.90;
0034 T1.0/0.8 3.70 / 5.00, 20 steps 3.70 / 4.95.

2 h Pareto (lat img ms, img fid_patch best sampler): 0033 w192 (3.66, ~3.9), 0034 w256 (5.74, 3.70), 0032 w384 (9.22, 3.55).
Video improves much less with compute (5.28 -> 5.06 for w192, 4.92 best; oracle 3.98): video is the weaker task.

## Round 10: new seed directions (generator dev/drafts/make_round_10.py), base 0028 (w192: 4.70 / 5.28; seed
variants of the same recipe 4.80 / 4.89 img)

Batch method: on 1 GPU at fixed time, samples/s is nearly batch independent (B2 56 samples/s, B16 71), so
30 min runs at different batch sizes are ~sample matched (B16 gets ~25% more samples). That is the 8 GPU
question: how much quality per sample does a larger global batch cost, and which LR recovers it.
8 GPUs x current per GPU batch (2 images) = global 16 images.

| Run ID | Run name | Parent runs | lat img / vid ms | Run purpose |
|---|---|---|---|---|
| 0036 | temporal attention -> causal gated depthwise temporal conv, kernel 2 (current + previous frame) | 0028 | 3.32 / 8.63 | seed 1 temporal locality (-17% video latency) |
| 0037 | axial 2D RoPE (base 100, window coordinates) instead of relative position bias | 0028 | 3.31 / 10.4 | seed 2 RoPE |
| 0038 | token budget 16 x 128^2 (16 images / 4 videos), LR 1.2e-3 (28.5 GB) | 0028 | - | seed 3 batch, fixed LR |
| 0039 | token budget 16 x 128^2, LR 3.4e-3 (sqrt(8) rule) | 0038 | - | seed 3 batch, sqrt LR |

Checks: temporal conv is causal (perturbing frame 5 changes outputs 5, 6 only); RoPE preserves norms and
scores depend only on relative offsets.

Launched: ./wan_infra/start_round.sh 0036 0037 0038 0039

### Round 10 results

| run | img fid_patch | img fid | vid fid_patch | vid fid | lat img / vid ms | steps | samples |
|---|---|---|---|---|---|---|---|
| 0036 temporal conv k2 | 4.67 | 3.10 | 5.23 | 4.33 | 3.69 / 9.12 | 57.5k | ~115k |
| 0037 RoPE | 4.73 | 3.17 | 5.26 | 4.39 | 3.76 / 10.8 | 53.8k | ~108k |
| 0038 B16 LR 1.2e-3 | 5.41 | 3.54 | 5.58 | 4.80 | 3.66 / 10.8 | 8.1k | ~130k |
| 0039 B16 LR 3.4e-3 | 5.09 | 3.32 | 5.33 | 4.55 | 3.65 / 10.8 | 8.1k | ~130k |

Paired per step (@10k / 20k): total loss 0028 0.469 / 0.432, 0036 0.464 / 0.427, 0037 0.469 / 0.433; video pair
0028 0.384 / 0.349, 0036 0.376 / 0.340.

Gradient noise scale (dev/evaluations/gradient_noise_scale.jsonl, image 512->1024, B in images, 24 x 16 x 2 samples):
w192 0028 @7.5 min 9.7; 0033 @30 / 60 / 90 min 17.0 / 6.6 / 2.6, final ~260 (CI 160-980); w384 0032 @30 / 60 min
7.1 / 4.8, final ~1000 (unreliable). McCandlish reading with B_noise ~10: samples E = E_min (1 + B / B_noise) ->
B2 1.2 E_min, B16 2.6 E_min; steps S = S_min (1 + B_noise / B) -> B2 6 S_min, B16 1.6 S_min.

## Round 11: batch x LR grid (generator dev/drafts/make_round_11.py), base 0036

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0040 | B4 (4 x 128^2), LR 1.7e-3 (sqrt) | 0036 | sample efficiency curve |
| 0041 | B8, LR 2.4e-3 (sqrt) | 0036 | sample efficiency curve |
| 0042 | B16, LR 6.8e-3 (2x sqrt) | 0036, 0039 | LR optimum at B16 |
| 0043 | B16, LR 3.4e-3, warmup 1000 steps | 0036, 0039 | warmup at large batch |

Launched: ./wan_infra/start_round.sh 0040 0041 0042 0043

### Round 11 results (base 0036, batch in images)

| run | batch | LR | warmup steps | img fid_patch | img fid | vid fid_patch | vid fid | steps |
|---|---|---|---|---|---|---|---|---|
| 0036 | 2 | 1.2e-3 | 100 | 4.67 | 3.10 | 5.23 | 4.33 | 57.5k |
| 0040 | 4 | 1.7e-3 | 100 | 4.59 | 3.09 | 5.28 | 4.42 | 42.0k |
| 0041 | 8 | 2.4e-3 | 100 | 4.67 | 3.12 | 5.29 | 4.42 | 18.4k |
| 0042 | 16 | 6.8e-3 | 100 | 5.32 | 3.30 | 5.50 | 4.66 | 8.7k |
| 0043 | 16 | 3.4e-3 | 1000 | 4.80 | 3.24 | 5.30 | 4.46 | 8.6k |

Per sample loss (pre-decay) @16k / 32k / 48k samples: B2 0.473 / 0.421 / 0.410, B4 0.516 / 0.453 / 0.424,
B8 0.515 / 0.460 / 0.427, B16 (3.4e-3, wu 1000) 0.633 / 0.512 / 0.466, B16 (6.8e-3) 0.547 / 0.492 / 0.489.
Phase 1 full pretraining reference (reports/stage1_pretrain_report.md): global batch 112, LR 4e-4, 500 warmup
steps, 6.8M samples (~60x a 30 min run here).

## Round 12: batch rules continued (generator dev/drafts/make_round_12.py), base 0036

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0044 | B2, LR 1.2e-3, warmup 8000 steps (16k samples) | 0036 | control: long warmup at small batch |
| 0045 | B16, LR 4.8e-3, warmup 1000 | 0043 | LR optimum at B16 |
| 0046 | B32 (2 accumulated micro batches), LR 4.8e-3 (sqrt), warmup 500 (16k samples) | 0043 | next batch point, warmup in samples |
| 0047 | B8, LR 2.4e-3, warmup 2000 (16k samples) | 0041 | warmup at B8 |

Launched: ./wan_infra/start_round.sh 0044 0045 0046 0047

### Round 12 results (base 0036)

| run | batch | LR | warmup | img fid_patch | img fid | vid fid_patch | vid fid | steps | spikes |
|---|---|---|---|---|---|---|---|---|---|
| 0044 | 2 | 1.2e-3 | 8000 | 4.66 | 3.04 | 5.27 | 4.35 | 57.3k | 0 |
| 0045 | 16 | 4.8e-3 | 1000 | 4.84 | 3.25 | 5.28 | 4.46 | 8.6k | 12 (to ~1000) |
| 0046 | 32 (accum 2) | 4.8e-3 | 500 | 5.33 | 3.52 | 5.51 | 4.74 | 4.3k | 0 |
| 0047 | 8 | 2.4e-3 | 2000 | 4.61 | 3.15 | 5.21 | 4.34 | 18.4k | 0 |

Activation stats (dev/evaluations/activation_stats.jsonl, image input, sigma 0.5), max attention logit per block 0-7:
0042 @50%: 1424 33 28 19 18 17 807 162 (stream RMS end 57); 0043 @50%: 20 10 0.5 5 17 21 10 10 (19);
0036 final: 20 29 0.04 0.02 19 14 17 18 (24); 0033 2 h final: 19 26 0.02 0.02 18 16 17 15 (30).

## Round 13: QK-norm (generator dev/drafts/make_round_13.py), base 0036; new runs log train/grad_norm_mean / max

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0048 | B2 LR 1.2e-3 + QK-norm (3.48 / 9.04 ms) | 0036 | QK-norm at normal LR, dead blocks |
| 0049 | B16 LR 4.8e-3 wu 1000 + QK-norm | 0045 | spikes removed? |
| 0050 | B16 LR 6.8e-3 wu 1000 + QK-norm | 0042 | LR ceiling |
| 0051 | B16 LR 3.4e-3 wu 1000 + QK-norm | 0043 | QK-norm at stable large batch LR |

Launched: ./wan_infra/start_round.sh 0048 0049 0050 0051

### Round 13 results (QK-norm)

| run | batch | LR | img fid_patch | img fid | vid fid_patch | vid fid | lat img / vid ms | steps | spikes | grad norm max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0048 | 2 | 1.2e-3 | 4.66 | 3.12 | 5.20 | 4.33 | 3.96 / 9.62 | 51.9k | 0 | 6.4 |
| 0049 | 16 | 4.8e-3 | 4.99 | 3.33 | 5.36 | 4.54 | 3.89 / 9.46 | 7.6k | 13 | 5.8e5 |
| 0050 | 16 | 6.8e-3 | 5.18 | 3.34 | 5.42 | 4.59 | 3.91 / 9.41 | 7.7k | 9 | 3.7e6 |
| 0051 | 16 | 3.4e-3 | 4.89 | 3.33 | 5.32 | 4.51 | 3.97 / 9.43 | 7.6k | 0 | 6.8 |

Spike probes (dev/evaluations/spike_probe_*.pt): onsets / max loss / max grad over 10 min at B16 LR 6.8e-3 const:
original s0 9 / 1.2e4 / ~1e6, s2 6 / 2e5; scale 100 s0 1 / 401 / 5982, s2 0 / 2.3 / 7.8; condition RMS norm s0
0 / 2.1 / 6.6; both 0 / 2.1 / 6.8. Median loss @1900-2250 steps: original 0.522 / 0.510, scale 100 0.527 /
0.504, cond norm 0.542, both 0.540.

## Round 14: spike fixes in real runs (generator dev/drafts/make_round_14.py), base 0036

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0052 | B2 LR 1.2e-3 + condition RMS norm | 0036 | quality cost of the fix at normal LR |
| 0053 | B16 LR 4.8e-3 wu 1000 + condition RMS norm | 0045 | spikes removed, quality |
| 0054 | B16 LR 6.8e-3 wu 1000 + condition RMS norm | 0042, 0050 | LR ceiling |
| 0055 | B16 LR 4.8e-3 wu 1000 + embedding scale 100 | 0045, 0053 | alternative fix |

Launched: ./wan_infra/start_round.sh 0052 0053 0054 0055

### Round 14 results (spike fixes)

| run | batch | LR | fix | img fid_patch | img fid | vid fid_patch | vid fid | spikes | grad norm max |
|---|---|---|---|---|---|---|---|---|---|
| 0052 | 2 | 1.2e-3 | condition RMS norm | 4.66 | 3.12 | 5.21 | 4.35 | 0 | 7.2 |
| 0053 | 16 | 4.8e-3 | condition RMS norm | 4.94 | 3.30 | 5.31 | 4.50 | 0 | 6.5 |
| 0054 | 16 | 6.8e-3 | condition RMS norm | 5.18 | 3.39 | 5.38 | 4.57 | 0 | 8.4 |
| 0055 | 16 | 4.8e-3 | embedding scale 100 | 4.90 | 3.31 | 5.29 | 4.46 | 2 | 1.3e4 |

## Round 15: Attention Residuals (generator dev/drafts/make_round_15.py + dev/drafts/attnres_patch.py), base 0052

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0056 | Block AttnRes, 8 blocks (1 transformer block each, <= 10 sources), SEED 0 | 0052 | AttnRes |
| 0057 | Block AttnRes, 4 blocks (2 transformer blocks each, <= 6 sources), SEED 0 | 0052 | cheaper AttnRes |
| 0058 | Block AttnRes, 8 blocks, SEED 1 | 0056 | replicate |
| 0059 | 0052 SEED 1 | 0052 | paired reference for 0058 |

Judged on paired per step loss (same seed) and fid at fixed time (AttnRes pays ~30% throughput at D = 192).

Profiles (warm, 60 s): 0059 base 1784 steps, 3.27 / 8.62 ms; 0056 / 0058 block-8 1333-1362 steps (-25%), 4.3 / 10.4 ms;
0057 block-4 1494 steps (-16%), 3.78 / 9.81 ms.

Launched: ./wan_infra/start_round.sh 0056 0057 0058 0059

### Round 15 results (Attention Residuals)

| run | variant | seed | img fid_patch | vid fid_patch | lat img / vid ms | mem img / vid MB | steps | paired loss @30k |
|---|---|---|---|---|---|---|---|---|
| 0052 | base | 0 | 4.66 | 5.21 | 3.69 / 9.15 | 203 / 373 | 56.3k | 0.406 |
| 0056 | block AttnRes, 8 blocks | 0 | 4.96 | 5.29 | 4.71 / 10.9 | 311 / 626 | 41.1k | 0.391 |
| 0057 | block AttnRes, 4 blocks | 0 | 4.61 | 5.31 | 4.18 / 10.3 | 269 / 533 | 46.4k | 0.395 |
| 0058 | block AttnRes, 8 blocks | 1 | 5.05 | 5.32 | 4.67 / 10.7 | 311 / 626 | 41.7k | 0.396 |
| 0059 | base | 1 | 4.71 | 5.25 | 3.71 / 9.02 | 203 / 373 | 55.2k | 0.410 |

Memory breakdown, image 1024 eager forward (above 31 MB weights): base 0052 +172.5 MB = SDPA math path 128 MB (9
buffers) + window partition 18 + 4 x 6 MB stream sized; 4D layout +89.5 MB = FFN input projection 32 + depthwise
conv output 32 + 4 x 6 MB. Video 512: +339 -> +203 MB (FFN 72 MB + 4 x 13.5).

## Round 16: memory / latency architecture changes (generator dev/drafts/make_round_16.py), base 0060

Candidates measured without training (dev/architecture_cost.py, random weights; eager inference peak above weights,
compiled latency):

| variant | params | image 1024 | video 512 |
|---|---|---|---|
| base 0060 | 15.6M | +89.5 MB, 3.08 ms | +202.4 MB, 8.06 ms |
| A depthwise conv after the SwiGLU gate | 15.6M | +90.0 MB, 3.68 ms | +202.9 MB, 7.68 ms |
| B RoPE + unshifted windows | 15.6M | +89.5 MB, 2.97 ms | +202.9 MB, 7.55 ms |
| C SwiGLU hidden 384 | 15.0M | +73.7 MB, 2.90 ms | +166.9 MB, 7.19 ms |
| D unshifted windows (bias kept) | 15.6M | +89.5 MB, 3.01 ms | +202.9 MB, 7.55 ms |
| E A + B | 15.6M | +90.0 MB, 3.57 ms | +202.9 MB, 7.17 ms |
| F RoPE (with shift) | 15.6M | +89.5 MB, 3.05 ms | +202.9 MB, 8.07 ms |
| G GELU MLP hidden 4D (same params) | 15.6M | +80.5 MB, 3.04 ms | +180.9 MB, 7.48 ms |
| H B + C | 15.0M | +74.0 MB, 2.79 ms | +166.9 MB, 6.73 ms |
| I G + B | 15.6M | +80.5 MB, 2.93 ms | +181.4 MB, 6.97 ms |

A dropped (no memory gain: projection output + gate temporaries peak the same; slower compiled image path).
The FFN width is the memory lever; shift removal is a latency lever only.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0061 | RoPE + unshifted windows | 0060, 0037 | latency, cost of no shift |
| 0062 | SwiGLU hidden 384 | 0060 | memory / latency via FFN width |
| 0063 | GELU MLP 4D instead of SwiGLU | 0060 | memory at equal params |
| 0064 | RoPE + unshifted + SwiGLU 384 | 0061, 0062 | lean combination |

Reference: 0052 / 0059 (same function as 0060, seeds 0 / 1): 4.66 / 4.71 img, 5.21 / 5.25 vid.
Launched: ./wan_infra/start_round.sh 0061 0062 0063 0064

### Round 16 results

| run | img fid_patch | img fid | vid fid_patch | vid fid | lat img / vid ms | mem img / vid MB | params | steps | paired loss @20k total / img / vid |
|---|---|---|---|---|---|---|---|---|---|
| 0052 base (5D layout) | 4.66 | 3.12 | 5.21 | 4.35 | 3.69 / 9.15 | 203 / 373 | 15.6M | 56.3k | 0.432 / 0.504 / 0.346 |
| 0061 RoPE + unshifted | 4.76 | 3.18 | 5.27 | 4.42 | 3.34 / 7.95 | 120 / 235 | 15.6M | 52.0k | 0.435 / 0.507 / 0.349 |
| 0062 SwiGLU 384 | 4.81 | 3.23 | 5.30 | 4.45 | 3.22 / 7.55 | 103 / 199 | 15.0M | 53.3k | 0.436 / 0.508 / 0.351 |
| 0063 GELU MLP 4D | 4.81 | 3.22 | 5.33 | 4.47 | 3.36 / 7.83 | 111 / 215 | 15.6M | 54.6k | 0.431 / 0.502 / 0.345 |
| 0064 lean combination | 4.94 | 3.24 | 5.32 | 4.46 | 3.13 / 7.19 | 102 / 199 | 15.0M | 57.4k | 0.441 / 0.512 / 0.356 |

## Round 17: 2 h batch / LR dynamics + production schedule (generator dev/drafts/make_round_17.py), base 0060

Accumulation over successive loader batches with a time based schedule (ACCUMULATION_SCHEDULE), LR x sqrt(accumulation),
logs train/samples and train/accumulation. Square root LR rule from B2 / 1.2e-3; decay over the last 50%.
Profiled samples/s: B4 94, B16 80, B64 80 -> ~equal samples in 2 h (~0.6M).

| Run ID | Run name | Parent runs | ~steps | Run purpose |
|---|---|---|---|---|
| 0065 | B4, LR 1.7e-3, wu 100 | 0060, 0040 | 170k | small batch reference |
| 0066 | B16 (micro 16 = 8 GPUs x 2), LR 3.4e-3, wu 1000 | 0060, 0043 | 36k | constant production size batch |
| 0067 | B64 (16 x 4 accumulation), LR 6.8e-3, wu 1000 | 0066 | 9k | large batch at 0.6M samples |
| 0068 | PRODUCTION: micro 16, ramp 16 -> 32 -> 64 at 25 / 50%, LR 3.4e-3 x sqrt(acc), wu 1000 | 0066, 0067 | 18k | production recipe, 1 GPU version |

30 min references: B2 0052 4.66, B4 0040 4.59, B8 0041 4.67, B16 0043 4.80 (5D layout, old base). 2 h reference w192 B2:
0033 4.03 / 5.06 (older base without temporal conv / condition norm).

Launched: TRAINING_TIME_LIMIT=135m ./wan_infra/start_round.sh 0065 0066 0067 0068

### Round 17 first launch (0065-0068): STOPPED at ~43 min, not benchmarked (user decision)

Stopped after 0065 diverged (attention logit growth, no QK-norm). Partial results (dev/logs/round_17_partial.txt), at
~158k samples, all pre-decay: loss 0066 B16 0.386, 0068 ramp 0.386, 0067 B64 0.397, 0065 B4 0.48 (diverging, max grad
1681); mean grad norm B16 0.10, B64 0.08, ramp 0.09. Early read: the ramp = constant B16 per sample, B64 slightly behind.

## Round 17 relaunch: 2 h batch / LR dynamics on one identical large model (generator dev/drafts/make_round_17b.py)

Model (all runs): run_0060 + QK-norm + GELU MLP (hidden 4D) at width 384: 45.0M params, lat 7.5 / 20.8 ms, window transformer
(8 blocks, shifted 8x8 windows, relative bias), causal temporal conv, learned source path, residual parameterization,
condition RMS norm, 4D attention layout. Only batch and LR differ: sqrt rule (LR = 1.2e-3 x sqrt(B / 2)); warmup over the
same 16k samples; linear decay over the last 50%; 2 h; micro batch 8 with accumulation (4 for B4). Profiled ~33-37
samples/s -> ~250k samples per run.

| Run ID | Run name | ~steps | Run purpose |
|---|---|---|---|
| 0069 | B4, LR 1.7e-3, warmup 4000 | 67k | small batch reference |
| 0070 | B16 (8 x 2), LR 3.4e-3, warmup 1000 | 15k | constant production size batch |
| 0071 | B64 (8 x 8), LR 6.8e-3, warmup 250 | 3.7k | large batch |
| 0072 | PRODUCTION: ramp 16 -> 32 -> 64 at 25 / 50%, LR 3.4e-3 x sqrt(B / 16), warmup 1000 | 7.5k | production recipe, 1 GPU version |

Launched: TRAINING_TIME_LIMIT=135m ./wan_infra/start_round.sh 0069 0070 0071 0072

### Round 17 results (relaunch; 2 h; identical 45M w384 model; only batch / LR differ)

| run | batch | LR | img fid_patch | img fid | img mse_lr e3 | vid fid_patch | vid fid | steps | samples | final loss | grad norm max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0069 | 4 | 1.7e-3 | 3.77 | 2.62 | 1.07 | 4.98 | 3.98 | 68.2k | 182k | 0.328 | 11.9 |
| 0070 | 16 | 3.4e-3 | 3.98 | 2.73 | 1.10 | 4.95 | 3.98 | 14.8k | 169k | 0.337 | 5.7 |
| 0071 | 64 | 6.8e-3 | 4.10 | 2.85 | 1.14 | 5.06 | 4.09 | 3.7k | 170k | 0.344 | 3.4 |
| 0072 | 16 -> 32 -> 64 | 3.4 -> 6.8e-3 | 3.98 | 2.78 | 1.11 | 4.97 | 3.99 | 7.4k | 166k | 0.337 | 5.8 |

Loss at equal samples (pre-decay; samples = latent batch entries): @30k B4 0.414 / B16 0.430 / ramp 0.432 / B64 0.487;
@60k 0.381 / 0.386 / 0.387 / 0.411; @90k 0.372 / 0.372 / 0.370 / 0.389; @120k 0.357 / 0.350 / 0.354 / 0.367.
Latency 8.0-8.2 / 21.3-21.6 ms, memory 239 / 440 MB (45M params; w384 2 h 0032 without the new stack: 3.76 / 4.92).

## PRODUCTION RECOMMENDATION (after round 17)

Architecture (scale the width / depth to ~0.5B; keep every component):
- Window transformer on the full resolution latent grid (no UNet): DiT style adaLN-zero blocks, LayerNorm without affine,
  shifted 8x8 window attention with a learned relative position bias (or axial RoPE, equal), causal gated depthwise
  temporal convolution (kernel 2) instead of temporal attention, feed forward with a depthwise 3x3 conv on the hidden
  (GELU MLP 4D: same quality per step as SwiGLU, -8% inference memory).
- Stabilizers (both required): RMS norm of the condition vector (sigma + hff embeddings; fixes single sample loss
  spikes from narrow sigma bands) and QK-norm (fixes slow attention logit growth over long training).
- Inputs: learned source path (LR encoder, pixel shuffle, causal phase temporal upsampling) + causal frame alignment of
  the concatenated source; residual parameterization (flow on (x - aligned source) / per channel residual std); copy of
  frame 0 for temporal only upscaling.
- Implementation: windows folded into the SDPA batch dimension (4D) so the fused kernel is used (-48% inference memory).
Training (8 GPUs):
- Global batch 16 images / 4 videos at start (8 GPUs x 2), ramp to 32 and 64 via gradient accumulation (equal quality per
  sample to constant 16 at 0.17M samples, fewer all-reduces; the critical batch grows with training, so larger batches
  later in a multi-million-sample run are safe). Do not start at 64.
- LR: square root rule with batch (x sqrt(2) per batch doubling) relative to the 0.5B model's own small batch LR; the
  w192-w384 models used 1.2e-3 at B2 -> 3.4e-3 at B16. For 0.5B, retune the base LR (phase 1 451M UNet: 4e-4 at batch 112);
  expect lower than here (wider model). Warmup >= 16k samples (1000 steps at B16) - essential at large batch / high LR.
  Linear decay to 0 over the last 30-50% (essential, 0.3-0.8 fid_patch in phase 1). EMA neutral (not needed). AdamW
  (0.9, 0.95), grad clip 1, bf16, fused AdamW.
- Data: train on the benchmark pair types (image 512 -> 1024, video 17 -> 33 @512) at image : video 5 : 4 (auxiliary pairs
  did not help at 30 min; re-check with the large data set).
Sampling: Euler, 14-20 steps, shift 4, initial noise temperature ~0.9-1.0 for spatial / 0.7-0.8 for temporal upscaling
(better trained models want temperature closer to 1 and more steps; re-sweep on the final model).
Open: small batch (B4) gave the best final image fid at 0.17M samples (3.77 vs 3.98) though B16 led per sample before the
decay -> the annealing phase benefits from many small steps; consider a longer decay or a smaller batch during the decay
in production (untested).

## OPEN QUESTION: training scales, small details and "benchmark maxing" (discussion after round 17, untested)

Context: rounds 3 / 12 converged on training only the two benchmark pair types (0009 vs 0008 at 30 min: 5.84 / 5.63
vs 5.94 / 5.90, within noise: the extra pair types cost steps but did not clearly hurt). Phase 1 production showed that a
model trained at one scale forgets others (CONTEXT.md, pretrain phase 2). So "benchmark pairs only" is a 30 min budget
optimum, not a recommendation for production.

Hypothesis (user): upscalers (this one and others) mostly fail on small but important details (text, background
faces). In high resolution photos such content is rare per pixel area, so the model's prior for "small face / small
text" (a few latent tokens, reconstructed from ~2x2 LR tokens) is undertrained. Downscaled images (256 -> 512, 128 ->
256) or 1024 mosaics of 4x4 downscaled images contain these features densely and at exactly that footprint.
Assessment: plausible. The model is local (receptive field ~40 latent tokens; global tokens / REPA / text gave nothing
in phase 1), so the gain is a better local prior for small objects, not "world understanding". For this local model,
mosaics ~= small image buckets (16 independent 128 -> 256 pairs) but keep the 1024 training shape. Caveat: downscaling
makes small objects unnaturally clean (real background faces are often defocused / compressed) -> risk of
hallucinated sharp faces in bokeh; hff conditioning helps; keep such buckets a minority (~20-30%). Also likely more
valuable than 256 -> 512 for 2K / 4K / 8K production: high resolution crops (2048 -> 4096 crops at 1024 target size),
whose latents are much smoother per token than anything trained on so far.

Proposed experiment (needs a new eval more than new training):
1. Mosaic benchmark: 1024 canvases of 4x4 held-out images downscaled to 256 px, ground truth = VAE encoding of the
   mosaic, LR = the usual 2x downscale + encode; same 512 -> 1024 task and fid_patch pipeline but dense in small faces
   / text / objects (one encoding pass of ~1000 held-out images with the existing dataset script).
2. Held-out 256 -> 512 and 128 -> 256 evals (keys already in the dataset).
3. Training, same seed (paired loss): A benchmark pairs only (control); B + 20-30% 256 -> 512 / 128 -> 256 pairs (no new
   data); C + 20-30% 1024 mosaic pairs (new encoding, same shape). Evaluate all on standard + mosaic + small-scale evals.
   Prediction: B / C clearly better on the mosaic and small-scale evals, standard benchmark within noise. If the mosaic
   eval does not move, the bottleneck is capacity / semantics for small objects -> next lever: targeted data (crops with
   faces / text found by a detector, oversampled).
4. Budget: the effect concerns rare content and 30 min runs are step limited -> 2 h round (or 30 min with seed
   replicates judged on the mosaic eval). Complement fid with a visual check of face / text tiles (and OCR / face
   similarity if available).
