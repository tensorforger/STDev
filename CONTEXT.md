# Research context

Incremental history and knowledge overview for agents and decision making. Newest notes are at the bottom. Details live in `reports/` and `autoresearch/`; this file only summarizes and links. It does not describe the codebase.

## Goal

Generative latent upscaler(s) that support many generative models and many VAE latent spaces.

- A single-step "latent upscale adapter" is a regression model. A generative latent upscaler is itself a generative model: it needs no extra denoise passes of the base model, and its output can be decoded directly by the VAE into a high quality, high resolution image.
- Related work:
  - https://github.com/tensorforger/comfyui-flow-upscaler: earlier project of the author, FLUX.2 only.
  - https://arxiv.org/abs/2511.10629: main LUA paper.
  - Model providers start to ship their own latent upscalers: LTX 2.3 (open weights), MiniMax-H3 (API only so far).

## History

### 1. Autoresearch on FLUX.2 latents (2x, 512 -> 1024 px)

Setup: agent-driven rounds of 8 parallel runs, 20k FLUX.2-klein-generated samples, trade-off of patch FID (primary metric, native 256 px tiles) and latency. About 12 h on 8x RTX 5090. Log and final recipe: `autoresearch/experiment_logs.md` (section "RECOMMENDED SOLUTION FOR SCALING").

Key insights:
- Capacity belongs at full latent resolution: a deep and wide decoder there matters most. Depth is more latency efficient than width. Parameters at low resolution do not pay off.
- Local context reasoning matters: window attention (8x8) helps. SDXL-style window transformer blocks everywhere are best per step and the best option for larger budgets.
- Optimal LR (4e-4 with a linear decay over the last ~30%, the decay is essential) and sigma shifts were found. Sampling step optimum grows with model quality.
- Negative (no gain per step): REPA, global tokens, text cross-attention, coupled noise, 2x2 patchify, flip augmentation, EMA for short runs.
- Compute-optimal size grows with budget. Patch FID: baseline 4.54, best 1.30 (304M SDXL-style or 88M deep decoder, 2 h on one GPU).
- Caveat: 20k samples is small, the smallest models start to saturate.

### 2. Pretraining on ReLAION (online pipeline)

Model: 451M UNet combining the best autoresearch findings, conditioned on a pooled text embedding and 4 high frequency (HF) features. About 18 h on 8x RTX 5090 in total. Pipeline: `reports/pretrain_pipeline_plan.md`.

- Phase 1, 256 -> 512: patch FID 2.42 (EMA) after 9.7 h. Saturated before the decay. `reports/stage1_pretrain_report.md`.
- Phase 2, 512 -> 1024: patch FID 1.72 (EMA) after 8.7 h. Almost no improvement during training (flat from 30 min, only the LR decay helped), and the run was data starved (each sample seen ~6x). `reports/stage2_pretrain_report.md`.
- Later evaluation showed that the phase 2 model forgot 256 -> 512 upscaling.
- Conclusion: sequential resolution phases are wrong. Move to mixed training with resolution buckets, where the highest resolution is the rarest.

### 3. Post-training experiments (FLUX-generated data, stage 2 model as start)

- SFT on FLUX-generated data (`reports/sft_vs_stage2_pretrain.md`): patch FID 1.90 -> 0.82 on the FLUX benchmark, but almost no visual improvement on real images. It only fits the generated distribution and makes CFG less robust. Not recommended as a general step.
- Rectified flow distillation to 1-4 steps (`reports/rectified_flow_distillation_vs_sft.md`): works well, keeps ~98% of the teacher gain. 1 step is enough.
- Distillation into a small model (`reports/small_single_step_distillation.md`): 48.6M, single forward pass, ~98x faster than the 20-step teacher, still works well.
- Architecture vs training recipe (`reports/flow_upscaler_architecture_vs_training.md`): both the architecture and the training recipe improved over the earlier FlowUpscaler.
- CFG (judged visually in ComfyUI, "user preference"):
  - Prompt conditioning has almost no impact, even with strong CFG. CFG works through HF conditioning.
  - Without CFG, results are sometimes slightly blurry.
  - Best: CFG ~6.0, negative = no HF conditioning, positive = ~0.9 on all HF features. 1.0 is too noisy, larger CFG gives grid artifacts.
  - Only the first two features matter ("noise_level", "edge_sharpness") and the model barely distinguishes them. "texture_detail" and "detail_scale" have almost no impact.
  - Suggested simplification: one "sharpness" scalar condition (CPBD or similar).

### 4. HFF conditioning standard

Sharpness metric comparison on 1000 ReLAION samples (`notebooks/relaion_sharpness_metrics.ipynb`): the Immerkaer estimator (whole image, luminance) best captures the high frequency content the model needs. It is now the standard high frequency feature (HFF) conditioning for future models: `src/stdev/hff/immerkaer.py` (batched torch, images and videos, output [0...1]). It is quantile-normalized with a fixed table calibrated once on ~10k ReLAION images (`dev/calibrate_immerkaer.py`), so values are absolute and independent of any later dataset.

### 5. VAE wrappers

8 VAEs (images and videos) behind a common interface, plus 3 tiny VAEs for the video ones. Latent statistics, roundtrip quality, frame rules and speed: `reports/all_vaes_test.md`.
Notable: latent std ranges 0.7-1.3 and channels 4-128 across VAEs, Wan 2.1 is under-scaled, MiniMax-H3 needs T = 5 + 17n.

### 6. Multi-VAE vs per-VAE test

Test of the hypothesis "one model for many VAEs": 53M UNet with shared core and per-VAE adapters (`src/stdev/multi_vae_upscalers/small_universal.py`), trained on 9 VAEs at once, against a fresh same-seed model per VAE with the same step budget. Details: `reports/multi_vae_vs_single_vae.md`.

- Per-VAE models are better on all 9 VAEs (patch FID 1-48% lower, mostly 25-48%). The VAEs interfere inside the shared model; interference is small only for tiny_minimax_h3 and tiny_wan_2_1.
- Caveats: single runs got ~10% more steps, evaluation on a training shard, both settings are undertrained (most are still worse than bilinear).
- Decision: train separate models from scratch for every VAE.

## Open problems

- Pretrain phase 2 progress and forgetting of lower resolutions (data limited and sequential schedule).
- Only FLUX.2 latents are covered so far; per-VAE models are not yet trained at full scale.
- Per-VAE models multiply the training cost; the recipe must be cheap or reusable across VAEs.
- The small 53M model is still worse than bilinear for most VAEs after ~6k steps.
- Condition design: text is nearly useless, 4 HF features reduce to ~1 effective feature.

## Plan

1. Separate model per VAE, each trained from scratch (the universal shared-core model lost to per-VAE models on every VAE, see history 6). The universal model is shelved.
   - Reuse the same architecture and recipe for all VAEs, only adapting channel counts and video handling.
2. Pretrain the new base models from scratch:
   - more training than before,
   - no prompt conditioning,
   - a single sharpness conditioning scalar,
   - better budget allocation with resolution bucket training.
