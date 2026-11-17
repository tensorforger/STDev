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

### 7. VAE implementation priority list

Asked two agents for deep research on what VAEs should be implementd fist. More detailed analysis: `reports/prioritized_vaes_by_demand_qwen.md`

Summarizing it, we should start from **Wan2.1-VAE**, then **MiniMax-H3 VisualVAE** and **Qwen-Image-2.1 VAE**. 

### 8. Autoresearch on Wan 2.1 latents (phase 2: images 2x, videos 2x in time)

Setup: agent-driven rounds of 4 parallel 30 min runs (plus two 2 h rounds), Wan 2.1 VAE, 19k images + 9k videos, benchmark image 512 -> 1024 (full VAE) and video 17 -> 33 frames at 512 (tiny VAE), primary metric image patch FID. ~70 runs (run_0000 - run_0072). Log, results and the production recommendation: `autoresearch_2/experiment_logs.md` (section "PRODUCTION RECOMMENDATION").

Key insights:
- Wan latents are smooth (HR latents correlate 0.95 with the upsampled LR; residual std 0.22 vs 0.79 for FLUX.2). The flow must run on the per channel normalized residual over the aligned LR latents; on raw latents the detail is buried under the noise (-3 patch FID). The same applies to any smooth / low channel VAE.
- The LR input path matters: a learned LR encoder (pixel shuffle, causal phase temporal upsampling) halved patch FID vs. bilinear latent upsampling. Causal frame alignment is exact for the Wan VAE (target frame j <- source frame ceil(j / 2), frame 0 identical).
- A plain window transformer at full latent resolution (DiT style adaLN-zero, shifted 8x8 windows) beats the UNet; a causal depthwise temporal convolution (previous frame only) replaces temporal attention with no loss (temporal locality works). SwiGLU / GELU equal; RoPE = relative bias; AttnRes better per step but costs memory (rejected: inference buffers).
- Stability needs two fixes: RMS norm of the sigma / hff condition vector (single-sample loss spikes from narrow sigma bands at high LR) and QK-norm (slow attention logit growth over long training).
- Inference memory: 5D window attention silently fell back to the math SDPA path; folding windows into the batch halved the memory peak (-7% latency). The FFN hidden is the remaining peak.
- Batch scaling: square root LR rule with long warmup; the critical batch grows during training; a ramp 16 -> 64 matches constant 16 per sample; do not start large. 30 min runs are step limited (small batch, fast models win); training seed noise ~0.1-0.3 patch FID.
- Sampling: few steps (14-20), shift ~4, initial noise temperature < 1 (0.9-1.0 images, 0.7-0.8 video); optimum moves towards more steps / temperature 1 with training.
- Results: image / video patch FID 49.1 / 13.9 (baseline) -> 3.77 / 4.98 (45M, 2 h on one GPU); pixel bicubic 8.21, LR frame copy 5.69, VAE oracle 1.30 / 3.98.

### 9. Wan 2.1 pretraining (started 2026-11-16, 8x RTX 5090)

222M window transformer (autoresearch_2 recipe at width 768, depth 16) + 10% hff dropout to a null embedding (CFG). Online pipeline: GPUs 6-7 encode ReLAION images (full VAE; 90% 512 -> 1024, 10% 256 -> 512) and vk-50k clips from S3 (tiny VAE; mostly 17 -> 33 frames at 512, rare 1024 / combined temporal + spatial pairs) into a rolling cache and runs an hourly 100-image patch FID benchmark; 6 GPUs train (DDP, same pair type per micro-step on all ranks, batch ramp by accumulation, sqrt LR). Open-ended: LR decay over 30% more steps starts on a control file. Plan, measurements and LR probe: `reports/wan_pretrain_plan.md`; code `src/stdev/wan_pretrain/`, launcher `scripts/run_wan_pretrain.py`.

### 10. ComfyUI development node pack

The repository root doubles as a ComfyUI custom node pack for local testing of the pretraining checkpoints: clone to `custom_nodes/`, put checkpoints into `models/latent_upscale_models/`. Entry points `__init__.py` / `nodes.py` only register modules; logic lives in `src/stdev/comfy_nodes/<vae>/`, one mostly independent node set per VAE upscaler (new ones are added to `NODE_MODULES`). Checkpoints are safetensors with metadata (`vae`, `model_config`), converted by `model_ops/download_checkpoint.py` (newest EMA weights of a run from the `checkpoints` bucket). 

Supported packs:

1. Wan 2.1 (load, image, video nodes; steps, CFG on the hff condition, shift, noise scale, spatial / temporal toggles).

## Open problems

1. Flux.2 tested at scale as the direct hr latent generator. Autoresearch 2 found that Wan 2.1 VAE requires generation of residuals. So, is there a single optimal approach for all VAEs? Or universal abstraction that covers both modes?
2. Per-VAE models multiply the training cost; the recipe must be cheap or reusable across VAEs.
3. Condition design: text is nearly useless, 4 HF features reduce to ~1 effective feature. Need to test the 1 feature at scale.
4. Small important details (text, background faces) are where upscalers fail. Hypothesis: they are rare per pixel area in high resolution training images, so their prior is undertrained. See `autoresearch_2/experiment_logs.md` ("OPEN QUESTION: training scales").

## Plan

1. Separate model per VAE, each trained from scratch (the universal shared-core model lost to per-VAE models on every VAE). The universal model is shelved. 
   - Implement VAE upscalers in optimal order (`reports/prioritized_vaes_by_demand_qwen.md`)
   - Reuse the same architecture and recipe for all VAEs, only adapting channel counts and video handling, if that is possible (open problem 1)
2. Pretrain the new base models from scratch:
   - more training than before
   - autoresearch 2 recipe
