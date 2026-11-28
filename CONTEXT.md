# Research context

Incremental history and knowledge overview for agents and decision making. Newest notes are at the bottom. Details live in `reports/` and `autoresearch/`; this file only summarizes and links. It does not describe the codebase.

## Goals

- Explore generative latent upscalers in different VAE latent spaces. 
- Find what can be practical.
- Scale it up, train better models.
- Make it practical.

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

Status at 26.8 h (before decay, `reports/wan_pretrain_plan.md`): 1000-image image fid_patch 2.78 (EMA) vs oracle 1.37 / pixel bicubic 5.49; research best 3.77 (oracle 1.30). Still improving log-linearly in time. The 100-image hourly benchmark is strongly biased (model looks near bicubic there): trends only. Noted: video 512 loss flat since ~16 h (50k clips), wrong residual table for tiny-VAE spatial video (target std 1.5-2), steady weight norm growth (wd 0.01 too weak), small text still garbled.

Finished 2026-11-18 at 45.7 h (step 161.6k, stopped mid LR decay: no gain). Final EMA fid_patch 2.76 on the fixed 1000-image set (oracle 1.37, bicubic 5.49). Patch FID saturated from ~1 day on (paired eval: -0.02 to -0.04 over the last 47k steps); the decay still lowered the training loss and mse_lr, but not patch FID or visuals (90k vs 150k nearly identical). Internals: no dead blocks, feed forward adaLN gates up to ~15 in deep blocks, last block feed forward off, hff changes the velocity by only 1-4% (CFG needs large scales). Checkpoint on R2 `checkpoints/wan_pretrain_v1/checkpoint_000161590.pt`; latent cache kept for distillation. Unpaired 1000-image evals on rebuilt sets are not comparable (only the first ~99 images match): always reuse `wan_eval_1000`.

### 10. ComfyUI development node pack

The repository root doubles as a ComfyUI custom node pack for local testing of the pretraining checkpoints: clone to `custom_nodes/`, put checkpoints into `models/latent_upscale_models/`. Entry points `__init__.py` / `nodes.py` only register modules; logic lives in `src/stdev/comfy_nodes/<vae>/`, one mostly independent node set per VAE upscaler (new ones are added to `NODE_MODULES`). Checkpoints are safetensors with metadata (`vae`, `model_config`), converted by `model_ops/download_checkpoint.py` (newest EMA weights of a run from the `checkpoints` bucket). 

Supported packs:

1. Wan 2.1 (load, image, video nodes; steps, CFG on the hff condition, shift, noise scale, spatial / temporal toggles). Distilled single step students (large and small, `architecture` metadata `WanStudentTransformer`) have their own load / image / video nodes (guidance, hff, seed); `model_ops/download_checkpoint.py --run wan_distill_v1 wan_small_s20m ...` converts them.

### 11. Wan 2.1 single step rectified flow distillation (2026-11-18)

40k teacher pairs from the pretraining cache (pretraining type shares; hff true / random / null, CFG 1-3 on half of the non-null, negative = null hff), 2 h on 8 GPUs. Student = teacher + zero initialized guidance condition, trained from the teacher EMA on sigma 1 only. One epoch (~15 min) is enough: 1-step EMA fid_patch 2.87 vs teacher 2.76 with 20 steps (~96% of the gain over bicubic), same mse_lr. Raw weights oscillate (lr 1e-3 is high), the guidance input changes little. Report: `reports/wan_rf_distillation.md`; code `src/stdev/wan_distill/`; checkpoint `distill_runs/wan_distill_v1` (R2 `checkpoints/wan_distill_v1/`).

### 12. Small single step students (size ladder, 2026-11-19)

The large student architecture scaled down (19.5M -> 0.52M), trained from scratch on one GPU each for ~9 h on pairs generated online by the large 1-step student. Final 1-step fid_patch: 19.5M 3.79, 10.8M 3.98, 5.0M 4.46, 2.3M 5.09, 1.1M 6.22, 0.52M 8.00 (bicubic 5.49): ~2M is the smallest size that beats bicubic. Not saturated at 7 h (decay still gave -0.25 to -0.9). Below 20M the models are overhead bound (6-9 ms vs 4.6 ms for 20M at 1024 px; head dim 32 slower): smaller is not faster without CUDA graphs. Small text degrades with size. Report: `reports/wan_small_students.md`.

### 13. Unified benchmark against other 2x upscaling solutions (2026-11-20)

All solutions get only the 64x64 Wan 2.1 latent of a 512 image and return the decoded 1024 image; 1000 Axis-V1 photos, metrics fid, patch_fid, roundtrip MSE / CIEDE2000, batch 1 latency (decode included, eager bf16, RTX 5090), peak VRAM. Compared: our base (20 steps) and students (200M, 20M, 2M; null hff, guidance 1, full or tiny VAE decoder) against bicubic, SPAN, Real-ESRGAN, spacepxl Wan2.1-VAE-upscale2x, bicubic / NN (ComfyUI-VAE-Utils) / SesquiLSR latent upscalers. Patch FID: base 4.31, 200M 4.36, 20M 5.85, spacepxl 6.33 (29 ms), 20M + tiny VAE 10.43 (26 ms), SPAN 11.19, Real-ESRGAN 13.83, bicubic 14.13, SesquiLSR 29, NN latent 51.
- Ours is best at >= 140 ms (-70% patch FID vs bicubic on this sharper domain, -50% on ReLAION); 1 step distillation costs +1%.
- Latency is dominated by the full VAE decode at 1024 (~115 ms more than the tiny decoder), not by the upscaler; the tiny decoder costs +78% patch FID and makes model size irrelevant (2M 11.60 vs 20M 10.43). spacepxl's decoder (integrated 2x at the cost of a 512 decode) is the best low latency option; nothing of ours lies between 30 and 140 ms.
- Feeding our latents to the spacepxl decoder (4x, downscaled back) is worse than the original VAE (4.51 vs 4.36). Ground truth hff does not help at guidance 1 (+-0.1 patch FID), null hff is enough. Third party latent upscalers are unusable without a refinement pass.
- Next: a fast decoder for our latents. Report: `reports/wan_upscaling_benchmark.md`; code `scripts/run_upscale_benchmark.py`, `src/stdev/upscale_benchmark/`; plot `scripts/plot_upscale_benchmark.py`.

### 14. Public release "Steppe" (2026-11-21 / 22)

Released under the "Steppe" brand (MIT, Wan VAE Apache-2.0 credited in NOTICE) on the TensorForger accounts. Production ComfyUI node pack [ComfyUI-Steppe](https://github.com/TensorForger/ComfyUI-Steppe): a clean reimplementation, independent of this repo (pure PyTorch `steppe` package plus thin ComfyUI nodes: loader, basic/advanced upscale for images and videos, spatial and temporal, measure hff). It is in the Comfy registry and ComfyUI-Manager (`steppe`, v1.0.3) and auto-downloads models. Three models were uploaded to [TensorForger/Steppe-Wan2.1](https://huggingface.co/TensorForger/Steppe-Wan2.1): 200M base (teacher, 20 steps), 200M 1-step (default) and 20M 1-step (lite). Its `config.json` is the model catalog the node fetches, which makes HF count downloads. A Gradio ZeroGPU demo Space, [TensorForger/Steppe](https://huggingface.co/spaces/TensorForger/Steppe), runs 2x and chained 4x image upscaling with the 200M 1-step model. Defaults: guidance 1, hff 0.9. Verified in ComfyUI with `wan_2.1_vae` / `qwen_image_vae`, chained 2x and any aspect ratio. Card and image generators are in `scripts/cards/`.

New insights from release: community asks for a bit different VAE support then initial plan, most liked comment asks for MOSS-Image (**Flux.2 VAE**) and Qwen Image 2.1 (**Qwen Image 2.1 VAE**)

### 15. Autoresearch on FLUX.2 latents (phase 3: 512 -> 1024, 2026-11-22 / 23)

Setup: 5 rounds of 4 parallel 30 min runs (run_0000 - run_0019, ~50M models), 20k ReLAION + 20k 4KLSDB images (center crop, FLUX.2 VAE; low resolution inputs also stored with other downscaling filters and JPEG), fixed Axis-V1 1k benchmark (patch FID, FID, low res MSE, chroma dE00, single forward latency p95, peak VRAM). Log, answers and the pretrain recipe: `autoresearch_3/experiment_logs.md` ("RECOMMENDED PRETRAIN RECIPE"); code `src/stdev/flux_upscalers/`, `src/stdev/flux_benchmark/`, `flux_infra/`; data `scripts/download_flux2_sources.py`, `scripts/encode_flux2_datasets.py`.

- Phase 1 and 2 findings compound: the phase 2 plain window transformer + normalized residual + learned source path beats the phase 1 UNet on FLUX.2 too (2.45 vs 2.95 with all shifts on both). The residual gain is small on FLUX.2 (-0.3 vs -3 on Wan) because FLUX.2 latents are not smooth (HR / upsampled LR correlation 0.69 vs 0.95), but it never hurts: residual is the universal parameterization (open problem 1).
- Do not transfer: initial noise temperature < 1 and sampling shift 4 (FLUX.2: 30 steps, shift 2, temperature 1.0).
- Data: 4KLSDB is worth more per sample than ReLAION; ReLAION : 4KLSDB = 1 : 2 is best (small, replicated gain).
- Input corruption: clean-trained models collapse on JPEG inputs (patch FID 47 vs 2.5) and degrade on blurry inputs; 20% corruption (JPEG weighted) costs +0.11 on the bilinear benchmark and fixes most of it (JPEG 10.4). Adopted for the pretrain.
- References: VAE oracle 0.71 (Wan 1.30), pixel bicubic 8.12; best 30 min model 2.45 (clean) / 2.55 (with corruption).
- Fixed: `crop_maximal_rectangle` silently used INTER_LINEAR (the cv2 flag was passed as `dst`); it is now antialiased by default, so new benchmark sets differ slightly from the Wan axis set.

### 16. FLUX.2 pretraining (2026-11-23 / 24, 8x RTX 5090)

192M window transformer (run_0016 recipe at width 768, depth 18) with hff (10% dropout to null, CFG), trained for a fixed 24 h. Online pipeline: 1 encoder GPU (compiled FLUX.2 encoder, 19 samples/s, ~3x reuse) and 7 training GPUs. Data: ReLAION : 4KLSDB = 1 : 1, with 4KLSDB streamed from the HF Hub (879 GB, no local copy). Large square crops (>= 50% of the short side, median ~85%). Buckets: 90% 512 -> 1024, 10% 256 -> 512, with one residual table for both. Input corruption: 20% (JPEG quality spread 95 -> 20, mostly light). Schedule set by training time: batch ramp x1 -> x2 (3.6 h) -> x4 (10.8 h) with sqrt LR (peak 2.6e-3), linear decay over the last 30%, weight decay 0.05. Report: `reports/flux_pretrain.md`; code `src/stdev/flux_pretrain/`, launcher `scripts/run_flux_pretrain.py`; checkpoint R2 `checkpoints/flux_pretrain_v1/checkpoint_000129590.pt`.

- Full autoresearch_3 benchmark (1000 Axis images, ground truth hff, 30 steps / shift 2): final raw patch_fid **1.69** / fid 1.93 (EMA 1.89 / 1.95); oracle 0.71, pixel bicubic 8.12, best 30 min research model 2.45.
- The LR decay was essential: patch FID was flat at 2.15-2.18 at constant LR (8.5 -> 14 h) and fell 22% in the decay. This is unlike Wan, where the decay did not help patch FID. After the decay the raw weights beat EMA 0.9999, so use raw.
- The x4 batch step (LR x sqrt) slightly raised fidelity errors (visual low_res_mse +1.5%, weight norm jump) without hurting patch FID. Next run: keep the x2 LR for the x4 step and try a longer decay.
- Stability is fixed by weight decay 0.05 + QK-norm + condition norm: weight norm plateaued (~1365 vs Wan 3162 and growing); max adaLN gate ~7 (Wan 15); all 216 attention heads are active, with no collapse. No memorization at 3x reuse (paired loss on seen vs unseen shards).
- hff has a much stronger effect than in Wan: 0.9 vs 0.1 changes the velocity by 8-10% (Wan 2-4%).

### 17. FLUX.2 single step rectified flow distillation (2026-11-24)

Same sized student (192M, teacher + zero-initialized guidance input, initialized from the teacher's raw weights), trained online: 7 GPUs generate teacher pairs and 1 GPU trains at the same time. The teacher runs 30 steps at shift 2, with CFG uniform in [1, 2] on half of the non-null samples (negative = null hff); hff is true 45% / random 45% / null 10%. Training ran for 20 min (lr 3e-4, decay over the last 30%) on 9.4k samples (7.4k pairs, reuse 1.26), ~35 min of wall time in total. Result: 1-step raw patch_fid **1.76** / fid 2.02 vs teacher 1.69 / 1.93 (30 steps), i.e. ~99% of the gain over bicubic. It was still improving at the end, and raw beats EMA. cfg 2 costs +0.02 patch_fid (its visual effect needs checking in ComfyUI). Report: `reports/flux_rf_distillation.md`; code `src/stdev/flux_distill/`, launcher `scripts/run_flux_distill.py`; checkpoint `distill_runs/flux_distill_v1` (R2 `checkpoints/flux_distill_v1/`).

### 18. FLUX.2 small single step student (20.2M, 2026-11-24)

The Wan small-student recipe at 256 x 14 (head dim 64), trained from scratch for a fixed 6 h. Pairs are generated online by the large 1-step student on 4 GPUs (~30 pairs/s each), while 4 GPUs train with DDP (~177 samples/s, near-linear scaling, global batch 32 @1024, LR 2.1e-3). Pairs were reused 1.48x; 3.79M samples were seen. 4 + 4 was the measured balance point; putting a generator and a trainer on every GPU gave no extra training throughput. Generator input loading must decode only the source latents: full pretraining cache shards are ~80 MB, and decoding them halved generation. Development notes for porting the whole pipeline (pretrain + both distillations) to the next VAE (code map, commands, pitfalls, measured throughputs): `reports/steppe_flux_dev_notes.md`. Result: 1-step patch_fid **2.21** / fid 2.62 (large student 1.76, teacher 1.69, bicubic 8.12), i.e. ~93% of the large student's gain (Wan s20m: ~65%). Model latency is 5.9 ms vs 36.6 ms for the large student (512 -> 1024, compiled, batch 1). It was still improving at the end (-0.03 per 30 min in the decay). Report: `reports/flux_small_student.md`; launcher `scripts/run_flux_small_distill.py`; checkpoint `distill_runs/flux_small_s20m` (R2 `checkpoints/flux_small_s20m/`).

### 19. Unified benchmark of FLUX.2 2x upscaling solutions (2026-11-25)

Same pattern as section 13: the normalized FLUX.2 latent of a 512 image in (no prompt, no hff, no ground truth), the decoded 1024 image out; the 1000 image autoresearch_3 Axis set (`axis_v1_1k_flux2_2x`), patch FID primary, roundtrip MSE / ΔE00, batch 1 latency with decode, peak VRAM, RTX 5090. Ours with raw weights, null hff, guidance 1: base 30 steps 1.72, 200M 1 step 1.75 (172 ms), 20M 2.20 (131 ms); with the tiny TAEF2 decoder 3.22 (60 ms) / 3.81 (19 ms). Others: FlowUpscaler 4.28 (4.45 with TAEF2, 16 ms), SPAN 6.18, SesquiLSR FLUX.2 7.71, bicubic 8.12, Real-ESRGAN 9.03, VOSR 25.9-33.6, latent bicubic 114.
- Ours is the best at every latency above 17 ms (-78% vs bicubic, -59% vs FlowUpscaler). Distillation costs +2%, the 20M student +26%. The full decode (~110 ms) still dominates latency. TAEF2 costs only ~+1.5 patch FID (Wan tiny VAE: +78%), and the 20M + TAEF2 pipeline beats every third party solution at any latency.
- VOSR (4x real world restorer) fails at 2x on clean inputs: it repaints fine detail. 256 -> 4x looks cleaner but scores worse. Verified with the official script.
- Hildegard (klein 9B fp8 + LoRA, ComfyUI, constant README prompt) was measured on 30 images only, without FID (~7 s per image, 11.9 GB). With the official reference weights (0.9 / 0.8 / 0.8) it redraws subjects (ΔE00 6.6). With weights 1.0 it is aligned and detailed, but ΔE00 3.8 vs 2.2 for ours.
- Report: `reports/flux_upscaling_benchmark.md`. Code: `scripts/run_flux_upscale_benchmark.py`, `src/stdev/flux_upscale_benchmark/`, `scripts/run_hildegard_benchmark.py` (ComfyUI venv), `src/stdev/vaes/tiny_flux_2_vae.py`. Results: `raw_results/flux_upscale_benchmark_axis_v1_1k/`.

### 20. Public release "Steppe-Flux.2" (2026-11-26)

- Steppe-ComfyUI was updated to support new models
- New weights were uploaded to [Steppe-Flux.2](https://huggingface.co/TensorForger/Steppe-Flux.2)

New insights from release and testing: 

- Community asks for **Minimax-H3** VAE support.
- There was also a request to lower VRAM requirements.
- From testing: requires more anime and vector images in dataset.

### 21. Autoresearch on Qwen-Image 2.1 latents (phase 4: 512 -> 1024, 2026-11-26 / 27): VAE marked problematic

Setup: 8 rounds on 4 RTX 5090 (run_0000 - run_0030, mostly 30 min, last round 1 h), 40k 4KLSDB images, fixed Axis-V1 1k benchmark decoded with the texture-fix decoder (the official one roughly doubles the oracle patch FID: 2.44 vs 1.07). New research setup: single dataset, no hff, no corruption, ~50M model, single resolution. Log and final summary: `autoresearch_4/experiment_logs.md` ("FINAL SUMMARY"); code `src/stdev/qwen_upscalers/`, `src/stdev/qwen_benchmark/`, `qwen_infra/`, data `scripts/encode_qwen_datasets.py`.

- The VAE is Qwen-Image-VAE-2.0 f16c64 (arXiv 2605.13565): 64 channels, 16x, no KL, DINOv2 aligned latents, decoder without GAN. A token covers 16x16 px, so the "high" latent band holds coarse, input-determined content, and the decoder turns any small off-manifold latent error into colour blotches (10% relative noise: 29.7 dB vs FLUX.2 40.2 / Wan 33.3). Latent L2 is a poor proxy: a re-encoded bicubic image is further away in L2 than the models but decodes far better.
- Best latent-only result: patch FID 5.75 (97M, 1 h; bicubic 8.69), but FID 4.4 vs bicubic 2.2 and worse colour. It is a hard plateau: neither 4x tokens, nor 6x parameters, nor 2x time move it (flow loss and latent MSE keep improving, decoded quality does not). Decoder-in-the-loop losses fail in every form (implementation verified).
- Transferable findings: per-channel shift + scale residual normalization, a frequency weighted flow loss (velocity error above half Nyquist x16), sampling shift 3-4, RoPE + mask-free shifted windows (+5% speed, slightly better), whole-model compile with max-autotune (+14% training speed).
- Precision: FP8 training keeps quality but gives only +1-5% speed (GEMMs are ~36% of a step); torchao MXFP8 / NVFP4 training does not run on this stack. FP8 post-training quantization is lossless and -22% latency.
- A factorization diagnosis (decode -> bicubic -> encode base + generated delta) reaches 4.71, but it needs the VAE round trip at inference and was rejected as a recipe. Next pretrain uses another VAE; ideas for a revisit are in the log.

### 22. Ming-Image 0.2 Design VAE check (2026-11-27)

Quick check of the new RGBA VAE of Ming-Image-0.2-Design (released 2026-11-25), whether it shares the Qwen-Image 2.1 problems. It is byte-identical to the Qwen-Image-Layered RGBA VAE (Wan 2.1 / Qwen-Image 1 architecture, 8x, 16 channels), but its latent space differs from Wan's (raw std ~0.116, per channel linear R2 0.80 to Wan). Verdict: not likely problematic. Robust decoder (10% latent noise 39.5 dB vs Qwen-Image 2.1 29.8, FLUX.2 40.6), band tests like Wan / FLUX.2. The 2x task is detail heavy like FLUX.2 (the bilinear source explains 21% of the target variance; FLUX.2 41%, Wan 85%). KL use cannot be determined (unconstrained log-variance head), but Wan 2.1 is effectively KL-free too, so this is not the deciding factor. Report: `reports/ming_image_vae_check.md`; wrapper `src/stdev/vaes/ming_image_vae.py` (own per channel statistics; the Qwen-Image-Layered config's Wan statistics do not fit). One 30 min round (`autoresearch_5/experiment_logs.md`, 40k 4KLSDB, Axis 1k): the VAE round trip is lossy on photos (oracle patch FID 3.57 vs FLUX.2 0.71; pixel bicubic 10.58). The FLUX.2 recipe transfers as is (6.92); with the autoresearch_4 package (mean centered residual, RoPE / mask-free windows, full compile) and initial noise temperature 0.9 the best is 6.34 / fid 3.25, ~60% of the bicubic -> oracle gap (FLUX.2 77%, Qwen-Image 2.1 36%). Residual vs direct: no difference; the frequency weighted loss does not help on this VAE.

### 23. Autoresearch 6: MiniMax-H3-Image latents -> FLUX.2 latents (2026-11-27, 4x RTX 5090)

Experimental side project: a generative "VAE translator" (no pixel upscaling) that turns the H3-Image latent (16x, 24 channels; its ViT decoder loses small faces and text, the ostris fine-tune oversmooths) into the FLUX.2 latent of the same image, decoded by a FLUX.2 decoder. Viability tiers set by the user: beat the native H3 decoder on latency, VRAM, fid and patch_fid (fully viable), or on quality only (still useful). Setup, references and log: `autoresearch_6/instruction.md`, `autoresearch_6/experiment_logs.md`; code `src/stdev/h3_translators/`, `src/stdev/h3_benchmark/`, `h3_infra/`, data `scripts/encode_h3_datasets.py` (40.9k 4KLSDB 1024 crops), wrapper `src/stdev/vaes/minimax_h3_image_vae.py`.

- The H3 decoder must run tiled (released 256 px tiles; untiled decode collapses). Its diffusers decode is launch bound; batching the tiles gives the fair baseline: 40 / 207 / 763 ms at 1024 / 2048 / 4096, ~1-1.5 GB. Axis 1k: H3 patch_fid 2.47 / fid 4.37, ostris 2.92 / 2.85, FLUX.2 oracle 0.71 / 0.72, TAEF2 oracle 2.25 / 3.62.
- `black-forest-labs/FLUX.2-small-decoder` shares the FLUX.2 encoder and latent space (equal after bf16 rounding): same quality as the full decoder (oracle 0.68 / 0.78) at -28% latency and -26% VRAM. It is the quality decoder of this project.
- An H3 latent of a 1024 image has the 64x64 grid of the Steppe-Flux.2 input, so the Steppe model transfers with only the source input conv replaced. A linear 3x3 H3 -> FLUX.2 map explains 36% of the target variance (Steppe's bilinear source: 41%) and serves as the residual base.
- Two rounds (8 runs): the direct approach learns; Steppe-Flux.2 200M init is best: 60 min, patch_fid 1.84 / fid 4.10 vs H3 2.47 / 4.37, still improving; a 47.5M scratch model reaches 2.03 / 4.26 in 60 min. Remaining gap: low-frequency colour drift (PSNR 29.46 vs 29.78, ΔE00 2.49 vs 2.29), which shrinks with training time. A low-frequency weighted loss, a lower sampling temperature, and H3 decoder features as conditioning do not help; training sigma shift 2 helps slightly. TAEF2 decoding still loses to H3.
- Verdict so far: "still useful" (better patch_fid / fid with the small decoder, but 1024 end to end ~77-103 ms and ~2.6 GB vs 40 ms / ~1 GB for batched H3). Moved to a full 8 GPU training run on another machine.

### 24. H3 -> FLUX.2 translator pretraining and distillation (2026-11-28, 8x RTX 5090)

192M run_0006 translator (Steppe-Flux.2 base init: 308 / 309 tensors incl. the hff projection; linear 3x3 residual base, training sigma shift 2) trained with the FLUX.2 pretrain recipe: online ReLAION : 4KLSDB = 1 : 1, buckets 90% 1024 / 10% 512 px, hff with 10% dropout, 20% corruption of the H3 encoder input (JPEG / other resize filters; target clean). 6 training + 2 encoder GPUs (~19 samples/s, reuse ~2.2). The H3 encoder was made 4.7x faster exactly: all 256 px tiles in one compiled call, and its causal 3D convolutions run as 2D convolutions on a single frame (zero causal padding: only the last temporal slice touches data); bf16 (1.7% latent error). Untiled H3 encoding is not usable (30% different latents). Open ended schedule: peak LR 6e-4, batch x2 at 1.5 h (LR held), manual decay via `control/decay` (hours): 3.66 h constant + 2 h decay = 5.66 h. Report: `reports/h3_translator_pretrain.md`; code `src/stdev/h3_pretrain/`, `src/stdev/h3_distill/`, launchers `scripts/run_h3_pretrain.py`, `scripts/run_h3_distill.py`, full benchmark on all GPUs `scripts/run_h3_pretrain_benchmark.py`; checkpoints R2 `checkpoints/h3_translator_v1/`, `checkpoints/h3_distill_v1/`, `checkpoints/h3_distill_v2/` (linear base and residual statistics stored as buffers).

- Axis 1k, FLUX.2 small decoder, 30 steps, null hff: patch_fid **1.57** / fid **2.92** / PSNR 30.34 / ΔE00 2.13 vs native H3 2.47 / 4.37 / 29.78 / 2.29 and autoresearch_6 best 1.84 / 4.10. **Beats the H3 decoder on every quality metric**; the colour / fidelity gap of autoresearch_6 is closed. As in the FLUX.2 pretrain, metrics plateaued at constant LR (~2 h) and the decay gave most of the late gain; raw ≈ EMA (use raw). Ground truth hff barely matters (-0.02): null hff is fine for generated latents.
- 1-step distillation (FLUX.2 recipe, 7 generator GPUs + 1 trainer, ~42 min): patch_fid **1.67** / fid 3.17, PSNR / ΔE00 equal to the teacher (+6% / +9% vs teacher, ~89% of the gain over H3). A 60 min run with 80% decay did not improve it (1.73 / 3.15; single evaluation patch_fid noise ~0.1-0.3): the remaining gap is not a training length problem.
- Viability unchanged: "still useful" (better quality; 1024 end to end ~103 ms and ~2.6 GB vs 40 ms / ~1 GB for batched H3 tiles). TAEF2 decoding still loses to H3 (3.06 / 5.17).

## Open problems

1. Flux.2 tested at scale as the direct hr latent generator. Autoresearch 2 found that Wan 2.1 VAE requires generation of residuals. So, is there a single optimal approach for all VAEs? Or universal abstraction that covers both modes? Need to find out if recipe advantages can be transfered across VAEs.
2. Small important details (text, background faces) are where upscalers fail. Hypothesis: they are rare per pixel area in high resolution training images, so their prior is undertrained. See `autoresearch_2/experiment_logs.md` ("OPEN QUESTION: training scales").
3. How to make training faster and train better model with same compute budget? Low-precision training? (Partly answered in section 21: FP8 keeps quality but gives only +1-5% at these sizes; compile and kernel-friendly attention give more.)

## Plan

1. Separate model per VAE, each trained from scratch. 
   - Implement VAE upscalers in optimal order (oreder can be slightly changed, reference is `reports/prioritized_vaes_by_demand_qwen.md`)
   - Reuse the same architecture and recipe for all VAEs, only adapting channel counts and video handling, if that is possible (open problem 1)
2. Pretrain the new base models from scratch:
   - more training than before
   - autoresearch 2 recipe
