# FlowUpscaler architecture vs training recipe, and comparison with external latent upscalers

## Summary

Three questions were answered with the unmodified `scripts/run_benchmark.py` (`flux_1024`, all 1,000 images,
bf16, seed 0; `axis_1024`, all 1,000 real images, for the models marked below).

1. **External models.** The released FlowUpscaler (59.3M, 1 step) is clearly worse than the small
   student of `reports/small_single_step_distillation.md` (patch FID 1.873 vs 1.351) but 1.4x faster
   and uses 0.6x of its peak memory. SesquiLSR (~3M, deterministic regression) is the fastest and
   smallest model, but its patch FID is 6.05, 3x worse than FlowUpscaler.
2. **Architecture or recipe?** The FlowUpscaler architecture, trained from random init with the
   recipe of the small student (single step distillation from the SFT teacher, 42 epochs), reaches
   patch FID **1.451**, a **22.5% improvement** over the released weights (1.873), without any
   change in the network (same state dict, same latency). Most of the previous gap to the small
   student was therefore the training recipe. But the remaining gap is the architecture: after
   2.3x more epochs it is still 7.4% behind the small student on patch FID (1.451 vs 1.351), and
   it never reaches the small student's patch FID of epoch 14 (1.446) before the learning rate decay.
3. **Why does it behave like a smaller model?** Of its 59.3M parameters only 18.9M are
   convolutions at full resolution, and its full resolution blocks have no MLP or attention. The
   small student does 1.9x more full resolution compute (1,115 vs 596 GFLOPs), and 21% of the
   FlowUpscaler parameters compute a constant when the model is used at a single timestep. See [Why it behaves like a smaller model](#why-the-flowupscaler-architecture-behaves-like-a-smaller-model).
   The mechanism is an interpretation of measured facts, not a controlled ablation.

The retrained weights ("FlowUpscaler 2.0", `distill_runs/flow_arch_v1/export/student_ema.safetensors`)
also improve on the out-of-domain real image benchmark `axis_1024` (patch FID 5.661 to 4.618, FID
4.886 to 3.560).

## Models

| Name | Params | Steps | Description |
|---|---:|---:|---|
| SesquiLSR | ~3M | 1 (deterministic) | `src/stdev/upscalers/sesqui_upscaler.py`, external pretrained regression upscaler (Flux2 weights); no noise, no text. Raw VAE latents via the batch norm adaptor. |
| FlowUpscaler (released) | 59.3M | 1 | `src/stdev/upscalers/flow_upscaler.py`, external pretrained weights, timestep 1000, single Euler step as in the ComfyUI node. |
| FlowUpscaler 2.0 (`flow_arch_v1`) | 59.3M | 1 | Same architecture, random init, trained with `scripts/run_small_distill.py --architecture flow` (recipe below). |
| Small student (`small_v1`) | 48.6M | 1 | `src/stdev/upscalers/distill_small.py`, residual + window attention + MLP blocks, trained for 18 epochs. |
| Large 1-step student, SFT teacher, Stage-2 pretrained | 450.9M | 1, 20, 20 | References from the existing reports. |

Recipe of the small student and of FlowUpscaler 2.0 (identical): rectified flow distillation pairs of
the SFT model (13,200 pairs, `train_datasets/flux_sft_distill_pairs`), input is the teacher noise
(pure noise, t = 1) and the low resolution latents, target is the velocity `noise - teacher_output`,
batch 16, AdamW (0.9, 0.95), LR 4e-4 constant after 100 warmup steps, gradient clip 1, EMA 0.999,
bf16 autocast, linear decay to zero in the last epoch. FlowUpscaler 2.0: 41 epochs at constant LR
plus 1 decay epoch (34,650 steps, 554,400 samples, 6,881 s of training on one GPU). Small student:
17 + 1 epochs (14,850 steps, 237,600 samples, 5,710 s). Benchmarked weights are the EMA weights.

## Benchmark results

### `flux_1024` (in-domain teacher generations, lower is better)

| Model | Params | Steps | Patch FID | FID | HR image MSE | LR image MSE | HR latent MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SesquiLSR | ~3M | 1 | 6.0545 | 5.0730 | 0.003534 | 0.001567 | 0.5317 |
| FlowUpscaler (released) | 59.3M | 1 | 1.8725 | 2.7465 | 0.003173 | 0.001384 | 0.3883 |
| **FlowUpscaler 2.0** | 59.3M | 1 | **1.4511** | **2.1723** | **0.002254** | **0.000873** | **0.3220** |
| Small student | 48.6M | 1 | 1.3514 | 1.9871 | 0.002216 | 0.000853 | 0.3200 |
| Large 1-step student | 450.9M | 1 | 0.8413 | 1.0402 | 0.001865 | 0.000640 | 0.3039 |
| SFT EMA (teacher) | 450.9M | 20 | 0.8241 | 0.9488 | 0.001841 | 0.000618 | 0.3044 |
| Stage-2 pretrained EMA | 450.9M | 20 | 1.9031 | 1.4535 | 0.002163 | 0.000700 | 0.2935 |

Change of FlowUpscaler 2.0 relative to the released FlowUpscaler: patch FID -22.5%, FID -20.9%, HR
image MSE -29.0%, LR image MSE -36.9%, HR latent MSE -17.1%. Relative to the small student:
patch FID +7.4%, FID +9.3%, HR image MSE +1.7%, LR image MSE +2.3%, HR latent MSE +0.6%.

### `axis_1024` (real images, out-of-domain; only models run on it)

| Model | Patch FID | FID | HR image MSE | LR image MSE | HR latent MSE |
|---|---:|---:|---:|---:|---:|
| SesquiLSR | 8.0627 | 7.8631 | 0.003636 | 0.001441 | 0.5109 |
| FlowUpscaler (released) | 5.6608 | 4.8864 | 0.003900 | 0.001585 | 0.4745 |
| **FlowUpscaler 2.0** | **4.6179** | **3.5602** | **0.002965** | **0.001020** | **0.4027** |

FlowUpscaler 2.0 vs released: patch FID -18.4%, FID -27.1%, HR image MSE -24.0%, LR image MSE
-35.6%, HR latent MSE -15.1%. The improvement carries over to real images although the
distillation pairs are FLUX generations. The small student and the large models were not run on
`axis_1024`.

### Efficiency (compiled single forward, ms; peak memory of the eager model, MB)

| Model | Forward 512 | Forward 1024 | Forward 2048 | Peak memory 512 / 1024 / 2048 |
|---|---:|---:|---:|---|
| SesquiLSR | **0.62** | **1.23** | **4.08** | **21 / 64 / 237** |
| FlowUpscaler (released) | 2.10 | 6.23 | 22.41 | 155 / 265 / 700 |
| FlowUpscaler 2.0 | 2.12 | 6.34 | 22.09 | 155 / 265 / 700 |
| Small student | 2.91 | 8.50 | 34.38 | 183 / 452 / 1,528 |
| Large 1-step student | 12.24 | 42.65 | 219.44 | 997 / 1,407 / 3,045 |
| SFT EMA (teacher), 20 steps | 11.81 | 41.90 | 216.14 | 997 / 1,408 / 3,049 |

The two FlowUpscaler rows differ by at most 2% (measurement noise; the architecture is identical).
SesquiLSR's forward excludes the batch norm conversion and patchify steps of its `upscale`.

## Learning curves: FlowUpscaler architecture vs small student

Same recipe, EMA weights benchmarked after each epoch on `flux_1024`. Epochs up to 17 (small) and 41
(flow) are constant LR; epoch 18 of the small student and 42 of the flow model are the decay epoch.

| Epoch | Train loss, flow | Train loss, small | Patch FID, flow | Patch FID, small | FID, flow | FID, small |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2893 | 0.3219 | 10.016 | 10.161 | 13.133 | 13.636 |
| 2 | 0.1518 | 0.1576 | 4.298 | 4.438 | 4.508 | 6.054 |
| 3 | 0.1279 | 0.1266 | 3.384 | 3.098 | 3.889 | 4.289 |
| 5 | 0.1078 | 0.0949 | 2.701 | 2.095 | 3.379 | 2.907 |
| 8 | 0.0934 | 0.0775 | 2.270 | 1.705 | 2.995 | 2.430 |
| 10 | 0.0868 | 0.0710 | 2.093 | 1.583 | 2.833 | 2.290 |
| 14 | 0.0783 | 0.0628 | 1.849 | 1.446 | 2.611 | 2.132 |
| 17 | 0.0739 | 0.0586 | 1.738 | 1.383 | 2.527 | 2.039 |
| 18 | 0.0728 | 0.0535 (decay) | 1.701 | **1.351** (decay) | 2.473 | **1.987** (decay) |
| 20 | 0.0706 | - | 1.663 | - | 2.440 | - |
| 25 | 0.0667 | - | 1.588 | - | 2.369 | - |
| 30 | 0.0638 | - | 1.535 | - | 2.297 | - |
| 35 | 0.0616 | - | 1.497 | - | 2.248 | - |
| 41 | 0.0594 | - | 1.470 | - | 2.208 | - |
| 42 (decay) | 0.0559 | - | **1.451** | - | **2.172** | - |

Observations:

- **Faster start.** The flow architecture is ahead in epochs 1 and 2 (train loss 0.289 vs 0.322 and
  0.152 vs 0.158; patch FID 10.0 vs 10.2 and 4.30 vs 4.44) and falls behind from epoch 3 on.
- **Earlier plateau.** Patch FID gain per epoch of the flow model: -0.043 (epochs 10 to 20),
  -0.013 (20 to 30), -0.006 (30 to 41). The small student was still at -0.021 per epoch at epochs
  14 to 17. The flow model needs 10 epochs for the patch FID the small student has at epoch 5
  (2.09), about 25 epochs for its epoch 10 (1.58), and does not reach its epoch 14 value (1.446)
  within the constant LR phase (1.470 at epoch 41); only the decay epoch gets there (1.451).
- **Wall clock.** An epoch of the flow model takes 164 s, the small student's 317 s (1.9x), so the
  comparison in training time is closer than in epochs but still favours the small student: at
  the small student's 5,710 s the flow model is at epoch 35 (patch FID 1.497, no decay) vs 1.351.
- **Train loss is not a common scale for the two models.** At a similar training loss of about
  0.060 the small student (epoch 16, 0.0600) has patch FID 1.402 and the flow model (epoch 41,
  0.0594) has 1.470. The flow model also trains with dropout 0.01, which slightly raises its
  logged loss, so the difference in loss per unit of patch FID is, if anything, understated. The
  flow model's remaining error is relatively more in the detail statistics measured by patch FID
  than in the pixel MSE: the final gaps to the small student are +7.4% (patch FID) vs +1.7% (HR
  image MSE).

## Why the FlowUpscaler architecture behaves like a smaller model

Parameter count (59.3M vs 48.6M) is a poor measure of capacity for this task. Measured with
`dev/compare_model_capacity.py` and `dev/count_level_parameters.py` (forward FLOPs for a 1024 px
image, 128x128 latents; one FLOP counted per multiply and per add):

### 1. A fifth of the parameters computes a constant

FlowUpscaler is conditioned on the timestep through a sinusoidal embedding MLP and a linear
projection (scale and shift) in every residual block. In single step distillation, and in the
single Euler step of the released ComfyUI inference, the timestep is always 1000, so the whole
timestep path is a function of a constant. It is equivalent to one learned scale and shift vector
per block (a few thousand parameters) and cannot carry image-dependent information.

| Parameter group of FlowUpscaler | Params | Share |
|---|---:|---:|
| Main UNet convolutions (levels H, H/2, H/4) | 35.15M | 59.3% |
| Low resolution encoder convolutions | 10.64M | 17.9% |
| Timestep projections in the main UNet blocks | 8.66M | 14.6% |
| Timestep projections in the low resolution encoder | 2.36M | 4.0% |
| Timestep embedding MLP | 1.57M | 2.7% |
| FiLM projections of the low resolution features | 0.89M | 1.5% |
| **Total** | **59.27M** | |

Timestep related parameters: 12.59M (21.2%). Without them the model has 46.7M parameters that
process content, fewer than the 48.6M of the small student, which has no timestep input. The
released weights were presumably trained on a range of timesteps (the interface is a flow matching
model taking a noisy sample), where this machinery was useful, but at deployment, and in this
distillation, it is idle.

### 2. The detail is made at full resolution, where FlowUpscaler is thin

| Quantity | FlowUpscaler | Small student |
|---|---:|---:|
| Forward GFLOPs at 1024 px | 739.8 | 1,185.1 |
| GFLOPs at full resolution (share) | 596.4 (81%) | 1,115.0 (94%) |
| Block layout at full resolution | 3 + 3 residual blocks, width 384 | 1 + 3 residual blocks with attention and MLP, width 512 |
| GFLOPs of one full resolution block | 87.0 (convolutions only) | 259.8 (residual + 8x8 window attention + MLP) |
| Parameters at full resolution | 23.6M, of which 4.7M timestep projections | 33.1M, of which 12.6M attention and MLP |
| Residual convolution parameters at full resolution | 18.9M | 20.5M |

The full resolution convolution capacity of the two models is similar (18.9M vs 20.5M parameters,
596 vs about 670 GFLOPs). What FlowUpscaler lacks is the 12.6M parameters and about 445 GFLOPs of
window attention and per pixel MLP at full resolution, which the small student spends on top of
the convolutions. The remaining FlowUpscaler parameters sit in places that do little for detail
generation: the H/2 and H/4 levels (8.2M + 8.5M + 3.4M including 3.9M timestep projections,
15% of FLOPs), and the low resolution encoder (10.6M parameters but only 32 GFLOPs, 4%, because it
runs on the low resolution latents).

Consistent with this, the autoresearch (`autoresearch/experiment_logs.md`, round 10) found that a
wide full resolution level (width >= 512) and a decoder heavy layout carry quality, while wide or
deep lower levels do not pay off. The small student was designed from that finding; FlowUpscaler
spends 384 channels uniformly on all three levels.

### 3. How the low resolution input enters

FlowUpscaler never concatenates the low resolution latents with the sample. They are encoded
separately and enter through FiLM (scale and shift from a 1x1 convolution) at three points of the
encoder only, before the first full resolution stage and before the two lower stages, after
bilinear interpolation of the features. The decoder sees them only through the skip connections.
In the small student the bilinearly upsampled low resolution latents are concatenated with the
noise at the input and reach every layer through the residual stream. The target here is
`noise - hr_latents`: the model must reproduce the noise and subtract the image, so a model whose
only route for the low resolution signal is a modulation of features computed from noise is
expected to be at a disadvantage for fine, low resolution aligned structure. This part is the
least tested of the explanations.

### How this explains the learning curves

The fast start and the early plateau fit a model whose capacity is dominated by simple, local,
convolutional structure. Early loss reduction comes from the coarse part of the mapping (the
low frequency content, local sharpening, learnable by 3x3 convolutions at any width) and the flow
model gets there at least as fast as the small student, which starts from zero initialized
blocks. The late phase of the small student is driven by the fine detail that nonlinear
per pixel MLPs and window attention at full resolution can add; the flow model lacks this
capacity, so its per epoch gains vanish earlier (see above) and the gap shows up mostly in patch
FID, the detail metric, and much less in MSE. Both the capacity measurements and the metric pattern
support this, but the explanation was not isolated experimentally.

### Experiments that would test it (not run)

- Remove the timestep path of FlowUpscaler (replace it with learned constants). Prediction: same
  metrics with 12.6M fewer parameters. If so, FlowUpscaler could be a 46.7M model at no cost.
- Add a window attention and MLP to the full resolution blocks of FlowUpscaler. Prediction: patch
  FID moves toward the small student's.
- Concatenate the upsampled low resolution latents to the input of FlowUpscaler. Prediction: the
  early speed is kept and the gap in the MSE metrics closes.
- Spend the 12.6M timestep parameters as width 512 at full resolution (equal parameter count).
  Prediction: better than the original layout, since compute moves to where detail is made.

## Scope and interpretation

- Single run and single seed per model; the FlowUpscaler 2.0 benchmark reproduces the
  per-epoch benchmark of epoch 42 exactly on `flux_1024` (patch FID 1.451067), so the metric itself is
  deterministic. The measured seed noise of patch FID in the autoresearch (about 0.03 for the
  training seed, 0.05 for the inference seed) is 2 to 3 times smaller than the 0.100 gap between
  the flow architecture and the small student.
- The two training runs differ in length (42 vs 18 epochs); a matched epoch comparison is in the
  learning curve table, a matched training time comparison above.
- FlowUpscaler 2.0 was trained for a single timestep (t = 1000), with the same interface as the
  ComfyUI node uses (one Euler step, `noise - velocity`). Multi-step or multi-timestep use of
  the new weights was not evaluated and is not expected to work. The state dict has the same keys
  as the released model (loaded strictly by `FlowUpscalerUNet`, a subclass without extra
  parameters); the ComfyUI node itself was not run with the new weights.
- All distillation pairs come from the SFT teacher, which was pretrained on much more data than the
  released FlowUpscaler saw. The recipe change therefore includes the teacher and pair source, not
  only the optimisation; this was not isolated. `axis_1024` evidence suggests the gain is not
  specific to FLUX generations.
- SesquiLSR was designed for arbitrary scales and perceptual (LPIPS) quality on images; it is
  benchmarked here only at 2x on latent statistics (patch FID, FID, MSE), which is the benchmark
  of this repository and not the metric it targets.
- The small student and the large models have no `axis_1024` results.
- No visual comparison of output images is part of this report.

## Artifacts and reproduction

- Wrappers: `src/stdev/upscalers/flow_upscaler.py` (FlowUpscaler, `FlowUpscalerUNet`),
  `src/stdev/upscalers/sesqui_upscaler.py` (SesquiLSR). Training: `scripts/run_small_distill.py`
  with `--architecture flow`.
- Run: `distill_runs/flow_arch_v1/` (EMA and raw weights in `export/`, tensorboard in `tensorboard/`);
  per-epoch benchmark results `raw_results/flow_arch_v1_epoch_NNN/`; final two dataset benchmark
  `raw_results/flow_arch_v1_final/`; released FlowUpscaler `raw_results/flow_upscaler/`; SesquiLSR
  `raw_results/sesqui_lsr/`.
- Analysis scripts: `dev/compare_flow_upscaler.py` (results table), `dev/compare_model_capacity.py`
  (parameters and FLOPs), `dev/count_level_parameters.py` (parameters by level),
  `dev/summarize_small_run.py --run-name flow_arch_v1` (per-epoch table).

```bash
python scripts/run_small_distill.py --run-name flow_arch_v1 --architecture flow
touch distill_runs/flow_arch_v1/control/decay   # after epoch 41
FLOW_UPSCALER_WEIGHTS=distill_runs/flow_arch_v1/export/student_ema.safetensors \
FLOW_UPSCALER_NAME=flow_arch_v1_final \
python scripts/run_benchmark.py --datasets benchmark_datasets/flux_1024 benchmark_datasets/axis_1024 \
  --upscaler src/stdev/upscalers/flow_upscaler.py
```
