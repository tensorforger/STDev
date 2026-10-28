# Experiment log

## Reference baseline

Run 0000 uses a 106.61M-parameter attention-free convolutional UNet, 20-step flow matching, and 13,096 training steps. It reached Axis FID 7.9988 and FLUX FID 5.0951, with 3.7709 ms average single-forward latency and 299.46 MB peak inference memory at 1024 px. Training loss was still decreasing at the end of the run.

## Round 1

Hypothesis: ramped EMA should improve sample quality by smoothing short-run optimization noise without changing architecture, inference cost, or the training objective. All other settings remain identical to run 0000 so the comparison isolates EMA. The decay is `min(0.9999, (1 + step) / (10 + step))`, and only EMA weights are saved for benchmarking.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0001 | Ramped weight EMA | 0000 | Test whether weight averaging improves FID at zero inference cost |

Result: EMA improved Axis FID from 7.9988 to 7.8501 and FLUX FID from 5.0951 to 4.8858. The 0.15–0.21 FID gains are directionally consistent but below the single-seed noise threshold. Latency and memory were unchanged. Training throughput was effectively unchanged at 13,077 steps. Keep EMA as the default parent, while treating its quality benefit as provisional.

## Round 2

Hypothesis: constant learning rate leaves avoidable optimization noise at the end of a short fixed-budget run. Cosine decay to zero should produce a better final optimum and complement EMA without affecting inference. Keep the peak learning rate, warmup, model, data, and EMA schedule fixed. Use 13,000 expected steps based on runs 0000–0001 and clamp the schedule at zero for any additional steps.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0002 | Cosine learning-rate decay | 0001 | Test whether end-of-run LR decay improves the final optimum |

Result: cosine decay substantially regressed to Axis FID 9.7816 and FLUX FID 6.2269. Late-window training loss also worsened relative to run 0001 (about 0.39 versus 0.37), despite the same architecture and 13,091 steps. The reduced learning-rate area leaves the model less optimized within the fixed budget; reject this schedule and do not use it as a parent.

## Round 3

Hypothesis: the still-decreasing loss and cosine-schedule regression indicate under-optimization rather than excessive final optimization noise. Doubling the constant learning rate from 2e-4 to 4e-4 may reach a better solution within the same number of updates. Branch from run 0001 and retain its warmup, gradient clipping, EMA, and all other settings.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0003 | Double constant learning rate | 0001 | Test whether faster optimization improves quality within the time budget |

Result: 4e-4 is a clear improvement, reaching Axis FID 6.5060 and FLUX FID 4.0782. This improves both datasets by more than the 0.5 FID noise threshold over run 0001, with unchanged inference latency and memory. Early and late training loss improved, ending near 0.355–0.364 at 13,143 steps. Run 0003 is the new quality parent.

## Round 4

Hypothesis: the strong gain from doubling learning rate and continuing late loss improvement suggest the optimum may be higher still. Test 8e-4 with the same warmup, clipping, and EMA to locate the edge of the useful learning-rate range before spending runs on architecture changes.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0004 | Quadruple baseline learning rate | 0003 | Test 8e-4 and locate the learning-rate optimum |

Result: 8e-4 reached Axis FID 6.4592 and FLUX FID 3.9148. The 0.05 Axis and 0.16 FLUX gains over 4e-4 are below the noise threshold, while early optimization was somewhat worse and late loss was similar. Treat 4e-4–8e-4 as a plateau and stop the LR sweep. Run 0004 remains the nominal best parent, with run 0003 as an equivalent alternative.

### Sampler-shift small experiment

The training and inference schedules both use shift 3. Reuse run 0004 weights and evaluate inference shifts 1, 2, and 4 without retraining to determine whether the sampling trajectory is well matched before choosing the next training direction. Compare against the official shift-3 metrics; keep efficiency unchanged because this only alters sampler sigmas.

Result: quality improves monotonically over the tested range. Axis FID is 6.8881 / 6.7277 / 6.2724 for shifts 1 / 2 / 4, versus 6.4592 for shift 3. FLUX FID is 4.2545 / 4.1269 / 3.7881, versus 3.9148 for shift 3. Shift 4 also improves all image and latent MSE metrics over shift 3. The gains are below 0.5 FID but consistent across both datasets. Use shift-4 inference on run 0004 as the control for a matched training-schedule experiment.

## Round 5

Hypothesis: shift 4 gives a better inference trajectory for the current model. Training with the same shift may align noise weighting with that trajectory and improve further. Change the shared sigma shift from 3 to 4, retaining run 0004's architecture, 8e-4 learning rate, and EMA. Compare both with official run 0004 and its inference-only shift-4 control.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0005 | Matched sigma shift 4 | 0004 | Align training noise weighting with the better sampling trajectory |

Result: matched shift-4 training reached Axis FID 6.8157 and FLUX FID 4.1073, worse than run 0004 at both shift 3 and inference-only shift 4. Training loss fell to about 0.31, but it is not comparable across different sigma distributions. Reject matched shift-4 training. Decouple training shift 3 from inference shift 4 in future runs.

## Round 6

Hypothesis: 1,000 samples and roughly 210 epochs per run create a memorization risk. Random horizontal flips of paired HR/LR latents provide valid spatial augmentation and may improve generalization at zero inference cost. Branch from run 0004, keep training shift 3, adopt the validated inference shift 4, and add independent per-sample horizontal flips with probability 0.5. The inference-only shift-4 control isolates the augmentation effect.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0006 | Horizontal flip augmentation | 0004 | Test paired latent flipping for data-limited generalization |

Result: horizontal flips regressed to Axis FID 6.8868 and FLUX FID 4.2347, compared with 6.2724 / 3.7881 for the same run 0004 weights using inference shift 4. Training loss was higher, as expected for the harder augmented distribution, but benchmark quality did not benefit. Reject this augmentation.

## Round 7

Hypothesis: channel-concatenating LR latents only at the input may dilute conditioning inside the network. A separate LR encoder can extract features at the native LR grid and a 2x-downsampled grid, then inject them into every UNet block using spatial scale, shift, and residual gates. Zero-initialized gates preserve an identity residual path at initialization; the previous zero initialization of the second convolution must be removed to avoid a zero-gradient deadlock. Retain training shift 3, inference shift 4, 8e-4 LR, and EMA. No augmentation.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0007 | Multi-scale zero-gated LR conditioning | 0004 | Inject a separate LR encoder into every UNet residual block |

Result: run 0007 reached Axis FID 6.9019 and FLUX FID 4.0920, worse than the run 0004 shift-4 control at 6.2724 / 3.7881. Latency increased to 3.9569 ms at 1024 px, and throughput dropped to 11,385 steps. Training MSE improved to about 0.32 despite fewer updates, showing that lower velocity MSE did not translate into better multi-step FID for this branch. Reject the added conditioning path rather than inferring that the idea failed from loss alone.

## Round 8

Hypothesis: at high noise levels, velocity direction may matter more for the sampled image than its magnitude, which dominates MSE. Add a cosine-direction loss with weight 0.1 to the existing velocity MSE. Restore the plain run 0004 backbone with training shift 3 and inference shift 4; no LR-conditioning encoder or data augmentation. Inference is unchanged.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0008 | Cosine velocity direction loss | 0004 | Test whether directional supervision improves multi-step sample quality |

Result: run 0008 reached Axis FID 6.2390 and FLUX FID 3.7519 with unchanged inference cost. This is only 0.033 / 0.036 FID better than the run 0004 shift-4 control, so the incremental cosine-loss effect is below noise. Retain it provisionally as the current best official run, but do not consider the 0.1 weight validated.

## Round 9

Hypothesis: LR latents provide local appearance but may not preserve all global semantics. SDXL prompt embeddings can supply complementary style and semantic conditioning through cross-attention at the 32x32 bottleneck. Each image token attends to only 77 fixed text tokens, so complexity is linear in image area. Zero-initialize the attention output projection to preserve the initial model. Branch from run 0008 and keep all optimization and loss settings fixed.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0009 | Bottleneck text cross-attention | 0008 | Test global prompt conditioning with linear-area attention |
