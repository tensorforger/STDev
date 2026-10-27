# Autoresearch experiment log

## Reference baseline

Run 0000 is a 59.27M-parameter one-step flow-matching UNet with low-resolution latent FiLM conditioning. It trained for 5451 steps with batch size 16 and uniformly sampled sigmas.

| Run ID | Axis FID | Flux FID | Latency 1024 (ms) | Peak VRAM 1024 (MB) |
|---|---:|---:|---:|---:|
| 0000 | 16.4405 | 12.1390 | 6.2252 | 268.3892 |

## Round 1

### Hypothesis

One-step inference always evaluates the model at sigma=1, but baseline training distributes capacity and optimization across uniform sigmas. Training only at sigma=1 should align the training and inference distributions and improve the direct noise-to-latent mapping without changing inference cost. The architecture, optimizer, data, and sampler are held fixed to isolate this effect.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0001 | Endpoint-only flow matching | 0000 | Test whether training exclusively at pure noise improves one-step quality |

### Result

Run 0001 completed 5457 steps. Endpoint-only training improved Axis FID from 16.4405 to 11.8234 and Flux FID from 12.1390 to 7.4647. High-resolution latent MSE also improved from 0.5583/0.5036 to 0.5083/0.4466. Latency and memory were unchanged within normal variation. This validates inference-aligned training as the new baseline direction.

| Run ID | Axis FID | Flux FID | Latency 1024 (ms) | Peak VRAM 1024 (MB) |
|---|---:|---:|---:|---:|
| 0001 | 11.8234 | 7.4647 | 6.0564 | 268.3892 |

## Round 2

### Hypothesis

The endpoint model trains with a constant learning rate and its loss is still decreasing at the end of the run. An exponential moving average should reduce optimization noise and improve generalization without changing inference cost. Use decay 0.999 and save only the averaged weights.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0002 | Endpoint training with weight EMA | 0001 | Test whether averaging the last optimization trajectory improves quality |

### Result

EMA with decay 0.999 caused a large regression: Axis FID 17.9568 versus 11.8234 and Flux FID 12.6722 versus 7.4647. The training trajectory and final raw loss matched run 0001, so this is specifically an averaged-weight effect rather than an optimization failure. Weight averaging likely damages the generative output distribution despite comparable reconstruction. Do not use EMA in subsequent branches.

| Run ID | Axis FID | Flux FID | Latency 1024 (ms) | Peak VRAM 1024 (MB) |
|---|---:|---:|---:|---:|
| 0002 | 17.9568 | 12.6722 | 5.8865 | 268.3892 |

## Round 3

### Hypothesis

The 1000-sample dataset is reused for roughly 87 epochs, while real-image Axis quality still trails Flux quality. Dihedral augmentation should reduce memorization and enforce the spatial equivariance expected from 2x upscaling. Apply the same random flip/transpose to noise, target latents, and low-resolution latents, with all other settings inherited from run 0001.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0003 | Endpoint training with dihedral augmentation | 0001 | Test whether geometric augmentation improves generalization |

### Result

Dihedral augmentation caused a severe regression to Axis FID 44.7419 and Flux FID 38.2454, with visible structural artifacts. FLUX latent spatial rotations do not preserve the decoded image distribution; latent-space dihedral augmentation is therefore invalid for this dataset. Reject this direction and preserve run 0001 as the quality leader.

| Run ID | Axis FID | Flux FID | Latency 1024 (ms) | Peak VRAM 1024 (MB) |
|---|---:|---:|---:|---:|
| 0003 | 44.7419 | 38.2454 | 5.8716 | 268.3892 |

## Round 4

### Hypothesis

Random crops can increase spatial diversity without the invalid rotations used in round 3. Cropping high-resolution latents from 128x128 to 96x96, with aligned 48x48 low-resolution inputs, should also provide roughly 1.8x more optimization steps at the same wall-clock budget. This tests crop-based training and increased update count together against full-frame run 0001.

| Run ID | Run name | Parent runs | Run purpose |
|---|---|---|---|
| 0004 | Endpoint training on random 96px latent crops | 0001 | Test valid spatial augmentation and faster optimization |
