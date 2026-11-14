# Experiment logs

## Setup notes (before round 1)

- Baseline `run_0000`: 96.5M parameter window-attention UNet on frames folded into the batch (5D latents, 16 channels), temporal attention (identity for single frame inputs), sinusoidal frame index embedding, conditioning = sigma + hff embeddings. Low resolution latents are resized trilinearly to the target size and concatenated to the noisy latents.
- Recipe from phase 1: lr 4e-4, linear decay over the last 30%, logit-normal sigmas with shift 1, Euler 20 steps at sampling shift 2, no EMA, no augmentation.
- Sampling weights of pair types in `run_0000`: image 512->1024 (4), video 17->33 frames at 512 (3), image 256->512 (1), video 17->33 frames at 1024 (1), other pair types have weight 0 (their shapes are supported).
- Batch size comes from a token budget: 16 x 128 x 128 target latent positions, capped at 64 samples.
- Compile: residual and transformer blocks are compiled, all training shapes are warmed up before the clock starts (about 160 s while sharing the GPU with dataset encoding).
- The benchmark set is currently taken from the training samples (1000 images, 100 videos), so metrics are optimistic. To be replaced with held-out data.
