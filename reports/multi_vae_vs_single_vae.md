# Multi-VAE model vs per-VAE models

Question: can one shared model (core plus per-VAE adapters) upscale latents of many VAEs without losing quality compared to a separate model per VAE?

## Setup

- Model: 53M UNet (levels 512/256/256, SDXL-style window transformer blocks), conditioned on sigma, VAE one-hot and HFF. Per-VAE 1x1 input and output adapters. `tiny_wan_2_1` shares the adapter of `wan_2_1`. Code: `src/stdev/multi_vae_upscalers/small_universal.py`.
- Data: `precomputes/universal_512_dev`, 256 -> 512 px latents. Images: 10k samples for each of sdxl, sd_3_5, flux_1, flux_2, qwen_image_2_1, wan_2_1. Videos: 2065 clips for each of tiny_wan_2_1, tiny_ltx_2_5, tiny_minimax_h3.
- Training: flow matching, AdamW lr 4e-4 with a linear decay over the last 30%, EMA 0.999, bf16, 10 epochs, one VAE per batch.
- Multi-VAE run (`test_2`): all 9 VAEs interleaved by weighted sampling.
- Single-VAE runs: for each VAE a fresh model with the same seed (same core and adapter init), trained only on that VAE for 10 epochs.
- Metric: patch FID at 256 px native patches, EMA weights, 256 samples of shard 0. Real = VAE reconstruction of the ground truth high-res latents, fake = decoded upscaled latents. Bilinear latent upsampling is the reference.
- Shard 0 is a training shard, so the absolute numbers are optimistic for both models.

## Results

| VAE | multi FID | single FID | bilinear FID | single vs multi |
|---|---|---|---|---|
| sdxl | 88.49 | 65.88 | 39.70 | -26% |
| sd_3_5 | 53.45 | 35.78 | 36.96 | -33% |
| flux_1 | 87.36 | 53.27 | 37.50 | -39% |
| flux_2 | 40.70 | 21.28 | 32.65 | -48% |
| qwen_image_2_1 | 156.20 | 124.76 | 35.56 | -20% |
| wan_2_1 | 45.76 | 41.74 | 37.70 | -9% |
| tiny_ltx_2_5 | 103.42 | 70.67 | 38.98 | -32% |
| tiny_minimax_h3 | 44.83 | 44.38 | 29.67 | -1% |
| tiny_wan_2_1 | 24.60 | 22.78 | 23.13 | -7% |

Training steps per VAE: multi-VAE 5648-5723 (images) and 9216-9507 (videos); single-VAE 6240 and 10240. The single models received ~10% more steps because the weighted interleaving of the multi-VAE run did not give every VAE its full per-epoch batch count.

## Findings

- Single-VAE models are better on all 9 VAEs. VAEs interfere with each other inside the shared model; the shared features do not compensate for it at this size and budget.
- Interference differs a lot per VAE: large for flux_2, flux_1, sd_3_5, tiny_ltx_2_5 and sdxl; negligible for tiny_minimax_h3 and tiny_wan_2_1.
- Confound: ~10% more steps for the single models. It is too small to explain gaps of 30-48%, but it is not an exactly matched budget.
- Absolute quality is low in both settings. Only flux_2 (clearly), sd_3_5 and tiny_wan_2_1 (marginally) beat bilinear. The 53M model after ~6k steps is undertrained, so the comparison says little about converged models.
- qwen_image_2_1 is bad in both settings (decoder is fragile on imperfect latents).
- The per-epoch previews of the single runs are in `multi_vae_runs/single_<vae>/previews`.

## Conclusion

A single universal model costs quality for most VAEs at this scale. The plan moves to separate models trained from scratch per VAE.

Not tested: a larger shared model, more training, a balanced step budget, held-out evaluation.

Scripts: `scripts/run_multi_vae.py`, `scripts/eval_multi_vae.py`, `scripts/run_single_vae_baselines.py`.
