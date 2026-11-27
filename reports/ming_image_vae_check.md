# Ming-Image 0.2 Design VAE: is it likely to share the Qwen-Image 2.1 VAE problems?

Date: 2026-11-27. Model: https://huggingface.co/inclusionAI/Ming-Image-0.2-Design (released 2026-11-25).
Question: Qwen-Image 2.1 (autoresearch_4) turned out problematic for latent upscaling (decoder very sensitive to small
off-manifold errors, plateau at patch FID ~5.8-6.0). Is the new Ming VAE (8x, RGBA) likely to behave the same?

## Identity

- `vae/config.json`: `AutoencoderKLQwenImage` (the Wan 2.1 / Qwen-Image 1 architecture: base_dim 96, z_dim 16,
  dim_mult [1, 2, 4, 4], temporal downsampling) with `input_channels: 4` (RGBA), no per channel statistics, a single
  `scaling_factor 8.0064`. DiT: 6B, patch size 2, flow matching, fixed shift 6, 12 steps recommended.
- The weights are byte-identical to the Ming-Image-0.1-Design VAE and to the Qwen/Qwen-Image-Layered RGBA VAE (whose
  config keeps the Wan 2.1 per channel latents_mean / latents_std).
- Not a light finetune of Wan 2.1 (relative weight difference > 1 in the blocks), but functionally related: a per
  channel linear map from Wan 2.1 latents explains R2 0.80 of the Ming latents (min 0.68).
- Wrapper: `src/stdev/vaes/ming_image_vae.py` (normalization with the Qwen-Image-Layered / Wan statistics).

## Measurements (dev/autoresearch_4/vae_geometry_compare.py, 64 4KLSDB crops at 1024 / bilinear 512, 16 decoded)

Correction (same day): the first measurement normalized the Ming latents with the Qwen-Image-Layered / Wan per channel
statistics, which do not fit these weights (raw Ming latents have std ~0.116 = 1 / scaling_factor 8.0064, Wan ~2); the
large per channel offsets inflated all statistics pooled over channels ("98% explained by the source" was an artifact).
The table uses Ming's own per channel statistics (512 crops, `src/stdev/vaes/ming_image_vae.py`). Noise and band tests
were unaffected.

| | Ming-Image (= Qwen-Image-Layered) | Wan 2.1 | Qwen-Image 2.1 | FLUX.2 |
|---|---|---|---|---|
| latent grid at 1024 px | 128 x 128 x 16 | 128 x 128 x 16 | 64 x 64 x 64 | 128 x 128 x 32 |
| target std / residual std | 0.97 / 0.87 | 0.51 / 0.20 | 1.10 / 0.67 | 0.95 / 0.72 |
| variance explained by the bilinear source | 20.7% | 84.9% | 62.2% | 41.3% |
| residual mean offset (max abs) | 0.06 | 0.08 | 0.79 | 0.18 |
| lag-1 autocorrelation target / residual | 0.55 / 0.41 | 0.90 / 0.58 | 0.89 / 0.81 | 0.63 / 0.40 |
| high band (> half Nyquist) energy fraction | 30% | 4.0% | 4.1% | 27% |
| decoder PSNR, white noise 5 / 10 / 20% | 45.2 / 39.5 / 33.7 | 38.4 / 33.3 / 28.6 | 34.4 / 29.8 / 24.8 | 46.3 / 40.6 / 34.8 |
| decoder PSNR, smooth noise 10% | 35.7 | 35.3 | 35.6 | 36.3 |
| band tests low_res_mse 1e-4: gt / low only / low + other high / low + gt high shifted 1 token | 2.3 / 44.7 / 77.6 / 85.5 | 2.4 / 44.9 / 71.2 / 82.7 | 1.4 / 79.6 / 123.8 / 150.1 | 0.8 / 46.3 / 61.9 / 81.7 |
| E(D(z)) self-consistency (relative error) | 0.024 | 0.007 | 0.006 | 0.036 |
| posterior std / spread of posterior means | 9.9 (meaningless log-variance head) | 0.0004 | 0.00006 | 0.26 |

Noise is relative to the mean channel std of each VAE. The band cut is half Nyquist of each latent grid (32 px pixel
wavelength for the 8x VAEs, 64 px for Qwen-Image 2.1).

## Verdict: not likely to have the Qwen-Image 2.1 problems; a FLUX.2-like (detail heavy) task

- Same architecture family as Wan 2.1 / Qwen-Image 1, not the 16x DINO-aligned Qwen-Image-VAE-2.0, but a different
  latent space than Wan (per channel linear R2 0.80, different scale).
- Robust decoder (10% noise: 39.5 dB, close to FLUX.2 40.6, ~10 dB better than Qwen-Image 2.1); band tests like Wan /
  FLUX.2.
- The 2x task is detail heavy, not smooth: the bilinear source explains only 21% of the target variance (FLUX.2 41%,
  Wan 85%), 30% of the energy is above half Nyquist. Expect the FLUX.2 picture (AR3: the residual parameterization
  helps only a little), not the Wan one.
- KL: not determinable from the weights; the log-variance head is unconstrained (posterior std 10x the spread of the
  means), so a KL term was absent or negligible. Not the deciding factor: Wan 2.1 is also effectively KL-free (0.0004)
  and worked well.
- Caveats: Ming's pipeline uses a global scaling factor (the wrapper uses per channel statistics); only opaque RGB was
  tested (alpha untested); 64 images, decoder tests only. A 30 min autoresearch round follows (autoresearch_5).
