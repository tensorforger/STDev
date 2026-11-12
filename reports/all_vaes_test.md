# All VAEs test report

Setup: RTX 5090, batch size 1, no compilation. Quality: 8 images from `datasets/relaion_high_resolution_subset` (center crop, 512x512)
and 6 videos from `datasets/strf-128h/videos` (256x256; 33 frames for Wan/LTX, 22 for MiniMax). Raw data: `dev/vae_results/*.json`,
script: `dev/test_all_vaes.py`.

## Findings

- All VAEs share the `VAE` interface (`src/stdev/vaes/vae.py`): latents `[B, C, T, H, W]` (T = 1 for images), pixels in [0...1],
  float32 inputs and outputs. Input shapes are validated in `encode` (see per-class docstrings for the rules).
- Pixel range handling is correct for every VAE: roundtrip PSNR is 29.8-36.8 dB (images) and 32-34.5 dB (videos); no shifted or scaled outputs.
- Latent normalization: all VAEs are within roughly mean 0 / std 0.7-1.3. SDXL has a mean offset of +0.3 (no shift factor in its config).
  Wan 2.1 std is 0.70 (images) and 0.77 (videos), i.e. somewhat under-scaled compared to the others. Per-channel stds vary 0.4-2.2, which is a model property.
- bf16 works for all VAEs (no NaN, same MSE as fp32, relative latent error <= 4e-3; SDXL is the worst). MiniMax-H3 keeps its encoder in fp32 by design,
  so its bf16 latents are identical to fp32 and bf16 mainly halves decoder memory.
- Wan 2.1 works both as an image VAE (`as_image=True`) and as a video VAE with T = 1; both give identical results.
- Qwen-Image 2.1: alpha is encoded and decoded correctly. Alpha MSE is ~2e-4, opaque alpha is restored to within 1e-5.
  Plain RGB MSE on the synthetic alpha test is high only in the fully transparent area where RGB is unconstrained; alpha-weighted RGB MSE is ~7e-4.
- MiniMax-H3: T must be 5 + 17n with n >= 1 (or 1). A single latent frame (T = 1) encodes but cannot be decoded
  (the diffusers decoder requires >= 3 latent frames), and T = 5 yields 2 latents which also cannot be decoded. Hence the image row for minimax is absent
  and its decode latency for T = 1 shows an error.
- Frame maps verified by real encode/decode: Wan T = 1 + 4k -> 1 + k; LTX T = 1 + 8k -> 1 + k; MiniMax T = 5 + 17n -> 2 + 5n -> 5 + 17n.
- Tiny VAEs (`tiny_*`, TAEHV, `src/stdev/vaes/tiny_*_vae.py`) are drop-in video counterparts of Wan 2.1 (`taew2_1`), LTX 2.5 (`taeltx2_3_wide`)
  and MiniMax-H3 (`taeh3`): same latent shapes, frame rules and normalized latent space. See the tiny comparison section below.
  Tiny Wan latents are very close to full (relative MSE 0.9%); tiny LTX (24%) and tiny MiniMax (8%) are approximate. Roundtrip PSNR is 31 / 26 / 26.5 dB
  (full: 34.5 / 32.2 / 32.0). Tiny models are 5-90x smaller in weights and 5-100x faster, mostly in the encoder and with lower peak VRAM.


## Weights memory

| vae | fp32 MiB | bf16 MiB |
|---|---|---|
| sdxl | 321 | 161 |
| sd_3_5 | 321 | 161 |
| flux_1 | 321 | 161 |
| flux_2 | 323 | 162 |
| qwen_image_2_1 | 1303 | 651 |
| wan_2_1_image | 506 | 243 |
| wan_2_1_video | 506 | 243 |
| ltx_2_5 | 2793 | 1395 |
| minimax_h3 | 9941 | 5312 |
| tiny_wan_2_1 | 44 | 22 |
| tiny_ltx_2_5 | 116 | 58 |
| tiny_minimax_h3 | 44 | 22 |

## Roundtrip quality and latent statistics

MSE is on clamped [0...1] pixels. Latent mean/std are global over all test latents (fp32); per-channel columns show max |channel mean| and range of channel std. bf16 latent error is MSE(latents_bf16, latents_fp32) / var(latents_fp32).

| vae | data | latent shape | MSE fp32 | PSNR fp32 | MSE bf16 | PSNR bf16 | latent err bf16 | mean | std | max abs ch. mean | ch. std range | shape | finite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdxl | image | [1, 4, 1, 64, 64] | 1.72e-03 | 29.8 | 1.77e-03 | 29.7 | 3.68e-03 | +0.305 | 1.117 | 0.69 | 0.83-1.49 | ok | ok |
| sd_3_5 | image | [1, 16, 1, 64, 64] | 7.52e-04 | 33.1 | 7.49e-04 | 33.2 | 3.99e-05 | +0.095 | 1.267 | 1.15 | 0.48-1.68 | ok | ok |
| flux_1 | image | [1, 16, 1, 64, 64] | 5.03e-04 | 35.2 | 5.02e-04 | 35.2 | 1.26e-03 | +0.050 | 1.265 | 0.79 | 0.54-2.20 | ok | ok |
| flux_2 | image | [1, 32, 1, 64, 64] | 3.22e-04 | 36.2 | 3.25e-04 | 36.2 | 1.00e-04 | +0.009 | 0.904 | 0.50 | 0.71-1.10 | ok | ok |
| qwen_image_2_1 | image | [1, 64, 1, 32, 32] | 3.41e-04 | 36.8 | 3.45e-04 | 36.8 | 1.71e-04 | -0.036 | 1.257 | 1.71 | 0.61-1.67 | ok | ok |
| wan_2_1_image | image | [1, 16, 1, 64, 64] | 9.28e-04 | 32.9 | 9.32e-04 | 32.9 | 6.01e-05 | +0.059 | 0.702 | 0.71 | 0.41-0.78 | ok | ok |
| wan_2_1_video | image | [1, 16, 1, 64, 64] | 9.28e-04 | 32.9 | 9.32e-04 | 32.9 | 6.01e-05 | +0.059 | 0.702 | 0.71 | 0.41-0.78 | ok | ok |
| wan_2_1_video | video | [1, 16, 9, 32, 32] | 3.63e-04 | 34.5 | 3.66e-04 | 34.4 | 4.94e-05 | +0.010 | 0.768 | 0.40 | 0.54-0.88 | ok | ok |
| ltx_2_5 | image | [1, 128, 1, 16, 16] | 9.80e-04 | 32.4 | 9.94e-04 | 32.3 | 1.99e-04 | +0.027 | 1.234 | 1.17 | 0.68-1.86 | ok | ok |
| ltx_2_5 | video | [1, 128, 5, 8, 8] | 6.23e-04 | 32.2 | 6.32e-04 | 32.1 | 1.47e-04 | -0.004 | 1.029 | 0.55 | 0.61-1.31 | ok | ok |
| minimax_h3 | video | [1, 24, 7, 16, 16] | 6.36e-04 | 32.0 | 6.40e-04 | 32.0 | 0.00e+00 | +0.103 | 1.025 | 1.13 | 0.53-1.39 | ok | ok |
| tiny_wan_2_1 | image | [1, 16, 1, 64, 64] | 1.63e-03 | 29.8 | 1.63e-03 | 29.8 | 6.16e-05 | +0.059 | 0.699 | 0.70 | 0.41-0.77 | ok | ok |
| tiny_wan_2_1 | video | [1, 16, 9, 32, 32] | 8.11e-04 | 31.0 | 8.14e-04 | 30.9 | 7.61e-05 | +0.008 | 0.764 | 0.40 | 0.53-0.87 | ok | ok |
| tiny_ltx_2_5 | image | [1, 128, 1, 16, 16] | 3.77e-03 | 25.3 | 3.81e-03 | 25.3 | 2.98e-04 | +0.023 | 1.181 | 1.11 | 0.55-1.83 | ok | ok |
| tiny_ltx_2_5 | video | [1, 128, 5, 8, 8] | 2.43e-03 | 26.2 | 2.44e-03 | 26.1 | 2.96e-04 | -0.002 | 0.965 | 0.48 | 0.63-1.21 | ok | ok |
| tiny_minimax_h3 | video | [1, 24, 7, 16, 16] | 2.25e-03 | 26.5 | 2.25e-03 | 26.5 | 8.32e-05 | +0.106 | 0.953 | 0.89 | 0.52-1.24 | ok | ok |

## Qwen alpha channel

| dtype | RGB MSE | alpha MSE | alpha-weighted RGB MSE | opaque alpha abs error |
|---|---|---|---|---|
| fp32 | 4.71e-02 | 2.28e-04 | 6.88e-04 | 9.37e-06 |
| bf16 | 4.24e-02 | 2.28e-04 | 6.92e-04 | 3.57e-06 |

## Tiny vs full VAE latents

6 videos, 256x256, fp32. Latent MSE between tiny and full encoder on the same video (`dev/compare_tiny_vaes.py`); relative = MSE / var(full latents). Roundtrip MSE is tiny encode + tiny decode on clamped [0...1] pixels.

| tiny vae | latent shape | latent MSE | relative latent MSE | roundtrip pixel MSE |
|---|---|---|---|---|
| tiny_wan_2_1 | [1, 16, 9, 32, 32] | 5.41e-03 | 9.24e-03 | 8.11e-04 |
| tiny_ltx_2_5 | [1, 128, 5, 8, 8] | 2.52e-01 | 2.41e-01 | 2.43e-03 |
| tiny_minimax_h3 | [1, 24, 7, 16, 16] | 8.23e-02 | 8.02e-02 | 2.25e-03 |

## Latency and peak VRAM: sdxl

Batch size 1, median of 2-3 runs. MiB is peak allocation above the pre-call state (includes input tensor).

| size | frames | latent shape | enc ms fp32 | dec ms fp32 | enc ms bf16 | dec ms bf16 | enc MiB fp32 | dec MiB fp32 | enc MiB bf16 | dec MiB bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | [1, 4, 1, 128, 128] | 120.4 | 215.6 | 69.2 | 129.3 | 2573 | 3843 | 1542 | 2434 |
| 512 | 1 | [1, 4, 1, 64, 64] | 24.8 | 48.6 | 12.7 | 26.6 | 644 | 962 | 386 | 610 |
| 256 | 1 | [1, 4, 1, 32, 32] | 5.8 | 11.4 | 3.5 | 6.7 | 161 | 242 | 97 | 154 |
| 128 | 1 | [1, 4, 1, 16, 16] | 2.0 | 3.6 | 1.3 | 2.3 | 41 | 62 | 24 | 39 |

## Latency and peak VRAM: sd_3_5

Batch size 1, median of 2-3 runs. MiB is peak allocation above the pre-call state (includes input tensor).

| size | frames | latent shape | enc ms fp32 | dec ms fp32 | enc ms bf16 | dec ms bf16 | enc MiB fp32 | dec MiB fp32 | enc MiB bf16 | dec MiB bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | [1, 16, 1, 128, 128] | 119.9 | 227.0 | 72.8 | 135.9 | 2573 | 3843 | 1542 | 2434 |
| 512 | 1 | [1, 16, 1, 64, 64] | 25.7 | 51.7 | 13.5 | 27.6 | 644 | 963 | 386 | 610 |
| 256 | 1 | [1, 16, 1, 32, 32] | 6.0 | 11.8 | 3.6 | 6.8 | 161 | 242 | 97 | 154 |
| 128 | 1 | [1, 16, 1, 16, 16] | 2.0 | 3.8 | 1.3 | 2.4 | 41 | 62 | 24 | 39 |

## Latency and peak VRAM: flux_1

Batch size 1, median of 2-3 runs. MiB is peak allocation above the pre-call state (includes input tensor).

| size | frames | latent shape | enc ms fp32 | dec ms fp32 | enc ms bf16 | dec ms bf16 | enc MiB fp32 | dec MiB fp32 | enc MiB bf16 | dec MiB bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | [1, 16, 1, 128, 128] | 125.4 | 225.2 | 73.2 | 136.9 | 2573 | 3843 | 1542 | 2434 |
| 512 | 1 | [1, 16, 1, 64, 64] | 25.7 | 51.0 | 13.4 | 27.8 | 644 | 963 | 386 | 610 |
| 256 | 1 | [1, 16, 1, 32, 32] | 6.1 | 11.9 | 3.7 | 6.9 | 161 | 242 | 97 | 154 |
| 128 | 1 | [1, 16, 1, 16, 16] | 2.2 | 3.6 | 1.3 | 2.4 | 41 | 62 | 24 | 39 |

## Latency and peak VRAM: flux_2

Batch size 1, median of 2-3 runs. MiB is peak allocation above the pre-call state (includes input tensor).

| size | frames | latent shape | enc ms fp32 | dec ms fp32 | enc ms bf16 | dec ms bf16 | enc MiB fp32 | dec MiB fp32 | enc MiB bf16 | dec MiB bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | [1, 32, 1, 128, 128] | 125.7 | 223.7 | 72.4 | 135.9 | 2573 | 3846 | 1542 | 2438 |
| 512 | 1 | [1, 32, 1, 64, 64] | 25.7 | 51.7 | 13.6 | 27.7 | 644 | 963 | 386 | 611 |
| 256 | 1 | [1, 32, 1, 32, 32] | 6.1 | 11.9 | 3.6 | 6.9 | 161 | 243 | 97 | 154 |
| 128 | 1 | [1, 32, 1, 16, 16] | 2.1 | 3.7 | 1.4 | 2.4 | 41 | 62 | 24 | 39 |

## Latency and peak VRAM: qwen_image_2_1

Batch size 1, median of 2-3 runs. MiB is peak allocation above the pre-call state (includes input tensor).

| size | frames | latent shape | enc ms fp32 | dec ms fp32 | enc ms bf16 | dec ms bf16 | enc MiB fp32 | dec MiB fp32 | enc MiB bf16 | dec MiB bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | [1, 64, 1, 64, 64] | 94.4 | 368.3 | 56.3 | 225.6 | 4260 | 13436 | 2322 | 6718 |
| 512 | 1 | [1, 64, 1, 32, 32] | 22.4 | 91.1 | 12.0 | 51.4 | 1067 | 3362 | 582 | 1681 |
| 256 | 1 | [1, 64, 1, 16, 16] | 5.9 | 21.6 | 3.9 | 12.0 | 267 | 849 | 146 | 423 |
| 128 | 1 | [1, 64, 1, 8, 8] | 3.0 | 7.7 | 2.5 | 4.8 | 77 | 208 | 38 | 107 |

## Latency and peak VRAM: wan_2_1_image

Batch size 1, median of 2-3 runs. MiB is peak allocation above the pre-call state (includes input tensor).

| size | frames | latent shape | enc ms fp32 | dec ms fp32 | enc ms bf16 | dec ms bf16 | enc MiB fp32 | dec MiB fp32 | enc MiB bf16 | dec MiB bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | [1, 16, 1, 128, 128] | 166.9 | 261.7 | 92.2 | 149.2 | 5027 | 8078 | 2705 | 4230 |
| 512 | 1 | [1, 16, 1, 64, 64] | 38.3 | 63.0 | 19.7 | 32.1 | 1260 | 2023 | 679 | 1060 |
| 256 | 1 | [1, 16, 1, 32, 32] | 9.8 | 15.4 | 5.3 | 8.0 | 317 | 507 | 172 | 267 |
| 128 | 1 | [1, 16, 1, 16, 16] | 3.7 | 5.4 | 2.3 | 3.7 | 90 | 134 | 43 | 68 |

## Latency and peak VRAM: wan_2_1_video

Batch size 1, median of 2-3 runs. MiB is peak allocation above the pre-call state (includes input tensor).

| size | frames | latent shape | enc ms fp32 | dec ms fp32 | enc ms bf16 | dec ms bf16 | enc MiB fp32 | dec MiB fp32 | enc MiB bf16 | dec MiB bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | [1, 16, 1, 128, 128] | 164.8 | 258.5 | 81.1 | 134.3 | 5027 | 8078 | 2705 | 4230 |
| 1024 | 5 | [1, 16, 2, 128, 128] | 654.1 | 1081.9 | 334.6 | 561.3 | 14074 | 20245 | 7804 | 10890 |
| 1024 | 17 | [1, 16, 5, 128, 128] | 2150.3 | 3593.2 | 1107.4 | 1871.5 | 15835 | 21213 | 8684 | 11374 |
| 512 | 1 | [1, 16, 1, 64, 64] | 38.1 | 62.0 | 17.4 | 28.6 | 1259 | 2022 | 679 | 1060 |
| 512 | 5 | [1, 16, 2, 64, 64] | 156.5 | 266.5 | 78.9 | 133.9 | 3526 | 5068 | 1955 | 2727 |
| 512 | 17 | [1, 16, 5, 64, 64] | 519.6 | 873.5 | 266.4 | 454.8 | 3966 | 5309 | 2175 | 2847 |
| 512 | 33 | [1, 16, 9, 64, 64] | 886.0 | 1505.9 | 516.5 | 886.6 | 4015 | 5359 | 2201 | 2871 |
| 512 | 65 | [1, 16, 17, 64, 64] | 1740.7 | 2961.2 | 1043.4 | 1754.2 | 4115 | 5459 | 2252 | 2922 |
| 256 | 1 | [1, 16, 1, 32, 32] | 8.7 | 14.0 | 4.7 | 7.2 | 317 | 507 | 172 | 267 |
| 256 | 5 | [1, 16, 2, 32, 32] | 34.7 | 58.6 | 18.7 | 30.8 | 886 | 1271 | 492 | 687 |
| 256 | 17 | [1, 16, 5, 32, 32] | 114.3 | 195.0 | 61.6 | 103.2 | 997 | 1333 | 549 | 719 |
| 256 | 33 | [1, 16, 9, 32, 32] | 220.5 | 376.9 | 118.8 | 199.7 | 1009 | 1344 | 555 | 723 |
| 256 | 65 | [1, 16, 17, 32, 32] | 475.3 | 741.2 | 233.2 | 438.7 | 1034 | 1371 | 566 | 734 |
| 256 | 129 | [1, 16, 33, 32, 32] | 894.6 | 1495.1 | 462.0 | 784.3 | 1085 | 1423 | 591 | 761 |
| 128 | 1 | [1, 16, 1, 16, 16] | 3.4 | 5.0 | 2.2 | 3.4 | 90 | 134 | 43 | 68 |
| 128 | 5 | [1, 16, 2, 16, 16] | 10.3 | 16.4 | 6.7 | 10.5 | 254 | 320 | 125 | 174 |
| 128 | 17 | [1, 16, 5, 16, 16] | 32.4 | 51.0 | 20.1 | 32.2 | 274 | 342 | 140 | 183 |
| 128 | 33 | [1, 16, 9, 16, 16] | 59.3 | 97.2 | 37.9 | 65.2 | 278 | 345 | 141 | 183 |
| 128 | 65 | [1, 16, 17, 16, 16] | 115.3 | 204.9 | 73.6 | 118.7 | 284 | 352 | 144 | 187 |
| 128 | 129 | [1, 16, 33, 16, 16] | 248.0 | 401.9 | 144.9 | 234.0 | 297 | 365 | 151 | 193 |

## Latency and peak VRAM: ltx_2_5

Batch size 1, median of 2-3 runs. MiB is peak allocation above the pre-call state (includes input tensor).

| size | frames | latent shape | enc ms fp32 | dec ms fp32 | enc ms bf16 | dec ms bf16 | enc MiB fp32 | dec MiB fp32 | enc MiB bf16 | dec MiB bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | [1, 128, 1, 32, 32] | 61.6 | 59.9 | 29.4 | 28.4 | 499 | 517 | 260 | 272 |
| 1024 | 9 | [1, 128, 2, 32, 32] | 284.8 | 287.7 | 144.0 | 146.4 | 2173 | 1779 | 1159 | 1033 |
| 1024 | 17 | [1, 128, 3, 32, 32] | 500.0 | 506.6 | 258.9 | 264.1 | 3933 | 3188 | 2103 | 1866 |
| 512 | 1 | [1, 128, 1, 16, 16] | 17.6 | 17.4 | 9.8 | 10.3 | 168 | 474 | 86 | 326 |
| 512 | 9 | [1, 128, 2, 16, 16] | 69.4 | 70.8 | 34.1 | 35.2 | 545 | 509 | 290 | 259 |
| 512 | 17 | [1, 128, 3, 16, 16] | 123.4 | 135.3 | 63.4 | 64.8 | 985 | 798 | 526 | 467 |
| 512 | 33 | [1, 128, 5, 16, 16] | 239.8 | 243.9 | 120.3 | 122.7 | 1734 | 1503 | 998 | 883 |
| 512 | 65 | [1, 128, 9, 16, 16] | 466.2 | 473.7 | 236.4 | 240.8 | 3366 | 2911 | 1942 | 1715 |
| 256 | 1 | [1, 128, 1, 8, 8] | 7.3 | 7.5 | 4.8 | 4.8 | 134 | 449 | 63 | 229 |
| 256 | 9 | [1, 128, 2, 8, 8] | 19.4 | 20.0 | 10.7 | 11.5 | 178 | 452 | 88 | 269 |
| 256 | 17 | [1, 128, 3, 8, 8] | 33.0 | 34.0 | 17.1 | 18.3 | 294 | 487 | 144 | 309 |
| 256 | 33 | [1, 128, 5, 8, 8] | 60.8 | 61.2 | 30.4 | 31.9 | 546 | 484 | 253 | 589 |
| 256 | 65 | [1, 128, 9, 8, 8] | 119.0 | 119.9 | 58.0 | 59.4 | 843 | 729 | 489 | 432 |
| 256 | 129 | [1, 128, 17, 8, 8] | 242.0 | 243.2 | 117.3 | 120.4 | 2058 | 1822 | 961 | 849 |
| 128 | 1 | [1, 128, 1, 4, 4] | 5.5 | 6.0 | 3.6 | 3.3 | 117 | 445 | 59 | 228 |
| 128 | 9 | [1, 128, 2, 4, 4] | 9.4 | 9.9 | 4.4 | 4.7 | 129 | 446 | 67 | 229 |
| 128 | 17 | [1, 128, 3, 4, 4] | 11.3 | 12.2 | 6.3 | 6.7 | 139 | 447 | 63 | 229 |
| 128 | 33 | [1, 128, 5, 4, 4] | 18.6 | 19.5 | 9.7 | 10.3 | 160 | 448 | 80 | 240 |
| 128 | 65 | [1, 128, 9, 4, 4] | 33.2 | 33.5 | 17.2 | 18.6 | 213 | 464 | 124 | 292 |
| 128 | 129 | [1, 128, 17, 4, 4] | 58.6 | 59.4 | 29.8 | 31.8 | 417 | 693 | 241 | 572 |

## Latency and peak VRAM: minimax_h3

Batch size 1, median of 2-3 runs. MiB is peak allocation above the pre-call state (includes input tensor).

| size | frames | latent shape | enc ms fp32 | dec ms fp32 | enc ms bf16 | dec ms bf16 | enc MiB fp32 | dec MiB fp32 | enc MiB bf16 | dec MiB bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | [1, 24, 1, 64, 64] | 304.4 | ValueError:  | 304.4 | ValueError:  | 338 | ValueError:  | 338 | ValueError:  |
| 512 | 1 | [1, 24, 1, 32, 32] | 109.7 | ValueError:  | 109.9 | ValueError:  | 329 | ValueError:  | 330 | ValueError:  |
| 512 | 22 | [1, 24, 7, 32, 32] | 2342.7 | 1999.9 | 2349.0 | 537.5 | 3620 | 436 | 3620 | 233 |
| 512 | 39 | [1, 24, 12, 32, 32] | 3518.6 | 4014.5 | 3667.9 | 1116.6 | 3723 | 521 | 3725 | 354 |
| 256 | 1 | [1, 24, 1, 16, 16] | 12.3 | ValueError:  | 12.3 | ValueError:  | 325 | ValueError:  | 326 | ValueError:  |
| 256 | 22 | [1, 24, 7, 16, 16] | 260.9 | 222.8 | 276.3 | 66.3 | 3465 | 268 | 3465 | 148 |
| 256 | 39 | [1, 24, 12, 16, 16] | 391.3 | 445.7 | 394.6 | 119.4 | 3491 | 289 | 3491 | 159 |
| 256 | 73 | [1, 24, 22, 16, 16] | 652.1 | 891.8 | 654.0 | 262.6 | 3542 | 336 | 3542 | 183 |
| 256 | 124 | [1, 24, 37, 16, 16] | 1044.1 | 1564.6 | 1048.5 | 421.6 | 3619 | 374 | 3620 | 281 |
| 128 | 1 | [1, 24, 1, 8, 8] | 4.8 | ValueError:  | 4.9 | ValueError:  | 117 | ValueError:  | 117 | ValueError:  |
| 128 | 22 | [1, 24, 7, 8, 8] | 67.9 | 61.8 | 74.8 | 19.4 | 872 | 68 | 872 | 39 |
| 128 | 39 | [1, 24, 12, 8, 8] | 101.8 | 124.8 | 111.8 | 35.8 | 879 | 74 | 879 | 41 |
| 128 | 73 | [1, 24, 22, 8, 8] | 169.5 | 250.5 | 176.9 | 76.6 | 891 | 84 | 892 | 48 |
| 128 | 124 | [1, 24, 37, 8, 8] | 271.2 | 439.4 | 296.5 | 132.7 | 911 | 94 | 912 | 71 |

## Latency and peak VRAM: tiny_wan_2_1

Batch size 1, median of 2-3 runs. MiB is peak allocation above the pre-call state (includes input tensor).

| size | frames | latent shape | enc ms fp32 | dec ms fp32 | enc ms bf16 | dec ms bf16 | enc MiB fp32 | dec MiB fp32 | enc MiB bf16 | dec MiB bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | [1, 16, 1, 128, 128] | 24.8 | 38.8 | 12.7 | 19.4 | 1620 | 4416 | 1585 | 2209 |
| 1024 | 5 | [1, 16, 2, 128, 128] | 49.7 | 78.3 | 26.8 | 40.4 | 3204 | 8768 | 1632 | 4385 |
| 1024 | 17 | [1, 16, 5, 128, 128] | 129.3 | 197.2 | 66.5 | 105.1 | 7956 | 21824 | 4080 | 10915 |
| 512 | 1 | [1, 16, 1, 64, 64] | 6.0 | 8.6 | 3.2 | 4.7 | 405 | 1104 | 396 | 552 |
| 512 | 5 | [1, 16, 2, 64, 64] | 10.8 | 17.6 | 5.4 | 9.0 | 801 | 2192 | 409 | 1096 |
| 512 | 17 | [1, 16, 5, 64, 64] | 34.3 | 47.1 | 16.6 | 25.4 | 1989 | 5456 | 1021 | 2729 |
| 512 | 33 | [1, 16, 9, 64, 64] | 55.3 | 86.8 | 30.6 | 46.8 | 3573 | 9808 | 1837 | 4905 |
| 512 | 65 | [1, 16, 17, 64, 64] | 106.7 | 168.9 | 57.9 | 91.1 | 6741 | 18512 | 3469 | 9258 |
| 256 | 1 | [1, 16, 1, 32, 32] | 1.5 | 2.7 | 1.0 | 1.5 | 137 | 276 | 99 | 138 |
| 256 | 5 | [1, 16, 2, 32, 32] | 6.2 | 8.5 | 1.5 | 2.7 | 526 | 548 | 103 | 274 |
| 256 | 17 | [1, 16, 5, 32, 32] | 6.0 | 10.3 | 3.5 | 5.6 | 498 | 1364 | 255 | 682 |
| 256 | 33 | [1, 16, 9, 32, 32] | 12.3 | 19.4 | 6.3 | 10.2 | 894 | 2452 | 460 | 1226 |
| 256 | 65 | [1, 16, 17, 32, 32] | 26.8 | 41.1 | 13.0 | 20.5 | 1686 | 4628 | 868 | 2315 |
| 256 | 129 | [1, 16, 33, 32, 32] | 50.6 | 79.6 | 28.9 | 41.8 | 3270 | 8980 | 1684 | 4491 |
| 128 | 1 | [1, 16, 1, 16, 16] | 0.7 | 1.5 | 0.8 | 0.9 | 34 | 326 | 25 | 35 |
| 128 | 5 | [1, 16, 2, 16, 16] | 0.9 | 1.7 | 0.8 | 1.1 | 68 | 137 | 50 | 69 |
| 128 | 17 | [1, 16, 5, 16, 16] | 1.6 | 3.1 | 1.0 | 1.6 | 169 | 341 | 124 | 171 |
| 128 | 33 | [1, 16, 9, 16, 16] | 4.1 | 6.0 | 1.7 | 2.7 | 317 | 613 | 223 | 307 |
| 128 | 65 | [1, 16, 17, 16, 16] | 5.3 | 8.6 | 3.0 | 5.0 | 421 | 1157 | 421 | 579 |
| 128 | 129 | [1, 16, 33, 16, 16] | 10.8 | 17.5 | 5.9 | 9.5 | 817 | 2245 | 817 | 1123 |

## Latency and peak VRAM: tiny_ltx_2_5

Batch size 1, median of 2-3 runs. MiB is peak allocation above the pre-call state (includes input tensor).

| size | frames | latent shape | enc ms fp32 | dec ms fp32 | enc ms bf16 | dec ms bf16 | enc MiB fp32 | dec MiB fp32 | enc MiB bf16 | dec MiB bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | [1, 128, 1, 32, 32] | 6.8 | 27.0 | 1.8 | 9.6 | 698 | 1554 | 273 | 840 |
| 1024 | 9 | [1, 128, 2, 32, 32] | 6.5 | 37.7 | 3.8 | 19.4 | 980 | 2833 | 544 | 1673 |
| 1024 | 17 | [1, 128, 3, 32, 32] | 10.0 | 56.3 | 5.8 | 31.7 | 1428 | 4241 | 816 | 2505 |
| 512 | 1 | [1, 128, 1, 16, 16] | 1.1 | 4.8 | 0.8 | 3.1 | 134 | 357 | 73 | 210 |
| 512 | 9 | [1, 128, 2, 16, 16] | 1.6 | 8.5 | 1.0 | 5.0 | 246 | 709 | 145 | 418 |
| 512 | 17 | [1, 128, 3, 16, 16] | 2.4 | 14.4 | 1.3 | 7.2 | 358 | 1061 | 217 | 626 |
| 512 | 33 | [1, 128, 5, 16, 16] | 3.9 | 22.5 | 2.1 | 11.5 | 582 | 1765 | 361 | 1042 |
| 512 | 65 | [1, 128, 9, 16, 16] | 7.2 | 42.3 | 4.5 | 22.6 | 1030 | 3173 | 649 | 1875 |
| 256 | 1 | [1, 128, 1, 8, 8] | 0.8 | 2.6 | 0.8 | 2.1 | 33 | 90 | 19 | 53 |
| 256 | 9 | [1, 128, 2, 8, 8] | 0.8 | 3.4 | 0.8 | 2.5 | 61 | 279 | 38 | 105 |
| 256 | 17 | [1, 128, 3, 8, 8] | 0.9 | 4.7 | 0.8 | 2.8 | 89 | 267 | 55 | 157 |
| 256 | 33 | [1, 128, 5, 8, 8] | 1.0 | 5.7 | 0.8 | 3.8 | 145 | 442 | 92 | 261 |
| 256 | 65 | [1, 128, 9, 8, 8] | 1.8 | 9.4 | 1.2 | 5.2 | 257 | 794 | 163 | 469 |
| 256 | 129 | [1, 128, 17, 8, 8] | 3.6 | 18.4 | 2.0 | 9.7 | 481 | 1498 | 307 | 885 |
| 128 | 1 | [1, 128, 1, 4, 4] | 0.8 | 2.1 | 0.8 | 2.0 | 12 | 29 | 5 | 14 |
| 128 | 9 | [1, 128, 2, 4, 4] | 0.8 | 2.2 | 0.8 | 2.0 | 15 | 45 | 9 | 26 |
| 128 | 17 | [1, 128, 3, 4, 4] | 0.8 | 2.5 | 0.8 | 2.1 | 22 | 93 | 14 | 40 |
| 128 | 33 | [1, 128, 5, 4, 4] | 0.8 | 3.2 | 0.8 | 2.2 | 37 | 111 | 21 | 66 |
| 128 | 65 | [1, 128, 9, 4, 4] | 0.8 | 3.5 | 0.8 | 2.6 | 66 | 199 | 41 | 117 |
| 128 | 129 | [1, 128, 17, 4, 4] | 1.0 | 5.4 | 0.9 | 3.3 | 122 | 376 | 78 | 221 |

## Latency and peak VRAM: tiny_minimax_h3

Batch size 1, median of 2-3 runs. MiB is peak allocation above the pre-call state (includes input tensor).

| size | frames | latent shape | enc ms fp32 | dec ms fp32 | enc ms bf16 | dec ms bf16 | enc MiB fp32 | dec MiB fp32 | enc MiB bf16 | dec MiB bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 22 | [1, 24, 7, 32, 32] | 16.1 | 15.7 | 7.9 | 7.9 | 1080 | 1908 | 813 | 954 |
| 512 | 39 | [1, 24, 12, 32, 32] | 25.4 | 30.1 | 12.7 | 14.1 | 1620 | 3268 | 1229 | 1635 |
| 256 | 22 | [1, 24, 7, 16, 16] | 3.3 | 4.1 | 2.0 | 2.1 | 270 | 477 | 264 | 239 |
| 256 | 39 | [1, 24, 12, 16, 16] | 5.1 | 6.3 | 2.8 | 3.9 | 406 | 817 | 397 | 409 |
| 256 | 73 | [1, 24, 22, 16, 16] | 8.8 | 11.9 | 4.9 | 6.6 | 676 | 1497 | 666 | 749 |
| 256 | 124 | [1, 24, 37, 16, 16] | 16.3 | 21.1 | 8.1 | 10.8 | 1080 | 2517 | 1067 | 1259 |
| 128 | 22 | [1, 24, 7, 8, 8] | 1.0 | 1.4 | 0.8 | 1.0 | 108 | 120 | 66 | 60 |
| 128 | 39 | [1, 24, 12, 8, 8] | 1.4 | 1.8 | 1.0 | 1.3 | 162 | 204 | 100 | 103 |
| 128 | 73 | [1, 24, 22, 8, 8] | 2.1 | 3.3 | 1.3 | 1.8 | 238 | 374 | 167 | 187 |
| 128 | 124 | [1, 24, 37, 8, 8] | 3.4 | 5.2 | 1.9 | 3.1 | 430 | 629 | 268 | 315 |
