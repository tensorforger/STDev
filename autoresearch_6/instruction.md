# Autoresearch 6: MiniMax-H3-Image latents -> FLUX.2 latents ("generative VAE translator")

Experimental side project, 4x RTX 5090. Idea: MiniMax-H3-Image generates directly at 4K / 8K, but its VAE (16x, 24
channels) loses small faces and text; the ostris decoder fine-tune (`ostris/ostris_h3_image_vae`) oversmooths. Translate
the H3 latent of an image into the FLUX.2 latent (8x, 32 channels) of the **same** image with a Steppe-like generative
(rectified flow) model, then decode with the FLUX.2 VAE (best FID of all VAEs) or TAEF2. Not an upscaler: output
resolution = input resolution. Log: `./autoresearch_6/experiment_logs.md`.

## Viability criteria (user)

- 100% viable: the stack beats the native H3 decoder on all of latency, VRAM, fid, patch_fid.
- Still useful: better quality metrics, but slower or more VRAM.
- Not worth it: still worse quality after all attempts.

Speed: batch 1 latency and peak VRAM at 1024, 2048 and 4096 px (square), the full latent -> pixels path. The model may
be assumed one step (after RF distillation); approximate the distillation quality drop as +5% on the metrics.
Quality: Axis-V1 1k at 1024 only.

## VAE facts (checked 2026-11-27)

- H3-Image VAE (`AutoencoderKLMiniMaxH3Image`, diffusers 0.43): causal 3D CNN encoder on one frame + **ViT decoder**: 24
  transformer blocks (1024 wide) on the 1/16 grid, 4 register tokens, RoPE on normalized [-1, 1] coordinates, then one
  linear projection to a 16x16x3 pixel patch. There is **no progressive upsampling stage** to borrow; "reuse the first
  decoder blocks" means reusing intermediate ViT token features (1024 ch at 1/16) as extra conditioning.
- Released tiling (256 px tiles, >= 64 px blend) is essential for the decoder: whole-image decode at 1024 collapses
  (PSNR 20.2, patch_fid 126) because the RoPE coordinates are resolution normalized. Encoder tiling barely matters
  (tiled slightly better; tiled is used for all data). Wrapper: `src/stdev/vaes/minimax_h3_image_vae.py` (also loads
  the ostris checkpoint, same encoder).
- Grid match: an H3 latent of a 1024 image is 64x64, exactly the low resolution input grid of Steppe-Flux.2 (512 px
  FLUX.2 latent 64x64 -> 128x128). Steppe-Flux.2 200M (`models/Steppe-Flux.2`, width 768, depth 18, source encoder
  512 wide with a pixel-shuffle 2x upsample) can be initialized as is except `source_encoder.input_conv`
  (32 -> 24 channels) and the bilinear-source half of `input_conv` (64 = 32 noisy + 32 upsampled source channels; there
  is no FLUX.2 low resolution latent here, so the residual base must be learned, e.g. a linear H3 -> FLUX.2 map).

## Dataset

`./train_datasets/h3_flux2_4klsdb_40k`: 4KLSDB 1024 center crops (65 parquet files, ~40.9k images, same source as
autoresearch_4 / 5), `block_<file>_<row>.vz.safetensors`, fp16 normalized latents:

| Key | Shape | Content |
|---|---|---|
| `latents_h3` | [1, 24, 64, 64] | condition: H3-Image encoder (fp32, tiled, posterior mode), config mean / std |
| `latents_flux2` | [1, 32, 128, 128] | target: FLUX.2 VAE (bf16, posterior mean), batch norm statistics |
| `hff_1024` | [1] | Immerkaer hff (unused) |

Generation: `torchrun --standalone --nproc_per_node=4 scripts/encode_h3_datasets.py --watch`.

## Benchmark

`./benchmark_datasets/axis_v1_1k_h3_flux2`: the 1000 Axis-V1 1024 crops (same ground truth as autoresearch_3-5),
`latents_h3.pt` [N, 24, 64, 64], `latents_flux2.pt` [N, 32, 128, 128]. Module `src/stdev/h3_benchmark/benchmark.py`;
references `python scripts/run_h3_benchmark.py --reference h3|h3_untiled|ostris|flux2|taef2 [--speed]`; results
`./raw_results/h3/<name>/metrics.json`. Metrics: patch_fid (main), fid, psnr (vs ground truth 1024), low_res_mse /
chroma ΔE00 (roundtrip to the 512 bilinear image), speed at 1024 / 2048 / 4096.

### References (Axis 1k, 1024 px, bf16 decoders)

| Decoder | Latents | patch_fid | fid | PSNR | low_res_mse | ΔE00 |
|---|---|---|---|---|---|---|
| H3 native (tiled) — **baseline** | H3 | 2.47 | 4.37 | 29.78 | 1.03e-3 | 2.29 |
| H3 native, untiled | H3 | 126.4 | 85.8 | 20.19 | 9.4e-3 | 6.47 |
| ostris fine-tune (tiled) | H3 | 2.92 | 2.85 | 31.02 | 7.4e-4 | 1.88 |
| FLUX.2 VAE (oracle) | FLUX.2 | 0.71 | 0.72 | 36.57 | 1.0e-4 | 1.08 |
| FLUX.2 small decoder (oracle) | FLUX.2 | 0.68 | 0.78 | 36.53 | 1.0e-4 | 1.12 |
| TAEF2 (oracle) | FLUX.2 | 2.25 | 3.62 | 32.87 | 3.2e-4 | 1.82 |

Speed (decode, batch 1, bf16): H3 native diffusers 134 / 637 / 2363 ms, all tiles batched 40 / 207 / 763 ms (fair
baseline), ~0.8-1.5 GB; FLUX.2 97 / 475 ms / OOM at 4096; TAEF2 8 / 36 / 148 ms but 10 GB at 4096 untiled. Details in
`experiment_logs.md`.

## Rules

- Scratch work in `./dev/autoresearch_6/`. Run files, infra and rounds follow the autoresearch_4 / 5 pattern
  (to be set up in round 1).
