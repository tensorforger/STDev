# FLUX.2 VAE throughput on RTX 5090

GPU: NVIDIA GeForce RTX 5090, torch 2.16.0.dev20261023+cu134, diffusers AutoencoderKLFlux2 (`models/FLUX.2-klein-4B/vae`), square RGB images, `torch.no_grad`, `cudnn.benchmark=True`.

Batch size was doubled from 1 until OOM, each point timed for >= 3 s after warmup, for eager and `torch.compile` (default mode). Encode uses the posterior mean. Script: `dev/benchmark_flux2_vae_speed.py`.

## Optimal throughput (images/s)

Best over batch sizes and eager/compiled.

| Resolution | dtype | Encode | Decode |
|---|---|---|---|
| 256 | fp32 | **313.8** (bs=64, compiled) | **149.4** (bs=64, compiled) |
| 256 | bf16 | **522.8** (bs=4, compiled) | **275.0** (bs=32, compiled) |
| 512 | fp32 | **73.9** (bs=32, compiled) | **30.3** (bs=32, compiled) |
| 512 | bf16 | **123.4** (bs=8, compiled) | **67.8** (bs=32, compiled) |
| 1024 | fp32 | **13.6** (bs=8, compiled) | **6.0** (bs=8, eager) |
| 1024 | bf16 | **26.5** (bs=8, compiled) | **15.8** (bs=16, compiled) |
| 2048 | fp32 | **2.0** (bs=2, compiled) | **1.1** (bs=1, eager) |
| 2048 | bf16 | **4.2** (bs=2, compiled) | **2.2** (bs=2, eager) |

## Full results (images/s, encode / decode)

### 256 px, fp32, eager

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 204.7 | 106.3 |
| 2 | 226.5 | 108.7 |
| 4 | 222.5 | 108.1 |
| 8 | 219.3 | 109.4 |
| 16 | 241.8 | 108.6 |
| 32 | 239.0 | 108.3 |
| 64 | 238.7 | 123.1 |

### 256 px, fp32, compiled

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 223.5 | 113.4 |
| 2 | 260.4 | 122.9 |
| 4 | 268.3 | 117.2 |
| 8 | 264.9 | 121.2 |
| 16 | 298.4 | 122.8 |
| 32 | 308.7 | 120.6 |
| 64 | 313.8 | 149.4 |
| 128 | 280.1 | 131.8 |

### 512 px, fp32, eager

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 49.6 | 25.4 |
| 2 | 50.3 | 24.3 |
| 4 | 51.3 | 25.6 |
| 8 | 51.9 | 26.2 |
| 16 | 56.6 | 26.1 |
| 32 | 56.6 | 26.2 |

### 512 px, fp32, compiled

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 54.2 | 27.6 |
| 2 | 63.3 | 26.0 |
| 4 | 62.1 | 26.9 |
| 8 | 65.0 | 27.2 |
| 16 | 73.7 | 27.1 |
| 32 | 73.9 | 30.3 |

### 1024 px, fp32, eager

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 10.1 | 5.7 |
| 2 | 10.7 | 5.6 |
| 4 | 11.0 | 5.9 |
| 8 | 11.1 | 6.0 |

### 1024 px, fp32, compiled

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 10.3 | 5.8 |
| 2 | 13.2 | 5.4 |
| 4 | 13.5 | 5.4 |
| 8 | 13.6 | 5.4 |

### 2048 px, fp32, eager

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 1.7 | 1.1 |
| 2 | 1.7 | 1.1 |

### 2048 px, fp32, compiled

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 1.6 | 1.1 |
| 2 | 2.0 | 1.0 |

### 256 px, bf16, eager

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 342.3 | 182.8 |
| 2 | 422.7 | 210.9 |
| 4 | 448.0 | 228.5 |
| 8 | 422.2 | 228.7 |
| 16 | 409.4 | 224.8 |
| 32 | 408.1 | 224.8 |
| 64 | 406.1 | 203.0 |
| 128 | 403.7 | 202.7 |

### 256 px, bf16, compiled

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 426.1 | 221.3 |
| 2 | 494.3 | 251.5 |
| 4 | 522.8 | 264.5 |
| 8 | 511.5 | 269.5 |
| 16 | 509.4 | 271.7 |
| 32 | 510.5 | 275.0 |
| 64 | 508.3 | 238.6 |
| 128 | 509.5 | 240.3 |
| 256 | 495.4 | 126.0 |

### 512 px, bf16, eager

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 95.0 | 48.4 |
| 2 | 97.3 | 47.1 |
| 4 | 92.8 | 51.3 |
| 8 | 98.3 | 54.8 |
| 16 | 95.6 | 55.1 |
| 32 | 92.7 | 54.5 |

### 512 px, bf16, compiled

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 112.1 | 54.2 |
| 2 | 122.4 | 57.2 |
| 4 | 122.5 | 57.6 |
| 8 | 123.4 | 58.2 |
| 16 | 122.9 | 58.4 |
| 32 | 122.4 | 67.8 |
| 64 | 118.1 | 39.8 |

### 1024 px, bf16, eager

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 19.4 | 10.7 |
| 2 | 20.9 | 11.0 |
| 4 | 21.6 | 12.3 |
| 8 | 21.6 | 12.9 |

### 1024 px, bf16, compiled

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 24.7 | 12.1 |
| 2 | 26.1 | 12.1 |
| 4 | 26.3 | 12.1 |
| 8 | 26.5 | 11.9 |
| 16 | 25.6 | 15.8 |

### 2048 px, bf16, eager

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 3.5 | 2.2 |
| 2 | 3.6 | 2.2 |

### 2048 px, bf16, compiled

| Batch size | Encode | Decode |
|---|---|---|
| 1 | 3.7 | 2.2 |
| 2 | 4.2 | 2.2 |
