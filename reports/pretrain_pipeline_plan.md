# Pretraining pipeline plan (8x RTX 5090, DDP)

Three stage latent upscaler pretraining: 256->512 and 512->1024 on ReLAION-high-resolution,
1024->2048 (SFT-like, mixed with earlier stages) on 4KLSDB + ReLAION >= 2048.

The document has three parts:

- **A. Measured**: numbers measured on this machine (2026-10-30), with the script that produced them.
- **B. Derived**: estimates computed from A. Marked "est." where something was extrapolated.
- **C. Plan**: what we are going to do, and why.

---

## A. Measured

### A1. Download: ReLAION-high-resolution

Script: `dev/relaion_download_speed.py` (the notebook downloader split over processes, 90 s
windows, images validated with PIL `verify`, not saved). Shard 0, different url ranges per run.

| Total concurrency | Images/s | MB/s | Alive links |
|---|---|---|---|
| 256 (1 process, notebook setting) | 31 | 14 | 41% |
| 2048 (8 processes x 256) | 381 | 189 | 55% |
| 8192 (32 processes x 256) | 648 | 290 | 29% (timeouts and http 429 rise) |

Mean image size 0.45-0.50 MB. The notebook's 12.7 images/s (other machine) was limited by
concurrency, not by the network.

Metadata of shard 0 (1.3 M rows): min image side >= 1024 for 100%, >= 1536 for 28.8%,
>= 2048 for 11.7%, >= 3000 for 3.5%; pwatermark > 0.5 for 14.5%, punsafe > 0.5 for 1.7%.
The dataset has 128 such shards.

### A2. 4KLSDB size

Hugging Face metadata: `train_x4` split is 879 GB in 206 parquet files at ~7 MB per row
(4K HR + LR image), ~125 k images. `train_x8` / `train_x16` (823 / 807 GB) are other degradation
scales.

### A3. CPU preprocessing

Script: `dev/pretrain_cpu_preprocessing_timing.py`, one OpenCV thread, 3 JPEGs (1600x1053 / 376 KB,
2127x1400 / 2.8 MB, 4544x2805 / 2.4 MB).

| Step | ms per image |
|---|---|
| JPEG decode | 5 / 33 / 61 |
| Area resize to HR + LR, 512/256 | 8-44 |
| Area resize to HR + LR, 1024/512 | 14-52 |
| Area resize to HR + LR, 2048/1024 | 77 |
| High frequency metrics on HR at 512 / 1024 / 2048 | 11 / 60-65 / 207 |

### A4. VAE encode

From `reports/flux2_vae_speed.md` (FLUX.2 VAE, one GPU, bf16, compiled, best batch):
256: 523 images/s, 512: 123, 1024: 26.5, 2048: 4.2.

### A5. Training throughput

Script: `dev/pretrain_training_throughput.py`, results in `dev/data/pretrain_training_throughput.jsonl`.
One GPU, synthetic batches already on the GPU, run_0078 model code with overrides,
`torch.compile`, bf16 autocast, fused AdamW, grad clipping. Large variants use decoder blocks
(8, 2, 2), window transformer blocks after every block on all levels.

| Model | Params | HR px | Batch | Grad ckpt | Samples/s | Peak memory |
|---|---|---|---|---|---|---|
| 0078 (512,512,1024) dec (3,2,2) | 304M | 512 | 64 | yes | 73.6 | 17.3 GB |
| | | 1024 | 4 | no | 22.3 | 13.7 GB |
| | | 1024 | 8 | no | 23.4 | 22.3 GB |
| | | 1024 | 16 | yes | 18.2 | 17.3 GB |
| | | 1024 | 24 | yes | 19.3 | 23.9 GB |
| | | 2048 | 4 | yes | 4.4 | 17.3 GB |
| 768_dec8 (768,512,1024) | 445M | 512 | 16 | no | 28.6 | 29.4 GB |
| | | 1024 | 2 | no | 7.4 | 18.4 GB |
| | | 1024 | 4 | no | 7.8 | 29.4 GB |
| | | 1024 | 8 | yes | 6.2 | 21.6 GB |
| 1024_dec8 (1024,512,1024) | 576M | 512 | 8 | no | 20.2 | 23.8 GB |
| | | 1024 | 2 | no | 5.0 | 23.8 GB |
| | | 1024 | 2 / 4 | yes | 4.0 / 4.1 | 12.8 / 17.9 GB |
| | | 2048 | 1 | yes | 1.07 | 18.2 GB |
| 1280_dec8 (1280,640,1280) | 887M | 1024 | 1 | no | 3.3 | 23.8 GB |
| | | 1024 | 2 / 4 | yes | 2.7 / 2.8 | 18.4 / 24.7 GB |

Out of memory (32 GB): 768_dec8 none tested; 1024_dec8 at 1024 px batch 4 without and batch 8
with checkpointing; 1280_dec8 at 1024 px batch 2 without and batch 8 with checkpointing.

### A6. Gradient all-reduce

Script: `dev/allreduce_bandwidth.py` (torchrun, 8 processes, NCCL). Bus bandwidth 25.8 GB/s for
25-1600 MB messages, bf16 and fp32 alike. Topology: PCIe only, GPUs 0-3 on NUMA node 0,
4-7 on node 1, no NVLink.

### A7. Machine

2x AMD EPYC 9B14 (384 threads), 503 GB RAM, 480 GB disk; `/workspace` is NOT a persistent volume.

### A8. From the research log (`autoresearch/experiment_logs.md`)

- LR decay to zero at the end is worth 0.3-0.8 fid_patch; checkpoints before the decay are only a
  rough signal of the curve shape (round 10).
- fid_patch noise: ~0.03 between training seeds, ~0.05 between inference seeds, with 1000 samples.
  500 sample evaluations are biased and too noisy (round 7).
- EMA was neutral, but only tested on runs of <= 2 h that already had LR decay.
- Only batch 16 was tested; larger batches untested at equal samples.
- The FLUX.2 VAE is not flip equivariant: no flip augmentation on latents.

---

## B. Derived

### B1. Per sample costs (single GPU)

| Stage | VAE encode LR + HR | Train 768_dec8 | Train 1024_dec8 | Online encode overhead (768 / 1024) |
|---|---|---|---|---|
| 1: 256 -> 512 | 10.0 ms | 35 ms | 50 ms | 29% / 20% |
| 2: 512 -> 1024 | 45.8 ms | 128 ms | 200 ms | 36% / 23% |
| 3: 1024 -> 2048 | 276 ms | ~590 ms (est.) | 935 ms | 47% / 30% |

768_dec8 at 2048 px (1.7 samples/s) is extrapolated from the 1024_dec8 1024 -> 2048 ratio.

### B2. Observations

1. Training cost scales with HR pixel count: x4 per stage (0078: 73.6 / 18.2 / 4.4 at equal
   pixels per batch; 1024_dec8: 20.2 / 5.0).
2. Training cost is driven by the full resolution width, not by the parameter count: 768 / 1024 /
   1280 wide cost 1.0 / 1.56 / 2.4x at 1024 px. 887M (1280 wide) only fits batch 1 at 1024 px
   without checkpointing: not practical on 32 GB cards.
3. Gradient checkpointing costs 20-30% throughput.
4. Download (~400 images/s) and CPU preprocessing (~2000+ images/s on 384 threads) are never the
   bottleneck: the fastest training stage consumes ~180 samples/s.
5. All-reduce of 445M bf16 gradients takes ~60 ms (fp32 ~120 ms) against a ~510 ms step at
   batch 4 / 1024 px: mostly hidden behind backward, < 10% with gradient accumulation 2.
6. AdamW state is 16 B/param (445M: 7 GB, 576M: 9 GB) before activations.

### B3. Encoder GPUs vs training GPUs

DDP efficiency assumed 0.9. "Fresh" = share of training samples that can be newly encoded; the
rest are reused from the cache (average uses per sample = 1 / fresh).

| Stage | Split (encode + train) | Encode /s | Train /s, 768 / 1024 | Fresh, 768 / 1024 |
|---|---|---|---|---|
| 1 | 1 + 7 | 100 | 180 / 127 | 55% / 79% |
| 2 | 1 + 7 | 22 | 49 / 32 | 45% / 69% |
| 2 | 2 + 6 | 44 | 42 / 27 | 100% / 100% |
| 3 | 1 + 7 | 3.6 | 11 / 6.7 | 33% / 54% |
| 3 | 2 + 6 | 7.2 | 9.2 / 5.8 | 78% / 100% |

Download needed per encoder GPU (after filtering): stage 1 ~100/s; stage 2 ~75/s (only the 29%
with min side >= 1536 qualify); stage 3 ~31/s (11.7% >= 2048). All below 400/s.

### B4. Storage per sample (8 bit latents, + ~4 KB pooled text, caption, HF stats, metadata)

| Stage | Bytes per sample | Samples per 400 GB local cache | Local cache lasts (768, 1 + 7) |
|---|---|---|---|
| 1 | ~164 KB | ~2.4 M | ~3.7 h of training |
| 2 | ~644 KB | ~0.6 M | ~3.5 h |
| 3 | ~2.6 MB | ~150 k | ~4 h |

### B5. Evaluation cost (est.)

fid_patch on 1000 samples, 20 Euler steps: ~20 k forward passes of a 445M model (~25 ms each at
1024 px, extrapolated from run_0078 latency 14.5 ms) + 1000 VAE decodes (~60 s) + Inception:
~10 minutes on one GPU.

### B6. Example: what a stage costs in time (768_dec8, 1 + 7)

At 180 samples/s, stage 1 sees 15.6 M samples/day from 8.6 M newly encoded images/day. At
49 samples/s, stage 2 sees 4.2 M samples/day (1 + 7) from 1.9 M new images/day, or a single
pass at 3.6 M/day with 2 + 6. Stage lengths are not fixed in advance (see C3).

---

## C. Plan

### C1. Decision: online producer with a rolling latent cache, adaptive stage lengths

The model is a local 2x upscaler, not a general generative model; it may saturate long before a
fixed data budget is spent, and we do not want to guess the stage lengths. Instead of encoding a
fixed dataset up front, a dedicated encoder process produces latents continuously, training
reads from a rolling cache, and each stage ends when the fid_patch curve flattens.

Rejected alternatives:
- Offline build of a fixed dataset (~30 M / 8 M / 0.6 M images, ~1.1 days on 8 GPUs before
  training): cheaper in GPU time (~8% vs 12.5-25%) but fixes the data amount in advance.
- Pure online (encode every training sample, no cache): 20-47% overhead per sample (B1) and the
  encoder cannot match the trainers anyway (B3), so samples must be reused regardless.

### C2. Pipeline layout

```
download (CPU, 16-32 procs x 256 conns) -> filter -> decode / crop / resize / HF stats (CPU, ~200 procs)
    -> encoder GPU(s): FLUX.2 VAE (bf16, compiled) + CLIP-L pooled text
    -> rolling cache (local disk ~400 GB, every shard also uploaded to R2)
    -> trainer GPUs (DDP): sample a mix of new and cached latents
```

- **GPU split per stage:** stage 1: 1 encoder + 7 trainers. Stages 2 and 3: 2 encoders + 6
  trainers (single pass / mostly fresh data, B3). The split is a launch parameter; changing it
  means restarting the trainer at a checkpoint.
- **Encoder GPU also runs evaluation** (C4), ~5% of its time.
- **Filters:** pwatermark < 0.5, punsafe < 0.5; stage 2: min side >= 1536; stage 3: min side >= 2048
  (plus 4KLSDB). Watermarks are high frequency patterns the upscaler would learn.
- **Per sample stored:** LR and HR latents (vaezip lbrq_8bit), pooled CLIP-L embedding (fp16,
  1.5 KB), caption, high frequency stats of the HR image, source size and crop box. Full 77 token
  embeddings are not stored (118 KB per sample, token level text gave no gain in research).
- **Cache policy:** shards of ~1 GB; the trainer samples uniformly over the local cache window;
  oldest shards are evicted when the disk fills. Every shard is kept in R2 for stage 3 replay and
  for restarts (`/workspace` is not persistent). R2 or B2 rather than AWS S3 to avoid egress fees.
- **Stage transition:** when a stage is about to end, the encoder switches to the next stage's
  resolution pair and filter, so the next stage starts with a warm cache.
- **Process placement:** encoder(s) on one NUMA node's last GPU(s); DataLoader workers pinned to
  the NUMA node of their GPU.

### C3. Training schedule without a fixed length

- Warmup, then **constant LR** (no decay planned in advance).
- **EMA weights** for evaluation (rough stand-in for an annealed model).
- **Stopping rule per stage:** EMA fid_patch improves by less than ~0.1 (2x inference seed noise)
  over the last ~12 h -> start the **decay phase**: linear decay to zero over 10-20% of the steps
  done in that stage, then evaluate the annealed model and move on.
- **Optional check, ~daily:** fork a short decay from the current checkpoint (on the encoder GPU
  or by pausing) to confirm that the annealed quality still improves, not only the EMA.
- Next stage starts from the annealed weights with a new short warmup.

### C4. Evaluation

- fid_patch with 1000 samples, every ~3-4 h, on the encoder GPU (~10 min each, B5).
- Validation sets fixed before training:
  - held-out ReLAION images at the stage resolution pair (real photos, matches training data),
  - flux_1024 benchmark (teacher images, comparable to the research rounds),
  - stage 3: held-out 4KLSDB at 1024 -> 2048.
- Log train loss, fid_patch (EMA), and the stage's sample count / reuse factor.

### C5. Training configuration

- DDP on the trainer GPUs, bf16 autocast, fused AdamW, bf16 gradient compression hook,
  `ZeroRedundancyOptimizer` to shard optimizer state.
- Micro-batch at equal pixels: 768_dec8: 16 @ 512, 4 @ 1024 (29.4 GB, tight: ZeRO-1 sharding of the Adam moments frees ~3 GB),
  1 @ 2048. Gradient accumulation to a global batch of 64-128.
- Stage 3 mixing: one resolution per optimizer step for all ranks, pixel-balanced micro-batches
  (1 @ 2048 = 4 @ 1024 = 16 @ 512), e.g. 60% 1024 -> 2048, 25% 512 -> 1024, 15% 256 -> 512 replayed
  from R2.

### C6. Open items before launch

1. **Model size:** 768 wide (445M) vs 1024 wide (576M): two ~12 h stage 1 runs on 4 GPUs each,
   same data, compare fid_patch per hour. 1280 wide is excluded (B2.2).
2. **Batch size and LR** at global batch 64-128: short sweep on stage 1 (research only used 16).
3. **EMA decay** for long runs (untested beyond 2 h).
4. **Latent cropping:** if crop(encode(image)) ~= encode(crop(image)) away from borders for the
   FLUX.2 VAE, store larger latents (e.g. full 4K for 4KLSDB) and take random crops at training time;
   HF stats would then need a per tile map.
5. **Sustained download** over hours (DNS, rate limits, alive rate per shard).
6. **Object storage** account and bucket (R2 / B2) for the shard upload.
