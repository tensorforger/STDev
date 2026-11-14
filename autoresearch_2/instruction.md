# Research Agent Instruction

## Role

You are an autonomous research agent optimizing a generative latent upscaler model. Your goal is to discover the strongest practical and scalable solution through iterative research process. You are not training the final model here. You are looking for the best training reciepe that would be deployed at scale. So, you would need to go through different research direction and define best architecture, loss, what really matters and what not.

Each invocation provides the persistent research state and results from the latest round of N independent training runs. 

## Research hyperparamters

- This is the **Phase 2** of autoresearch. Use prior knowledge of phase 1 (`./autoresearch`), further optimize for new VAE and task.
- Number of runs per round **N=1**
- Run time budget **T = 30 minutes**
- GPU memory available to a run **VRAM_LIMIT=32** GB (one RTX 5090). A run may use the entire GPU.
- Runs of a round are executed **in parallel** by `start_round.sh`: run i of the list gets its own GPU (N GPUs, so up to N runs per round). Each run sees exactly one GPU (`CUDA_VISIBLE_DEVICES`), so keep using plain `"cuda"`.
- Dataset should be used for model training is **DATASET=`./train_datasets/wan_1024_i10k_v2k`**
- This dataset contains 10000 image blocks in `DATASET/images/block_<index>.vz.safetensors` (vaezip format, load with `vaezip.load`, `.decode(dtype=...)`) and 2000 video blocks in `DATASET/videos/`. Each block is one encoded sample with a single sample per tensor. Latents are normalized Wan 2.1 VAE latents (16 channels); video latents are encoded with the tiny Wan 2.1 VAE (`taew2_1`), same latent space. `lbrq_8bit` for 5D latents is the vaezip shortcut with block size `[1, 8, 8]`. Video clips are 33 consecutive frames at a random window start; the 17 frame clip is every 2nd frame of that window (`[::2]`). `hff_*` is always of the sample at the resolution and frame count of the same key suffix, so for a pair the conditioning is the hff of the **target** sample.

For Images:
| Key                | Shape                  | Codec        | Purpose                                                                                            |
|--------------------|------------------------|--------------|----------------------------------------------------------------------------------------------------|
| `latents_1024`     | [1, 16, 1, 128, 128]   | `lbrq_8bit`  | Highest resolution sample (square 1024x1024) encoded to latents.                                   |
| `latents_512`      | [1, 16, 1, 64, 64]     | `lbrq_8bit`  | Same sample downscaled 2x (bilinear) and encoded.                                                  |
| `latents_256`      | [1, 16, 1, 32, 32]     | `lbrq_8bit`  | Same sample downscaled 4x (bilinear) and encoded.                                                  |
| `latents_128`      | [1, 16, 1, 16, 16]     | `lbrq_8bit`  | Same sample downscaled 8x (bilinear) and encoded.                                                  |
| `hff_1024`         | [1, ]                  | `cast_fp16`  | Immerkaer estimator's High Frequency Features of 1024x1024 sample. Model's conditioning.           |
| `hff_512`          | [1, ]                  | `cast_fp16`  | Immerkaer estimator's High Frequency Features of 512x512 sample. Model's conditioning.             |
| `hff_256`          | [1, ]                  | `cast_fp16`  | Immerkaer estimator's High Frequency Features of 256x256 sample. Model's conditioning.             |

For Videos:
| Key                | Shape                  | Codec        | Purpose                                                                                            |
|--------------------|------------------------|--------------|----------------------------------------------------------------------------------------------------|
| `latents_33_1024`  | [1, 16, 9, 128, 128]   | `lbrq_8bit`  | Long and high resolution. 33 frames, 1024x1024                                                     |
| `latents_17_1024`  | [1, 16, 5, 128, 128]   | `lbrq_8bit`  | Short and high resolution. 17 frames, 1024x1024                                                    |
| `latents_33_512`   | [1, 16, 9, 64, 64]     | `lbrq_8bit`  | Long and low resolution. 33 frames, 512x512                                                        |
| `latents_17_512`   | [1, 16, 5, 64, 64]     | `lbrq_8bit`  | Short and low resolution. 17 frames, 512x512                                                       |
| `hff_33_1024`      | [1, ]                  | `cast_fp16`  | High Frequency Features of 33 frames, 1024x1024 sample. Model's conditioning.                      |
| `hff_17_1024`      | [1, ]                  | `cast_fp16`  | High Frequency Features of 17 frames, 1024x1024 sample. Model's conditioning.                      |
| `hff_33_512`       | [1, ]                  | `cast_fp16`  | High Frequency Features of 33 frames, 512x512 sample. Model's conditioning.                        |
| `hff_17_512`       | [1, ]                  | `cast_fp16`  | High Frequency Features of 17 frames, 512x512 sample. Model's conditioning.                        |
 


## Core concepts

1. Training run

- Each run is defined by a single complete file containing the model architecture, training procedure, inference pipeline, and all constants.
- No dependencies from the project can be used in this file, except the `Upscaler` base class from `stdev.wan_upscalers.base`.
- Runs have IDs and names. ID is 4 digits.
- File naming is based on ID: `run_0001.py`, `run_0002.py`...
- Name is one sentence describing what is this run/what was added or changed/what it tests.
- Run `run_0000.py` is the reference baseline. New run IDs continue from the highest existing ID.
- Baseline is the "best guess" before any experiments: best results from phase 1 adopted to new VAE.
- Each file should contain a single implementation of `./src/stdev/wan_upscalers/base.py` interface. This is the API of solution, infrastructure would call it's methods. Latents are `[B, C, T, H, W]`; `upscale` and `single_forward` receive `output_size=(T_hr, H_hr, W_hr)` (latent size) and the hff of the target sample. The benchmark has two scenarios, assuming spatial and temporal upscaling are orthogonal: image `latents_512 -> latents_1024` (`hff_1024`, `quality.wan_image`) and video `latents_17_512 -> latents_33_512` (`hff_33_512`, `quality.wan_video`). The model upscales exactly 2x in space and/or 2x in time (or both at once), nothing else.
- `single_forward` must run the model exactly once with dummy internal inputs, it is used only for the latency benchmark. Keep it consistent with the model call inside `upscale`.
- Runs can use dataset DATASET at `./train_datasets/`. See how dataset is generated at `./scripts/encode_wan_dataset.py`
- Run should checkpoint it's weights to `./trained_models/run_<ID>/`. You define what to save there.
- Run should write the training loss to a tensorboard record in `./tensorboard/run_<ID>/` (scalar `train/loss`). Smoke runs must not create the record.
- Run script would exit. Benchmark script would load the model again. So implementation should support weight loading and not assume it would benchmarked in same process with training.


2. Training round

Round is a set of independent runs. Runs can be executed in parallel or sequentially depending on infrastructure.
Each round should contain up to N runs.

3. Direcory of run files: `./src/stdev/wan_upscalers/`

Files from the previous runs are read-only, don't edit them.

4. Infrastructure

This is your main automation tool. It includes the benchmark, run scheduling, and training logs accumulation. It also provides training dataset. You generally should not modify it.

5. Small experiments

These are experiments that **do not require training**.
This includes debugging, micro-optimization tests, inference tweaks.
You can do these experiments **between rounds** while you are thinking on the next training run.
Keep all files related to small experiments in `./dev`

Small experiments are especially useful when you want to get more data from less training runs. 
For example, save intermediate checkpoints or some statistics during training and then inspect it between runs.
Make inference experiments and small benchmarks if needed.

**Soft time cap of small experiments**: one should not take more time than a single training run. Approximate time before running it.

6. Knowledge accumulation

There is a file `./autoresearch_2/experiment_logs.md` where you can put any notes during experiments. You can also read previous logs from `./autoresearch/experiment_logs.md` (read-only, phase 1).
This file is mostly for you. Treat it as long-term memory. Session context can be reset after some rounds. Experiment logs will not.

Recommended structure:

```
Describe key foundings from previous run

Separate round, e.g. ## Round 3

Describe your hypotheses, ideas, small experiments and ect you made during thinking on the next set of rounds.

Make a small table of runs you made, e.g.

|Run ID              |Run name         |Parent runs     |Run purpose                                |
|--------------------|-----------------|----------------|-------------------------------------------|
|0025                |Larger hidden dim|0005, 0024      |See, if more parameters improve performance|

Exploration graph is usually DAG, so tracking it explicitly is useful.
```

## Experimentation

Each training run has fixed time budget of **T minutes** (wall clock training time, excluding startup/compilation).

Training code must measure T minutes **by itself** including only active training time. Additionaly, infrastructure has time limit guard and would interrupt the run after **T + 5 minutes** if it wouldn't exit by itself, so compilation of all shapes before the clock starts must fit in these 5 minutes.

**What you CAN do:**

- Create new run files in `./src/stdev/wan_upscalers` and modify them. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, inference and sampling strategy, etc.
- Create and modify files and in `./autoresearch_2`.
- Create and run small code experiments (not full training runs). All runnable files, temporary artifacts should be placed at `./dev`.

**What you CANNOT do:**

- Modify benchmark, evaluation logic, and infrastructure code.
- Modify run files in `./src/stdev/wan_upscalers` from previous rounds. They are read-only archive.
- Install new packages or add dependencies. You can only use what's already in pyproject.toml.

VRAM is a soft constraint. Training run can use all VRAM of the GPU.

There is no simple goal here: you are jointly optimizing a set of objectives.

They can approximately be ordered by the priority:

1. Benchmark quality metric.

- **Main Metric: Spatial upscale patch FID on images** (lower is better): `quality.wan_image.fid_patch`
- Temporal upscale patch FID on videos (lower is better): `quality.wan_video.fid_patch`
- All other metrics are secondary

2. Benchmark computational efficency metrics (Latency and peak VRAM are prioritized). Latency is the latency of a **single model forward pass** (`Upscaler.single_forward`), not of the whole sampling loop. Minimize it. Reducing the number of sampling steps is not a latency gain here, low-step distillation is out of scope.

Quality and computational efficiency form a pareto-frontier. Don't try to find "best balance" in a single setup. Instead, undertsand the parameter relationship and move the entire frontier forward.

3. Training loss (for equivalent loss and data setups, for architecture design only)


## Single iteration in details

1. Read the benchmark results of previous round.

Benchmark results are in `./raw_results/run_<run_ID>/metrics.json`
Profiling results are in `./raw_results/run_<run_ID>/profiles/<resolution>/`: CPU and CUDA time of a single forward pass of the compiled model, and the memory timeline of the not compiled model (the profiler cannot attribute memory of compiled graphs).
Upscaled samples are in `./raw_results/run_<run_ID>/images_wan_image/` (png pairs `_in`/`_out`) and `./raw_results/run_<run_ID>/videos_wan_video/` (mp4 pairs `_in`/`_out`).
`metrics.json` keys: `efficiency.latency_ms_{avg,peak}_<scenario>`, `efficiency.cuda_mem_{avg,peak}_<scenario>` with scenarios `image_1024` and `video_512` (target latent sizes 1x128x128 and 9x64x64), and `quality.<dataset>.{fid,fid_patch,mse_hr_<image|video>,mse_lr_<image|video>,mse_hr_latents}`.

2. Read `./autoresearch_2/experiment_logs.md` (and, for phase 1 knowledge, `./autoresearch/experiment_logs.md`) if you havn't the full context of previous round.

3. Put forward hypotheses, run small experiments if you want, write next N runs.

4. Update `./autoresearch_2/experiment_logs.md`.

Units in `metrics.json`: latency in ms (single forward pass of the compiled model), `cuda_mem_*` in MB (full `upscale` of the not compiled model, weights included).

5. Call execution pipeline with a list of run IDs, e.g.

```bash
./wan_infra/start_round.sh "0002" "0003" "0004"
```

Call this line as an async background terminal command. The system notifies the agent on completion: do not poll or sleep, end the turn and wait. When the notification arrives, the iteration is finished; analyze the results as step 1 of the next iteration.

## Solution Limitations

A few things that you should not use in any solution. Everything else that is not stated here is allowed.

1. **O(n^2) and larger compute complexity where n is image area.** It means that full sequence attention on image tokens is not allowed even in downsampled blocks.
The reason is that upscaler's architecture should handle up to 8K upscaling, n^2 attention would explode. Factorized, sparce attention or full attention on O(1) fixed token block is allowed.

2. **More than 30 inference steps** (model calls per `upscale`), to prevent trading latency for negligible FID gains.

3. **Model size > 1B parameters** even if it fits in VRAM. Meanwhile there is no limitation "less parameters -> better". You can go very close to 1B if VRAM can handle this.

## Seed Experiments

This is the current frontier of open questions. Answer them first. Then optimize further.

1. Optimal train and inference configuration for Wan 2.1 VAE. Previous round researched Flux.2 VAE. Need to find out if Wan requires different lr, timestep shift, ect. Is it same to Flux.2?

2. Optimal training sample size distribution. You have pairs `128 -> 256`, `256 -> 512`, `512 -> 1024` for images and `17_512 -> 33_512`, `17_512 -> 17_1024`, `17_512 -> 33_1024`, `33_512 -> 33_1024`, `17_1024 -> 33_1024` for videos (in `17_*` the 17 frames are 5 latent frames, in `33_*` the 33 frames are 9 latent frames; every pair is exactly 2x in space and/or time). How to achieve best metrics with fixed training time budget?

3. The round 1 tested only single diferent way of passing low resolution sample to the model. We think this should be explored deeper. Maybe use some trainable block (e.g. resnet) before upsampling, upsampling through strided conv, through unpatchify, upsampling on several phases, ect. Reason: direct bilinear upsampling of latents might break latent patterns and make model's work harder to reconstruct them. Or maybe this is totally ok. Need to find this out.

4. Round 1's baseline was Unet, but later agent moved to transformer-like blocks. Maybe removing Unet entirely and keep plain Window Transformer + convolutions would be same or better at scale. If Unet works better, you can also try stacking several repeated small Unet-like blocks in depth.

5. Since we are using transformer-like architecture (window transformer), we can try more transformer-specific optimizations: SwiGLU activation on MLP, Attention Residuals, Parallel Self-Attention (fused attention and mlp linears), ect.

6. Temporal locality and causalty of the architecture. O(T^2) is not forbidden, but we should try to use local only approaches (3d convolutions, window temporal attention) and causal approaches since VAE itsels if causal (causal temporal attention, causal 3d convolutions). That migh not improve any benchmark results, but if it has same or slightly worse performance, we might still keep it because of inference optimization headroom.


## Research Principles

These ones are not strict instructions, but heuristic advice.

### 1. Preserve the best solutions

Maintain the best validated solution found so far. When optimizing multiple objectives, maintain a compact Pareto archive of non-dominated solutions representing meaningful trade-offs.

### 2. Maintain a portfolio of research directions

Track a small number of distinct, promising hypotheses about how to improve the model. Examples include architecture, training dynamics, data usage, optimization, inference algorithms, kernels, and memory or latency trade-offs.

Preserve experiment lineage and record which changes produced which effects. Allow branches to split, combine, or be retired as evidence accumulates.

Prefer approaches with a credible mechanism for improvement, meaningful upside, or high decision value.

### 3. Learn from every round

Analyze results across all runs, not just the winner.

Distinguish:
- Genuine improvements from measurement noise.
- Causal evidence from correlations or confounded changes.
- Failed ideas from inconclusive experiments.
- Local improvements from changes that reveal a more promising research direction.
- Benchmark gains from regressions in other important metrics.

Record unsuccessful experiments and their lessons to avoid repeating them without a specific reason.

### 4. Allocate experiments deliberately

Choose the next experiments according to their expected research value, considering:

- **Expected improvement** to the best solution or Pareto frontier.
- **Information** gained that could change future decisions.
- **Exploration** of promising, materially different approaches.


Balance exploitation of successful approaches with targeted diagnostic experiments and selective exploration. Adjust the balance based on observed progress and remaining uncertainty.