# Experiment logs

## Round 0

Run 0000: flow-matching conditional UNet (96 base channels, 3 levels), 30 min, 31k steps, EMA, Euler 30 steps.
Results (axis_1024): FID 9.89, RQI 0.501, latency 24.5 ms at 1024, peak mem 30.8 MB at 1024 (units: ms, MB). Training loss ~0.695.
Old bilinear baseline.py and its results were removed.

## Round 1 (debug-mode round)

Two simple variants of 0000 to exercise the pipeline.

|Run ID|Run name                      |Parent runs|Run purpose                                          |
|------|------------------------------|-----------|-----------------------------------------------------|
|0001  |Uniform time sampling         |0000       |See if uniform t instead of logit-normal helps FID  |
|0002  |Wider UNet (128 base channels)|0000       |See if more parameters improve FID within 30 minutes |

Results on axis_1024:

|Run ID|FID ↓|RQI ↑|HR latent MSE ↓|1024 latency (ms) ↓|1024 peak VRAM (MB) ↓|Steps|Final 100-step loss|
|------|-----|-----|---------------|-------------------|---------------------|-----|-------------------|
|0000  |9.887|0.501|0.4505         |24.49              |30.81                |31347|0.6950             |
|0001  |10.627|0.501|0.4554         |24.55              |30.81                |31655|0.7001             |
|0002  |8.876|0.506|0.4444         |30.40              |41.14                |23736|0.6662             |

Increasing base channels from 96 to 128 improved FID by 1.01 and reduced the comparable training loss, at the cost of 24% higher 1024 latency. Run 0002 is the quality leader. Uniform time sampling worsened FID by 0.74 relative to the otherwise identical run 0000, so subsequent runs keep logit-normal sampling. RQI differences remain below the 0.01 noise threshold.

## Round 2

N=1 for this and later rounds. The capacity result motivates testing a stronger model, but throughput and training augmentation need deliberate trade-offs because run 0002 already sees about 76 dataset passes in 30 minutes.

Small experiment before the next training run: `dev/round_2/sampler_sweep.py` evaluates run 0002 on all 1000 benchmark images with 30/16/10/6 full-trajectory Euler steps and condition-initialized trajectories starting at t=0.75/0.5/0.25. This tests the quality-latency frontier without retraining; FID, HR latent MSE, and HR image MSE are recorded in `dev/round_2/sampler_results.json`.

### Sampler results

The custom full-30 evaluation reproduced the official FID (8.8726 versus 8.8757), validating the comparison. Fewer full-trajectory steps improve both FID and reconstruction until the trajectory becomes too coarse:

|Euler steps|FID ↓|HR latent MSE ↓|HR image MSE ↓|
|-----------|-----|---------------|--------------|
|30         |8.873|0.4444         |0.004357      |
|16         |8.515|0.4358         |0.004276      |
|10         |8.117|0.4257         |0.004181      |
|8          |7.891|0.4197         |0.004123      |
|6          |7.583|0.4106         |0.004028      |
|4          |7.332|0.3960         |0.003870      |
|3          |7.416|0.3859         |0.003745      |
|2          |7.817|0.3792         |0.003600      |
|1          |10.268|0.4500        |0.004108      |

Four steps is the nominal FID optimum, while three and two steps trade slightly higher FID for lower MSE and latency. Condition initialization is retired for this model: starts at t=0.75/0.5/0.25 give FID 9.466/13.883/32.065. These are same-seed, full-benchmark comparisons, but differences below 0.5 FID are still treated as noise for model selection claims.

### Training throughput microbenchmark

`dev/round_2/training_throughput.py` measures synthetic full training steps at 128x128 latents, batch 16. Measurements exclude EMA and data loading:

|Base channels|Precision / compile|Parameters|ms/step|Samples/s|Peak VRAM (GB)|
|-------------|-------------------|----------|-------|---------|--------------|
|128          |fp32 / no          |23.0M     |70.4   |227      |5.50          |
|128          |bf16 / no          |23.0M     |52.4   |305      |4.49          |
|128          |bf16 / yes         |23.0M     |40.5   |395      |2.43          |
|160          |bf16 / no          |35.5M     |79.6   |201      |5.67          |
|192          |bf16 / no          |50.7M     |97.6   |164      |6.90          |
|192          |bf16 / yes         |50.7M     |79.0   |202      |3.75          |
|128, batch 32|bf16 / no          |23.0M     |107.3  |298      |8.58          |

The 192-channel compiled configuration approximately preserves run 0002's measured samples/s while providing 2.2x parameters and fits comfortably in VRAM. Compilation warmup is excluded from the run's internal 30-minute active-training timer but remains inside the infrastructure's 35-minute limit.

|Run ID|Run name|Parent runs|Run purpose|
|------|--------|-----------|-----------|
|0003  |Wider bf16-compiled UNet with four-step sampling|0002|Test 192 channels at comparable training throughput, with the independently validated sampler|

Run 0003 passed smoke training and the complete smoke benchmark. Fine-step sweep values are stored in `dev/round_2/sampler_fine_results.json`; throughput values are in `dev/round_2/training_throughput.json`.

### Run 0003 results

|Run ID|Parameters|Steps|Final 100-step loss|FID ↓|RQI ↑|HR latent MSE ↓|1024 latency (ms) ↓|1024 peak VRAM (MB)|
|------|----------|-----|-------------------|-----|-----|---------------|-------------------|-------------------|
|0003  |50.75M    |21129|0.6252             |6.281|0.515|0.3930         |9.69               |60.28              |

At the matched four-step sampler, run 0002 has FID 7.332 and HR latent/image MSE 0.3960/0.003870; run 0003 has FID 6.281 and HR latent/image MSE 0.3930/0.003694. The capacity/precision package improves FID by 1.05 at comparable training throughput. Against the official 30-step run 0002, FID improves by 2.59 and 1024 latency drops from 30.40 to 9.69 ms, while peak VRAM rises from 41.14 to 60.28 MB. The RQI gain of 0.009 remains just below the 0.01 noise threshold.

Run 0003's sampler sweep (`dev/round_2/sampler_run_0003_results.json`) confirms four steps: 8/6/4/3/2 steps give FID 6.711/6.461/6.280/6.447/7.018. HR latent MSE continues to decrease with fewer steps, but FID rises below four. Keep four steps for the next run.

## Round 3

A same-noise, logit-normal flow-loss evaluation on all 1000 held-out benchmark latent pairs (`dev/round_2/validation_loss.py`) gives:

|Run ID|Base channels|Benchmark flow loss ↓|
|------|-------------|---------------------|
|0000  |96           |1.02519              |
|0002  |128          |1.01937              |
|0003  |192          |1.01517              |

These EMA/bf16 values are not directly comparable to raw-model training losses. Capacity gains transfer only weakly to this held-out loss despite large training-loss gains, suggesting that domain generalization is becoming more valuable than another scale increase. As a scale control, 256-channel bf16 compiled training measures 131.3 ms/step, 121.9 samples/s, and 5.18 GB peak VRAM (`dev/round_2/training_throughput_256.json`); it would reduce dataset passes from about 68 to 42. Defer 256 channels.

The training distribution contains only 1625/5000 detailed-realism blocks, with the remainder anime and oil-painting styles. The next hypothesis is that emphasizing the realism subset and applying exact dihedral augmentations will improve transfer to the real-image benchmark without changing model capacity or inference cost. Realism blocks are repeated 3x (59.1% effective share), and horizontal/vertical flips plus transpose augment aligned latent pairs. These coupled data interventions will be kept or reverted together; follow-up runs can ablate them if the effect is material.

|Run ID|Run name|Parent runs|Run purpose|
|------|--------|-----------|-----------|
|0004  |Realism-oversampled D4-augmented UNet|0003|Test domain-focused sampling and augmentation at fixed capacity|

Run 0004 passed smoke training and the complete smoke benchmark. Dataset checks confirm 5000 unique blocks, 8250 weighted blocks, and 4875 realism entries.

### Run 0004 results

|Run ID|Steps|Final 100-step loss|FID ↓|RQI ↑|HR latent MSE ↓|HR image MSE ↓|1024 latency (ms)|
|------|-----|-------------------|-----|-----|---------------|--------------|-----------------|
|0004  |21458|0.7581             |11.349|0.493|0.4307        |0.005013      |9.02             |

The realism-oversampling plus D4 package is a clear failure relative to run 0003: FID worsens by 5.07, RQI by 0.022, and reconstruction metrics also regress. Training loss is substantially higher (0.758 versus 0.625), and saved samples appear smoother. Revert both changes and branch from run 0003. Because the interventions were coupled, this run does not establish their individual effects; do not repeat either without a distinct reason or an isolated ablation.

## Round 4

The selected four-step Euler sampler evaluates exactly t=1, 0.75, 0.5, and 0.25. Logit-normal training instead covers the continuous unit interval and places little mass near the endpoints. The next hypothesis is that uniform discrete sampling of these four inference times will specialize the vector field for the actual sampler and reduce trajectory error. Model, original unaugmented dataset, optimizer, and inference remain identical to run 0003. Training losses from this run are not directly comparable because the time distribution changes.

|Run ID|Run name|Parent runs|Run purpose|
|------|--------|-----------|-----------|
|0005  |Four-step-aligned discrete time training|0003|Specialize flow training to the four sampler evaluation times|
