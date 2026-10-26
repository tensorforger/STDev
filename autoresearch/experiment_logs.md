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
