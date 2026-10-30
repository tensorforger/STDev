# Pretrain pipeline details

## Model

Runs 0077 and 0078 combined. Deep decoder with 8 blocks, dim 768, all blocks are SDXL-style with window attention.
Implement it as a separate file upscalers/pretrain_large.py
Takes pooled text embed and hight frequency conditioning (see ./notebooks/high_frequency_attributes). Features are calculated over a high res image that model should generate (after crop and resize), not original image. They are normalized using real image statistrics of first ~ 10000 samples. Passed to model as a vector of 4 values, then each is transformed into embed same way as timestep. All combined to single conditioning vector and passed to modulation. Independent 20% dropout for text embed and each high frequency feature.

## Pipeline

First ~ 30 minutes is warmup: download set of images, encode to cache, calculate all statistics.
Fully online, no connection to S3, local cache only. For stage 1 one GPU for encoding, 7 for train.
Save checkpoints including optimizer state and current variables state (as learning rate).

## Training control

- Plot training loss to tensorboard.
- Benchmark on same set of 2000 images every 30 minutes (first 4 hours), then every 2 hours. Plot benchmark results curve to tensorboard. Save 100 image pairs each benchmark (orig low res + generated). During benchmark model sees text and high frequency features.

