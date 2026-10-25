# task_1

Your task is to implement the latent upscaler benchmark logic. This same benchmark would be used for various models and sampling approaches.

Conceptually, there are these layers:

- Benchmark dataset, represented as a single directory
- Unified image latent upscaler abstraction
- Script that runs the benchmark for the particular upscaler and the particular dataset

### Benchmark Dataset

Should be a folder in the `./benchmark_datasets/`, for example, `./benchmark_datasets/axis_1024`

Should contain two directories:

- "images": contain `.png` images resized to same size. Default is height=1024, width=1024.
- "labels": contain `.txt` text files with image descriptions.


### Upscaler Abstraction_

Minimal version is in `src/stdev/upscalers/base.py`

Benchmark should approximately do:

```python
upscaler = Upscaler()
upscaler.prepare_to_benchmark()

for sample in benchmark_dataset:
    upscaled = upscaler.upscale(sample)
```

And then calculate all metrics.

### Benchmark script

Takes the path to benchmark dataset folder and path to upscaler definition file:

```bash
uv run scripts/run_benchmark.py --dataset benchmark_datasets/axis_1024 --upscaler src/stdev/upscalers/upscaler_1.py
```
Use torch.compile of the model for all benchmarks

It should measure two sets of metrics:

1. Computational efficiency

Can be computed on dummy data.

- Latency (use warm up, then loop some iterations)
- CUDA memory usage
- GPU utilization
- CPU usage

Use torch.profiler.profile for this.
Save peak and average values for numerical results.
For CUDA and CPU usage, memory usage additionaly save plots (images).
Measure for different image resolutions. Resolutions are [512, 1024, 2048].
For example, if image is in 1024x1024 resolution, latents would have shape [1, 32, 128, 128].
In this case, metric name would be e.g. `cuda_mem_peak_1024`.
Use batch size of 1 for all computational efficiency metrics.


2. Quality

`lr` = low resolution
`hr` = high resolution

Take an image from the dataset in the original resolution, it is `image_hr_gt`
Downscale it with bilinear interpolation, it is `image_lr_gt`
Encode both to `latents_hr_gt` and `latents_lr_gt`
Upscale `latents_lr_gt` to `latents_hr_pred` by passing it to `upscaler.upscale()`
Decode `latents_hr_pred` to `image_hr_pred`
Downscale `image_hr_pred` with bilinear interpolation to `image_lr_pred`

The metrics would be:
- `mse_hr_image` = MSE of `image_hr_gt` and `image_hr_pred` (can not be perfect, just for reference)
- `mse_lr_image` = MSE of `image_lr_gt` and `image_lr_pred` (measures roundtrip error, can become zero)
- `mse_hr_latents` = MSE of `latents_hr_pred` and `latents_hr_gt` (mse in latents space)
- `fid` = global FID between all `image_hr_gt` images and all `image_hr_pred`
- `rqi` = average of rqi(`image_hr_pred`, `image_hr_gt`) 

About RQI metric: https://github.com/tensorforger/rqi
or notebook `./notebooks/benchmark_dev.ipynb`

You would have a result dict `dict[str, float]` and a set of images.
Save them to `./raw_results/<dataset_name>/<upscaler_name>/` as a json file `metrics.json` and images in `images` subdir there.
Also print the result dict when script is completed.

Script should also support the `--smoke` argument that would do essentially everything same but use only 2 passes of each resolution for efficency metrics and 2 samples for quality ones, also no torch.compile and no warmup. This would add suffix `_smoke` to `<dataset_name>` in the path, e.g. `./raw_results/axis_1024_smoke/baseline`.

All other numbers and behaviours (e.g. list of resolutions) are just constants on the top of script file. No cli args for them.

#### Script optimization:

On the first run with new dataset, script should create cache in `./benchmark_cache/<dataset_name>` that contains
1. Statistics for ground truth distribution for FID
2. Encoded image latents `latents_lr_gt` and `latents_hr_gt` compressed with vaezip (example in `./notebooks/benchmark_dev.ipynb`)


##  Quick check

Make a small smoke test this way:

1. Implement the minimal upscaler definition `baseline.py` that would use just bilinear interpolation on latents.
2. Make a script, that would make a benchmark dataset `axis_1024` from the `datasets/axis-v1_1k`. This one is reusable, keep it in `./scripts`. No cli args. Just resizes all images to 1024x1024 with `crop_maximal_rectangle` and structutes the files in required way.
3. Run approximately this:
```bash
uv run scripts/run_benchmark.py --dataset benchmark_datasets/axis_1024 --upscaler src/stdev/upscalers/baseline.py --smoke
```
4. Check values are saved.