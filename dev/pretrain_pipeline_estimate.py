"""
Rough time, data and storage budget of the three stage pretraining on one 8x RTX 5090 machine,
from measured throughputs (VAE encode: reports/flux2_vae_speed.md, training:
dev/pretrain_training_throughput.py, download: dev/relaion_download_speed.py).

Usage: python dev/pretrain_pipeline_estimate.py [--days 14] [--model 768_dec8]
"""

import argparse
from dataclasses import dataclass

GPU_COUNT = 8
SECONDS_PER_DAY = 86400.0
DDP_EFFICIENCY = 0.9
DOWNLOAD_IMAGES_PER_SECOND = 400.0
CPU_PREPROCESS_IMAGES_PER_SECOND = 2000.0
VAE_ENCODE_IMAGES_PER_SECOND = {256: 522.8, 512: 123.4, 1024: 26.5, 2048: 4.2}
LATENT_CHANNELS = 32
VAE_DOWNSCALE = 8
BYTES_PER_LATENT_VALUE = 1.0
TEXT_AND_METADATA_BYTES = 4096

TRAIN_SAMPLES_PER_SECOND_PER_GPU: dict[str, dict[int, float]] = {
    "0078": {512: 90.0, 1024: 23.4, 2048: 5.5},
    "768_dec8": {512: 28.6, 1024: 7.8, 2048: 1.7},
    "1024_dec8": {512: 20.2, 1024: 5.0, 2048: 1.07},
    "1280_dec8": {512: 13.0, 1024: 3.3, 2048: 0.7},
}


@dataclass
class Stage:
    name: str
    low_resolution: int
    high_resolution: int
    compute_fraction: float
    unique_images: float


STAGES = [
    Stage("stage 1: 256 -> 512", 256, 512, 0.55, 30e6),
    Stage("stage 2: 512 -> 1024", 512, 1024, 0.35, 8e6),
    Stage("stage 3: 1024 -> 2048", 1024, 2048, 0.10, 0.6e6),
]


def encode_seconds(resolutions: tuple[int, ...]) -> float:
    """
    returns:
        single GPU seconds to VAE encode one image at all resolutions
    """
    return sum(1.0 / VAE_ENCODE_IMAGES_PER_SECOND[resolution] for resolution in resolutions)


def latent_bytes(resolution: int) -> float:
    side = resolution // VAE_DOWNSCALE
    return LATENT_CHANNELS * side * side * BYTES_PER_LATENT_VALUE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=14.0)
    parser.add_argument("--model", default="768_dec8", choices=TRAIN_SAMPLES_PER_SECOND_PER_GPU)
    arguments = parser.parse_args()
    train_speed = TRAIN_SAMPLES_PER_SECOND_PER_GPU[arguments.model]

    shared_encode_gpu_seconds = (
        STAGES[0].unique_images * encode_seconds((256, 512))
        + STAGES[1].unique_images * encode_seconds((1024,))
        + STAGES[2].unique_images * encode_seconds((1024, 2048))
    )
    offline_encode_days = shared_encode_gpu_seconds / GPU_COUNT / SECONDS_PER_DAY
    training_days = arguments.days - offline_encode_days
    print(f"model {arguments.model}, total {arguments.days:.1f} days, "
          f"offline encode {offline_encode_days:.2f} days, training {training_days:.2f} days\n")

    online_extra_days = 0.0
    total_storage_bytes = 0.0
    for stage_index, stage in enumerate(STAGES):
        stage_days = training_days * stage.compute_fraction
        per_gpu_speed = train_speed[stage.high_resolution]
        machine_speed = per_gpu_speed * GPU_COUNT * DDP_EFFICIENCY
        samples = machine_speed * stage_days * SECONDS_PER_DAY
        train_seconds = 1.0 / per_gpu_speed
        stage_encode_seconds = encode_seconds((stage.low_resolution, stage.high_resolution))
        online_overhead = stage_encode_seconds / train_seconds
        online_extra_days += stage_days * online_overhead
        new_resolutions = {0: (256, 512), 1: (1024,), 2: (1024, 2048)}[stage_index]
        storage = stage.unique_images * (
            sum(latent_bytes(resolution) for resolution in new_resolutions) + TEXT_AND_METADATA_BYTES
        )
        total_storage_bytes += storage
        read_megabytes_per_second = machine_speed * (
            latent_bytes(stage.low_resolution) + latent_bytes(stage.high_resolution)
        ) / 1e6
        print(stage.name)
        print(f"  training days          {stage_days:6.2f}")
        print(f"  train speed            {per_gpu_speed:6.1f} /s/GPU, {machine_speed:6.0f} /s machine")
        print(f"  samples seen           {samples / 1e6:6.1f} M")
        print(f"  unique images          {stage.unique_images / 1e6:6.1f} M -> {samples / stage.unique_images:4.1f} epochs")
        print(f"  online encode overhead {100 * online_overhead:6.0f} % of training time")
        print(f"  download needed online {machine_speed:6.0f} /s vs {DOWNLOAD_IMAGES_PER_SECOND:.0f} /s available")
        print(f"  latent read rate       {read_megabytes_per_second:6.1f} MB/s")
        print(f"  new storage (8 bit)    {storage / 1e12:6.2f} TB")
    print()
    print(f"offline: encode once {offline_encode_days:.2f} days, storage {total_storage_bytes / 1e12:.1f} TB")
    print(f"online:  encode every sample, extra {online_extra_days:.2f} days for the same samples")
    download_days = (STAGES[0].unique_images + STAGES[2].unique_images) / DOWNLOAD_IMAGES_PER_SECOND / SECONDS_PER_DAY
    print(f"download of all unique images at {DOWNLOAD_IMAGES_PER_SECOND:.0f}/s: {download_days:.2f} days")


if __name__ == "__main__":
    main()
