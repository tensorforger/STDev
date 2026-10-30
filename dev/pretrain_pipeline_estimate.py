from dataclasses import dataclass

GPU_COUNT = 8
SECONDS_PER_DAY = 86400.0
TRAIN_SAMPLES_PER_SECOND_AT_1024 = 6.0
VAE_ENCODE_IMAGES_PER_SECOND = {256: 522.8, 512: 123.4, 1024: 26.5, 2048: 4.2}
LATENT_CHANNELS = 32
VAE_DOWNSCALE = 8
BYTES_PER_LATENT_VALUE = 1.0


@dataclass
class Stage:
    name: str
    low_resolution: int
    high_resolution: int
    train_speed_factor: float


STAGES = [
    Stage("stage 1: 256 -> 512", 256, 512, 3.5),
    Stage("stage 2: 512 -> 1024", 512, 1024, 1.0),
    Stage("stage 3: 1024 -> 2048 full", 1024, 2048, 0.22),
    Stage("stage 3: 1024 HR crops of 2048", 512, 1024, 1.0),
]


def latent_megabytes(resolution: int) -> float:
    side = resolution // VAE_DOWNSCALE
    return LATENT_CHANNELS * side * side * BYTES_PER_LATENT_VALUE / 1e6


if __name__ == "__main__":
    for stage in STAGES:
        train_seconds = 1.0 / (TRAIN_SAMPLES_PER_SECOND_AT_1024 * stage.train_speed_factor)
        encode_seconds = (
            1.0 / VAE_ENCODE_IMAGES_PER_SECOND[stage.low_resolution]
            + 1.0 / VAE_ENCODE_IMAGES_PER_SECOND[stage.high_resolution]
        )
        online_samples_per_second = GPU_COUNT / (train_seconds + encode_seconds)
        offline_samples_per_second = GPU_COUNT / train_seconds
        encode_only_samples_per_second = GPU_COUNT / encode_seconds
        storage = latent_megabytes(stage.low_resolution) + latent_megabytes(stage.high_resolution)
        print(stage.name)
        print(f"  train only        {offline_samples_per_second:7.1f} samples/s  {offline_samples_per_second * SECONDS_PER_DAY / 1e6:6.2f} M/day")
        print(f"  online with VAE   {online_samples_per_second:7.1f} samples/s  overhead {100 * encode_seconds / (train_seconds + encode_seconds):4.1f}%")
        print(f"  encode only 8 GPU {encode_only_samples_per_second:7.1f} samples/s  {encode_only_samples_per_second * SECONDS_PER_DAY / 1e6:6.2f} M/day")
        print(f"  latent storage    {storage:7.3f} MB/sample (8 bit)")
    shared_encode_seconds = sum(1.0 / VAE_ENCODE_IMAGES_PER_SECOND[resolution] for resolution in (256, 512, 1024))
    print(f"shared 256+512+1024 encode: {GPU_COUNT / shared_encode_seconds:.0f} images/s on 8 GPUs, "
          f"{GPU_COUNT / shared_encode_seconds * SECONDS_PER_DAY / 1e6:.1f} M/day, "
          f"{1 / shared_encode_seconds:.1f} images/s on 1 GPU")
