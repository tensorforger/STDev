"""
Configuration of the FLUX.2 latent upscaler pretraining (autoresearch_3 recipe, scaled): resolution buckets
(pair types) and their mix, data sources and input corruption, process layout, the fixed time training
schedule and the visual benchmark. Run directories reuse PretrainPaths.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from stdev.flux_pretrain.model import ModelConfig, SamplerConfig
from stdev.pretrain.config import PretrainPaths

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PRETRAIN_RUNS_DIRECTORY = REPOSITORY_ROOT / "pretrain_runs"
VAE_DIRECTORY = REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B" / "vae"
AXIS_DIRECTORY = REPOSITORY_ROOT / "datasets" / "axis-v1_1k"
ENV_PATH = REPOSITORY_ROOT / ".env"

# Per channel std of latents_target - bilinear 2x(latents_source) on the pretrain mix (dev/flux_pretrain_residual_std.py,
# 4.3k samples; one table for both buckets: 256 -> 512 differs by +3.5%).
RESIDUAL_STD = (0.713, 0.677, 0.727, 0.603, 0.720, 0.709, 0.593, 0.734, 0.619, 0.724, 0.745, 0.613, 0.722, 0.682, 0.760, 0.717, 0.739, 0.713, 0.731, 0.706, 0.763, 0.738, 0.687, 0.731, 0.739, 0.713, 0.690, 0.699, 0.708, 0.713, 0.714, 0.665)
MODEL = ModelConfig(width=768, depth=18, head_dim=64, mlp_hidden=3072, condition_dim=512, source_encoder_width=512, residual_std=RESIDUAL_STD)
BENCHMARK_SAMPLER = SamplerConfig(steps=30, shift=2.0, noise_scale=1.0)

TOKENS_PER_UNIT = 128 * 128


@dataclass(frozen=True)
class PairType:
    """
    One resolution bucket: source / target side in pixels, share by sample count, micro-batch size, number of
    gradient checkpointed transformer blocks and samples per cache shard.
    """

    name: str
    source_side: int
    target_side: int
    count_share: float
    micro_batch_size: int
    checkpointed_blocks: int
    shard_samples: int
    encode_batch_size: int

    @property
    def target_latent_size(self) -> tuple[int, int]:
        return (self.target_side // 8, self.target_side // 8)

    @property
    def source_latent_size(self) -> tuple[int, int]:
        return (self.source_side // 8, self.source_side // 8)

    @property
    def tokens_per_sample(self) -> int:
        return math.prod(self.target_latent_size)


PAIR_TYPES: tuple[PairType, ...] = (
    PairType("image_512_1024", 512, 1024, 0.9, 2, 0, 128, 4),
    PairType("image_256_512", 256, 512, 0.1, 8, 0, 256, 8),
)
PAIR_TYPE_BY_NAME = {pair_type.name: pair_type for pair_type in PAIR_TYPES}


def micro_step_probabilities() -> dict[str, float]:
    """
    Probability of each pair type per micro-step, so that consumed samples follow count_share.
    """
    weights = {pair_type.name: pair_type.count_share / pair_type.micro_batch_size for pair_type in PAIR_TYPES}
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


# Image sources, mixed 1 : 1 by produced sample count.
DATASETS = ("relaion", "lsdb")
DATASET_SHARES = {"relaion": 0.5, "lsdb": 0.5}
RELAION_SHARD_COUNT = 128
EXCLUDED_RELAION_SHARD_INDEX = 127
MIN_IMAGE_SOURCE_SIDE = 1024
LSDB_DATASET_NAME = "SingleBicycle/4KLSDB"
LSDB_FILE_PREFIX = "data/train_x4-"

# Square crop side = max(target side, fraction * short side), fraction = 1 - (1 - MIN_CROP_FRACTION) * u^2,
# u uniform: mostly close to the full image, never below MIN_CROP_FRACTION of the short side.
MIN_CROP_FRACTION = 0.5

# Input corruption (source only; target and hff stay clean). Bilinear = the benchmark downscale. JPEG is applied on
# top of the bilinear downscale with quality = JPEG_QUALITY_MAX - (JPEG_QUALITY_MAX - JPEG_QUALITY_MIN) * u^2:
# mostly almost invisible, the autoresearch level (20-50) as the heavy tail.
SOURCE_VARIANTS = ("bilinear", "jpeg", "nearest", "antialias", "bicubic", "lanczos")
SOURCE_VARIANT_WEIGHTS = {"bilinear": 80.0, "jpeg": 6.0, "nearest": 4.0, "antialias": 4.0, "bicubic": 3.0, "lanczos": 3.0}
JPEG_QUALITY_MIN = 20
JPEG_QUALITY_MAX = 95

DOWNLOAD_PROCESSES = 3
LSDB_READER_PROCESSES = 3
PREPROCESS_PROCESSES_PER_DATASET = 20
IMAGE_JOB_QUEUE_SIZE = 4000
IMAGE_BYTES_QUEUE_SIZE = 200
SAMPLE_QUEUE_SIZE = 128
CACHE_SIZE_LIMIT_BYTES = 250 * 1024**3
EVICTION_INTERVAL_SECONDS = 60.0

TRAINING_GPUS = (0, 1, 2, 3, 4, 5, 6)
ENCODER_GPUS = (7,)
WARMUP_MIN_SHARDS_PER_TYPE = 4
WARMUP_MIN_SAMPLES = 15000

BENCHMARK_IMAGE_COUNT = 30
BENCHMARK_INTERVAL_SECONDS = 1800.0
BENCHMARK_BATCH = 6

SEED = 0
TRAINING_HOURS = 24.0
BASE_LEARNING_RATE = 1.3e-3
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.05
WARMUP_UNITS = 20000
DECAY_FRACTION = 0.3
# (fraction of the training time, gradient accumulation factor); LR scales with sqrt(accumulation).
ACCUMULATION_SCHEDULE = ((0.0, 1), (0.15, 2), (0.45, 4))
GRADIENT_CLIP_NORM = 1.0
TRAINING_SIGMA_SHIFT = 1.0
HFF_DROPOUT_PROBABILITY = 0.1
EMA_DECAY = 0.9999
LOG_INTERVAL_STEPS = 25
CONTROL_CHECK_INTERVAL_STEPS = 10
CHECKPOINT_INTERVAL_SECONDS = 1800.0
KEPT_CHECKPOINTS = 2
UPLOAD_INTERVAL_SECONDS = 3 * 3600.0
UPLOAD_BUCKET = "checkpoints"
KEPT_UPLOADS = 2
DATA_WORKERS_PER_TYPE = 3
DYNAMO_CACHE_SIZE_LIMIT = 64


def run_paths(run_name: str) -> PretrainPaths:
    return PretrainPaths(PRETRAIN_RUNS_DIRECTORY / run_name)
