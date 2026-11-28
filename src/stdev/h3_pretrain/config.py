"""
Configuration of the H3-Image -> FLUX.2 latent translator pretraining (autoresearch_6 run_0006 + the FLUX.2 pretrain
recipe): resolution buckets (pair types), data sources and input corruption of the H3 encoder input, process layout,
the open ended schedule with a manually triggered LR decay, and the visual benchmark. Run directories reuse
PretrainPaths.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from stdev.flux_pretrain.model import ModelConfig, SamplerConfig
from stdev.pretrain.config import PretrainPaths

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PRETRAIN_RUNS_DIRECTORY = REPOSITORY_ROOT / "pretrain_runs"
FLUX_VAE_DIRECTORY = REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B" / "vae"
FLUX_SMALL_DECODER_DIRECTORY = REPOSITORY_ROOT / "models" / "FLUX.2-small-decoder"
H3_VAE_DIRECTORY = REPOSITORY_ROOT / "models" / "MiniMax-H3-Image" / "vae_diffusers"
STEPPE_CHECKPOINT_PATH = REPOSITORY_ROOT / "models" / "Steppe-Flux.2" / "steppe_flux_2_200M_base.safetensors"
LINEAR_BASE_PATH = REPOSITORY_ROOT / "src" / "stdev" / "h3_translators" / "assets" / "linear_base_k3.pt"
AXIS_DIRECTORY = REPOSITORY_ROOT / "datasets" / "axis-v1_1k"
ENV_PATH = REPOSITORY_ROOT / ".env"

# Per channel mean / std of latents_flux2 - linear_base(latents_h3) on the pretrain mix (dev/h3_pretrain/residual_statistics.py,
# 4.2k produced samples, buckets weighted by count share; std within 0.94-1.15 of the autoresearch_6 4KLSDB table, the
# means differ by up to 0.23; the linear base explains R2 0.31 of the target variance on this mix).
RESIDUAL_MEAN = (0.0905, -0.0221, -0.2292, -0.0500, -0.1751, 0.1451, -0.0111, -0.0110, 0.0505, 0.0651, -0.0737, 0.0248, 0.0899, 0.0502, 0.0766, 0.0805, 0.0518, 0.0433, 0.0341, 0.0400, 0.0286, 0.0730, 0.0289, 0.1233, 0.0738, -0.0148, 0.0459, 0.0114, -0.1267, 0.0910, -0.0041, 0.0261)
RESIDUAL_STD = (0.7880, 0.5836, 0.8547, 0.3762, 0.8784, 0.6964, 0.6593, 0.8793, 0.3850, 0.8016, 0.8252, 0.6328, 0.8106, 0.8249, 0.8341, 0.6065, 0.8417, 0.7938, 0.7993, 0.5800, 0.8339, 0.8488, 0.8181, 0.8285, 0.7177, 0.8014, 0.6384, 0.8140, 0.8879, 0.7136, 0.7986, 0.5943)
MODEL = ModelConfig(width=768, depth=18, head_dim=64, mlp_hidden=3072, condition_dim=512, source_encoder_width=512, residual_std=RESIDUAL_STD)
BENCHMARK_SAMPLER = SamplerConfig(steps=30, shift=2.0, noise_scale=1.0)

TOKENS_PER_UNIT = 128 * 128


@dataclass(frozen=True)
class PairType:
    """
    One resolution bucket: image side in pixels (H3 latent side / 16, FLUX.2 latent side / 8), share by sample count,
    micro-batch size, number of gradient checkpointed transformer blocks, samples per cache shard, encode batch.
    """

    name: str
    side: int
    count_share: float
    micro_batch_size: int
    checkpointed_blocks: int
    shard_samples: int
    encode_batch_size: int

    @property
    def target_side(self) -> int:
        return self.side

    @property
    def target_latent_size(self) -> tuple[int, int]:
        return (self.side // 8, self.side // 8)

    @property
    def source_latent_size(self) -> tuple[int, int]:
        return (self.side // 16, self.side // 16)

    @property
    def tokens_per_sample(self) -> int:
        return math.prod(self.target_latent_size)


PAIR_TYPES: tuple[PairType, ...] = (
    PairType("image_1024", 1024, 0.9, 2, 0, 128, 4),
    PairType("image_512", 512, 0.1, 8, 0, 256, 8),
)
PAIR_TYPE_BY_NAME = {pair_type.name: pair_type for pair_type in PAIR_TYPES}


def micro_step_probabilities() -> dict[str, float]:
    """
    Probability of each pair type per micro-step, so that consumed samples follow count_share.
    """
    weights = {pair_type.name: pair_type.count_share / pair_type.micro_batch_size for pair_type in PAIR_TYPES}
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


# Image sources, mixed 1 : 1 by produced sample count (same as the FLUX.2 pretrain).
DATASETS = ("relaion", "lsdb")
DATASET_SHARES = {"relaion": 0.5, "lsdb": 0.5}
RELAION_SHARD_COUNT = 128
EXCLUDED_RELAION_SHARD_INDEX = 127
MIN_IMAGE_SOURCE_SIDE = 1024
LSDB_DATASET_NAME = "SingleBicycle/4KLSDB"
LSDB_FILE_PREFIX = "data/train_x4-"

# Square crop side = max(side, fraction * short side), fraction = 1 - (1 - MIN_CROP_FRACTION) * u^2.
MIN_CROP_FRACTION = 0.5

# Input corruption of the H3 encoder input (the FLUX.2 target and the hff stay clean). "clean": the H3 encoder sees the
# target pixels (= benchmark input). The filter variants resize the crop to the side with another filter than the
# target's INTER_AREA (they fall back to "clean" when the crop needs no resize); "jpeg" compresses the target pixels
# with quality = JPEG_QUALITY_MAX - (JPEG_QUALITY_MAX - JPEG_QUALITY_MIN) * u^2. Weights as in the FLUX.2 pretrain.
SOURCE_VARIANTS = ("clean", "jpeg", "nearest", "antialias", "bicubic", "lanczos")
SOURCE_VARIANT_WEIGHTS = {"clean": 80.0, "jpeg": 6.0, "nearest": 4.0, "antialias": 4.0, "bicubic": 3.0, "lanczos": 3.0}
JPEG_QUALITY_MIN = 20
JPEG_QUALITY_MAX = 95

DOWNLOAD_PROCESSES = 3
LSDB_READER_PROCESSES = 3
PREPROCESS_PROCESSES_PER_DATASET = 16
IMAGE_JOB_QUEUE_SIZE = 4000
IMAGE_BYTES_QUEUE_SIZE = 200
SAMPLE_QUEUE_SIZE = 128
CACHE_SIZE_LIMIT_BYTES = 200 * 1024**3
EVICTION_INTERVAL_SECONDS = 60.0
H3_ENCODER_BFLOAT16 = True

TRAINING_GPUS = (0, 1, 2, 3, 4, 5)
ENCODER_GPUS = (6, 7)
WARMUP_MIN_SHARDS_PER_TYPE = 4
WARMUP_MIN_SAMPLES = 8000

BENCHMARK_IMAGE_COUNT = 30
BENCHMARK_INTERVAL_SECONDS = 1800.0
BENCHMARK_BATCH = 6
LOW_FREQUENCY_SIDE = 64

SEED = 0
# Open ended schedule: constant LR after warmup until `control/decay` appears (its content: decay length in hours),
# then linear decay to zero over that length and the end of training. MAX_TRAINING_HOURS is a safety stop.
MAX_TRAINING_HOURS = 24.0
BASE_LEARNING_RATE = 6e-4
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.05
WARMUP_UNITS = 15000
# (training hours, gradient accumulation factor); the LR is held at the batch step.
ACCUMULATION_SCHEDULE = ((0.0, 1), (1.5, 2))
GRADIENT_CLIP_NORM = 1.0
TRAINING_SIGMA_SHIFT = 2.0
HFF_DROPOUT_PROBABILITY = 0.1
EMA_DECAY = 0.9999
LOG_INTERVAL_STEPS = 25
CONTROL_CHECK_INTERVAL_STEPS = 10
CHECKPOINT_INTERVAL_SECONDS = 1800.0
KEPT_CHECKPOINTS = 2
UPLOAD_INTERVAL_SECONDS = 2 * 3600.0
UPLOAD_BUCKET = "checkpoints"
KEPT_UPLOADS = 2
DATA_WORKERS_PER_TYPE = 3
DYNAMO_CACHE_SIZE_LIMIT = 64


def run_paths(run_name: str) -> PretrainPaths:
    return PretrainPaths(PRETRAIN_RUNS_DIRECTORY / run_name)
