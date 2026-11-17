"""
Configuration of the Wan 2.1 latent upscaler pretraining: pair types (resolution / frame buckets), their
training mix, data production, training schedule and benchmark. Run directories reuse PretrainPaths.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from stdev.pretrain.config import PretrainPaths
from stdev.wan_pretrain.model import ModelConfig, SamplerConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PRETRAIN_RUNS_DIRECTORY = REPOSITORY_ROOT / "pretrain_runs"
FULL_VAE_DIRECTORY = REPOSITORY_ROOT / "models" / "Wan2.1-T2V-1.3B-Diffusers" / "vae"
TINY_VAE_PATH = REPOSITORY_ROOT / "models" / "taew2_1.safetensors"
ENV_PATH = REPOSITORY_ROOT / ".env"

MODEL = ModelConfig(width=768, depth=16, head_dim=64, mlp_hidden=3072, condition_dim=512, source_encoder_width=512)
BENCHMARK_SAMPLER = SamplerConfig(steps=20, shift=4.0, noise_scale_spatial=1.0, noise_scale_temporal=0.8)

TOKENS_PER_UNIT = 128 * 128
IMAGE_TOKEN_SHARE = 5.0 / 9.0


@dataclass(frozen=True)
class PairType:
    """
    One training bucket: source and target sizes in pixels (frames, side), the share of its kind (by sample
    count), the micro-batch size, and how many transformer blocks are gradient checkpointed.
    Video sources are every 2nd frame of the target clip and / or its 2x box downscale.
    """

    name: str
    kind: str
    source_frames: int
    source_side: int
    target_frames: int
    target_side: int
    count_share: float
    micro_batch_size: int
    checkpointed_blocks: int
    shard_samples: int

    @property
    def target_latent_size(self) -> tuple[int, int, int]:
        return (1 + (self.target_frames - 1) // 4, self.target_side // 8, self.target_side // 8)

    @property
    def source_latent_size(self) -> tuple[int, int, int]:
        return (1 + (self.source_frames - 1) // 4, self.source_side // 8, self.source_side // 8)

    @property
    def tokens_per_sample(self) -> int:
        return math.prod(self.target_latent_size)

    @property
    def is_temporal(self) -> bool:
        return self.target_frames != self.source_frames

    @property
    def is_spatial(self) -> bool:
        return self.target_side != self.source_side


PAIR_TYPES: tuple[PairType, ...] = (
    PairType("image_512_1024", "image", 1, 512, 1, 1024, 0.9, 2, 2, 128),
    PairType("image_256_512", "image", 1, 256, 1, 512, 0.1, 8, 2, 256),
    PairType("video_17_33_512", "video", 17, 512, 33, 512, 0.9, 1, 7, 64),
    PairType("video_9_17_1024", "video", 9, 1024, 17, 1024, 1.0 / 30.0, 1, 13, 32),
    PairType("video_17_512_1024", "video", 17, 512, 17, 1024, 1.0 / 30.0, 1, 13, 32),
    PairType("video_9_512_17_1024", "video", 9, 512, 17, 1024, 1.0 / 30.0, 1, 13, 32),
)
PAIR_TYPE_BY_NAME = {pair_type.name: pair_type for pair_type in PAIR_TYPES}


def micro_step_probabilities() -> dict[str, float]:
    """
    Probability of each pair type per micro-step, so that the target tokens split IMAGE_TOKEN_SHARE between
    images and videos and the sample counts inside a kind follow count_share.
    """
    weights: dict[str, float] = {}
    for kind, kind_share in (("image", IMAGE_TOKEN_SHARE), ("video", 1.0 - IMAGE_TOKEN_SHARE)):
        kind_types = [pair_type for pair_type in PAIR_TYPES if pair_type.kind == kind]
        kind_tokens = sum(pair_type.count_share * pair_type.tokens_per_sample for pair_type in kind_types)
        for pair_type in kind_types:
            token_share = kind_share * pair_type.count_share * pair_type.tokens_per_sample / kind_tokens
            weights[pair_type.name] = token_share / (pair_type.micro_batch_size * pair_type.tokens_per_sample)
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def production_shares() -> dict[str, float]:
    """
    Share of each pair type in the produced samples, equal to its share in the consumed samples (equal reuse).
    """
    probabilities = micro_step_probabilities()
    counts = {pair_type.name: probabilities[pair_type.name] * pair_type.micro_batch_size for pair_type in PAIR_TYPES}
    total = sum(counts.values())
    return {name: count / total for name, count in counts.items()}


RELAION_SHARD_COUNT = 128
BENCHMARK_SHARD_INDEX = 127
MIN_IMAGE_SOURCE_SIDE = 1024
VIDEO_BUCKET = "vk-50k"
VIDEO_KEY_PREFIX = "videos/"
HELD_OUT_VIDEOS = 500
VIDEO_SEED = 0

DOWNLOAD_PROCESSES = 3
IMAGE_PREPROCESS_PROCESSES = 24
VIDEO_PROCESSES = 16
IMAGE_JOB_QUEUE_SIZE = 4000
IMAGE_BYTES_QUEUE_SIZE = 400
IMAGE_SAMPLE_QUEUE_SIZE = 256
VIDEO_SAMPLE_QUEUE_SIZE = 24
CACHE_SIZE_LIMIT_BYTES = 150 * 1024**3
EVICTION_INTERVAL_SECONDS = 60.0

BENCHMARK_IMAGE_COUNT = 100
BENCHMARK_SAVED_IMAGES = 100
BENCHMARK_INTERVAL_SECONDS = 3600.0
BENCHMARK_DECODE_BATCH = 2

TRAINING_GPUS = (0, 1, 2, 3, 4, 5)
ENCODER_GPUS = (6, 7)
WARMUP_MIN_SHARDS_PER_TYPE = 2
WARMUP_MIN_SECONDS = 20 * 60

SEED = 0
BASE_LEARNING_RATE = 1.3e-3
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
WARMUP_UNITS = 20000
DEFAULT_DECAY_FRACTION = 0.3
GRADIENT_CLIP_NORM = 1.0
TRAINING_SIGMA_SHIFT = 1.0
HFF_DROPOUT_PROBABILITY = 0.1
EMA_DECAY = 0.9999
ACCUMULATION_SCHEDULE = ((0, 1), (500_000, 2), (1_500_000, 4))
LOG_INTERVAL_STEPS = 25
CONTROL_CHECK_INTERVAL_STEPS = 10
CHECKPOINT_INTERVAL_SECONDS = 1800.0
KEPT_CHECKPOINTS = 2
UPLOAD_INTERVAL_SECONDS = 3 * 3600.0
UPLOAD_BUCKET = "checkpoints"
KEPT_UPLOADS = 2
DATA_WORKERS_PER_TYPE = 2
DYNAMO_CACHE_SIZE_LIMIT = 64


def run_paths(run_name: str) -> PretrainPaths:
    return PretrainPaths(PRETRAIN_RUNS_DIRECTORY / run_name)
