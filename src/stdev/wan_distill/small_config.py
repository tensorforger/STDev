"""
Configuration of the small single step students of the Wan 2.1 upscaler (size ladder ~20M -> ~0.5M): the student
architecture scaled down in width and depth, trained from scratch on one GPU each, on pairs generated online by
the large single step student (distill_runs/wan_distill_v1) on two GPUs.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from stdev.wan_pretrain.model import ModelConfig


@dataclass(frozen=True)
class SmallStudent:
    name: str
    config: ModelConfig
    learning_rate: float


def scaled_learning_rate(width: int) -> float:
    """
    Conservative width scaling: 1.5e-3 at width 256, x sqrt(256 / width), at most 3e-3.
    """
    return min(3e-3, 1.5e-3 * math.sqrt(256 / width))


def make_student(name: str, width: int, depth: int, head_dim: int, condition_dim: int, source_encoder_width: int) -> SmallStudent:
    config = ModelConfig(
        width=width,
        depth=depth,
        head_dim=head_dim,
        mlp_hidden=4 * width,
        condition_dim=condition_dim,
        source_encoder_width=source_encoder_width,
    )
    return SmallStudent(name, config, scaled_learning_rate(width))


STUDENTS: tuple[SmallStudent, ...] = (
    make_student("s20m", 256, 10, 64, 256, 192),
    make_student("s10m", 192, 10, 32, 192, 128),
    make_student("s5m", 128, 10, 32, 128, 96),
    make_student("s2m", 96, 8, 32, 96, 64),
    make_student("s1m", 64, 8, 32, 64, 48),
    make_student("s05m", 48, 6, 16, 48, 32),
)
STUDENT_BY_NAME = {student.name: student for student in STUDENTS}

LARGE_STUDENT_CHECKPOINT = Path("distill_runs/wan_distill_v1/checkpoints/checkpoint_0004306.pt")
PRETRAIN_CACHE = Path("pretrain_runs/wan_data/cache")
ONLINE_PAIRS_DIRECTORY = Path("distill_runs/wan_small_pairs")
RUN_PREFIX = "wan_small_"

GENERATOR_GPUS = (6, 7)
TRAINING_GPUS = (0, 1, 2, 3, 4, 5)
GENERATOR_BATCH_SIZE = {
    "image_512_1024": 16,
    "image_256_512": 64,
    "video_17_33_512": 8,
    "video_9_17_1024": 4,
    "video_17_512_1024": 4,
    "video_9_512_17_1024": 4,
}
SHARD_SAMPLES = {
    "image_512_1024": 64,
    "image_256_512": 256,
    "video_17_33_512": 32,
    "video_9_17_1024": 16,
    "video_17_512_1024": 16,
    "video_9_512_17_1024": 16,
}
CACHE_SIZE_LIMIT_BYTES = 40 * 1024**3
WARMUP_MIN_SHARDS_PER_TYPE = 2

STEP_BATCH_MULTIPLIER = 8
WARMUP_STEPS = 1000
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
EMA_DECAY = 0.999
DEFAULT_DECAY_FRACTION = 0.3
BENCHMARK_INTERVAL_SECONDS = 30 * 60.0
LOG_INTERVAL_STEPS = 100
CONTROL_CHECK_INTERVAL_STEPS = 50
KEPT_CHECKPOINTS = 2
KEPT_UPLOADS = 2
DATA_WORKERS_PER_TYPE = 2
