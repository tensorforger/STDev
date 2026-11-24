"""
Configuration of the small single step student of the FLUX.2 upscaler (~20M): the student architecture scaled down
(Wan 2.1 small students recipe), trained from scratch with DDP on pairs generated online by the large single step
student (distill_runs/flux_distill_v1, raw weights) on the generator GPUs.
"""

import math

from stdev.flux_distill.config import RUNS_DIRECTORY
from stdev.flux_pretrain.config import MODEL
from stdev.flux_pretrain.model import ModelConfig

SMALL_MODEL = ModelConfig(
    width=256, depth=14, head_dim=64, mlp_hidden=1024, condition_dim=256, source_encoder_width=192, residual_std=MODEL.residual_std
)
LARGE_STUDENT_CHECKPOINT = RUNS_DIRECTORY / "flux_distill_v1" / "checkpoints" / "checkpoint_001083.pt"
LARGE_STUDENT_WEIGHTS = "raw"

GENERATOR_GPUS = (4, 5, 6, 7)
TRAINING_GPUS = (0, 1, 2, 3)
GENERATOR_BATCH_SIZE = {"image_512_1024": 16, "image_256_512": 64}
SHARD_SAMPLES = {"image_512_1024": 64, "image_256_512": 256}
PAIR_CACHE_SIZE_LIMIT_BYTES = 120 * 1024**3
WARMUP_MIN_SHARDS_PER_TYPE = 4

SEED = 0
TRAINING_HOURS = 6.0
MICRO_BATCH_SIZE = {"image_512_1024": 8, "image_256_512": 32}
BASE_LEARNING_RATE = 1.5e-3
REFERENCE_GLOBAL_BATCH = 16
WARMUP_STEPS = 1000
DECAY_FRACTION = 0.3
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
EMA_DECAY = 0.999
LOG_INTERVAL_STEPS = 100
CONTROL_CHECK_INTERVAL_STEPS = 20
SNAPSHOT_INTERVAL_SECONDS = 30 * 60.0
CHECKPOINT_INTERVAL_SECONDS = 30 * 60.0
DATA_WORKERS_PER_TYPE = 4


def learning_rate_for(global_batch: int) -> float:
    """
    Wan small student LR (1.5e-3 at 16 images @1024 per step), square root scaled to the global batch.
    """
    return BASE_LEARNING_RATE * math.sqrt(global_batch / REFERENCE_GLOBAL_BATCH)
