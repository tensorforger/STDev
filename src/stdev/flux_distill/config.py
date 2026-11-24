"""
Configuration of the single step rectified flow distillation of the FLUX.2 latent upscaler: online teacher pairs
(generator GPUs) and a single GPU student trainer running at the same time, then a parallel 1-step benchmark.
"""

from pathlib import Path

from stdev.flux_pretrain.config import REPOSITORY_ROOT

TEACHER_CHECKPOINT = REPOSITORY_ROOT / "pretrain_runs" / "flux_pretrain_v1" / "checkpoints" / "checkpoint_000129590.pt"
TEACHER_WEIGHTS = "raw"
PRETRAIN_CACHE = REPOSITORY_ROOT / "pretrain_runs" / "flux_data" / "cache"
RUNS_DIRECTORY = REPOSITORY_ROOT / "distill_runs"

TEACHER_STEPS = 30
TEACHER_SHIFT = 2.0
NULL_HFF_PROBABILITY = 0.1
RANDOM_HFF_PROBABILITY = 0.45
GUIDED_PROBABILITY = 0.5
GUIDANCE_RANGE = (1.0, 2.0)

GENERATOR_GPUS = (0, 1, 2, 3, 4, 5, 6)
TRAINER_GPU = 7
SHARD_SAMPLES = {"image_512_1024": 32, "image_256_512": 64}
TEACHER_BATCH_SIZE = {"image_512_1024": 8, "image_256_512": 32}
MIN_PAIRS_TO_START = 1500

SEED = 0
TRAINING_MINUTES = 20.0
ACCUMULATION = 4
LEARNING_RATE = 3e-4
WARMUP_STEPS = 50
DECAY_FRACTION = 0.3
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
EMA_DECAY = 0.995
LOG_INTERVAL_STEPS = 25
SNAPSHOT_INTERVAL_SECONDS = 240.0
DATA_WORKERS_PER_TYPE = 3
UPLOAD_BUCKET = "checkpoints"
