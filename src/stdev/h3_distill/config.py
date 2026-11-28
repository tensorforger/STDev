"""
Configuration of the single step rectified flow distillation of the H3 -> FLUX.2 latent translator (FLUX.2 distillation
recipe, `reports/flux_rf_distillation.md`): online teacher pairs (generator GPUs) and a single GPU student trainer
running at the same time, then a parallel 1-step benchmark on the autoresearch_6 Axis 1k set.
"""

from stdev.h3_pretrain.config import REPOSITORY_ROOT

TEACHER_CHECKPOINT = REPOSITORY_ROOT / "pretrain_runs" / "h3_translator_v1" / "checkpoints" / "checkpoint_000050590.pt"
TEACHER_WEIGHTS = "raw"
PRETRAIN_CACHE = REPOSITORY_ROOT / "pretrain_runs" / "h3_data" / "cache"
RUNS_DIRECTORY = REPOSITORY_ROOT / "distill_runs"

TEACHER_STEPS = 30
TEACHER_SHIFT = 2.0

GENERATOR_GPUS = (0, 1, 2, 3, 4, 5, 6)
TRAINER_GPU = 7
SHARD_SAMPLES = {"image_1024": 32, "image_512": 64}
TEACHER_BATCH_SIZE = {"image_1024": 8, "image_512": 32}
MIN_PAIRS_TO_START = 1500

SEED = 0
TRAINING_MINUTES = 25.0
ACCUMULATION = 4
LEARNING_RATE = 3e-4
WARMUP_STEPS = 50
DECAY_FRACTION = 0.3
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
EMA_DECAY = 0.995
LOG_INTERVAL_STEPS = 25
SNAPSHOT_INTERVAL_SECONDS = 300.0
DATA_WORKERS_PER_TYPE = 3
UPLOAD_BUCKET = "checkpoints"
