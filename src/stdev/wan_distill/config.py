"""
Configuration of the single step rectified flow distillation of the Wan 2.1 latent upscaler. Paths are relative
to a data root (default: the repository root), so code in a worktree can use the main checkout's data.
"""

from dataclasses import dataclass
from pathlib import Path

from stdev.wan_pretrain.config import REPOSITORY_ROOT

DEFAULT_DATA_ROOT = REPOSITORY_ROOT
TEACHER_CHECKPOINT = Path("pretrain_runs/wan_pretrain_v1/checkpoints/checkpoint_000161590.pt")
PAIRS_DIRECTORY = Path("pretrain_runs/wan_distill_pairs")
EVALUATION_DATA_RUN = Path("pretrain_runs/wan_eval_1000")
FULL_VAE_DIRECTORY = Path("models/Wan2.1-T2V-1.3B-Diffusers/vae")
RUNS_DIRECTORY = Path("distill_runs")

SEED = 0
LEARNING_RATE = 1e-3
WARMUP_STEPS = 200
ACCUMULATION = 4
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
EMA_DECAY = 0.999
DECAY_EPOCHS = 1
LOG_INTERVAL_STEPS = 25
CONTROL_CHECK_INTERVAL_STEPS = 10
KEPT_CHECKPOINTS = 2
UPLOAD_BUCKET = "checkpoints"
KEPT_UPLOADS = 2
DYNAMO_CACHE_SIZE_LIMIT = 64

EVALUATION_BATCH = 4
EVALUATION_SAVED_IMAGES = 16
EVALUATION_NOISE_SEED = 0


@dataclass(frozen=True)
class EvaluationVariant:
    name: str
    weights: str
    guidance: float


EVALUATION_VARIANTS = (
    EvaluationVariant("ema_cfg1", "ema", 1.0),
    EvaluationVariant("ema_cfg2", "ema", 2.0),
    EvaluationVariant("raw_cfg1", "raw", 1.0),
)
