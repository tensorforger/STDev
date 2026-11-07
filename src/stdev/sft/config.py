"""
Configuration of the single-GPU SFT pipeline of the pretrain_large latent upscaler: the
resolution buckets of the generated dataset, their directory layout and the generation /
training hyperparameters.
"""

from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FLUX_MODEL_DIRECTORY = REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B"
SDXL_DIRECTORY = REPOSITORY_ROOT / "models" / "stable-diffusion-xl-base-1.0"
BASE_PROMPTS_PATH = REPOSITORY_ROOT / "prompts" / "base_prompts.json"
MODIFIERS_PATH = REPOSITORY_ROOT / "prompts" / "modifiers.json"

DATASET_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "flux_sft_buckets"
SFT_RUNS_DIRECTORY = REPOSITORY_ROOT / "sft_runs"

BENCHMARK_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "flux_1024"
BENCHMARK_IMAGE_SIZE = 1024
NUM_BENCHMARK_SAMPLES = 1000
BENCHMARK_SEED_OFFSET = 1_000_000


@dataclass(frozen=True)
class ResolutionBucket:
    """
    One target (high resolution) latent bucket of the generated dataset, named "{width}x{height}".
    """

    name: str
    width: int
    height: int
    num_blocks: int


# Adjust freely: names, resolutions (multiples of 16, as required by the VAE and patchify) and
# the number of blocks generated for each bucket.
RESOLUTION_BUCKETS: tuple[ResolutionBucket, ...] = (
    ResolutionBucket("512x512", width=512, height=512, num_blocks=1000),
    ResolutionBucket("1024x512", width=1024, height=512, num_blocks=500),
    ResolutionBucket("512x1024", width=512, height=1024, num_blocks=500),
    ResolutionBucket("768x512", width=768, height=512, num_blocks=600),
    ResolutionBucket("512x768", width=512, height=768, num_blocks=600),
    ResolutionBucket("768x1024", width=768, height=1024, num_blocks=1000),
    ResolutionBucket("1024x768", width=1024, height=768, num_blocks=1000),
    ResolutionBucket("1024x1024", width=1024, height=1024, num_blocks=8000),
)
BUCKET_SEED_STRIDE = 10_000_000

NUM_INFERENCE_STEPS = 4
MAX_SEQUENCE_LENGTH = 512
TEXT_ENCODER_OUTPUT_LAYERS = (9, 18, 27)
LATENT_CHANNELS = 32
VAE_SPATIAL_COMPRESSION = 8
LATENT_CODEC = "lbrq_8bit"
TEXT_EMBEDDING_CODEC = "cast_fp16"
HIGH_FREQUENCY_CODEC = "cast_fp32"
PROMPT_EMBEDS_CODEC = "cast_fp16"

HIGH_FREQUENCY_QUANTILE_COUNT = 1001
HIGH_FREQUENCY_STATISTICS_SAMPLES = 10000

SEED = 0
TEXT_DROPOUT_PROBABILITY = 0.2
HIGH_FREQUENCY_DROPOUT_PROBABILITY = 0.2
TRAINING_SIGMA_SHIFT = 1.0

LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 200
GRADIENT_CLIP_NORM = 1.0
EMA_DECAY = 0.999
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
GRADIENT_CHECKPOINTING = True
TRAINING_STEPS = 20_000
LOG_INTERVAL_STEPS = 50
CHECKPOINT_INTERVAL_STEPS = 1000
KEPT_CHECKPOINTS = 2
DATA_WORKERS = 4


def high_frequency_quantiles_path(dataset_directory: Path) -> Path:
    return dataset_directory / "high_frequency_quantiles.json"


def run_directory(run_name: str) -> Path:
    return SFT_RUNS_DIRECTORY / run_name
