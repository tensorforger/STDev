"""
Configuration of the rectified flow (reflow) distillation of the pretrained upscaler.

A distillation run is a loop of rounds: all GPUs generate teacher pairs (initial noise -> teacher
output) on 4KLSDB crops for GENERATION_SECONDS, then all GPUs train the student on interpolations
of these pairs (each pair TRAINING_REPEATS times on average), then sample images of the student
at SAMPLE_STEPS are saved. The final LR decay is started manually, as in pretraining.
"""

from dataclasses import dataclass
from pathlib import Path

from stdev.pretrain.config import REPOSITORY_ROOT, STAGES

DISTILL_RUNS_DIRECTORY = REPOSITORY_ROOT / "distill_runs"
TEACHER_RUN_DIRECTORY = REPOSITORY_ROOT / "pretrain_runs" / "stage2_v1"
TEACHER_WEIGHTS_PATH = TEACHER_RUN_DIRECTORY / "export" / "stage2_final_000054108_ema.safetensors"
TEACHER_QUANTILES_PATH = TEACHER_RUN_DIRECTORY / "statistics" / "high_frequency_quantiles.json"

DATASET_NAME = "SingleBicycle/4KLSDB"
DATASET_DIRECTORY = REPOSITORY_ROOT / "datasets" / "4klsdb"
TRAIN_FILE_PREFIX = "data/train_x4-"
SAMPLE_SET_FILE = "data/validation_x4-00000-of-00004.parquet"
CAPTION_COLUMN = "caption"
PREFETCHED_FILES_PER_RANK = 2

STAGE = STAGES["stage_2"]
GPUS = (0, 1, 2, 3, 4, 5, 6, 7)

CROPS_PER_IMAGE = 2
PREPROCESS_WORKERS_PER_GPU = 6
PREPROCESS_LOOKAHEAD_IMAGES = 64
GENERATION_SECONDS = 600
TEACHER_BATCH_SIZE = 8
TEACHER_STEPS = 20
TEACHER_SIGMA_SHIFT = 2.0
PAIR_SHARD_SAMPLES = 64
TEXT_DROPOUT_PROBABILITY = 0.2
HIGH_FREQUENCY_DROPOUT_PROBABILITY = 0.2

TRAINING_REPEATS = 3
# Student sampling uses uniform Euler steps; it is trained only at the noise levels of the 1 to
# STUDENT_MAX_STEPS step schedules. Every schedule gets the same weight, split over its steps:
# t = 1 (pure noise, used by all schedules) gets 52% of the samples.
STUDENT_MAX_STEPS = 4
LEARNING_RATE = 1e-4
WARMUP_STEPS = 100
ADAM_BETAS = (0.9, 0.95)
GRADIENT_CLIP_NORM = 1.0
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
EMA_DECAY = 0.999
DEFAULT_DECAY_FRACTION = 0.15
KEPT_CHECKPOINTS = 2
DATA_WORKERS_PER_RANK = 4
LOG_INTERVAL_STEPS = 50
CONTROL_CHECK_INTERVAL_STEPS = 25
SEED = 0

SAMPLE_IMAGE_COUNT = 16
SAMPLE_STEPS = (1, 2, 3, 4)
SAMPLE_STRIP_TILE = 512
SAMPLE_SEED = 0


@dataclass(frozen=True)
class DistillPaths:
    """
    Directory layout of one distillation run.
    """

    root: Path

    @property
    def rounds(self) -> Path:
        return self.root / "rounds"

    def pairs(self, round_index: int) -> Path:
        return self.rounds / f"round_{round_index:04d}" / "pairs"

    def round_results(self, round_index: int) -> Path:
        return self.rounds / f"round_{round_index:04d}"

    @property
    def sample_set(self) -> Path:
        return self.root / "sample_set"

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def export(self) -> Path:
        return self.root / "export"

    @property
    def tensorboard(self) -> Path:
        return self.root / "tensorboard"

    @property
    def control(self) -> Path:
        return self.root / "control"

    @property
    def decay_request(self) -> Path:
        return self.control / "decay"

    @property
    def stop_request(self) -> Path:
        return self.control / "stop"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def state(self) -> Path:
        return self.root / "run_state.json"

    @property
    def training_done(self) -> Path:
        return self.root / "training_done"

    def create(self) -> None:
        for directory in (
            self.rounds,
            self.sample_set,
            self.samples,
            self.checkpoints,
            self.export,
            self.tensorboard,
            self.control,
            self.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def run_paths(run_name: str) -> DistillPaths:
    return DistillPaths(DISTILL_RUNS_DIRECTORY / run_name)
