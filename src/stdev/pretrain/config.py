"""
Configuration of the online pretraining pipeline: stages, data sources, process layout,
training and benchmark schedule. Paths of one pretraining run live under PretrainPaths.
"""

from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PRETRAIN_RUNS_DIRECTORY = REPOSITORY_ROOT / "pretrain_runs"
VAE_DIRECTORY = REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B" / "vae"
SDXL_DIRECTORY = REPOSITORY_ROOT / "models" / "stable-diffusion-xl-base-1.0"

RELAION_DATASET_NAME = "laion/relaion-high-resolution"
RELAION_SHARD_COUNT = 128
BENCHMARK_SHARD_INDEX = 127
MAX_WATERMARK_PROBABILITY = 0.5
MAX_UNSAFE_PROBABILITY = 0.5
METADATA_SHUFFLE_SEED = 0

DOWNLOAD_PROCESSES = 8
DOWNLOAD_CONCURRENCY_PER_PROCESS = 256
MAX_CONNECTIONS_PER_HOST = 4
CONNECT_TIMEOUT_SECONDS = 5.0
TOTAL_TIMEOUT_SECONDS = 20.0
MAX_IMAGE_BYTES = 30 * 1024 * 1024
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) research-dataset-downloader"

PREPROCESS_PROCESSES = 48
URL_QUEUE_SIZE = 20000
IMAGE_QUEUE_SIZE = 2000
SAMPLE_QUEUE_SIZE = 2000

ENCODE_BATCH_SIZE = 32
SHARD_SAMPLE_COUNT = 512
CACHE_SIZE_LIMIT_BYTES = 300 * 1024**3
EVICTION_INTERVAL_SECONDS = 30.0

WARMUP_SECONDS = 30 * 60
HIGH_FREQUENCY_STATISTICS_SAMPLES = 10000
HIGH_FREQUENCY_QUANTILE_COUNT = 1001
BENCHMARK_SAMPLE_COUNT = 2000
BENCHMARK_SAVED_PAIRS = 100
BENCHMARK_BATCH_SIZE = 16
BENCHMARK_EARLY_INTERVAL_SECONDS = 30 * 60
BENCHMARK_EARLY_PERIOD_SECONDS = 4 * 3600
BENCHMARK_LATE_INTERVAL_SECONDS = 2 * 3600

TRAINING_GPUS = (0, 1, 2, 3, 4, 5, 6)
ENCODER_GPUS = (7,)
WARMUP_ENCODER_GPUS = (0, 1, 2, 3, 4, 5, 6, 7)

SEED = 0
LEARNING_RATE = 4e-4
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 500
DEFAULT_DECAY_FRACTION = 0.15
GRADIENT_CLIP_NORM = 1.0
TRAINING_SIGMA_SHIFT = 1.0
EMA_DECAY = 0.9999
TEXT_DROPOUT_PROBABILITY = 0.2
HIGH_FREQUENCY_DROPOUT_PROBABILITY = 0.2
LOG_INTERVAL_STEPS = 50
CONTROL_CHECK_INTERVAL_STEPS = 50
CHECKPOINT_INTERVAL_SECONDS = 3600
KEPT_CHECKPOINTS = 2
DATA_WORKERS_PER_RANK = 6


@dataclass(frozen=True)
class StageConfig:
    """
    One training stage: the resolution pair, the minimal source image side, the crop policy and
    the per GPU micro-batch (gradient accumulation gives the per GPU batch).
    Crop side is uniform in [high_resolution, short side] before the area resize.
    """

    name: str
    low_resolution: int
    high_resolution: int
    min_source_side: int
    micro_batch_size: int
    gradient_accumulation_steps: int


STAGES: dict[str, StageConfig] = {
    "stage_1": StageConfig("stage_1", 256, 512, 1024, 8, 2),
    "stage_2": StageConfig("stage_2", 512, 1024, 1536, 2, 2),
    "stage_3": StageConfig("stage_3", 1024, 2048, 2048, 1, 1),
}


@dataclass(frozen=True)
class PretrainPaths:
    """
    Directory layout of one pretraining run.
    """

    root: Path

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def benchmark_set(self) -> Path:
        return self.root / "benchmark_set"

    @property
    def statistics(self) -> Path:
        return self.root / "statistics"

    @property
    def high_frequency_quantiles(self) -> Path:
        return self.statistics / "high_frequency_quantiles.json"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"

    @property
    def benchmarks(self) -> Path:
        return self.root / "benchmarks"

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
    def encoder_pause(self) -> Path:
        return self.control / "encoder_pause"

    @property
    def stop_request(self) -> Path:
        return self.control / "stop"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def producer_state(self) -> Path:
        return self.root / "producer_state.json"

    @property
    def producer_statistics(self) -> Path:
        return self.root / "producer_statistics.json"

    @property
    def warmup_done(self) -> Path:
        return self.root / "warmup_done"

    @property
    def training_done(self) -> Path:
        return self.root / "training_done"

    def create(self) -> None:
        for directory in (
            self.cache,
            self.benchmark_set,
            self.statistics,
            self.checkpoints,
            self.snapshots,
            self.benchmarks,
            self.tensorboard,
            self.control,
            self.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def run_paths(run_name: str) -> PretrainPaths:
    return PretrainPaths(PRETRAIN_RUNS_DIRECTORY / run_name)
