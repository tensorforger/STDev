"""
Benchmark entry of the stage 2 pretrained model (ema weights) for scripts/run_benchmark.py.
High frequency features are not given (null embeddings).
"""

from pathlib import Path

from stdev.upscalers.pretrain_large import PretrainLargeUpscaler

WEIGHTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "pretrain_runs"
    / "stage2_v1"
    / "export"
    / "stage2_final_000054108_ema.safetensors"
)


class Stage2FinalEmaUpscaler(PretrainLargeUpscaler):
    def __init__(self) -> None:
        super().__init__(WEIGHTS_PATH)

    def get_name(self) -> str:
        return "pretrain_stage2_final_ema"
