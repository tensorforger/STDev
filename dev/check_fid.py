import tempfile
from pathlib import Path

import torch

from stdev.benchmark.dataset import BenchmarkDataset
from stdev.utils.fid import FIDEvaluator, FIDReference

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NUM_IMAGES = 32

real = BenchmarkDataset(
    REPOSITORY_ROOT / "benchmark_datasets" / "axis_1024", max_samples=NUM_IMAGES
)
teacher = BenchmarkDataset(
    REPOSITORY_ROOT / "benchmark_datasets" / "flux_1024", max_samples=NUM_IMAGES
)

with tempfile.TemporaryDirectory() as temporary_directory:
    evaluator = FIDEvaluator(
        references=[
            FIDReference(
                "real",
                str(real.images_directory),
                f"{temporary_directory}/real.pt",
                real.get_image_file_names(),
            ),
            FIDReference(
                "teacher",
                str(teacher.images_directory),
                f"{temporary_directory}/teacher.pt",
                teacher.get_image_file_names(),
            ),
        ],
        image_size=1024,
        batch_size=16,
    )
    for name, statistics in evaluator.real_statistics.items():
        print(
            name,
            statistics["real_features_sum"].sum().item(),
            statistics["real_features_num_samples"].item(),
        )
    for sample_name in real.sample_names:
        evaluator.update(real.load_image(sample_name))
    print("real images as fake:", evaluator.compute())
    for sample_name in teacher.sample_names:
        evaluator.update(teacher.load_image(sample_name))
    print("teacher images as fake:", evaluator.compute())
