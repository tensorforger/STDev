"""
Distributed single step benchmark of the Wan distillation on the fixed 1000-image set (wan_eval_1000, the set of
the pretraining evaluations): every rank upscales a contiguous slice 512 -> 1024 (true hff, fixed per image
noise, given guidance), decodes it and feeds patch / whole image FID; the fake FID statistics and the mse_lr
sums are all-reduced (when a process group exists; a single process evaluates everything), so the numbers equal a
single process evaluation (teacher, 20 steps: fid_patch 2.76).
"""

from pathlib import Path

import cv2
import torch
import torch.distributed as distributed
import torch.nn.functional as functional

import stdev.wan_pretrain.benchmark as benchmark
from stdev.pretrain.config import PretrainPaths
from stdev.utils.fid import FIDEvaluator
from stdev.utils.image_tools import torch_to_cv2
from stdev.vaes.wan_2_1_vae import Wan21VAE
from stdev.wan_distill.config import EVALUATION_BATCH, EVALUATION_NOISE_SEED, EVALUATION_SAVED_IMAGES
from stdev.wan_distill.student import WanStudentTransformer, upscale_single_step

FAKE_STATE_NAMES = ("fake_features_sum", "fake_features_cov_sum", "fake_features_num_samples")


class DistributedEvaluation:
    def __init__(self, data_paths: PretrainPaths, vae_directory: Path, rank: int, world_size: int, count: int = 0) -> None:
        """
        count: evaluate only the first `count` images (smoke tests; 0 = all)
        """
        self.benchmark_set = benchmark.BenchmarkSet(data_paths)
        self.fid_evaluator: FIDEvaluator = benchmark.build_fid_evaluator(data_paths, self.benchmark_set)
        for metric in self.fid_evaluator.metrics.values():
            metric.sync_on_compute = False
            metric._to_sync = False
        self.vae = Wan21VAE(str(vae_directory), device="cuda", dtype=torch.bfloat16, as_image=True)
        count = min(count, len(self.benchmark_set)) if count > 0 else len(self.benchmark_set)
        self.count = count
        self.indices = range(rank * count // world_size, (rank + 1) * count // world_size)
        self.target_size = tuple(self.benchmark_set.latents_target.shape[2:])

    def noise(self, indices: range) -> torch.Tensor:
        """
        returns:
            noise: [B, 16, 1, 128, 128], unit gaussian seeded per image index (noise scale 1 for spatial upscaling)
        """
        return torch.stack(
            [
                torch.randn((16, *self.target_size), generator=torch.Generator().manual_seed(EVALUATION_NOISE_SEED + index))
                for index in indices
            ]
        )

    @torch.no_grad()
    def evaluate(self, student: WanStudentTransformer, guidance: float, output_directory: Path | None, tag: str) -> dict[str, float]:
        """
        args:
            student: inference copy (eval mode) with the weights to evaluate
        returns:
            fid, fid_patch, mse_lr over the whole set (identical on all ranks)
        """
        squared_error_sum = torch.zeros((), device="cuda", dtype=torch.float64)
        for start in range(self.indices.start, self.indices.stop, EVALUATION_BATCH):
            indices = range(start, min(start + EVALUATION_BATCH, self.indices.stop))
            latents = upscale_single_step(
                student,
                self.benchmark_set.latents_source[indices.start : indices.stop].cuda(),
                self.benchmark_set.hff[indices.start : indices.stop].cuda(),
                torch.full((len(indices),), guidance, device="cuda"),
                self.noise(indices).cuda(),
            )
            images = benchmark.decode(self.vae, latents)
            self.fid_evaluator.update(images)
            ground_truth = self.benchmark_set.load_images(indices)
            squared_error_sum += (
                (functional.avg_pool2d(images, 2) - functional.avg_pool2d(ground_truth, 2)) ** 2
            ).mean(dim=(1, 2, 3)).double().sum()
            if output_directory is not None:
                for offset, index in enumerate(indices):
                    if index < EVALUATION_SAVED_IMAGES:
                        cv2.imwrite(str(output_directory / f"{index:05d}_{tag}.png"), torch_to_cv2(images[offset : offset + 1]))
        if distributed.is_initialized():
            for metric in self.fid_evaluator.metrics.values():
                for name in FAKE_STATE_NAMES:
                    distributed.all_reduce(getattr(metric, name))
            distributed.all_reduce(squared_error_sum)
        metrics = self.fid_evaluator.compute()
        metrics["mse_lr"] = squared_error_sum.item() / self.count
        return metrics
