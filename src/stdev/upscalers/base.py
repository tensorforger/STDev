from abc import ABC, abstractmethod
from typing import Any, TypeVar

import torch


class Upscaler(ABC):
    """
    Self-contained high-level wrapper of upscaler model.
    Each upscaler definition file should contain the implementation of this interface.
    """

    @abstractmethod
    def train(self, smoke_mode: bool = False) -> None:
        """
        Strats the training process.
        In smoke_mode should return after several training steps.
        """

    @abstractmethod
    def prepare_to_benchmark(self) -> None:
        """
        Should be called before benchmarking the upscaler.
        """

    @abstractmethod
    def compile(self) -> None:
        """
        Compiles the model with torch.compile. Called after prepare_to_benchmark().
        """

    @abstractmethod
    def upscale(self, tensor: torch.Tensor, prompt: str) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            prompt: text description of the image
        returns:
            tensor: [B, C, H_hr, W_hr]
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of upsclaer
        """
