import torch
import torch.nn.functional as functional

from stdev.upscalers.base import Upscaler


class BaselineUpscaler(Upscaler):
    """
    Bilinear interpolation of latents.
    """

    def __init__(self, scale_factor: int = 2) -> None:
        self.scale_factor = scale_factor

    def train(self) -> None:
        pass

    def prepare_to_benchmark(self) -> None:
        pass

    def compile(self) -> None:
        self._interpolate = torch.compile(self._interpolate)

    def _interpolate(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H_lr, W_lr]
        returns:
            tensor: [B, C, H_lr * scale_factor, W_lr * scale_factor]
        """
        return functional.interpolate(
            tensor, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )

    def upscale(self, tensor: torch.Tensor, prompt: str) -> torch.Tensor:
        """
        args:
            tensor: [B, C, H_lr, W_lr], dtype = float32 (or float16, bfloat16), device = "cuda"
            prompt: text description of the image, unused
        returns:
            tensor: [B, C, H_hr, W_hr]
        """
        return self._interpolate(tensor)

    def get_name(self) -> str:
        return "baseline"
