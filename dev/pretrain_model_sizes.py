"""
Parameter counts of candidate pretraining architectures built from the run_0078 template.

Usage: python dev/pretrain_model_sizes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tools import load_run_module, resolve_run_file

CANDIDATES: dict[str, list[str]] = {
    "0078": [],
    "0077": [
        "LEVEL_CHANNELS=(512, 256, 512)",
        "LEVEL_BLOCKS=(1, 1, 1)",
        "DECODER_LEVEL_BLOCKS=(8, 1, 1)",
        "MIDDLE_BLOCKS=1",
        "ATTENTION_LEVELS=(False, True, True)",
        "TRANSFORMER_MLP=False",
    ],
    "512_dec8_transformer": ["DECODER_LEVEL_BLOCKS=(8, 2, 2)"],
    "640_dec8_transformer": ["LEVEL_CHANNELS=(640, 512, 1024)", "DECODER_LEVEL_BLOCKS=(8, 2, 2)"],
    "768_dec6_transformer": ["LEVEL_CHANNELS=(768, 512, 1024)", "DECODER_LEVEL_BLOCKS=(6, 2, 2)"],
    "768_dec8_transformer": ["LEVEL_CHANNELS=(768, 512, 1024)", "DECODER_LEVEL_BLOCKS=(8, 2, 2)"],
    "1024_dec6_transformer": ["LEVEL_CHANNELS=(1024, 512, 1024)", "DECODER_LEVEL_BLOCKS=(6, 2, 2)"],
    "1024_dec8_transformer": ["LEVEL_CHANNELS=(1024, 512, 1024)", "DECODER_LEVEL_BLOCKS=(8, 2, 2)"],
}


def count_parameters(overrides: list[str]) -> float:
    """
    returns:
        parameter count in millions
    """
    module = load_run_module(resolve_run_file("0078"), overrides)
    model = module.UpscalerUNet()
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


if __name__ == "__main__":
    for candidate_name, candidate_overrides in CANDIDATES.items():
        print(f"{candidate_name:28s} {count_parameters(candidate_overrides):7.1f}M")
