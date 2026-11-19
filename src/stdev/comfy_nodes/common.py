"""
Shared helpers of the ComfyUI development node pack: model folder lookup, checkpoint metadata and
device / progress integration. Checkpoints are safetensors files in models/latent_upscale_models whose
metadata names the VAE they belong to (see model_ops/download_checkpoint.py).
"""

import json
import os
from contextlib import contextmanager
from typing import Iterator

import torch
from safetensors import safe_open

import folder_paths

MODEL_FOLDER_NAME = "latent_upscale_models"
METADATA_VAE_KEY = "vae"
METADATA_CONFIG_KEY = "model_config"
METADATA_ARCHITECTURE_KEY = "architecture"
DEFAULT_ARCHITECTURE = "WanUpscalerTransformer"


def register_model_folder() -> None:
    if MODEL_FOLDER_NAME not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[MODEL_FOLDER_NAME] = (
            [os.path.join(folder_paths.models_dir, MODEL_FOLDER_NAME)],
            {".safetensors"},
        )


def read_metadata(file_name: str) -> dict[str, str]:
    path = folder_paths.get_full_path_or_raise(MODEL_FOLDER_NAME, file_name)
    with safe_open(path, framework="pt") as file:
        return dict(file.metadata() or {})


def list_checkpoints(vae_name: str, architecture: str = DEFAULT_ARCHITECTURE) -> list[str]:
    """
    returns:
        file names in the model folder whose metadata marks them as upscalers of the given VAE and
        architecture (files without architecture metadata are the default architecture)
    """
    register_model_folder()
    matching = []
    for file_name in folder_paths.get_filename_list(MODEL_FOLDER_NAME):
        metadata = read_metadata(file_name)
        file_architecture = metadata.get(METADATA_ARCHITECTURE_KEY, DEFAULT_ARCHITECTURE)
        if metadata.get(METADATA_VAE_KEY) == vae_name and file_architecture == architecture:
            matching.append(file_name)
    return matching


def load_checkpoint(file_name: str) -> tuple[dict[str, torch.Tensor], dict]:
    """
    returns:
        weights: parameter name -> tensor, model_config: constructor arguments stored in the metadata
    """
    from safetensors.torch import load_file

    path = folder_paths.get_full_path_or_raise(MODEL_FOLDER_NAME, file_name)
    metadata = read_metadata(file_name)
    return load_file(path), json.loads(metadata[METADATA_CONFIG_KEY])


def compute_device() -> torch.device:
    import comfy.model_management

    return comfy.model_management.get_torch_device()


def offload_device() -> torch.device:
    import comfy.model_management

    return comfy.model_management.unet_offload_device()


def release_memory() -> None:
    import comfy.model_management

    comfy.model_management.soft_empty_cache()


def check_interrupted() -> None:
    import comfy.model_management

    comfy.model_management.throw_exception_if_processing_interrupted()


@contextmanager
def on_device(model: torch.nn.Module) -> Iterator[torch.device]:
    """
    Moves the model to the compute device for the duration of the block and back afterwards.
    """
    device = compute_device()
    model.to(device)
    try:
        yield device
    finally:
        model.to(offload_device())
        release_memory()


def make_progress_bar(total: int):
    import comfy.utils

    return comfy.utils.ProgressBar(total)
