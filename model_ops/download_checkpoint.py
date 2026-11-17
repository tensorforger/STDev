"""
Downloads the newest checkpoint of a pretraining run from the checkpoints bucket and converts it to a
ComfyUI ready safetensors file (EMA or raw weights plus model config metadata).

Usage: python model_ops/download_checkpoint.py [--run wan_pretrain_v1] [--weights ema|raw] [--output-dir DIR]
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from stdev.s3_tools import download_archive, get_s3_client
from stdev.wan_pretrain.model import ModelConfig, WanUpscalerTransformer

BUCKET_NAME = "checkpoints"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "models" / "latent_upscale_models"
VAE_BY_RUN_PREFIX = {"wan_pretrain": "wan_2_1"}


def find_latest_key(client, run: str) -> str:
    listed = client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"{run}/").get("Contents", [])
    keys = sorted(item["Key"] for item in listed if item["Key"].endswith(".pt"))
    if not keys:
        raise FileNotFoundError(f"no checkpoints under s3://{BUCKET_NAME}/{run}/")
    return keys[-1]


def convert(
    checkpoint_path: Path, weights: str, vae_name: str, run: str, output_directory: Path
) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = dict(checkpoint["model_config"])
    step = checkpoint["state"]["step"]
    if weights == "ema":
        names = [
            name
            for name, _ in WanUpscalerTransformer(ModelConfig(**model_config)).named_parameters()
        ]
        tensors = {
            name: tensor.float().contiguous()
            for name, tensor in zip(names, checkpoint["ema"], strict=True)
        }
    else:
        tensors = {
            name: tensor.float().contiguous() for name, tensor in checkpoint["model"].items()
        }
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{vae_name}_{run}_{weights}_step{step:09d}.safetensors"
    metadata = {
        "vae": vae_name,
        "model_config": json.dumps(model_config),
        "run": run,
        "weights": weights,
        "step": str(step),
    }
    save_file(tensors, str(output_path), metadata=metadata)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="wan_pretrain_v1")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    arguments = parser.parse_args()

    vae_name = next(
        vae for prefix, vae in VAE_BY_RUN_PREFIX.items() if arguments.run.startswith(prefix)
    )
    client = get_s3_client()
    key = find_latest_key(client, arguments.run)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.output_dir / Path(key).name
    download_archive(client, BUCKET_NAME, key, str(checkpoint_path))
    output_path = convert(
        checkpoint_path, arguments.weights, vae_name, arguments.run, arguments.output_dir
    )
    checkpoint_path.unlink()
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
