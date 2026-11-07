"""
Generates the FLUX.2-klein-4B teacher dataset for SFT of the pretrain_large latent upscaler:
samples the teacher at each configured resolution bucket (`stdev.sft.config.RESOLUTION_BUCKETS`),
decodes the high resolution image, re-encodes its bilinear 2x downscale and measures the high
frequency features of the decoded image. Each sample is written as one `.vz.safetensors` block
(`latents_orig`, `latents_div_2`, `text_embedding` [pooled SDXL embedding], `high_frequency_raw`)
in its bucket's directory. `--benchmark` instead writes the older fixed resolution image /
prompt_embeds benchmark set used by scripts/run_benchmark.py.

Usage: torchrun --standalone --nproc_per_node=8 scripts/generate_flux_dataset.py
           torchrun --standalone --nproc_per_node=8 scripts/generate_flux_dataset.py --benchmark
           python scripts/generate_flux_dataset.py --smoke   (tiny dataset for a quick check)
"""

import argparse
import dataclasses
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import vaezip
from diffusers.models import Flux2Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from stdev.pretrain.high_frequency import compute_feature_vector
from stdev.sft.config import (
    BASE_PROMPTS_PATH,
    BENCHMARK_DIRECTORY,
    BENCHMARK_IMAGE_SIZE,
    BENCHMARK_SEED_OFFSET,
    BUCKET_SEED_STRIDE,
    DATASET_DIRECTORY as DEFAULT_DATASET_DIRECTORY,
    FLUX_MODEL_DIRECTORY,
    LATENT_CHANNELS,
    MAX_SEQUENCE_LENGTH,
    MODIFIERS_PATH,
    NUM_BENCHMARK_SAMPLES,
    NUM_INFERENCE_STEPS,
    PROMPT_EMBEDS_CODEC,
    RESOLUTION_BUCKETS,
    SDXL_DIRECTORY,
    TEXT_ENCODER_OUTPUT_LAYERS,
    VAE_SPATIAL_COMPRESSION,
    ResolutionBucket,
)
from stdev.sft.data import block_path, write_block
from stdev.text_encoders.sdxl_text_encoder import SDXLTextEncoder
from stdev.utils.image_tools import resize_torch, torch_to_cv2
from stdev.vaes.flux_2_vae import Flux2VAE

DTYPE = torch.bfloat16
SMOKE_BLOCKS_PER_BUCKET = 2

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Generate the fixed resolution benchmark set instead of the training blocks.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Overrides the training blocks directory (default: stdev.sft.config.DATASET_DIRECTORY).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Generate only {SMOKE_BLOCKS_PER_BUCKET} blocks per bucket, for a quick check.",
    )
    return parser.parse_args()


def load_prompt_sources() -> tuple[list[str], list[str]]:
    base_prompts = json.loads(BASE_PROMPTS_PATH.read_text())
    modifiers = json.loads(MODIFIERS_PATH.read_text())["modifiers"]
    return base_prompts, modifiers


def build_block_specification(seed: int, base_prompts: list[str], modifiers: list[str]) -> dict:
    random_generator = random.Random(seed)
    modifier = random_generator.choice(modifiers)
    base_prompt = random_generator.choice(base_prompts)

    return {"model_prompt": f"{modifier} {base_prompt}", "seed": seed}


@torch.no_grad()
def encode_prompts(
    prompts: list[str],
    text_encoder: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    args:
        prompts: list of B strings
    returns:
        prompt_embeds: [B, MAX_SEQUENCE_LENGTH, 3 * hidden_dim], dtype: torch.bfloat16
        text_ids: [B, MAX_SEQUENCE_LENGTH, 4], dtype: torch.int64
    """
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    inputs = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    ).to(DEVICE)

    output = text_encoder.model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )

    selected_states = torch.stack(
        [output.hidden_states[layer] for layer in TEXT_ENCODER_OUTPUT_LAYERS], dim=1
    )  # [B, L, S, D]
    batch_size, num_layers, sequence_length, hidden_dim = selected_states.shape
    prompt_embeds = selected_states.permute(0, 2, 1, 3).reshape(
        batch_size, sequence_length, num_layers * hidden_dim
    )

    text_ids = torch.zeros(sequence_length, 4, dtype=torch.int64, device=DEVICE)
    text_ids[:, 3] = torch.arange(sequence_length, device=DEVICE)
    text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1)

    return prompt_embeds, text_ids


def build_latent_ids(batch_size: int, latent_height: int, latent_width: int) -> torch.Tensor:
    """
    returns:
        latent_ids: [B, H*W, 4], dtype: torch.int64, coordinates (time=0, row, column, layer=0)
    """
    rows = torch.arange(latent_height, device=DEVICE)
    columns = torch.arange(latent_width, device=DEVICE)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    latent_ids = torch.zeros(latent_height * latent_width, 4, dtype=torch.int64, device=DEVICE)
    latent_ids[:, 1] = row_grid.reshape(-1)
    latent_ids[:, 2] = column_grid.reshape(-1)

    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1)


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    args:
        latents: [B, H*W, C]
    returns:
        latents: [B, C, H, W]
    """
    batch_size, _, num_channels = latents.shape
    return latents.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)


def compute_empirical_mu(image_sequence_length: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_sequence_length > 4300:
        return float(a2 * image_sequence_length + b2)

    mu_200 = a2 * image_sequence_length + b2
    mu_10 = a1 * image_sequence_length + b1
    slope = (mu_200 - mu_10) / 190.0
    intercept = mu_200 - 200.0 * slope

    return float(slope * num_steps + intercept)


def compute_high_frequency_batch(image: torch.Tensor) -> torch.Tensor:
    """
    args:
        image: [B, 3, H, W], channel_order: RGB, pixel_values: [0...1]
    returns:
        high_frequency_raw: [B, 4], raw (not quantile normalized) feature values, dtype: float32
    """
    features = [
        compute_feature_vector(torch_to_cv2(image[index : index + 1]))
        for index in range(image.shape[0])
    ]
    return torch.from_numpy(np.stack(features)).float()


@torch.no_grad()
def generate_samples(
    specifications: list[dict],
    text_encoder: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
    transformer: Flux2Transformer2DModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    vae: Flux2VAE,
    sdxl_text_encoder: SDXLTextEncoder,
    height: int,
    width: int,
) -> dict[str, torch.Tensor]:
    """
    Teacher generation at one target resolution: samples the FLUX.2 model, decodes the high
    resolution image, re-encodes its bilinear 2x downscale and measures its high frequency
    features.

    returns:
        sdxl_prompt_embeds: [B, 77, 768], dtype: torch.float32, full SDXL sequence embeddings
        latents_orig: [B, 32, height/8, width/8], dtype: torch.bfloat16
        latents_div_2: [B, 32, height/16, width/16], dtype: torch.bfloat16
        high_frequency_raw: [B, 4], dtype: torch.float32
    """
    batch_size = len(specifications)
    packed_height = height // (VAE_SPATIAL_COMPRESSION * 2)
    packed_width = width // (VAE_SPATIAL_COMPRESSION * 2)
    patchified_channels = LATENT_CHANNELS * 4

    prompt_embeds, text_ids = encode_prompts(
        [specification["model_prompt"] for specification in specifications],
        text_encoder,
        tokenizer,
    )

    noise_items = []
    for specification in specifications:
        generator = torch.Generator(device=DEVICE).manual_seed(specification["seed"])
        noise_items.append(
            torch.randn(
                (1, patchified_channels, packed_height, packed_width),
                generator=generator,
                device=DEVICE,
                dtype=DTYPE,
            )
        )
    noise = torch.cat(noise_items, dim=0)  # [B, 128, H/16, W/16]
    latents = noise.reshape(batch_size, patchified_channels, packed_height * packed_width).permute(
        0, 2, 1
    )  # [B, H*W/256, 128]
    latent_ids = build_latent_ids(batch_size, packed_height, packed_width)

    sigmas = np.linspace(1.0, 1 / NUM_INFERENCE_STEPS, NUM_INFERENCE_STEPS)
    mu = compute_empirical_mu(image_sequence_length=latents.shape[1], num_steps=NUM_INFERENCE_STEPS)
    scheduler.set_timesteps(sigmas=sigmas, device=DEVICE, mu=mu)
    scheduler.set_begin_index(0)

    for timestep in scheduler.timesteps:
        timestep_batch = timestep.expand(batch_size).to(DTYPE)
        noise_prediction = transformer(
            hidden_states=latents,
            timestep=timestep_batch / 1000,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            return_dict=False,
        )[0]
        latents = scheduler.step(noise_prediction, timestep, latents, return_dict=False)[0]

    latents_orig = vae.unpatchify_latents(unpack_latents(latents, packed_height, packed_width))

    image = vae.decode(latents_orig).float().clamp(0, 1)
    image_div_2 = resize_torch(
        image, height=height // 2, width=width // 2, interpolation="bilinear"
    )
    latents_div_2 = vae.encode(image_div_2)

    sdxl_prompt_embeds = sdxl_text_encoder.encode_text_batch(
        [specification["model_prompt"] for specification in specifications]
    ).float()

    return {
        "sdxl_prompt_embeds": sdxl_prompt_embeds,
        "latents_orig": latents_orig,
        "latents_div_2": latents_div_2,
        "high_frequency_raw": compute_high_frequency_batch(image),
    }


def save_training_block(
    dataset_directory: Path,
    bucket: ResolutionBucket,
    block_index: int,
    tensors: dict[str, torch.Tensor],
    item_index: int,
) -> None:
    pooled_text_embedding = tensors["sdxl_prompt_embeds"][item_index : item_index + 1].mean(dim=1)
    write_block(
        dataset_directory,
        bucket,
        block_index,
        latents_orig=tensors["latents_orig"][item_index : item_index + 1],
        latents_div_2=tensors["latents_div_2"][item_index : item_index + 1],
        text_embedding=pooled_text_embedding,
        high_frequency_raw=tensors["high_frequency_raw"][item_index : item_index + 1],
    )


def get_benchmark_paths(sample_index: int) -> tuple[Path, Path]:
    sample_name = f"flux_{sample_index:06d}"
    return (
        BENCHMARK_DIRECTORY / "images" / f"{sample_name}.png",
        BENCHMARK_DIRECTORY / "prompt_embeds" / f"{sample_name}.vz.safetensors",
    )


def save_benchmark_sample(
    sample_index: int, tensors: dict[str, torch.Tensor], item_index: int
) -> None:
    """
    args:
        tensors: output of generate_samples(), with an extra "decoded_image" entry
            ([B, 3, H, W], channel_order: RGB, pixel_values: [0...1])
    """
    image_path, prompt_embeds_path = get_benchmark_paths(sample_index)
    cv2.imwrite(
        str(image_path), torch_to_cv2(tensors["decoded_image"][item_index : item_index + 1])
    )
    vaezip.save_tensor(
        tensors["sdxl_prompt_embeds"][item_index : item_index + 1],
        prompt_embeds_path,
        codec=PROMPT_EMBEDS_CODEC,
    )


def main() -> None:
    torch.cuda.set_device(DEVICE)
    arguments = parse_arguments()
    benchmark_mode = arguments.benchmark
    dataset_directory = arguments.output_directory or DEFAULT_DATASET_DIRECTORY
    resolution_buckets = RESOLUTION_BUCKETS
    if arguments.smoke:
        resolution_buckets = tuple(
            dataclasses.replace(bucket, num_blocks=SMOKE_BLOCKS_PER_BUCKET)
            for bucket in RESOLUTION_BUCKETS
        )
    base_prompts, modifiers = load_prompt_sources()

    if benchmark_mode:
        (BENCHMARK_DIRECTORY / "images").mkdir(parents=True, exist_ok=True)
        (BENCHMARK_DIRECTORY / "prompt_embeds").mkdir(parents=True, exist_ok=True)
        num_benchmark_samples = (
            SMOKE_BLOCKS_PER_BUCKET if arguments.smoke else NUM_BENCHMARK_SAMPLES
        )
        all_items: list[tuple[ResolutionBucket | None, int]] = [
            (None, index) for index in range(1, num_benchmark_samples + 1)
        ]
        is_done = lambda item: all(path.exists() for path in get_benchmark_paths(item[1]))
    else:
        for bucket in resolution_buckets:
            (dataset_directory / bucket.name).mkdir(parents=True, exist_ok=True)
        all_items = [
            (bucket, block_index)
            for bucket in resolution_buckets
            for block_index in range(1, bucket.num_blocks + 1)
        ]
        is_done = lambda item: block_path(dataset_directory, item[0], item[1]).exists()

    # Each rank takes a fixed stride of the full item list, so ranks never overlap regardless of
    # which files already exist when they start.
    pending_items = [item for item in all_items[RANK::WORLD_SIZE] if not is_done(item)]
    if not pending_items:
        print(f"[rank {RANK}] Everything already exists", flush=True)
        return

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        FLUX_MODEL_DIRECTORY / "text_encoder", local_files_only=True, dtype=DTYPE
    ).to(DEVICE)
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        FLUX_MODEL_DIRECTORY / "tokenizer", local_files_only=True
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(FLUX_MODEL_DIRECTORY / "scheduler")
    transformer = Flux2Transformer2DModel.from_pretrained(
        FLUX_MODEL_DIRECTORY / "transformer", torch_dtype=DTYPE
    ).to(DEVICE)
    vae = Flux2VAE(str(FLUX_MODEL_DIRECTORY / "vae"), device=DEVICE, dtype=DTYPE)
    sdxl_text_encoder = SDXLTextEncoder(str(SDXL_DIRECTORY), device=DEVICE, dtype=torch.float32)

    text_encoder.eval()
    transformer.eval()

    text_encoder.model = torch.compile(text_encoder.model)
    transformer = torch.compile(transformer)
    vae.model.encoder = torch.compile(vae.model.encoder)
    vae.model.decoder = torch.compile(vae.model.decoder)

    with tqdm(
        total=len(pending_items), desc=f"Generating [rank {RANK}]", position=RANK
    ) as progress_bar:
        for bucket, item_index in pending_items:
            if benchmark_mode:
                height = width = BENCHMARK_IMAGE_SIZE
                seed = BENCHMARK_SEED_OFFSET + item_index
            else:
                height, width = bucket.height, bucket.width
                seed = resolution_buckets.index(bucket) * BUCKET_SEED_STRIDE + item_index
            specification = build_block_specification(seed, base_prompts, modifiers)
            tensors = generate_samples(
                [specification],
                text_encoder,
                tokenizer,
                transformer,
                scheduler,
                vae,
                sdxl_text_encoder,
                height,
                width,
            )
            if benchmark_mode:
                tensors["decoded_image"] = vae.decode(tensors["latents_orig"]).float().clamp(0, 1)
                save_benchmark_sample(item_index, tensors, 0)
            else:
                save_training_block(dataset_directory, bucket, item_index, tensors, 0)
            progress_bar.update(1)


if __name__ == "__main__":
    main()
