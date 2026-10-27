import argparse
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

from stdev.text_encoders.sdxl_text_encoder import SDXLTextEncoder
from stdev.utils.image_tools import resize_torch, torch_to_cv2
from stdev.vaes.flux_2_vae import Flux2VAE

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = REPOSITORY_ROOT / "models" / "FLUX.2-klein-4B"
SDXL_DIRECTORY = REPOSITORY_ROOT / "models" / "stable-diffusion-xl-base-1.0"
BASE_PROMPTS_PATH = REPOSITORY_ROOT / "prompts" / "base_prompts.json"
MODIFIERS_PATH = REPOSITORY_ROOT / "prompts" / "modifiers.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "train_datasets" / "flux_1024_1k"
BENCHMARK_DIRECTORY = REPOSITORY_ROOT / "benchmark_datasets" / "flux_1024"

NUM_BLOCKS = 1000
NUM_BENCHMARK_SAMPLES = 1000
BENCHMARK_SEED_OFFSET = 1_000_000
BATCH_SIZE = 1
BASE_SEED = 0
IMAGE_SIZE = 1024
NUM_INFERENCE_STEPS = 4
SINGLE_BLOCK_INDEX = 15
MAX_SEQUENCE_LENGTH = 512
TEXT_ENCODER_OUTPUT_LAYERS = (9, 18, 27)
LATENT_CHANNELS = 32
VAE_SPATIAL_COMPRESSION = 8
LATENT_CODEC = "lbrq_8bit"
ACTIVATION_CODEC = "lbrq_4bit"
PROMPT_EMBEDS_CODEC = "cast_fp16"
DEVICE = "cuda"
DTYPE = torch.bfloat16


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Generate the benchmark dataset instead of the training blocks.",
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


def get_block_path(block_index: int) -> Path:
    return OUTPUT_DIRECTORY / f"block_{block_index:06d}.vz.safetensors"


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


@torch.no_grad()
def generate_blocks(
    specifications: list[dict],
    text_encoder: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
    transformer: Flux2Transformer2DModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    vae: Flux2VAE,
    captured_activations: dict[str, torch.Tensor],
    sdxl_text_encoder: SDXLTextEncoder,
) -> dict[str, torch.Tensor]:
    """
    returns:
        prompt_embeds: [B, 77, 768], dtype: torch.float32, SDXL text encoder embeddings
        init_latents: [B, 32, H/8, W/8], dtype: torch.bfloat16
        latents_orig: [B, 32, H/8, W/8], dtype: torch.bfloat16
        latents_div_2: [B, 32, H/16, W/16], dtype: torch.bfloat16
        activations_grid: [B, inner_dim, H/16, W/16], dtype: torch.bfloat16
    """
    batch_size = len(specifications)
    packed_size = IMAGE_SIZE // (VAE_SPATIAL_COMPRESSION * 2)
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
                (1, patchified_channels, packed_size, packed_size),
                generator=generator,
                device=DEVICE,
                dtype=DTYPE,
            )
        )
    noise = torch.cat(noise_items, dim=0)  # [B, 128, H/16, W/16]
    latents = noise.reshape(batch_size, patchified_channels, packed_size * packed_size).permute(
        0, 2, 1
    )  # [B, H*W/256, 128]
    latent_ids = build_latent_ids(batch_size, packed_size, packed_size)

    init_latents = vae.unpatchify_latents(noise)

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

    num_text_tokens = prompt_embeds.shape[1]
    block_activations = captured_activations["hidden_states"][:, num_text_tokens:, :]
    activations_grid = unpack_latents(block_activations, packed_size, packed_size)

    latents_orig = vae.unpatchify_latents(unpack_latents(latents, packed_size, packed_size))

    image = vae.decode(latents_orig)
    image_div_2 = resize_torch(
        image, height=IMAGE_SIZE // 2, width=IMAGE_SIZE // 2, interpolation="bilinear"
    )
    latents_div_2 = vae.encode(image_div_2)

    sdxl_prompt_embeds = sdxl_text_encoder.encode_text_batch(
        [specification["model_prompt"] for specification in specifications]
    )

    return {
        "prompt_embeds": sdxl_prompt_embeds.float(),
        "init_latents": init_latents,
        "latents_orig": latents_orig,
        "latents_div_2": latents_div_2,
        "activations_grid": activations_grid,
    }


def save_block(block_index: int, tensors: dict[str, torch.Tensor], item_index: int) -> None:
    safetensors_path = get_block_path(block_index)
    codecs = {
        "prompt_embeds": PROMPT_EMBEDS_CODEC,
        "init_latents": LATENT_CODEC,
        "latents_orig": LATENT_CODEC,
        "latents_div_2": LATENT_CODEC,
        "activations_grid": ACTIVATION_CODEC,
    }
    encoded_tensors = {
        name: vaezip.encode(tensors[name][item_index : item_index + 1], codec=codec)
        for name, codec in codecs.items()
    }

    temporary_path = safetensors_path.with_name(safetensors_path.name + ".tmp")
    vaezip.save(encoded_tensors, path=temporary_path)
    os.replace(temporary_path, safetensors_path)


def get_benchmark_paths(sample_index: int) -> tuple[Path, Path]:
    sample_name = f"flux_{sample_index:06d}"
    return (
        BENCHMARK_DIRECTORY / "images" / f"{sample_name}.png",
        BENCHMARK_DIRECTORY / "prompt_embeds" / f"{sample_name}.vz.safetensors",
    )


@torch.no_grad()
def save_benchmark_sample(
    sample_index: int, tensors: dict[str, torch.Tensor], item_index: int, vae: Flux2VAE
) -> None:
    """
    args:
        tensors: output of generate_blocks()
    """
    image_path, prompt_embeds_path = get_benchmark_paths(sample_index)
    latents = tensors["latents_orig"][item_index : item_index + 1]  # [1, 32, H/8, W/8]
    image = vae.decode(latents).float().clamp(0, 1)
    cv2.imwrite(str(image_path), torch_to_cv2(image))
    vaezip.save_tensor(
        tensors["prompt_embeds"][item_index : item_index + 1],
        prompt_embeds_path,
        codec=PROMPT_EMBEDS_CODEC,
    )


def main() -> None:
    arguments = parse_arguments()
    benchmark_mode = arguments.benchmark
    output_directory = BENCHMARK_DIRECTORY if benchmark_mode else OUTPUT_DIRECTORY
    if benchmark_mode:
        (BENCHMARK_DIRECTORY / "images").mkdir(parents=True, exist_ok=True)
        (BENCHMARK_DIRECTORY / "prompt_embeds").mkdir(parents=True, exist_ok=True)
    else:
        output_directory.mkdir(parents=True, exist_ok=True)
    base_prompts, modifiers = load_prompt_sources()

    if benchmark_mode:
        item_indices = range(1, NUM_BENCHMARK_SAMPLES + 1)
        seed_offset = BENCHMARK_SEED_OFFSET
        is_done = lambda index: all(path.exists() for path in get_benchmark_paths(index))
    else:
        item_indices = range(1, NUM_BLOCKS + 1)
        seed_offset = BASE_SEED
        is_done = lambda index: get_block_path(index).exists()

    pending_indices = [index for index in item_indices if not is_done(index)]
    if not pending_indices:
        print(f"Everything already exists in {output_directory}")
        return

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        MODEL_DIRECTORY / "text_encoder", local_files_only=True, dtype=DTYPE
    ).to(DEVICE)
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        MODEL_DIRECTORY / "tokenizer", local_files_only=True
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_DIRECTORY / "scheduler")
    transformer = Flux2Transformer2DModel.from_pretrained(
        MODEL_DIRECTORY / "transformer", torch_dtype=DTYPE
    ).to(DEVICE)
    vae = Flux2VAE(str(MODEL_DIRECTORY / "vae"), device=DEVICE, dtype=DTYPE)
    sdxl_text_encoder = SDXLTextEncoder(str(SDXL_DIRECTORY), device=DEVICE, dtype=torch.float32)

    text_encoder.eval()
    transformer.eval()

    # The hook stays registered for the whole run so the compiled graph is not invalidated.
    captured_activations: dict[str, torch.Tensor] = {}

    def capture_hook(module, module_inputs, module_output):
        captured_activations["hidden_states"] = module_output

    transformer.single_transformer_blocks[SINGLE_BLOCK_INDEX].register_forward_hook(capture_hook)

    text_encoder.model = torch.compile(text_encoder.model)
    transformer = torch.compile(transformer)
    vae.model.encoder = torch.compile(vae.model.encoder)
    vae.model.decoder = torch.compile(vae.model.decoder)

    with tqdm(total=len(pending_indices), desc="Generating blocks") as progress_bar:
        for batch_start in range(0, len(pending_indices), BATCH_SIZE):
            batch_indices = pending_indices[batch_start : batch_start + BATCH_SIZE]
            specifications = [
                build_block_specification(seed_offset + block_index, base_prompts, modifiers)
                for block_index in batch_indices
            ]
            tensors = generate_blocks(
                specifications,
                text_encoder,
                tokenizer,
                transformer,
                scheduler,
                vae,
                captured_activations,
                sdxl_text_encoder,
            )
            for item_index, block_index in enumerate(batch_indices):
                if benchmark_mode:
                    save_benchmark_sample(block_index, tensors, item_index, vae)
                else:
                    save_block(block_index, tensors, item_index)
            progress_bar.update(len(batch_indices))


if __name__ == "__main__":
    main()
