"""
Generates the round 7 run files from run_0043 by exact text substitutions.
Round 7 is step matched: every run trains exactly TRAINING_STEPS steps (LR decay over the last 30%
of steps) and saves checkpoints at 25/50/75%, so REPA and global context modules are compared at
equal data and steps (sample efficiency, the regime that matters at scale), not at equal time.
"""

from pathlib import Path

UPSCALERS_DIRECTORY = Path(__file__).resolve().parents[1] / "src" / "stdev" / "upscalers"
BASE_RUN = "0043"
BASE_DOCSTRING = """    Stacked recipe (training shift 1, LR 4e-4, linear decay over the last 30%, pooled text
    condition) with levels (256, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1)."""
DOCSTRING_PREFIX = """    Step matched (TRAINING_STEPS, checkpoints at 25/50/75%) run_0043: stacked recipe, levels
    (256, 256, 512), encoder blocks (1, 1, 1), decoder blocks (3, 1, 1)"""

STEP_MATCHED = [
    (
        "TRAINING_TIME_SECONDS = 30 * 60\n",
        "TRAINING_TIME_SECONDS = 30 * 60\nTRAINING_STEPS = 9_000\n"
        "CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75)\n",
    ),
    (
        """                remaining_fraction = 1.0 - (time.perf_counter() - training_start_time) / (
                    TRAINING_TIME_SECONDS
                )
""",
        """                remaining_fraction = 1.0 - step / TRAINING_STEPS
""",
    ),
    (
        """                is_finished = elapsed_seconds >= TRAINING_TIME_SECONDS""",
        """                is_finished = (
                    step >= TRAINING_STEPS or elapsed_seconds >= TRAINING_TIME_SECONDS
                )""",
    ),
    (
        """            if smoke_mode:
                is_finished = step >= SMOKE_STEPS""",
        """            if not smoke_mode and step in checkpoint_steps:
                OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
                checkpoint_path = OUTPUT_DIRECTORY / f"checkpoint_{step:06d}.safetensors"
                save_file(model.state_dict(), str(checkpoint_path))

            if smoke_mode:
                is_finished = step >= SMOKE_STEPS""",
    ),
    (
        """        step = 0
        loss_sum = torch.zeros((), device=DEVICE)""",
        """        step = 0
        checkpoint_steps = {round(fraction * TRAINING_STEPS) for fraction in CHECKPOINT_FRACTIONS}
        loss_sum = torch.zeros((), device=DEVICE)""",
    ),
]

UNET_ANCHOR = "class UpscalerUNet(nn.Module):"


def representation_alignment(level_name: str) -> list[tuple[str, str]]:
    """
    REPA: an MLP projection of UNet features matches the teacher activations (cosine similarity).
    level_name "level_1": encoder output at 1/2 resolution (64x64 for training, the teacher grid).
    level_name "middle": middle block output at 1/4 resolution, teacher average pooled 2x.
    """
    if level_name == "level_1":
        channels_expression = "LEVEL_CHANNELS[1]"
        features_expression = "skips[1]"
        teacher_expression = "teacher_activations"
    else:
        channels_expression = "LEVEL_CHANNELS[-1]"
        features_expression = "middle_features"
        teacher_expression = "functional.avg_pool2d(teacher_activations, 2)"
    replacements = [
        (
            "LOG_INTERVAL_STEPS = 100\n",
            "LOG_INTERVAL_STEPS = 100\n"
            'ACTIVATIONS_KEY = "activations_grid"\n'
            "ACTIVATION_CHANNELS = 3072\n"
            "ACTIVATION_GRID_SIZE = 64\n"
            "PROJECTOR_HIDDEN_DIM = 2048\n"
            "ALIGNMENT_LOSS_WEIGHT = 0.5\n",
        ),
        (
            '''    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
            latents_full: [C, H, W], dtype: torch.float32
            latents_div_2: [C, H / 2, W / 2], dtype: torch.float32
            prompt_embeds: [77, 768], dtype: torch.float32
        """
        encoded_block = vaezip.load(self.block_paths[index])
        latents_full, latents_div_2, prompt_embeds = (
            encoded_block[name].decode(dtype=torch.float32)[0]
            for name in ("latents_orig", "latents_div_2", "prompt_embeds")
        )
        return latents_full, latents_div_2, prompt_embeds''',
            '''    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        returns:
            latents_full: [C, H, W], dtype: torch.float32
            latents_div_2: [C, H / 2, W / 2], dtype: torch.float32
            prompt_embeds: [77, 768], dtype: torch.float32
            encoded_activations: vaezip tensors of activations_grid without the shape entry,
                each [1, ...], decoded on the GPU in batches
        """
        encoded_block = vaezip.load(self.block_paths[index])
        latents_full, latents_div_2, prompt_embeds = (
            encoded_block[name].decode(dtype=torch.float32)[0]
            for name in ("latents_orig", "latents_div_2", "prompt_embeds")
        )
        encoded_activations = {
            name: tensor
            for name, tensor in encoded_block[ACTIVATIONS_KEY].tensors.items()
            if name != "shape"
        }
        return latents_full, latents_div_2, prompt_embeds, encoded_activations


def read_activations_codec_metadata(dataset_directory: Path) -> dict[str, str]:
    first_block_path = next(iter(sorted(dataset_directory.glob("block_*.vz.safetensors"))))
    return dict(vaezip.load(first_block_path)[ACTIVATIONS_KEY].metadata)


def decode_activations(
    encoded_activations: dict[str, torch.Tensor], metadata: dict[str, str]
) -> torch.Tensor:
    """
    args:
        encoded_activations: collated vaezip tensors, each [B, 1, ...], on the GPU
    returns:
        activations: [B, ACTIVATION_CHANNELS, ACTIVATION_GRID_SIZE, ACTIVATION_GRID_SIZE],
            dtype: torch.bfloat16
    """
    tensors = {name: tensor.flatten(0, 1) for name, tensor in encoded_activations.items()}
    batch_size = next(iter(tensors.values())).shape[0]
    tensors["shape"] = torch.tensor(
        [batch_size, ACTIVATION_CHANNELS, ACTIVATION_GRID_SIZE, ACTIVATION_GRID_SIZE],
        dtype=torch.int64,
    )
    return vaezip.EncodedTensor(tensors=tensors, metadata=metadata).decode(dtype=torch.bfloat16)''',
        ),
        (
            "def iterate_batches(data_loader: DataLoader) -> Iterator[tuple[torch.Tensor, ...]]:",
            "def iterate_batches(data_loader: DataLoader) -> Iterator[tuple]:",
        ),
        (
            UNET_ANCHOR,
            f'''class FeatureProjector(nn.Module):
    """
    Per position MLP mapping UNet features to the teacher activation space, training only.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d({channels_expression}, PROJECTOR_HIDDEN_DIM, 1),
            nn.SiLU(),
            nn.Conv2d(PROJECTOR_HIDDEN_DIM, PROJECTOR_HIDDEN_DIM, 1),
            nn.SiLU(),
            nn.Conv2d(PROJECTOR_HIDDEN_DIM, ACTIVATION_CHANNELS, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, {channels_expression}, h, w]
        returns:
            projected: [B, ACTIVATION_CHANNELS, h, w]
        """
        return self.layers(features)


{UNET_ANCHOR}''',
        ),
        (
            """        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"
        args:
            noisy_latents: [B, C, H, W]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_low_resolution: [B, C, H / 2, W / 2]
            prompt_embeds: [B, 77, TEXT_EMBEDDING_DIM]
        returns:
            velocity: [B, C, H, W]
        \"\"\"""",
            f"""        prompt_embeds: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        \"\"\"
        args:
            noisy_latents: [B, C, H, W]
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_low_resolution: [B, C, H / 2, W / 2]
            prompt_embeds: [B, 77, TEXT_EMBEDDING_DIM]
        returns:
            velocity: [B, C, H, W]
            features: [B, {channels_expression}, h, w] for the alignment loss, only if
                return_features
        \"\"\"""",
        ),
        (
            "        return self.output_conv(functional.silu(self.output_norm(features)))",
            f"""        velocity = self.output_conv(functional.silu(self.output_norm(features)))
        if return_features:
            return velocity, {features_expression}
        return velocity""",
        ),
        (
            """        model = UpscalerUNet().to(DEVICE)
        compiled_model = torch.compile(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)""",
            """        model = UpscalerUNet().to(DEVICE)
        projector = FeatureProjector().to(DEVICE)
        compiled_model = torch.compile(model)
        compiled_projector = torch.compile(projector)
        trained_parameters = list(model.parameters()) + list(projector.parameters())
        optimizer = torch.optim.AdamW(trained_parameters, lr=LEARNING_RATE, betas=ADAM_BETAS)
        activations_metadata = read_activations_codec_metadata(DATASET_DIRECTORY)""",
        ),
        (
            """        loss_sum = torch.zeros((), device=DEVICE)
        for batch in batches:
            latents_full, latents_div_2, prompt_embeds = (
                tensor.to(DEVICE, non_blocking=True) for tensor in batch
            )  # [B, C, H, W], [B, C, H / 2, W / 2], [B, 77, 768]""",
            """        loss_sum = torch.zeros((), device=DEVICE)
        alignment_loss_sum = torch.zeros((), device=DEVICE)
        for batch in batches:
            latents_full, latents_div_2, prompt_embeds = (
                tensor.to(DEVICE, non_blocking=True) for tensor in batch[:3]
            )  # [B, C, H, W], [B, C, H / 2, W / 2], [B, 77, 768]
            teacher_activations = decode_activations(
                {name: tensor.to(DEVICE, non_blocking=True) for name, tensor in batch[3].items()},
                activations_metadata,
            )  # [B, ACTIVATION_CHANNELS, H / 2, W / 2]""",
        ),
        (
            """            with torch.autocast(DEVICE, dtype=TRAINING_DTYPE):
                velocity = compiled_model(noisy_latents, sigmas, latents_div_2, prompt_embeds)
            loss = functional.mse_loss(velocity.float(), noise - latents_full)""",
            f"""            with torch.autocast(DEVICE, dtype=TRAINING_DTYPE):
                velocity, features = compiled_model(
                    noisy_latents, sigmas, latents_div_2, prompt_embeds, return_features=True
                )
                projected = compiled_projector(features)
            flow_loss = functional.mse_loss(velocity.float(), noise - latents_full)
            alignment_loss = 1.0 - functional.cosine_similarity(
                projected.float(), {teacher_expression}.float(), dim=1
            ).mean()
            loss = flow_loss + ALIGNMENT_LOSS_WEIGHT * alignment_loss""",
        ),
        (
            "            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)",
            "            torch.nn.utils.clip_grad_norm_(trained_parameters, GRADIENT_CLIP_NORM)",
        ),
        (
            """            step += 1
            loss_sum += loss.detach()""",
            """            step += 1
            loss_sum += flow_loss.detach()
            alignment_loss_sum += alignment_loss.detach()""",
        ),
        (
            """                mean_loss = loss_sum.item() / LOG_INTERVAL_STEPS
                print(
                    f"step {step} loss {mean_loss:.4f} elapsed {elapsed_seconds:.0f}s", flush=True
                )
                if summary_writer is not None:
                    summary_writer.add_scalar("train/loss", mean_loss, step)
                loss_sum.zero_()""",
            """                mean_loss = loss_sum.item() / LOG_INTERVAL_STEPS
                mean_alignment_loss = alignment_loss_sum.item() / LOG_INTERVAL_STEPS
                print(
                    f"step {step} loss {mean_loss:.4f} alignment {mean_alignment_loss:.4f} "
                    f"elapsed {elapsed_seconds:.0f}s",
                    flush=True,
                )
                if summary_writer is not None:
                    summary_writer.add_scalar("train/loss", mean_loss, step)
                    summary_writer.add_scalar("train/alignment_loss", mean_alignment_loss, step)
                loss_sum.zero_()
                alignment_loss_sum.zero_()""",
        ),
    ]
    if level_name == "middle":
        replacements.append(
            (
                "        features = self.middle_blocks(features, condition)\n",
                "        features = self.middle_blocks(features, condition)\n"
                "        middle_features = features\n",
            )
        )
    return replacements


ATTENTION_CONSTANT = ("TIMESTEP_SCALE = 1000.0\n", "TIMESTEP_SCALE = 1000.0\nATTENTION_HEAD_DIM = 64\n")


def context_at_levels(module_name: str) -> list[tuple[str, str]]:
    """
    Wraps the encoder and decoder stages of levels >= 1 and the middle blocks with a context module
    taking only the features.
    """
    return [
        (
            """        self.output_norm = make_group_norm(LEVEL_CHANNELS[0])""",
            f"""        self.encoder_contexts = nn.ModuleList(
            {module_name}(LEVEL_CHANNELS[level]) if level >= 1 else nn.Identity()
            for level in range(len(LEVEL_CHANNELS))
        )
        self.middle_context = {module_name}(LEVEL_CHANNELS[-1])
        self.decoder_contexts = nn.ModuleList(
            {module_name}(LEVEL_CHANNELS[level]) if level >= 1 else nn.Identity()
            for level in range(len(LEVEL_CHANNELS))
        )
        self.output_norm = make_group_norm(LEVEL_CHANNELS[0])""",
        ),
        (
            "            features = encoder_stage(features, condition)\n",
            "            features = self.encoder_contexts[level](encoder_stage(features, condition))\n",
        ),
        (
            "        features = self.middle_blocks(features, condition)\n",
            "        features = self.middle_context(self.middle_blocks(features, condition))\n",
        ),
        (
            "            features = self.decoder_stages[level](torch.cat([features, skip], dim=1), condition)\n",
            """            features = self.decoder_contexts[level](
                self.decoder_stages[level](torch.cat([features, skip], dim=1), condition)
            )
""",
        ),
    ]


WINDOW_ATTENTION = [
    ATTENTION_CONSTANT,
    ("ATTENTION_HEAD_DIM = 64\n", "ATTENTION_HEAD_DIM = 64\nWINDOW_SIZE = 8\n"),
    (
        UNET_ANCHOR,
        f'''class WindowAttention(nn.Module):
    """
    Multi-head self-attention inside non-overlapping WINDOW_SIZE x WINDOW_SIZE windows (linear in
    image area). The output projection is zero initialized, so the module starts as the identity.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = make_group_norm(channels)
        self.query_key_value = nn.Conv2d(channels, 3 * channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C, H, W], H and W divisible by WINDOW_SIZE
        returns:
            features: [B, C, H, W]
        """
        batch_size, channels, height, width = features.shape
        heads = channels // ATTENTION_HEAD_DIM
        rows, columns = height // WINDOW_SIZE, width // WINDOW_SIZE
        hidden = self.query_key_value(self.norm(features)).reshape(
            batch_size, 3, heads, ATTENTION_HEAD_DIM, rows, WINDOW_SIZE, columns, WINDOW_SIZE
        )
        hidden = hidden.permute(1, 0, 2, 4, 6, 5, 7, 3).reshape(
            3, batch_size, heads, rows * columns, WINDOW_SIZE * WINDOW_SIZE, ATTENTION_HEAD_DIM
        )  # [3, B, heads, windows, window tokens, D]
        attended = functional.scaled_dot_product_attention(hidden[0], hidden[1], hidden[2])
        attended = attended.reshape(
            batch_size, heads, rows, columns, WINDOW_SIZE, WINDOW_SIZE, ATTENTION_HEAD_DIM
        ).permute(0, 1, 6, 2, 4, 3, 5)  # [B, heads, D, rows, window, columns, window]
        return features + self.output(attended.reshape(batch_size, channels, height, width))


{UNET_ANCHOR}''',
    ),
] + context_at_levels("WindowAttention")

LINEAR_ATTENTION = [
    ATTENTION_CONSTANT,
    (
        UNET_ANCHOR,
        f'''class LinearAttention(nn.Module):
    """
    Global multi-head linear attention (efficient attention: softmax of queries over channels, of
    keys over positions), cost linear in image area. The output projection is zero initialized.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = make_group_norm(channels)
        self.query_key_value = nn.Conv2d(channels, 3 * channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C, H, W]
        returns:
            features: [B, C, H, W]
        """
        batch_size, channels, height, width = features.shape
        heads = channels // ATTENTION_HEAD_DIM
        query, key, value = self.query_key_value(self.norm(features)).reshape(
            batch_size, 3, heads, ATTENTION_HEAD_DIM, height * width
        ).unbind(1)  # [B, heads, D, HW]
        query = query.float().softmax(dim=2)
        key = key.float().softmax(dim=3)
        context = torch.einsum("bhdn,bhen->bhde", key, value.float())  # [B, heads, D, D]
        attended = torch.einsum("bhde,bhdn->bhen", context, query)  # [B, heads, D, HW]
        attended = attended.reshape(batch_size, channels, height, width).to(features.dtype)
        return features + self.output(attended)


{UNET_ANCHOR}''',
    ),
] + context_at_levels("LinearAttention")

CONTROL_STREAM_CLASSES = f'''class ControlTransformerLayer(nn.Module):
    """
    Pre-norm transformer layer (self-attention and MLP) on the control tokens.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(CONTROL_DIM)
        self.query_key_value = nn.Linear(CONTROL_DIM, 3 * CONTROL_DIM)
        self.attention_output = nn.Linear(CONTROL_DIM, CONTROL_DIM)
        self.mlp_norm = nn.LayerNorm(CONTROL_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(CONTROL_DIM, 4 * CONTROL_DIM),
            nn.GELU(),
            nn.Linear(4 * CONTROL_DIM, CONTROL_DIM),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        args:
            tokens: [B, T, CONTROL_DIM]
        returns:
            tokens: [B, T, CONTROL_DIM]
        """
        batch_size, num_tokens, _ = tokens.shape
        heads = CONTROL_DIM // ATTENTION_HEAD_DIM
        query, key, value = (
            self.query_key_value(self.attention_norm(tokens))
            .reshape(batch_size, num_tokens, 3, heads, ATTENTION_HEAD_DIM)
            .permute(2, 0, 3, 1, 4)
        )  # [B, heads, T, D] each
        attended = functional.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch_size, num_tokens, CONTROL_DIM)
        tokens = tokens + self.attention_output(attended)
        return tokens + self.mlp(self.mlp_norm(tokens))


class ControlStream(nn.Module):
    """
    A fixed set of learned control tokens gathers information from the image features with
    cross-attention, mixes it globally with self-attention, and writes it back to the image
    features with cross-attention. Cost is linear in image area. The write projection is zero
    initialized, so the module starts as the identity.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.tokens = nn.Parameter(0.02 * torch.randn(CONTROL_TOKENS, CONTROL_DIM))
        self.condition_projection = nn.Linear(CONDITION_DIM, CONTROL_DIM)
        self.read_token_norm = nn.LayerNorm(CONTROL_DIM)
        self.read_query = nn.Linear(CONTROL_DIM, CONTROL_DIM)
        self.read_image_norm = make_group_norm(channels)
        self.read_key_value = nn.Conv2d(channels, 2 * CONTROL_DIM, 1)
        self.read_output = nn.Linear(CONTROL_DIM, CONTROL_DIM)
        self.layers = nn.ModuleList(ControlTransformerLayer() for _ in range(CONTROL_LAYERS))
        self.write_image_norm = make_group_norm(channels)
        self.write_query = nn.Conv2d(channels, CONTROL_DIM, 1)
        self.write_token_norm = nn.LayerNorm(CONTROL_DIM)
        self.write_key_value = nn.Linear(CONTROL_DIM, 2 * CONTROL_DIM)
        self.write_output = nn.Conv2d(CONTROL_DIM, channels, 1)
        nn.init.zeros_(self.write_output.weight)
        nn.init.zeros_(self.write_output.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        args:
            features: [B, C, H, W]
            condition: [B, CONDITION_DIM]
        returns:
            features: [B, C, H, W]
        """
        batch_size, _, height, width = features.shape
        heads = CONTROL_DIM // ATTENTION_HEAD_DIM
        tokens = self.tokens[None] + self.condition_projection(condition)[:, None]  # [B, T, C_t]

        read_query = self.read_query(self.read_token_norm(tokens)).reshape(
            batch_size, CONTROL_TOKENS, heads, ATTENTION_HEAD_DIM
        ).transpose(1, 2)  # [B, heads, T, D]
        read_key, read_value = self.read_key_value(self.read_image_norm(features)).reshape(
            batch_size, 2, heads, ATTENTION_HEAD_DIM, height * width
        ).transpose(3, 4).unbind(1)  # [B, heads, HW, D]
        gathered = functional.scaled_dot_product_attention(read_query, read_key, read_value)
        tokens = tokens + self.read_output(
            gathered.transpose(1, 2).reshape(batch_size, CONTROL_TOKENS, CONTROL_DIM)
        )
        for layer in self.layers:
            tokens = layer(tokens)

        write_query = self.write_query(self.write_image_norm(features)).reshape(
            batch_size, heads, ATTENTION_HEAD_DIM, height * width
        ).transpose(2, 3)  # [B, heads, HW, D]
        write_key, write_value = (
            self.write_key_value(self.write_token_norm(tokens))
            .reshape(batch_size, CONTROL_TOKENS, 2, heads, ATTENTION_HEAD_DIM)
            .permute(2, 0, 3, 1, 4)
        )  # [B, heads, T, D]
        written = functional.scaled_dot_product_attention(write_query, write_key, write_value)
        written = written.transpose(2, 3).reshape(batch_size, CONTROL_DIM, height, width)
        return features + self.write_output(written)


{UNET_ANCHOR}'''

CONTROL_STREAM = [
    ATTENTION_CONSTANT,
    (
        "ATTENTION_HEAD_DIM = 64\n",
        "ATTENTION_HEAD_DIM = 64\nCONTROL_TOKENS = 64\nCONTROL_DIM = 512\nCONTROL_LAYERS = 2\n",
    ),
    (UNET_ANCHOR, CONTROL_STREAM_CLASSES),
    (
        """        self.output_norm = make_group_norm(LEVEL_CHANNELS[0])""",
        """        self.control_stream = ControlStream(LEVEL_CHANNELS[-1])
        self.output_norm = make_group_norm(LEVEL_CHANNELS[0])""",
    ),
    (
        "        features = self.middle_blocks(features, condition)\n",
        "        features = self.control_stream(self.middle_blocks(features, condition), condition)\n",
    ),
]

MULTI_SCALE_CONTROL_STREAM = CONTROL_STREAM + [
    ("CONTROL_TOKENS = 64\n", "CONTROL_TOKENS = 128\n"),
    (
        """        self.control_stream = ControlStream(LEVEL_CHANNELS[-1])""",
        """        self.control_stream = ControlStream(LEVEL_CHANNELS[-1])
        self.level_1_control_stream = ControlStream(LEVEL_CHANNELS[1])""",
    ),
    (
        "            features = self.decoder_stages[level](torch.cat([features, skip], dim=1), condition)\n",
        """            features = self.decoder_stages[level](torch.cat([features, skip], dim=1), condition)
            if level == 1:
                features = self.level_1_control_stream(features, condition)
""",
    ),
]

RUNS = {
    "0048": ([], "control."),
    "0049": (representation_alignment("level_1"), "+ REPA (weight 0.5) on the level 1 encoder output."),
    "0050": (representation_alignment("middle"), "+ REPA (weight 0.5) on the middle block output."),
    "0051": (WINDOW_ATTENTION, "+ 8x8 window self-attention after the level 1, 2 and middle stages."),
    "0052": (LINEAR_ATTENTION, "+ global linear attention after the level 1, 2 and middle stages."),
    "0053": (CONTROL_STREAM, "+ control stream (64 tokens, 2 layers) at the middle block."),
    "0054": (
        CONTROL_STREAM + representation_alignment("level_1"),
        "+ control stream at the middle block + REPA on level 1.",
    ),
    "0055": (
        MULTI_SCALE_CONTROL_STREAM,
        "+ control streams (128 tokens) at the middle block and the level 1 decoder.",
    ),
}


def main() -> None:
    base_text = (UPSCALERS_DIRECTORY / f"run_{BASE_RUN}.py").read_text()
    for run_id, (replacements, docstring_suffix) in RUNS.items():
        path = UPSCALERS_DIRECTORY / f"run_{run_id}.py"
        if path.exists():
            continue
        text = base_text.replace(f"run_{BASE_RUN}", f"run_{run_id}").replace(
            f"Run{BASE_RUN}Upscaler", f"Run{run_id}Upscaler"
        )
        docstring = f"{DOCSTRING_PREFIX}\n    {docstring_suffix}"
        for old, new in STEP_MATCHED + replacements + [(BASE_DOCSTRING, docstring)]:
            if text.count(old) != 1:
                raise RuntimeError(f"run_{run_id}: expected one occurrence of {old[:70]!r}")
            text = text.replace(old, new)
        long_lines = [line for line in text.split("\n") if len(line) > 100]
        if long_lines:
            raise RuntimeError(f"run_{run_id}: long lines {long_lines}")
        path.write_text(text)
        print("wrote", path.name)


if __name__ == "__main__":
    main()
