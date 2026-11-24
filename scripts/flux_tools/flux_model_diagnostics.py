"""
Internals of FLUX.2 pretrain checkpoints: residual stream growth, adaLN gates / scales, per head attention statistics
(max logit, entropy, peak probability, share of the attention output), QK-norm gains and relative bias, on visual
benchmark images at several noise levels. Blocks are replayed in fp32 with explicit attention (checked against the
model's own attention).

Usage: CUDA_VISIBLE_DEVICES=7 python scripts/flux_tools/flux_model_diagnostics.py DATA_RUN CHECKPOINT [CHECKPOINT ...] [--weights ema]
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as functional

from stdev.flux_pretrain.config import MODEL, run_paths
from stdev.flux_pretrain.model import (
    FluxUpscalerTransformer,
    ModelConfig,
    TransformerBlock,
    WINDOW_SIZE,
    align_source,
    shifted_window_mask,
    to_residual,
    window_merge,
    window_partition,
)

SIGMAS = (0.1, 0.5, 0.9)
NUM_IMAGES = 4


def load_weights(checkpoint_path: Path, weights: str) -> tuple[FluxUpscalerTransformer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ModelConfig(**{**checkpoint["model_config"], "residual_std": tuple(checkpoint["model_config"]["residual_std"])})
    model = FluxUpscalerTransformer(config)
    model.load_state_dict(checkpoint["model"])
    if weights == "ema":
        with torch.no_grad():
            for parameter, ema_parameter in zip(model.parameters(), checkpoint["ema"]):
                parameter.copy_(ema_parameter)
    return model.cuda().float().eval().requires_grad_(False), checkpoint["state"]


def attention_statistics(block: TransformerBlock, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    args:
        tokens: [N, H, W, D], modulated normed tokens
    returns:
        attended: [N, H, W, D] (after the output projection), statistics per head: [heads]
    """
    count, height, width, dim = tokens.shape
    heads, head_dim = block.heads, block.head_dim
    if block.shift:
        tokens = torch.roll(tokens, (-block.shift, -block.shift), dims=(1, 2))
    windows = window_partition(block.qkv(tokens))  # [N, windows, t, 3D]
    num_windows, window_tokens = windows.shape[1], windows.shape[2]
    qkv = windows.reshape(count * num_windows, window_tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    query, key, value = block.query_norm(qkv[0]), block.key_norm(qkv[1]), qkv[2]  # [M, heads, t, Dh]
    logits = query @ key.transpose(-1, -2) / math.sqrt(head_dim)  # [M, heads, t, t]
    bias = block.relative_bias[:, block.relative_index][None]
    if block.shift:
        mask = shifted_window_mask(height, width, block.shift, tokens.device)
        bias = bias.masked_fill(~mask[:, None], float("-inf"))
        bias = bias.expand(count, -1, -1, -1, -1).reshape(count * num_windows, heads, window_tokens, window_tokens)
    content_logit_max = logits.abs().amax(dim=(0, 2, 3))
    logits = logits + bias
    probabilities = logits.softmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)  # [M, heads, t]
    valid_keys = torch.isfinite(logits).sum(-1).float()  # [M, heads, t]
    per_head = probabilities @ value  # [M, heads, t, Dh]
    weight = block.attention_output.weight.view(dim, heads, head_dim)
    attended = block.attention_output.bias.expand(count * num_windows, window_tokens, dim).clone()  # [M, t, D]
    head_norms = torch.zeros(heads, device=tokens.device)
    for head in range(heads):
        head_output = per_head[:, head] @ weight[:, head].T  # [M, t, D]
        attended += head_output
        head_norms[head] = head_output.square().sum(-1).mean().sqrt()
    attended = attended.reshape(count, num_windows, window_tokens, dim)
    attended = window_merge(attended, height, width)
    if block.shift:
        attended = torch.roll(attended, (block.shift, block.shift), dims=(1, 2))
    statistics = {
        "content_logit_max": content_logit_max,
        "entropy_ratio": (entropy / valid_keys.log()).mean(dim=(0, 2)),
        "peak_probability": probabilities.amax(-1).mean(dim=(0, 2)),
        "output_share": head_norms / head_norms.sum(),
    }
    return attended, statistics


@torch.no_grad()
def diagnose(model: FluxUpscalerTransformer, latents_source: torch.Tensor, latents_target: torch.Tensor, hff: torch.Tensor) -> None:
    config = model.config
    residual = to_residual(latents_target, latents_source, config.residual_std)
    generator = torch.Generator(device="cuda").manual_seed(0)
    noise = torch.randn(residual.shape, generator=generator, device="cuda")
    rows = []
    head_rows = {}
    for sigma in SIGMAS:
        sigmas = torch.full((residual.shape[0],), sigma, device="cuda")
        noisy = (1 - sigma) * residual + sigma * noise
        condition = model.conditioning(sigmas, hff, torch.ones_like(hff, dtype=torch.bool))
        stacked = torch.cat([noisy, align_source(latents_source)], dim=1)
        tokens = (model.input_conv(stacked) + model.source_encoder(latents_source)).permute(0, 2, 3, 1)
        stem_rms = tokens.square().mean().sqrt().item()
        for index, block in enumerate(model.blocks):
            modulation = block.modulation(condition)[:, None, None, :].chunk(6, dim=-1)
            shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation
            normed = block.norm(tokens) * (1 + scale_a) + shift_a
            attended, statistics = attention_statistics(block, normed)
            reference = block._attention(normed)
            check = ((attended - reference).norm() / reference.norm()).item()
            after_attention = tokens + gate_a * attended
            feed_forward = block._feed_forward(block.norm(after_attention) * (1 + scale_f) + shift_f)
            output = after_attention + gate_f * feed_forward
            rows.append(
                (
                    sigma,
                    index,
                    tokens.square().mean().sqrt().item(),
                    ((output - tokens).norm() / tokens.norm()).item(),
                    (gate_a * attended).square().mean().sqrt().item(),
                    (gate_f * feed_forward).square().mean().sqrt().item(),
                    gate_a.abs().mean().item(),
                    gate_f.abs().mean().item(),
                    gate_f.abs().max().item(),
                    scale_a.abs().max().item(),
                    scale_f.abs().max().item(),
                    check,
                )
            )
            head_rows.setdefault(index, []).append(statistics)
            tokens = output
        final_rms = tokens.square().mean().sqrt().item()
        print(f"sigma {sigma}: stem rms {stem_rms:.2f}, final stream rms {final_rms:.2f}")
    print("sigma block stream_rms rel_update attn_out_rms ff_out_rms |gate_a| |gate_f| max|gate_f| max|scale_a| max|scale_f| attn_check")
    for row in rows:
        print(f"{row[0]:.1f} {row[1]:5d} {row[2]:10.2f} {row[3]:10.3f} {row[4]:12.3f} {row[5]:10.3f} {row[6]:8.3f} {row[7]:8.3f} {row[8]:10.3f} {row[9]:11.3f} {row[10]:11.3f} {row[11]:.1e}")
    print("per head (averaged over sigmas): block | content logit max (max over heads) | entropy ratio min/mean/max | peak prob max | output share min/max | low-share heads (<0.25 / heads)")
    for index, statistics_list in head_rows.items():
        merged = {key: torch.stack([statistics[key] for statistics in statistics_list]).mean(0) for key in statistics_list[0]}
        share = merged["output_share"]
        heads = share.numel()
        low = (share < 0.25 / heads).sum().item()
        print(
            f"  {index:2d} | {merged['content_logit_max'].max():6.2f} | {merged['entropy_ratio'].min():.3f} / {merged['entropy_ratio'].mean():.3f} / "
            f"{merged['entropy_ratio'].max():.3f} | {merged['peak_probability'].max():.3f} | {share.min():.3f} / {share.max():.3f} | {low}"
        )


def parameter_summary(model: FluxUpscalerTransformer) -> None:
    print("block | qk-norm gain rms (q / k) | relative bias max | qkv / out / ff_in / ff_out weight rms | modulation weight rms")
    for index, block in enumerate(model.blocks):
        rms = lambda tensor: tensor.square().mean().sqrt().item()
        print(
            f"  {index:2d} | {rms(block.query_norm.weight):.3f} / {rms(block.key_norm.weight):.3f} | {block.relative_bias.abs().max():6.2f} | "
            f"{rms(block.qkv.weight):.4f} / {rms(block.attention_output.weight):.4f} / {rms(block.feed_forward_input.weight):.4f} / "
            f"{rms(block.feed_forward_output.weight):.4f} | {rms(block.modulation[1].weight):.4f}"
        )
    total = torch.linalg.vector_norm(torch.stack([parameter.norm() for parameter in model.parameters()])).item()
    print(f"total weight norm {total:.1f}, null_hff norm {model.conditioning.null_hff.norm():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_run")
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    arguments = parser.parse_args()
    tensors = torch.load(run_paths(arguments.data_run).benchmark_set / "latents.pt")
    latents_source = tensors["latents_source"][:NUM_IMAGES].cuda()
    latents_target = tensors["latents_target"][:NUM_IMAGES].cuda()
    hff = tensors["hff"][:NUM_IMAGES].cuda()
    for checkpoint_path in arguments.checkpoints:
        model, state = load_weights(checkpoint_path, arguments.weights)
        assert model.config.depth == MODEL.depth
        print(f"===== {checkpoint_path.name}: step {state['step']}, {state['training_seconds'] / 3600:.2f} h, {arguments.weights}")
        parameter_summary(model)
        diagnose(model, latents_source, latents_target, hff)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
