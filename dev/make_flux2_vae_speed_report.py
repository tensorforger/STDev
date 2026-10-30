import json

import torch

INPUT_PATH = "dev/data/flux2_vae_speed.json"
OUTPUT_PATH = "reports/flux2_vae_speed.md"


def best_entry(configurations: list[dict], dtype: str, resolution: int, key: str) -> tuple:
    best = None
    for configuration in configurations:
        if configuration["dtype"] != dtype or configuration["resolution"] != resolution:
            continue
        for entry in configuration["results"]:
            if best is None or entry[key] > best[0]:
                best = (entry[key], entry["batch_size"], configuration["compiled"])
    return best


def format_cell(best: tuple) -> str:
    throughput, batch_size, compiled = best
    mode = "compiled" if compiled else "eager"
    return f"**{throughput:.1f}** (bs={batch_size}, {mode})"


def main() -> None:
    with open(INPUT_PATH) as file:
        configurations = json.load(file)

    lines = [
        "# FLUX.2 VAE throughput on RTX 5090",
        "",
        f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
        "diffusers AutoencoderKLFlux2 (`models/FLUX.2-klein-4B/vae`), square RGB images, "
        "`torch.no_grad`, `cudnn.benchmark=True`.",
        "",
        "Batch size was doubled from 1 until OOM, each point timed for >= 3 s after warmup, "
        "for eager and `torch.compile` (default mode). "
        "Encode uses the posterior mean. Script: `dev/benchmark_flux2_vae_speed.py`.",
        "",
        "## Optimal throughput (images/s)",
        "",
        "Best over batch sizes and eager/compiled.",
        "",
        "| Resolution | dtype | Encode | Decode |",
        "|---|---|---|---|",
    ]
    for resolution in [256, 512, 1024, 2048]:
        for dtype in ["fp32", "bf16"]:
            encode = best_entry(configurations, dtype, resolution, "encode_ips")
            decode = best_entry(configurations, dtype, resolution, "decode_ips")
            lines.append(
                f"| {resolution} | {dtype} | {format_cell(encode)} | {format_cell(decode)} |"
            )

    lines += ["", "## Full results (images/s, encode / decode)", ""]
    for configuration in configurations:
        mode = "compiled" if configuration["compiled"] else "eager"
        lines.append(f"### {configuration['resolution']} px, {configuration['dtype']}, {mode}")
        lines += ["", "| Batch size | Encode | Decode |", "|---|---|---|"]
        for entry in configuration["results"]:
            lines.append(
                f"| {entry['batch_size']} | {entry['encode_ips']:.1f} | {entry['decode_ips']:.1f} |"
            )
        lines.append("")

    with open(OUTPUT_PATH, "w") as file:
        file.write("\n".join(lines))


if __name__ == "__main__":
    main()
