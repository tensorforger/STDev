import gc
import json
import sys
import time

import torch
from diffusers.models import AutoencoderKLFlux2

VAE_PATH = "models/FLUX.2-klein-4B/vae"
RESOLUTIONS = [256, 512, 1024, 2048]
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
MAX_BATCH_SIZE = 1024
WARMUP_ITERATIONS = 3
MIN_BENCHMARK_SECONDS = 3.0
OUTPUT_PATH = "dev/data/flux2_vae_speed.json"


def synchronized_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def measure_images_per_second(function, batch_size: int) -> float:
    for _ in range(WARMUP_ITERATIONS):
        function()
    iterations = 0
    start = synchronized_time()
    while True:
        function()
        iterations += 1
        elapsed = synchronized_time() - start
        if elapsed >= MIN_BENCHMARK_SECONDS and iterations >= 3:
            break
    return iterations * batch_size / elapsed


def benchmark_configuration(
    model: AutoencoderKLFlux2, resolution: int, dtype: torch.dtype, compiled: bool
) -> list[dict]:
    if compiled:
        encode_function = torch.compile(lambda x: model.encode(x).latent_dist.mean)
        decode_function = torch.compile(lambda z: model.decode(z, return_dict=False)[0])
    else:
        encode_function = lambda x: model.encode(x).latent_dist.mean
        decode_function = lambda z: model.decode(z, return_dict=False)[0]

    results = []
    batch_size = 1
    while batch_size <= MAX_BATCH_SIZE:
        entry = {"batch_size": batch_size}
        try:
            with torch.no_grad():
                image = torch.randn(batch_size, 3, resolution, resolution, device="cuda", dtype=dtype)
                entry["encode_ips"] = measure_images_per_second(
                    lambda: encode_function(image), batch_size
                )
                latents = encode_function(image)
                del image
                entry["decode_ips"] = measure_images_per_second(
                    lambda: decode_function(latents), batch_size
                )
                del latents
            entry["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            results.append(entry)
            print(resolution, dtype, compiled, entry, flush=True)
        except (torch.OutOfMemoryError, RuntimeError) as error:
            print(resolution, dtype, compiled, batch_size, "FAILED", str(error)[:120], flush=True)
            break
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        batch_size *= 2
    return results


def main() -> None:
    torch.backends.cudnn.benchmark = True
    all_results = []
    for dtype_name, dtype in DTYPES.items():
        model = AutoencoderKLFlux2.from_pretrained(VAE_PATH).to("cuda", dtype).eval()
        for resolution in RESOLUTIONS:
            for compiled in [False, True]:
                torch._dynamo.reset()
                results = benchmark_configuration(model, resolution, dtype, compiled)
                all_results.append(
                    {
                        "dtype": dtype_name,
                        "resolution": resolution,
                        "compiled": compiled,
                        "results": results,
                    }
                )
                with open(OUTPUT_PATH, "w") as file:
                    json.dump(all_results, file, indent=1)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
