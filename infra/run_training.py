import argparse
import time
from pathlib import Path

from stdev.upscalers.loading import load_upscaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
UPSCALERS_DIRECTORY = REPOSITORY_ROOT / "src" / "stdev" / "upscalers"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains a single run.")
    parser.add_argument("run_id", help="4 digit run ID, e.g. 0001")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_name = f"run_{arguments.run_id}"
    upscaler = load_upscaler(UPSCALERS_DIRECTORY / f"{run_name}.py")
    if upscaler.get_name() != run_name:
        raise RuntimeError(f"get_name() returned {upscaler.get_name()!r}, expected {run_name!r}")

    start_time = time.perf_counter()
    upscaler.train(smoke_mode=arguments.smoke)
    print(f"{run_name}: training finished in {time.perf_counter() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
