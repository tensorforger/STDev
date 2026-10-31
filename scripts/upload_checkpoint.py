"""Upload a training checkpoint to an S3-compatible bucket (Cloudflare R2).

Credentials are read from the environment, falling back to the repo's `.env`:
    S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY

Examples:
    # newest checkpoint across all runs in pretrain_runs/
    python scripts/upload_checkpoint.py
    # newest checkpoint of one run
    python scripts/upload_checkpoint.py pretrain_runs/stage1_v1
    # a specific file, custom bucket and key
    python scripts/upload_checkpoint.py pretrain_runs/stage1_v1/checkpoints/checkpoint_000056599.pt \
        --bucket checkpoints --key stage1_v1/final.pt
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "pretrain_runs"
REQUIRED_VARS = ("S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY")


def load_env_file(path: Path) -> None:
    """Minimal KEY=VALUE parser; variables already set in the environment win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


def find_latest_checkpoint(root: Path) -> Path:
    checkpoints = sorted(root.rglob("checkpoints/*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints/*.pt found under {root}")
    return checkpoints[-1]


def default_key(path: Path) -> str:
    # pretrain_runs/<run>/checkpoints/<file>.pt -> <run>/<file>.pt
    if path.parent.name == "checkpoints":
        return f"{path.parent.parent.name}/{path.name}"
    return path.name


class Progress:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.start = time.monotonic()
        self.last_print = 0.0
        self.lock = threading.Lock()

    def __call__(self, nbytes: int) -> None:
        with self.lock:
            self.done += nbytes
            now = time.monotonic()
            if now - self.last_print < 5 and self.done < self.total:
                return
            self.last_print = now
            elapsed = max(now - self.start, 1e-6)
            print(
                f"  {self.done / 2**30:.2f}/{self.total / 2**30:.2f} GiB "
                f"({100 * self.done / self.total:.1f}%) {self.done / elapsed / 2**20:.1f} MiB/s",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=RUNS_DIR,
        help="checkpoint file, or a directory to search for the newest checkpoint (default: pretrain_runs/)",
    )
    parser.add_argument("--bucket", default="checkpoints")
    parser.add_argument("--key", help="object key (default: <run_name>/<checkpoint_file>)")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--force", action="store_true", help="upload even if an object of the same size exists")
    parser.add_argument("--chunk-mb", type=int, default=64, help="multipart chunk size in MiB")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    load_env_file(args.env_file)
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        sys.exit(f"missing credentials: {', '.join(missing)} (set them or add to {args.env_file})")

    path = args.path if args.path.is_file() else find_latest_checkpoint(args.path)
    key = args.key or default_key(path)
    size = path.stat().st_size

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )

    if not args.force:
        try:
            remote_size = s3.head_object(Bucket=args.bucket, Key=key)["ContentLength"]
            if remote_size == size:
                print(f"s3://{args.bucket}/{key} already exists with the same size, skipping (--force to overwrite)")
                return
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise

    print(f"uploading {path} ({size / 2**30:.2f} GiB) -> s3://{args.bucket}/{key}")
    chunk = args.chunk_mb * 2**20
    transfer = TransferConfig(
        multipart_threshold=chunk, multipart_chunksize=chunk, max_concurrency=args.concurrency
    )
    start = time.monotonic()
    s3.upload_file(str(path), args.bucket, key, Config=transfer, Callback=Progress(size))

    remote_size = s3.head_object(Bucket=args.bucket, Key=key)["ContentLength"]
    if remote_size != size:
        sys.exit(f"size mismatch after upload: local {size}, remote {remote_size}")
    print(f"done in {time.monotonic() - start:.0f}s, verified size {remote_size} bytes")


if __name__ == "__main__":
    main()
