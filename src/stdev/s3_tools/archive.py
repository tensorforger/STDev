import os
import tarfile
import time

import requests
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm

from .progress import ProgressCallback

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB

# Errors that indicate a dropped/broken connection mid-stream, worth retrying with a Range resume.
RETRYABLE_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def create_archive(archive_path, entries):
    """Create a tar archive at `archive_path` from (path_on_disk, arcname) entries."""
    with tarfile.open(archive_path, "w") as tar:
        for path, arcname in entries:
            tar.add(path, arcname=arcname)


def extract_archive(archive_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(dest_dir)


def upload_archive(s3, archive_path, bucket_name, object_key):
    progress = ProgressCallback(archive_path, desc="Uploading")
    transfer_config = TransferConfig(multipart_chunksize=1024 * 1024 * 1024)
    try:
        s3.upload_file(
            archive_path,
            bucket_name,
            object_key,
            Callback=progress,
            Config=transfer_config,
        )
    finally:
        progress.close()


def download_archive(
    s3, bucket_name, object_key, archive_path, max_retries=8, backoff_seconds=5
):
    head = s3.head_object(Bucket=bucket_name, Key=object_key)
    total = head.get("ContentLength") or None

    downloaded = os.path.getsize(archive_path) if os.path.exists(archive_path) else 0

    with tqdm(
        total=total,
        initial=downloaded,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=object_key,
    ) as bar:
        attempt = 0
        while total is None or downloaded < total:
            # Presigned URL is regenerated each attempt since a retry may happen well after ExpiresIn.
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": object_key},
                ExpiresIn=600,
            )
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
            try:
                with requests.get(
                    url, stream=True, headers=headers, timeout=(30, 300)
                ) as r:
                    r.raise_for_status()
                    mode = "ab" if downloaded else "wb"
                    with open(archive_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)
                            bar.update(len(chunk))
                attempt = 0
            except RETRYABLE_ERRORS as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                wait = min(backoff_seconds * attempt, 60)
                print(
                    f"Download interrupted ({e}); retrying {attempt}/{max_retries} "
                    f"from byte {downloaded} after {wait}s..."
                )
                time.sleep(wait)

    if total is not None and downloaded != total:
        raise IOError(
            f"Downloaded size ({downloaded}) does not match expected size ({total}) "
            f"for '{object_key}'"
        )
