import argparse
import os

from stdev.s3_tools import download_archive, extract_archive, get_s3_client

BUCKET_NAME = "stored-models"
STORED_MODELS_DIR = "stored_models"


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract a stored_models subdirectory from S3."
    )
    parser.add_argument("subdir", help="Name of the subdirectory under stored_models/ to download")
    args = parser.parse_args()

    object_key = f"{args.subdir}.tar"
    archive_path = f"{args.subdir}.tar"

    s3 = get_s3_client()

    try:
        download_archive(s3, BUCKET_NAME, object_key, archive_path)

        print(f"Extracting '{archive_path}' -> '{STORED_MODELS_DIR}/' ...")
        extract_archive(archive_path, STORED_MODELS_DIR)

        print(f"Done. Models extracted to '{STORED_MODELS_DIR}/{args.subdir}'")
    finally:
        if os.path.exists(archive_path):
            os.remove(archive_path)


if __name__ == "__main__":
    main()
