import argparse
import os

from stdev.s3_tools import create_archive, ensure_bucket, get_s3_client, upload_archive

BUCKET_NAME = "stored-models"
STORED_MODELS_DIR = "stored_models"


def main():
    parser = argparse.ArgumentParser(
        description="Archive and upload a stored_models subdirectory to S3."
    )
    parser.add_argument("subdir", help="Name of the subdirectory under stored_models/ to upload")
    args = parser.parse_args()

    src_dir = os.path.join(STORED_MODELS_DIR, args.subdir)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"No such directory: {src_dir}")

    archive_path = f"{args.subdir}.tar"
    object_key = f"{args.subdir}.tar"

    print(f"Archiving '{src_dir}' -> '{archive_path}' ...")
    create_archive(archive_path, [(src_dir, args.subdir)])

    try:
        s3 = get_s3_client()
        ensure_bucket(s3, BUCKET_NAME)
        upload_archive(s3, archive_path, BUCKET_NAME, object_key)
        print(f"\nUploaded '{archive_path}' to '{BUCKET_NAME}/{object_key}'")
    finally:
        os.remove(archive_path)


if __name__ == "__main__":
    main()
