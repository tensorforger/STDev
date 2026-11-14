import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


def get_s3_client():
    cfg = Config(
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        connect_timeout=30,
        read_timeout=3600,
        retries={
            "max_attempts": 10,
            "mode": "adaptive",
        },
    )

    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        region_name="auto",
        config=cfg,
    )


def ensure_bucket(s3, bucket_name):
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError:
        print(f"Bucket '{bucket_name}' not found, creating it...")
        s3.create_bucket(Bucket=bucket_name)
        print(f"Created bucket: {bucket_name}")
