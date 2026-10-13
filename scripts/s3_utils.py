# scripts/s3_utils.py
# All S3 read/write operations used by the pipeline and Lambda.
# Uses boto3 — install with: pip install boto3

import boto3
import io
import sys
from pathlib import Path

# Allow running from scripts/ or project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import S3_BUCKET, AWS_REGION

s3 = boto3.client("s3", region_name=AWS_REGION)


# ── Upload ─────────────────────────────────────────────────────────────────────

def upload_file(local_path: Path, s3_key: str) -> None:
    """Upload a local file to S3."""
    local_path = Path(local_path)
    print(f"  ↑ Uploading {local_path.name} → s3://{S3_BUCKET}/{s3_key}")
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)


def upload_folder(local_dir: Path, s3_prefix: str) -> None:
    """Upload every file in a local directory to S3 under a prefix."""
    local_dir = Path(local_dir)
    for f in local_dir.iterdir():
        if f.is_file():
            upload_file(f, f"{s3_prefix}{f.name}")


def upload_bytes(data: bytes, s3_key: str, content_type: str = "application/octet-stream") -> None:
    """Upload raw bytes to S3 (useful for DataFrames serialised in memory)."""
    print(f"  ↑ Writing bytes → s3://{S3_BUCKET}/{s3_key}")
    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=data, ContentType=content_type)


# ── Download ───────────────────────────────────────────────────────────────────

def download_file(s3_key: str, local_path: Path) -> None:
    """Download a file from S3 to a local path."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ Downloading s3://{S3_BUCKET}/{s3_key} → {local_path.name}")
    s3.download_file(S3_BUCKET, s3_key, str(local_path))


def download_to_buffer(s3_key: str) -> io.BytesIO:
    """Download a file from S3 directly into memory (no disk needed — ideal for Lambda)."""
    print(f"  ↓ Streaming s3://{S3_BUCKET}/{s3_key} into memory")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    return io.BytesIO(obj["Body"].read())


def download_folder(s3_prefix: str, local_dir: Path) -> None:
    """Download all files under an S3 prefix to a local directory."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            if filename:
                download_file(key, local_dir / filename)


# ── List ───────────────────────────────────────────────────────────────────────

def list_keys(s3_prefix: str) -> list[str]:
    """Return all S3 keys under a prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


# ── CSV helpers ────────────────────────────────────────────────────────────────

def df_to_s3_csv(df, s3_key: str) -> None:
    """Save a pandas DataFrame directly to S3 as CSV (no temp file)."""
    import pandas as pd
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    upload_bytes(buf.getvalue().encode("utf-8"), s3_key, "text/csv")


def df_from_s3_csv(s3_key: str):
    """Read a CSV from S3 directly into a pandas DataFrame."""
    import pandas as pd
    buf = download_to_buffer(s3_key)
    return pd.read_csv(buf, low_memory=False)


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Listing contents of s3://{S3_BUCKET}/")
    all_keys = list_keys("")
    if all_keys:
        for k in all_keys:
            print(f"  {k}")
    else:
        print("  (bucket is empty — upload some files first)")
