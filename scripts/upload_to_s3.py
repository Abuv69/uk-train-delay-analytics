# scripts/upload_to_s3.py
# One-time script: uploads your local raw files, processed CSVs, and
# trained model into S3. Run this once to seed the bucket.
# After this, Lambda handles all future uploads automatically.
#
# Run from project root: python scripts/upload_to_s3.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import S3_BUCKET, S3_RAW, S3_PROCESSED, S3_MODELS
from scripts.s3_utils import upload_folder, upload_file


def main():
    print("=" * 55)
    print(f"UK Train Delay Analytics — Seeding S3 bucket")
    print(f"Bucket: {S3_BUCKET}")
    print("=" * 55)

    # 1. Raw ORR files
    raw_dir = ROOT / "data" / "raw"
    if raw_dir.exists() and any(raw_dir.iterdir()):
        print("\n[1/3] Uploading raw ORR files ...")
        upload_folder(raw_dir, S3_RAW)
    else:
        print("[1/3] No raw files found — skipping.")

    # 2. Processed CSVs
    proc_dir = ROOT / "data" / "processed"
    if proc_dir.exists() and any(proc_dir.iterdir()):
        print("\n[2/3] Uploading processed CSVs ...")
        upload_folder(proc_dir, S3_PROCESSED)
    else:
        print("[2/3] No processed files found. Run clean_data.py first.")

    # 3. Trained model
    model_dir = ROOT / "models"
    if model_dir.exists():
        print("\n[3/3] Uploading trained model ...")
        for f in model_dir.iterdir():
            if f.suffix in (".joblib", ".pkl"):
                upload_file(f, f"{S3_MODELS}{f.name}")
    else:
        print("[3/3] No model files found. Run ml_model.py first.")

    print("\n✓ S3 seed complete. Your data is now in the cloud.")
    print(f"  View at: https://s3.console.aws.amazon.com/s3/buckets/{S3_BUCKET}")


if __name__ == "__main__":
    main()
