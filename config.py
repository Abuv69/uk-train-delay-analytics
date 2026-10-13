# config.py
# Central configuration — all AWS settings live here.
# Credentials are read from environment variables or ~/.aws/credentials
# NEVER hardcode keys here. NEVER commit .env to GitHub.

import os
from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
DATA_DIR  = ROOT / "data"
RAW_DIR   = DATA_DIR / "raw"
PROC_DIR  = DATA_DIR / "processed"
DB_DIR    = ROOT / "database"
MODEL_DIR = ROOT / "models"
VIS_DIR   = ROOT / "visuals"

# ── AWS S3 ────────────────────────────────────────────────────────────────────
# Set these as environment variables, or they fall back to defaults below.
# In Lambda, set them as Lambda environment variables (no .env file needed).
S3_BUCKET   = os.environ.get("S3_BUCKET", "uk-train-delay-abu")
S3_RAW      = "raw/"
S3_PROCESSED = "processed/"
S3_MODELS   = "models/"
AWS_REGION  = os.environ.get("AWS_REGION", "eu-west-2")

# ── Pipeline settings ─────────────────────────────────────────────────────────
# Map S3 raw filenames → which cleaner to call
RAW_FILE_MAP = {
    "Table_3184": "delay_causes",
    "table-3130": "station_performance",
    "table-3133": "operator_punctuality",
}
