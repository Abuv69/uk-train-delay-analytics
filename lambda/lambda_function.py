# lambda/lambda_function.py
# AWS Lambda function — automatically triggered when a new ORR file
# is uploaded to s3://uk-train-delay-abu/raw/
#
# What it does:
#   1. Receives the S3 event (which file was uploaded and where)
#   2. Streams the raw file from S3 into memory (no disk I/O)
#   3. Runs the appropriate cleaning logic
#   4. Writes the cleaned CSV back to s3://.../processed/
#   5. Logs everything to CloudWatch automatically
#
# Deploy with: bash lambda/deploy_lambda.sh

import json
import logging
import io
import re
import urllib.parse
import boto3
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

# ── Shared helpers (inline — Lambda has no local imports) ──────────────────────

def to_num(val):
    if val is None:
        return float("nan")
    s = str(val).strip()
    if s in ("[z]", "[u]", "[p]", "", "nan"):
        return float("nan")
    s = re.sub(r"\s*\[p\]$", "", s)
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _tolerate_ods_errors():
    """ORR .ods files contain error cells that pandas rejects; treat them as missing."""
    try:
        from pandas.io.excel import _odfreader as _odf
        if getattr(_odf.ODFReader, "_patched", False):
            return
        orig = _odf.ODFReader._get_cell_value
        def safe(self, cell):
            try:
                return orig(self, cell)
            except ValueError:
                return float("nan")
        _odf.ODFReader._get_cell_value = safe
        _odf.ODFReader._patched = True
    except Exception:
        pass


def stream_excel(bucket, key, engine="openpyxl", **kwargs):
    """Read an Excel/ODS file from S3 directly into a DataFrame."""
    _tolerate_ods_errors()
    obj = s3.get_object(Bucket=bucket, Key=key)
    buf = io.BytesIO(obj["Body"].read())
    if kwargs.get("sheet_name") in ("3130a Stations",):
        names = pd.ExcelFile(buf, engine=engine).sheet_names
        kwargs["sheet_name"] = next(n for n in names if n.lower().replace("_", " ").startswith("3130a"))
        buf.seek(0)
    return pd.read_excel(buf, engine=engine, **kwargs)


def write_csv_to_s3(df, bucket, key):
    """Write a DataFrame as CSV to S3 without touching disk."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    logger.info(f"Wrote {len(df):,} rows → s3://{bucket}/{key}")


# ── Cleaners (stripped-down versions of scripts/clean_data.py) ────────────────

CAUSE_COLS = [
    "nr_external", "nr_network_mgmt", "nr_non_track_assets",
    "nr_severe_weather", "nr_track",
    "toc_self_fleet", "toc_self_operations", "toc_self_stations",
    "toc_self_other", "toc_self_traincrew",
    "toc_on_toc_fleet", "toc_on_toc_operations", "toc_on_toc_stations",
    "toc_on_toc_other", "toc_on_toc_traincrew",
]


def parse_period_3184(raw_period: str):
    s = str(raw_period).strip()
    m = re.search(r"Period\s+(\d+)", s)
    period_num = int(m.group(1)) if m else None
    m2 = re.match(r"(\w+)\s+(\d{4})", s)
    year_label, start_year = None, None
    if m2:
        start_year = int(m2.group(2))
        year_label = f"{start_year}-{str(start_year + 1)[-2:]}"
    return year_label, period_num, start_year


def clean_3184(bucket, key):
    logger.info("Cleaning Table 3184 ...")
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    if key.lower().endswith(".csv"):
        d = pd.read_csv(io.BytesIO(body), header=0)
        rows = [tuple(d.columns)] + [tuple(r) for r in d.itertuples(index=False)]
    else:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(body), read_only=True)
        rows = list(wb["Sheet 1"].iter_rows(values_only=True))
    records = []
    for i, row in enumerate(rows):
        if i == 0 or not row[0] or not row[1]:
            continue
        year_label, period_num, start_year = parse_period_3184(str(row[0]))
        causes = [to_num(row[j]) for j in range(2, 17)]
        total = sum(c for c in causes if not pd.isna(c))
        nr_total = sum(c for c in causes[:5] if not pd.isna(c))
        toc_total = sum(c for c in causes[5:] if not pd.isna(c))
        records.append({
            "raw_period": str(row[0]).strip(),
            "year_label": year_label,
            "period_num": period_num,
            "start_year": start_year,
            "operator": str(row[1]).strip(),
            **dict(zip(CAUSE_COLS, causes)),
            "total_delay_mins": total,
            "nr_responsible_mins": nr_total,
            "toc_responsible_mins": toc_total,
        })
    df = pd.DataFrame(records)
    df = df[df["operator"] != "train operating company name"]
    df = df.dropna(subset=["year_label", "operator"])
    write_csv_to_s3(df, bucket, "processed/delay_causes.csv")


def clean_3133(bucket, key):
    logger.info("Cleaning Table 3133 ...")
    engine = "odf" if key.endswith(".ods") else "openpyxl"
    df_raw = stream_excel(bucket, key, engine=engine,
                          sheet_name="3133_Punctuality_(quarterly)", header=None)
    header_row = None
    for idx, row in df_raw.iterrows():
        if any(str(c).strip().lower() == "time period" for c in row if c is not None):
            header_row = idx
            break
    if header_row is None:
        raise ValueError("Could not find header in Table 3133")

    df = stream_excel(bucket, key, engine=engine,
                      sheet_name="3133_Punctuality_(quarterly)", header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "time period" in cl:             col_map[c] = "period"
        elif "national or operator" in cl:  col_map[c] = "operator"
        elif "number of recorded" in cl and "moving" not in cl: col_map[c] = "recorded_stops"
        elif "59 seconds" in cl and "moving" not in cl:         col_map[c] = "pct_on_time_1m"
        elif "3 minutes" in cl and "moving" not in cl:          col_map[c] = "pct_on_time_3m"
        elif "15 minutes" in cl and "moving" not in cl:         col_map[c] = "pct_on_time_15m"
    df = df.rename(columns=col_map)
    df = df.dropna(subset=["period", "operator"])
    EXCLUDE = {"Great Britain", "England and Wales", "Scotland", "National or Operator", "nan"}
    df = df[~df["operator"].astype(str).isin(EXCLUDE)]

    def parse_quarter(p):
        p = str(p).strip()
        if p.startswith("Apr"):    return "Q1"
        elif p.startswith("Jul"): return "Q2"
        elif p.startswith("Oct"): return "Q3"
        elif p.startswith("Jan"): return "Q4"
        return None

    def parse_year(p):
        m = re.search(r"(\d{4})", str(p))
        return int(m.group(1)) if m else None

    df["quarter"] = df["period"].apply(parse_quarter)
    df["year"] = df["period"].apply(parse_year)
    df["fin_year"] = df.apply(
        lambda r: f"{r['year']}-{str(r['year']+1)[-2:]}"
        if r["quarter"] in ("Q1","Q2","Q3")
        else f"{r['year']-1}-{str(r['year'])[-2:]}"
        if r["quarter"] == "Q4" else None, axis=1
    )
    # ── same fixes as scripts/clean_data.py (keep the two in sync) ──
    for col in ["pct_on_time_1m", "pct_on_time_3m", "pct_on_time_15m", "recorded_stops"]:
        if col in df.columns:
            df[col] = df[col].apply(to_num)
    df["operator"] = (df["operator"].astype(str)
                      .str.replace(r"\s*\[note\s*\d+\]", "", regex=True).str.strip())
    df["provisional"] = df["period"].astype(str).str.contains(r"\[p\]")
    df["period"] = (df["period"].astype(str)
                    .str.replace(r"\s*\[\w+\]", "", regex=True).str.strip())
    df = df[~df["period"].str.startswith("Time")]
    if "recorded_stops" in df.columns:
        df = df[df["recorded_stops"].fillna(0) > 0]
    df = df.dropna(subset=["quarter", "year"])
    first_month = {"Q1": 4, "Q2": 7, "Q3": 10, "Q4": 1}
    df["qdate"] = pd.to_datetime(dict(year=df["year"].astype(int),
                                      month=df["quarter"].map(first_month), day=1))
    keep = [c for c in ["period", "operator", "recorded_stops", "pct_on_time_1m", "pct_on_time_3m",
                        "pct_on_time_15m", "quarter", "year", "fin_year", "qdate", "provisional"]
            if c in df.columns]
    df = df[keep].drop_duplicates(["operator", "period"], keep="last")
    write_csv_to_s3(df, bucket, "processed/operator_punctuality.csv")


def clean_3130(bucket, key):
    logger.info("Cleaning Table 3130 (large file — may take ~30s) ...")
    eng = "odf" if key.lower().endswith(".ods") else "openpyxl"
    df_raw = stream_excel(bucket, key, engine=eng,
                          sheet_name="3130a Stations", header=None)
    header_row = None
    for idx, row in df_raw.iterrows():
        if any(str(c).strip().lower().startswith("time period") for c in row if c is not None):
            header_row = idx
            break
    if header_row is None:
        raise ValueError("Could not find header in Table 3130")

    df = stream_excel(bucket, key, engine=eng,
                      sheet_name="3130a Stations", header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "time period" in cl:          col_map[c] = "period"
        elif "station name" in cl:       col_map[c] = "station"
        elif "scheduled" in cl:          col_map[c] = "scheduled_stops"
        elif "time to 3" in cl:          col_map[c] = "time_to_3_pct"
        elif "cancellations" in cl:      col_map[c] = "cancellations_pct"
    df = df.rename(columns=col_map)
    df = df.dropna(subset=["period", "station"])
    for col in ["time_to_3_pct", "cancellations_pct", "scheduled_stops"]:
        if col in df.columns:
            df[col] = df[col].apply(to_num)
    df["year_label"] = df["period"].astype(str).str.extract(r"(Apr \d{4} to Mar \d{4})")
    df["period_num"] = df["period"].astype(str).str.extract(r"Period\s+(\d+)").astype(float)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df["station"] = (df["station"].astype(str)
                     .str.replace(r"\s*\[note\s*\d+\]", "", regex=True).str.strip())
    write_csv_to_s3(df, bucket, "processed/station_performance.csv")


# ── Main handler ───────────────────────────────────────────────────────────────

CLEANER_MAP = {
    "3184": clean_3184,
    "3130": clean_3130,
    "3133": clean_3133,
}


def lambda_handler(event, context):
    """
    Triggered by S3 PutObject event on the raw/ prefix.
    Determines which table was uploaded and runs the right cleaner.
    """
    logger.info(f"Event received: {json.dumps(event)}")

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key    = urllib.parse.unquote_plus(record["s3"]["object"]["key"])

        logger.info(f"New file: s3://{bucket}/{key}")

        if not key.startswith("raw/"):
            logger.info("Not in raw/ prefix — ignoring.")
            continue

        filename = key.split("/")[-1]

        # Match to the right cleaner
        matched = False
        for table_id, cleaner_fn in CLEANER_MAP.items():
            if table_id in filename:
                logger.info(f"Matched Table {table_id} → running cleaner")
                cleaner_fn(bucket, key)
                matched = True
                break

        if not matched:
            logger.warning(f"No cleaner matched for file: {filename}")

    return {
        "statusCode": 200,
        "body": json.dumps("Pipeline complete"),
    }
