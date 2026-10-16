# scripts/clean_data.py
# Cleans all three ORR data files and saves processed CSVs to data/processed/
# Run from the project root: python scripts/clean_data.py

from pathlib import Path
import pandas as pd
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

LOG = {}   # what cleaning set aside or changed (shown on the dashboard's Data and method page)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def to_num(val):
    """Convert a cell value to float, returning NaN for non-numeric markers."""
    if val is None:
        return float("nan")
    s = str(val).strip()
    if s in ("[z]", "[u]", "[p]", "", "nan"):
        return float("nan")
    # strip trailing [p] provisional markers
    s = re.sub(r"\s*\[p\]$", "", s)
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_period_3184(raw_period: str):
    """
    '   Apr 2019 to Mar 2020 (Period 01)'  ->  year_label='2019-20', period_num=1
    Returns (year_label, period_num, period_start_year)
    """
    s = str(raw_period).strip()
    # extract period number
    m = re.search(r"Period\s+(\d+)", s)
    period_num = int(m.group(1)) if m else None
    # extract start month/year
    m2 = re.match(r"(\w+)\s+(\d{4})", s)
    year_label = None
    period_start_year = None
    if m2:
        period_start_year = int(m2.group(2))
        end_year = period_start_year + 1
        year_label = f"{period_start_year}-{str(end_year)[-2:]}"
    return year_label, period_num, period_start_year


CAUSE_COLS = [
    "nr_external", "nr_network_mgmt", "nr_non_track_assets",
    "nr_severe_weather", "nr_track",
    "toc_self_fleet", "toc_self_operations", "toc_self_stations",
    "toc_self_other", "toc_self_traincrew",
    "toc_on_toc_fleet", "toc_on_toc_operations", "toc_on_toc_stations",
    "toc_on_toc_other", "toc_on_toc_traincrew",
]

# ──────────────────────────────────────────────
# TABLE 3184 — Delay minutes by operator & cause
# ──────────────────────────────────────────────

def clean_3184():
    candidates = list(RAW.glob("*3184*")) + list(RAW.glob("*Delay_minutes*")) + list(RAW.glob("*delay_minutes*"))
    if not candidates:
        print("[SKIP] Table 3184 file not found in data/raw/ — skipping.")
        return
    path = candidates[0]
    print(f"[3184] Reading {path.name} ...")

    wb_rows = []
    if path.suffix == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True)
        ws = wb["Sheet 1"]
        for row in ws.iter_rows(values_only=True):
            wb_rows.append(row)
        wb.close()
    else:
        df_raw = pd.read_excel(path, engine="odf", header=0)
        wb_rows = [tuple(df_raw.columns)] + [tuple(r) for r in df_raw.itertuples(index=False)]

    header = wb_rows[0]
    records = []
    for row in wb_rows[1:]:
        if not row[0] or not row[1]:
            continue
        year_label, period_num, start_year = parse_period_3184(str(row[0]))
        causes = [to_num(row[i]) for i in range(2, 17)]
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
    out = OUT / "delay_causes.csv"
    df.to_csv(out, index=False)
    print(f"[3184] Saved {len(df):,} rows → {out.relative_to(ROOT)}")


# ──────────────────────────────────────────────
# TABLE 3130 — Station performance
# ──────────────────────────────────────────────

def clean_3130():
    candidates = list(RAW.glob("*3130*")) + list(RAW.glob("*station*"))
    if not candidates:
        print("[SKIP] Table 3130 file not found in data/raw/ — skipping.")
        return
    path = candidates[0]
    print(f"[3130] Reading {path.name} (this is large, may take ~30s) ...")

    sheet = "3130a Stations"
    if path.suffix == ".xlsx":
        df_raw = pd.read_excel(path, sheet_name=sheet, header=None)
    else:
        df_raw = pd.read_excel(path, engine="odf", sheet_name=sheet, header=None)

    # Find the actual header row (contains 'Time period')
    header_row = None
    for idx, row in df_raw.iterrows():
        if any(str(c).strip().lower().startswith("time period") for c in row if c is not None):
            header_row = idx
            break

    if header_row is None:
        print("[3130] Could not find header row — skipping.")
        return

    df = pd.read_excel(path, sheet_name=sheet, header=header_row, engine="openpyxl" if path.suffix == ".xlsx" else "odf")
    df.columns = [str(c).strip() for c in df.columns]

    # Rename columns
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "time period" in cl:
            col_map[c] = "period"
        elif "station name" in cl:
            col_map[c] = "station"
        elif "scheduled" in cl:
            col_map[c] = "scheduled_stops"
        elif "recorded station stops" in cl and "percentage" in cl:
            col_map[c] = "recorded_pct"
        elif "recorded station stops" in cl:
            col_map[c] = "recorded_stops"
        elif "time to 3" in cl or "time to3" in cl:
            col_map[c] = "time_to_3_pct"
        elif "cancellations" in cl:
            col_map[c] = "cancellations_pct"
    df = df.rename(columns=col_map)

    needed = ["period", "station"]
    df = df.dropna(subset=[c for c in needed if c in df.columns])
    df = df[df["period"].astype(str).str.strip() != ""]

    for col in ["time_to_3_pct", "cancellations_pct", "scheduled_stops"]:
        if col in df.columns:
            df[col] = df[col].apply(to_num)

    # Extract financial year and period number
    df["year_label"] = df["period"].astype(str).str.extract(r"(Apr \d{4} to Mar \d{4})")
    df["period_num"] = df["period"].astype(str).str.extract(r"Period\s+(\d+)").astype(float)

    LOG["station_rows_with_no_3min_figure"] = int(df["time_to_3_pct"].isna().sum()) if "time_to_3_pct" in df.columns else None
    LOG["station_rows_total"] = int(len(df))
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]   # empty junk columns
    df["station"] = (df["station"].astype(str)
                     .str.replace(r"\s*\[note\s*\d+\]", "", regex=True).str.strip())

    out = OUT / "station_performance.csv"
    df.to_csv(out, index=False)
    print(f"[3130] Saved {len(df):,} rows → {out.relative_to(ROOT)}")


# ──────────────────────────────────────────────
# TABLE 3133 — Operator punctuality (quarterly)
# ──────────────────────────────────────────────

def clean_3133():
    candidates = list(RAW.glob("*3133*")) + list(RAW.glob("*punctuality*"))
    if not candidates:
        print("[SKIP] Table 3133 file not found in data/raw/ — skipping.")
        return
    path = candidates[0]
    print(f"[3133] Reading {path.name} ...")

    engine = "odf" if path.suffix == ".ods" else "openpyxl"
    df_raw = pd.read_excel(path, engine=engine, sheet_name="3133_Punctuality_(quarterly)", header=None)

    # Find actual data header (row with 'Time period')
    header_row = None
    for idx, row in df_raw.iterrows():
        if any(str(c).strip().lower() == "time period" for c in row if c is not None):
            header_row = idx
            break

    if header_row is None:
        print("[3133] Could not find header row — skipping.")
        return

    df = pd.read_excel(path, engine=engine, sheet_name="3133_Punctuality_(quarterly)", header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    # Rename to clean names
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "time period" in cl:
            col_map[c] = "period"
        elif "national or operator" in cl:
            col_map[c] = "operator"
        elif "number of recorded" in cl and "moving" not in cl:
            col_map[c] = "recorded_stops"
        elif "59 seconds" in cl and "moving" not in cl:
            col_map[c] = "pct_on_time_1m"
        elif "3 minutes" in cl and "moving" not in cl:
            col_map[c] = "pct_on_time_3m"
        elif "15 minutes" in cl and "moving" not in cl:
            col_map[c] = "pct_on_time_15m"
        elif "moving annual total" in cl:
            col_map[c] = "mat_stops"
        elif "moving annual average" in cl and "59" in cl:
            col_map[c] = "mat_pct_1m"
        elif "moving annual average" in cl and "3 minutes" in cl:
            col_map[c] = "mat_pct_3m"
        elif "moving annual average" in cl and "15 minutes" in cl:
            col_map[c] = "mat_pct_15m"
    df = df.rename(columns=col_map)

    df = df.dropna(subset=["period", "operator"])
    df = df[df["period"].astype(str).str.strip() != ""]
    df = df[~df["period"].astype(str).str.startswith("Time")]

    # Filter out national/regional aggregates — keep individual operators
    EXCLUDE = {"Great Britain", "England and Wales", "Scotland",
               "National or Operator", "nan"}
    df = df[~df["operator"].astype(str).isin(EXCLUDE)]

    for col in ["pct_on_time_1m", "pct_on_time_3m", "pct_on_time_15m", "recorded_stops"]:
        if col in df.columns:
            df[col] = df[col].apply(to_num)

    # Parse quarter info
    def parse_quarter(p):
        p = str(p).strip()
        if p.startswith("Apr"):    return "Q1"
        elif p.startswith("Jul"):  return "Q2"
        elif p.startswith("Oct"):  return "Q3"
        elif p.startswith("Jan"):  return "Q4"
        return None

    def parse_year(p):
        m = re.search(r"(\d{4})", str(p))
        return int(m.group(1)) if m else None

    df["quarter"] = df["period"].apply(parse_quarter)
    df["year"] = df["period"].apply(parse_year)
    # Financial year label (Q1 Apr-Jun starts the financial year)
    df["fin_year"] = df.apply(
        lambda r: f"{r['year']}-{str(r['year']+1)[-2:]}" if r["quarter"] in ("Q1","Q2","Q3") else
                  f"{r['year']-1}-{str(r['year'])[-2:]}" if r["quarter"] == "Q4" else None, axis=1
    )

    # ── Fixes (found while reviewing the live dashboard) ──────────────────
    # 1. strip ORR footnote markers: "Lumo [note 3]" -> "Lumo"
    LOG["punctuality_operator_names_with_footnote_tags_cleaned"] = int(
        df["operator"].astype(str).str.contains(r"\[note", regex=True).sum())
    df["operator"] = (df["operator"].astype(str)
                      .str.replace(r"\s*\[note\s*\d+\]", "", regex=True).str.strip())
    # 2. provisional flag, then clean the period label ("Oct to Dec 2025 [p]")
    df["provisional"] = df["period"].astype(str).str.contains(r"\[p\]")
    df["period"] = (df["period"].astype(str)
                    .str.replace(r"\s*\[\w+\]", "", regex=True).str.strip())
    # 3. zero recorded stops = no service (e.g. Hull Trains in COVID), not 0% punctuality
    if "recorded_stops" in df.columns:
        zero = df["recorded_stops"].fillna(0) <= 0
        LOG["punctuality_quarters_excluded_no_trains_recorded"] = [
            f"{r.operator} ({r.period})" for r in df[zero].itertuples()]
        df = df[~zero]
    # 4. real quarter start date -> chronological sorting (Jan-Mar belongs to the PREVIOUS
    #    financial year, so sorting by calendar year + Q label mis-orders quarters)
    first_month = {"Q1": 4, "Q2": 7, "Q3": 10, "Q4": 1}
    df = df.dropna(subset=["quarter", "year"])
    df["qdate"] = pd.to_datetime(dict(year=df["year"].astype(int),
                                      month=df["quarter"].map(first_month), day=1))
    keep = [c for c in ["period", "operator", "recorded_stops", "pct_on_time_1m",
                        "pct_on_time_3m", "pct_on_time_15m", "quarter", "year",
                        "fin_year", "qdate", "provisional"] if c in df.columns]
    before = len(df)
    df = df[keep].drop_duplicates(["operator", "period"], keep="last")
    LOG["punctuality_duplicate_rows_removed"] = int(before - len(df))
    LOG["punctuality_rows_with_no_3min_figure"] = int(df["pct_on_time_3m"].isna().sum())

    out = OUT / "operator_punctuality.csv"
    df.to_csv(out, index=False)
    print(f"[3133] Saved {len(df):,} rows → {out.relative_to(ROOT)}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("UK Train Delay Analytics — Data Cleaning")
    print("=" * 55)
    missing = [f for f in [RAW] if not f.exists()]
    if missing:
        print(f"[ERROR] data/raw/ folder not found at {RAW}")
        print("Create it and place the 3 ORR files inside.")
        sys.exit(1)

    clean_3184()
    clean_3130()
    clean_3133()
    import json
    (OUT / "cleaning_log.json").write_text(json.dumps(LOG, indent=2))
    print("\n✓ All done. Cleaned files saved to data/processed/")
