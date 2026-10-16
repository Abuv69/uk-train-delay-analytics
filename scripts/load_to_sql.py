# scripts/load_to_sql.py
# Loads all processed CSVs into a single SQLite database (database/delays.db)
# Run from the project root: python scripts/load_to_sql.py

from pathlib import Path
import pandas as pd
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DB_DIR = ROOT / "database"
DB_PATH = DB_DIR / "delays.db"

TABLE_MAP = {
    "delay_causes.csv":          "delay_causes",
    "station_performance.csv":   "station_performance",
    "operator_punctuality.csv":  "operator_punctuality",
}


def load():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    if not PROCESSED.exists() or not any(PROCESSED.iterdir()):
        print("[ERROR] No processed files found. Run scripts/clean_data.py first.")
        sys.exit(1)

    print(f"Connecting to {DB_PATH.relative_to(ROOT)} ...")
    conn = sqlite3.connect(DB_PATH.as_posix())

    for filename, table_name in TABLE_MAP.items():
        csv_path = PROCESSED / filename
        if not csv_path.exists():
            print(f"[SKIP] {filename} not found — skipping table '{table_name}'")
            continue

        print(f"Loading {filename} → table '{table_name}' ...")
        df = pd.read_csv(csv_path, low_memory=False)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"  ✓ {len(df):,} rows inserted")

    # Create helpful views for analysis / Streamlit
    conn.executescript("""
        DROP VIEW IF EXISTS v_operator_delay_summary;
        CREATE VIEW v_operator_delay_summary AS
        SELECT
            operator,
            year_label,
            SUM(total_delay_mins)       AS total_delay_mins,
            SUM(nr_responsible_mins)    AS nr_delay_mins,
            SUM(toc_responsible_mins)   AS toc_delay_mins,
            COUNT(*)                    AS periods_counted
        FROM delay_causes
        WHERE operator IS NOT NULL
        GROUP BY operator, year_label;

        DROP VIEW IF EXISTS v_annual_punctuality;
        CREATE VIEW v_annual_punctuality AS
        SELECT
            operator,
            fin_year,
            quarter,
            period,
            year,
            pct_on_time_3m,
            pct_on_time_15m,
            recorded_stops
        FROM operator_punctuality
        WHERE pct_on_time_3m IS NOT NULL;
    """)

    conn.commit()

    # build info: shown in the app as "data last updated" + what cleaning set aside
    import json
    from datetime import datetime, timezone
    info = {"built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "latest_punctuality_quarter_start": str(pd.read_sql(
                "SELECT MAX(qdate) m FROM operator_punctuality", conn)["m"].iloc[0])[:10],
            "latest_delay_financial_year": pd.read_sql(
                "SELECT MAX(year_label) m FROM delay_causes", conn)["m"].iloc[0]}
    log_path = PROCESSED / "cleaning_log.json"
    info["cleaning"] = json.loads(log_path.read_text()) if log_path.exists() else {}
    (DB_DIR / "build_info.json").write_text(json.dumps(info, indent=2))
    conn.close()
    print(f"\n✓ Database ready at {DB_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    load()
