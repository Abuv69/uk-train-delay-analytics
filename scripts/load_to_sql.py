# scripts/load_to_sql.py
# Loads the cleaned CSVs into the database chosen by DATABASE_URL
# (PostgreSQL), or into database/delays.db (SQLite) when it is not set.
#
#   python scripts/load_to_sql.py                 # checks the data first, then loads
#   python scripts/load_to_sql.py --skip-checks   # load even if a check fails (not recommended)
#
# The whole load is ONE transaction: if anything fails, the old tables stay
# exactly as they were, so the app never sees a half-loaded database.

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db
import quality_checks as qc

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DB_DIR = ROOT / "database"

TABLE_MAP = {
    "delay_causes.csv":          "delay_causes",
    "station_performance.csv":   "station_performance",
    "operator_punctuality.csv":  "operator_punctuality",
}

VIEWS = {
    "v_operator_delay_summary": """
        SELECT operator, year_label,
               SUM(total_delay_mins)     AS total_delay_mins,
               SUM(nr_responsible_mins)  AS nr_delay_mins,
               SUM(toc_responsible_mins) AS toc_delay_mins,
               COUNT(*)                  AS periods_counted
        FROM delay_causes
        WHERE operator IS NOT NULL
        GROUP BY operator, year_label""",
    "v_annual_punctuality": """
        SELECT operator, fin_year, quarter, period, year,
               pct_on_time_3m, pct_on_time_15m, recorded_stops
        FROM operator_punctuality
        WHERE pct_on_time_3m IS NOT NULL""",
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_punct_op_q ON operator_punctuality (operator, qdate)",
    "CREATE INDEX IF NOT EXISTS ix_delay_op_year ON delay_causes (operator, year_label)",
    "CREATE INDEX IF NOT EXISTS ix_station_name ON station_performance (station)",
]


def load(skip_checks: bool = False) -> int:
    if not PROCESSED.exists() or not any(PROCESSED.glob("*.csv")):
        print("[ERROR] No processed files found. Run scripts/clean_data.py first.")
        return 1

    if not skip_checks:
        results = qc.run_all()
        summary = qc.write_report(results)
        if summary["errors"]:
            qc.print_report([r for r in results if not r.passed])
            print(f"\n[BLOCKED] {summary['errors']} data-quality check(s) failed. "
                  f"Nothing was loaded. See {qc.REPORT.relative_to(ROOT)}.")
            return 2
        print(f"Data-quality checks passed ({summary['checks']} checks, {summary['warnings']} warnings).")

    print(f"Target: {db.describe()}")
    engine = db.get_engine()
    chunk = 2000 if db.is_postgres() else None
    method = "multi" if db.is_postgres() else None

    with engine.begin() as conn:                      # one transaction
        for view in VIEWS:                            # Postgres will not drop a table that a view uses
            conn.execute(text(f"DROP VIEW IF EXISTS {view}"))

        for filename, table in TABLE_MAP.items():
            path = PROCESSED / filename
            if not path.exists():
                print(f"[SKIP] {filename} not found, skipping table '{table}'")
                continue
            print(f"Loading {filename} -> table '{table}' ...")
            df = pd.read_csv(path, low_memory=False)
            if "provisional" in df.columns:
                df["provisional"] = df["provisional"].astype(str).str.lower().isin(["true", "1"]).astype(int)
            df.to_sql(table, conn, if_exists="replace", index=False, chunksize=chunk, method=method)
            print(f"  ok {len(df):,} rows")

        for name, sql in VIEWS.items():
            conn.execute(text(f"CREATE VIEW {name} AS {sql}"))
        for stmt in INDEXES:
            conn.execute(text(stmt))

        info = {"built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "database": "postgresql" if db.is_postgres() else "sqlite",
                "latest_punctuality_quarter_start": str(conn.execute(
                    text("SELECT MAX(qdate) FROM operator_punctuality")).scalar())[:10],
                "latest_delay_financial_year": conn.execute(
                    text("SELECT MAX(year_label) FROM delay_causes")).scalar()}
        log_path = PROCESSED / "cleaning_log.json"
        info["cleaning"] = json.loads(log_path.read_text()) if log_path.exists() else {}
        qpath = qc.REPORT
        if qpath.exists():
            q = json.loads(qpath.read_text())
            info["quality"] = {k: q[k] for k in ("checks", "errors", "warnings")}
        # keep the latest build info inside the database too, so an app that only
        # has a DATABASE_URL (no repo files) can still show "data last updated"
        conn.execute(text("DROP TABLE IF EXISTS build_info"))
        conn.execute(text("CREATE TABLE build_info (info TEXT)"))
        conn.execute(text("INSERT INTO build_info (info) VALUES (:j)"), {"j": json.dumps(info)})

    DB_DIR.mkdir(parents=True, exist_ok=True)
    (DB_DIR / "build_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nDone. Data loaded into {db.describe()}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-checks", action="store_true")
    sys.exit(load(ap.parse_args().skip_checks))
