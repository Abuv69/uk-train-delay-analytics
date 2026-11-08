"""Data-quality gate. Runs on the cleaned CSVs BEFORE they are loaded into the database.

Each check returns a result with a severity:
  error -> the load is blocked (the data would mislead the app)
  warn  -> the load goes ahead, but the report says what looked odd

Run on its own:  python scripts/quality_checks.py
Exit code 1 if any error-level check fails, so Docker / CI / a scheduler can react to it.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORT = PROCESSED / "quality_report.json"

CAUSE_COLS = [
    "nr_external", "nr_network_mgmt", "nr_non_track_assets", "nr_severe_weather", "nr_track",
    "toc_self_fleet", "toc_self_operations", "toc_self_stations", "toc_self_other", "toc_self_traincrew",
    "toc_on_toc_fleet", "toc_on_toc_operations", "toc_on_toc_stations", "toc_on_toc_other", "toc_on_toc_traincrew",
]

REQUIRED = {
    "operator_punctuality": ["operator", "qdate", "recorded_stops", "pct_on_time_1m", "pct_on_time_3m",
                             "pct_on_time_15m", "quarter", "fin_year", "provisional"],
    "delay_causes": ["operator", "year_label", "period_num", "total_delay_mins",
                     "nr_responsible_mins", "toc_responsible_mins"] + CAUSE_COLS,
    "station_performance": ["period", "station", "scheduled_stops", "recorded_stops",
                            "time_to_3_pct", "cancellations_pct", "year_label", "period_num"],
}
MIN_ROWS = {"operator_punctuality": 500, "delay_causes": 1000, "station_performance": 50000}


@dataclass
class Result:
    table: str
    check: str
    severity: str      # "error" | "warn"
    passed: bool
    detail: str


def _r(table, check, severity, ok, detail_ok, detail_bad):
    return Result(table, check, severity, bool(ok), detail_ok if ok else detail_bad)


def check_table_shape(name: str, df: pd.DataFrame) -> list[Result]:
    out = []
    missing = [c for c in REQUIRED[name] if c not in df.columns]
    out.append(_r(name, "required columns present", "error", not missing,
                  f"all {len(REQUIRED[name])} present", f"missing: {missing}"))
    out.append(_r(name, "enough rows", "error", len(df) >= MIN_ROWS[name],
                  f"{len(df):,} rows", f"only {len(df):,} rows (expected at least {MIN_ROWS[name]:,})"))
    junk = [c for c in df.columns if str(c).lower() in ("nan", "") or str(c).startswith("Unnamed")]
    out.append(_r(name, "no blank-header columns", "error", not junk, "none", f"found: {junk}"))
    dupe_cols = pd.Index(df.columns).duplicated().sum()
    out.append(_r(name, "no duplicate column names", "error", dupe_cols == 0, "none", f"{dupe_cols} duplicated"))
    return out


def check_punctuality(df: pd.DataFrame) -> list[Result]:
    t = "operator_punctuality"
    out = check_table_shape(t, df)
    if out[0].passed is False:
        return out
    n_dupe = int(df.duplicated(["operator", "qdate"]).sum())
    out.append(_r(t, "one row per operator and quarter", "error", n_dupe == 0, "no duplicates", f"{n_dupe} duplicate rows"))
    for col in ("pct_on_time_1m", "pct_on_time_3m", "pct_on_time_15m"):
        bad = int(((df[col] < 0) | (df[col] > 100)).sum())
        out.append(_r(t, f"{col} between 0 and 100", "error", bad == 0, "ok", f"{bad} values outside 0-100"))
    bad = int(((df.pct_on_time_1m > df.pct_on_time_3m + 1e-9) | (df.pct_on_time_3m > df.pct_on_time_15m + 1e-9)).sum())
    out.append(_r(t, "within 1 min <= within 3 min <= within 15 min", "error", bad == 0, "holds for every row", f"{bad} rows break it"))
    bad = int((df.recorded_stops <= 0).sum())
    out.append(_r(t, "every row has recorded stops", "error", bad == 0, "ok", f"{bad} rows with zero stops"))
    nulls = int(df.pct_on_time_3m.isna().sum())
    out.append(_r(t, "no missing on-time figures", "warn", nulls == 0, "ok", f"{nulls} rows have no 3-minute figure"))

    q = pd.to_datetime(df.qdate)
    latest = q.max()
    n_latest = int((q == latest).sum())
    out.append(_r(t, "latest quarter covers most operators", "error", n_latest >= 15,
                  f"{n_latest} operators in {latest:%b %Y}", f"only {n_latest} operators in {latest:%b %Y}"))
    expected = pd.date_range(q.min(), latest, freq="QS-JAN")
    expected = [d for d in expected if d.month in (1, 4, 7, 10)]
    gaps = sorted(set(expected) - set(q.unique()))
    out.append(_r(t, "no missing quarters between first and last", "error", not gaps,
                  f"{len(expected)} consecutive quarters", f"missing: {[f'{g:%b %Y}' for g in gaps]}"))
    prov = df[df.provisional.astype(str).str.lower().isin(["true", "1"])]
    only_latest = prov.empty or pd.to_datetime(prov.qdate).min() >= latest - pd.DateOffset(months=3)
    out.append(_r(t, "provisional flag only on the latest quarters", "warn", only_latest,
                  f"{len(prov)} provisional rows", "provisional rows found in older quarters"))
    return out


def check_delays(df: pd.DataFrame) -> list[Result]:
    t = "delay_causes"
    out = check_table_shape(t, df)
    if out[0].passed is False:
        return out
    n_dupe = int(df.duplicated(["operator", "year_label", "period_num"]).sum())
    out.append(_r(t, "one row per operator and period", "error", n_dupe == 0, "no duplicates", f"{n_dupe} duplicate rows"))
    neg = int((df[CAUSE_COLS] < 0).any(axis=1).sum())
    out.append(_r(t, "no negative delay minutes", "error", neg == 0, "ok", f"{neg} rows with negative minutes"))
    diff = (df[CAUSE_COLS].sum(axis=1) - df.total_delay_mins).abs()
    out.append(_r(t, "total equals the sum of the 15 causes", "error", bool((diff < 0.5).all()),
                  "holds for every row", f"{int((diff >= 0.5).sum())} rows differ"))
    diff = (df.nr_responsible_mins + df.toc_responsible_mins - df.total_delay_mins).abs()
    out.append(_r(t, "Network Rail + train company = total", "error", bool((diff < 0.5).all()),
                  "holds for every row", f"{int((diff >= 0.5).sum())} rows differ"))
    out.append(_r(t, "period number is 1 to 13", "error", bool(df.period_num.between(1, 13).all()), "ok", "out of range"))
    n_ops = df[df.year_label == df.year_label.max()].operator.nunique()
    out.append(_r(t, "latest year has most operators", "warn", n_ops >= 15, f"{n_ops} operators", f"only {n_ops}"))
    return out


def check_stations(df: pd.DataFrame) -> list[Result]:
    t = "station_performance"
    out = check_table_shape(t, df)
    if out[0].passed is False:
        return out
    n_dupe = int(df.duplicated(["period", "station"]).sum())
    out.append(_r(t, "one row per station and period", "error", n_dupe == 0, "no duplicates", f"{n_dupe} duplicate rows"))
    for col in ("time_to_3_pct", "cancellations_pct"):
        bad = int(((df[col] < 0) | (df[col] > 100)).sum())
        out.append(_r(t, f"{col} between 0 and 100", "error", bad == 0, "ok", f"{bad} values outside 0-100"))
    share = df.time_to_3_pct.isna().mean()
    out.append(_r(t, "most rows have an on-time figure", "warn", share < 0.25,
                  f"{share:.0%} blank (ORR withholds figures for low-volume stops)", f"{share:.0%} blank"))
    return out


def run_all(processed: Path = PROCESSED) -> list[Result]:
    files = {"operator_punctuality": check_punctuality, "delay_causes": check_delays,
             "station_performance": check_stations}
    results: list[Result] = []
    for name, fn in files.items():
        path = processed / f"{name}.csv"
        if not path.exists():
            results.append(Result(name, "file exists", "error", False, f"{path.name} not found"))
            continue
        results.extend(fn(pd.read_csv(path, low_memory=False)))
    return results


def write_report(results: list[Result], path: Path = REPORT) -> dict:
    summary = {"errors": sum(1 for r in results if not r.passed and r.severity == "error"),
               "warnings": sum(1 for r in results if not r.passed and r.severity == "warn"),
               "checks": len(results), "results": [asdict(r) for r in results]}
    path.write_text(json.dumps(summary, indent=2))
    return summary


def print_report(results: list[Result]) -> None:
    for r in results:
        mark = "PASS" if r.passed else ("FAIL" if r.severity == "error" else "WARN")
        print(f"  [{mark}] {r.table}: {r.check} ({r.detail})")


def main() -> int:
    print("Running data-quality checks ...")
    results = run_all()
    print_report(results)
    s = write_report(results)
    print(f"\n{s['checks']} checks: {s['errors']} errors, {s['warnings']} warnings -> {REPORT.relative_to(ROOT)}")
    return 1 if s["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
