"""Data-quality checks (hand-built frames with known answers) and the database switch."""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import db, quality_checks as qc  # noqa: E402


def punct(**over):
    rows = []
    for op in ["A", "B"]:
        for i, d in enumerate(["2025-01-01", "2025-04-01", "2025-07-01"]):
            rows.append(dict(operator=op, qdate=d, recorded_stops=1000, pct_on_time_1m=60.0 + i,
                             pct_on_time_3m=80.0 + i, pct_on_time_15m=95.0, quarter="Q1", fin_year="2024-25",
                             provisional=False))
    df = pd.DataFrame(rows)
    for k, v in over.items():
        df[k] = v
    return df


def result(results, check):
    return next(r for r in results if r.check == check)


def test_clean_punctuality_passes_the_content_checks():
    res = qc.check_punctuality(punct())
    for name in ["one row per operator and quarter", "pct_on_time_3m between 0 and 100",
                 "within 1 min <= within 3 min <= within 15 min", "no missing quarters between first and last"]:
        assert result(res, name).passed, name


def test_duplicate_operator_quarter_is_an_error():
    df = pd.concat([punct(), punct().iloc[:1]])
    r = result(qc.check_punctuality(df), "one row per operator and quarter")
    assert not r.passed and r.severity == "error" and "1 duplicate" in r.detail


def test_percentage_above_100_is_an_error():
    df = punct()
    df.loc[0, "pct_on_time_15m"] = 101
    assert not result(qc.check_punctuality(df), "pct_on_time_15m between 0 and 100").passed


def test_percentages_must_nest():
    df = punct()
    df.loc[0, "pct_on_time_1m"] = 90          # within 1 min can never exceed within 3 min (80)
    r = result(qc.check_punctuality(df), "within 1 min <= within 3 min <= within 15 min")
    assert not r.passed and "1 rows" in r.detail


def test_missing_quarter_is_named():
    df = punct()
    df = df[df.qdate != "2025-04-01"]
    r = result(qc.check_punctuality(df), "no missing quarters between first and last")
    assert not r.passed and "Apr 2025" in r.detail


def test_blank_header_columns_are_caught():
    df = punct()
    df["nan"] = None
    assert not result(qc.check_punctuality(df), "no blank-header columns").passed


def test_delay_total_must_equal_sum_of_causes():
    row = {c: 10.0 for c in qc.CAUSE_COLS}
    df = pd.DataFrame([dict(operator="A", year_label="2025-26", period_num=1, total_delay_mins=150.0,
                            nr_responsible_mins=50.0, toc_responsible_mins=100.0, **row)])
    res = qc.check_delays(df)
    assert result(res, "total equals the sum of the 15 causes").passed
    assert result(res, "Network Rail + train company = total").passed
    df.loc[0, "total_delay_mins"] = 999.0
    assert not result(qc.check_delays(df), "total equals the sum of the 15 causes").passed


def test_report_counts_errors_and_warnings(tmp_path):
    res = [qc.Result("t", "a", "error", False, "x"), qc.Result("t", "b", "warn", False, "y"),
           qc.Result("t", "c", "error", True, "ok")]
    s = qc.write_report(res, tmp_path / "r.json")
    assert (s["errors"], s["warnings"], s["checks"]) == (1, 1, 3)


# ── database switch ───────────────────────────────────────────────────────────
def test_sqlite_is_the_default(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert not db.is_postgres() and db.database_url().startswith("sqlite:///")


@pytest.mark.parametrize("given", ["postgres://u:p@h:5432/d", "postgresql://u:p@h:5432/d"])
def test_postgres_urls_are_normalised(monkeypatch, given):
    monkeypatch.setenv("DATABASE_URL", given)
    assert db.is_postgres() and db.database_url().startswith("postgresql+psycopg2://")


def test_describe_never_shows_the_password(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:SECRET123@db.example.com:5432/trains")
    text = db.describe()
    assert "SECRET123" not in text and "db.example.com" in text and "trains" in text


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against PostgreSQL")
def test_postgres_has_the_same_tables(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    with db.connect() as conn:
        for t in ["operator_punctuality", "delay_causes", "station_performance", "build_info"]:
            assert db.table_exists(conn, t), t
        n = pd.read_sql("SELECT COUNT(*) n FROM operator_punctuality", conn)["n"].iloc[0]
    assert n > 1000
