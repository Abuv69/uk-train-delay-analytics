"""Forecast pipeline: lags, targets and delay features checked against hand calculations."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from scripts.features import FEATURES, PERIOD_TO_QUARTER, build_feature_table


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    # Operator A: five consecutive quarters, 1,000 recorded stops each
    punct = [("A", "2023-24", "Q1", "2023-04-01", 90.0), ("A", "2023-24", "Q2", "2023-07-01", 70.0),
             ("A", "2023-24", "Q3", "2023-10-01", 85.0), ("A", "2023-24", "Q4", "2024-01-01", 60.0),
             ("A", "2024-25", "Q1", "2024-04-01", 95.0),
             # Operator B: a missing quarter (no Jul 2023)
             ("B", "2023-24", "Q1", "2023-04-01", 80.0), ("B", "2023-24", "Q3", "2023-10-01", 90.0)]
    pd.DataFrame(punct, columns=["operator", "fin_year", "quarter", "qdate", "pct_on_time_3m"]).assign(
        recorded_stops=1000.0).to_sql("operator_punctuality", c, index=False)
    # Delay data for A, financial year 2023-24: every 4-week period has 100 delay minutes,
    # of which 60 Network Rail, 10 weather, 5 own traincrew
    d = [("A", "2023-24", p, 100.0, 60.0, 10.0, 5.0) for p in range(1, 14)]
    pd.DataFrame(d, columns=["operator", "year_label", "period_num", "total_delay_mins",
                             "nr_responsible_mins", "nr_severe_weather", "toc_self_traincrew"]).to_sql(
        "delay_causes", c, index=False)
    return c


def row(df, op, qdate):
    return df[(df.operator == op) & (df.qdate == pd.Timestamp(qdate))].iloc[0]


def test_period_to_quarter_mapping():
    assert [PERIOD_TO_QUARTER[p] for p in range(1, 14)] == ["Q1"] * 3 + ["Q2"] * 3 + ["Q3"] * 3 + ["Q4"] * 4


def test_lags_for_latest_row(conn):
    r = row(build_feature_table(conn), "A", "2024-04-01")
    # t=Apr 2024: t-1=Jan 2024 (60), t-2=Oct 2023 (85), t-3=Jul 2023 (70)
    assert (r.cur_pct, r.prev1_pct, r.prev2_pct, r.seasonal_pct) == (95, 60, 85, 70)
    assert r.roll4_pct == pytest.approx((95 + 60 + 85 + 70) / 4)      # 77.5
    assert r.delta1 == 35
    assert np.isnan(r.target)                                         # no following quarter yet -> unknown, not 0


def test_target_is_next_quarter_below_80(conn):
    df = build_feature_table(conn)
    assert row(df, "A", "2024-01-01").target == 0     # next = 95 (>= 80)
    assert row(df, "A", "2023-04-01").target == 1     # next = 70 (< 80)
    assert row(df, "A", "2023-10-01").target == 1     # next = 60 (< 80)
    r = row(df, "A", "2024-01-01")
    assert (r.cur_pct, r.prev1_pct, r.prev2_pct, r.seasonal_pct) == (60, 85, 70, 90)
    assert r.roll4_pct == pytest.approx(76.25) and r.delta1 == -25


def test_missing_history_is_not_filled_in(conn):
    df = build_feature_table(conn)
    first = row(df, "A", "2023-04-01")
    assert np.isnan(first.prev1_pct) and np.isnan(first.roll4_pct)    # only 1 of 4 values known -> no average
    third = row(df, "A", "2023-10-01")
    assert np.isnan(third.seasonal_pct)
    assert third.roll4_pct == pytest.approx((85 + 70 + 90) / 3)       # 3 of 4 known -> mean of those 3


def test_gap_in_an_operators_history_does_not_shift_lags(conn):
    df = build_feature_table(conn)
    r = row(df, "B", "2023-10-01")
    assert np.isnan(r.prev1_pct)            # Jul 2023 is missing -> not silently replaced by Apr 2023
    assert r.prev2_pct == 80                # Apr 2023 is exactly two quarters earlier
    assert np.isnan(row(df, "B", "2023-04-01").target)   # Jul 2023 missing -> target unknown


def test_delay_features_hand_calculated(conn):
    df = build_feature_table(conn)
    q1 = row(df, "A", "2023-04-01")          # periods 1-3: 3 x 100 = 300 delay minutes
    assert q1.total_delay_mins == 300
    assert q1.delay_per_1k_stops == pytest.approx(300 / 1000 * 1000)
    assert q1.nr_share == pytest.approx(0.6) and q1.weather_share == pytest.approx(0.1)
    assert q1.traincrew_share == pytest.approx(0.05)
    q4 = row(df, "A", "2024-01-01")          # periods 10-13: 4 x 100 = 400 delay minutes
    assert q4.total_delay_mins == 400 and q4.delay_per_1k_stops == pytest.approx(400)
    assert np.isnan(row(df, "A", "2024-04-01").nr_share)   # no delay data for FY 2024-25 -> NaN, never 0


def test_forecast_quarter_dummies(conn):
    r = row(build_feature_table(conn), "A", "2024-01-01")   # current Q4 (Jan-Mar) -> forecasting Q1 (Apr-Jun)
    assert (r.tgt_q1, r.tgt_q2, r.tgt_q3, r.tgt_q4) == (1, 0, 0, 0)
    assert set(FEATURES) <= set(build_feature_table(conn).columns)
