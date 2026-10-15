# scripts/features.py
# Single source of truth for ML features. Used by BOTH ml_model.py (training)
# and the Streamlit app (prediction), so the two can never drift apart.
#
# Task: given everything known at the END of quarter t, predict whether the
# operator's on-time % (within 3 mins) in quarter t+1 will be below THRESHOLD.

import numpy as np
import pandas as pd

THRESHOLD = 80.0

# ORR's 4-weekly delay periods -> quarters (P1-3 Apr-Jun, P4-6 Jul-Sep, P7-9 Oct-Dec, P10-13 Jan-Mar)
PERIOD_TO_QUARTER = {1: "Q1", 2: "Q1", 3: "Q1", 4: "Q2", 5: "Q2", 6: "Q2",
                     7: "Q3", 8: "Q3", 9: "Q3", 10: "Q4", 11: "Q4", 12: "Q4", 13: "Q4"}

CORE = ["cur_pct", "prev1_pct", "prev2_pct", "seasonal_pct"]
FEATURES = [
    "cur_pct",             # on-time % in the latest quarter (t)
    "prev1_pct",           # t-1
    "prev2_pct",           # t-2
    "seasonal_pct",        # same quarter as the one being predicted, one year earlier (t-3)
    "roll4_pct",           # mean of t, t-1, t-2, t-3
    "delta1",              # cur_pct - prev1_pct (momentum)
    "delay_per_1k_stops",  # delay minutes per 1,000 recorded stops in quarter t (size-normalised)
    "nr_share",            # share of delay minutes caused by Network Rail
    "weather_share",       # share caused by severe weather
    "traincrew_share",     # share caused by the operator's own traincrew issues
    "tgt_q1", "tgt_q2", "tgt_q3", "tgt_q4",   # which quarter is being predicted (seasonality)
]


def quarter_index(qdate: pd.Series) -> pd.Series:
    return qdate.dt.year * 4 + (qdate.dt.month - 1) // 3


def quarter_label(ts) -> str:
    ts = pd.Timestamp(ts)
    start = ts.strftime("%b")
    end = (ts + pd.DateOffset(months=2)).strftime("%b")
    return f"{start}–{end} {ts.year}"


def next_quarter_start(ts) -> pd.Timestamp:
    return pd.Timestamp(ts) + pd.DateOffset(months=3)


def build_feature_table(conn) -> pd.DataFrame:
    punct = pd.read_sql(
        """SELECT operator, fin_year, quarter, qdate, pct_on_time_3m AS cur_pct, recorded_stops
           FROM operator_punctuality WHERE pct_on_time_3m IS NOT NULL""",
        conn, parse_dates=["qdate"])

    causes = pd.read_sql(
        """SELECT operator, year_label AS fin_year, period_num, total_delay_mins,
                  nr_responsible_mins, nr_severe_weather, toc_self_traincrew
           FROM delay_causes""", conn)
    causes["quarter"] = causes["period_num"].astype(int).map(PERIOD_TO_QUARTER)
    cq = (causes.groupby(["operator", "fin_year", "quarter"], as_index=False)
          [["total_delay_mins", "nr_responsible_mins", "nr_severe_weather", "toc_self_traincrew"]].sum())

    df = punct.merge(cq, on=["operator", "fin_year", "quarter"], how="left")
    df["qidx"] = quarter_index(df["qdate"])

    # lags by calendar position (robust to gaps in an operator's history)
    base = df[["operator", "qidx", "cur_pct"]]
    for k, name in [(1, "prev1_pct"), (2, "prev2_pct"), (3, "seasonal_pct")]:
        s = base.copy()
        s["qidx"] = s["qidx"] + k
        df = df.merge(s.rename(columns={"cur_pct": name}), on=["operator", "qidx"], how="left")

    nxt = base.copy()
    nxt["qidx"] = nxt["qidx"] - 1
    df = df.merge(nxt.rename(columns={"cur_pct": "next_pct"}), on=["operator", "qidx"], how="left")

    window = df[["cur_pct", "prev1_pct", "prev2_pct", "seasonal_pct"]]
    df["roll4_pct"] = window.mean(axis=1).where(window.notna().sum(axis=1) >= 3)
    df["delta1"] = df["cur_pct"] - df["prev1_pct"]

    tot = df["total_delay_mins"].where(df["total_delay_mins"] > 0)
    df["delay_per_1k_stops"] = df["total_delay_mins"] / df["recorded_stops"].where(df["recorded_stops"] > 0) * 1000
    df["nr_share"] = df["nr_responsible_mins"] / tot
    df["weather_share"] = df["nr_severe_weather"] / tot
    df["traincrew_share"] = df["toc_self_traincrew"] / tot

    cur_q = df["quarter"].str[1].astype(int)
    tgt_q = cur_q % 4 + 1
    for q in range(1, 5):
        df[f"tgt_q{q}"] = (tgt_q == q).astype(int)

    df["target"] = np.where(df["next_pct"].notna(), (df["next_pct"] < THRESHOLD).astype(float), np.nan)
    return df.sort_values(["operator", "qdate"]).reset_index(drop=True)
