"""The deployed forecast: calibration checks and the 'never 0% or 100%' rule."""
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts import insights as ins
from scripts.features import FEATURES, build_feature_table

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def metrics():
    return json.loads((ROOT / "models" / "metrics.json").read_text())


@pytest.fixture(scope="module")
def latest_forecasts():
    bundle = joblib.load(ROOT / "models" / "risk_model.joblib")
    with sqlite3.connect(ROOT / "database" / "delays.db") as conn:
        feats = build_feature_table(conn)
    q = feats.qdate.max()
    ok = feats[feats.qdate == q].dropna(subset=["cur_pct", "prev1_pct", "prev2_pct", "seasonal_pct"]).copy()
    ok["p_below"] = bundle["model"].predict_proba(ok[FEATURES].astype(float).fillna(pd.Series(bundle["medians"])))[:, 1]
    return ok, bundle


def test_calibration_bands_account_for_every_test_case(metrics):
    bands = metrics["calibration_bands"]
    assert sum(b["n"] for b in bands) == metrics["test_rows"]
    for b in bands:
        if b["n"]:
            assert 0 <= b["actual_rate"] <= 1 and 0 <= b["mean_predicted"] <= 1


def test_overconfidence_is_reduced_and_reported(metrics):
    cal = metrics["calibration"]
    assert cal["deployed_unweighted_tunedC"]["share_extreme_forecasts"] < cal["previous_balanced_C1"]["share_extreme_forecasts"]
    assert cal["deployed_unweighted_tunedC"]["share_extreme_forecasts"] < 0.25
    assert cal["deployed_unweighted_tunedC"]["ece"] < 0.15        # documented limitation, guarded against regression


def test_model_beats_no_skill_and_is_sensible(metrics):
    res = metrics["results"]["Logistic regression (deployed)"]
    assert res["roc_auc"] > 0.9 and res["accuracy"] > 0.85


def test_risk_goes_up_when_on_time_falls(latest_forecasts):
    ok, _ = latest_forecasts
    assert ok.nsmallest(3, "cur_pct")["p_below"].mean() > ok.nlargest(3, "cur_pct")["p_below"].mean()


def test_no_operator_ever_displays_as_certain(latest_forecasts):
    ok, bundle = latest_forecasts
    assert len(ok) >= 20
    bands = json.loads((ROOT / "models" / "metrics.json").read_text())["calibration_bands"]
    for _, r in ok.iterrows():
        p = float(r["p_below"])
        assert 0.0 <= p <= 1.0
        for shown in (ins.format_probability(p), ins.format_probability(1 - p)):
            assert shown not in ("0%", "100%", "about 0%", "about 100%")
        assert ins.outlook(p)[0] in {"very likely", "likely", "uncertain", "unlikely", "very unlikely"}
        c = ins.confidence(p, [], bands)
        assert c["level"] in {"high", "medium", "low"}
