# scripts/ml_model.py
# Predicts whether an operator's on-time % (3 min) will fall below 80% NEXT quarter.
#
# Honest evaluation:
#   * TIME-BASED split: train on the past, test on later quarters (no random shuffling)
#   * one-quarter embargo between train and test so no label leaks across the boundary
#   * compared against naive baselines
#   * CALIBRATION is measured, not assumed: Brier score, log-loss, expected calibration error,
#     and a table of "when we said X, how often did it happen?" saved for the dashboard.
#
# Run from the project root: python scripts/ml_model.py

import json
import sqlite3
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.features import (CORE, FEATURES, THRESHOLD, build_feature_table,  # noqa: E402
                              quarter_label)

DB_PATH = ROOT / "database" / "delays.db"
MODEL_DIR = ROOT / "models"
VIS_DIR = ROOT / "visuals"

TRAIN_END = pd.Timestamp("2022-10-01")   # last training quarter start (its label = Jan-Mar 2023)
TEST_START = pd.Timestamp("2023-04-01")  # first test quarter (one-quarter embargo in between)
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
BANDS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.0001)]


def scores(y, pred, proba):
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 3),
        "f1": round(float(f1_score(y, pred)), 3),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y, pred)), 3),
        "accuracy": round(float(accuracy_score(y, pred)), 3),
    }


def calibration(y, p, n_bins=5):
    """Brier, log-loss, expected calibration error, and share of 'extreme' (<2% or >98%) forecasts."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    idx = np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1])
    ece = sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean())
              for b in range(n_bins) if (idx == b).any())
    return {"brier": round(float(brier_score_loss(y, p)), 4),
            "log_loss": round(float(log_loss(y, p)), 4),
            "ece": round(float(ece), 4),
            "share_extreme_forecasts": round(float(((p < 0.02) | (p > 0.98)).mean()), 3)}


def band_table(y, p):
    """'When the model said a risk in this range, how often did it really fall below 80%?'"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    rows = []
    for lo, hi in BANDS:
        m = (p >= lo) & (p < hi)
        rows.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()),
                     "mean_predicted": round(float(p[m].mean()), 3) if m.any() else None,
                     "actual_rate": round(float(y[m].mean()), 3) if m.any() else None})
    return rows


def lr(C, balanced):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=C, max_iter=3000, class_weight="balanced" if balanced else None))


def pick_C(lab):
    """Choose regularisation by rolling-origin validation INSIDE the training period only."""
    train = lab[lab["qdate"] <= TRAIN_END]
    cutoffs = [pd.Timestamp("2016-10-01"), pd.Timestamp("2018-10-01"), pd.Timestamp("2020-10-01")]
    out = {}
    for C in C_GRID:
        losses = []
        for cut in cutoffs:
            tr = train[train["qdate"] <= cut]
            va = train[(train["qdate"] >= cut + pd.DateOffset(months=6)) &
                       (train["qdate"] < cut + pd.DateOffset(months=30))]
            if tr["target"].nunique() < 2 or va.empty:
                continue
            m = lr(C, False).fit(tr[FEATURES], tr["target"])
            losses.append(log_loss(va["target"], np.clip(m.predict_proba(va[FEATURES])[:, 1], 1e-6, 1 - 1e-6),
                                   labels=[0, 1]))
        out[C] = float(np.mean(losses))
    best = min(out, key=out.get)
    return best, out


def main():
    if not DB_PATH.exists():
        print("[ERROR] Database not found. Run load_to_sql.py first.")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH.as_posix())
    df = build_feature_table(conn)
    conn.close()

    lab = df.dropna(subset=["target"] + CORE).copy()
    medians = lab.loc[lab["qdate"] <= TRAIN_END, FEATURES].median().to_dict()
    lab[FEATURES] = lab[FEATURES].fillna(pd.Series(medians))

    train = lab[lab["qdate"] <= TRAIN_END]
    test = lab[lab["qdate"] >= TEST_START]
    Xtr, ytr, Xte, yte = train[FEATURES], train["target"], test[FEATURES], test["target"]
    print(f"Train: {len(train)} rows up to {quarter_label(train.qdate.max())}  |  "
          f"Test: {len(test)} rows {quarter_label(test.qdate.min())} → {quarter_label(test.qdate.max())}")

    # ── calibration investigation ─────────────────────────────────────────────
    prev = lr(1.0, True).fit(Xtr, ytr)               # the version first deployed
    p_prev = prev.predict_proba(Xte)[:, 1]
    bestC, cv = pick_C(lab)
    new = lr(bestC, False).fit(Xtr, ytr)             # unweighted + regularisation tuned on past data only
    p_new = new.predict_proba(Xte)[:, 1]
    cal = {"previous_balanced_C1": calibration(yte, p_prev),
           "deployed_unweighted_tunedC": calibration(yte, p_new), "chosen_C": bestC,
           "cv_log_loss_by_C": {str(k): round(v, 4) for k, v in cv.items()}}
    print("\nCalibration on future quarters (lower Brier / log-loss / ECE is better):")
    print("  previous (balanced, C=1):   ", cal["previous_balanced_C1"])
    print(f"  deployed (unweighted, C={bestC}):", cal["deployed_unweighted_tunedC"])
    print("  share of test forecasts under 2% or over 98%: previous "
          f"{cal['previous_balanced_C1']['share_extreme_forecasts']:.0%} → "
          f"deployed {cal['deployed_unweighted_tunedC']['share_extreme_forecasts']:.0%}")

    # ── model comparison ──────────────────────────────────────────────────────
    results = {}
    results["Baseline: same as this quarter"] = scores(
        yte, (test["cur_pct"] < THRESHOLD).astype(int), 100 - test["cur_pct"])
    results["Baseline: same quarter last year"] = scores(
        yte, (test["seasonal_pct"] < THRESHOLD).astype(int), 100 - test["seasonal_pct"])
    rf = RandomForestClassifier(n_estimators=400, max_depth=5, min_samples_leaf=5,
                                class_weight="balanced", random_state=42, n_jobs=-1).fit(Xtr, ytr)
    results["Random forest"] = scores(yte, rf.predict(Xte), rf.predict_proba(Xte)[:, 1])
    results["Logistic regression (deployed)"] = scores(yte, (p_new >= 0.5).astype(int), p_new)
    table = pd.DataFrame(results).T
    print("\nTest-set results (future quarters only):")
    print(table.to_string())

    bands = band_table(yte, p_new)
    print("\nWhen the deployed model said …, how often did the operator really fall below 80%?")
    for b in bands:
        print(f"  predicted {b['lo']:.0%}–{b['hi']:.0%}: n={b['n']:3d}  actual {b['actual_rate']}")

    # ── final model: refit on ALL labelled quarters ───────────────────────────
    final = lr(bestC, False).fit(lab[FEATURES], lab["target"])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES, "medians": medians,
                 "threshold": THRESHOLD}, MODEL_DIR / "risk_model.joblib")

    coefs = new.named_steps["logisticregression"].coef_[0]
    metrics = {
        "task": f"Predict on-time % (3 min) below {THRESHOLD:.0f}% next quarter",
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_until": quarter_label(train.qdate.max()),
        "test_from": quarter_label(test.qdate.min()), "test_to": quarter_label(test.qdate.max()),
        "at_risk_share_test": round(float(yte.mean()), 3),
        "results": results, "deployed_model": f"Logistic regression, C={bestC}, no class weights",
        "coefficients": dict(zip(FEATURES, [round(float(v), 3) for v in coefs])),
        "calibration": cal, "calibration_bands": bands,
        "deployed_model_trained_on": int(len(lab)),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    VIS_DIR.mkdir(exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    table[["roc_auc", "f1"]].plot.barh(ax=a1, color=["#185FA5", "#1D9E75"])
    a1.set_title("Test on future quarters: model vs baselines", fontweight="bold")
    a1.set_xlim(0, 1)
    pd.Series(metrics["coefficients"]).sort_values().plot.barh(ax=a2, color="#185FA5")
    a2.set_title("Logistic regression coefficients (standardised)", fontweight="bold")
    fig.tight_layout()
    fig.savefig((VIS_DIR / "7_ml_model_results.png").as_posix(), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved models/risk_model.joblib, models/metrics.json, visuals/7_ml_model_results.png")


if __name__ == "__main__":
    main()
