# scripts/ml_model.py
# Predicts whether an operator's on-time % (3 min) will fall below 80% NEXT quarter.
#
# Honest evaluation:
#   * TIME-BASED split: train on the past, test on later quarters (no random shuffling)
#   * one-quarter embargo between train and test so no label leaks across the boundary
#   * compared against naive baselines - a model that can't beat "same as this quarter"
#     isn't adding value
#
# Run from the project root: python scripts/ml_model.py

import json
import sqlite3
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
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


def scores(y, pred, proba):
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 3),
        "f1": round(float(f1_score(y, pred)), 3),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y, pred)), 3),
        "accuracy": round(float(accuracy_score(y, pred)), 3),
    }


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
    print(f"At-risk share  train {ytr.mean():.1%}  test {yte.mean():.1%}")

    results = {}

    # Baseline 1: persistence - "next quarter will be at risk if THIS quarter was"
    results["Baseline: same as this quarter"] = scores(
        yte, (test["cur_pct"] < THRESHOLD).astype(int), 100 - test["cur_pct"])
    # Baseline 2: seasonal - "same as this quarter last year"
    results["Baseline: same quarter last year"] = scores(
        yte, (test["seasonal_pct"] < THRESHOLD).astype(int), 100 - test["seasonal_pct"])

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(Xtr, ytr)
    results["Logistic regression"] = scores(yte, lr.predict(Xte), lr.predict_proba(Xte)[:, 1])

    rf = RandomForestClassifier(n_estimators=400, max_depth=5, min_samples_leaf=5,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    results["Random forest"] = scores(yte, rf.predict(Xte), rf.predict_proba(Xte)[:, 1])

    table = pd.DataFrame(results).T
    print("\nTest-set results (future quarters only):")
    print(table.to_string())

    # Final deployed model: logistic regression (simplest model that matched/beat the rest),
    # refit on ALL labelled quarters
    final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    final.fit(lab[FEATURES], lab["target"])

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES, "medians": medians,
                 "threshold": THRESHOLD}, MODEL_DIR / "risk_model.joblib")

    coefs = lr.named_steps["logisticregression"].coef_[0]   # standardised -> comparable magnitudes
    importances = dict(zip(FEATURES, [round(float(v), 3) for v in coefs]))
    metrics = {
        "task": f"Predict on-time % (3 min) below {THRESHOLD:.0f}% next quarter",
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_until": quarter_label(train.qdate.max()),
        "test_from": quarter_label(test.qdate.min()), "test_to": quarter_label(test.qdate.max()),
        "at_risk_share_test": round(float(yte.mean()), 3),
        "results": results, "deployed_model": "Logistic regression", "coefficients": importances,
        "deployed_model_trained_on": int(len(lab)),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    VIS_DIR.mkdir(exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    table[["roc_auc", "f1"]].plot.barh(ax=a1, color=["#185FA5", "#1D9E75"])
    a1.set_title("Test on future quarters: model vs baselines", fontweight="bold")
    a1.set_xlim(0, 1)
    pd.Series(importances).sort_values().plot.barh(ax=a2, color="#185FA5")
    a2.set_title("Logistic regression coefficients (standardised)", fontweight="bold")
    fig.tight_layout()
    fig.savefig((VIS_DIR / "7_ml_model_results.png").as_posix(), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved models/risk_model.joblib, models/metrics.json, visuals/7_ml_model_results.png")


if __name__ == "__main__":
    main()
