# scripts/ml_model.py
# Trains a Random Forest model to predict whether an operator will have
# below-80% punctuality in the next quarter.
# Run from the project root: python scripts/ml_model.py

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_curve, auc)

ROOT     = Path(__file__).resolve().parent.parent
DB_PATH  = ROOT / "database" / "delays.db"
VIS_DIR  = ROOT / "visuals"
MODEL_DIR = ROOT / "models"

THRESHOLD = 80.0   # below this = "delayed" (poor performance)


def load_features(conn):
    """
    Join operator_punctuality + delay_causes to build a feature-rich table.
    Returns a DataFrame ready for ML.
    """
    punct = pd.read_sql("""
        SELECT operator, year, quarter, fin_year, pct_on_time_3m, recorded_stops
        FROM operator_punctuality
        WHERE pct_on_time_3m IS NOT NULL
        ORDER BY operator, year, quarter
    """, conn)

    causes = pd.read_sql("""
        SELECT operator, year_label AS fin_year, period_num,
               total_delay_mins, nr_responsible_mins, toc_responsible_mins,
               nr_severe_weather, toc_self_traincrew, toc_on_toc_traincrew
        FROM delay_causes
    """, conn)

    # Aggregate causes to quarterly (13 periods ≈ 1 year, ~3.25 periods per quarter)
    causes_q = causes.groupby(["operator", "fin_year"]).agg(
        total_delay_mins     = ("total_delay_mins", "sum"),
        nr_responsible_mins  = ("nr_responsible_mins", "sum"),
        toc_responsible_mins = ("toc_responsible_mins", "sum"),
        severe_weather_mins  = ("nr_severe_weather", "sum"),
        traincrew_mins       = ("toc_self_traincrew", "sum"),
    ).reset_index()

    df = punct.merge(causes_q, on=["operator", "fin_year"], how="left")

    # ── Feature engineering ──────────────────────────────────────────────────
    df = df.sort_values(["operator", "year", "quarter"]).reset_index(drop=True)

    # Lag features: last quarter's punctuality
    df["lag1_pct"] = df.groupby("operator")["pct_on_time_3m"].shift(1)
    df["lag2_pct"] = df.groupby("operator")["pct_on_time_3m"].shift(2)
    df["lag4_pct"] = df.groupby("operator")["pct_on_time_3m"].shift(4)  # same Q, last year

    # Rolling 4-quarter average
    df["roll4_pct"] = df.groupby("operator")["pct_on_time_3m"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).mean()
    )

    # Trend: slope over last 4 quarters
    def trend(series, n=4):
        vals = series.shift(1).rolling(n, min_periods=n).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
        )
        return vals

    df["trend_4q"] = df.groupby("operator")["pct_on_time_3m"].transform(trend)

    # Seasonality: quarter dummies
    df["is_q1"] = (df["quarter"] == "Q1").astype(int)
    df["is_q2"] = (df["quarter"] == "Q2").astype(int)
    df["is_q3"] = (df["quarter"] == "Q3").astype(int)
    df["is_q4"] = (df["quarter"] == "Q4").astype(int)

    # Delay ratios (where available)
    df["nr_delay_ratio"] = df["nr_responsible_mins"] / (df["total_delay_mins"] + 1)
    df["severe_weather_ratio"] = df["severe_weather_mins"] / (df["total_delay_mins"] + 1)
    df["traincrew_ratio"] = df["traincrew_mins"] / (df["total_delay_mins"] + 1)

    # Encode operator
    le = LabelEncoder()
    df["operator_enc"] = le.fit_transform(df["operator"].fillna("Unknown"))

    # Target: next quarter below threshold
    df["target"] = (df.groupby("operator")["pct_on_time_3m"].shift(-1) < THRESHOLD).astype(int)

    return df, le


FEATURE_COLS = [
    "lag1_pct", "lag2_pct", "lag4_pct", "roll4_pct", "trend_4q",
    "is_q1", "is_q2", "is_q3", "is_q4",
    "total_delay_mins", "nr_responsible_mins", "toc_responsible_mins",
    "nr_delay_ratio", "severe_weather_ratio", "traincrew_ratio",
    "operator_enc",
]


def train_and_evaluate(conn):
    print("[ML] Building features ...")
    df, le = load_features(conn)

    # Drop rows with any NaN in features or target
    df_model = df[FEATURE_COLS + ["target"]].dropna()
    if len(df_model) < 80:
        print(f"[ML] Only {len(df_model)} usable rows after dropna — model may be weak.")

    X = df_model[FEATURE_COLS]
    y = df_model["target"]

    print(f"[ML] Dataset: {len(X)} rows | class balance: {y.mean():.1%} 'underperforming'")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.sum() >= 10 else None
    )

    # ── Random Forest ──────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="f1")
    print(f"[ML] Cross-val F1 scores: {cv_scores.round(3)} — mean={cv_scores.mean():.3f}")

    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    print("\n[ML] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["On track (≥80%)", "At risk (<80%)"]))

    # ── ROC ───────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # ── Charts ────────────────────────────────────────────────────────────
    VIS_DIR.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # Feature importance
    ax1 = fig.add_subplot(gs[0])
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
    importances.plot.barh(ax=ax1, color="#185FA5")
    ax1.set_title("Feature Importance", fontweight="bold")
    ax1.set_xlabel("Mean decrease in impurity")

    # Confusion matrix
    ax2 = fig.add_subplot(gs[1])
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["On track", "At risk"]).plot(
        ax=ax2, colorbar=False, cmap="Blues"
    )
    ax2.set_title("Confusion Matrix", fontweight="bold")

    # ROC curve
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(fpr, tpr, color="#A32D2D", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax3.plot([0, 1], [0, 1], "k--", lw=1)
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve", fontweight="bold")
    ax3.legend(loc="lower right")

    fig.suptitle(
        f"ML Model — Predicting Underperformance (<{THRESHOLD}% on-time)\n"
        f"Random Forest | CV F1 = {cv_scores.mean():.3f} | ROC AUC = {roc_auc:.3f}",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    out = VIS_DIR / "7_ml_model_results.png"
    fig.savefig(out.as_posix(), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ML] Chart saved → visuals/7_ml_model_results.png")

    # Save model + encoder
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(rf, MODEL_DIR / "rf_model.joblib")
    joblib.dump(le, MODEL_DIR / "operator_encoder.joblib")
    print(f"[ML] Model saved → models/rf_model.joblib")

    return rf, le, FEATURE_COLS


if __name__ == "__main__":
    print("=" * 55)
    print("UK Train Delay Analytics — ML Model")
    print("=" * 55)
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found. Run load_to_sql.py first.")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH.as_posix())
    train_and_evaluate(conn)
    conn.close()
    print("\n✓ Done.")
