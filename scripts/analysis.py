# scripts/analysis.py
# Generates 6 analytical charts from the SQLite database.
# Run from the project root: python scripts/analysis.py

from pathlib import Path
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sys

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "database" / "delays.db"
VIS_DIR = ROOT / "visuals"

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
BLUE   = "#185FA5"
TEAL   = "#1D9E75"
AMBER  = "#BA7517"
RED    = "#A32D2D"
PURPLE = "#534AB7"
GRAY   = "#73726c"


def get_conn():
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found at {DB_PATH.relative_to(ROOT)}")
        print("Run scripts/load_to_sql.py first.")
        sys.exit(1)
    return sqlite3.connect(DB_PATH.as_posix())


def save(fig, name):
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    p = VIS_DIR / name
    fig.savefig(p.as_posix(), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → visuals/{name}")


# ── Chart 1: Top 10 operators by total delay (all years) ──────────────────────
def chart_worst_operators(conn):
    df = pd.read_sql("""
        SELECT operator, SUM(total_delay_mins) / 1e6 AS delay_M
        FROM delay_causes
        GROUP BY operator
        ORDER BY delay_M DESC
        LIMIT 10
    """, conn)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["operator"][::-1], df["delay_M"][::-1], color=BLUE)
    ax.set_xlabel("Total Delay (millions of minutes)")
    ax.set_title("Top 10 Operators by Total Delay — Apr 2019 to Present", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.bar_label(bars, labels=[f"{v:.1f}M" for v in df["delay_M"][::-1]],
                 padding=4, fontsize=9, color=GRAY)
    fig.tight_layout()
    save(fig, "1_worst_operators.png")


# ── Chart 2: Delay trend over time (total across all operators) ────────────────
def chart_delay_trend(conn):
    df = pd.read_sql("""
        SELECT year_label, period_num,
               SUM(total_delay_mins) / 1e6 AS delay_M,
               SUM(nr_responsible_mins)  / 1e6 AS nr_M,
               SUM(toc_responsible_mins) / 1e6 AS toc_M
        FROM delay_causes
        GROUP BY year_label, period_num
        ORDER BY year_label, period_num
    """, conn)
    df["x"] = df["year_label"] + " P" + df["period_num"].astype(int).astype(str).str.zfill(2)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(range(len(df)), df["nr_M"],  alpha=0.4, color=BLUE,   label="Network Rail responsible")
    ax.fill_between(range(len(df)), df["toc_M"], alpha=0.4, color=AMBER,  label="TOC responsible")
    ax.plot(range(len(df)), df["delay_M"], color=RED, linewidth=1.8, label="Total delay")

    # Annotate COVID dip
    covid_mask = df["year_label"] == "2020-21"
    if covid_mask.any():
        idx = df.index[covid_mask][0]
        ax.annotate("COVID\nlockdowns", xy=(idx, df.loc[idx, "delay_M"]),
                    xytext=(idx - 8, df["delay_M"].max() * 0.85),
                    arrowprops=dict(arrowstyle="->", color=GRAY),
                    fontsize=9, color=GRAY)

    # x-axis: show year labels only
    year_starts = df.groupby("year_label").apply(lambda g: g.index.min()).values
    ax.set_xticks(year_starts)
    ax.set_xticklabels(df.loc[year_starts, "year_label"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Delay (millions of minutes)")
    ax.set_title("UK Rail Delay Trend by Period — April 2019 to Present", fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "2_delay_trend.png")


# ── Chart 3: Delay cause breakdown (latest full year) ─────────────────────────
def chart_cause_breakdown(conn):
    latest_year = pd.read_sql(
        "SELECT year_label FROM delay_causes ORDER BY start_year DESC LIMIT 1", conn
    ).iloc[0, 0]

    cause_labels = {
        "nr_external": "NR — External",
        "nr_network_mgmt": "NR — Network Mgmt",
        "nr_non_track_assets": "NR — Non-Track Assets",
        "nr_severe_weather": "NR — Severe Weather",
        "nr_track": "NR — Track",
        "toc_self_fleet": "TOC — Fleet",
        "toc_self_operations": "TOC — Operations",
        "toc_self_stations": "TOC — Stations",
        "toc_self_traincrew": "TOC — Traincrew",
        "toc_self_other": "TOC — Other",
    }

    df = pd.read_sql(f"""
        SELECT {', '.join(f'SUM({c}) AS {c}' for c in cause_labels)}
        FROM delay_causes
        WHERE year_label = '{latest_year}'
    """, conn)

    values = {cause_labels[c]: df[c].iloc[0] / 1e6 for c in cause_labels}
    values = dict(sorted(values.items(), key=lambda x: x[1], reverse=True))

    colors = [BLUE if "NR" in k else AMBER for k in values]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(list(values.keys())[::-1], list(values.values())[::-1],
                   color=colors[::-1])
    ax.set_xlabel("Delay (millions of minutes)")
    ax.set_title(f"Delay Causes — {latest_year}", fontweight="bold")
    ax.bar_label(bars, labels=[f"{v:.2f}M" for v in list(values.values())[::-1]],
                 padding=4, fontsize=9, color=GRAY)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Network Rail"), Patch(color=AMBER, label="TOC")],
              loc="lower right")
    fig.tight_layout()
    save(fig, "3_delay_causes.png")


# ── Chart 4: Best & worst 20 stations (latest year, min 100 scheduled stops) ──
def chart_stations(conn):
    try:
        df = pd.read_sql("""
            SELECT station,
                   AVG(time_to_3_pct)       AS avg_time_to_3,
                   AVG(cancellations_pct)   AS avg_cancellations,
                   SUM(scheduled_stops)     AS total_scheduled
            FROM station_performance
            WHERE time_to_3_pct IS NOT NULL
              AND scheduled_stops IS NOT NULL
            GROUP BY station
            HAVING total_scheduled >= 2000
            ORDER BY avg_time_to_3 DESC
        """, conn)
    except Exception as e:
        print(f"  [SKIP chart 4] {e}")
        return

    if df.empty:
        return

    best  = df.head(15)
    worst = df.tail(15).iloc[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.barh(best["station"], best["avg_time_to_3"], color=TEAL)
    ax1.set_xlabel("Avg % trains on time (within 3 mins)")
    ax1.set_title("15 Best Stations (Time-to-3)", fontweight="bold")
    ax1.set_xlim(80, 100)

    ax2.barh(worst["station"], worst["avg_time_to_3"], color=RED)
    ax2.set_xlabel("Avg % trains on time (within 3 mins)")
    ax2.set_title("15 Worst Stations (Time-to-3)", fontweight="bold")
    ax2.set_xlim(0, 80)

    fig.tight_layout()
    save(fig, "4_best_worst_stations.png")


# ── Chart 5: Operator punctuality over time (pct_on_time_3m) ──────────────────
def chart_operator_punctuality(conn):
    # Focus on major operators with long history
    MAJOR = [
        "Avanti West Coast", "CrossCountry", "East Midlands Railway",
        "Govia Thameslink Railway", "Great Western Railway",
        "London North Eastern Railway", "Northern Trains", "ScotRail",
        "South Western Railway", "TransPennine Express",
    ]
    placeholders = ",".join("?" * len(MAJOR))
    df = pd.read_sql(f"""
        SELECT operator, year, quarter, pct_on_time_3m
        FROM operator_punctuality
        WHERE operator IN ({placeholders})
          AND pct_on_time_3m IS NOT NULL
        ORDER BY operator, year, quarter
    """, conn, params=MAJOR)

    if df.empty:
        return

    df["x"] = df["year"].astype(str) + " " + df["quarter"]
    pivot = df.pivot_table(index="x", columns="operator", values="pct_on_time_3m")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    palette = sns.color_palette("tab10", n_colors=len(pivot.columns))
    for col, color in zip(pivot.columns, palette):
        ax.plot(range(len(pivot)), pivot[col], label=col, linewidth=1.5, alpha=0.85, color=color)

    ax.axhline(80, color=RED, linestyle="--", linewidth=1, label="80% benchmark")

    # x-axis: show year labels at Q1
    q1_idx = [i for i, v in enumerate(pivot.index) if "Q1" in v]
    ax.set_xticks(q1_idx)
    ax.set_xticklabels([pivot.index[i][:4] for i in q1_idx], fontsize=9)

    ax.set_ylabel("% Trains on time (within 3 mins)")
    ax.set_title("Operator Punctuality — Quarterly Trend (2014–2025)", fontweight="bold")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "5_operator_punctuality_trend.png")


# ── Chart 6: COVID impact — before / during / after comparison ────────────────
def chart_covid_impact(conn):
    df = pd.read_sql("""
        SELECT year_label, start_year,
               AVG(total_delay_mins) AS avg_period_delay
        FROM delay_causes
        GROUP BY year_label, start_year
        ORDER BY start_year
    """, conn)

    if df.empty:
        return

    def era(y):
        if y < 2020:   return "Pre-COVID\n(2019–20)"
        elif y == 2020: return "COVID\n(2020–21)"
        elif y == 2021: return "Recovery\n(2021–22)"
        else:           return "Post-COVID\n(2022+)"

    df["era"] = df["start_year"].apply(era)
    era_avg = df.groupby("era")["avg_period_delay"].mean().reindex(
        ["Pre-COVID\n(2019–20)", "COVID\n(2020–21)", "Recovery\n(2021–22)", "Post-COVID\n(2022+)"]
    )

    colors = [BLUE, RED, AMBER, TEAL]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(era_avg.index, era_avg.values / 1e6, color=colors, width=0.5)
    ax.set_ylabel("Avg delay per period (millions of mins)")
    ax.set_title("COVID Impact on UK Rail Delays", fontweight="bold")
    ax.bar_label(bars, labels=[f"{v:.2f}M" for v in era_avg.values / 1e6],
                 padding=4, fontsize=10)
    fig.tight_layout()
    save(fig, "6_covid_impact.png")


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("UK Train Delay Analytics — Generating Charts")
    print("=" * 55)
    conn = get_conn()
    print("\n[1/6] Worst operators ...")
    chart_worst_operators(conn)
    print("[2/6] Delay trend over time ...")
    chart_delay_trend(conn)
    print("[3/6] Delay cause breakdown ...")
    chart_cause_breakdown(conn)
    print("[4/6] Best & worst stations ...")
    chart_stations(conn)
    print("[5/6] Operator punctuality trend ...")
    chart_operator_punctuality(conn)
    print("[6/6] COVID impact ...")
    chart_covid_impact(conn)
    conn.close()
    print("\n✓ All 6 charts saved to visuals/")
