# app/streamlit_app.py
# Interactive dashboard for UK Train Delay Analytics
# Run from the project root: streamlit run app/streamlit_app.py

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent.parent
DB_PATH   = ROOT / "database" / "delays.db"
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="UK Train Delay Analytics",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem 1.2rem; border: 1px solid #e0e0e0;
    }
    .metric-value { font-size: 2rem; font-weight: 600; color: #185FA5; }
    .metric-label { font-size: 0.85rem; color: #555; margin-top: 2px; }
    .section-header { font-size: 1.1rem; font-weight: 600;
                       color: #2c3e50; margin: 1.2rem 0 0.4rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_conn():
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(DB_PATH.as_posix(), check_same_thread=False)


@st.cache_data
def load_delay_causes():
    conn = get_conn()
    if conn is None: return pd.DataFrame()
    return pd.read_sql("SELECT * FROM delay_causes", conn)


@st.cache_data
def load_punctuality():
    conn = get_conn()
    if conn is None: return pd.DataFrame()
    return pd.read_sql("SELECT * FROM operator_punctuality WHERE pct_on_time_3m IS NOT NULL", conn)


@st.cache_data
def load_stations():
    conn = get_conn()
    if conn is None: return pd.DataFrame()
    return pd.read_sql("""
        SELECT station,
               AVG(time_to_3_pct) AS avg_time_to_3,
               AVG(cancellations_pct) AS avg_cancellations,
               SUM(scheduled_stops) AS total_scheduled
        FROM station_performance
        WHERE time_to_3_pct IS NOT NULL
        GROUP BY station
        HAVING total_scheduled >= 2000
    """, conn)


def load_model():
    rf_path  = MODEL_DIR / "rf_model.joblib"
    enc_path = MODEL_DIR / "operator_encoder.joblib"
    if rf_path.exists() and enc_path.exists():
        return joblib.load(rf_path), joblib.load(enc_path)
    return None, None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚂 UK Train Delays")
    st.markdown("**Powered by ORR official data**")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "📉 Delay Causes",
        "🏆 Operator Performance",
        "🚉 Station Rankings",
        "🤖 ML Predictor",
        "🗃 Raw Data",
    ])

conn = get_conn()
if conn is None:
    st.error("⚠️ Database not found. Run `python scripts/load_to_sql.py` first.")
    st.stop()

dc = load_delay_causes()
pq = load_punctuality()

# ── PAGE: Overview ────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("UK Train Delay Analytics Dashboard")
    st.caption("Official ORR data — Table 3184, 3133, 3130 · April 2019 to present")

    # Sidebar filters
    with st.sidebar:
        all_ops = sorted(dc["operator"].dropna().unique()) if not dc.empty else []
        sel_ops = st.multiselect("Filter operators", all_ops, default=[])

    dff = dc if not sel_ops else dc[dc["operator"].isin(sel_ops)]

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(dc['operator'].unique()):,}</div>
            <div class='metric-label'>Train Operators</div></div>""", unsafe_allow_html=True)
    with c2:
        total_d = dff["total_delay_mins"].sum() / 1e9
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{total_d:.1f}B</div>
            <div class='metric-label'>Total Delay Minutes</div></div>""", unsafe_allow_html=True)
    with c3:
        avg_pct = pq["pct_on_time_3m"].mean() if not pq.empty else 0
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{avg_pct:.1f}%</div>
            <div class='metric-label'>Avg On-Time (3 min)</div></div>""", unsafe_allow_html=True)
    with c4:
        years = dc["year_label"].nunique() if not dc.empty else 0
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{years}</div>
            <div class='metric-label'>Financial Years</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Delay over time
    if not dff.empty:
        trend = dff.groupby(["year_label", "period_num"])["total_delay_mins"].sum().reset_index()
        trend["x"] = trend["year_label"] + " P" + trend["period_num"].astype(int).astype(str).str.zfill(2)
        fig = px.area(trend, x="x", y="total_delay_mins",
                      title="Total Delay Minutes per Period",
                      labels={"total_delay_mins": "Delay (mins)", "x": "Period"},
                      color_discrete_sequence=["#185FA5"])
        fig.update_layout(xaxis_tickangle=-45, xaxis_nticks=20)
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if not dff.empty:
            top10 = dff.groupby("operator")["total_delay_mins"].sum().nlargest(10).reset_index()
            fig2 = px.bar(top10, x="total_delay_mins", y="operator", orientation="h",
                          title="Top 10 Operators — Total Delay",
                          labels={"total_delay_mins": "Minutes", "operator": ""},
                          color_discrete_sequence=["#A32D2D"])
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        if not pq.empty:
            latest_pq = pq.sort_values("year").groupby("operator").tail(4)
            avg_latest = latest_pq.groupby("operator")["pct_on_time_3m"].mean().reset_index()
            avg_latest = avg_latest.sort_values("pct_on_time_3m", ascending=True)
            color_vals = ["#A32D2D" if v < 80 else "#1D9E75" for v in avg_latest["pct_on_time_3m"]]
            fig3 = go.Figure(go.Bar(
                x=avg_latest["pct_on_time_3m"],
                y=avg_latest["operator"],
                orientation="h",
                marker_color=color_vals,
            ))
            fig3.add_vline(x=80, line_dash="dash", line_color="#BA7517",
                           annotation_text="80% target")
            fig3.update_layout(title="Recent On-Time % (last 4 quarters)",
                               xaxis_title="% On Time (3 min)",
                               yaxis_title="", xaxis_range=[60, 100])
            st.plotly_chart(fig3, use_container_width=True)


# ── PAGE: Delay Causes ────────────────────────────────────────────────────────
elif page == "📉 Delay Causes":
    st.title("Delay Causes Analysis")

    with st.sidebar:
        all_ops = sorted(dc["operator"].dropna().unique())
        sel_op  = st.selectbox("Operator", ["All"] + all_ops)
        all_years = sorted(dc["year_label"].dropna().unique())
        sel_year  = st.selectbox("Financial Year", ["All"] + all_years)

    dff = dc.copy()
    if sel_op != "All":   dff = dff[dff["operator"] == sel_op]
    if sel_year != "All": dff = dff[dff["year_label"] == sel_year]

    CAUSE_MAP = {
        "nr_external": "NR External",
        "nr_network_mgmt": "NR Network Mgmt",
        "nr_non_track_assets": "NR Non-Track Assets",
        "nr_severe_weather": "NR Severe Weather",
        "nr_track": "NR Track",
        "toc_self_fleet": "TOC Fleet",
        "toc_self_operations": "TOC Operations",
        "toc_self_stations": "TOC Stations",
        "toc_self_traincrew": "TOC Traincrew",
        "toc_self_other": "TOC Other",
        "toc_on_toc_fleet": "TOC-on-TOC Fleet",
        "toc_on_toc_operations": "TOC-on-TOC Ops",
        "toc_on_toc_stations": "TOC-on-TOC Stations",
        "toc_on_toc_traincrew": "TOC-on-TOC Traincrew",
        "toc_on_toc_other": "TOC-on-TOC Other",
    }

    cols_present = [c for c in CAUSE_MAP if c in dff.columns]
    cause_totals = dff[cols_present].sum().rename(CAUSE_MAP)
    cause_df = cause_totals.reset_index()
    cause_df.columns = ["cause", "delay_mins"]
    cause_df = cause_df.sort_values("delay_mins", ascending=False)
    cause_df["responsible"] = cause_df["cause"].apply(lambda x: "Network Rail" if x.startswith("NR") else "TOC")

    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig = px.bar(cause_df, x="delay_mins", y="cause", orientation="h",
                     color="responsible",
                     color_discrete_map={"Network Rail": "#185FA5", "TOC": "#BA7517"},
                     title="Delay Minutes by Cause",
                     labels={"delay_mins": "Minutes", "cause": ""})
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, legend_title="Responsible")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        nr_total  = dff["nr_responsible_mins"].sum()
        toc_total = dff["toc_responsible_mins"].sum()
        fig2 = go.Figure(go.Pie(
            labels=["Network Rail", "TOC (operators)"],
            values=[nr_total, toc_total],
            marker_colors=["#185FA5", "#BA7517"],
            hole=0.4,
        ))
        fig2.update_layout(title="NR vs TOC Responsibility Split")
        st.plotly_chart(fig2, use_container_width=True)

    # Year-on-year trend by cause category
    if sel_op != "All":
        cause_yr = dff.groupby("year_label")[cols_present].sum().reset_index()
        cause_yr_long = cause_yr.melt(id_vars="year_label", var_name="cause", value_name="mins")
        cause_yr_long["cause"] = cause_yr_long["cause"].map(CAUSE_MAP)
        fig3 = px.line(cause_yr_long, x="year_label", y="mins", color="cause",
                       title=f"Delay Cause Trend — {sel_op}",
                       labels={"mins": "Minutes", "year_label": "Financial Year"})
        st.plotly_chart(fig3, use_container_width=True)


# ── PAGE: Operator Performance ────────────────────────────────────────────────
elif page == "🏆 Operator Performance":
    st.title("Operator Punctuality Performance")

    with st.sidebar:
        all_ops = sorted(pq["operator"].dropna().unique()) if not pq.empty else []
        sel_ops = st.multiselect("Select operators", all_ops,
                                  default=all_ops[:6] if len(all_ops) >= 6 else all_ops)

    dff = pq[pq["operator"].isin(sel_ops)] if sel_ops else pq

    if not dff.empty:
        dff = dff.copy()
        dff["x"] = dff["year"].astype(str) + " " + dff["quarter"].astype(str)
        dff = dff.sort_values(["operator", "year", "quarter"])

        fig = px.line(dff, x="x", y="pct_on_time_3m", color="operator",
                      title="Quarterly Punctuality — % trains on time (within 3 mins)",
                      labels={"pct_on_time_3m": "% On Time (3 min)", "x": "Quarter"})
        fig.add_hline(y=80, line_dash="dash", line_color="red",
                      annotation_text="80% benchmark")
        fig.update_layout(xaxis_tickangle=-45, xaxis_nticks=20)
        st.plotly_chart(fig, use_container_width=True)

    # League table
    st.subheader("League Table — Latest 4 Quarters")
    if not pq.empty:
        recent = pq.sort_values("year").groupby("operator").tail(4)
        league = recent.groupby("operator").agg(
            avg_pct=("pct_on_time_3m", "mean"),
            min_pct=("pct_on_time_3m", "min"),
            max_pct=("pct_on_time_3m", "max"),
        ).reset_index().sort_values("avg_pct", ascending=False).reset_index(drop=True)
        league.index += 1
        league.columns = ["Operator", "Avg On-Time %", "Worst Quarter %", "Best Quarter %"]
        league["Avg On-Time %"] = league["Avg On-Time %"].round(1)
        league["Worst Quarter %"] = league["Worst Quarter %"].round(1)
        league["Best Quarter %"]  = league["Best Quarter %"].round(1)

        def color_row(val):
            if isinstance(val, float):
                if val >= 85: return "background-color: #d4edda"
                elif val < 75: return "background-color: #f8d7da"
            return ""

        st.dataframe(league.style.map(color_row, subset=["Avg On-Time %"]),
                     use_container_width=True, height=500)


# ── PAGE: Station Rankings ────────────────────────────────────────────────────
elif page == "🚉 Station Rankings":
    st.title("Station Performance Rankings")

    stations = load_stations()
    if stations.empty:
        st.warning("Station data not loaded. Make sure station_performance.csv was processed.")
    else:
        c1, c2 = st.columns([2, 1])
        with c2:
            min_stops = st.slider("Min scheduled stops", 2000, 50000, 5000, step=1000)

        dff = stations[stations["total_scheduled"] >= min_stops]

        col1, col2 = st.columns(2)
        with col1:
            best = dff.nlargest(20, "avg_time_to_3")
            fig = px.bar(best, x="avg_time_to_3", y="station", orientation="h",
                         title="🏅 20 Best Stations (Time-to-3 %)",
                         color="avg_time_to_3", color_continuous_scale="Greens",
                         labels={"avg_time_to_3": "% On Time", "station": ""})
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            worst = dff.nsmallest(20, "avg_time_to_3")
            fig2 = px.bar(worst, x="avg_time_to_3", y="station", orientation="h",
                          title="🔴 20 Worst Stations (Time-to-3 %)",
                          color="avg_time_to_3", color_continuous_scale="Reds_r",
                          labels={"avg_time_to_3": "% On Time", "station": ""})
            fig2.update_layout(yaxis={"categoryorder": "total ascending"},
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Cancellation Rate — Top 20 worst")
        worst_canc = dff.nlargest(20, "avg_cancellations")
        fig3 = px.bar(worst_canc, x="avg_cancellations", y="station", orientation="h",
                      color="avg_cancellations", color_continuous_scale="Oranges",
                      labels={"avg_cancellations": "Cancellations %", "station": ""},
                      title="Stations with Highest Cancellation Rates")
        fig3.update_layout(yaxis={"categoryorder": "total ascending"},
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)


# ── PAGE: ML Predictor ────────────────────────────────────────────────────────
elif page == "🤖 ML Predictor":
    st.title("ML — Underperformance Predictor")
    st.markdown(
        "This model predicts the **probability that an operator will fall below "
        "80% on-time punctuality next quarter**, using lag features, seasonality, "
        "and delay cause data. Trained on ORR data (2014–2025)."
    )

    model, le = load_model()
    if model is None:
        st.warning("Model not found. Run `python scripts/ml_model.py` to train it.")
        st.info("Once trained, the model file will appear in `models/rf_model.joblib`")
    else:
        # Show model results image if it exists
        model_chart = ROOT / "visuals" / "7_ml_model_results.png"
        if model_chart.exists():
            st.image(str(model_chart), caption="Model evaluation results", use_container_width=True)

        st.subheader("🎯 Live Prediction")
        st.markdown("Fill in the values below to predict the risk for next quarter.")

        all_ops = list(le.classes_)
        col1, col2, col3 = st.columns(3)
        with col1:
            sel_op  = st.selectbox("Operator", all_ops)
            quarter = st.selectbox("Current Quarter", ["Q1", "Q2", "Q3", "Q4"])
        with col2:
            lag1 = st.slider("Last quarter on-time %", 60.0, 100.0, 82.0, step=0.5)
            lag2 = st.slider("2 quarters ago on-time %", 60.0, 100.0, 81.0, step=0.5)
        with col3:
            lag4 = st.slider("Same Q last year on-time %", 60.0, 100.0, 83.0, step=0.5)
            roll4 = st.slider("4-quarter rolling average %", 60.0, 100.0, 82.0, step=0.5)

        col4, col5 = st.columns(2)
        with col4:
            total_delay = st.number_input("Total delay mins (this period)", 0, 5000000, 100000, step=10000)
            nr_delay    = st.number_input("NR responsible delay mins", 0, 3000000, 50000, step=10000)
        with col5:
            toc_delay   = st.number_input("TOC responsible delay mins", 0, 3000000, 50000, step=10000)
            weather_mins = st.number_input("Severe weather delay mins", 0, 500000, 5000, step=1000)

        if st.button("🔮 Predict Next Quarter Risk", type="primary"):
            nr_ratio = nr_delay / (total_delay + 1)
            wx_ratio = weather_mins / (total_delay + 1)
            tc_ratio = 0.1  # default traincrew ratio

            features = pd.DataFrame([{
                "lag1_pct": lag1, "lag2_pct": lag2, "lag4_pct": lag4,
                "roll4_pct": roll4, "trend_4q": lag1 - lag4,
                "is_q1": int(quarter == "Q1"), "is_q2": int(quarter == "Q2"),
                "is_q3": int(quarter == "Q3"), "is_q4": int(quarter == "Q4"),
                "total_delay_mins": total_delay,
                "nr_responsible_mins": nr_delay,
                "toc_responsible_mins": toc_delay,
                "nr_delay_ratio": nr_ratio,
                "severe_weather_ratio": wx_ratio,
                "traincrew_ratio": tc_ratio,
                "operator_enc": le.transform([sel_op])[0],
            }])

            prob = model.predict_proba(features)[0][1]
            pred = prob >= 0.5

            if pred:
                st.error(f"⚠️ **AT RISK** — predicted probability of underperformance: **{prob:.1%}**")
            else:
                st.success(f"✅ **ON TRACK** — predicted probability of underperformance: **{prob:.1%}**")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Risk Score (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#A32D2D" if pred else "#1D9E75"},
                    "steps": [
                        {"range": [0, 50], "color": "#d4edda"},
                        {"range": [50, 75], "color": "#fff3cd"},
                        {"range": [75, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "value": 50},
                },
            ))
            st.plotly_chart(fig, use_container_width=True)


# ── PAGE: Raw Data ────────────────────────────────────────────────────────────
elif page == "🗃 Raw Data":
    st.title("Raw Data Explorer")

    table = st.selectbox("Table", ["delay_causes", "operator_punctuality", "station_performance"])
    try:
        df_raw = pd.read_sql(f"SELECT * FROM {table} LIMIT 5000", conn)
        st.caption(f"{len(df_raw):,} rows (capped at 5,000 for display)")
        st.dataframe(df_raw, use_container_width=True, height=500)
        csv = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button(f"⬇️ Download {table}.csv", csv, f"{table}.csv", "text/csv")
    except Exception as e:
        st.error(f"Could not load table '{table}': {e}")
