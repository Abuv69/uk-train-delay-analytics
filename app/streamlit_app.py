# app/streamlit_app.py
# UK Train Delay Analytics dashboard.  Run: streamlit run app/streamlit_app.py

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.features import (FEATURES, THRESHOLD, build_feature_table,  # noqa: E402
                              next_quarter_start, quarter_label)

DB_PATH = ROOT / "database" / "delays.db"
MODEL_PATH = ROOT / "models" / "risk_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

BLUE, TEAL, AMBER, RED = "#4C8BF5", "#1D9E75", "#E3A008", "#E5484D"

st.set_page_config(page_title="UK Train Delay Analytics", page_icon="🚂",
                   layout="wide", initial_sidebar_state="expanded")

if not DB_PATH.exists():
    st.error("Database not found. Run `python scripts/load_to_sql.py` first.")
    st.stop()


# ── Data loading (cached) ─────────────────────────────────────────────────────
def _query(sql, **kw):
    with closing(sqlite3.connect(DB_PATH.as_posix())) as conn:
        return pd.read_sql(sql, conn, **kw)


@st.cache_data
def load_causes():
    d = _query("SELECT * FROM delay_causes")
    d["period_start"] = (pd.to_datetime(d["start_year"].astype(int).astype(str) + "-04-01")
                         + pd.to_timedelta((d["period_num"] - 1) * 28, unit="D"))
    return d


@st.cache_data
def load_punct():
    return _query("SELECT * FROM operator_punctuality WHERE pct_on_time_3m IS NOT NULL",
                  parse_dates=["qdate"]).sort_values(["operator", "qdate"])


@st.cache_data
def load_stations():
    d = _query("""SELECT period, station, scheduled_stops, time_to_3_pct, cancellations_pct,
                         year_label, period_num FROM station_performance""")
    start = d["year_label"].str.extract(r"Apr (\d{4})")[0].astype(float)
    d["period_start"] = (pd.to_datetime(start.astype("Int64").astype(str) + "-04-01", errors="coerce")
                         + pd.to_timedelta((d["period_num"] - 1) * 28, unit="D"))
    return d


@st.cache_data
def load_features():
    with closing(sqlite3.connect(DB_PATH.as_posix())) as conn:
        return build_feature_table(conn)


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def load_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


dc, pq, feats = load_causes(), load_punct(), load_features()
latest_q = pq["qdate"].max()


def wavg(g, col, w):
    m = g[col].notna() & g[w].notna() & (g[w] > 0)
    return (g.loc[m, col] * g.loc[m, w]).sum() / g.loc[m, w].sum() if m.any() else np.nan


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚂 UK Train Delays")
    st.caption("Official ORR data · Tables 3184, 3133, 3130")
    page = st.radio("Navigate", ["📊 Overview", "📉 Delay Causes", "🏆 Operator Performance",
                                 "🚉 Stations", "🤖 Risk Predictor", "🗃 Raw Data"])
    st.divider()
    st.caption(f"Punctuality data to **{quarter_label(latest_q)}**"
               f"{' (provisional)' if pq.loc[pq.qdate == latest_q, 'provisional'].any() else ''}")
    st.caption(f"Delay-cause data to **{dc['year_label'].max()}**")


# ═════════════════════════════ OVERVIEW ══════════════════════════════════════
if page == "📊 Overview":
    st.title("UK Train Delay Analytics")
    st.caption("How reliable is UK rail, who is struggling, and why?")

    cur = pq[pq.qdate == latest_q]
    prev_year = pq[pq.qdate == latest_q - pd.DateOffset(years=1)]
    nat_now = wavg(cur, "pct_on_time_3m", "recorded_stops")
    nat_prev = wavg(prev_year, "pct_on_time_3m", "recorded_stops")
    last_fy = dc["year_label"].max()
    fy_total = dc.loc[dc.year_label == last_fy, "total_delay_mins"].sum() / 1e6

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Operators (latest quarter)", f"{cur['operator'].nunique()}")
    c2.metric(f"On time (3 min), {quarter_label(latest_q)}", f"{nat_now:.1f}%",
              f"{nat_now - nat_prev:+.1f} pts vs a year earlier" if pd.notna(nat_prev) else None)
    c3.metric(f"Delay minutes, {last_fy}", f"{fy_total:.1f}M")
    c4.metric("Punctuality history", f"{pq.qdate.min().year}–{latest_q.year}")
    st.caption("On-time % is weighted by each operator's recorded stops (national picture, not a simple average).")

    ops = sorted(dc["operator"].unique())
    sel = st.multiselect("Filter delay charts by operator (optional)", ops)
    dff = dc[dc.operator.isin(sel)] if sel else dc

    trend = dff.groupby("period_start", as_index=False)["total_delay_mins"].sum()
    fig = px.area(trend, x="period_start", y="total_delay_mins",
                  title="Delay minutes per 4-week period",
                  labels={"period_start": "", "total_delay_mins": "Delay minutes"})
    fig.update_traces(line_color=BLUE)
    fig.add_vrect(x0="2020-03-23", x1="2020-07-04", fillcolor=AMBER, opacity=0.2, line_width=0,
                  annotation_text="COVID lockdown", annotation_position="top left")
    st.plotly_chart(fig)

    t1, t2 = st.tabs(["Operators by on-time %", "Operators by delay"])
    with t1:
        last4 = sorted(pq.qdate.unique())[-4:]
        r = (pq[pq.qdate.isin(last4)].groupby("operator", as_index=False)["pct_on_time_3m"].mean()
             .sort_values("pct_on_time_3m"))
        r["status"] = np.where(r.pct_on_time_3m < THRESHOLD, "Below 80%", "80% or better")
        fig = px.bar(r, x="pct_on_time_3m", y="operator", orientation="h", color="status",
                     color_discrete_map={"Below 80%": RED, "80% or better": TEAL},
                     title=f"Average on-time % (3 min), last 4 quarters to {quarter_label(latest_q)}",
                     labels={"pct_on_time_3m": "% on time", "operator": "", "status": ""}, height=620)
        fig.update_yaxes(dtick=1)
        fig.update_xaxes(range=[50, 100])
        fig.add_vline(x=THRESHOLD, line_dash="dash", line_color=AMBER)
        st.plotly_chart(fig)
    with t2:
        mode = st.radio("Measure", ["Total delay minutes", "Delay minutes per 1,000 stops (fairer)"],
                        horizontal=True)
        if mode.startswith("Total"):
            r = dff.groupby("operator", as_index=False)["total_delay_mins"].sum()
            r = r.nlargest(12, "total_delay_mins").sort_values("total_delay_mins")
            fig = px.bar(r, x="total_delay_mins", y="operator", orientation="h",
                         title="Top 12 operators by total delay minutes (all years)",
                         labels={"total_delay_mins": "Minutes", "operator": ""}, height=480)
            fig.update_traces(marker_color=RED)
            st.plotly_chart(fig)
            st.caption("⚠️ Big operators run more trains, so they naturally accumulate more delay minutes. "
                       "Use the per-1,000-stops view for a fair comparison.")
        else:
            f4 = feats[feats.qdate.isin(last4)].dropna(subset=["total_delay_mins", "recorded_stops"])
            r = (f4.groupby("operator").apply(
                lambda g: g.total_delay_mins.sum() / g.recorded_stops.sum() * 1000, include_groups=False)
                .rename("per_1k").reset_index().sort_values("per_1k").tail(12))
            fig = px.bar(r, x="per_1k", y="operator", orientation="h",
                         title="Delay minutes per 1,000 recorded stops, last 4 quarters",
                         labels={"per_1k": "Delay mins per 1,000 stops", "operator": ""}, height=480)
            fig.update_traces(marker_color=AMBER)
            st.plotly_chart(fig)
            st.caption("Long-distance and sleeper services make few stops per journey, so they look worse on "
                       "this measure than commuter operators.")


# ═════════════════════════════ DELAY CAUSES ══════════════════════════════════
elif page == "📉 Delay Causes":
    st.title("Delay Causes")
    f1, f2 = st.columns(2)
    sel_op = f1.selectbox("Operator", ["All"] + sorted(dc.operator.unique()))
    sel_year = f2.selectbox("Financial year", ["All"] + sorted(dc.year_label.unique()))
    d = dc.copy()
    if sel_op != "All":
        d = d[d.operator == sel_op]
    if sel_year != "All":
        d = d[d.year_label == sel_year]

    CAUSES = {"nr_external": "NR External", "nr_network_mgmt": "NR Network Mgmt",
              "nr_non_track_assets": "NR Non-Track Assets", "nr_severe_weather": "NR Severe Weather",
              "nr_track": "NR Track", "toc_self_fleet": "TOC Fleet",
              "toc_self_operations": "TOC Operations", "toc_self_stations": "TOC Stations",
              "toc_self_traincrew": "TOC Traincrew", "toc_self_other": "TOC Other",
              "toc_on_toc_fleet": "On-TOC Fleet", "toc_on_toc_operations": "On-TOC Ops",
              "toc_on_toc_stations": "On-TOC Stations", "toc_on_toc_traincrew": "On-TOC Traincrew",
              "toc_on_toc_other": "On-TOC Other"}
    tot = d[list(CAUSES)].sum().rename(CAUSES).reset_index()
    tot.columns = ["cause", "mins"]
    tot["who"] = np.where(tot.cause.str.startswith("NR"), "Network Rail", "Train operators")
    tot = tot.sort_values("mins")

    a, b = st.columns([1.6, 1])
    fig = px.bar(tot, x="mins", y="cause", orientation="h", color="who", height=520,
                 color_discrete_map={"Network Rail": BLUE, "Train operators": AMBER},
                 title="Delay minutes by cause", labels={"mins": "Minutes", "cause": "", "who": ""})
    fig.update_yaxes(dtick=1)
    a.plotly_chart(fig)
    share = d[["nr_responsible_mins", "toc_responsible_mins"]].sum()
    fig = go.Figure(go.Pie(labels=["Network Rail", "Train operators"],
                           values=[share.nr_responsible_mins, share.toc_responsible_mins],
                           hole=0.5, marker_colors=[BLUE, AMBER]))
    fig.update_layout(title="Who is responsible?", height=520)
    b.plotly_chart(fig)

    yr = (dc if sel_op == "All" else dc[dc.operator == sel_op]).groupby("year_label")[
        ["nr_responsible_mins", "toc_responsible_mins"]].sum().reset_index()
    fig = px.bar(yr, x="year_label", y=["nr_responsible_mins", "toc_responsible_mins"],
                 title=f"Delay minutes by year — {sel_op}",
                 labels={"value": "Minutes", "year_label": "", "variable": ""})
    fig.for_each_trace(lambda t: t.update(
        name="Network Rail" if t.name.startswith("nr") else "Train operators",
        marker_color=BLUE if t.name.startswith("nr") else AMBER))
    st.plotly_chart(fig)


# ═════════════════════════ OPERATOR PERFORMANCE ══════════════════════════════
elif page == "🏆 Operator Performance":
    st.title("Operator Performance")
    st.caption("% of trains arriving within 3 minutes of schedule, per quarter.")
    active = pq[pq.qdate == latest_q].nlargest(6, "recorded_stops")["operator"].tolist()
    ops = st.multiselect("Operators", sorted(pq.operator.unique()), default=active)
    d = pq[pq.operator.isin(ops)]

    t1, t2, t3 = st.tabs(["Trend", "Heatmap (last 12 quarters)", "League table"])
    with t1:
        fig = px.line(d, x="qdate", y="pct_on_time_3m", color="operator",
                      labels={"qdate": "", "pct_on_time_3m": "% on time (3 min)", "operator": ""})
        fig.add_hline(y=THRESHOLD, line_dash="dash", line_color=AMBER, annotation_text="80% line")
        st.plotly_chart(fig)
    with t2:
        q12 = sorted(pq.qdate.unique())[-12:]
        h = pq[pq.qdate.isin(q12)].pivot_table(index="operator", columns="qdate", values="pct_on_time_3m")
        h.columns = [quarter_label(c) for c in h.columns]
        h = h.loc[h.mean(axis=1).sort_values().index]
        fig = px.imshow(h, color_continuous_scale="RdYlGn", zmin=60, zmax=100, aspect="auto",
                        text_auto=".0f", height=max(420, 26 * len(h)),
                        labels={"color": "% on time"})
        fig.update_xaxes(side="top")
        st.plotly_chart(fig)
        st.caption("Notice the red columns: Oct–Dec is the worst quarter almost every year.")
    with t3:
        last4 = sorted(pq.qdate.unique())[-4:]
        lg = (pq[pq.qdate.isin(last4)].groupby("operator")["pct_on_time_3m"]
              .agg(avg="mean", worst="min", best="max", quarters="count").reset_index()
              .sort_values("avg", ascending=False))
        lg.columns = ["Operator", "Average %", "Worst quarter %", "Best quarter %", "Quarters"]
        st.dataframe(lg, hide_index=True, column_config={
            "Average %": st.column_config.ProgressColumn("Average %", min_value=50, max_value=100, format="%.1f"),
            "Worst quarter %": st.column_config.NumberColumn(format="%.1f"),
            "Best quarter %": st.column_config.NumberColumn(format="%.1f")})
        st.caption(f"Last 4 quarters to {quarter_label(latest_q)}. Operators with fewer than 4 quarters are new or restarted.")


# ═════════════════════════════ STATIONS ══════════════════════════════════════
elif page == "🚉 Stations":
    st.title("Station Performance")
    st0 = load_stations()
    st.caption(f"Data covers {st0.period_start.min():%b %Y} to {st0.period_start.max():%b %Y}. "
               "Averages are weighted by scheduled stops.")
    min_stops = st.slider("Minimum scheduled stops (filters out tiny stations)", 1000, 100000, 10000, step=1000)

    def agg(g):
        return pd.Series({"punctuality": wavg(g, "time_to_3_pct", "scheduled_stops"),
                          "cancellations": wavg(g, "cancellations_pct", "scheduled_stops"),
                          "stops": g["scheduled_stops"].sum()})
    s = st0.groupby("station").apply(agg, include_groups=False).reset_index()
    s = s[(s.stops >= min_stops)].dropna(subset=["punctuality"])

    t1, t2, t3 = st.tabs(["Punctuality", "Cancellations", "Station lookup"])
    with t1:
        c1, c2 = st.columns(2)
        best = s.nlargest(20, "punctuality").sort_values("punctuality")
        worst = s.nsmallest(20, "punctuality").sort_values("punctuality", ascending=False)
        for col, data, title, colour in [(c1, best, "20 best — % arriving within 3 min", TEAL),
                                         (c2, worst, "20 worst — % arriving within 3 min", RED)]:
            fig = px.bar(data, x="punctuality", y="station", orientation="h", title=title,
                         labels={"punctuality": "% on time", "station": ""}, height=640)
            fig.update_traces(marker_color=colour)
            fig.update_yaxes(dtick=1)
            col.plotly_chart(fig)
    with t2:
        c = s.dropna(subset=["cancellations"]).nlargest(20, "cancellations").sort_values("cancellations")
        fig = px.bar(c, x="cancellations", y="station", orientation="h", height=640,
                     title="20 highest cancellation rates", labels={"cancellations": "% cancelled", "station": ""})
        fig.update_traces(marker_color=AMBER)
        fig.update_yaxes(dtick=1)
        st.plotly_chart(fig)
    with t3:
        name = st.selectbox("Station", sorted(st0.station.unique()))
        one = st0[st0.station == name].dropna(subset=["time_to_3_pct"]).sort_values("period_start")
        if one.empty:
            st.info("No punctuality data for this station.")
        else:
            fig = px.line(one, x="period_start", y="time_to_3_pct", markers=True,
                          title=f"{name} — % arriving within 3 min by 4-week period",
                          labels={"period_start": "", "time_to_3_pct": "% on time"})
            fig.add_hline(y=THRESHOLD, line_dash="dash", line_color=AMBER)
            st.plotly_chart(fig)


# ═════════════════════════════ RISK PREDICTOR ════════════════════════════════
elif page == "🤖 Risk Predictor":
    st.title("Risk Predictor")
    bundle, metrics = load_bundle(), load_metrics()
    if bundle is None:
        st.warning("Model not found. Run `python scripts/ml_model.py`.")
        st.stop()
    model, medians = bundle["model"], bundle["medians"]
    next_q = next_quarter_start(latest_q)
    st.markdown(f"Will an operator's on-time % fall **below {THRESHOLD:.0f}%** in "
                f"**{quarter_label(next_q)}**? Everything below is filled in automatically from the "
                f"latest real data (**{quarter_label(latest_q)}**) — you can then test 'what-if' changes.")

    def predict(row_df):
        X = row_df[FEATURES].astype(float).fillna(pd.Series(medians))
        return model.predict_proba(X)[:, 1]

    latest = feats[feats.qdate == latest_q].dropna(subset=["cur_pct", "prev1_pct", "prev2_pct", "seasonal_pct"]).copy()
    latest["risk"] = predict(latest)

    tab_op, tab_all, tab_model = st.tabs(["Operator forecast", "All operators", "How good is it?"])

    with tab_all:
        t = latest[["operator", "cur_pct", "seasonal_pct", "risk"]].sort_values("risk", ascending=False)
        t.columns = ["Operator", f"On time {quarter_label(latest_q)} %", "Same quarter last year %", "Risk next quarter"]
        t["Risk next quarter"] = (t["Risk next quarter"] * 100).round(0)
        st.dataframe(t, hide_index=True, column_config={
            "Risk next quarter": st.column_config.ProgressColumn("Risk next quarter", min_value=0, max_value=100, format="%d%%"),
            f"On time {quarter_label(latest_q)} %": st.column_config.NumberColumn(format="%.1f"),
            "Same quarter last year %": st.column_config.NumberColumn(format="%.1f")})
        st.caption("Operators without a full year of history are excluded.")

    with tab_op:
        ordered = latest.sort_values("risk", ascending=False)["operator"].tolist()
        op = st.selectbox("Operator", ordered)
        row = latest[latest.operator == op].iloc[0]
        base = float(row["risk"])

        g1, g2 = st.columns([1, 1.4])
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=base * 100, number={"suffix": "%"},
            title={"text": f"Risk of <{THRESHOLD:.0f}% in {quarter_label(next_q)}"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": RED if base >= 0.5 else TEAL},
                   "steps": [{"range": [0, 40], "color": "rgba(29,158,117,0.25)"},
                             {"range": [40, 60], "color": "rgba(227,160,8,0.25)"},
                             {"range": [60, 100], "color": "rgba(229,72,77,0.25)"}]}))
        fig.update_layout(height=320)
        g1.plotly_chart(fig)
        g1.markdown(f"**{'⚠️ At risk' if base >= 0.5 else '✅ Likely on track'}** — "
                    f"{op} ran at **{row.cur_pct:.1f}%** in {quarter_label(latest_q)}.")

        hist = pq[(pq.operator == op)].tail(16)
        fig = px.line(hist, x="qdate", y="pct_on_time_3m", markers=True,
                      title=f"{op}: recent quarters", labels={"qdate": "", "pct_on_time_3m": "% on time"})
        fig.add_hline(y=THRESHOLD, line_dash="dash", line_color=AMBER)
        fig.update_layout(height=320)
        g2.plotly_chart(fig)

        with st.expander("🎛 What-if: change the inputs and watch the risk move"):
            st.caption("Sliders start at the real values. The model recalculates instantly.")
            k = op.replace(" ", "_")
            s1, s2 = st.columns(2)
            cur_pct = s1.slider("On time this quarter %", 40.0, 100.0, float(round(row.cur_pct, 1)), 0.1, key=f"cur_{k}")
            prev1 = s1.slider("On time last quarter %", 40.0, 100.0, float(round(row.prev1_pct, 1)), 0.1, key=f"p1_{k}")
            prev2 = s1.slider("On time 2 quarters ago %", 40.0, 100.0, float(round(row.prev2_pct, 1)), 0.1, key=f"p2_{k}")
            seas = s2.slider("Same quarter last year %", 40.0, 100.0, float(round(row.seasonal_pct, 1)), 0.1, key=f"se_{k}")
            dmax = float(np.nanmax(feats["delay_per_1k_stops"]) * 1.1)
            dpk = row.delay_per_1k_stops if pd.notna(row.delay_per_1k_stops) else medians["delay_per_1k_stops"]
            delay = s2.slider("Delay mins per 1,000 stops", 0.0, round(dmax, 0), float(min(round(dpk, 0), dmax)), 1.0, key=f"d_{k}")
            nr = s2.slider("Share of delay from Network Rail", 0.0, 1.0,
                           float(round(row.nr_share if pd.notna(row.nr_share) else medians["nr_share"], 2)), 0.01, key=f"nr_{k}")
            wx = s2.slider("Share of delay from severe weather", 0.0, 1.0,
                           float(round(row.weather_share if pd.notna(row.weather_share) else medians["weather_share"], 2)), 0.01, key=f"wx_{k}")

            w = row.copy()
            w["cur_pct"], w["prev1_pct"], w["prev2_pct"], w["seasonal_pct"] = cur_pct, prev1, prev2, seas
            w["roll4_pct"] = np.mean([cur_pct, prev1, prev2, seas])
            w["delta1"] = cur_pct - prev1
            w["delay_per_1k_stops"], w["nr_share"], w["weather_share"] = delay, nr, wx
            risk_wi = float(predict(pd.DataFrame([w]))[0])
            m1, m2 = st.columns(2)
            m1.metric("Risk with real data", f"{base:.0%}")
            m2.metric("Risk with your changes", f"{risk_wi:.0%}", f"{(risk_wi - base) * 100:+.0f} pts", delta_color="inverse")

    with tab_model:
        if metrics:
            st.markdown(f"**Task:** {metrics['task']}.  \n"
                        f"**Tested on future quarters only:** {metrics['test_from']} → {metrics['test_to']} "
                        f"({metrics['test_rows']} operator-quarters, trained on data up to {metrics['train_until']}).")
            res = pd.DataFrame(metrics["results"]).T.reset_index()
            res.columns = ["Model", "ROC AUC", "F1", "Precision", "Recall", "Accuracy"]
            st.dataframe(res, hide_index=True)
            st.info("**Honest reading:** UK rail is strongly seasonal — Oct–Dec is reliably the worst quarter — so "
                    "simple baselines like 'same quarter last year' already score very high. The model matches or "
                    "slightly beats them. It's a sound early-warning signal, not a crystal ball.")
            coef = pd.Series(metrics["coefficients"]).sort_values()
            fig = px.bar(coef, orientation="h", height=460,
                         title="What drives the prediction (standardised coefficients; + = more risk)",
                         labels={"value": "Coefficient", "index": ""})
            fig.update_layout(showlegend=False)
            fig.update_traces(marker_color=BLUE)
            st.plotly_chart(fig)


# ═════════════════════════════ RAW DATA ══════════════════════════════════════
elif page == "🗃 Raw Data":
    st.title("Raw Data Explorer")
    table = st.selectbox("Table", ["delay_causes", "operator_punctuality", "station_performance"])
    d = _query(f"SELECT * FROM {table} LIMIT 5000")
    st.caption(f"Showing up to 5,000 rows · {len(d.columns)} columns")
    st.dataframe(d)
    st.download_button(f"⬇️ Download {table}.csv (shown rows)", d.to_csv(index=False).encode("utf-8"),
                       f"{table}.csv", "text/csv")
