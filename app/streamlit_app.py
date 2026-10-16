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

AUTHOR = "Abubakar Shaikh"
GITHUB_URL = "https://github.com/abuv69/uk-train-delay-analytics"
LINKEDIN_URL = "https://www.linkedin.com/in/mohammedabubakarshaikh"
ORR_URL = "https://dataportal.orr.gov.uk/"

PAGES = ["🏠 Overview", "📉 Delay Causes", "🏆 Operator Performance", "🚉 Stations",
         "🤖 Risk Predictor", "ℹ️ About & Data", "🗃 Raw Data"]

st.set_page_config(page_title="UK Train Delay Analytics", page_icon="🚂",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.hero {padding: 1.3rem 1.5rem; border-radius: 14px; margin-bottom: 1rem;
       background: linear-gradient(135deg, rgba(76,139,245,.20), rgba(29,158,117,.14));
       border: 1px solid rgba(128,128,128,.28);}
.hero h1 {margin: 0 0 .35rem 0; font-size: 2.1rem; line-height: 1.2;}
.hero p {margin: 0 0 .6rem 0; opacity: .9; font-size: 1.02rem;}
.chip {display: inline-block; padding: .15rem .65rem; border-radius: 999px; margin: 0 .35rem .3rem 0;
       background: rgba(128,128,128,.20); font-size: .8rem;}
.note {padding: .7rem .95rem; border-left: 4px solid #4C8BF5; background: rgba(76,139,245,.10);
       border-radius: 6px; font-size: .93rem; margin: .3rem 0 1rem;}
.warn {border-left-color: #E3A008; background: rgba(227,160,8,.10);}
</style>
""", unsafe_allow_html=True)


def note(html, kind=""):
    st.markdown(f'<div class="note {kind}">{html}</div>', unsafe_allow_html=True)


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
    st.caption("How reliable is UK rail? Official data, explored interactively.")
    page = st.radio("Go to", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("**Data freshness**")
    st.caption(f"Punctuality: up to **{quarter_label(latest_q)}**"
               f"{' (provisional)' if pq.loc[pq.qdate == latest_q, 'provisional'].any() else ''}")
    st.caption(f"Delay causes: up to **{dc['year_label'].max()}** (financial year)")
    st.divider()
    st.link_button("⭐ View code on GitHub", GITHUB_URL)
    st.caption(f"Built by {AUTHOR}")


def page_intro(text):
    note(f"<b>What this page shows:</b> {text}")


# ═════════════════════════════ OVERVIEW ══════════════════════════════════════
if page == "🏠 Overview":
    cur = pq[pq.qdate == latest_q]
    prev_year = pq[pq.qdate == latest_q - pd.DateOffset(years=1)]
    nat_now = wavg(cur, "pct_on_time_3m", "recorded_stops")
    nat_prev = wavg(prev_year, "pct_on_time_3m", "recorded_stops")
    last_fy = dc["year_label"].max()
    fy_total = dc.loc[dc.year_label == last_fy, "total_delay_mins"].sum() / 1e6

    st.markdown(f"""
<div class="hero">
  <h1>🚂 UK Train Delay Analytics</h1>
  <p>An interactive look at how punctual Britain's train operators are, what causes delays,
  and which operators are at risk of falling short next quarter — built entirely on official
  Office of Rail and Road (ORR) data.</p>
  <span class="chip">{cur['operator'].nunique()} operators</span><span class="chip">{pq.qdate.min().year}–{latest_q.year} punctuality</span>
  <span class="chip">100,000+ station records</span>
  <span class="chip">Official ORR open data</span>
</div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Operators running", f"{cur['operator'].nunique()}",
              help=f"Operators with punctuality data in {quarter_label(latest_q)}.")
    c2.metric(f"Trains on time, {quarter_label(latest_q)}", f"{nat_now:.1f}%",
              f"{nat_now - nat_prev:+.1f} pts vs {quarter_label(latest_q - pd.DateOffset(years=1))}"
              if pd.notna(nat_prev) else None,
              help="Share of trains arriving within 3 minutes of timetable, across all operators "
                   "(weighted by how many stops each operator makes).")
    c3.metric(f"Delay minutes, {last_fy}", f"{fy_total:.1f}M",
              help="Total minutes of delay attributed to a cause, across all operators, "
                   "for the latest full financial year (April–March).")
    c4.metric("Operators below 80% on time", f"{int((cur.pct_on_time_3m < THRESHOLD).sum())} of {len(cur)}",
              help=f"In {quarter_label(latest_q)}. 80% is the line the Risk Predictor uses.")

    # ---- Key takeaways (computed live from the data) ----
    st.subheader("Key takeaways")
    best = cur.nlargest(1, "pct_on_time_3m").iloc[0]
    worst = cur.nsmallest(1, "pct_on_time_3m").iloc[0]
    recent = pq[pq.qdate >= "2022-04-01"]
    oct_dec = recent.loc[recent.qdate.dt.month == 10, "pct_on_time_3m"].mean()
    other_q = recent.loc[recent.qdate.dt.month != 10, "pct_on_time_3m"].mean()
    nr_share = dc.nr_responsible_mins.sum() / dc.total_delay_mins.sum()
    t1, t2, t3, t4 = st.columns(4)
    with t1.container(border=True):
        st.markdown(f"**🥇 Best vs weakest**\n\n{best.operator} led at **{best.pct_on_time_3m:.1f}%**; "
                    f"{worst.operator} trailed at **{worst.pct_on_time_3m:.1f}%** ({quarter_label(latest_q)}).")
    with t2.container(border=True):
        st.markdown(f"**❄️ Autumn is the weak spot**\n\nOct–Dec averages **{oct_dec:.0f}%** on time versus "
                    f"**{other_q:.0f}%** in other quarters (since Apr 2022).")
    with t3.container(border=True):
        st.markdown(f"**🛤 Infrastructure drives delay**\n\nNetwork Rail is responsible for **{nr_share:.0%}** of "
                    f"delay minutes; train operators for the rest.")
    with t4.container(border=True):
        st.markdown("**🦠 COVID reset the picture**\n\nDelay minutes more than halved in 2020-21 as services "
                    "were cut, and have since climbed back to pre-COVID levels.")

    # ---- Guide ----
    with st.expander("🧭 What's on each page?", expanded=True):
        g = st.columns(3)
        guide = [
            ("📉 Delay Causes", "Why trains are delayed, and whether Network Rail or the operators are responsible."),
            ("🏆 Operator Performance", "Quarterly on-time trends, a colour-coded heatmap and a league table."),
            ("🚉 Stations", "Best and worst stations for punctuality and cancellations, plus a station lookup."),
            ("🤖 Risk Predictor", "Which operators are likely to fall below 80% on time next quarter — with a what-if panel."),
            ("ℹ️ About & Data", "Where the data comes from, how the pipeline and model work, and their limits."),
            ("🗃 Raw Data", "Browse and download the tables behind every chart."),
        ]
        for i, (t, d) in enumerate(guide):
            with g[i % 3].container(border=True):
                st.markdown(f"**{t}**\n\n{d}")

    with st.expander("📖 Glossary — plain-English definitions"):
        st.markdown("""
- **On time (3 min):** a train counts as on time if it arrives within **3 minutes** of its timetabled time at a recorded station stop.
- **Delay minutes:** total minutes of delay that ORR attributes to a cause (e.g. a signalling failure or a train fault).
- **Network Rail (NR):** the organisation that owns and maintains tracks, signals and stations. **TOC:** a *train operating company* such as Northern or GWR.
- **On-TOC delay:** delay caused by *another* operator's train getting in the way.
- **Quarter:** Apr–Jun, Jul–Sep, Oct–Dec, Jan–Mar. **Financial year:** April to March (e.g. 2025-26).
- **Period:** ORR's 4-week reporting block; there are 13 per financial year.
- **Recorded stops:** how many station stops were timed — used to compare operators fairly.
""")

    st.divider()
    ops = sorted(dc["operator"].unique())
    st.subheader("Explore the trend")
    sel = st.multiselect("Narrow the delay charts to specific operators (optional)", ops,
                         help="Leave empty to include all operators.")
    dff = dc[dc.operator.isin(sel)] if sel else dc

    trend = dff.groupby("period_start", as_index=False)["total_delay_mins"].sum()
    fig = px.area(trend, x="period_start", y="total_delay_mins",
                  title="Delay minutes per 4-week period",
                  labels={"period_start": "", "total_delay_mins": "Delay minutes"})
    fig.update_traces(line_color=BLUE)
    fig.add_vrect(x0="2020-03-23", x1="2020-07-04", fillcolor=AMBER, opacity=0.2, line_width=0,
                  annotation_text="COVID lockdown", annotation_position="top left")
    st.plotly_chart(fig)

    t1, t2 = st.tabs(["Who is most punctual?", "Who has the most delay?"])
    with t1:
        last4 = sorted(pq.qdate.unique())[-4:]
        r = (pq[pq.qdate.isin(last4)].groupby("operator", as_index=False)["pct_on_time_3m"].mean()
             .sort_values("pct_on_time_3m"))
        r["status"] = np.where(r.pct_on_time_3m < THRESHOLD, "Below 80%", "80% or better")
        fig = px.bar(r, x="pct_on_time_3m", y="operator", orientation="h", color="status",
                     color_discrete_map={"Below 80%": RED, "80% or better": TEAL},
                     title=f"Average % of trains on time (within 3 min) — {quarter_label(last4[0])} to {quarter_label(last4[-1])}",
                     labels={"pct_on_time_3m": "% of trains on time", "operator": "", "status": ""}, height=620)
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
                         title="Top 12 operators by total delay minutes (Apr 2019 – Mar 2026)",
                         labels={"total_delay_mins": "Delay minutes", "operator": ""}, height=480)
            fig.update_traces(marker_color=RED)
            st.plotly_chart(fig)
            note("Big operators run more trains, so they naturally accumulate more delay minutes. "
                 "Switch to <b>per 1,000 stops</b> for a fairer comparison.", "warn")
        else:
            f4 = feats[feats.qdate.isin(last4)].dropna(subset=["total_delay_mins", "recorded_stops"])
            r = (f4.groupby("operator").apply(
                lambda g: g.total_delay_mins.sum() / g.recorded_stops.sum() * 1000, include_groups=False)
                .rename("per_1k").reset_index().sort_values("per_1k").tail(12))
            fig = px.bar(r, x="per_1k", y="operator", orientation="h",
                         title=f"Delay minutes per 1,000 recorded stops — {quarter_label(last4[0])} to {quarter_label(last4[-1])}",
                         labels={"per_1k": "Delay minutes per 1,000 stops", "operator": ""}, height=480)
            fig.update_traces(marker_color=AMBER)
            st.plotly_chart(fig)
            note("Long-distance and sleeper services make few stops per journey, so they look worse on "
                 "this measure than commuter operators.", "warn")


# ═════════════════════════════ DELAY CAUSES ══════════════════════════════════
elif page == "📉 Delay Causes":
    st.title("📉 Delay Causes")
    page_intro("Where delay minutes come from. <b>NR</b> = Network Rail (tracks, signals, weather, external events); "
               "<b>TOC / On-TOC</b> = train operators themselves, or another operator's train blocking the way.")
    f1, f2 = st.columns(2)
    sel_op = f1.selectbox("Operator", ["All"] + sorted(dc.operator.unique()))
    sel_year = f2.selectbox("Financial year (April–March)", ["All"] + sorted(dc.year_label.unique()))
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
    top = tot.iloc[-1]
    st.markdown(f"**Biggest single cause for this selection:** {top.cause} — "
                f"{top.mins / 1e6:.1f}M delay minutes.")

    a, b = st.columns([1.6, 1])
    fig = px.bar(tot, x="mins", y="cause", orientation="h", color="who", height=520,
                 color_discrete_map={"Network Rail": BLUE, "Train operators": AMBER},
                 title="Delay minutes by cause", labels={"mins": "Delay minutes", "cause": "", "who": ""})
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
                 title=f"Delay minutes by financial year — {sel_op}",
                 labels={"value": "Delay minutes", "year_label": "Financial year", "variable": ""})
    fig.for_each_trace(lambda t: t.update(
        name="Network Rail" if t.name.startswith("nr") else "Train operators",
        marker_color=BLUE if t.name.startswith("nr") else AMBER))
    st.plotly_chart(fig)
    note("2020-21 is low because services were heavily cut during COVID-19 lockdowns, not because the railway got better.")


# ═════════════════════════ OPERATOR PERFORMANCE ══════════════════════════════
elif page == "🏆 Operator Performance":
    st.title("🏆 Operator Performance")
    page_intro("The percentage of trains arriving within <b>3 minutes</b> of their timetabled time, each quarter. "
               "Higher is better; the dashed line marks 80%.")
    active = pq[pq.qdate == latest_q].nlargest(6, "recorded_stops")["operator"].tolist()
    ops = st.multiselect("Operators to compare", sorted(pq.operator.unique()), default=active,
                         help="Defaults to the six operators with the most timed stops.")
    d = pq[pq.operator.isin(ops)]

    t1, t2, t3 = st.tabs(["📈 Trend", "🟩 Heatmap (last 12 quarters)", "🏁 League table"])
    with t1:
        fig = px.line(d, x="qdate", y="pct_on_time_3m", color="operator",
                      labels={"qdate": "Quarter start", "pct_on_time_3m": "% of trains on time (3 min)", "operator": ""})
        fig.add_hline(y=THRESHOLD, line_dash="dash", line_color=AMBER, annotation_text="80% line")
        st.plotly_chart(fig)
        note("Look for the repeating dip: Oct–Dec is the weakest quarter almost every year. "
             "The spike in 2020 is the COVID lockdown, when far fewer trains ran.")
    with t2:
        q12 = sorted(pq.qdate.unique())[-12:]
        h = pq[pq.qdate.isin(q12)].pivot_table(index="operator", columns="qdate", values="pct_on_time_3m")
        h.columns = [quarter_label(c) for c in h.columns]
        h = h.loc[h.mean(axis=1).sort_values().index]
        fig = px.imshow(h, color_continuous_scale="RdYlGn", zmin=60, zmax=100, aspect="auto",
                        text_auto=".0f", height=max(420, 26 * len(h)), labels={"color": "% on time"})
        fig.update_xaxes(side="top")
        st.plotly_chart(fig)
        note("Each cell is % of trains on time. Red = struggling, green = strong. Operators are sorted worst (top) to best (bottom).")
    with t3:
        last4 = sorted(pq.qdate.unique())[-4:]
        lg = (pq[pq.qdate.isin(last4)].groupby("operator")["pct_on_time_3m"]
              .agg(avg="mean", worst="min", best="max", quarters="count").reset_index()
              .sort_values("avg", ascending=False))
        lg.columns = ["Operator", "Average %", "Worst quarter %", "Best quarter %", "Quarters of data"]
        st.dataframe(lg, hide_index=True, column_config={
            "Average %": st.column_config.ProgressColumn("Average %", min_value=50, max_value=100, format="%.1f"),
            "Worst quarter %": st.column_config.NumberColumn(format="%.1f"),
            "Best quarter %": st.column_config.NumberColumn(format="%.1f")})
        st.caption(f"Covers the last four quarters: {quarter_label(last4[0])} to {quarter_label(last4[-1])}. "
                   "Operators with fewer than 4 quarters are new or restarted.")


# ═════════════════════════════ STATIONS ══════════════════════════════════════
elif page == "🚉 Stations":
    st.title("🚉 Stations")
    st0 = load_stations()
    page_intro(f"How punctual individual stations are, from {st0.period_start.min():%b %Y} to {st0.period_start.max():%b %Y}. "
               "<b>Punctuality</b> = % of trains arriving within 3 minutes of timetable; "
               "<b>cancellations</b> = % of scheduled trains that didn't run. Averages are weighted by scheduled stops.")
    min_stops = st.slider("Minimum scheduled stops (hides tiny stations whose averages are unreliable)",
                          1000, 100000, 10000, step=1000)

    def agg(g):
        return pd.Series({"punctuality": wavg(g, "time_to_3_pct", "scheduled_stops"),
                          "cancellations": wavg(g, "cancellations_pct", "scheduled_stops"),
                          "stops": g["scheduled_stops"].sum()})
    s = st0.groupby("station").apply(agg, include_groups=False).reset_index()
    s = s[(s.stops >= min_stops)].dropna(subset=["punctuality"])
    st.caption(f"{len(s):,} stations meet this filter.")

    t1, t2, t3 = st.tabs(["⏱ Punctuality", "🚫 Cancellations", "🔎 Station lookup"])
    with t1:
        c1, c2 = st.columns(2)
        best = s.nlargest(20, "punctuality").sort_values("punctuality")
        worst = s.nsmallest(20, "punctuality").sort_values("punctuality", ascending=False)
        for col, data, title, colour in [(c1, best, "20 most punctual stations", TEAL),
                                         (c2, worst, "20 least punctual stations", RED)]:
            fig = px.bar(data, x="punctuality", y="station", orientation="h", title=title,
                         labels={"punctuality": "% of trains on time (3 min)", "station": ""}, height=640)
            fig.update_traces(marker_color=colour)
            fig.update_yaxes(dtick=1)
            col.plotly_chart(fig)
        note("Many of the least punctual stations are on rural or single-track lines, where one late train delays the next.")
    with t2:
        c = s.dropna(subset=["cancellations"]).nlargest(20, "cancellations").sort_values("cancellations")
        fig = px.bar(c, x="cancellations", y="station", orientation="h", height=640,
                     title="20 stations with the highest cancellation rates",
                     labels={"cancellations": "% of scheduled trains cancelled", "station": ""})
        fig.update_traces(marker_color=AMBER)
        fig.update_yaxes(dtick=1)
        st.plotly_chart(fig)
    with t3:
        name = st.selectbox("Pick a station", sorted(st0.station.unique()))
        one = st0[st0.station == name].dropna(subset=["time_to_3_pct"]).sort_values("period_start")
        if one.empty:
            st.info("No punctuality data for this station.")
        else:
            fig = px.line(one, x="period_start", y="time_to_3_pct", markers=True,
                          title=f"{name}: % of trains on time by 4-week period",
                          labels={"period_start": "", "time_to_3_pct": "% on time (3 min)"})
            fig.add_hline(y=THRESHOLD, line_dash="dash", line_color=AMBER)
            st.plotly_chart(fig)


# ═════════════════════════════ RISK PREDICTOR ════════════════════════════════
elif page == "🤖 Risk Predictor":
    st.title("🤖 Risk Predictor")
    bundle, metrics = load_bundle(), load_metrics()
    if bundle is None:
        st.warning("Model not found. Run `python scripts/ml_model.py`.")
        st.stop()
    model, medians = bundle["model"], bundle["medians"]
    next_q = next_quarter_start(latest_q)
    L_latest, L_next = quarter_label(latest_q), quarter_label(next_q)
    note(f"<b>What this does:</b> using each operator's punctuality up to <b>{L_latest}</b>, a statistical model estimates "
         f"the chance that fewer than <b>80%</b> of its trains will be on time in <b>{L_next}</b>. "
         "Everything is pre-filled from real data — pick an operator, or open the what-if panel to experiment.")

    def predict(row_df):
        X = row_df[FEATURES].astype(float).fillna(pd.Series(medians))
        return model.predict_proba(X)[:, 1]

    latest = feats[feats.qdate == latest_q].dropna(subset=["cur_pct", "prev1_pct", "prev2_pct", "seasonal_pct"]).copy()
    latest["risk"] = predict(latest)

    def band(p):
        return ("High risk", RED) if p >= 0.6 else (("Uncertain", AMBER) if p >= 0.4 else ("Low risk", TEAL))

    tab_op, tab_all, tab_model = st.tabs(["🎯 Operator forecast", "📋 All operators", "🧪 How good is it?"])

    # ---- all operators ----
    with tab_all:
        t = latest[["operator", "cur_pct", "seasonal_pct", "risk"]].sort_values("risk", ascending=False)
        t.columns = ["Operator", f"On time {L_latest} (%)",
                     f"On time {quarter_label(next_q - pd.DateOffset(years=1))} (%)", f"Risk of <80% in {L_next}"]
        t.iloc[:, 3] = (t.iloc[:, 3] * 100).round(0)
        st.dataframe(t, hide_index=True, column_config={
            t.columns[3]: st.column_config.ProgressColumn(t.columns[3], min_value=0, max_value=100, format="%d%%"),
            t.columns[1]: st.column_config.NumberColumn(format="%.1f"),
            t.columns[2]: st.column_config.NumberColumn(format="%.1f")})
        st.caption("Sorted from highest to lowest risk. Operators without a full year of history are left out.")

    # ---- single operator ----
    with tab_op:
        ordered = latest.sort_values("risk", ascending=False)["operator"].tolist()
        op = st.selectbox("Choose an operator (listed highest risk first)", ordered)
        row = latest[latest.operator == op].iloc[0]
        base = float(row["risk"])
        qc = row["qdate"]
        L_cur = quarter_label(qc)
        L_p1 = quarter_label(qc - pd.DateOffset(months=3))
        L_p2 = quarter_label(qc - pd.DateOffset(months=6))
        L_seas = quarter_label(qc - pd.DateOffset(months=9))
        label, colour = band(base)

        g1, g2 = st.columns([1, 1.4])
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=base * 100, number={"suffix": "%"},
            title={"text": f"Chance of <80% on time in {L_next}"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": colour},
                   "steps": [{"range": [0, 40], "color": "rgba(29,158,117,0.25)"},
                             {"range": [40, 60], "color": "rgba(227,160,8,0.25)"},
                             {"range": [60, 100], "color": "rgba(229,72,77,0.25)"}]}))
        fig.update_layout(height=300, margin=dict(t=70, b=10))
        g1.plotly_chart(fig)
        g1.markdown(f"**{label}.** {op} ran at **{row.cur_pct:.1f}%** on time in {L_cur}. "
                    f"Same quarter a year earlier ({L_seas}): **{row.seasonal_pct:.1f}%**.")
        g1.caption("0–40% low · 40–60% uncertain · 60–100% high. Treat this as a risk *ranking*, not an exact probability.")

        hist = pq[(pq.operator == op)].tail(16)
        fig = px.line(hist, x="qdate", y="pct_on_time_3m", markers=True,
                      title=f"{op}: % of trains on time, last 16 quarters",
                      labels={"qdate": "Quarter start", "pct_on_time_3m": "% on time (3 min)"})
        fig.add_hline(y=THRESHOLD, line_dash="dash", line_color=AMBER, annotation_text="80% line")
        fig.update_layout(height=300)
        g2.plotly_chart(fig)

        with st.expander("🎛 What-if: change the numbers and watch the risk move", expanded=True):
            st.caption("The sliders start at this operator's real figures. Move any of them to test a scenario, "
                       "then press reset to return to the real values.")
            k = op.replace(" ", "_")
            keys = {n: f"{n}_{k}" for n in ["cur", "p1", "p2", "seas", "delay", "nr", "wx"]}
            dmax = float(round(np.nanpercentile(feats["delay_per_1k_stops"].dropna(), 99.5) * 1.2))
            dpk = row.delay_per_1k_stops if pd.notna(row.delay_per_1k_stops) else medians["delay_per_1k_stops"]
            nr0 = row.nr_share if pd.notna(row.nr_share) else medians["nr_share"]
            wx0 = row.weather_share if pd.notna(row.weather_share) else medians["weather_share"]
            defaults = {
                keys["cur"]: float(np.clip(round(row.cur_pct, 1), 40, 100)),
                keys["p1"]: float(np.clip(round(row.prev1_pct, 1), 40, 100)),
                keys["p2"]: float(np.clip(round(row.prev2_pct, 1), 40, 100)),
                keys["seas"]: float(np.clip(round(row.seasonal_pct, 1), 40, 100)),
                keys["delay"]: float(np.clip(round(dpk), 0, dmax)),
                keys["nr"]: float(np.clip(round(nr0, 2), 0, 1)),
                keys["wx"]: float(np.clip(round(wx0, 2), 0, 1)),
            }
            for kk, v in defaults.items():
                st.session_state.setdefault(kk, v)

            def _reset(vals):
                for kk, v in vals.items():
                    st.session_state[kk] = v

            st.button("↺ Reset to real values", on_click=_reset, args=(defaults,), key=f"reset_{k}")
            s1, s2 = st.columns(2)
            cur_pct = s1.slider(f"% on time in {L_cur} (latest quarter)", 40.0, 100.0, step=0.1, key=keys["cur"])
            prev1 = s1.slider(f"% on time in {L_p1}", 40.0, 100.0, step=0.1, key=keys["p1"])
            prev2 = s1.slider(f"% on time in {L_p2}", 40.0, 100.0, step=0.1, key=keys["p2"])
            seas = s2.slider(f"% on time in {L_seas} (same quarter, a year earlier)", 40.0, 100.0, step=0.1, key=keys["seas"])
            delay = s2.slider(f"Delay minutes per 1,000 stops in {L_cur}", 0.0, dmax, step=1.0, key=keys["delay"])
            nr = s2.slider(f"Share of {L_cur} delay caused by Network Rail", 0.0, 1.0, step=0.01, key=keys["nr"])
            wx = s2.slider(f"Share of {L_cur} delay caused by severe weather", 0.0, 1.0, step=0.01, key=keys["wx"])

            w = row.copy()
            w["cur_pct"], w["prev1_pct"], w["prev2_pct"], w["seasonal_pct"] = cur_pct, prev1, prev2, seas
            w["roll4_pct"] = np.mean([cur_pct, prev1, prev2, seas])
            w["delta1"] = cur_pct - prev1
            w["delay_per_1k_stops"], w["nr_share"], w["weather_share"] = delay, nr, wx
            risk_wi = float(predict(pd.DataFrame([w]))[0])
            diff = (risk_wi - base) * 100
            m1, m2 = st.columns(2)
            m1.metric(f"Risk with real data ({L_next})", f"{base:.0%}")
            m2.metric("Risk with your scenario", f"{risk_wi:.0%}",
                      None if abs(diff) < 0.5 else f"{diff:+.0f} pts", delta_color="inverse")

            # what is driving the score for this scenario
            scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
            Xw = pd.DataFrame([w])[FEATURES].astype(float).fillna(pd.Series(medians))
            contrib = pd.Series(scaler.transform(Xw)[0] * lr.coef_[0], index=FEATURES)
            names = {"cur_pct": f"On time {L_cur}", "prev1_pct": f"On time {L_p1}", "prev2_pct": f"On time {L_p2}",
                     "seasonal_pct": f"On time {L_seas} (same quarter last year)",
                     "roll4_pct": "Average of the last 4 quarters", "delta1": "Change since the previous quarter",
                     "delay_per_1k_stops": "Delay minutes per 1,000 stops", "nr_share": "Network Rail share of delay",
                     "weather_share": "Weather share of delay", "traincrew_share": "Traincrew share of delay",
                     "tgt_q1": "Forecast quarter is Apr–Jun", "tgt_q2": "Forecast quarter is Jul–Sep",
                     "tgt_q3": "Forecast quarter is Oct–Dec", "tgt_q4": "Forecast quarter is Jan–Mar"}
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6).iloc[::-1]
            dfc = pd.DataFrame({"factor": [names[i] for i in top.index], "effect": top.values})
            dfc["direction"] = np.where(dfc.effect > 0, "Raises risk", "Lowers risk")
            fig = px.bar(dfc, x="effect", y="factor", orientation="h", color="direction", height=320,
                         color_discrete_map={"Raises risk": RED, "Lowers risk": TEAL},
                         title="What is pushing the risk up or down in this scenario",
                         labels={"effect": "Effect on risk score", "factor": "", "direction": ""})
            st.plotly_chart(fig)
            st.caption("Compared with a typical operator-quarter. Bars to the right push risk up; bars to the left pull it down.")

    # ---- model quality ----
    with tab_model:
        if metrics:
            st.markdown(f"**The task:** {metrics['task']}.  \n"
                        f"**How it was tested:** trained on data up to {metrics['train_until']}, then scored only on "
                        f"*future* quarters ({metrics['test_from']} → {metrics['test_to']}, {metrics['test_rows']} "
                        "operator-quarters) — the model never saw them during training.")
            res = pd.DataFrame(metrics["results"]).T.reset_index()
            res.columns = ["Model", "ROC AUC", "F1", "Precision", "Recall", "Accuracy"]
            st.dataframe(res, hide_index=True)
            note("<b>Honest reading:</b> UK rail is strongly seasonal — Oct–Dec is reliably the worst quarter — so "
                 "simple rules like <i>'same quarter last year'</i> already score very high. The model matches or slightly "
                 "beats them. It is a sound early-warning signal, not a crystal ball.")
            with st.expander("What do these scores mean?"):
                st.markdown("""
- **ROC AUC** — how well the model ranks risky operators above safe ones. 1.0 is perfect; 0.5 is a coin flip.
- **Precision** — when it says "at risk", how often is it right?
- **Recall** — of the operators that really fell below 80%, how many did it catch?
- **F1** — a single score balancing precision and recall.
- **Accuracy** — share of all predictions that were correct.
- **Baselines** are deliberately simple rules (e.g. "next quarter will look like this quarter"). A model only adds value if it beats them.
""")
            coef = pd.Series(metrics["coefficients"]).sort_values()
            fig = px.bar(coef, orientation="h", height=460,
                         title="What drives predictions overall (standardised coefficients; positive = more risk)",
                         labels={"value": "Coefficient", "index": ""})
            fig.update_layout(showlegend=False)
            fig.update_traces(marker_color=BLUE)
            st.plotly_chart(fig)


# ═════════════════════════════ ABOUT ═════════════════════════════════════════
elif page == "ℹ️ About & Data":
    st.title("ℹ️ About this project")
    st.markdown(f"""
**UK Train Delay Analytics** is an end-to-end data project that turns official UK rail statistics into
answers: *Which operators are punctual? Why do trains get delayed? Who is at risk next quarter?*

It was built by **{AUTHOR}** as a portfolio project covering the full data journey — collecting data,
cleaning it automatically in the cloud, analysing it, modelling it, and presenting it here.
""")
    b1, b2 = st.columns(2)
    b1.link_button("💻 Source code on GitHub", GITHUB_URL)
    b2.link_button("💼 Connect on LinkedIn", LINKEDIN_URL)

    st.subheader("📊 Where the data comes from")
    st.markdown(f"""
All data is published free by the **Office of Rail and Road (ORR)**, Britain's rail regulator, on the
[ORR Data Portal]({ORR_URL}) under the Open Government Licence v3.0.

| ORR table | What it contains | Used for |
|---|---|---|
| **3184** — Delay minutes by operator and cause | Delay minutes per 4-week period, split into 15 causes (Network Rail vs operators) | Delay Causes page, delay features in the model |
| **3133** — Train punctuality by operator | % of trains on time (within 59 sec, 3 min and 15 min) each quarter since 2014 | Operator Performance, Risk Predictor |
| **3130** — Time-to-3 and cancellations by station | On-time % and cancellation % for every station, per 4-week period | Stations page |

Data currently runs to **{quarter_label(latest_q)}** for punctuality and **{dc['year_label'].max()}** for delay causes.
The most recent punctuality quarter is marked *provisional* by ORR and may be revised.
""")

    st.subheader("⚙️ How it works")
    st.code("""ORR data portal (Excel / ODS files)
        │  manual download
        ▼
Amazon S3  raw/  ───── upload triggers ─────►  AWS Lambda
                                                 • detects which ORR table arrived
                                                 • cleans it in memory (pandas)
                                                 • logs to CloudWatch
                                                      │
Amazon S3  processed/  ◄── cleaned CSVs ──────────────┘
        │
        ▼
SQLite database  ──►  analysis + model training (Python)  ──►  this Streamlit dashboard""", language="text")
    st.markdown("""
**Honest status:** the cleaning pipeline runs automatically in AWS (S3 → Lambda). The dashboard currently reads a
database snapshot built from those cleaned files; connecting it directly to a cloud database and automating
the refresh are planned next steps.
""")

    st.subheader("🤖 The risk model, in one paragraph")
    st.markdown("""
A **logistic regression** model looks at an operator's last four quarters of punctuality, its momentum, how many delay
minutes it suffers per 1,000 stops, what share of delay comes from Network Rail or weather, and which season is
coming. It outputs the chance that the operator's on-time rate will fall below 80% next quarter. It was validated
on **future quarters only** and compared against simple baselines — see the *How good is it?* tab on the Risk Predictor page.
""")

    st.subheader("⚠️ Limitations worth knowing")
    st.markdown("""
- **Seasonality dominates.** Much of the model's accuracy comes from the fact that Oct–Dec is reliably the worst quarter.
- **Risk scores are rankings**, not calibrated probabilities.
- **Delay minutes aren't normalised for size** unless you use the "per 1,000 stops" view.
- **Station figures cover Apr 2023 onwards** only, and small stations are noisy (hence the minimum-stops filter).
- **ORR revises recent data**, so the latest quarter can change slightly.
- This is an independent portfolio project, **not an official ORR product**.
""")

    st.subheader("🧰 Tech stack")
    st.markdown("Python · pandas · scikit-learn · SQLite · Plotly · Streamlit · AWS S3 · AWS Lambda · CloudWatch · GitHub")
    st.subheader("🛣 Roadmap")
    st.markdown("Planned: Amazon RDS (cloud database) → Docker + AWS ECS (always-on hosting) → AWS Glue + Athena (data lake).")
    st.caption("Contains public sector information licensed under the Open Government Licence v3.0.")


# ═════════════════════════════ RAW DATA ══════════════════════════════════════
elif page == "🗃 Raw Data":
    st.title("🗃 Raw Data Explorer")
    page_intro("The cleaned tables behind every chart. Pick a table to browse it, then download it as CSV.")
    desc = {"delay_causes": "Delay minutes by operator, 4-week period and cause (ORR Table 3184).",
            "operator_punctuality": "Quarterly % of trains on time by operator (ORR Table 3133).",
            "station_performance": "Station on-time and cancellation figures by 4-week period (ORR Table 3130)."}
    table = st.selectbox("Table", list(desc))
    st.caption(desc[table])
    d = _query(f"SELECT * FROM {table} LIMIT 5000")
    st.caption(f"Showing up to 5,000 rows · {len(d.columns)} columns")
    st.dataframe(d)
    st.download_button(f"⬇️ Download {table}.csv (rows shown)", d.to_csv(index=False).encode("utf-8"),
                       f"{table}.csv", "text/csv")
