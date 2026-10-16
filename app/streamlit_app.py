# app/streamlit_app.py
# UK Train Delay Analytics — a plain-English guide to how reliable UK train operators are.
# Run: streamlit run app/streamlit_app.py
#
# Design rules: one simple question -> one clear answer; every change the user makes is explained;
# every number shows its unit, period and source; nothing is silently dropped or filled in.
# All calculations live in scripts/insights.py (unit-tested); this file only lays them out.

import html
import json
import logging
import os
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
from scripts import insights as ins  # noqa: E402
from scripts.features import (FEATURES, THRESHOLD, build_feature_table,  # noqa: E402
                              next_quarter_start, quarter_label)

log = logging.getLogger("train_app")
DEBUG = bool(os.environ.get("TRAIN_APP_DEBUG"))          # tests set this so errors are not hidden

DB_PATH = ROOT / "database" / "delays.db"
INFO_PATH = ROOT / "database" / "build_info.json"
MODEL_PATH = ROOT / "models" / "risk_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

BLUE, TEAL, AMBER, RED, GREY = "#4C8BF5", "#1D9E75", "#E3A008", "#E5484D", "#8a8f98"
AUTHOR = "Abubakar Shaikh"
GITHUB_URL = "https://github.com/abuv69/uk-train-delay-analytics"
LINKEDIN_URL = "https://www.linkedin.com/in/mohammedabubakarshaikh"            # paste your LinkedIn profile URL here; the button stays hidden while this is empty
ORR_URL = "https://dataportal.orr.gov.uk/"

st.set_page_config(page_title="UK Train Delays: is my train company reliable?", page_icon="🚂",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container {padding-top: 4.5rem; max-width: 1100px;}
div[data-baseweb="select"] > div {min-height: 48px;}
.stButton > button, .stDownloadButton > button, .stLinkButton > a {min-height: 48px; width: 100%;}
.answer {padding: 1.1rem 1.3rem; border-radius: 16px; margin: .6rem 0 1rem;
         border: 1px solid rgba(128,128,128,.3); border-left: 10px solid #8a8f98;
         background: rgba(128,128,128,.08);}
.answer.green {border-left-color: #1D9E75; background: rgba(29,158,117,.10);}
.answer.amber {border-left-color: #E3A008; background: rgba(227,160,8,.10);}
.answer.red {border-left-color: #E5484D; background: rgba(229,72,77,.10);}
.answer .light {font-size: 1.15rem; font-weight: 700; margin-bottom: .3rem;}
.answer h2 {margin: .1rem 0 .5rem; font-size: 1.55rem; line-height: 1.3; padding: 0;}
.answer p {margin: .25rem 0; font-size: 1.02rem;}
.answer .small {font-size: .9rem; opacity: .85;}
.note {padding: .65rem .9rem; border-left: 4px solid #4C8BF5; background: rgba(76,139,245,.10);
       border-radius: 6px; font-size: .95rem; margin: .4rem 0 .9rem;}
.warn {border-left-color: #E3A008; background: rgba(227,160,8,.10);}
.changed {border-left-color: #1D9E75; background: rgba(29,158,117,.10);}
@media (max-width: 640px) {
  .answer h2 {font-size: 1.25rem;} h1 {font-size: 1.7rem !important;}
  .block-container {padding-left: .8rem; padding-right: .8rem;}
}
</style>
""", unsafe_allow_html=True)


def note(text, kind=""):
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def style(fig, height=None):
    fig.update_layout(margin=dict(l=0, r=10, t=55, b=10), legend=dict(orientation="h", y=-0.18, title=None),
                      title_font_size=16, title_x=0)
    if height:
        fig.update_layout(height=height)
    fig.update_yaxes(automargin=True)
    return fig


def show(fig, height=None):
    st.plotly_chart(style(fig, height), config={"displayModeBar": False})


# ── data ──────────────────────────────────────────────────────────────────────
if not DB_PATH.exists():
    st.error("The data file is missing. Run `python scripts/load_to_sql.py` first.")
    st.stop()


@st.cache_data(show_spinner="Loading the latest rail data…")
def load_data():
    with closing(sqlite3.connect(DB_PATH.as_posix())) as conn:
        pq = pd.read_sql("SELECT * FROM operator_punctuality WHERE pct_on_time_3m IS NOT NULL", conn,
                         parse_dates=["qdate"])
        dc = pd.read_sql("SELECT * FROM delay_causes", conn)
        stn = pd.read_sql("""SELECT period, station, scheduled_stops, time_to_3_pct, cancellations_pct,
                                    year_label, period_num FROM station_performance""", conn)
        feats = build_feature_table(conn)
    pq = pq.sort_values(["operator", "qdate"]).reset_index(drop=True)
    dc["period_start"] = (pd.to_datetime(dc["start_year"].astype(int).astype(str) + "-04-01")
                          + pd.to_timedelta((dc["period_num"] - 1) * 28, unit="D"))
    return pq, dc, ins.add_station_quarter(stn), feats


@st.cache_data
def load_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@st.cache_data
def station_names():
    return sorted(load_data()[2]["station"].dropna().unique())


@st.cache_data
def station_summary():
    return ins.station_table(load_data()[2])


@st.cache_data
def csv_bytes(table):
    with closing(sqlite3.connect(DB_PATH.as_posix())) as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn).to_csv(index=False).encode("utf-8")


pq, dc, stn, feats = load_data()
bundle, metrics, info = load_bundle(), load_json(METRICS_PATH), load_json(INFO_PATH)
latest_q = pq["qdate"].max()
next_q = next_quarter_start(latest_q)
L_LATEST, L_NEXT = quarter_label(latest_q), quarter_label(next_q)
latest_provisional = bool(pq.loc[pq.qdate == latest_q, "provisional"].any())
operators = sorted(pq["operator"].unique())


def all_forecasts():
    """Forecast for every operator that has the history the model needs (others get an explicit reason)."""
    if bundle is None:
        return pd.DataFrame()
    ok = feats[feats.qdate == latest_q].dropna(subset=["cur_pct", "prev1_pct", "prev2_pct", "seasonal_pct"]).copy()
    X = ok[FEATURES].astype(float).fillna(pd.Series(bundle["medians"]))
    ok["p_below"] = bundle["model"].predict_proba(X)[:, 1]
    return ok


# ── navigation, URL and "what changed" plumbing ───────────────────────────────
HOME = "🏠 Home"
PAGES = {HOME: "home", "📉 Why trains are late": "late", "🏆 Compare operators": "compare",
         "🚉 Station lookup": "stations", "🔮 Forecast for next quarter": "forecast",
         "ℹ️ Data and method": "method", "⬇️ Download the data": "download"}
SLUG = {v: k for k, v in PAGES.items()}

if "page" not in st.session_state:
    st.session_state["page"] = SLUG.get(st.query_params.get("page", "home"), HOME)


def goto(p):
    st.session_state["page"] = p


page = st.sidebar.radio("Menu", list(PAGES), key="page")
st.sidebar.caption("On a phone, tap the arrow at the top left to open this menu.")

if st.session_state.get("_last_page") != page:           # new page: forget the previous page's changes
    for k in [k for k in st.session_state if str(k).startswith(("_prev_", "_msg_"))]:
        del st.session_state[k]
    if st.session_state.get("_last_page") is not None:    # moved within the app (not opening a shared link)
        st.query_params.from_dict({"page": PAGES[page]})  # so one page's choices never leak into the next
    st.session_state["_last_page"] = page


def sync_url(**params):
    """Put the current selections in the address bar so a result can be shared and re-opened identically."""
    p = {k: str(v) for k, v in params.items() if v not in (None, "", [])}
    p["page"] = PAGES[page]
    if dict(st.query_params) != p:
        st.query_params.from_dict(p)


def url_choice(name, allowed, default, key, label):
    """Read a URL parameter once, validate it, and fall back with a friendly note if it is invalid."""
    if key in st.session_state:
        return
    raw = st.query_params.get(name)
    if raw is None:
        st.session_state[key] = default
    elif raw in allowed:
        st.session_state[key] = raw
    else:
        st.session_state[key] = default
        note(f"We couldn't find {label} “{esc(raw)}” from the link, so we're showing {esc(default)} instead.", "warn")


def change_message(key, sel, vals, builder):
    """Remember the previous selection for this page; when it changes, build the 'What changed' line."""
    prev = st.session_state.get(f"_prev_{key}")
    st.session_state[f"_prev_{key}"] = (sel, vals)
    if prev is not None and prev[0] != sel:
        st.session_state[f"_msg_{key}"] = builder(prev[0], prev[1], sel, vals)
    return st.session_state.get(f"_msg_{key}")


def show_change(msg):
    if msg:
        note(f"<b>What changed:</b> {esc(msg)}", "changed")


def back_button():
    st.button("← Back to home", on_click=goto, args=(HOME,), key=f"back_{PAGES[page]}")


def data_footer():
    prov = " The most recent quarter is provisional and may be revised." if latest_provisional else ""
    st.caption(f"Data last updated: {info.get('built_at_utc', 'unknown')} UTC · Train figures run to {L_LATEST}.{prov} "
               f"Source: Office of Rail and Road (ORR), Open Government Licence v3.0.")


def friendly_error(e):
    if DEBUG:
        raise e
    log.exception("page failed")
    st.error("Sorry, something went wrong while building this page. Please try a different choice or refresh. "
             "If it keeps happening, please let the developer know.")


# ══════════════════════════════ HOME ═════════════════════════════════════════
def forecast_for(op):
    """(forecast dict or None, reason or None) for an operator, starting from the latest quarter."""
    reason = ins.forecast_unavailable_reason(feats, op, latest_q)
    if reason or bundle is None:
        return None, reason or "the forecast model is not available."
    row = feats[(feats.operator == op) & (feats.qdate == latest_q)]
    fc = ins.predict_risk(bundle["model"], bundle["medians"], row, FEATURES)
    fc["conf"] = ins.confidence(fc["p_below"], fc["imputed"], metrics.get("calibration_bands", []))
    fc["row"] = row.iloc[0]
    return fc, None


def conf_line(fc):
    c = fc["conf"]
    why = "; ".join(c["reasons"])
    return (f"<b>Confidence: {c['level']}.</b> {esc(why[:1].upper() + why[1:])}. Forecasts are never certain."
            if why else f"<b>Confidence: {c['level']}.</b> Forecasts are never certain.")


def answer_html(op, kind, qd, actual, qoq, fc, fc_reason):
    ql = quarter_label(qd)
    if kind == "actual" and actual is None:
        return (f'<div class="answer"><div class="light">ℹ️ No data</div>'
                f'<h2>No data for {esc(op)} in {ql}</h2><p>That is because {esc(ins.missing_reason(pq, op, qd))}</p></div>')
    if kind == "actual":
        pct, above = actual["pct"], actual["pct"] >= THRESHOLD
        prov = " <span class='small'>(provisional figure)</span>" if actual["provisional"] else ""
        prev_l = quarter_label(qd - pd.DateOffset(months=3))
        if qoq is None:
            delta = f" (there is no {prev_l} figure to compare with)"
        elif qoq == 0:
            delta = f", the same as in {prev_l}"
        else:
            delta = f", {'up' if qoq > 0 else 'down'} {abs(qoq):.1f} percentage points from {prev_l}"
        head = f"{esc(op)}: {ins.in_ten(pct)} trains arrived on time in {ql}."
        body = f"<p><b>{pct:.1f}%</b> of trains arrived on time{delta}.{prov}</p>"
        if qd == latest_q and fc:
            emoji, text, colour = ins.traffic_light(fc["p_below"], above)
            return (f'<div class="answer {colour}"><div class="light">{emoji} {esc(text)}</div><h2>{head}</h2>{body}'
                    f'<p>{esc(ins.outlook_sentence(fc["p_below"], above, L_NEXT))}</p>'
                    f'<p class="small">{conf_line(fc)}</p></div>')
        extra = (f"<p class='small'>No forecast for {esc(op)} because {esc(fc_reason)}</p>"
                 if qd == latest_q else
                 f"<p class='small'>Forecasts start from the latest quarter. Choose “{L_LATEST} (latest)” "
                 f"above to see what to expect in {L_NEXT}.</p>")
        return f'<div class="answer"><h2>{head}</h2>{body}{extra}</div>'
    # forecast quarter selected
    if not fc:
        return (f'<div class="answer"><div class="light">ℹ️ No forecast</div><h2>No forecast for {esc(op)} in {L_NEXT}</h2>'
                f'<p>That is because {esc(fc_reason)}</p></div>')
    above = fc["row"]["cur_pct"] >= THRESHOLD
    emoji, text, colour = ins.traffic_light(fc["p_below"], above)
    return (f'<div class="answer {colour}"><div class="light">{emoji} {esc(text)}</div>'
            f'<h2>Our forecast for {esc(op)} in {L_NEXT}</h2>'
            f'<p>{esc(ins.outlook_sentence(fc["p_below"], above, L_NEXT))} We put the chance that more than '
            f'{THRESHOLD:.0f}% of its trains arrive on time at <b>{ins.format_probability(fc["p_above"])}</b>.</p>'
            f'<p class="small">Latest actual figure: {fc["row"]["cur_pct"]:.1f}% in {L_LATEST}.</p>'
            f'<p class="small">{conf_line(fc)}</p></div>')


def why_number_md(op, kind, qd, actual, qoq, fc):
    if kind == "actual" and actual:
        prev_l = quarter_label(qd - pd.DateOffset(months=3))
        return (f"**Source:** Office of Rail and Road, Table 3133 (percentage of trains arriving on time, by operator).  \n"
                f"**Period:** {quarter_label(qd)}{' (provisional)' if actual['provisional'] else ''}.  \n"
                f"**How it is calculated:** the ORR times each train at each station stop; a stop counts as on time if the "
                f"train arrives within 3 minutes of the timetable. {actual['pct']:.1f}% is the share of "
                f"{actual['stops']:,.0f} timed stops that were on time. “About {ins.in_ten(actual['pct']).replace('about ', '')}” "
                f"is that percentage rounded to the nearest tenth.  \n"
                f"**Change since {prev_l}:** " + (f"{qoq:+.1f} percentage points (difference of the two rounded figures)."
                                                  if qoq is not None else "not available (no figure for that quarter)."))
    if fc:
        r = fc["row"]
        past = [f"{quarter_label(r['qdate'] - pd.DateOffset(months=m))}: {v:.1f}%" for m, v in
                [(0, r.cur_pct), (3, r.prev1_pct), (6, r.prev2_pct), (9, r.seasonal_pct)]]
        res = metrics.get("results", {}).get("Logistic regression (deployed)", {})
        return ("**What the forecast is:** a statistical model (logistic regression) estimates the chance that fewer than "
                f"{THRESHOLD:.0f}% of this operator's trains arrive on time in {L_NEXT}.  \n"
                f"**What it used:** the operator's % on time in {'; '.join(past)}; delay minutes per 1,000 stops "
                f"({r.delay_per_1k_stops:,.0f} in {L_LATEST}); the share of delay caused by Network Rail, weather and the "
                f"operator's own staff; and the season being forecast.  \n"
                f"**How accurate it has been:** tested on {metrics.get('test_rows', '?')} operator-quarters from "
                f"{metrics.get('test_from', '?')} to {metrics.get('test_to', '?')} that it never saw in training, it called "
                f"{res.get('accuracy', float('nan')):.0%} of them correctly.  \n"
                f"**Safety limits:** forecasts are shown as ranges (never 0% or 100%) because nothing is certain.")
    return ""


def home():
    _, mid, _ = st.columns([1, 10, 1])
    with mid:
        st.markdown("# 🚂 Is your train company reliable?")
        st.markdown("**This site tells you how often a UK train company's trains arrive on time, why they run late, "
                    "and what to expect next quarter.**")

        # ---- search panel -----
        opts = [(f"{L_LATEST} (latest)", "actual", latest_q), (f"{L_NEXT} (forecast)", "forecast", next_q)]
        older = [q for q in sorted(pq.qdate.unique(), reverse=True) if q != latest_q][:12]
        opts += [(quarter_label(q), "actual", pd.Timestamp(q)) for q in older]
        opt_map = {lab: (k, q) for lab, k, q in opts}
        iso = {("forecast" if k == "forecast" else pd.Timestamp(q).strftime("%Y-%m-%d")): lab for lab, k, q in opts}
        default_op = pq[pq.qdate == latest_q].nlargest(1, "recorded_stops")["operator"].iloc[0]
        url_choice("operator", operators, default_op, "sel_operator", "the operator")
        if "sel_quarter" not in st.session_state:
            raw = st.query_params.get("quarter")
            st.session_state["sel_quarter"] = iso.get(raw, opts[0][0])
            if raw and raw not in iso:
                note(f"We couldn't find the quarter “{esc(raw)}” from the link, so we're showing {L_LATEST}.", "warn")
        if "sel_station" not in st.session_state and st.query_params.get("station") in station_names():
            st.session_state["sel_station"] = st.query_params.get("station")

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            op = c1.selectbox("Which train company do you use?", operators, key="sel_operator",
                              help="Type to search. These are the train operating companies in the ORR data.")
            station = c2.selectbox("Which station? (optional)", station_names(), index=None, key="sel_station",
                                   placeholder="Type a station name…",
                                   help="Optional. Shows how punctual that station is.")
            qlabel_sel = c3.selectbox("Which quarter?", list(opt_map), key="sel_quarter",
                                      help="A quarter is three months. The forecast looks ahead to the next one.")
        kind, qd = opt_map[qlabel_sel]
        qd = pd.Timestamp(qd)
        st.session_state["_carry_op"] = op          # lets "Why trains are late" and "Forecast" open on your company
        st.caption(f"Showing **{esc(op)}**. Not your train company? Choose another above."
                   if op == default_op and not st.query_params.get("operator") else "")

        # ---- numbers (one snapshot used by the card AND the "what changed" line) -----
        actual = ins.operator_quarter(pq, op, qd) if kind == "actual" else None
        qoq = ins.qoq_change(pq, op, qd) if kind == "actual" else None
        fc, fc_reason = forecast_for(op)
        st_sum = ins.station_quarter_summary(stn, station, qd) if station else None
        snapshot = {"operator": op, "qlabel": quarter_label(qd), "kind": kind,
                    "pct": actual["pct"] if actual else None, "p_above": fc["p_above"] if fc else None,
                    "station": station, "station_pct": st_sum["pct"] if st_sum else None}
        msg = change_message("home", (op, station, qlabel_sel), snapshot,
                             lambda ps, pv, s, v: ins.describe_change(pv, v))
        sync_url(operator=op, station=station, quarter=("forecast" if kind == "forecast" else qd.strftime("%Y-%m-%d")))

        show_change(msg)
        st.markdown(answer_html(op, kind, qd, actual, qoq, fc, fc_reason), unsafe_allow_html=True)
        with st.expander("Why this number?"):
            st.markdown(why_number_md(op, kind, qd, actual, qoq, fc) or "There is no figure to explain for this selection.")

        if station:
            if st_sum:
                nat = ins.weighted_mean(*(lambda d: (d["time_to_3_pct"], d["scheduled_stops"]))(
                    stn[(stn.qdate == qd) & stn.time_to_3_pct.notna()]))
                st.markdown(f"#### 🚉 {esc(station)} in {quarter_label(qd)}")
                st.markdown(f"**{st_sum['pct']:.1f}%** of trains arrived on time at {esc(station)} "
                            f"(average across all stations that quarter: {nat:.1f}%).")
            else:
                note(f"No data for {esc(station)} in {quarter_label(qd)} because "
                     f"{esc(ins.station_missing_reason(stn, station, qd))}", "warn")
            with st.expander("Why this number? (station)"):
                st.markdown("**Source:** ORR Table 3130 (on-time and cancellations by station). **How:** each station is timed "
                            "in 4-week periods; we average the periods that fall in the quarter, weighted by scheduled "
                            "stops, counting only periods with a published figure. A train is on time if it arrives within "
                            "3 minutes of the timetable.")

        # ---- three supporting cards -----
        ref_q = qd if kind == "actual" else latest_q
        ref_l = quarter_label(ref_q)
        st.markdown("### More about " + esc(op))
        card_compare(op, ref_q, ref_l, kind)
        card_causes(op)
        card_seasons(op, qd if kind == "actual" else next_q)

        # ---- explore more -----
        st.markdown("### Explore more")
        items = [("📉 Why trains are late", "What causes delays, and who is responsible."),
                 ("🏆 Compare operators", "Side-by-side trends, a heatmap and a league table."),
                 ("🚉 Station lookup", "Find any station, plus the best and worst."),
                 ("🔮 Forecast for next quarter", "Which operators are at risk of falling short."),
                 ("ℹ️ Data and method", "Where the data comes from and its limits."),
                 ("⬇️ Download the data", "Get the tables behind every chart.")]
        cols = st.columns(3)
        for i, (name, desc) in enumerate(items):
            with cols[i % 3]:
                st.button(name, on_click=goto, args=(name,), key=f"go_{i}")
                st.caption(desc)
        st.divider()
        data_footer()


def card_compare(op, ref_q, ref_l, kind):
    with st.container(border=True):
        st.markdown("#### How does it compare with other operators?")
        r = ins.rank_in_quarter(pq, op, ref_q)
        if r is None:
            st.markdown(f"No comparison for {esc(op)} in {ref_l} because {esc(ins.missing_reason(pq, op, ref_q))}")
            return
        own = ins.operator_quarter(pq, op, ref_q)["pct"]
        gap = round(round(own, 1) - round(r["national"], 1), 1)
        word = "above" if gap >= 0 else "below"
        title = f"{op} ranked {r['rank']} of {r['n']}, {abs(gap):.1f} points {word} the national average"
        st.markdown(f"In {ref_l}, **{esc(op)}** ranked **{r['rank']} out of {r['n']}** operators with **{own:.1f}%** of trains "
                    f"on time, against **{r['national']:.1f}%** nationally."
                    + (f" (Comparison uses {ref_l}, the latest actual quarter.)" if kind == "forecast" else ""))
        rows = pq[pq.qdate == ref_q].sort_values("pct_on_time_3m")
        fig = go.Figure(go.Bar(
            x=rows.pct_on_time_3m, y=rows.operator, orientation="h",
            marker_color=[BLUE if o == op else GREY for o in rows.operator],
            text=[f"{v:.0f}%" + (" ◀ selected" if o == op else "") for o, v in zip(rows.operator, rows.pct_on_time_3m)],
            textposition="outside", cliponaxis=False))
        fig.add_vline(x=THRESHOLD, line_dash="dash", line_color=AMBER)
        fig.update_layout(title=title, xaxis_title=f"% of trains on time, {ref_l}", xaxis_range=[0, 115])
        show(fig, 24 * len(rows) + 110)
        with st.expander("Why this number?"):
            st.markdown(f"**Source:** ORR Table 3133. **Period:** {ref_l}. Operators are ranked by the % of timed station stops "
                        f"that were within 3 minutes of the timetable. Only the {r['n']} operators with a published figure "
                        f"for {ref_l} are ranked; operators that were not running or reported then are not included. The "
                        f"national figure is the average across those operators, weighted by how many stops each makes.")


def card_causes(op):
    with st.container(border=True):
        st.markdown("#### What usually causes delays for this operator?")
        own = dc[dc.operator == op]
        if own.empty:
            st.markdown(f"No delay-cause figures for {esc(op)}: the ORR's delay-cause table has no rows for it.")
            return
        fy = own.year_label.max()
        sh = ins.cause_group_shares(dc, op, fy)
        y0 = int(fy[:4])
        span = f"April {y0} to March {y0 + 1}"
        top = sh.iloc[0]
        st.markdown(f"In **{fy}** ({span}), {esc(op)} lost **{sh.minutes.sum():,.0f} minutes** to delays. The biggest cause (**{top.share:.0%}** of the "
                    f"minutes) was: **{top.group}**.")
        d = sh.sort_values("share")
        fig = go.Figure(go.Bar(x=d.share * 100, y=d.group, orientation="h", marker_color=BLUE,
                               text=[f"{v:.0f}%" for v in d.share * 100], textposition="outside", cliponaxis=False))
        fig.update_layout(title=f"{op}'s biggest cause of delay: {top.group.split(' (')[0].lower()} ({top.share:.0%})",
                          xaxis_title=f"% of delay minutes, {fy}", xaxis_range=[0, max(d.share * 100) * 1.25])
        show(fig, 330)
        with st.expander("Why this number?"):
            st.markdown(f"**Source:** ORR Table 3184 (delay minutes by operator and cause). **Period:** financial year {fy} "
                        f"({span}), the latest year in the data. ORR lists 15 causes; we group them into six plain-English "
                        f"groups. Each bar is that group's delay minutes divided by the operator's total delay minutes.")


def card_seasons(op, season_q):
    with st.container(border=True):
        st.markdown("#### Is this quarter usually better or worse than others?")
        prof = ins.seasonal_profile(pq, op)
        if prof is None:
            n = int((pq.operator == op).sum())
            st.markdown(f"No seasonal pattern for {esc(op)}: it has only {n} quarters of figures and we need at least 4.")
            return
        order = ["Apr–Jun", "Jul–Sep", "Oct–Dec", "Jan–Mar"]
        by = prof["by_season"].reindex(order)
        this = ins.season_name(season_q)
        weakest = by["mean"].idxmin()
        span = f"{quarter_label(prof['from'])} to {quarter_label(prof['to'])}"
        if pd.notna(by.loc[this, "mean"]):
            diff = round(round(by.loc[this, "mean"], 1) - round(prof["overall"], 1), 1)
            st.markdown(f"**{this}** is usually **{abs(diff):.1f} percentage points {'better' if diff > 0 else 'worse'}** than "
                        f"{esc(op)}'s average of **{prof['overall']:.1f}%** (last {prof['n_quarters']} quarters, {span}).")
        fig = go.Figure(go.Bar(x=by.index, y=by["mean"], marker_color=[BLUE if s == this else GREY for s in by.index],
                               text=[f"{v:.0f}%" if pd.notna(v) else "no data" for v in by["mean"]], textposition="outside"))
        fig.update_layout(title=f"{weakest} is {op}'s weakest season", yaxis_title="% of trains on time",
                          yaxis_range=[0, 112])
        show(fig, 320)
        with st.expander("Why this number?"):
            st.markdown(f"**Source:** ORR Table 3133. **Period:** the operator's last {prof['n_quarters']} quarters ({span}). "
                        f"Each bar averages the quarters that fall in that season (so each uses about "
                        f"{prof['n_quarters'] // 4} quarters). Blue marks the season you are looking at. Averages over "
                        f"only a few years can be moved by one unusual quarter.")


# ══════════════════════════════ WHY TRAINS ARE LATE ══════════════════════════
def period_text(fy):
    if fy == "All years":
        return f"April {int(dc.year_label.min()[:4])} to March {int(dc.year_label.max()[:4]) + 1}"
    return f"April {int(fy[:4])} to March {int(fy[:4]) + 1}"


def page_late():
    back_button()
    st.title("📉 Why trains are late")
    st.markdown("Every delay is given a cause by the rail industry. This page shows the biggest causes, and whether "
                "**Network Rail** (which looks after tracks, signals and stations) or the **train companies** themselves "
                "are responsible.")
    ops = ["All operators"] + sorted(dc.operator.unique())
    years = ["All years"] + sorted(dc.year_label.unique())
    carried = st.session_state.get("_carry_op")
    url_choice("operator", ops, carried if carried in ops else "All operators", "late_op", "the train company")
    url_choice("year", years, "All years", "late_year", "the year")
    c1, c2 = st.columns(2)
    op = c1.selectbox("Which train company?", ops, key="late_op")
    fy = c2.selectbox("Which year? (April to March)", years, key="late_year")
    d = dc if op == "All operators" else dc[dc.operator == op]
    if fy != "All years":
        d = d[d.year_label == fy]
    sync_url(operator=op, year=fy)
    total = d.total_delay_mins.sum()
    show_change(change_message(
        "late", (op, fy), total / 1e6,
        lambda ps, pv, s, v: ins.describe_generic_change(
            "the selection", f"{ps[0]}, {ps[1].lower()}", f"{s[0]}, {s[1].lower()}", "Total delay minutes",
            ins.fmt_value(pv, " million"), ins.fmt_value(v, " million"), ins.fmt_delta(pv, v, " million"),
            period_text(s[1]))))
    if total <= 0:
        note(f"No delay figures for {esc(op)} in {esc(fy)} because the ORR's delay-cause table has no rows for that "
             "combination (the company may not have been running).", "warn")
        return

    nr = d.nr_responsible_mins.sum() / total
    st.markdown(f"**{total / 1e6:,.1f} million minutes** of delay were recorded for **{esc(op)}** "
                f"({period_text(fy)}).")
    groups = pd.DataFrame({"group": list(ins.CAUSE_GROUPS),
                           "minutes": [d[c].sum().sum() for c in ins.CAUSE_GROUPS.values()]})
    groups["share"] = groups.minutes / groups.minutes.sum() * 100
    g = groups.sort_values("share")
    fig = go.Figure(go.Bar(x=g.share, y=g.group, orientation="h", marker_color=BLUE,
                           text=[f"{v:.0f}%" for v in g.share], textposition="outside", cliponaxis=False))
    fig.update_layout(title=f"Network Rail caused {nr:.0%} of delay minutes ({period_text(fy)})",
                      xaxis_title="% of delay minutes", xaxis_range=[0, g.share.max() * 1.25])
    show(fig, 380)
    with st.expander("Why this number?"):
        st.markdown(f"**Source:** ORR Table 3184 (delay minutes by operator and cause). **Period:** {period_text(fy)}. "
                    "ORR gives 15 causes; we group them into six. Network Rail's share adds up the five causes ORR assigns "
                    "to Network Rail (track, other equipment, management, outside events, severe weather) and divides by "
                    "all delay minutes in the selection.")

    yr = (dc if op == "All operators" else dc[dc.operator == op]).groupby("year_label")[
        ["nr_responsible_mins", "toc_responsible_mins"]].sum() / 1e6
    tot = yr.sum(axis=1)
    first, last = tot.index.min(), tot.index.max()
    chg = (tot[last] / tot[first] - 1) * 100
    fig = px.bar(yr.reset_index(), x="year_label", y=["nr_responsible_mins", "toc_responsible_mins"],
                 labels={"value": "Million delay minutes", "year_label": "Financial year", "variable": ""},
                 title=f"Delay minutes in {last} were {abs(chg):.0f}% {'lower' if chg < 0 else 'higher'} than in {first}")
    fig.for_each_trace(lambda t: t.update(name="Network Rail" if t.name.startswith("nr") else "Train companies",
                                          marker_color=BLUE if t.name.startswith("nr") else AMBER))
    show(fig, 360)
    note("2020-21 is low because services were heavily cut during COVID-19 lockdowns, not because the railway got better.")

    with st.expander("For analysts: all 15 ORR cause categories"):
        names = {"nr_external": "NR External", "nr_network_mgmt": "NR Network management/other",
                 "nr_non_track_assets": "NR Non-track assets", "nr_severe_weather": "NR Severe weather",
                 "nr_track": "NR Track", "toc_self_fleet": "TOC Fleet", "toc_self_operations": "TOC Operations",
                 "toc_self_stations": "TOC Stations", "toc_self_traincrew": "TOC Traincrew", "toc_self_other": "TOC Other",
                 "toc_on_toc_fleet": "On-TOC Fleet", "toc_on_toc_operations": "On-TOC Operations",
                 "toc_on_toc_stations": "On-TOC Stations", "toc_on_toc_traincrew": "On-TOC Traincrew",
                 "toc_on_toc_other": "On-TOC Other"}
        t = d[list(names)].sum().rename(names).reset_index()
        t.columns = ["Cause (ORR category)", "Delay minutes"]
        st.dataframe(t.sort_values("Delay minutes", ascending=False), hide_index=True)
        st.caption("NR = Network Rail. TOC = the train operating company itself. On-TOC = caused by another operator's train.")


# ══════════════════════════════ COMPARE OPERATORS ════════════════════════════
def page_compare():
    back_button()
    st.title("🏆 Compare operators")
    st.markdown("How often do trains arrive on time? **“On time” means within 3 minutes of the timetable.** Higher is "
                "better, and the dashed line marks 80%.")
    top6 = pq[pq.qdate == latest_q].nlargest(6, "recorded_stops")["operator"].tolist()
    if "cmp_ops" not in st.session_state:
        raw = st.query_params.get("operators")
        wanted = raw.split(",") if raw else top6
        valid = [o for o in wanted if o in operators]
        if raw and len(valid) != len(wanted):
            note("Some train companies in the link couldn't be found, so they were left out.", "warn")
        st.session_state["cmp_ops"] = valid or top6
    sel = st.multiselect("Which train companies do you want to compare?", operators, key="cmp_ops",
                         help="Defaults to the six companies with the most timed stops. Type to search.")
    sync_url(operators=",".join(sel))
    vals = {o: (ins.operator_quarter(pq, o, latest_q) or {}).get("pct") for o in sel}
    show_change(change_message(
        "compare", tuple(sorted(sel)), vals,
        lambda ps, pv, s, v: ins.describe_set_change(sorted(set(s) - set(ps)), sorted(set(ps) - set(s)),
                                                     {**pv, **v}, L_LATEST)))
    if not sel:
        st.info("Choose at least one train company above to see its trend.")

    q12 = sorted(pq.qdate.unique())[-12:]
    t1, t2, t3 = st.tabs(["📈 Trend over time", "🟩 Last 12 quarters", "🏁 League table"])
    with t1:
        if sel:
            d = pq[pq.operator.isin(sel)]
            r = d[d.qdate >= q12[0]].assign(season=lambda x: x.qdate.map(ins.season_name))
            by = r.groupby("season")["pct_on_time_3m"].mean()
            weak = by.idxmin()
            others = r[r.season != weak]["pct_on_time_3m"].mean()
            fig = px.line(d, x="qdate", y="pct_on_time_3m", color="operator",
                          labels={"qdate": "Quarter", "pct_on_time_3m": "% of trains on time (within 3 minutes)",
                                  "operator": ""},
                          title=f"{weak} is the weakest season: {by[weak]:.0f}% on time vs {others:.0f}% in other seasons "
                                f"(last 3 years)")
            fig.add_hline(y=THRESHOLD, line_dash="dash", line_color=AMBER, annotation_text="80%")
            show(fig, 430)
            note("The big jump in 2020 is the COVID lockdown, when far fewer trains ran.")
    with t2:
        h = pq[pq.qdate.isin(q12)].pivot_table(index="operator", columns="qdate", values="pct_on_time_3m")
        means = h.mean(axis=1)
        h.columns = [quarter_label(c) for c in h.columns]
        h = h.loc[means.sort_values().index]
        fig = px.imshow(h, color_continuous_scale="RdYlGn", zmin=60, zmax=100, aspect="auto", text_auto=".0f",
                        labels={"color": "% on time"},
                        title=f"{means.idxmax()} had the highest on-time rate over {quarter_label(q12[0])} to "
                              f"{quarter_label(q12[-1])} ({means.max():.0f}%); {means.idxmin()} the lowest ({means.min():.0f}%)")
        fig.update_xaxes(side="top")
        show(fig, max(420, 28 * len(h) + 120))
        st.caption("Each cell is the % of trains on time in that quarter. Green = good, red = poor. "
                   "A blank cell means there is no figure for that quarter (the company was not running or not reported).")
    with t3:
        q4 = sorted(pq.qdate.unique())[-4:]
        lg = (pq[pq.qdate.isin(q4)].groupby("operator")["pct_on_time_3m"]
              .agg(avg="mean", worst="min", best="max", n="count").reset_index().sort_values("avg", ascending=False))
        lg.columns = ["Train company", "Average % on time", "Worst quarter %", "Best quarter %", "Quarters with data"]
        st.markdown(f"**Ranked by average % of trains on time, {quarter_label(q4[0])} to {quarter_label(q4[-1])}**")
        st.dataframe(lg, hide_index=True, column_config={
            "Average % on time": st.column_config.ProgressColumn("Average % on time", min_value=50, max_value=100, format="%.1f"),
            "Worst quarter %": st.column_config.NumberColumn(format="%.1f"),
            "Best quarter %": st.column_config.NumberColumn(format="%.1f")})
        absent = sorted(set(operators) - set(lg["Train company"]))
        st.caption(f"{len(lg)} companies have figures in these four quarters. Not shown (no figures in this period, "
                   f"for example because they stopped running): {', '.join(absent) if absent else 'none'}. "
                   "Companies with fewer than 4 quarters are new or restarted.")


# ══════════════════════════════ STATION LOOKUP ═══════════════════════════════
def page_stations():
    back_button()
    st.title("🚉 Station lookup")
    names = station_names()
    sq = stn.qdate.max()
    st.markdown(f"Pick a station to see how often its trains arrive on time. Station figures cover "
                f"{quarter_label(stn.qdate.min())} to {quarter_label(sq)}.")
    if "stn_sel" not in st.session_state and st.query_params.get("station"):
        raw = st.query_params.get("station")
        if raw in names:
            st.session_state["stn_sel"] = raw
        else:
            note(f"We couldn't find the station “{esc(raw)}” from the link.", "warn")
    station = st.selectbox("Find a station", names, index=None, key="stn_sel", placeholder="Type a station name…")
    summ = ins.station_quarter_summary(stn, station, sq) if station else None
    sync_url(station=station)
    show_change(change_message(
        "stn", station, summ["pct"] if summ else None,
        lambda ps, pv, s, v: ins.describe_generic_change(
            "the station", ps or "none", s or "none", "Trains on time", ins.fmt_value(pv), ins.fmt_value(v),
            ins.fmt_delta(pv, v), quarter_label(sq))))
    if station:
        if summ:
            allst = ins.station_table(stn[stn.qdate == sq])
            avg = ins.weighted_mean(allst["punctuality"], allst["stops"])
            gap = round(round(summ["pct"], 1) - round(avg, 1), 1)
            st.markdown(f"In **{quarter_label(sq)}**, **{summ['pct']:.1f}%** of trains arrived on time at **{esc(station)}**, "
                        f"against {avg:.1f}% across all stations.")
            one = stn[(stn.station == station) & stn.time_to_3_pct.notna()].copy()
            one["when"] = one.qdate
            q = one.groupby("when").apply(lambda x: ins.weighted_mean(x.time_to_3_pct, x.scheduled_stops),
                                          include_groups=False).reset_index(name="pct")
            fig = px.line(q, x="when", y="pct", markers=True,
                          labels={"when": "Quarter", "pct": "% of trains on time (within 3 minutes)"},
                          title=f"{station}: {abs(gap):.1f} points {'above' if gap >= 0 else 'below'} the average station "
                                f"in {quarter_label(sq)}")
            fig.add_hline(y=THRESHOLD, line_dash="dash", line_color=AMBER, annotation_text="80%")
            show(fig, 340)
        else:
            note(f"No data for {esc(station)} in {quarter_label(sq)} because "
                 f"{esc(ins.station_missing_reason(stn, station, sq))}", "warn")
        with st.expander("Why this number?"):
            st.markdown("**Source:** ORR Table 3130 (on-time and cancellations by station). Each station is timed in 4-week "
                        "periods; a quarter averages the periods inside it, weighted by scheduled stops, using only periods "
                        "with a published figure. On time = within 3 minutes of the timetable.")

    st.markdown("### Best and worst stations")
    with st.expander("For analysts: filter out small stations"):
        min_stops = st.slider("Minimum scheduled stops over the whole period", 1000, 100000, 10000, step=1000,
                              key="stn_min", help="Stations with very few stops have unreliable averages.")
    min_stops = st.session_state.get("stn_min", 10000)
    s = station_summary()
    shown = s[(s.stops >= min_stops)].dropna(subset=["punctuality"])
    show_change(change_message(
        "stnmin", min_stops, len(shown),
        lambda ps, pv, s_, v: ins.describe_generic_change(
            "the minimum number of stops", f"{ps:,}", f"{s_:,}", "Stations shown", f"{pv:,}", f"{v:,}",
            ins.fmt_delta(pv, v, " stations", 0), "the rankings below")))
    st.caption(f"{len(shown):,} stations are ranked. {len(s) - len(shown):,} are hidden because they have fewer than "
               f"{min_stops:,} scheduled stops in the data (or no published figure), which makes their average unreliable.")
    t1, t2 = st.tabs(["⏱ On time", "🚫 Cancellations"])
    with t1:
        best = shown.nlargest(15, "punctuality").sort_values("punctuality")
        worst = shown.nsmallest(15, "punctuality").sort_values("punctuality", ascending=False)
        c1, c2 = st.columns(2)
        for col, data, title, colour in [
                (c1, best, f"Best: {best.iloc[-1].station} at {best.iloc[-1].punctuality:.0f}% on time", TEAL),
                (c2, worst, f"Worst: {worst.iloc[-1].station} at {worst.iloc[-1].punctuality:.0f}% on time", RED)]:
            fig = go.Figure(go.Bar(x=data.punctuality, y=data.station, orientation="h", marker_color=colour,
                                   text=[f"{v:.0f}%" for v in data.punctuality], textposition="outside", cliponaxis=False))
            fig.update_layout(title=title, xaxis_title="% of trains on time (within 3 minutes)", xaxis_range=[0, 115])
            fig.update_yaxes(dtick=1)
            with col:
                show(fig, 24 * 15 + 110)
        st.caption(f"Averages cover {quarter_label(stn.qdate.min())} to {quarter_label(sq)}, weighted by scheduled stops.")
    with t2:
        c = shown.dropna(subset=["cancellations"]).nlargest(15, "cancellations").sort_values("cancellations")
        fig = go.Figure(go.Bar(x=c.cancellations, y=c.station, orientation="h", marker_color=AMBER,
                               text=[f"{v:.1f}%" for v in c.cancellations], textposition="outside", cliponaxis=False))
        fig.update_layout(title=f"{c.iloc[-1].station} cancels the most trains: {c.iloc[-1].cancellations:.1f}% of those scheduled",
                          xaxis_title="% of scheduled trains cancelled", xaxis_range=[0, c.cancellations.max() * 1.25])
        fig.update_yaxes(dtick=1)
        show(fig, 24 * 15 + 110)


# ══════════════════════════════ FORECAST ═════════════════════════════════════
def page_forecast():
    back_button()
    st.title("🔮 Forecast for next quarter")
    if bundle is None:
        st.warning("The forecast model is not available right now.")
        return
    fcs = all_forecasts()
    st.markdown(f"Using each company's results up to **{L_LATEST}**, we estimate how likely it is that **more than "
                f"{THRESHOLD:.0f}% of its trains will arrive on time in {L_NEXT}**. "
                "Forecasts are estimates, never certainties.")
    ordered = fcs.sort_values("p_below", ascending=False)["operator"].tolist()
    carried = st.session_state.get("_carry_op")
    url_choice("operator", ordered, carried if carried in ordered else ordered[0], "fc_op", "the train company")
    op = st.selectbox("Which train company?", ordered, key="fc_op", help="Listed from least to most likely to be above 80%.")
    sync_url(operator=op)
    fc, reason = forecast_for(op)
    show_change(change_message(
        "fc", op, fc["p_above"],
        lambda ps, pv, s, v: ins.describe_generic_change(
            "the train company", ps, s, f"The chance of being above {THRESHOLD:.0f}% on time",
            ins.format_probability(pv), ins.format_probability(v), None, L_NEXT)))
    st.markdown(answer_html(op, "forecast", next_q, None, None, fc, reason), unsafe_allow_html=True)
    with st.expander("Why this number?"):
        st.markdown(why_number_md(op, "forecast", next_q, None, None, fc))

    st.markdown(f"### All train companies: outlook for {L_NEXT}")
    t = fcs.sort_values("p_below", ascending=False)
    tbl = pd.DataFrame({
        "Train company": t.operator,
        f"% on time, {L_LATEST}": t.cur_pct.round(1),
        f"Outlook for {L_NEXT}": [f"{ins.traffic_light(p, c >= THRESHOLD)[0]} {ins.traffic_light(p, c >= THRESHOLD)[1]}"
                                  for p, c in zip(t.p_below, t.cur_pct)],
        f"Chance of being above {THRESHOLD:.0f}%": [ins.format_probability(1 - p) for p in t.p_below]})
    st.dataframe(tbl, hide_index=True, column_config={f"% on time, {L_LATEST}": st.column_config.NumberColumn(format="%.1f")})
    active = pq[pq.qdate == latest_q].operator.tolist()
    missing = [(o, ins.forecast_unavailable_reason(feats, o, latest_q)) for o in active if o not in set(fcs.operator)]
    if missing:
        st.caption("Not included: " + " ".join(f"{o}: {r}" for o, r in missing))
    st.caption("🟢 likely to be above 80% · 🟠 uncertain · 🔴 unlikely to be above 80%. The colour is always paired with words.")

    with st.expander("For analysts: what-if panel, accuracy and model details"):
        w, a, m = st.tabs(["🎛 What-if panel", "🎯 How accurate has it been?", "🧪 Model details"])
        with w:
            what_if(op, fc)
        with a:
            accuracy_tab()
        with m:
            model_tab()


def what_if(op, fc):
    r = fc["row"]
    qc = r["qdate"]
    L = [quarter_label(qc), quarter_label(qc - pd.DateOffset(months=3)), quarter_label(qc - pd.DateOffset(months=6)),
         quarter_label(qc - pd.DateOffset(months=9))]
    med = bundle["medians"]
    real = {"cur": float(r.cur_pct), "p1": float(r.prev1_pct), "p2": float(r.prev2_pct), "seas": float(r.seasonal_pct),
            "delay": float(r.delay_per_1k_stops if pd.notna(r.delay_per_1k_stops) else med["delay_per_1k_stops"]),
            "nr": float(r.nr_share if pd.notna(r.nr_share) else med["nr_share"]),
            "wx": float(r.weather_share if pd.notna(r.weather_share) else med["weather_share"])}
    dmax = float(round(np.nanpercentile(feats["delay_per_1k_stops"].dropna(), 99.5) * 1.2))
    k = op.replace(" ", "_")
    key = {n: f"wi_{n}_{k}" for n in real}
    shown = {"cur": round(real["cur"], 1), "p1": round(real["p1"], 1), "p2": round(real["p2"], 1),
             "seas": round(real["seas"], 1), "delay": float(np.clip(round(real["delay"]), 0, dmax)),
             "nr": float(np.clip(round(real["nr"] * 100), 0, 100)), "wx": float(np.clip(round(real["wx"] * 100), 0, 100))}
    for n, v in shown.items():
        st.session_state.setdefault(key[n], float(np.clip(v, 40, 100)) if n in ("cur", "p1", "p2", "seas") else v)

    def _reset(vals):
        for kk, v in vals.items():
            st.session_state[kk] = v
    st.caption("Sliders start at this company's real figures. Move any of them to test a scenario.")
    st.button("↺ Reset to real values", on_click=_reset, args=({key[n]: (float(np.clip(shown[n], 40, 100)) if n in
              ("cur", "p1", "p2", "seas") else shown[n]) for n in shown},), key=f"reset_{k}")
    s1, s2 = st.columns(2)
    new = {}
    new["cur"] = s1.slider(f"% on time in {L[0]} (latest quarter)", 40.0, 100.0, step=0.1, key=key["cur"])
    new["p1"] = s1.slider(f"% on time in {L[1]}", 40.0, 100.0, step=0.1, key=key["p1"])
    new["p2"] = s1.slider(f"% on time in {L[2]}", 40.0, 100.0, step=0.1, key=key["p2"])
    new["seas"] = s1.slider(f"% on time in {L[3]} (same quarter, a year earlier)", 40.0, 100.0, step=0.1, key=key["seas"])
    new["delay"] = s2.slider(f"Delay minutes per 1,000 stops in {L[0]}", 0.0, dmax, step=1.0, key=key["delay"])
    new["nr"] = s2.slider(f"% of {L[0]} delay caused by Network Rail", 0.0, 100.0, step=1.0, key=key["nr"])
    new["wx"] = s2.slider(f"% of {L[0]} delay caused by severe weather", 0.0, 100.0, step=1.0, key=key["wx"])

    # untouched sliders use the exact real value; changed ones use the user's value
    use = {n: (new[n] / 100 if n in ("nr", "wx") else new[n]) if new[n] != shown[n] else real[n] for n in real}
    changed = {n: use[n] for n in real if new[n] != shown[n]}
    base = ins.predict_risk(bundle["model"], bundle["medians"], ins.scenario_row(r, real), FEATURES)
    scen = ins.predict_risk(bundle["model"], bundle["medians"], ins.scenario_row(r, use), FEATURES)
    above = r.cur_pct >= THRESHOLD
    word = "stay above" if above else "get back above"
    b_disp, s_disp = round(ins.capped(base["p_above"]) * 100), round(ins.capped(scen["p_above"]) * 100)

    labels = {"cur": f"% on time, {L[0]}", "p1": f"% on time, {L[1]}", "p2": f"% on time, {L[2]}",
              "seas": f"% on time, {L[3]}", "delay": "Delay minutes per 1,000 stops", "nr": "% of delay from Network Rail",
              "wx": "% of delay from severe weather"}
    fmt = {"cur": "{:.1f}%", "p1": "{:.1f}%", "p2": "{:.1f}%", "seas": "{:.1f}%", "delay": "{:,.0f}", "nr": "{:.0f}%", "wx": "{:.0f}%"}
    rows = [{"Input": labels[n], "Original": fmt[n].format(shown[n]), "Your value": fmt[n].format(new[n]),
             "Changed?": "✔ changed" if n in changed else ""} for n in real]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    m1, m2 = st.columns(2)
    m1.metric(f"Original forecast ({L_NEXT})", f"{ins.format_probability(base['p_above'])}",
              help=f"Chance of being above {THRESHOLD:.0f}% on time")
    m2.metric("Your scenario", f"{ins.format_probability(scen['p_above'])}",
              None if b_disp == s_disp else f"{s_disp - b_disp:+d} points")
    st.caption(f"Both numbers are the chance of being above {THRESHOLD:.0f}% on time, shown between 5% and 95% because "
               "nothing is certain.")
    if not changed:
        st.markdown("You haven't changed anything yet. Drag a slider to try a scenario.")
    else:
        single = ins.one_at_a_time(bundle["model"], bundle["medians"], FEATURES, r, real, changed)
        eff = [(abs(round(ins.capped(1 - x["p_below"]) * 100) - b_disp), x) for x in single]
        top_eff, top = max(eff, key=lambda z: z[0])
        if top_eff < 1:
            st.markdown("None of your changes moves the forecast noticeably.")
        else:
            n = top["key"]
            st.markdown(f"**Mostly because** you changed **{labels[n].lower()}** from {fmt[n].format(shown[n])} to "
                        f"{fmt[n].format(new[n])}. On its own, that moves the chance of being above {THRESHOLD:.0f}% from "
                        f"{ins.format_probability(base['p_above'])} to {ins.format_probability(1 - top['p_below'])}. "
                        "Changes affect each other, so individual effects don't simply add up.")


def accuracy_tab():
    bands = metrics.get("calibration_bands", [])
    if not bands:
        st.info("Accuracy details are not available.")
        return
    st.markdown(f"We tested the model on **{metrics['test_rows']} operator-quarters** from {metrics['test_from']} to "
                f"{metrics['test_to']} that it had never seen. Here is how often it was right, by the risk it gave:")
    t = pd.DataFrame([{"Our estimated chance of falling below 80%": f"{b['lo']:.0%} to {b['hi']:.0%}",
                       "Past cases": b["n"],
                       "Times it really fell below 80%": f"{round(b['actual_rate'] * b['n'])} of {b['n']}" if b["actual_rate"] is not None else "–"}
                      for b in bands])
    st.dataframe(t, hide_index=True)
    cal = metrics.get("calibration", {})
    prev, now = cal.get("previous_balanced_C1", {}), cal.get("deployed_unweighted_tunedC", {})
    if prev and now:
        st.markdown(f"**Why the app never shows 0% or 100%:** an earlier version gave extreme forecasts (under 2% or over 98%) "
                    f"for {prev['share_extreme_forecasts']:.0%} of test cases. Those extremes were mostly right, but we only "
                    f"have {metrics['test_rows']} correlated test cases, which cannot prove certainty. We re-tuned the model so "
                    f"only {now['share_extreme_forecasts']:.0%} are extreme, and we cap what is displayed at 5% to 95%. "
                    f"The trade-off: it is a little under-confident in the middle ranges (see the table).")


def model_tab():
    res = pd.DataFrame(metrics.get("results", {})).T.reset_index()
    res.columns = ["Model", "ROC AUC", "F1", "Precision", "Recall", "Accuracy"]
    st.dataframe(res, hide_index=True)
    note("<b>Honest reading:</b> UK rail is strongly seasonal (Oct–Dec is reliably the worst quarter), so simple rules like "
         "“same quarter last year” already score very high. The model matches or slightly beats them.")
    with st.expander("What do these scores mean?"):
        st.markdown("- **ROC AUC:** how well it ranks risky operators above safe ones (1.0 perfect, 0.5 a coin flip).\n"
                    "- **Precision:** when it says “at risk”, how often is it right?\n"
                    "- **Recall:** of the operators that really fell below 80%, how many did it catch?\n"
                    "- **F1:** one score balancing precision and recall. **Accuracy:** share of all calls that were right.\n"
                    "- **Baselines** are simple rules. A model only adds value if it beats them.")
    names = {"cur_pct": "On time, latest quarter", "prev1_pct": "On time, 1 quarter earlier",
             "prev2_pct": "On time, 2 quarters earlier", "seasonal_pct": "On time, same quarter last year",
             "roll4_pct": "Average of last 4 quarters", "delta1": "Change since previous quarter",
             "delay_per_1k_stops": "Delay minutes per 1,000 stops", "nr_share": "Network Rail share of delay",
             "weather_share": "Weather share of delay", "traincrew_share": "Traincrew share of delay",
             "tgt_q1": "Forecast quarter is Apr–Jun", "tgt_q2": "Forecast quarter is Jul–Sep",
             "tgt_q3": "Forecast quarter is Oct–Dec", "tgt_q4": "Forecast quarter is Jan–Mar"}
    co = pd.Series(metrics.get("coefficients", {})).rename(index=names).sort_values()
    fig = px.bar(co, orientation="h", labels={"value": "Effect on risk (standardised)", "index": ""},
                 title="What drives the forecast overall (right = more risk of falling below 80%)")
    fig.update_layout(showlegend=False)
    fig.update_traces(marker_color=BLUE)
    show(fig, 460)
    st.caption(f"Model: {metrics.get('deployed_model', 'logistic regression')}.")


# ══════════════════════════════ DATA AND METHOD ══════════════════════════════
def page_method():
    back_button()
    sync_url()
    st.title("ℹ️ Data and method")
    st.markdown(f"**UK Train Delay Analytics** answers three questions: *which train companies are reliable, why do trains run "
                f"late, and who is at risk next quarter?* It is an independent portfolio project by **{AUTHOR}**, built on "
                "official rail statistics. It is **not** a live service: it does not tell you about today's trains.")
    c1, c2 = st.columns(2)
    c1.link_button("💻 Source code on GitHub", GITHUB_URL)
    if LINKEDIN_URL:
        c2.link_button("💼 Connect on LinkedIn", LINKEDIN_URL)

    st.subheader("Where the data comes from")
    st.markdown(f"All data is published free by the **Office of Rail and Road (ORR)**, Britain's rail regulator, on the "
                f"[ORR Data Portal]({ORR_URL}) under the Open Government Licence v3.0.\n\n"
                "| ORR table | What it contains | Used for |\n|---|---|---|\n"
                "| **3184** Delay minutes by operator and cause | Minutes of delay per 4-week period, split into 15 causes | Why trains are late |\n"
                "| **3133** Train punctuality by operator | % of trains within 3 minutes of timetable, per quarter since 2014 | Compare operators, forecast |\n"
                "| **3130** On-time and cancellations by station | Per station, per 4-week period (April 2023 onwards) | Station lookup |")
    st.markdown(f"**Data last updated:** {info.get('built_at_utc', 'unknown')} UTC. Train figures run to **{L_LATEST}**"
                f"{' (provisional, so ORR may revise it)' if latest_provisional else ''}; delay causes to "
                f"**{dc.year_label.max()}**. Updating is currently a manual download from ORR.")

    st.subheader("What we set aside while cleaning (nothing is dropped silently)")
    cl = info.get("cleaning", {})
    if cl:
        ex = cl.get("punctuality_quarters_excluded_no_trains_recorded", [])
        st.markdown(f"- **{len(ex)} operator-quarter(s) excluded because no trains were recorded:** "
                    f"{', '.join(ex) if ex else 'none'}. A quarter with no trains is missing data, not 0% on time.\n"
                    f"- **{cl.get('punctuality_operator_names_with_footnote_tags_cleaned', 0)} operator names** had ORR "
                    "footnote tags (such as “[note 4]”) removed so the same company isn't counted twice.\n"
                    f"- **{cl.get('punctuality_duplicate_rows_removed', 0)} duplicate rows** removed.\n"
                    f"- **{cl.get('station_rows_with_no_3min_figure', 0):,} of {cl.get('station_rows_total', 0):,} station rows** "
                    "have no published on-time figure. They are left out of station averages (not counted as 0%).")
    st.markdown("- Forecasts need four quarters in a row of history, so new operators get no forecast and the page says why.\n"
                "- If a delay detail is missing for a forecast, a typical value is used and the confidence is lowered. "
                "The forecast card tells you when this happens.")

    st.subheader("How the forecast works")
    st.markdown("A **logistic regression** model looks at an operator's last four quarters, its momentum, delay minutes per "
                "1,000 stops, the share of delay caused by Network Rail, weather and its own staff, and the season. It "
                f"estimates the chance of **fewer than {THRESHOLD:.0f}%** of trains being on time next quarter. It was "
                "tested only on future quarters and compared with simple rules (see the forecast page, “For analysts”).")

    st.subheader("Known limitations")
    cal = metrics.get("calibration", {})
    now = cal.get("deployed_unweighted_tunedC", {})
    st.markdown(
        "- **Seasonality does most of the work.** Oct–Dec is reliably the worst quarter, so “same quarter last year” is "
        "already a strong forecast. Our model matches or only slightly beats it.\n"
        f"- **Forecasts are estimates, not certainties.** The model was tested on just {metrics.get('test_rows', '?')} "
        "operator-quarters from 24 operators, and neighbouring quarters are closely linked, so the real evidence is "
        "thinner than the number suggests. An earlier version often gave 0% or 100%; we re-tuned it, cap what is shown at "
        "5% to 95%, and show past accuracy beside each forecast. It is slightly under-confident in the middle ranges"
        f"{' (calibration error ' + format(now.get('ece', 0), '.3f') + ')' if now else ''}.\n"
        f"- **The data lags.** Figures run to {L_LATEST}, and the “next quarter” forecast is for {L_NEXT}. Data is refreshed "
        "by hand, so ORR may have published newer figures.\n"
        "- **One measure, one threshold.** We use “within 3 minutes” and forecast against 80%. Cancellations and the 15-minute "
        "measure are not forecast.\n"
        "- **Delay causes are by 4-week period.** We group periods into quarters (periods 10 to 13 are Jan–Mar and cover "
        "16 weeks, not 12), so delay minutes per 1,000 stops is slightly inflated in Jan–Mar.\n"
        "- **Station data is separate from operator data.** Stations cover April 2023 onwards and are not split by "
        "operator here, so we cannot say which company's trains are late at a station.\n"
        "- **Delay minutes are not adjusted for size** unless you use the “per 1,000 stops” view.\n"
        "- **Provisional and revised figures.** ORR can revise recent quarters.\n"
        "- **Small samples:** averages over a few years, or for small stations, can be moved by one unusual quarter.\n"
        "- **Independent project.** Not an official ORR product, and not advice for travel or safety decisions.")

    st.subheader("Built with")
    st.markdown("Python · pandas · scikit-learn · SQLite · Plotly · Streamlit · AWS S3 and Lambda (automatic cleaning of "
                "newly uploaded ORR files) · GitHub. Planned: a cloud database, always-on hosting, and a data lake.")
    st.caption("Contains public sector information licensed under the Open Government Licence v3.0.")


# ══════════════════════════════ DOWNLOAD ═════════════════════════════════════
def page_download():
    back_button()
    sync_url()
    st.title("⬇️ Download the data")
    st.markdown("These are the cleaned tables behind every chart. Choose one to preview it, then download the full file as CSV.")
    desc = {"delay_causes": "Delay minutes by train company, 4-week period and cause (ORR Table 3184).",
            "operator_punctuality": "% of trains on time by train company and quarter (ORR Table 3133).",
            "station_performance": "On-time and cancellation figures by station and 4-week period (ORR Table 3130)."}
    table = st.selectbox("Which table?", list(desc))
    st.caption(desc[table])
    with closing(sqlite3.connect(DB_PATH.as_posix())) as conn:
        n = pd.read_sql(f"SELECT COUNT(*) n FROM {table}", conn)["n"].iloc[0]
        prev = pd.read_sql(f"SELECT * FROM {table} LIMIT 500", conn)
    st.caption(f"Preview of the first 500 of {n:,} rows. The download contains all {n:,} rows.")
    st.dataframe(prev)
    st.download_button(f"⬇️ Download {table}.csv ({n:,} rows)", csv_bytes(table), f"{table}.csv", "text/csv")


# ══════════════════════════════ ROUTER ═══════════════════════════════════════
PAGE_FUNCS = {HOME: home, "📉 Why trains are late": page_late, "🏆 Compare operators": page_compare,
              "🚉 Station lookup": page_stations, "🔮 Forecast for next quarter": page_forecast,
              "ℹ️ Data and method": page_method, "⬇️ Download the data": page_download}
try:
    PAGE_FUNCS[page]()
    if page != HOME:
        st.divider()
        data_footer()
except Exception as e:  # noqa: BLE001 - users see a friendly message, details go to the log
    friendly_error(e)
