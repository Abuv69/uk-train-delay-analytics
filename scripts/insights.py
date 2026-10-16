# scripts/insights.py
"""
Pure, testable calculations behind every sentence the dashboard shows.
No Streamlit imports here: the app only formats and displays what this module returns,
so what the tests check is exactly what the user reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.features import PERIOD_TO_QUARTER, THRESHOLD, quarter_label

SEASON_BY_MONTH = {1: "Jan–Mar", 4: "Apr–Jun", 7: "Jul–Sep", 10: "Oct–Dec"}

# Plain-English grouping of ORR's 15 delay causes (column names in delay_causes).
CAUSE_GROUPS = {
    "Track, signals and railway equipment (Network Rail)": ["nr_track", "nr_non_track_assets"],
    "Network Rail management and other causes": ["nr_network_mgmt"],
    "Events outside the railway (e.g. trespass, vandalism)": ["nr_external"],
    "Severe weather": ["nr_severe_weather"],
    "The train company's own trains, staff and stations": [
        "toc_self_fleet", "toc_self_operations", "toc_self_stations", "toc_self_traincrew", "toc_self_other"],
    "Other train companies getting in the way": [
        "toc_on_toc_fleet", "toc_on_toc_operations", "toc_on_toc_stations",
        "toc_on_toc_traincrew", "toc_on_toc_other"],
}


# ── formatting helpers ────────────────────────────────────────────────────────
def season_name(ts) -> str:
    return SEASON_BY_MONTH[pd.Timestamp(ts).month]


def in_ten(pct: float) -> str:
    """71.4 -> 'about 7 in 10'. 97+ -> 'almost all', under 3 -> 'almost none'. Clipped to 1-9 otherwise."""
    if pct is None or pd.isna(pct):
        return "no data"
    if pct >= 97:
        return "almost all"
    if pct < 3:
        return "almost none"
    return f"about {int(min(9, max(1, round(pct / 10))))} in 10"


def format_probability(p: float) -> str:
    """Never shows a bare 0% or 100%: forecasts are never certain."""
    if p >= 0.95:
        return "over 95%"
    if p <= 0.05:
        return "under 5%"
    return f"about {int(round(p * 100))}%"


def fmt_value(x, unit: str = "%", decimals: int = 1) -> str:
    if x is None or pd.isna(x):
        return "no data"
    return f"{round(float(x), decimals):.{decimals}f}{unit}"


def fmt_delta(old, new, unit: str = " percentage points", decimals: int = 1) -> str | None:
    """Difference computed from the *displayed* (rounded) values so text and numbers always agree."""
    if old is None or new is None or pd.isna(old) or pd.isna(new):
        return None
    d = round(round(float(new), decimals) - round(float(old), decimals), decimals)
    return f"{d:+.{decimals}f}{unit}"


# ── on-time calculations ──────────────────────────────────────────────────────
def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    m = values.notna() & weights.notna() & (weights > 0)
    return float((values[m] * weights[m]).sum() / weights[m].sum()) if m.any() else float("nan")


def national_on_time(pq: pd.DataFrame, qdate) -> float:
    """National % on time in a quarter, weighted by each operator's recorded stops."""
    rows = pq[pq["qdate"] == pd.Timestamp(qdate)]
    return weighted_mean(rows["pct_on_time_3m"], rows["recorded_stops"])


def operator_quarter(pq: pd.DataFrame, operator: str, qdate) -> dict | None:
    r = pq[(pq["operator"] == operator) & (pq["qdate"] == pd.Timestamp(qdate))]
    if r.empty:
        return None
    r = r.iloc[0]
    return {"pct": float(r["pct_on_time_3m"]), "stops": float(r["recorded_stops"]),
            "provisional": bool(r.get("provisional", False))}


def qoq_change(pq: pd.DataFrame, operator: str, qdate) -> float | None:
    """Change in % on time versus the previous quarter, in percentage points (from rounded values)."""
    cur = operator_quarter(pq, operator, qdate)
    prev = operator_quarter(pq, operator, pd.Timestamp(qdate) - pd.DateOffset(months=3))
    if cur is None or prev is None:
        return None
    return round(round(cur["pct"], 1) - round(prev["pct"], 1), 1)


def rank_in_quarter(pq: pd.DataFrame, operator: str, qdate) -> dict | None:
    rows = pq[pq["qdate"] == pd.Timestamp(qdate)].dropna(subset=["pct_on_time_3m"])
    if operator not in set(rows["operator"]):
        return None
    ordered = rows.sort_values("pct_on_time_3m", ascending=False).reset_index(drop=True)
    rank = int(ordered.index[ordered["operator"] == operator][0]) + 1
    return {"rank": rank, "n": len(ordered), "national": national_on_time(pq, qdate)}


def missing_reason(pq: pd.DataFrame, operator: str, qdate) -> str:
    """Plain-English reason an operator has no figure for a quarter (never silently blank)."""
    own = pq[pq["operator"] == operator]
    q = pd.Timestamp(qdate)
    if own.empty:
        return f"there are no figures for {operator} in the data."
    first, last = own["qdate"].min(), own["qdate"].max()
    if q < first:
        return (f"the first quarter on record for {operator} is {quarter_label(first)}, "
                "so it had not started (or was not reported) yet.")
    if q > last:
        return (f"the latest quarter on record for {operator} is {quarter_label(last)}. "
                "Either it stopped running or the Office of Rail and Road has not published a later figure.")
    return (f"the Office of Rail and Road recorded no trains for {operator} in that quarter "
            "(for example because services were suspended), so it is left out rather than shown as 0%.")


# ── causes and seasons ────────────────────────────────────────────────────────
def cause_group_shares(dc: pd.DataFrame, operator: str, year_label: str) -> pd.DataFrame:
    d = dc[(dc["operator"] == operator) & (dc["year_label"] == year_label)]
    rows = []
    for group, cols in CAUSE_GROUPS.items():
        rows.append({"group": group, "minutes": float(d[cols].sum().sum())})
    out = pd.DataFrame(rows)
    total = out["minutes"].sum()
    out["share"] = out["minutes"] / total if total > 0 else np.nan
    return out.sort_values("minutes", ascending=False, kind="stable").reset_index(drop=True)


def seasonal_profile(pq: pd.DataFrame, operator: str, quarters: int = 12) -> dict | None:
    """Average % on time per season over the operator's last N quarters (needs >= 4 quarters)."""
    own = pq[pq["operator"] == operator].sort_values("qdate").tail(quarters)
    if len(own) < 4:
        return None
    own = own.assign(season=own["qdate"].map(season_name))
    by = own.groupby("season")["pct_on_time_3m"].agg(["mean", "count"])
    return {"by_season": by, "overall": float(own["pct_on_time_3m"].mean()), "n_quarters": len(own),
            "from": own["qdate"].min(), "to": own["qdate"].max()}


# ── stations ──────────────────────────────────────────────────────────────────
def add_station_quarter(st: pd.DataFrame) -> pd.DataFrame:
    """Map ORR 4-week periods onto calendar quarters (P1-3 Apr-Jun, P4-6 Jul-Sep, P7-9 Oct-Dec, P10-13 Jan-Mar)."""
    out = st.copy()
    start = out["year_label"].str.extract(r"Apr (\d{4})")[0].astype(float)
    q = out["period_num"].astype("Int64").map(PERIOD_TO_QUARTER)
    month = q.map({"Q1": 4, "Q2": 7, "Q3": 10, "Q4": 1})
    year = start + (q == "Q4").astype(int)
    out["qdate"] = pd.to_datetime(dict(year=year.astype("Int64"), month=month.astype("Int64"), day=1), errors="coerce")
    return out


def station_quarter_summary(st: pd.DataFrame, station: str, qdate) -> dict | None:
    d = st[(st["station"] == station) & (st["qdate"] == pd.Timestamp(qdate))]
    d = d[d["time_to_3_pct"].notna() & (d["scheduled_stops"] > 0)]
    if d.empty:
        return None
    return {"pct": weighted_mean(d["time_to_3_pct"], d["scheduled_stops"]), "stops": float(d["scheduled_stops"].sum())}


# ── forecast helpers ──────────────────────────────────────────────────────────
def outlook(p_below: float) -> tuple[str, str]:
    """(label, colour) for 'will the operator be above 80%?'. Thresholds are on P(above 80%)."""
    p_above = 1 - p_below
    if p_above >= 0.90:
        return "very likely", "green"
    if p_above >= 0.70:
        return "likely", "green"
    if p_above > 0.30:
        return "uncertain", "amber"
    if p_above > 0.10:
        return "unlikely", "red"
    return "very unlikely", "red"


def outlook_sentence(p_below: float, currently_above: bool, quarter: str) -> str:
    label, _ = outlook(p_below)
    verb = "stay above" if currently_above else "get back above"
    if label == "uncertain":
        return f"It is uncertain whether it will {verb} {THRESHOLD:.0f}% in {quarter}."
    return f"It is {label} that it will {verb} {THRESHOLD:.0f}% in {quarter}."


TRAFFIC_LIGHT = {"green": ("🟢", "Likely to stay above 80%"),
                 "amber": ("🟠", "Uncertain"),
                 "red": ("🔴", "Unlikely to reach 80%")}


def band_for(p_below: float, bands: list[dict]) -> dict | None:
    for b in bands:
        if b["lo"] <= p_below < b["hi"] or (b["hi"] >= 1.0 and p_below >= b["lo"]):
            return b
    return None


def confidence(p_below: float, imputed: list[str], bands: list[dict]) -> dict:
    """high / medium / low, with the reason in plain English.
    Start from how decisive the forecast is, then downgrade one level for (a) filled-in missing inputs
    and (b) probability ranges that have too few past cases to judge."""
    p_above = 1 - p_below
    if p_above >= 0.90 or p_above <= 0.10:
        level = 2
    elif 0.30 < p_above < 0.70:
        level = 0
    else:
        level = 1
    reasons = []
    band = band_for(p_below, bands)
    if band and band["n"] < 10:
        level -= 1
        reasons.append(f"only {band['n']} past cases fell in this range, which is too few to judge it well")
    if imputed:
        level -= 1
        reasons.append("some delay details were missing, so typical values were used for them")
    name = ["low", "medium", "high"][max(0, level)]
    if band and band["n"] >= 10 and band["actual_rate"] is not None:
        right = band["actual_rate"] if p_below >= 0.5 else 1 - band["actual_rate"]
        k = int(round(right * band["n"]))      # counts, not a bare 100%: a small past sample cannot prove certainty
        reasons.insert(0, f"in past tests, forecasts in this range were right in {k} of {band['n']} cases")
    return {"level": name, "reasons": reasons}


def predict_risk(model, medians: dict, row: pd.DataFrame, features: list[str]) -> dict:
    X = row[features].astype(float)
    imputed = [c for c in X.columns if pd.isna(X.iloc[0][c])]
    p = float(model.predict_proba(X.fillna(pd.Series(medians)))[0, 1])
    return {"p_below": p, "p_above": 1 - p, "imputed": imputed}


def forecast_unavailable_reason(feats: pd.DataFrame, operator: str, latest_q) -> str | None:
    """None if a forecast exists, otherwise why not."""
    r = feats[(feats["operator"] == operator) & (feats["qdate"] == pd.Timestamp(latest_q))]
    if r.empty:
        return (f"{operator} has no figure for {quarter_label(latest_q)}, which the forecast needs as its starting point.")
    r = r.iloc[0]
    missing = [n for c, n in [("prev1_pct", "the previous quarter"), ("prev2_pct", "two quarters earlier"),
                              ("seasonal_pct", "the same quarter a year earlier")] if pd.isna(r[c])]
    if missing:
        return (f"{operator} is missing figures for {', '.join(missing)}. The forecast needs a full year of history "
                "(four quarters in a row).")
    return None


# ── "What changed" text ───────────────────────────────────────────────────────
def _value_text(s: dict) -> str:
    if s["kind"] == "forecast":
        return f"{format_probability(s['p_above'])} chance of being above {THRESHOLD:.0f}% on time" \
            if s.get("p_above") is not None else "no forecast available"
    return f"{fmt_value(s['pct'])} of trains on time" if s.get("pct") is not None else "no data"


def describe_change(prev: dict | None, new: dict) -> str | None:
    """One plain-English line describing what a change in the search panel did. Uses the same snapshot
    numbers that the answer card shows, so text and card cannot disagree."""
    if prev is None:
        return None
    parts = []
    op_changed = prev["operator"] != new["operator"]
    q_changed = prev["qlabel"] != new["qlabel"]
    both_actual = prev["kind"] == "actual" and new["kind"] == "actual"
    has_nums = both_actual and prev.get("pct") is not None and new.get("pct") is not None
    delta = fmt_delta(prev.get("pct"), new.get("pct")) if has_nums else None

    if op_changed or q_changed:
        if has_nums and op_changed and not q_changed:
            parts.append(f"You switched from {prev['operator']} to {new['operator']}. On-time trains went from "
                         f"{fmt_value(prev['pct'])} to {fmt_value(new['pct'])} ({delta}) in {new['qlabel']}.")
        elif has_nums and q_changed and not op_changed:
            parts.append(f"You switched from {prev['qlabel']} to {new['qlabel']} for {new['operator']}. On-time trains "
                         f"went from {fmt_value(prev['pct'])} to {fmt_value(new['pct'])} ({delta}).")
        elif has_nums:
            parts.append(f"You changed from {prev['operator']} in {prev['qlabel']} ({fmt_value(prev['pct'])}) to "
                         f"{new['operator']} in {new['qlabel']} ({fmt_value(new['pct'])}), a difference of {delta}.")
        else:
            parts.append(f"You changed from {prev['operator']} in {prev['qlabel']} ({_value_text(prev)}) to "
                         f"{new['operator']} in {new['qlabel']} ({_value_text(new)}).")

    if prev.get("station") != new.get("station"):
        if new.get("station") is None:
            parts.append("You cleared the station, so only the operator result is shown.")
        elif new.get("station_pct") is None:
            parts.append(f"You picked {new['station']}, but there is no station figure for {new['qlabel']}.")
        else:
            parts.append(f"You picked {new['station']}, where {fmt_value(new['station_pct'])} of trains arrived on "
                         f"time in {new['qlabel']}.")
    return " ".join(parts) or None


def describe_generic_change(subject: str, old_label: str, new_label: str, metric: str,
                            old_value: str, new_value: str, delta: str | None, period: str) -> str:
    base = (f"You changed {subject} from {old_label} to {new_label}. {metric} went from {old_value} to "
            f"{new_value}")
    return f"{base} ({delta}) for {period}." if delta else f"{base} for {period}."


def describe_set_change(added: list[str], removed: list[str], values: dict, period: str) -> str | None:
    def item(op):
        v = values.get(op)
        return f"{op} ({fmt_value(v)} on time in {period})" if v is not None and not pd.isna(v) else f"{op} (no data for {period})"
    parts = []
    if added:
        parts.append("added " + ", ".join(item(o) for o in added))
    if removed:
        parts.append("removed " + ", ".join(item(o) for o in removed))
    return ("You " + " and ".join(parts) + ".") if parts else None


# ── more helpers used by the pages ────────────────────────────────────────────
DISPLAY_CAP = (0.05, 0.95)


def capped(p: float) -> float:
    """Probability as it may be *displayed*: never closer than 5% to certain."""
    return float(min(max(p, DISPLAY_CAP[0]), DISPLAY_CAP[1]))


def traffic_light(p_below: float, currently_above: bool) -> tuple[str, str, str]:
    """(emoji, text, colour). Always paired with text so colour is never the only signal."""
    _, colour = outlook(p_below)
    verb = "stay above" if currently_above else "get back above"
    text = {"green": f"Likely to {verb} {THRESHOLD:.0f}%",
            "amber": f"Uncertain whether it will {verb} {THRESHOLD:.0f}%",
            "red": f"Unlikely to {verb} {THRESHOLD:.0f}%"}[colour]
    return {"green": "🟢", "amber": "🟠", "red": "🔴"}[colour], text, colour


def station_table(st: pd.DataFrame) -> pd.DataFrame:
    """Per-station on-time % and cancellation %, weighted by scheduled stops. Rows with no figure are
    excluded from the averages (not counted as 0)."""
    d = st[st["scheduled_stops"].fillna(0) > 0]
    t = d[d["time_to_3_pct"].notna()]
    c = d[d["cancellations_pct"].notna()]
    out = pd.DataFrame({
        "punctuality": (t["time_to_3_pct"] * t["scheduled_stops"]).groupby(t["station"]).sum()
        / t["scheduled_stops"].groupby(t["station"]).sum(),
        "cancellations": (c["cancellations_pct"] * c["scheduled_stops"]).groupby(c["station"]).sum()
        / c["scheduled_stops"].groupby(c["station"]).sum(),
        "stops": d.groupby("station")["scheduled_stops"].sum()})
    return out.reset_index()


def station_missing_reason(st: pd.DataFrame, station: str, qdate) -> str:
    if station not in set(st["station"]):
        return f"the Office of Rail and Road's station tables do not list {station}."
    lo, hi = st["qdate"].min(), st["qdate"].max()
    q = pd.Timestamp(qdate)
    if q < lo or q > hi:
        return (f"the station tables only cover {quarter_label(lo)} to {quarter_label(hi)}.")
    return "the Office of Rail and Road published no reliable on-time figure for this station in that quarter."


SCENARIO_COLS = {"cur": "cur_pct", "p1": "prev1_pct", "p2": "prev2_pct", "seas": "seasonal_pct",
                 "delay": "delay_per_1k_stops", "nr": "nr_share", "wx": "weather_share"}


def scenario_row(row: pd.Series, vals: dict) -> pd.DataFrame:
    """The model's input row with what-if values applied and the derived inputs recomputed."""
    w = row.copy()
    for k, c in SCENARIO_COLS.items():
        if k in vals:
            w[c] = vals[k]
    w["roll4_pct"] = float(np.mean([w["cur_pct"], w["prev1_pct"], w["prev2_pct"], w["seasonal_pct"]]))
    w["delta1"] = float(w["cur_pct"] - w["prev1_pct"])
    return pd.DataFrame([w])


def one_at_a_time(model, medians: dict, features: list[str], row: pd.Series, orig: dict, new: dict) -> list[dict]:
    """For each changed input: the forecast if ONLY that input were changed (others kept at original)."""
    out = []
    for k in new:
        if k in orig and new[k] != orig[k]:
            vals = dict(orig)
            vals[k] = new[k]
            p = predict_risk(model, medians, scenario_row(row, vals), features)["p_below"]
            out.append({"key": k, "p_below": p})
    return out
