"""Unit tests for every calculation behind a sentence on the dashboard.
All expected values are worked out by hand in the comments."""
import random
import re

import numpy as np
import pandas as pd
import pytest

from scripts import insights as ins


# ── fixed sample data ────────────────────────────────────────────────────────
@pytest.fixture
def pq():
    rows = [("A", "2025-07-01", 90.0, 100, False), ("B", "2025-07-01", 60.0, 300, False),
            ("A", "2025-10-01", 95.0, 100, True), ("B", "2025-10-01", 65.5, 300, False),
            ("C", "2025-10-01", 80.0, 200, False),
            ("E", "2025-04-01", 70.0, 100, False), ("E", "2025-10-01", 75.0, 100, False)]   # E has a gap in Jul
    d = pd.DataFrame(rows, columns=["operator", "qdate", "pct_on_time_3m", "recorded_stops", "provisional"])
    d["qdate"] = pd.to_datetime(d["qdate"])
    return d


def test_national_on_time_is_weighted_by_stops(pq):
    # Jul 2025: (90*100 + 60*300) / 400 = 27000/400 = 67.5   (a simple mean would wrongly give 75)
    assert ins.national_on_time(pq, "2025-07-01") == pytest.approx(67.5)
    # Oct 2025: (95*100 + 65.5*300 + 80*200 + 75*100) / 700 = 52650/700 ... E also in Oct, so recompute by hand:
    # 9500 + 19650 + 16000 + 7500 = 52650 ; stops 100+300+200+100 = 700 ; 52650/700 = 75.2142857
    assert ins.national_on_time(pq, "2025-10-01") == pytest.approx(52650 / 700)


def test_operator_quarter_and_provisional_flag(pq):
    r = ins.operator_quarter(pq, "A", "2025-10-01")
    assert r == {"pct": 95.0, "stops": 100.0, "provisional": True}
    assert ins.operator_quarter(pq, "A", "2024-01-01") is None


def test_quarter_on_quarter_change(pq):
    assert ins.qoq_change(pq, "A", "2025-10-01") == 5.0      # 95.0 - 90.0
    assert ins.qoq_change(pq, "B", "2025-10-01") == 5.5      # 65.5 - 60.0
    assert ins.qoq_change(pq, "C", "2025-10-01") is None     # no previous quarter
    assert ins.qoq_change(pq, "E", "2025-10-01") is None     # gap in July -> no silent comparison with April


def test_rank_in_quarter(pq):
    r = ins.rank_in_quarter(pq, "B", "2025-10-01")
    # Oct 2025 order: A 95, C 80, E 75, B 65.5  -> B is 4th of 4
    assert (r["rank"], r["n"]) == (4, 4)
    assert ins.rank_in_quarter(pq, "A", "2025-10-01")["rank"] == 1
    assert ins.rank_in_quarter(pq, "Z", "2025-10-01") is None


def test_missing_reason_explains_each_case(pq):
    assert "first quarter on record" in ins.missing_reason(pq, "C", "2025-07-01")
    assert "latest quarter on record" in ins.missing_reason(pq, "A", "2026-01-01")
    assert "recorded no trains" in ins.missing_reason(pq, "E", "2025-07-01")
    assert "no figures" in ins.missing_reason(pq, "Z", "2025-07-01")


def test_cause_group_shares_sum_to_100_and_match_hand_calc():
    cols = [c for cs in ins.CAUSE_GROUPS.values() for c in cs]
    row = {c: 0.0 for c in cols}
    row.update(nr_track=100, nr_non_track_assets=50, nr_network_mgmt=150, nr_external=200,
               nr_severe_weather=50, toc_self_fleet=300, toc_on_toc_fleet=150)
    dc = pd.DataFrame([{"operator": "A", "year_label": "2025-26", **row},
                       {"operator": "B", "year_label": "2025-26", **{c: 999.0 for c in cols}}])
    out = ins.cause_group_shares(dc, "A", "2025-26").set_index("group")
    # total = 100+50+150+200+50+300+150 = 1000
    assert out["minutes"].sum() == 1000
    assert out.loc["Track, signals and railway equipment (Network Rail)", "share"] == pytest.approx(0.15)
    assert out.loc["Events outside the railway (e.g. trespass, vandalism)", "share"] == pytest.approx(0.20)
    assert out.loc["The train company's own trains, staff and stations", "share"] == pytest.approx(0.30)
    assert out.loc["Severe weather", "share"] == pytest.approx(0.05)
    assert out["share"].sum() == pytest.approx(1.0)
    assert ins.cause_group_shares(dc, "A", "2025-26").iloc[0]["group"].startswith("The train company's own")


def test_seasonal_profile():
    qd = pd.to_datetime(["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01",
                         "2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"])
    d = pd.DataFrame({"operator": "A", "qdate": qd, "pct_on_time_3m": [80, 90, 88, 70, 82, 92, 86, 72]})
    p = ins.seasonal_profile(d, "A")
    # Jan-Mar (80+82)/2=81 ; Apr-Jun (90+92)/2=91 ; Jul-Sep (88+86)/2=87 ; Oct-Dec (70+72)/2=71 ; overall 660/8=82.5
    assert p["by_season"].loc["Jan–Mar", "mean"] == 81
    assert p["by_season"].loc["Apr–Jun", "mean"] == 91
    assert p["by_season"].loc["Jul–Sep", "mean"] == 87
    assert p["by_season"].loc["Oct–Dec", "mean"] == 71
    assert p["overall"] == 82.5 and p["n_quarters"] == 8
    assert ins.seasonal_profile(d.head(3), "A") is None      # fewer than 4 quarters -> no profile


def test_station_quarters_and_weighted_average():
    rows = []
    for p in range(1, 14):
        rows.append({"station": "X", "year_label": "Apr 2025 to Mar 2026", "period_num": p,
                     "scheduled_stops": 100.0, "time_to_3_pct": np.nan})
    st = pd.DataFrame(rows)
    st.loc[0, ["scheduled_stops", "time_to_3_pct"]] = [100, 90]    # P1
    st.loc[1, ["scheduled_stops", "time_to_3_pct"]] = [300, 70]    # P2
    st.loc[9, "time_to_3_pct"] = 50                                # P10 -> Jan-Mar 2026
    st = ins.add_station_quarter(st)
    assert st.loc[0, "qdate"] == pd.Timestamp("2025-04-01")
    assert st.loc[9, "qdate"] == pd.Timestamp("2026-01-01")        # P10-13 belong to Jan-Mar of the NEXT calendar year
    # Q1: (90*100 + 70*300) / 400 = 30000/400 = 75.0 ; P3 has no figure and is excluded, not counted as 0
    assert ins.station_quarter_summary(st, "X", "2025-04-01")["pct"] == pytest.approx(75.0)
    assert ins.station_quarter_summary(st, "X", "2026-01-01")["pct"] == pytest.approx(50.0)
    assert ins.station_quarter_summary(st, "X", "2025-07-01") is None
    assert ins.station_quarter_summary(st, "Nowhere", "2025-04-01") is None


# ── plain-language formatting ────────────────────────────────────────────────
@pytest.mark.parametrize("pct,text", [(71.4, "about 7 in 10"), (51.8, "about 5 in 10"), (96.9, "about 9 in 10"),
                                      (97.0, "almost all"), (2.9, "almost none"), (3.0, "about 1 in 10"),
                                      (float("nan"), "no data")])
def test_in_ten(pct, text):
    assert ins.in_ten(pct) == text


def test_probabilities_never_display_as_zero_or_hundred():
    for p in np.linspace(0, 1, 1001):
        s = ins.format_probability(float(p))
        assert s not in ("0%", "100%", "about 0%", "about 100%")
    assert ins.format_probability(1.0) == "over 95%"
    assert ins.format_probability(0.0) == "under 5%"
    assert ins.format_probability(0.5) == "about 50%"
    assert ins.format_probability(0.734) == "about 73%"


def test_fmt_delta_uses_displayed_numbers():
    assert ins.fmt_delta(71.44, 84.56) == "+13.2 percentage points"   # displayed 71.4 -> 84.6
    assert ins.fmt_delta(84.6, 71.4) == "-13.2 percentage points"
    assert ins.fmt_delta(None, 5) is None


def test_outlook_labels_and_sentences():
    assert ins.outlook(0.03) == ("very likely", "green")
    assert ins.outlook(0.20) == ("likely", "green")
    assert ins.outlook(0.50) == ("uncertain", "amber")
    assert ins.outlook(0.80) == ("unlikely", "red")
    assert ins.outlook(0.97) == ("very unlikely", "red")
    assert ins.outlook_sentence(0.03, True, "Jan–Mar 2026") == "It is very likely that it will stay above 80% in Jan–Mar 2026."
    assert ins.outlook_sentence(0.98, False, "Jan–Mar 2026") == "It is very unlikely that it will get back above 80% in Jan–Mar 2026."
    assert ins.outlook_sentence(0.50, True, "Jan–Mar 2026") == "It is uncertain whether it will stay above 80% in Jan–Mar 2026."


BANDS = [{"lo": 0.0, "hi": 0.1, "n": 68, "actual_rate": 0.0}, {"lo": 0.1, "hi": 0.3, "n": 40, "actual_rate": 0.0},
         {"lo": 0.3, "hi": 0.7, "n": 43, "actual_rate": 0.5}, {"lo": 0.7, "hi": 0.9, "n": 37, "actual_rate": 0.97},
         {"lo": 0.9, "hi": 1.0, "n": 52, "actual_rate": 1.0}]


def test_confidence_levels_and_reasons():
    hi = ins.confidence(0.97, [], BANDS)
    assert hi["level"] == "high" and "right in 52 of 52 cases" in hi["reasons"][0]
    assert ins.confidence(0.50, [], BANDS)["level"] == "low"
    m = ins.confidence(0.80, [], BANDS)
    assert m["level"] == "medium" and "right in 36 of 37 cases" in m["reasons"][0]   # 0.97 * 37 = 35.9 -> 36
    low_hi = ins.confidence(0.02, [], BANDS)
    assert low_hi["level"] == "high" and "right in 68 of 68 cases" in low_hi["reasons"][0]
    imp = ins.confidence(0.97, ["nr_share"], BANDS)
    assert imp["level"] == "medium" and any("typical values" in r for r in imp["reasons"])
    sparse = ins.confidence(0.97, [], [{"lo": 0.9, "hi": 1.0, "n": 5, "actual_rate": 1.0}])
    assert sparse["level"] == "medium" and any("too few" in r for r in sparse["reasons"])


# ── "What changed" text must match the numbers shown ─────────────────────────
def snap(op, q, pct, kind="actual", p_above=None, station=None, station_pct=None):
    return {"operator": op, "qlabel": q, "kind": kind, "pct": pct, "p_above": p_above,
            "station": station, "station_pct": station_pct}


def test_what_changed_exact_wording():
    a, b = snap("Avanti West Coast", "Oct–Dec 2025", 71.44), snap("Northern Trains", "Oct–Dec 2025", 84.56)
    assert ins.describe_change(a, b) == ("You switched from Avanti West Coast to Northern Trains. On-time trains went "
                                         "from 71.4% to 84.6% (+13.2 percentage points) in Oct–Dec 2025.")
    c = snap("Northern Trains", "Jul–Sep 2025", 86.04)
    assert ins.describe_change(b, c) == ("You switched from Oct–Dec 2025 to Jul–Sep 2025 for Northern Trains. On-time "
                                         "trains went from 84.6% to 86.0% (+1.4 percentage points).")
    assert ins.describe_change(a, c) == ("You changed from Avanti West Coast in Oct–Dec 2025 (71.4%) to Northern Trains in "
                                         "Jul–Sep 2025 (86.0%), a difference of +14.6 percentage points.")
    assert ins.describe_change(a, a) is None and ins.describe_change(None, a) is None


def test_what_changed_forecast_and_missing_data():
    a = snap("Avanti West Coast", "Oct–Dec 2025", 71.44)
    f = snap("Avanti West Coast", "Jan–Mar 2026", None, kind="forecast", p_above=0.01)
    assert ins.describe_change(a, f) == ("You changed from Avanti West Coast in Oct–Dec 2025 (71.4% of trains on time) to "
                                         "Avanti West Coast in Jan–Mar 2026 (under 5% chance of being above 80% on time).")
    none = snap("Lumo", "Oct–Dec 2019", None)
    assert "no data" in ins.describe_change(a, none)


def test_what_changed_station_sentences():
    a = snap("A", "Oct–Dec 2025", 70.0)
    with_st = snap("A", "Oct–Dec 2025", 70.0, station="Leeds", station_pct=88.04)
    assert ins.describe_change(a, with_st) == "You picked Leeds, where 88.0% of trains arrived on time in Oct–Dec 2025."
    assert ins.describe_change(with_st, a) == "You cleared the station, so only the operator result is shown."
    no_fig = snap("A", "Oct–Dec 2025", 70.0, station="Leeds", station_pct=None)
    assert "no station figure for Oct–Dec 2025" in ins.describe_change(a, no_fig)


def test_what_changed_numbers_always_agree_with_snapshots():
    """Property test: for 300 random pairs, the three numbers in the text satisfy new - old == delta."""
    rnd = random.Random(7)
    for _ in range(300):
        old, new = rnd.uniform(40, 99.9), rnd.uniform(40, 99.9)
        text = ins.describe_change(snap("A", "Q", old), snap("B", "Q", new))
        nums = [float(x) for x in re.findall(r"[-+]?\d+\.\d", text)]
        assert len(nums) == 3
        assert nums[0] == round(old, 1) and nums[1] == round(new, 1)
        assert round(nums[1] - nums[0], 1) == pytest.approx(nums[2], abs=1e-9)


def test_generic_and_set_changes():
    assert ins.describe_generic_change(
        "the operator", "All operators", "Northern Trains", "Total delay minutes", "96.2M", "13.2M",
        ins.fmt_delta(96.2, 13.2, "M"), "all years") == (
        "You changed the operator from All operators to Northern Trains. Total delay minutes went from 96.2M to "
        "13.2M (-83.0M) for all years.")
    values = {"Northern Trains": 72.94, "ScotRail": 84.8}
    assert ins.describe_set_change(["Northern Trains"], ["ScotRail"], values, "Oct–Dec 2025") == (
        "You added Northern Trains (72.9% on time in Oct–Dec 2025) and removed ScotRail (84.8% on time in Oct–Dec 2025).")
    assert ins.describe_set_change([], [], {}, "x") is None
    assert "no data for Oct–Dec 2025" in ins.describe_set_change(["Ghost"], [], {}, "Oct–Dec 2025")


# ── helpers added for the redesigned pages ───────────────────────────────────
def test_traffic_light_always_pairs_colour_with_words():
    assert ins.traffic_light(0.03, True) == ("🟢", "Likely to stay above 80%", "green")
    assert ins.traffic_light(0.50, True) == ("🟠", "Uncertain whether it will stay above 80%", "amber")
    assert ins.traffic_light(0.98, False) == ("🔴", "Unlikely to get back above 80%", "red")
    for p in np.linspace(0, 1, 101):
        emoji, text, colour = ins.traffic_light(float(p), True)
        assert text and emoji and colour in ("green", "amber", "red")


def test_capped_probability_range():
    assert ins.capped(0.0) == 0.05 and ins.capped(1.0) == 0.95 and ins.capped(0.4) == 0.4


def test_station_table_weighted_and_excludes_missing():
    st = pd.DataFrame({"station": ["X", "X", "X", "Y"], "scheduled_stops": [100, 300, 600, 50],
                       "time_to_3_pct": [90.0, 70.0, np.nan, 100.0],
                       "cancellations_pct": [1.0, 3.0, 2.0, np.nan]})
    t = ins.station_table(st).set_index("station")
    # X punctuality: (90*100 + 70*300) / 400 = 75 (the NaN row is excluded, not counted as 0)
    assert t.loc["X", "punctuality"] == pytest.approx(75.0)
    # X cancellations: (1*100 + 3*300 + 2*600) / 1000 = 2.2
    assert t.loc["X", "cancellations"] == pytest.approx(2.2)
    assert t.loc["X", "stops"] == 1000 and t.loc["Y", "punctuality"] == 100.0 and np.isnan(t.loc["Y", "cancellations"])


def test_station_missing_reason():
    st = pd.DataFrame({"station": ["X"], "qdate": pd.to_datetime(["2025-04-01"])})
    assert "do not list" in ins.station_missing_reason(st, "Nowhere", "2025-04-01")
    assert "only cover" in ins.station_missing_reason(st, "X", "2019-04-01")
    assert "no reliable" in ins.station_missing_reason(st, "X", "2025-04-01")


class _FakeModel:
    """P(below 80%) = (100 - cur_pct) / 100, so effects can be checked by hand."""
    def predict_proba(self, X):
        p = np.clip((100 - X["cur_pct"].to_numpy()) / 100, 0, 1)
        return np.c_[1 - p, p]


def _row():
    return pd.Series({"cur_pct": 60.0, "prev1_pct": 70.0, "prev2_pct": 80.0, "seasonal_pct": 90.0, "roll4_pct": 75.0,
                      "delta1": -10.0, "delay_per_1k_stops": 300.0, "nr_share": 0.6, "weather_share": 0.1})


FEATS = ["cur_pct", "prev1_pct", "prev2_pct", "seasonal_pct", "roll4_pct", "delta1", "delay_per_1k_stops",
         "nr_share", "weather_share"]


def test_scenario_row_recomputes_derived_inputs():
    r = ins.scenario_row(_row(), {"cur": 80.0}).iloc[0]
    assert r.cur_pct == 80 and r.roll4_pct == pytest.approx((80 + 70 + 80 + 90) / 4)   # 80.0
    assert r.delta1 == 10.0                                                            # 80 - 70


def test_one_at_a_time_attribution():
    orig = {"cur": 60.0, "p1": 70.0, "seas": 90.0}
    new = {"cur": 80.0, "p1": 50.0, "seas": 90.0}
    out = {x["key"]: x["p_below"] for x in ins.one_at_a_time(_FakeModel(), {}, FEATS, _row(), orig, new)}
    assert set(out) == {"cur", "p1"}                    # 'seas' unchanged -> not attributed
    assert out["cur"] == pytest.approx(0.20)            # cur 80 -> (100-80)/100
    assert out["p1"] == pytest.approx(0.40)             # model ignores prev1, so risk stays (100-60)/100


def test_predict_risk_reports_filled_in_inputs():
    row = pd.DataFrame([_row()])
    row.loc[0, "nr_share"] = np.nan
    out = ins.predict_risk(_FakeModel(), {"nr_share": 0.5}, row, FEATS)
    assert out["imputed"] == ["nr_share"] and out["p_below"] == pytest.approx(0.4)
    assert out["p_above"] == pytest.approx(0.6)


def test_forecast_unavailable_reasons():
    f = pd.DataFrame({"operator": ["A", "B"], "qdate": pd.to_datetime(["2025-10-01"] * 2),
                      "prev1_pct": [80.0, np.nan], "prev2_pct": [80.0, np.nan], "seasonal_pct": [80.0, np.nan]})
    assert ins.forecast_unavailable_reason(f, "A", "2025-10-01") is None
    assert "previous quarter" in ins.forecast_unavailable_reason(f, "B", "2025-10-01")
    assert "no figure" in ins.forecast_unavailable_reason(f, "C", "2025-10-01")
