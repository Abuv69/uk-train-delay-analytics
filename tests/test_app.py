"""End-to-end checks of the running app: what the user reads must match the data."""
import re
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from scripts import insights as ins

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def truth():
    with sqlite3.connect(ROOT / "database" / "delays.db") as c:
        pq = pd.read_sql("SELECT * FROM operator_punctuality WHERE pct_on_time_3m IS NOT NULL", c, parse_dates=["qdate"])
        stn = pd.read_sql("SELECT * FROM station_performance", c)
    return pq, ins.add_station_quarter(stn)


def fresh():
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    assert not at.exception
    return at


def texts(at):
    return [m.value for m in at.markdown]


def test_home_has_one_clear_answer_with_text_label():
    at = fresh()
    card = [t for t in texts(at) if 'class="answer' in t]
    assert len(card) == 1
    assert "arrived on time" in card[0] and re.search(r"(Likely|Uncertain|Unlikely)", card[0])   # words, not just colour
    assert "(latest)" in at.selectbox(key="sel_quarter").value
    assert at.sidebar  # sidebar exists but starts collapsed (set in page config)


def test_what_changed_matches_numbers_on_screen(truth):
    pq, _ = truth
    at = fresh()
    a = at.selectbox(key="sel_operator").value
    q = pq.qdate.max()
    b = "Avanti West Coast" if a != "Avanti West Coast" else "Merseyrail"
    at.selectbox(key="sel_operator").set_value(b).run()
    assert not at.exception
    change = [t for t in texts(at) if "What changed" in t]
    assert len(change) == 1
    pa = round(float(pq[(pq.operator == a) & (pq.qdate == q)].pct_on_time_3m.iloc[0]), 1)
    pb = round(float(pq[(pq.operator == b) & (pq.qdate == q)].pct_on_time_3m.iloc[0]), 1)
    nums = [float(x) for x in re.findall(r"[-+]?\d+\.\d", change[0])]
    assert nums == [pa, pb, round(pb - pa, 1)]                      # old, new, difference - computed independently
    card = [t for t in texts(at) if 'class="answer' in t][0]
    assert f"{pb:.1f}%" in card                                     # the card shows the same number the line quotes
    assert b in change[0] and a in change[0]


def test_quarter_change_is_explained(truth):
    pq, _ = truth
    at = fresh()
    op = at.selectbox(key="sel_operator").value
    old_label = at.selectbox(key="sel_quarter").value
    new_label = at.selectbox(key="sel_quarter").options[2]
    at.selectbox(key="sel_quarter").set_value(new_label).run()
    change = [t for t in texts(at) if "What changed" in t][0]
    assert old_label.replace(" (latest)", "") in change and new_label in change
    assert "percentage points" in change


def test_forecast_quarter_shows_range_not_certainty():
    at = fresh()
    at.selectbox(key="sel_quarter").set_value(at.selectbox(key="sel_quarter").options[1]).run()
    card = [t for t in texts(at) if 'class="answer' in t][0]
    assert "forecast" in card.lower() and "Confidence" in card and "never certain" in card
    assert not re.search(r"\b(0|100)% chance", card)


def test_station_pick_matches_data(truth):
    _, stn = truth
    at = fresh()
    q = stn.qdate.max()
    at.selectbox(key="sel_quarter").set_value([o for o in at.selectbox(key="sel_quarter").options if "forecast" in o][0]).run()
    at.selectbox(key="sel_station").set_value("Leeds").run()
    assert not at.exception
    want = round(ins.station_quarter_summary(stn, "Leeds", q)["pct"], 1)
    assert any(f"You picked Leeds, where {want:.1f}% of trains arrived on time" in t for t in texts(at))


def test_url_round_trip_and_sharing():
    at = AppTest.from_file(APP, default_timeout=240)
    at.query_params["operator"] = "Merseyrail"
    at.query_params["quarter"] = "forecast"
    at.run()
    assert not at.exception
    assert at.selectbox(key="sel_operator").value == "Merseyrail"
    assert "(forecast)" in at.selectbox(key="sel_quarter").value
    assert at.query_params["operator"] == "Merseyrail" and at.query_params["page"] == "home"


def test_bad_link_falls_back_with_friendly_message():
    at = AppTest.from_file(APP, default_timeout=240)
    at.query_params["operator"] = "Not A Real Railway"
    at.query_params["quarter"] = "1999-01-01"
    at.run()
    assert not at.exception
    msgs = " ".join(texts(at))
    assert "couldn't find the operator" in msgs and "couldn't find the quarter" in msgs
    assert "Traceback" not in msgs


def test_every_page_renders_without_errors():
    at = fresh()
    for p in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(p).run()
        assert not at.exception, p


def test_no_operator_forecast_is_shown_as_certain():
    at = fresh()
    at.sidebar.radio[0].set_value("🔮 Forecast for next quarter").run()
    ops = at.selectbox(key="fc_op").options
    assert len(ops) >= 20
    for op in ops:
        at.selectbox(key="fc_op").set_value(op).run()
        assert not at.exception, op
        card = [t for t in texts(at) if 'class="answer' in t][0]
        assert not re.search(r"(?<![\d.])(0|100)%", card.replace("80%", "")), op
        assert re.search(r"(over 95%|under 5%|about \d+%)", card), op


def test_what_if_panel_is_explained_and_resettable():
    at = fresh()
    at.sidebar.radio[0].set_value("🔮 Forecast for next quarter").run()
    op = at.selectbox(key="fc_op").value
    k = op.replace(" ", "_")
    tbl = lambda: [d for d in at.dataframe if "Your value" in d.value.columns][0].value
    assert (tbl()["Changed?"] == "").all()                              # nothing changed at first
    assert any("haven't changed anything" in t for t in texts(at))
    cur = at.slider(key=f"wi_cur_{k}")
    orig = cur.value
    cur.set_value(min(100.0, orig + 25)).run()
    assert not at.exception
    t = tbl()
    assert (t["Changed?"] != "").sum() == 1 and t.loc[0, "Changed?"] == "✔ changed"
    assert any("Mostly because" in x or "None of your changes" in x for x in texts(at))
    at.button(key=f"reset_{k}").click().run()
    assert at.slider(key=f"wi_cur_{k}").value == orig
    assert (tbl()["Changed?"] == "").all()


def test_causes_page_opens_on_my_operator_and_explains_changes(truth):
    at = fresh()
    mine = at.selectbox(key="sel_operator").value                    # operator chosen on Home
    at.sidebar.radio[0].set_value("📉 Why trains are late").run()
    assert at.selectbox(key="late_op").value == mine                 # carried over deliberately
    assert at.query_params["page"] == "late" and at.query_params.get("year") == "All years"
    other = "Northern Trains" if mine != "Northern Trains" else "Merseyrail"
    at.selectbox(key="late_op").set_value(other).run()
    msg = [t for t in texts(at) if "What changed" in t][0]
    nums = [float(x) for x in re.findall(r"[-+]?\d+\.\d", msg)]
    with sqlite3.connect(ROOT / "database" / "delays.db") as c:
        q = "SELECT SUM(total_delay_mins) s FROM delay_causes WHERE operator=?"
        a = pd.read_sql(q, c, params=[mine])["s"].iloc[0] / 1e6
        b = pd.read_sql(q, c, params=[other])["s"].iloc[0] / 1e6
    assert nums == [round(a, 1), round(b, 1), round(round(b, 1) - round(a, 1), 1)]


def test_leaving_a_page_clears_its_url_choices():
    at = fresh()
    at.sidebar.radio[0].set_value("🔮 Forecast for next quarter").run()
    assert "quarter" not in at.query_params and "station" not in at.query_params   # home's choices did not leak
    assert at.query_params["page"] == "forecast"


JARGON = ["punctuality", "PPM", "logistic", "regression", "calibrat", "ROC", "F1", "impute", "coefficient", "quantile"]


def test_home_page_has_no_jargon():
    at = fresh()
    visible = " ".join(texts(at) + [c.value for c in at.caption] + [s.label for s in at.selectbox]
                       + [b.label for b in at.button] + [t.value for t in at.title])
    for word in JARGON:
        assert word.lower() not in visible.lower(), f"jargon on Home: {word}"


def test_every_chart_title_on_home_states_a_takeaway():
    at = fresh()
    titles = []
    for c in at.get("plotly_chart"):
        titles.append(c.proto.spec if hasattr(c, "proto") else "")
    import json
    for spec in titles:
        t = json.loads(spec).get("layout", {}).get("title", {}).get("text", "")
        assert len(t.split()) >= 5 and t.lower() != "chart", t          # a sentence, not a label like "Delay minutes"
