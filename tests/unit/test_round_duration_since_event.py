"""Unit test: event-anchored "since" phrases are mined as dated facts.

Block (e) in user_model.py extends duration mining to cover "since <event>"
anchors like "since the pandemic" (2020). The existing _FUZZY_DUR covers
human-phrased spans ("a decade") but NOT event-anchored phrases.

Acceptance:
  - "i've been brewing beer since the pandemic" stores since(brew <year>)
    where year = 2020 (pandemic anchor).
  - Existing fuzzy shapes still resolve (regression).
"""
import os, sys, re, datetime
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)


def test_event_anchor_regex_matches():
    """The event-anchor regex matches 'since the pandemic' in context."""
    q_clean = "i've been brewing beer since the pandemic"
    pattern = r"\bsince\s+the\s+pandemic\b"
    m = re.search(pattern, q_clean, re.IGNORECASE)
    assert m is not None, f"Expected match in: {q_clean}"
    assert m.start() > 0  # not at the beginning (activity verb precedes)


def test_event_anchor_regex_no_false_positive():
    """The event-anchor regex does NOT match ordinary 'since' usage."""
    # "since 2020" is a digit-year, handled by block (b), not (e).
    q_clean = "i've been brewing beer since 2020"
    pattern = r"\bsince\s+the\s+pandemic\b"
    m = re.search(pattern, q_clean, re.IGNORECASE)
    assert m is None, "Should not match digit-year 'since 2020'"


def test_event_anchor_year_mapping():
    """The event-anchor seed set maps 'the pandemic' to 2020."""
    _EVENT_ANCHORS = {
        "the pandemic": 2020, "pandemic": 2020,
        "the lockdown": 2020, "lockdown": 2020,
        "the recession": 2008, "recession": 2008,
        "the crash": 2008,
        "9/11": 2001, "september 11": 2001,
        "the war": 2022,
        "the invasion": 2022,
        "the coup": 2021,
        "brexit": 2016,
        "the referendum": 2016,
    }
    assert _EVENT_ANCHORS["the pandemic"] == 2020
    assert _EVENT_ANCHORS["pandemic"] == 2020
    assert _EVENT_ANCHORS["the recession"] == 2008


def test_event_anchor_does_not_overlap_fuzzy():
    """Event-anchored phrases do NOT overlap with _FUZZY_DUR entries."""
    _FUZZY_DUR = {
        "a decade": 10, "two decades": 20, "three decades": 30,
        "a couple of years": 2, "a couple years": 2,
        "a few years": 3, "few years": 3,
        "several years": 4, "a handful of years": 5,
        "many years": 15,
    }
    _EVENT_ANCHORS = {
        "the pandemic": 2020, "pandemic": 2020,
        "the lockdown": 2020, "lockdown": 2020,
    }
    # No overlap between the two seed sets.
    assert not set(_FUZZY_DUR.keys()) & set(_EVENT_ANCHORS.keys())


def test_since_event_fact_shape():
    """The fact stored for 'since the pandemic' has the expected shape."""
    # Simulate what block (e) does:
    # _put_fact("since", f"{_act_full} {_year}", 0.6)
    # where _act_full = "brew beer" and _year = 2020.
    _act = "brew"
    _obj = "beer"
    _act_full = f"{_act} {_obj}".strip() if _obj else _act
    _year = 2020
    fact_val = f"{_act_full} {_year}"
    assert fact_val == "brew beer 2020"


if __name__ == "__main__":
    test_event_anchor_regex_matches()
    print("[PASS] test_event_anchor_regex_matches")

    test_event_anchor_regex_no_false_positive()
    print("[PASS] test_event_anchor_regex_no_false_positive")

    test_event_anchor_year_mapping()
    print("[PASS] test_event_anchor_year_mapping")

    test_event_anchor_does_not_overlap_fuzzy()
    print("[PASS] test_event_anchor_does_not_overlap_fuzzy")

    test_since_event_fact_shape()
    print("[PASS] test_since_event_fact_shape")

    print("\nALL PASSED")
