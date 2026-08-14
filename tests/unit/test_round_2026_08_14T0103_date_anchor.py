"""Round 2026-08-14T0103 — date-grounded recall (Q59) regression.

The chat round flagged a residual limitation: first-person temporal duration
disclosures that name ONLY a 4-digit year (``"firing since 2017"``,
``"i started in 2019"``, ``"since 2019 i have lived here"``) were stored with
NO dated fact, so date-grounded recall (``"when did i start firing"``) returned
empty. The DateGrounder had no path for a year-only START anchor (it required a
month name before trusting an assembled date).

These tests assert (they do NOT return a bool — a returning test is always
green under pytest). They fail without the new ``resolve_year_start_anchor``
capability and pass with it.
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))

from ravana.core.temporal_grounding import DateGrounder
from ravana.core.hippocampal_buffer import HippocampalBuffer, HippocampalConfig


def _check(name, cond):
    """Assert ``cond`` — never return a bool (pytest reports returns as PASS)."""
    assert cond, name


def test_year_start_anchor_since():
    g = DateGrounder()
    sd = datetime(2026, 8, 14)
    r = g.resolve_year_start_anchor("i have been firing since 2017", sd)
    _check("'since 2017' -> 2017-01-01",
           r is not None and r.date == datetime(2017, 1, 1)
           and r.granularity == "year")


def test_year_start_anchor_started_in():
    g = DateGrounder()
    sd = datetime(2026, 8, 14)
    r = g.resolve_year_start_anchor("i started in 2019", sd)
    _check("'started in 2019' -> 2019-01-01",
           r is not None and r.date == datetime(2019, 1, 1))


def test_year_start_anchor_leading_since():
    g = DateGrounder()
    sd = datetime(2026, 8, 14)
    r = g.resolve_year_start_anchor("since 2019 i have lived here", sd)
    _check("'since 2019 ...' -> 2019-01-01",
           r is not None and r.date == datetime(2019, 1, 1))


def test_year_start_anchor_in_and_back_in():
    g = DateGrounder()
    sd = datetime(2026, 8, 14)
    r1 = g.resolve_year_start_anchor("in 2018 we moved", sd)
    _check("'in 2018 ...' -> 2018-01-01",
           r1 is not None and r1.date == datetime(2018, 1, 1))
    r2 = g.resolve_year_start_anchor("back in 2015 i began", sd)
    _check("'back in 2015 ...' -> 2015-01-01",
           r2 is not None and r2.date == datetime(2015, 1, 1))


def test_year_start_anchor_quantity_guard():
    g = DateGrounder()
    sd = datetime(2026, 8, 14)
    # A year that is really a scalar (followed by a unit) must NOT anchor.
    r1 = g.resolve_year_start_anchor("i scored 2015 points", sd)
    _check("'scored 2015 points' is NOT a date anchor", r1 is None)
    r2 = g.resolve_year_start_anchor("in 2018 dollars", sd)
    _check("'in 2018 dollars' is NOT a date anchor", r2 is None)


def test_year_start_anchor_no_cue_is_none():
    g = DateGrounder()
    sd = datetime(2026, 8, 14)
    # Bare year with no temporal cue word left un-anchored (so it does not
    # shadow the session-date default upstream).
    r = g.resolve_year_start_anchor("just a random 1999 sentence", sd)
    _check("'random 1999 sentence' (no cue) -> None", r is None)


def test_ground_utterance_returns_year():
    g = DateGrounder()
    sd = datetime(2026, 8, 14)
    r = g.ground_utterance("i have been firing since 2017", sd)
    _check("ground_utterance('...since 2017') -> 2017-01-01",
           r is not None and r.date == datetime(2017, 1, 1)
           and r.granularity == "year")


def test_ground_utterance_month_precision_preserved():
    g = DateGrounder()
    sd = datetime(2026, 8, 14)
    # A more-precise month+year must still win over the year-only anchor.
    r = g.ground_utterance("in May 2019 we moved", sd)
    _check("'in May 2019' stays day-precision (2019-05-01)",
           r is not None and r.date == datetime(2019, 5, 1))


def test_dated_fact_stored_and_retrievable():
    # Buffer-level integration: the engine calls ground_utterance() and stores
    # the returned absolute_date; date-grounded recall reads it back.
    g = DateGrounder()
    sd = datetime(2026, 8, 14)
    hb = HippocampalBuffer(HippocampalConfig(max_facts=50))
    subj, obj = "firing", "i have been firing my kiln since 2017"
    grounded = g.ground_utterance(obj, sd)
    hb.store(subject=subj, predicate="is_about", object=obj,
             confidence=0.6, session_date=sd,
             absolute_date=grounded.date if grounded else sd,
             user_fact=True)
    dated = hb.retrieve_dated(subj)
    _check("buffer holds a dated fact for 'firing'",
           dated is not None and len(dated) == 1)
    _check("stored date == 2017-01-01 (Q59 anchor)",
           dated[0].absolute_date == datetime(2017, 1, 1))


if __name__ == "__main__":
    tests = [
        test_year_start_anchor_since, test_year_start_anchor_started_in,
        test_year_start_anchor_leading_since, test_year_start_anchor_in_and_back_in,
        test_year_start_anchor_quantity_guard, test_year_start_anchor_no_cue_is_none,
        test_ground_utterance_returns_year,
        test_ground_utterance_month_precision_preserved,
        test_dated_fact_stored_and_retrievable,
    ]
    print("Round 2026-08-14T0103 — date-grounded recall (Q59)")
    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed.append(t.__name__)
            print(f"  FAIL: {t.__name__}: {exc}")
        else:
            print(f"  PASS: {t.__name__}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if not failed:
        print("ALL PASS")
    else:
        sys.exit(1)
