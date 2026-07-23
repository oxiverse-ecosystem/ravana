"""Phase 1 unit tests: temporal grounding + date-aware hippocampal retrieval."""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))

from ravana.core.temporal_grounding import DateGrounder
from ravana.core.hippocampal_buffer import HippocampalBuffer, HippocampalConfig


def _check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}: {name}")
    return cond


def test_session_date_parsing():
    g = DateGrounder()
    d = g.parse_session_date("2:15 pm on 8 May, 2023")
    ok = d is not None and d.year == 2023 and d.month == 5 and d.day == 8
    return _check("parse '2:15 pm on 8 May, 2023' -> 2023-05-08", ok)


def test_relative_years_ago():
    g = DateGrounder()
    sd = datetime(2023, 5, 8)
    r = g.resolve_relative("i moved here 4 years ago", sd)
    ok = r is not None and r.date.year == 2019
    return _check("'4 years ago' from 2023 -> 2019", ok)


def test_relative_last_month():
    g = DateGrounder()
    sd = datetime(2023, 6, 15)
    r = g.resolve_relative("i saw her last month", sd)
    ok = r is not None and r.date.month == 5 and r.date.year == 2023
    return _check("'last month' from 2023-06 -> 2023-05", ok)


def test_relative_yesterday():
    g = DateGrounder()
    sd = datetime(2023, 5, 8)
    r = g.resolve_relative("i went yesterday", sd)
    ok = r is not None and r.date.day == 7 and r.date.month == 5
    return _check("'yesterday' from 2023-05-08 -> 2023-05-07", ok)


def test_relative_last_weekday():
    g = DateGrounder()
    sd = datetime(2023, 5, 10)  # Wednesday
    r = g.resolve_relative("last tuesday", sd)
    # last Tuesday before Wed 2023-05-10 is 2023-05-09
    ok = r is not None and r.date.weekday() == 1 and r.date.day == 9
    return _check("'last tuesday' from Wed 2023-05-10 -> 2023-05-09", ok)


def test_month_day_assumes_session_year():
    g = DateGrounder()
    sd = datetime(2023, 7, 1)
    r = g.resolve_relative("on june 15th", sd)
    ok = r is not None and r.date.month == 6 and r.date.day == 15 and r.date.year == 2023
    return _check("'june 15th' assumes session year 2023", ok)


def test_interval_describe():
    g = DateGrounder()
    ok = g.describe_interval(datetime(2019, 1, 1), datetime(2023, 1, 1)) == "4 years"
    return _check("interval 2019->2023 == '4 years'", ok)


def test_buffer_dated_retrieval():
    hb = HippocampalBuffer(HippocampalConfig(max_facts=50))
    hb.store("paris", "is_about", "went to paris",
             session_date=datetime(2023, 5, 8),
             absolute_date=datetime(2023, 5, 8))
    hb.store("paris", "is_about", "planning another paris trip",
             session_date=datetime(2023, 8, 1),
             absolute_date=datetime(2023, 8, 1))
    dated = hb.retrieve_dated("paris")
    ok = dated is not None and len(dated) == 2 and dated[0].absolute_date < dated[1].absolute_date
    return _check("retrieve_dated returns chronologically sorted", ok)


def test_buffer_latest():
    hb = HippocampalBuffer(HippocampalConfig(max_facts=50))
    hb.store("job", "works_at", "google",
             absolute_date=datetime(2020, 1, 1))
    hb.store("job", "works_at", "meta",
             absolute_date=datetime(2023, 1, 1))
    latest = hb.retrieve_latest("job")
    ok = latest is not None and latest.object == "meta"
    return _check("retrieve_latest returns most recent (meta)", ok)


def test_state_roundtrip_with_dates():
    hb = HippocampalBuffer(HippocampalConfig(max_facts=50))
    hb.store("paris", "is_about", "went to paris",
             session_date=datetime(2023, 5, 8),
             absolute_date=datetime(2023, 5, 8))
    st = hb.get_state()
    hb2 = HippocampalBuffer(HippocampalConfig(max_facts=50))
    hb2.set_state(st)
    f = hb2.retrieve("paris")
    ok = f is not None and f[0].absolute_date == datetime(2023, 5, 8)
    return _check("get_state/set_state preserves dates", ok)


if __name__ == "__main__":
    tests = [
        test_session_date_parsing, test_relative_years_ago,
        test_relative_last_month, test_relative_yesterday,
        test_relative_last_weekday, test_month_day_assumes_session_year,
        test_interval_describe, test_buffer_dated_retrieval,
        test_buffer_latest, test_state_roundtrip_with_dates,
    ]
    print("Phase 1 — temporal grounding tests")
    results = [t() for t in tests]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    if passed == len(results):
        print("ALL PASS")
    else:
        sys.exit(1)
