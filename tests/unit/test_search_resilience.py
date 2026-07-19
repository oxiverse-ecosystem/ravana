"""
Phase 20a — SearchEngine transient-retry resilience.

Verifies the measured root cause: the local engine (SearXNG on
localhost:4000) intermittently aborts/refuses a fraction of real
requests but answers the very next attempt. The OLD code counted every
abort as a circuit-breaker failure, so ~5 flaky aborts blacked out the
only backend -> bare SearchError -> "couldn't verify from web".

This test proves the retry loop absorbs transient errors WITHOUT recording
a breaker failure, and that a *sustained* failure still trips the breaker.

No network required — _call_api is monkeypatched.
"""
import sys, os
import pytest

import socket
import urllib.error

_proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_proj, "ravana", "src"))

from ravana.web.learner import SearchEngine, SearchConfig, SearchError


# A *transient* network error (matches the real aborts: ConnectionAborted /
# RemoteDisconnected / URLError). The engine's retry loop only swallows
# errors in its _TRANSIENT tuple, so the test must raise one of those.
class _ConnAbort(urllib.error.URLError):
    def __init__(self, msg):
        super().__init__(msg)


def _make_engine(transient_first=0, then_raise=None):
    """Engine whose _call_api fails `transient_first` times (transient),
    then either succeeds (returns results) or raises `then_raise`."""
    eng = SearchEngine(SearchConfig(transient_retries=3, cooldown=1, max_failures=5))
    results = [{"title": "ok", "url": "http://x", "content": "good"}]
    state = {"n": 0}

    def fake_call(api_name, url, timeout, max_results):
        state["n"] += 1
        if state["n"] <= transient_first:
            raise _ConnAbort(f"transient abort #{state['n']}")
        if then_raise is not None:
            raise then_raise
        return results

    eng._call_api = fake_call
    return eng


def test_transient_aborts_recover_without_breaker_trip():
    eng = _make_engine(transient_first=2, then_raise=None)
    out = eng.search("anything", max_results=5)
    assert out, "search should recover after transient aborts"
    assert eng._api_failure_counts["local_api"] == 0, \
        "transient errors must NOT count toward the breaker"


def test_transient_flakiness_through_full_search():
    # Real-world shape: ~40% of requests abort, but the very next
    # attempt succeeds. Drive many sequential searches through the full
    # search() path and assert NONE of them trips the breaker (the
    # old behavior blacked out the only backend after ~5 blips).
    eng = _make_engine(transient_first=1, then_raise=None)
    for i in range(20):
        out = eng.search(f"query number {i}", max_results=3)
        assert out, f"search #{i} must absorb the transient abort"
    assert eng._api_failure_counts["local_api"] == 0, \
        "20 flaky searches must NOT trip the breaker"


def test_sustained_failure_trips_breaker():
    eng = _make_engine(transient_first=0, then_raise=SearchError("dead"))
    with pytest.raises(SearchError):
        eng.search("anything", max_results=5)
    assert eng._api_failure_counts["local_api"] >= 1


def test_single_transient_then_success():
    eng = _make_engine(transient_first=1, then_raise=None)
    out = eng.search("x")
    assert out
    assert eng._api_failure_counts["local_api"] == 0
