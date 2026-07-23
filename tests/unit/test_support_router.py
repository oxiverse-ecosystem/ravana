"""Unit tests for the support / advice router.

The consult benchmark cases are the first tests — they must now be
recognized as support intent (and, when the live web is reachable,
route to a web-backed answer). Detection is what we assert
deterministically; the web fetch is exercised separately and
degrades gracefully to None when the network is down.
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana-v2", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.chat.support_router import SupportRouter, route_support


def _router():
    # Build without glove (heuristics carry detection).
    return SupportRouter(glove_fn=None)


def test_stress_detected():
    r = _router()
    assert r.is_support("I feel stressed. Healthy ways to manage stress?")


def test_programming_advice_detected():
    r = _router()
    assert r.is_support("I want to learn programming. Python or JavaScript for a beginner?")


def test_overwhelmed_detected():
    r = _router()
    assert r.is_support("I'm overwhelmed, any advice?")


def test_benign_factual_not_flagged():
    r = _router()
    assert not r.is_support("What is the capital of France?")


def test_benign_greeting_not_flagged():
    r = _router()
    assert not r.is_support("hi, how are you?")


def test_build_query_shapes_topic():
    r = _router()
    q = r.build_query("I feel stressed. Healthy ways to manage stress?")
    assert "stress" in q.lower()


def test_route_support_no_engine():
    # No engine -> None (fail-open).
    assert route_support(None, "I feel stressed, healthy ways?") is None


def test_route_support_clean_query():
    # A clean factual query is not a support intent -> None.
    r = _router()
    # build a fake engine exposing the router
    class _E:
        _support_router = r
    assert route_support(_E(), "What is gravity?") is None


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]:
        fn()
        print("PASS", fn.__name__)
    print("ALL SUPPORT-ROUTER TESTS PASSED")
