"""PROMPT 3 verification: comparative web-answer gate must ACCEPT a correctly-
sourced snippet (it used to discard anything scoring below the absolute 1.5
floor). We mock the search engine so this test is deterministic and independent
of the flaky localhost:4000 gateway.
"""
import pytest


@pytest.fixture(scope="module")
def engine():
    from ravana.chat.engine import CognitiveChatEngine
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
    return e


def _fake_result(snippet, url):
    return [{"url": url, "content": snippet, "title": "x"}]


def test_web_snippet_search_accepts_borderline_good_snippet(engine, monkeypatch):
    """A well-sourced encyclopedic snippet that scores just below the old 1.5
    absolute floor must still be selected (comparative, not absolute)."""
    from ravana.chat.models import CognitiveResponseContext

    class _FakeSearch:
        def search(self, term, max_results=6, local_only=False):
            return _fake_result(
                "the speed of light in vacuum is about 299,792,458 metres per second",
                "https://en.wikipedia.org/wiki/Speed_of_light")

    engine.search_engine = _FakeSearch()
    ctx = CognitiveResponseContext(
        subject="speed of light",
        raw_input="what is the speed of light",
        valence=0.5)
    best, term, attempted = engine._web_snippet_search(
        ["speed of light real", "speed of light science"], ctx, False, 1e18)
    assert attempted is True
    assert best is not None, "comparative gate discarded a valid sourced snippet"
    assert "299,792,458" in best
    # Source tagging must be stashed for the surfacer.
    assert engine._last_web_source == "Wikipedia"


def test_web_direct_answer_surfaces_source(engine, monkeypatch):
    """_web_direct_answer must return the snippet AND tag the source."""
    from ravana.chat.models import CognitiveResponseContext

    class _FakeSearch:
        def search(self, term, max_results=6, local_only=False):
            return _fake_result(
                "the speed of light in vacuum is about 299,792,458 metres per second",
                "https://en.wikipedia.org/wiki/Speed_of_light")

    engine.search_engine = _FakeSearch()
    ctx = CognitiveResponseContext(
        subject="speed of light",
        raw_input="what is the speed of light",
        valence=0.5)
    ctx.strategy = ""
    out = engine._web_direct_answer(ctx)
    assert out is not None, "web_direct_answer withheld a valid sourced snippet"
    text, strat = out
    assert "299,792,458" in text
    assert "according to Wikipedia" in text, f"source not tagged: {text!r}"
    assert strat == "web_direct_answer"
