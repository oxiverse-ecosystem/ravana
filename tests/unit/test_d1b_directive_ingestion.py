"""Regression tests for D1b (round 2026-08-19T1628Z): the hippocampal buffer
must NOT store imperative request/recall-directive utterances as if they were
experienced content.

Root cause: "tell me again what you know about mica the crow." (an imperative
directive to the agent) was ingested keyed under mica/crow/tell/know, so a
LATER semantically-overlapping query surfaced the user's OWN request verbatim as
RAVANA's reply (source-monitoring violation). The fix extends the existing
interrogative ingestion guard with a structural request-directive guard.

These tests FAIL on the pre-fix code and PASS after the fix. They exercise the
REAL engine (_ingest_episodic path) — no mocks.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ, f"{_PROJ}\\ravana_ml\\src", f"{_PROJ}\\ravana\\src", f"{_PROJ}\\ravana-v2\\src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest


@pytest.fixture(scope="module")
def engine():
    from ravana.chat.engine import CognitiveChatEngine
    # Unique suffix so no prior persisted weights are auto-resumed.
    eng = CognitiveChatEngine(
        dim=64, seed=42, baby_mode=True,
        user_suffix="d1b_test_" + str(os.getpid()))
    yield eng
    try:
        eng.stop_background_learning()
    except Exception:
        pass


def _buffer_objects(eng):
    fb = getattr(eng, "hippocampal_buffer", None)
    facts = getattr(fb, "facts", {}) if fb else {}
    out = []
    for _kfacts in facts.values():
        for f in _kfacts:
            out.append((getattr(f, "object", "") or ""))
    return out


def test_directive_not_ingested(engine):
    """An imperative recall-directive must not be stored as a retrievable fact."""
    directive = "tell me again what you know about mica the crow."
    # Run it through the real process_turn so ingestion logic executes.
    engine.process_turn(directive)
    objs = _buffer_objects(engine)
    assert not any("tell me again what you know about mica" in o.lower() for o in objs), (
        "imperative directive was ingested into the hippocampal buffer: "
        + str([o for o in objs if 'mica' in o.lower()]))


def test_directive_echo_does_not_surfaced(engine):
    """A later query sharing tokens must not echo the prior directive back."""
    # Seed a real related fact first.
    engine.process_turn("i have a pet crow named mica. she steals shiny things.")
    # User asks an imperative directive (no '?') — should NOT be stored.
    engine.process_turn("tell me again what you know about mica the crow.")
    # A later unrelated comparative/philosophical query that previously echoed
    # the directive verbatim must now NOT contain the directive text.
    r = engine.process_turn(
        "what's the deal with narrowboats versus houseboats, philosophically?")
    assert "tell me again what you know about mica" not in (r or "").lower(), (
        "reply echoed the user's prior directive back as RAVANA's answer: " + repr(r))


@pytest.mark.parametrize("directive", [
    "tell me about my brother tomas.",
    "show me what you remember about my cat.",
    "remind me what i told you about the houseboat.",
    "describe my sister priya for me.",
    "could you explain what lutefisk is.",
    "what do you know about my garden?",
    "do you remember what i said about ren?",
])
def test_various_directives_not_ingested(engine, directive):
    engine.process_turn(directive)
    objs = [o.lower() for o in _buffer_objects(engine)]
    assert not any(directive.lower() in o for o in objs), (
        f"directive was ingested: {directive!r}")


def test_real_disclosure_still_ingested(engine):
    """Genuine first-person disclosures must still be stored (no over-blocking)."""
    engine.process_turn("i have a pet crow named mica. she steals anything shiny.")
    objs = [o.lower() for o in _buffer_objects(engine)]
    assert any("pet crow named mica" in o for o in objs), (
        "legitimate self-disclosure was wrongly blocked from ingestion")
