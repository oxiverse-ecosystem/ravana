"""Regression tests for D3 (round 2026-08-19T1628Z): the in-prompt causal
reasoner must NOT echo the user's own affective self-report back as its
"answer" to an affective query.

Root cause: a combined "statement + question" turn like
    "that parking lot plan makes my blood boil. do you get why i'm furious?"
was intercepted by the combined-fact handler, which fell through to
`answer_in_prompt_causal`. That function mined the user's affective
statement ("X makes my blood boil") as a causal premise, and because the
query contained a "?" its loose question-gate admitted it; the query seeds
then overlapped the user's own words, so the reasoner replayed the effect
clause "my blood boil" verbatim as RAVANA's reply — a source-monitoring
failure (the agent parroted the user's utterance as its own reasoning).

The fix adds a source-monitoring guard in `parse_causal_edges`: a premise
whose EFFECT is a first-person affective self-report is a felt state, not a
world-state transition, so the edge is refused and the turn falls through to
the genuine affective-response path. Detection is SEED-DRIVEN (first-person
pronoun + an affect-bearing word read from RAVANA's learnable VAD lexicon),
not a hardcoded reply or frozen keyword table.

These tests FAIL on the pre-fix code and PASS after the fix. They exercise the
REAL reasoner and the REAL engine (no mocks).
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ, f"{_PROJ}\\ravana_ml\\src", f"{_PROJ}\\ravana\\src", f"{_PROJ}\\ravana-v2\\src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest

from ravana.core.in_prompt_reasoner import (
    parse_causal_edges,
    answer_in_prompt_causal,
)


@pytest.fixture(scope="module")
def engine():
    from ravana.chat.engine import CognitiveChatEngine
    # Unique suffix so no prior persisted weights are auto-resumed.
    eng = CognitiveChatEngine(
        dim=64, seed=42, baby_mode=True,
        user_suffix="d3_test_" + str(os.getpid()))
    yield eng
    try:
        eng.stop_background_learning()
    except Exception:
        pass


# ── Reasoner-level (fast, direct) ──────────────────────────────────────────

def test_affective_statement_not_mined_as_causal_edge():
    """An affective self-report must not be bound as a simulate-able edge."""
    text = "that parking lot plan makes my blood boil. do you get why i'm furious?"
    edges = parse_causal_edges(text)
    assert edges == [], (
        "affective self-report was mined as a causal edge: " + str(edges))


def test_affective_alias_variants_not_mined():
    """Other first-person affect idioms must also be refused as effects."""
    for text in (
        "the meeting makes me furious. why does it bother you?",
        "this injustice makes my blood boil. do you see the problem?",
        "waiting makes me seethe. can you believe it?",
        "the news makes my heart race. are you not worried too?",
    ):
        assert parse_causal_edges(text) == [], (
            "affective effect not refused for: " + text)


def test_legit_causal_edge_still_mined():
    """A genuine world-state conditional must still bind (no over-blocking)."""
    text = "when you turn on the lamp, it lights up. what happens if you turn on the lamp?"
    edges = parse_causal_edges(text)
    assert edges == [("you turn on the lamp", "it lights up")], (
        "legit causal edge wrongly refused: " + str(edges))


def test_answer_in_prompt_causal_does_not_echo_affect():
    """The reasoner returns None for an affective turn, not the user's clause."""
    text = "that parking lot plan makes my blood boil. do you get why i'm furious?"
    ans = answer_in_prompt_causal(text)
    assert ans is None, (
        "reasoner echoed the user's affective clause as the answer: " + repr(ans))
    assert "blood boil" not in (ans or "").lower()


# ── Engine-level (end-to-end, no parroting) ────────────────────────────────

def test_engine_does_not_echo_affective_clause(engine):
    """A combined affective statement+question must not parrot the clause."""
    r = engine.process_turn(
        "that parking lot plan makes my blood boil. do you get why i'm furious?")
    assert "blood boil" not in (r or "").lower(), (
        "engine echoed the user's affective clause verbatim: " + repr(r))


def test_engine_still_reasons_legit_causal(engine):
    """A genuine causal premise+query must not be answered by parroting; the
    reply is either a real chained answer OR an honest 'i don't know', never
    an echo of the user's own words."""
    r = engine.process_turn(
        "when you turn on the lamp, an explosion occurs. what happens if you turn on the lamp?")
    assert r is not None, "engine returned no reply for a legit causal query"
    low = (r or "").lower()
    # The user's own words must never be replayed as the answer.
    assert "explosion occurs" not in low and "turn on the lamp" not in low, (
        "engine echoed the user's causal premise verbatim: " + repr(r))
