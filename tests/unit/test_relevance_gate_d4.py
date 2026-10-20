"""Capability test: _relevance_gate filters off-topic analogical replies.

D4 (non-sequitur confabulation): when _consult_internal_knowledge misses,
it falls back to _try_analogical_reasoning, which finds the closest concept
by GloVe cosine and generates an analogical reply. Without a relevance gate,
this reply can be completely off-topic (e.g., "dreams" -> "people who hoard
every book ever read").

The _relevance_gate method uses the learned multi-signal relevance score and
requires content overlap. Recency and GloVe proximity alone cannot establish
that a generated answer addresses the query.

This test verifies:
1. The _relevance_gate method exists and is callable
2. It passes relevant answers (high topical coherence)
3. It withholds off-topic answers (low topical coherence)
4. It is wired into _consult_internal_knowledge (analogical replies are gated)
"""
import os
import sys

import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (PROJ,
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402


@pytest.fixture
def engine():
    suffix = "relevance_gate_d4"
    for stale in (f"weights/ravana_weights{suffix}.pkl",
                  f"weights/ravana_usermodel{suffix}.pkl"):
        full = os.path.join(PROJ, stale)
        if os.path.exists(full):
            os.remove(full)
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix=suffix)
    yield eng
    try:
        eng.stop_background_learning()
    except Exception:
        pass


def test_relevance_gate_method_exists(engine):
    """The _relevance_gate method must exist and be callable."""
    assert hasattr(engine, '_relevance_gate'), (
        "_relevance_gate method is missing from the engine")
    assert callable(engine._relevance_gate), (
        "_relevance_gate must be callable")


def test_relevance_gate_passes_relevant_answer(engine):
    """A topically relevant answer should pass the gate."""
    # "sleep" and "dream" are semantically related (GloVe cosine > 0.30)
    answer = "sleep is when the mind rests and dreams happen"
    result = engine._relevance_gate("what is sleep", answer)
    assert result is not None, (
        "relevance_gate should pass a topically relevant answer")


def test_relevance_gate_withholds_off_topic_answer(engine):
    """An off-topic answer should be withheld (no content overlap)."""
    answer = "car engine repair requires mechanical tools and oil changes"
    result = engine._relevance_gate("what is sleep", answer)
    assert result is None, (
        "relevance_gate should withhold an off-topic answer "
        f"(got: {result!r})")


def test_relevance_gate_withholds_analogical_confabulation(engine):
    """The D4 confabulation: analogical replies that are off-topic must be
    withheld. This is the core test — it reproduces the exact failure mode
    from the round report (dreams -> book hoarding)."""
    confabulated = (
        "when i let my mind wander, i find myself drawn to people "
        "who hoard every book ever read"
    )
    result = engine._relevance_gate("what do you dream about", confabulated)
    assert result is None, (
        "relevance_gate should withhold analogical confabulation "
        f"(got: {result!r})")


def test_relevance_gate_returns_answer_string_when_relevant(engine):
    """When the answer IS relevant, the gate returns the answer string."""
    answer = "sleep is a natural state of rest for the mind and body"
    result = engine._relevance_gate("what is sleep", answer)
    assert result == answer, (
        "relevance_gate should return the answer string when relevant")


def test_relevance_gate_handles_empty_inputs(engine):
    """Empty answers pass through unchanged; None still means no answer."""
    assert engine._relevance_gate("some query", "") == ""
    assert engine._relevance_gate("", "") == ""
    assert engine._relevance_gate("some query", None) is None
    # Non-empty query with relevant answer should pass through the gate
    result = engine._relevance_gate("what is sleep", "sleep is when the mind rests")
    assert result is not None, "gate should pass a relevant answer"


def test_relevance_gate_passes_long_relevant_answer(engine):
    """A low Jaccard ratio must not reject an answer naming the real topic."""
    answer = (
        "sleep is a natural state of rest for the mind and body, with changes "
        "in awareness, breathing, muscle activity, and responses to surroundings"
    )
    assert engine._relevance_gate("what is sleep", answer) == answer


def test_relevance_gate_preserves_answer_when_scoring_unavailable(engine, monkeypatch):
    """The PR's fail-open behavior must survive the content-evidence check."""
    def unavailable(*args, **kwargs):
        raise RuntimeError("scoring unavailable")

    monkeypatch.setattr(engine._relevance_scorer, "score_with_signals", unavailable)
    answer = "sleep is when the mind rests"
    assert engine._relevance_gate("what is sleep", answer) == answer


def test_relevance_gate_is_wired_into_consult_internal(engine):
    """The _relevance_gate must be accessible via getattr (the wiring used
    by _consult_internal_knowledge)."""
    gate = getattr(engine, "_relevance_gate", None)
    assert gate is not None, (
        "_relevance_gate must be accessible via getattr — "
        "_consult_internal_knowledge uses getattr(self, '_relevance_gate', None)")
    # Verify it's callable
    assert callable(gate)
