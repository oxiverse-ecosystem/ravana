"""Capability test: UNIVERSAL RELEVANCE GATING for _consult_internal_knowledge.

The MRS relevance gate was originally wired only into _structured_recall.
Then it was extended to _self_cued_episodic and _route_agent_own_recall.
But _consult_internal_knowledge was the LAST ungated recall path — it could
confabulate by making up associative relations from the concept graph when
the query's subject was unknown.

This test verifies that _consult_internal_knowledge answers are relevance-gated
at the process_turn call site: an answer that passes the internal knowledge
lookup but fails the relevance score must return None (honest uncertainty
beats confabulation).
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
    suffix = "univrelgate_internal"
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


def test_consult_internal_knowledge_is_relevance_gated(engine):
    """The capability: _consult_internal_knowledge answers pass through the
    relevance gate at the process_turn call site. An answer that passes
    the internal knowledge lookup but fails the relevance score must return
    None (honest uncertainty beats confabulation).

    This test sets the relevance threshold very high so that even a
    correct internal knowledge answer fails the relevance score, verifying
    the gate is actually wired into the _consult_internal_knowledge path.

    NOTE: the gate is applied at the call site (process_turn), NOT inside
    _consult_internal_knowledge itself. So we must call process_turn to
    exercise the wired path — calling _consult_internal_knowledge directly
    bypasses the gate.
    """
    # Use a definitional query that _consult_internal_knowledge will answer
    # from the concept graph (e.g. "what is sleep" has a seed concept)
    reply = engine.process_turn("what is sleep")
    assert reply is not None, "engine should reply to a definitional query"

    # Now set the threshold very high so the relevance gate should reject it
    original_threshold = engine._relevance_threshold.base_threshold
    try:
        engine._relevance_threshold.base_threshold = 1.1  # impossibly high
        gated_reply = engine.process_turn("what is sleep")
        # The reply should NOT contain the internal knowledge answer — it should
        # fall through to uncertainty or some other strategy.
        # We check that the strategy is NOT "internal_knowledge" (meaning the gate
        # rejected it and it fell through to another path).
        assert engine._last_strategy != "internal_knowledge", (
            "relevance gate did not reject a low-relevance internal knowledge answer — "
            "the gate is not wired into the _consult_internal_knowledge path")
    finally:
        engine._relevance_threshold.base_threshold = original_threshold

    # After restoring the threshold, the answer should pass again
    restored_reply = engine.process_turn("what is sleep")
    assert restored_reply is not None, "engine should reply after restoring threshold"


def test_consult_internal_knowledge_gate_rejects_irrelevant_answer(engine):
    """End-to-end: the relevance gate is applied when _consult_internal_knowledge
    is called from process_turn, not just when called directly.

    This verifies the wiring is at the call site, not inside
    _consult_internal_knowledge itself — the gate must be applied by the caller
    so it can be swapped or configured independently.
    """
    # With a very high threshold, the process_turn should NOT return
    # an internal_knowledge answer (it should fall through to uncertainty)
    original_threshold = engine._relevance_threshold.base_threshold
    try:
        engine._relevance_threshold.base_threshold = 1.1
        reply = engine.process_turn("what is gravity")
        # The reply should NOT be an internal_knowledge answer
        # (it should be uncertainty or some other strategy)
        if reply and engine._last_strategy == "internal_knowledge":
            pytest.fail(
                "relevance gate did not reject low-relevance internal knowledge answer "
                "at process_turn level")
    finally:
        engine._relevance_threshold.base_threshold = original_threshold


def test_structured_recall_still_gated(engine):
    """Regression: the existing _structured_recall gating still works.
    This verifies the universal gating change did not break the existing
    structured recall path. We test the gate directly on the structured
    recall answer, since process_turn may route to a fallback path
    (user_profile_recall) that also answers name queries.
    """
    engine.process_turn("my name is alice")

    # With default threshold, structured recall should work
    sr = engine._structured_recall("what is my name")
    assert sr is not None and "alice" in sr.lower(), (
        "structured recall should work at default threshold")

    # The gate should pass at default threshold
    gated = engine._relevance_gate("what is my name", sr)
    assert gated is not None, "gate should pass at default threshold"

    # With very high threshold, the gate should reject it
    original_threshold = engine._relevance_threshold.base_threshold
    try:
        engine._relevance_threshold.base_threshold = 1.1
        gated = engine._relevance_gate("what is my name", sr)
        assert gated is None, "structured recall gate should reject at high threshold"
    finally:
        engine._relevance_threshold.base_threshold = original_threshold
