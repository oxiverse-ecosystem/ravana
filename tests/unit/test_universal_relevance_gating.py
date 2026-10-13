"""Capability test: UNIVERSAL RELEVANCE GATING across all recall paths.

The MRS relevance gate was originally wired only into _structured_recall.
The episodic recall path (_self_cued_episodic) and agent-own recall path
(_route_agent_own_recall) could still confabulate — they matched by
content-word overlap without verifying the answer actually addresses the
query. This capability wires the same gate into the episodic recall path
so ALL recall paths are relevance-gated.

The test verifies that _self_cued_episodic answers are relevance-gated:
an answer that passes the cue match but fails the relevance score must
return None (honest uncertainty beats confabulation).
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
    suffix = "univrelgate"
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


def test_self_cued_episodic_answer_is_relevance_gated(engine):
    """The capability: _self_cued_episodic answers pass through the
    relevance gate at the process_turn call site. An answer that passes
    the cue match but fails the relevance score must return None.

    This test sets the relevance threshold very high so that even a
    correct cue match fails the relevance score, verifying the gate
    is actually wired into the episodic recall path.

    NOTE: the gate is applied at the call site (process_turn), NOT inside
    _self_cued_episodic itself. So we must call process_turn to exercise
    the wired path — calling _self_cued_episodic directly bypasses the gate.
    """
    engine.process_turn("my cousin meera restores antique clocks in pune")

    # With default threshold, the answer should pass through process_turn
    reply = engine.process_turn("where does meera restore clocks")
    assert reply is not None and "meera" in reply.lower(), (
        "capability did not fire on a cued question")

    # Now set the threshold very high so the relevance gate should reject it
    original_threshold = engine._relevance_threshold.base_threshold
    try:
        engine._relevance_threshold.base_threshold = 1.1  # impossibly high
        gated_reply = engine.process_turn("where does meera restore clocks")
        # The reply should NOT contain the episodic answer — it should
        # fall through to uncertainty or some other strategy
        if gated_reply and "meera" in gated_reply.lower() and "pune" in gated_reply.lower():
            pytest.fail(
                "relevance gate did not reject a low-relevance answer — "
                "the gate is not wired into the episodic recall path")
    finally:
        engine._relevance_threshold.base_threshold = original_threshold

    # After restoring the threshold, the answer should pass again
    restored_reply = engine.process_turn("where does meera restore clocks")
    assert restored_reply is not None and "meera" in restored_reply.lower(), (
        "answer should pass after restoring threshold")


def test_relevance_gate_is_applied_at_process_turn_level(engine):
    """End-to-end: the relevance gate is applied when _self_cued_episodic
    is called from process_turn, not just when called directly.

    This verifies the wiring is at the call site (line ~7125), not inside
    _self_cued_episodic itself — the gate must be applied by the caller
    so it can be swapped or configured independently.
    """
    engine.process_turn("my cousin meera restores antique clocks in pune")

    # With a very high threshold, the process_turn should NOT return
    # a self_cued_episodic answer (it should fall through to uncertainty)
    original_threshold = engine._relevance_threshold.base_threshold
    try:
        engine._relevance_threshold.base_threshold = 1.1
        reply = engine.process_turn("where does meera restore clocks")
        # The reply should NOT be a self_cued_episodic answer
        # (it should be uncertainty or some other strategy)
        if reply and "meera" in reply.lower() and "pune" in reply.lower():
            # If it still returned the episodic answer, the gate is not wired
            pytest.fail(
                "relevance gate did not reject low-relevance episodic answer "
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
