"""Capability test: SELF-CUED EPISODIC RETRIEVAL (retrieval-by-cue).

A follow-up question about content the user ALREADY disclosed must be
answered from RAVANA's own record, not from the world. Before this
capability the turn fell through to _consult_internal_knowledge /
web_search: the round probe showed "where does meera restore clocks"
firing IntentForge web search and returning Amazon clock listings while
RAVANA already held the answer from the user's own prior disclosure.

The same content as the motivating probe is used on purpose, so the fix is
measured against the real defect. Each test gets a FRESH engine so no
result depends on test ordering.
"""
import os
import sys

import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for _p in (PROJ,
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

DISCLOSURE = "my cousin meera restores antique clocks in pune"
QUESTION = "where does meera restore clocks"


@pytest.fixture
def engine():
    suffix = "selfcuedtest"
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


def test_question_about_disclosed_entity_is_answered_from_own_record(engine):
    """The capability: a cue-matched question recalls the user's disclosure."""
    reply = engine.process_turn(DISCLOSURE)
    assert reply  # disclosure is acked

    answer = engine._self_cued_episodic(QUESTION)
    assert answer is not None, "capability did not fire on a cued question"
    # The answer must carry the USER's stored content, not a world fact.
    assert "meera" in answer.lower()
    assert "pune" in answer.lower()


def test_full_turn_routes_to_self_cued_strategy_and_skips_web(engine):
    """End-to-end: the turn answers from memory and never reaches the web."""
    engine.process_turn(DISCLOSURE)
    engine._pending_web_evidence = None
    reply = engine.process_turn(QUESTION)
    assert engine._last_strategy == "self_cued_episodic", (
        f"wrong strategy: {engine._last_strategy} / reply={reply!r}")
    assert "pune" in reply.lower()
    assert engine._pending_web_evidence is None, (
        "web evidence was fetched despite the user already disclosing it")


def test_world_query_with_no_stored_cue_fails_closed(engine):
    """Honest failure: nothing stored about France -> no invented recall."""
    engine.process_turn(DISCLOSURE)
    assert engine._self_cued_episodic(
        "what is the capital of france") is None


def test_second_disclosure_is_independently_recallable(engine):
    """Generalization: the capability is not tied to the first disclosure."""
    engine.process_turn(DISCLOSURE)
    engine.process_turn("my brother kiran repairs bicycles in jaipur")
    hit = engine._self_cued_episodic("where does kiran repair bicycles")
    assert hit is not None
    assert "jaipur" in hit.lower()
    # A question carrying only scaffolding has no content cue at all.
    assert engine._self_cued_episodic("what is it about") is None
