#!/usr/bin/env python3
"""Regression tests for AUTOBIOGRAPHICAL RECALL of the USER (feature
t_a41f7e29, round 2026-08-18T0937Z).

Closes the round residual R34/R43/R57/U17: RAVANA's own USER-autobiography
queries were either unhandled (degenerate uncertainty) or MISROUTED into the
agent-own-reply echo store. These tests run through the REAL engine path
(process_turn) so routing regressions are caught, and they assert that the
answer content comes from the live user-model stores (facts / stances), not
from an agent-self echo and not from authored hardcoding.

Self-hardcoding-audit guard: every asserted string is content that RAVANA
mined from the seeded disclosures (e.g. "theo restores vintage radios",
"cold weather hiking"). No test pins an authored sentence that the engine
could not have produced from state.
"""
import os
import sys
import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana", "src"), os.path.join(PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine

SUFFIX = "test_autobio_20260818"


def _eng():
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=SUFFIX)


def _seed_user(e):
    """Drive the REAL miner via process_turn so the user-model stores hold the
    disclosures we then ask about."""
    for q in [
        "my brother theo restores vintage radios",
        "i love cold-weather hiking",
        "i spent my whole childhood in a village by the river",
    ]:
        e.process_turn(q)
    # Give RAVANA a couple of its OWN replies so the agent-reply store is
    # populated — this is the trap the old misroute fell into.
    for q in ["do you like music?", "what do you think about crows?"]:
        e.process_turn(q)


def test_salience_most_about_me_is_store_driven():
    """'what will you remember most about me' must compose from the real user
    profile, NOT be None (the old gap) and NOT be an agent-own echo."""
    e = _eng()
    _seed_user(e)
    out = e.process_turn("what will you remember most about me")
    out = out if isinstance(out, str) else str(out)
    strat = getattr(e, "_last_strategy", None)
    assert strat == "autobiographical_recall", \
        f"expected autobiographical_recall strategy, got {strat}"
    assert out is not None, "expected a composed answer, got None"
    assert out.startswith("the thing that stands out most is "
                           ) or "what stays with me most is" in out
    # content must come from the real facts (theo / river / hiking)
    assert ("theo restores vintage radios" in out
            or "village by the river" in out
            or "cold weather hiking" in out), \
        f"answer not grounded in user store: {out!r}"
    assert out.startswith("i said:") is False, \
        f"answer leaked into agent-own echo: {out!r}"


def test_confirmation_liked_topic_reads_real_user_stance():
    """'did i tell you i liked cold-weather hiking' must confirm from the USER's
    real stance, NOT echo RAVANA's own reply about 'hiking'."""
    e = _eng()
    _seed_user(e)
    out = e.process_turn("did i tell you i liked cold-weather hiking?")
    out = out if isinstance(out, str) else str(out)
    strat = getattr(e, "_last_strategy", None)
    assert strat == "autobiographical_recall", \
        f"expected autobiographical_recall strategy, got {strat}"
    assert out is not None, "expected a confirmation, got None"
    assert "yes" in out.lower()
    assert "cold weather hiking" in out, \
        f"confirmation did not cite the real user stance: {out!r}"
    assert out.startswith("i said:") is False, \
        f"misfired into agent-own echo: {out!r}"


def test_confirmation_unknown_returns_honest_no():
    """A confirmation about something never disclosed must be honest, not a
    fabricated yes."""
    e = _eng()
    _seed_user(e)
    out = e.process_turn("did i tell you i liked underwater basket weaving?")
    out = out if isinstance(out, str) else str(out)
    strat = getattr(e, "_last_strategy", None)
    assert strat == "autobiographical_recall", \
        f"expected autobiographical_recall strategy, got {strat}"
    assert "not that i recall" in out, f"expected honest no, got: {out!r}"


def test_family_mention_confirmation_reads_fact():
    """'have i told you about my brother' must answer from the real brother
    fact (theo restores vintage radios)."""
    e = _eng()
    _seed_user(e)
    out = e.process_turn("have i told you about my brother?")
    out = out if isinstance(out, str) else str(out)
    strat = getattr(e, "_last_strategy", None)
    assert strat == "autobiographical_recall", \
        f"expected autobiographical_recall strategy, got {strat}"
    assert out is not None and "theo" in out, \
        f"expected brother fact, got: {out!r}"


def test_contradiction_reconcile_reports_current_stance():
    """'does that still fit / have i changed' must report the user's CURRENT
    (already-reconciled) stance, not a stale agent echo (U17)."""
    e = _eng()
    _seed_user(e)
    # Simulate the user softening the stance, then reconciling.
    e.process_turn("wait, my knee's been acting up — i can't really hike like i used to")
    out = e.process_turn(
        "earlier i told you i loved cold-weather hiking. does that still fit, or have i changed?")
    out = out if isinstance(out, str) else str(out)
    strat = getattr(e, "_last_strategy", None)
    assert strat == "autobiographical_recall", \
        f"expected autobiographical_recall strategy, got {strat}"
    assert out is not None, "expected a reconcile answer, got None"
    assert "cold weather hiking" in out, \
        f"reconcile did not cite the real topic: {out!r}"
    # The stance was softened to ~0.0 (round 2026-08-18 Unit B fix); the answer
    # should reflect a softened / non-strongly-for state, not "still strongly for".
    assert "strongly for" not in out, \
        f"reconcile surfaced stale strong stance: {out!r}"
    assert out.startswith("i said:") is False, \
        f"reconcile leaked into agent-own echo: {out!r}"


def test_agent_self_question_still_falls_through():
    """A genuine agent-self query ('what did YOU say about music') must NOT be
    intercepted by autobiographical recall — it lacks first-person user framing."""
    e = _eng()
    _seed_user(e)
    e._record_own_reply("what do you think about music", "i think music is a way i process the day.", "music")
    out = e.process_turn("what did you say about music?")
    out = out if isinstance(out, str) else str(out)
    strat = getattr(e, "_last_strategy", None)
    assert strat != "autobiographical_recall", \
        f"autobiographical gate wrongly intercepted agent-self query (strategy={strat}): {out!r}"


def teardown_module(module):
    try:
        e = _eng()
        e.stop_background_learning()
    except Exception:
        pass
