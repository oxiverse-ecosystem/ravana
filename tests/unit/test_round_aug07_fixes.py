"""Regression tests for the 2026-08-07 chat round (persona: Tobias).

Covers the defects that rotating to a fresh persona exposed and that the round
fixed:
  D1  name miner must not store article-led descriptors ("call me a hypocrite")
  D6  stance reversal: softening cue governs; scope-widening ("acoustic-only")
      must NOT flip a broader held stance
  D4  positive disclosures must not be tagged "so mixed?" (joy -> "good")
  D5  non-death distress ("broke me for months") routes to empathy, not
      counterfactual simulation
  D2b preference / episodic fact mining must drop comparative tails
      ("acoustic music over anything produced on a laptop" -> "acoustic music")

No network; RAVANA_OFFLINE=1. Each test drives the engine in-process and reads
REAL state — no mocked replies.
"""
import os
import sys
import re

os.environ.setdefault("RAVANA_OFFLINE", "1")

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (
    _PROJ,
    os.path.join(_PROJ, "ravana_ml", "src"),
    os.path.join(_PROJ, "ravana", "src"),
    os.path.join(_PROJ, "ravana-v2", "src"),
):
    sys.path.insert(0, _p)

import pytest
from ravana.chat.engine import CognitiveChatEngine


@pytest.fixture
def eng():
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                             user_suffix="test_aug07_unittest")
    yield e
    try:
        e.stop_background_learning()
    except Exception:
        pass


def _suffix_rotate():
    """Ensure each test gets an isolated, fresh store."""
    import time
    return "test_aug07_" + str(int(time.time() * 1000))


# ── D1: descriptor names are not stored as identity ────────────────────────
def test_descriptor_name_not_stored(eng):
    eng.process_turn("hi, i'm tobias")
    eng.process_turn("i keep honeybees on my roof")
    eng.process_turn("i said i was a vegetarian but i ate fish, call me a hypocrite")
    fact = eng.user_model.personal_facts.get("i", "name")
    assert fact is None or "hypocrite" not in (fact.value or "").lower(), \
        f"descriptor 'a hypocrite' must not be stored as name, got {fact}"
    # a real bare name still works
    eng2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix=_suffix_rotate())
    try:
        eng2.process_turn("call me tobias")
        f2 = eng2.user_model.personal_facts.get("i", "name")
        assert f2 is not None and f2.value.lower() == "tobias"
    finally:
        eng2.stop_background_learning()


# ── D6: softening relaxes; scope-widening does not flip ─────────────────────
def test_softening_relaxes_then_narrowing_keeps(eng):
    eng.process_turn("i can't stand cilantro, it tastes like soap to me")
    eng.process_turn("i prefer acoustic music over anything produced on a laptop")
    s_cil = eng.user_model.opinions.stances["cilantro"]
    s_ac = eng.user_model.opinions.stances["acoustic music"]
    assert s_cil.polarity < -0.5, f"cilantro should start negative, got {s_cil.polarity}"
    assert s_ac.polarity > 0.5, f"acoustic should start positive, got {s_ac.polarity}"

    # softening recant: "i take it back -- it's not that bad" -> relax toward neutral
    eng.process_turn("actually i take it back — i tried cilantro in a curry and it's not that bad")
    s_cil2 = eng.user_model.opinions.stances["cilantro"]
    assert s_cil2.polarity > -0.5, \
        f"softening should relax cilantro toward neutral, not invert to {s_cil2.polarity}"

    # scope-widening recant: "i was wrong about acoustic-ONLY" must NOT flip
    eng.process_turn("i think i was wrong about acoustic-only, some laptop music is gorgeous")
    s_ac2 = eng.user_model.opinions.stances["acoustic music"]
    assert s_ac2.polarity > 0.5, \
        f"narrowing 'acoustic-only' must NOT flip the 'acoustic music' stance, got {s_ac2.polarity}"


# ── D4: positive disclosure tagged as good, not mixed ──────────────────────
def test_positive_disclosure_not_mixed(eng):
    resp = eng.process_turn("this morning i pulled my first full super of honey and i'm buzzing with joy")
    assert "mixed" not in resp.lower(), \
        f"positive disclosure must not be tagged 'mixed', got: {resp}"
    assert "good" in resp.lower(), f"positive disclosure should ask 'what made today good?', got: {resp}"


# ── D5: non-death distress routes to empathy ───────────────────────────────
def test_nondeath_distress_routes_to_empathy(eng):
    resp = eng.process_turn("losing the hive to foulbrood last year broke me for months")
    assert "what happened" in resp.lower() or "hard" in resp.lower(), \
        f"distress should reach empathy, got: {resp}"
    # must NOT be a counterfactual causal-chain answer
    assert "would likely go" not in resp.lower(), \
        f"distress must not fall through to counterfactual simulation, got: {resp}"


# ── D2b: comparative tail dropped from likes / episodic facts ───────────────
def test_comparative_tail_dropped_from_likes(eng):
    eng.process_turn("i prefer acoustic music over anything produced on a laptop")
    likes = eng.user_model.preferences.get("likes", [])
    assert any("over anything" not in l for l in likes), \
        f"likes must drop comparative tail, got {likes}"
    clean = [l for l in likes if "over anything" not in l]
    assert clean and "acoustic music" in clean[0], \
        f"likes should keep 'acoustic music', got {likes}"
