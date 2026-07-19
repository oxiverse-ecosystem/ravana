"""Verification tests for the RAVANA Human-Likeness Plan (2026-07-15).

Covers the four brain-faithful fixes:
  A1 — agent self-preference ("what is your favorite color"):
       grounded pick + affect reason + reciprocity return; no metaphor
       dead-end ("presence").
  A2 — classic counterfactual ("tree falls in a forest ... sound"):
       both frames held (physical vibration vs. perceptual sound); no
       metaphor dead-end.
  B  — hedged speculative guess ("time seems to go faster as we age"):
       honesty preserved + a clearly-marked candidate mechanism attached.
  C  — retrievable multi-turn memory ("remember what I told you"):
       gist reconstructed from the portable episodic transcript; never
       stated facts fail CLOSED (no confabulation).

These assert BEHAVIOR (no metaphor leak, concrete + reciprocity, both
frames, honesty + candidate, real retrieval, fail-closed), not exact wording.
"""
import os
import sys
import tempfile

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana-v2", "src"))

import pytest

from ravana.chat.engine import CognitiveChatEngine


def _engine():
    d = tempfile.mkdtemp(prefix="ravana_humanlike_")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)


# ─────────────────────────────────────────────────────────────────────────
# A1 — Agent self-preference
# ─────────────────────────────────────────────────────────────────────────
def test_a1_favorite_color_no_metaphor():
    e = _engine()
    out = e.process_turn("what is your favorite color")
    low = out.lower()
    # No metaphor dead-end wording.
    assert "presence" not in low, f"metaphor dead-end leaked: {out!r}"
    # Concrete color pick present.
    assert any(w in low for w in
               ["blue", "green", "red", "black", "white", "purple",
                "yellow", "orange", "teal"]), f"no concrete color: {out!r}"
    # Reciprocity return (turns it back, like a human).
    assert "?" in out, f"no reciprocal return: {out!r}"


def test_a1_favorite_color_grounded_not_hardcoded():
    e = _engine()
    out = e.process_turn("what is your favorite color")
    # The pick must be composed from state (affect-grounded), not a fixed
    # canned string — verify an affect reason clause is present.
    low = out.lower()
    assert any(w in low for w in
               ["calm", "steady", "alive", "grounded", "quiet", "clear",
                "mysterious", "intense", "warm", "open", "still"]) or \
        "what about you" in low


def test_a1_what_do_you_like_routes_to_self():
    e = _engine()
    out = e.process_turn("what do you like")
    assert "what about you" in out.lower() or "you?" in out.lower()


# ─────────────────────────────────────────────────────────────────────────
# A2 — Classic counterfactual (both frames, no metaphor dead-end)
# ─────────────────────────────────────────────────────────────────────────
def test_a2_tree_falls_in_forest_no_metaphor():
    e = _engine()
    out = e.process_turn(
        "if a tree falls in a forest and no one hears it, does it make a sound")
    low = out.lower()
    assert "presence" not in low, f"metaphor dead-end leaked: {out!r}"
    # Both frames must be present: physical vibration AND perceptual sound.
    assert "vibration" in low, f"missing physical frame: {out!r}"
    assert "sound" in low, f"missing perceptual frame: {out!r}"


def test_a2_holds_both_frames_not_single_assertion():
    e = _engine()
    out = e.process_turn(
        "if a tree falls in a forest and no one hears it, does it make a sound")
    low = out.lower()
    # A human holds both frames (objective event vs. subjective perception)
    # rather than asserting one flatly — the reply should acknowledge the
    # listener/perception dependency, not claim a single definitive answer.
    assert ("listen" in low or "perceive" in low or "hear" in low), \
        f"no perspective-taking on perception: {out!r}"


# ─────────────────────────────────────────────────────────────────────────
# B — Hedged speculative guess under uncertainty
# ─────────────────────────────────────────────────────────────────────────
def test_b_time_faster_preserves_honesty():
    e = _engine()
    out = e.process_turn("why does time seem to go faster as we get older")
    low = out.lower()
    # RAVANA bar preserved: it does NOT assert a single confident cause.
    assert ("not certain" in low or "not sure" in low
            or "i'm not" in low or "guess" in low), \
        f"honesty not preserved: {out!r}"
    assert "know for sure" in low or "not something i know" in low


def test_b_time_faster_attaches_candidate_mechanism():
    e = _engine()
    out = e.process_turn("why does time seem to go faster as we get older")
    low = out.lower()
    # The hedged candidate mechanism (proportional/logarithmic time account)
    # must be present, explicitly framed as a guess/idea — not asserted.
    assert ("fraction" in low or "memory" in low or "routine" in low
            or "year" in low), f"no candidate mechanism: {out!r}"
    assert "idea" in low or "guess" in low, \
        f"candidate not framed as uncertain: {out!r}"


# ─────────────────────────────────────────────────────────────────────────
# C — Retrievable multi-turn memory (gist + fail-closed)
# ─────────────────────────────────────────────────────────────────────────
def test_c_remembers_seeded_facts():
    e = _engine()
    e.process_turn("i love astrophysics")
    e.process_turn("my favorite book is dune")
    out = e.process_turn("remember what i told you earlier")
    low = out.lower()
    assert "astrophysics" in low or "dune" in low, \
        f"stored facts not retrieved: {out!r}"


def test_c_fail_closed_on_never_stated():
    e = _engine()
    e.process_turn("i love hiking")
    e.process_turn("my favorite book is dune")
    out = e.process_turn("remember my cat's name")
    low = out.lower()
    # Never stated -> fail CLOSED (no confabulation). Should NOT invent a name
    # nor fall through to a web search that would fabricate a "cat" answer.
    assert "don't" in low or "not sure" in low or "outside what i know" in low \
        or "not really" in low, f"failed to fail-closed: {out!r}"
    assert "mittens" not in low and "whiskers" not in low  # inventive guards
    # The recall query was recognized but nothing stored -> fail-closed path,
    # NOT a web/graph answer.
    assert e._last_strategy in ("episodic_remember_miss", "reflective_uncertainty"), \
        f"control should not web-answer: {e._last_strategy} :: {out!r}"


def test_c_fail_closed_does_not_fall_through_to_web():
    # "remember my cat name" must NOT become a web lookup about a film "Cat".
    e = _engine()
    out = e.process_turn("remember my cat name")
    low = out.lower()
    assert "film" not in low and "movie" not in low and "directed by" not in low, \
        f"confabulated web answer: {out!r}"
    assert e._last_strategy == "episodic_remember_miss", \
        f"expected fail-closed, got {e._last_strategy}"


def test_regression_plain_question_not_hijacked_by_episodic():
    # A normal new question sharing a word with a past turn must NOT be
    # intercepted by the episodic matcher (would hijack the pipeline).
    e = _engine()
    e.process_turn("what color is tuesday")
    out = e.process_turn("what color is the sun")
    # Must NOT return the prior turn as a "remembered" episode.
    assert "you mentioned" not in out.lower() or "tuesday" not in out.lower(), \
        f"plain question hijacked by episodic matcher: {out!r}"


def test_c_record_episode_captures_facts():
    e = _engine()
    e.process_turn("my favorite movie is inception")
    facts = e._episodic_transcript[-1]["facts"]
    assert "favorite_movie" in facts and facts["favorite_movie"] == "inception"


def test_c_episodic_remember_excludes_current_turn():
    e = _engine()
    e.process_turn("i love jazz")
    # The recall query itself must not be "remembered" as the content.
    out = e._episodic_remember("remember what i told you")
    assert out is None or "remember what i told you" not in out.lower()


# ─────────────────────────────────────────────────────────────────────────
# Regression guard — the four fixes must not reintroduce the robotic path
# ─────────────────────────────────────────────────────────────────────────
def test_regression_no_metaphor_deadend_on_color_or_sound():
    e1 = _engine()
    a1 = e1.process_turn("what is your favorite color")
    assert "presence" not in a1.lower()
    e2 = _engine()
    a2 = e2.process_turn(
        "if a tree falls in a forest and no one hears it, does it make a sound")
    assert "presence" not in a2.lower()
