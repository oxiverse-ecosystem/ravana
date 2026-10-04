#!/usr/bin/env python3
"""Regression tests — a repeated DECLARATION is reinforcement, not a correction
(round 2026-10-04T0827Z, defect D3).

Measured defect (spying on `_detect_correction`, scratch/_dbg8_0827z.log):

  Q "i think public libraries are underrated"
  Q "i think public libraries are underrated"     <- EXACT repeat
    CORRECTION FLAGGED INDIRECT_REASK sev=0.30 fact=None
  A "thanks — i'll be more careful there. what should i have said?"

  Q "my bike is a 1993 tourer"
  Q "my bike is a 1993 tourer"                  <- EXACT repeat
    CORRECTION FLAGGED INDIRECT_REASK sev=0.30 fact=None

The user was emphasising themselves, not correcting RAVANA. Root cause: the
re-ask stream tested token overlap and nothing else, so a restated DECLARATION
and a repeated QUESTION — which are lexically identical — were the same event.

The distinguishing feature is the speech act. These tests assert the class
property (a re-ask is interrogative or an explicit repeat request) using
phrasings and topics that appear nowhere in the probe above.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana", "src"), os.path.join(PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.chat.user_model import _is_reask


def test_declarations_are_not_reasks():
    """A restated declaration is reinforcement, however much it overlaps."""
    for q in (
        "i think public libraries are underrated",
        "my bike is a 1993 tourer painted the wrong shade",
        "my grandmother in kochi knits fishing nets",
        "quiet mornings are underrated",
        "the harbour road is badly maintained",
        "i have three sisters",
    ):
        assert not _is_reask(q), f"declaration misread as a re-ask: {q!r}"


def test_questions_are_reasks():
    """The capability is not disabled: a real re-ask still registers."""
    for q in (
        "what is my bike",
        "who is my grandmother",
        "where was the shop",
        "why did you say that",
        "how does a kiln work",
        "is that right",
        "do you remember what i told you",
        "which one did you mean",
        "remind me what you said about kilns",
        "say that again",
        "you didn't hear me",
    ):
        assert _is_reask(q), f"genuine re-ask not detected: {q!r}"


def test_declaration_containing_an_interrogative_word_is_not_a_reask():
    """A wh-word or auxiliary only counts interrogative at the FRONT.

    Without this, "my bike is a 1993 tourer" would match on "is" and every
    declaration would become a re-ask again — the bug in a new shape.
    """
    for q in (
        "my bike is a 1993 tourer",
        "the kiln where i learned pottery is broken",
        "i wonder what you think of it",
        "my uncle who ran the shop retired",
        "howard, who fixed my bike, moved away",
    ):
        assert not _is_reask(q), f"interior interrogative word leaked: {q!r}"


def test_empty_and_degenerate_input():
    assert not _is_reask("")
    assert not _is_reask("   ")
    assert not _is_reask(None)


def test_repeated_declaration_is_not_flagged_as_a_correction():
    """End-to-end at the detector: two identical declarations, no correction."""
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_d3_reask_0827z")
    seen = []
    orig = eng.user_model._detect_correction

    def spy(query, subject, valence):
        r = orig(query, subject, valence)
        if getattr(eng.user_model, "detected_correction", False):
            seen.append((query, getattr(eng.user_model.detected_correction_type,
                                        None),
                         eng.user_model.correction_severity))
        return r
    eng.user_model._detect_correction = spy

    eng.process_turn("i think lighthouses are underrated")
    eng.process_turn("i think lighthouses are underrated")
    assert not seen, f"a repeated declaration was flagged as a correction: {seen!r}"
    try:
        eng.stop_background_learning()
    except Exception:
        pass