"""Regression: the activity resolver must not match a verb as a SUBSTRING.

Defect (round 2026-09-30T1031Z, chat probe TURN 15): asked "what do i do for
a living" after disclosing a sourdough starter, RAVANA answered
"you start sourdough starter." — a confident confabulation.

Root cause: the `_ACT` alternation in engine.py contains the bare verb `do`,
so the occupation question matched with _verb="do"; the resolver then tested
`_verb in _val` as a raw substring, and "do" occurs inside "sourDOugh".

These tests pin the WORD-BOUNDARY contract generally, and pin the
confabulation itself. If a future edit reintroduces substring matching, the
verb must not match inside a larger word.
"""
import os, sys
os.environ["RAVANA_OFFLINE"] = "1"

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana", "src"),
          os.path.join(PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine, _has_word


def test_has_word_requires_whole_word():
    """_has_word matches whole words, not substrings."""
    assert _has_word("do", "do pottery") is True
    assert _has_word("do", "i do ceramics") is True
    # The exact regression: "do" is inside "sourdough".
    assert _has_word("do", "start sourdough starter") is False
    assert _has_word("run", "a runner went past") is False
    assert _has_word("make", "remake the bench") is False
    assert _has_word("eat", "a great eaten meal") is False
    # A real standalone word still matches even when surrounded by others.
    assert _has_word("read", "already read it") is True
    assert _has_word("", "anything") is False
    assert _has_word("do", "") is False


def test_occupation_question_does_not_confabulate_from_unrelated_activity():
    """A bare-verb occupation question must not echo an unrelated fact.

    This is the chat-probe transcript, pinned. The disclosure is about a
    sourdough starter; the question is about work. Answering with the starter
    is a confabulation, and an honest fallback is strictly better.
    """
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_verb_substring_0930")
    eng.process_turn("i started a sourdough starter on tuesday and named it barty")

    reply = str(eng.process_turn("what do i do for a living")).lower()
    assert "sourdough" not in reply, (
        "occupation question confabulated from an unrelated activity fact: "
        f"{reply!r}"
    )
    eng.stop_background_learning()


def test_legitimate_activity_recall_still_works():
    """Whole-word matching must not break genuine activity recall."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_verb_substring_0930b")
    eng.process_turn("i bake bread every sunday")
    reply = str(eng.process_turn("what do i bake")).lower()
    assert "bread" in reply, f"activity recall regressed: {reply!r}"
    eng.stop_background_learning()
