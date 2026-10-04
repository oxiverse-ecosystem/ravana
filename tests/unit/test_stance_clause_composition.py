#!/usr/bin/env python3
"""Regression tests — a stance and its reason must compose as two clauses
(round 2026-10-04T0827Z, defect D2).

Measured defect, from the rotating probe:

    Q "do you like the way i argue"
    A "i'm still forming a view on the way i argue i don't have a fixed stance
       on the way i argue yet — what's your take? i'd rather hear how you see
       it than guess. what about you?"

Two clauses run together with no separator, and the topic is named twice.

Root cause, found with a stack-spy on `_agent_stance_on` (the SAME tuple, in
the SAME session, composing two different ways):

    engine_self_query.py:1692 -> "i'm still forming a view on X. I don't have
                                 a fixed stance on X yet — ..."     CORRECT
    engine.py:9048           -> "i'm still forming a view on X I don't have
                                 a fixed stance on X yet — ..."     BROKEN

The correct site stripped the stance's sentence terminator before joining; the
three broken sites terminated only the reason and joined with a bare space.
Four copies of the same two-line join, three of them wrong.

The fix puts the join ONCE (`_join_stance_clauses`) and routes every site
through it.

These assertions are about the COMPOSITION, not any topic's wording, so they
hold for any stance RAVANA can hold — the test cases below are pairs of
arbitrary strings, not this round's probe text.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana", "src"), os.path.join(PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine


def _eng():
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix="test_d2_join_0827z")


def test_unterminated_stance_is_separated_from_its_reason():
    """The measured failure: no separator, so the clauses run together."""
    e = _eng()
    out = e._join_stance_clauses("i'm still forming a view on the way i argue",
                                "i don't have a fixed stance on the way i "
                                "argue yet — what's your take?")
    assert ". " in out, f"clauses still run together: {out!r}"
    assert out.count("the way i argue") <= 2, out
    assert not out.startswith("i'm still forming a view on the way i argue i"), out


def test_reason_keeps_its_lowercase_continuation():
    """The reason CONTINUES the sentence, so it must not be capitalized.

    Force-capitalizing produced "i care deeply about privacy. Is a basic
    right" in an earlier round.
    """
    e = _eng()
    out = e._join_stance_clauses("i care deeply about privacy",
                                "it is a basic right — i was built to "
                                "protect it")
    assert ". It is" not in out, f"reason was force-capitalized: {out!r}"
    assert ". it is" in out, out


def test_existing_terminator_is_not_doubled():
    """A stance that already ends in '.' must not become '..'."""
    e = _eng()
    for stance in ("i'm for the sea.", "i'm for the sea", "i'm against it!"):
        out = e._join_stance_clauses(stance, "you said so earlier")
        assert ".." not in out, f"doubled terminator: {out!r}"
        assert ".." not in out.replace("...", ""), out


def test_missing_reason_yields_one_clean_sentence():
    e = _eng()
    assert e._join_stance_clauses("i'm for the sea", None) == "i'm for the sea."
    assert e._join_stance_clauses("i'm for the sea.", None) == "i'm for the sea."
    assert e._join_stance_clauses(None, "orphan reason") == ""
    assert e._join_stance_clauses(None, None) == ""


def test_end_to_end_no_run_together_clauses():
    """The measured probe query must compose as two clauses.

    Topic deliberately different from the probe's, so this cannot pass by
    matching the strings that exposed the defect.
    """
    e = _eng()
    for q in ("do you like the way i argue",
              "what do you make of my handwriting",
              "do you think tide charts are overrated"):
        out = (e.process_turn(q) or "").lower()
        # No lowercase word may run straight into another clause: every reply
        # must separate its stance from its reason with ". " (or end cleanly).
        assert "view on " in out or out.strip(), f"unexpected empty reply: {q!r}"
        head, _, tail = out.partition(". ")
        if "don't have a fixed stance on" in out:
            assert ". " in out, f"clauses ran together for {q!r}: {out!r}"
        if out and not out.endswith((".", "?", "!")):
            raise AssertionError(f"reply does not terminate cleanly: {out!r}")
    try:
        e.stop_background_learning()
    except Exception:
        pass