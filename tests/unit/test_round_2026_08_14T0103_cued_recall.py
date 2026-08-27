"""Regression tests for round 2026-08-14T0103Z — Defect C (part 1): cued
episodic recall must retrieve the TARGETED episode, not an adjacent turn.

A "remember when i told you about the commission" query must return the
commission episode, not whatever turn happened to sit before it. The
root cause was (a) the conversational-recall cue extractor grabbing the
determiner "the" from "about the commission", and (b) the _CUE_STOP set
not excluding question-scaffold words (when/for), which diluted the real
content cue ("commission") to 1-of-3 and failed the 0.34 match bar.
"""

import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _eng():
    sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
    os.environ["RAVANA_OFFLINE"] = "1"
    from ravana.chat.engine import CognitiveChatEngine
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix="test_cue_commission")


def _seed(eng, turns):
    import io, contextlib
    for t in turns:
        with contextlib.redirect_stdout(io.StringIO()):
            eng.process_turn(t)
    eng.stop_background_learning()


def test_cued_recall_targets_commission_not_adjacent():
    eng = _eng()
    _seed(eng, [
        "i'm mira. i'm a ceramicist and printmaker.",
        "i keep a sourdough starter i named bishop and bake before the morning firing.",
        "i gather morels under the beeches but only after the last frost.",
        "i'm over the moon, a collector just commissioned a full dinner set with my river-bird glaze!",
        "this november marks eight years since i first lit a kiln.",
    ])
    out = eng.process_turn(
        "remember when i told you about the commission -- what did i say it was for?")
    assert out is not None, "cued recall returned None"
    assert "commission" in (out or "").lower(), \
        f"cued recall missed the commission episode: {out!r}"
    assert "morels" not in (out or "").lower(), \
        f"cued recall echoed the adjacent morels turn: {out!r}"


def test_cue_extraction_skips_determiner():
    # "what did i tell you about the commission" (cue follows a determiner)
    # must resolve to the commission episode, exercising the cue extractor's
    # leading-determiner skip end-to-end. This is a second phrasing of the
    # same contract as the first test, covering the "about the X" shape where
    # the old regex would have grabbed "the".
    eng = _eng()
    _seed(eng, [
        "i told you about the lighthouse on the cliff.",
        "a collector just commissioned a full dinner set.",
    ])
    out = eng.process_turn("what did i tell you about the commission?")
    assert out is not None and "commission" in (out or "").lower(), \
        f"cue 'commission' not resolved (got {out!r})"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
