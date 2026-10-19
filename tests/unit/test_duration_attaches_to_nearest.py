"""Test: duration mining attaches to the NEAREST preceding activity verb.

Round 2026-10-19T0513Z limitation #1: "my favourite instrument is the veena
and i have been playing it for five years" — the duration "for five years" was
dropped because the duration miner picked the FIRST activity-ok verb in the
clause ("is") instead of the LAST ("playing"). A duration attaches to the
nearest preceding activity, not the first verb in the clause.

This test verifies the fix: the since-fact stores the nearest activity verb,
and recall renders the correct duration.
"""
import os
import re
import pytest

os.environ["RAVANA_OFFLINE"] = "1"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ravana_ml", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ravana-v2", "src"))

from ravana.chat.engine import CognitiveChatEngine


@pytest.fixture
def eng():
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_dur_attach")
    yield e
    e.stop_background_learning()


def test_duration_attaches_to_nearest_activity(eng):
    """Duration 'for five years' attaches to 'playing', not 'is'."""
    eng.process_turn(
        "my favourite instrument is the veena and i have been playing it for five years"
    )
    pf = eng.user_model.personal_facts
    since_facts = {k: v for k, v in pf.facts.items()
                   if isinstance(k, tuple) and len(k) == 3 and k[1] == "since"}
    # There must be at least one since fact
    assert since_facts, "No since fact was mined"
    # The since fact must reference "play" (from "playing"), not "is"
    vals = [v.value.lower() for v in since_facts.values()]
    assert any("play" in v for v in vals), (
        f"Duration did not attach to 'playing'. Since facts: {vals}"
    )
    assert not any(v.startswith("is ") or v == "is" for v in vals), (
        f"Duration incorrectly attached to 'is'. Since facts: {vals}"
    )


def test_duration_recall_renders_correct_activity(eng):
    """Recall query returns the correct activity with duration."""
    eng.process_turn(
        "my favourite instrument is the veena and i have been playing it for five years"
    )
    reply = eng.process_turn("how long have i been playing the veena")
    # Must mention the duration and the activity
    assert "year" in reply.lower(), f"Duration not in reply: {reply}"
    assert "play" in reply.lower(), f"Activity 'play' not in reply: {reply}"


def test_duration_recall_start_year(eng):
    """Recall query for start year returns the correct year."""
    eng.process_turn(
        "my favourite instrument is the veena and i have been playing it for five years"
    )
    reply = eng.process_turn("when did i start playing the veena")
    # 2026 - 5 = 2021
    assert "2021" in reply, f"Start year 2021 not in reply: {reply}"


def test_simple_duration_still_works(eng):
    """Simple single-clause duration still works (no regression)."""
    eng.process_turn("i've been repairing tube amps for eleven years")
    reply = eng.process_turn("how long have i been repairing tube amps")
    assert "year" in reply.lower(), f"Duration not in reply: {reply}"
    assert "repair" in reply.lower(), f"Activity 'repair' not in reply: {reply}"


def test_multiclause_duration_picks_last_verb(eng):
    """In a multi-clause sentence, duration picks the nearest verb."""
    eng.process_turn(
        "i live in mumbai and i have been practicing yoga for three years"
    )
    pf = eng.user_model.personal_facts
    since_facts = {k: v for k, v in pf.facts.items()
                   if isinstance(k, tuple) and len(k) == 3 and k[1] == "since"}
    vals = [v.value.lower() for v in since_facts.values()]
    # Should attach to "practicing" (nearest), not "live" (first)
    assert any("practic" in v for v in vals), (
        f"Duration did not attach to 'practicing'. Since facts: {vals}"
    )
