"""Test that trailing discourse adverbs do NOT leak into stance keys.

Round 2026-10-04T0827Z limitation #1: "actually i love board games now"
created stance 'board games now' alongside 'board games', so the next
recall about board games reads the weaker one.

The fix is structural: add "now" to _OPINION_STOP so _opinion_topic
trims it from the head phrase. This is the same closed-class stop-word
pattern as the existing entries ("really", "very", "just", etc.).
"""
import pytest


def _eng(suffix):
    """Build a fresh CognitiveChatEngine with an isolated suffix."""
    import os, sys
    os.environ["RAVANA_OFFLINE"] = "1"
    PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
    for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src", f"{PROJ}\\ravana-v2\\src"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from ravana.chat.engine import CognitiveChatEngine
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


def test_trailing_now_not_in_stance_key():
    """'i love board games now' must NOT create a stance keyed on 'board games now'."""
    eng = _eng("test_trailing_now")
    eng.process_turn("i love board games now")
    stances = eng.user_model.opinions.stances
    # The stance key should be "board games", NOT "board games now"
    assert "board games" in stances, f"Expected 'board games' in stances, got: {list(stances.keys())}"
    assert "board games now" not in stances, (
        f"Trailing adverb 'now' leaked into stance key: {list(stances.keys())}"
    )


def test_trailing_now_stance_recall_works():
    """After 'i love board games now', recalling the stance must find it."""
    eng = _eng("test_trailing_now_recall")
    eng.process_turn("i love board games now")
    # Now ask about board games — should recall the stance
    r = eng.process_turn("what do i think about board games")
    # The reply should reflect the stance (not "still forming a view")
    assert "still forming" not in (r or ""), (
        f"Stance not recalled — trailing adverb broke the key: {r}"
    )


def test_trailing_today_not_in_stance_key():
    """'i love board games today' must NOT create a stance keyed on 'board games today'."""
    eng = _eng("test_trailing_today")
    eng.process_turn("i love board games today")
    stances = eng.user_model.opinions.stances
    assert "board games" in stances, f"Expected 'board games' in stances, got: {list(stances.keys())}"
    assert "board games today" not in stances, (
        f"Trailing adverb 'today' leaked into stance key: {list(stances.keys())}"
    )


def test_trailing_tomorrow_not_in_stance_key():
    """'i love board games tomorrow' must NOT create a stance keyed on 'board games tomorrow'."""
    eng = _eng("test_trailing_tomorrow")
    eng.process_turn("i love board games tomorrow")
    stances = eng.user_model.opinions.stances
    assert "board games" in stances, f"Expected 'board games' in stances, got: {list(stances.keys())}"
    assert "board games tomorrow" not in stances, (
        f"Trailing adverb 'tomorrow' leaked into stance key: {list(stances.keys())}"
    )


def test_trailing_yesterday_not_in_stance_key():
    """'i love board games yesterday' must NOT create a stance keyed on 'board games yesterday'."""
    eng = _eng("test_trailing_yesterday")
    eng.process_turn("i love board games yesterday")
    stances = eng.user_model.opinions.stances
    assert "board games" in stances, f"Expected 'board games' in stances, got: {list(stances.keys())}"
    assert "board games yesterday" not in stances, (
        f"Trailing adverb 'yesterday' leaked into stance key: {list(stances.keys())}"
    )
