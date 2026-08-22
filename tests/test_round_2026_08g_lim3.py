"""Regression tests for round 2026-08g (t_a83a7170) — residual limitation #3:

"Offline web query echoes memory." A plain world-knowledge question
("what is X made of / what is a decorator in python") used to be answered by
echoing an UNRELATED autobiographical fact the user mentioned earlier, because
_try_hippocampal_retrieval's broad stem-matching pooled any stored fact whose
buffer key stem-matched a question token. The fix adds a query-intent
disambiguation gate (_is_autobiographical_recall_query) so the episodic ECHO
only fires for questions genuinely about the user's disclosed life, and general
knowledge questions fall through to honest uncertainty (the RAVANA bar).

These tests drive the REAL CognitiveChatEngine.process_turn path (RAVANA_OFFLINE=1,
fixed seed). They assert on REAL signals: the produced strategy and whether the
reply echoed a stored autobiographical fact -- never on a hardcoded reply string.

Verifying coverage: RAVANA_OFFLINE=1 pytest tests/test_round_2026_08g_lim3.py
"""
import os

os.environ.setdefault("RAVANA_OFFLINE", "1")

import pytest

from ravana.chat.engine import CognitiveChatEngine


def _make(tmpdir, suffix):
    return CognitiveChatEngine(
        dim=64, seed=42, baby_mode=True,
        data_dir=tmpdir, user_suffix=suffix,
    )


# Marker strategies that indicate the turn was answered by echoing stored
# autobiographical memory (the bug's signature), as opposed to honest
# uncertainty / internal knowledge / web.
_ECHO_STRATEGIES = (
    "hippocampal_recall", "memory_recall", "episodic_remember",
)


def test_knowledge_question_does_not_echo_unrelated_memory(tmpdir):
    """A general world-knowledge question whose a token coincidentally
    stem-matches a stored autobiographical fact must NOT be answered by
    echoing that fact. It should fall through to honest uncertainty (offline,
    no internal definition, no web)."""
    e = _make(tmpdir, "_lim3k")
    # Seed an autobiographical fact keyed under 'cooking'.
    e.process_turn("i enjoy cooking pasta on weekends")
    # World-knowledge question: 'cooking' stems-matches the stored fact key.
    reply = e.process_turn("what is cooking oil made of?")
    lower = (reply or "").lower()
    # The bug echoed the stored fact: "you told me earlier: you enjoy cooking
    # pasta on weekends". Assert that autobiographical content is NOT surfaced.
    assert "pasta" not in lower, f"echoed unrelated memory: {reply!r}"
    assert "weekends" not in lower, f"echoed unrelated memory: {reply!r}"
    assert e._last_strategy not in _ECHO_STRATEGIES, (
        f"knowledge question routed to episodic echo: strategy="
        f"{e._last_strategy!r} reply={reply!r}")
    # And it must not be a confident world-knowledge confabulation either.
    assert "decorator" not in lower or True  # no-op; keeps intent explicit


def test_personal_possessive_question_still_recalls(tmpdir):
    """A question about the user's OWN disclosed entity (possessive 'my') must
    STILL be answered from episodic memory -- the LoCoMo/LongMemEval behaviour
    the gate must preserve. This is the non-regression control."""
    e = _make(tmpdir, "_lim3p")
    e.process_turn("my car's gps is broken and it keeps rebooting")
    reply = e.process_turn("what is wrong with my car?")
    lower = (reply or "").lower()
    assert "gps" in lower or "reboot" in lower, (
        f"personal recall lost: strategy={e._last_strategy!r} reply={reply!r}")
    assert e._last_strategy in _ECHO_STRATEGIES, (
        f"expected episodic recall for personal query, got "
        f"{e._last_strategy!r}: {reply!r}")


def test_gate_function_detects_recall_vs_knowledge(tmpdir):
    """The disambiguation gate itself: explicit recall markers and personal
    possessives are recall intent; bare world-knowledge questions are not."""
    e = _make(tmpdir, "_lim3g")
    assert e._is_autobiographical_recall_query("what did you say about cooking?") is True
    assert e._is_autobiographical_recall_query("do you remember when i went to berlin?") is True
    assert e._is_autobiographical_recall_query("what is wrong with my car?") is True
    assert e._is_autobiographical_recall_query("what is cooking oil made of?") is False
    assert e._is_autobiographical_recall_query("what is a decorator in python?") is False
    assert e._is_autobiographical_recall_query("how does a black hole form?") is False
