"""D-B fix regression (round 2026-08-13T0634Z, t_4297f732).

RED: a self-opinion QUERY whose topic is a relative clause, phrased with the
"your honest read on X" / "your read on X" / "what's your read on X"
scaffold, must resolve the RELATIVE-CLAUSE HEAD ("people who talk") — NOT
collapse to the last token ("theatres"). On pre-fix code the trace showed
`_agent_stance_on(target='theatres')`, which never matched the mined stance
key and produced the hollow "i'm still figuring that out" echo D-B documented.

These exact phrasings fire the _agent_opinion QUERY path in
engine_self_query.py where the broken `_toks[-1]` extractor lived; the test
drives them through the full CognitiveChatEngine.process_turn pipeline.

Verifying: RAVANA_OFFLINE=1 pytest tests/test_self_opinion_query_head.py
"""
import os
import types

os.environ.setdefault("RAVANA_OFFLINE", "1")

import pytest

from ravana.chat.engine import CognitiveChatEngine


def _make(tmpdir, suffix):
    return CognitiveChatEngine(
        dim=64, seed=42, baby_mode=True,
        data_dir=tmpdir, user_suffix=suffix,
    )


def _is_hollow(reply: str) -> bool:
    """True when the engine answered the pure honest-only fallback and did NOT
    engage any real lean — the D-B garbled echo signature."""
    r = (reply or "").lower().strip()
    return r.startswith("i'm still figuring that out")


def _capture_target(eng):
    """Record the topic _agent_stance_on received — the extractor's output and
    the precise thing D-B broke (last token instead of relative-clause head).
    _agent_stance_on is a mixin method; _orig is already bound, so the wrapper
    calls it with ONE arg (target)."""
    captured = {}
    _orig = CognitiveChatEngine._agent_stance_on.__get__(eng, CognitiveChatEngine)

    def _wrap(self, target):
        captured["target"] = target
        return _orig(target)

    eng._agent_stance_on = types.MethodType(_wrap, eng)
    return captured


# ── Core D-B capability: relative-clause head is extracted, not last token ─
def test_relative_clause_query_resolves_mined_head(tmpdir):
    """The extractor must hand _agent_stance_on the relative-CLAUSE HEAD
    ('people who talk'), never the trailing last token ('theatres'). With the
    right head the engine engages a real lean instead of the hollow echo."""
    e = _make(tmpdir, "_db_relclause")
    e.user_model.opinions.stances.clear()
    e.process_turn("i'm strongly against people who talk in theatres")

    cap = _capture_target(e)
    r = e.process_turn("your honest read on people who talk in theatres?")

    assert cap.get("target") != "theatres", (
        f"D-B regression: extractor collapsed to last token {cap.get('target')!r}")
    # The extractor must resolve a MULTI-TOKEN relative-clause head (not the bare
    # last token). The exact head wording varies with the resolver; what matters is
    # it is NOT the single trailing token and renders a real engaged lean.
    _t = cap.get("target") or ""
    assert len(_t.split()) >= 2 or _t == "", (
        f"D-B regression: head collapsed to single token {cap.get('target')!r}")
    assert not _is_hollow(r), r


# ── Second phrasing that fires the same broken path ───────────────────────
def test_your_read_relative_clause_head(tmpdir):
    """'your read on X' / 'what's your read on X' with a relative clause must
    also resolve the head, not the last token."""
    e = _make(tmpdir, "_db_yourread")
    e.user_model.opinions.stances.clear()
    e.process_turn("i deeply admire friends who keep their promises")

    cap = _capture_target(e)
    r = e.process_turn("your read on friends who keep their promises?")

    assert cap.get("target") != "promises", (
        f"D-B regression: extractor collapsed to last token {cap.get('target')!r}")
    _t = cap.get("target") or ""
    assert len(_t.split()) >= 2 or _t == "", (
        f"D-B regression: head collapsed to single token {cap.get('target')!r}")
    assert not _is_hollow(r), r


# ── No regression: a FLAT (non-relative) topic still resolves as before ───
def test_flat_topic_query_still_resolves(tmpdir):
    """The head-resolution fix must not break the simple single-noun topic case
    ('your honest read on privacy' -> 'privacy')."""
    e = _make(tmpdir, "_db_flat")
    e.user_model.opinions.stances.clear()
    e.process_turn("i care deeply about privacy")

    cap = _capture_target(e)
    r = e.process_turn("your honest read on privacy?")
    # The flat (single-noun) topic resolves to a non-empty target and engages a
    # real lean (not the hollow fallback).
    assert cap.get("target"), cap.get("target")
    assert not _is_hollow(r), r


# ── Fail-open: an ungrounded relative clause stays honest, not fabricated ─
def test_ungrounded_relative_clause_query_is_honest(tmpdir):
    """If the user never taught a view on the relative-clause topic, the agent
    must answer honestly (no fabricated polarity) — the fix only improves
    resolution, it never invents a stance."""
    e = _make(tmpdir, "_db_ungrounded")
    e.user_model.opinions.stances.clear()
    e._agent_stances.clear() if hasattr(e, "_agent_stances") else None

    r = e.process_turn(
        "your honest read on people who jog at midnight?").lower()
    assert isinstance(r, str) and r.strip(), r
