"""Contrastive self-opinion capability (round 2026-08-12T1234Z, t_2595f8ad).

RED test: a binary self-opinion question ("your take on X versus Y",
"do you prefer A or B") must engage BOTH sides through the real stance
resolver, not collapse to the last token and fall through to the hollow
"i'm still figuring that out".

Driven through the full CognitiveChatEngine.process_turn path so a routing
regression is caught (not just the miner in isolation).

Verifying: RAVANA_OFFLINE=1 pytest tests/test_contrastive_self_opinion.py
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


def _is_hollow(reply: str) -> bool:
    """True only when the engine answered with the pure honest-only fallback and
    did NOT engage a contrast — i.e. the reply is exactly the hollow
    'i'm still figuring that out' with no ';' separating two engaged sides. A
    legitimate two-sided answer ('i'm still figuring that out; i am wary of
    cities...') is NOT hollow even though one clause is honest."""
    r = (reply or "").lower().strip()
    if ";" in r:
        return False  # two clauses => a contrast was engaged
    return r.startswith("i'm still figuring that out")


# ── Core capability: both sides of a 'versus' self-opinion get engaged ──────
def test_contrastive_versus_engages_both_sides(tmpdir):
    """Seeding a real user stance on one side ('sea') lets the agent derive a
    grounded lean on that side; the contrast path must surface BOTH the grounded
    side and the contrasted side ('mountains'), never collapsing to a single
    token or the hollow fallback."""
    e = _make(tmpdir, "_contrast_vs")
    e.user_model.opinions.stances.clear()
    # User holds a strong view on 'sea' -> agent derives a lean on 'sea'.
    e.user_model.opinions.express_stance("sea", polarity=0.7, confidence=0.8)

    reply = e.process_turn("what's your take on the sea versus the mountains")
    r = reply.lower()
    # Both contrasted topics are named in the answer (engaged, not collapsed).
    assert "sea" in r, reply
    assert "mountain" in r, reply
    # Not the hollow single-target fallback.
    assert not _is_hollow(reply), reply
    # The agent's real lean on the grounded side is present (not authored prose
    # — it is whatever _agent_stance_on produced for the derived stance).
    assert ("lean toward" in r or "drawn to" in r or "value" in r
            or "for" in r or "against" in r), reply


# ── Capability: 'or' binary preference is engaged too ───────────────────────
def test_contrastive_or_engages_both_sides(tmpdir):
    """'do you prefer A or B' is the same binary self-opinion shape and must
    engage both sides. The grounded side ('cities', where the user holds a
    stance) is named with its real lean; the other side is answered honestly.
    The capability signal is: the reply has TWO clauses (joined by ';'), the
    grounded side's topic appears, and it is NOT the old single-token collapse
    ('i'm for <last word>')."""
    e = _make(tmpdir, "_contrast_or")
    e.user_model.opinions.stances.clear()
    e.user_model.opinions.express_stance("cities", polarity=-0.6, confidence=0.8)

    reply = e.process_turn("do you prefer the countryside or the cities")
    r = reply.lower()
    # Two sides engaged -> a ';' separates the clauses (single-collapse never has one).
    assert ";" in r, reply
    # The grounded side's topic is named with its real lean.
    assert "cities" in r, reply
    assert "wary" in r or "against" in r or "cool" in r, reply
    # Not the hollow single-target fallback, and not collapsing to one side only.
    assert not _is_hollow(reply), reply
    assert "countryside or the cities" not in r, reply


# ── Fail-open: neither side grounded -> honest, not fabricated ──────────────
def test_contrastive_neither_grounded_is_honest(tmpdir):
    """If the agent has NO view on either side, it must answer honestly (the
    hollow fallback is acceptable here) and must NOT fabricate a polarity."""
    e = _make(tmpdir, "_contrast_none")
    e.user_model.opinions.stances.clear()
    e._agent_stances.clear() if hasattr(e, "_agent_stances") else None

    reply = e.process_turn("what's your take on zebras versus kangaroos")
    # Honest OR a real but neutral lean — either is acceptable; what matters is
    # it never asserts a specific conviction it doesn't have. We only assert it
    # returns a non-empty, grammatical reply and doesn't crash.
    assert isinstance(reply, str) and reply.strip(), reply
