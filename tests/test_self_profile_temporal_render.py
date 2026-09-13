"""Regression test for self-profile rendering of temporal (since/since_age) facts.

Limitation addressed: the self-profile summary rendered raw ``since``/``since_age``
facts as ``"your since_age teach myself 14"`` instead of natural language. The fix
parses the stored ``<activity> <age_or_year>`` value into its components and renders
it naturally.

This test drives the REAL CognitiveChatEngine.process_turn path (RAVANA_OFFLINE=1,
fixed seed, isolated suffix). It asserts on REAL stored state and the rendered
reply grammar — never on a hardcoded reply string.

Verifying: RAVANA_OFFLINE=1 pytest tests/test_self_profile_temporal_render.py -v
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


def test_self_profile_renders_since_age_naturally(tmpdir):
    """A since_age fact must render as natural language, not 'your since_age is ...'."""
    e = _make(tmpdir, "_temporal1")
    e.process_turn("i picked up the harmonica when i was fourteen")
    # Use a query that reaches _try_memory_query's self-profile renderer
    # (not the _meta or _agg_early early-intercept)
    r = e.process_turn("what is something you remember about me")
    reply = (r or "").lower()
    # Must NOT contain raw attribute name
    assert "since_age" not in reply, f"raw attr leaked: {reply!r}"
    # Must contain natural-language rendering of the age
    assert "fourteen" in reply or "14" in reply, f"age missing: {reply!r}"
    assert "harmonica" in reply, f"activity missing: {reply!r}"
    assert "started" in reply or "when you were" in reply, f"not rendered naturally: {reply!r}"


def test_self_profile_renders_since_year_naturally(tmpdir):
    """A since fact must render as natural language, not 'your since is ...'."""
    e = _make(tmpdir, "_temporal2")
    e.process_turn("i have been building telescopes since 2019")
    r = e.process_turn("what is something you remember about me")
    reply = (r or "").lower()
    # Must contain natural-language rendering
    assert "2019" in reply, f"year missing: {reply!r}"
    assert "telescopes" in reply, f"activity missing: {reply!r}"
    assert "started" in reply or "since" in reply, f"not rendered naturally: {reply!r}"
    # Must NOT render as raw "your since is"
    assert "your since is" not in reply, f"raw attr leaked: {reply!r}"


def test_self_profile_since_age_fallback_for_unparseable(tmpdir):
    """A since_age fact whose anchor is not a valid age (e.g. 999) should
    gracefully fall back to 'your since_age is <val>' — never crash."""
    e = _make(tmpdir, "_temporal3")
    # Force-store a fact with an unparseable anchor
    e.user_model.personal_facts.assert_fact("i", "since_age", "some activity 999",
                                            confidence=0.6)
    r = e.process_turn("what is something you remember about me")
    reply = (r or "").lower()
    assert "some activity" in reply, f"value missing: {reply!r}"
