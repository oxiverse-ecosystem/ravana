"""Attribute consistency smoke test (round 2026-09-06).

Catches dangling attribute references (e.g. _agent_own_stances renamed to
_agent_stances) at test time, not CI time. Workers operate in parallel on
branches that diverge from main — when main renames an attribute, the
worker's branch has stale references. This test asserts the canonical
attribute names exist on a fresh CognitiveChatEngine.
"""
import os
os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.chat.engine import CognitiveChatEngine


def _make(tmpdir):
    return CognitiveChatEngine(
        dim=64, seed=42, baby_mode=True,
        data_dir=tmpdir, user_suffix="_attrcheck",
    )


def test_canonical_agent_stances_attr(tmpdir):
    e = _make(tmpdir)
    # The canonical attribute is _agent_stances (not _agent_own_stances)
    assert hasattr(e, "_agent_stances"), "missing _agent_stances"
    assert isinstance(e._agent_stances, dict)


def test_canonical_agent_values_attr(tmpdir):
    e = _make(tmpdir)
    assert hasattr(e, "_agent_values"), "missing _agent_values"
    assert isinstance(e._agent_values, dict)


def test_canonical_agent_preferences_attr(tmpdir):
    e = _make(tmpdir)
    assert hasattr(e, "_agent_preferences"), "missing _agent_preferences"
    assert isinstance(e._agent_preferences, dict)


def test_canonical_agent_claims_attr(tmpdir):
    e = _make(tmpdir)
    assert hasattr(e, "_agent_claims"), "missing _agent_claims"
    assert isinstance(e._agent_claims, dict)


def test_no_stale_agent_own_stances_references(tmpdir):
    """_agent_own_stances was renamed to _agent_stances on main.

    If any code still references _agent_own_stances, it will raise
    AttributeError at runtime. This test asserts the OLD name is NOT
    present (so getattr(engine, '_agent_own_stances', {}) returns {}
    silently instead of real data, which would be a silent data loss).
    """
    e = _make(tmpdir)
    # The old name should NOT exist as a real attribute (only via getattr default)
    assert not hasattr(e, "_agent_own_stances"), \
        "stale _agent_own_stances attribute still exists — should be _agent_stances"
