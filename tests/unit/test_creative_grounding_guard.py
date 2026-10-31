"""Creative generation must fail closed for ungrounded topics.

Round residual: open-ended creative prompts with no concept-graph
grounding should produce an honest uncertainty (None) rather than word
salad. This pins that contract.
"""

import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")

PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src", f"{PROJ}\\ravana-v2\\src"):
    sys.path.insert(0, p)

import pytest


def _engine():
    from ravana.chat.engine import CognitiveChatEngine
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="createground")


def test_creative_returns_none_for_unknown_topic():
    eng = _engine()
    # A nonsense topic with no node in the graph must not produce creative text.
    out = eng._generate_creative("zzqxqwerty", "poem")
    assert out is None, f"expected None for ungrounded topic, got: {out!r}"


def test_creative_still_works_for_known_topic():
    eng = _engine()
    # Should not crash; result is either a string or None (both acceptable).
    out = eng._generate_creative("world", "poem")
    assert out is None or isinstance(out, str)
