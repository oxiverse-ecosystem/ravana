"""Regression tests for the yes/no -> encyclopedic query-variant steering (M4 deeper half).

_is_yesno_factual_query routes yes/no questions to the web; but
_rewrite_query_for_web previously only rewrote wh- queries
("what is X" -> "X definition"), so yes/no questions like "is pluto a
planet?" fell through to the raw "is X a Y?" string, which the search
backend ranks poorly (entity-collision / junk). This recasts them as
definition-seeking queries ("what is pluto planet") so encyclopedic
pages surface.

Run from repo root:
    python -m pytest tests/unit/test_yesno_query_steering.py -v
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine


def _eng():
    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    eng._is_yesno_factual_query = CognitiveChatEngine._is_yesno_factual_query.__get__(eng)
    eng._is_conditional_query = CognitiveChatEngine._is_conditional_query.__get__(eng)
    return eng


def test_yesno_rewritten_to_definition_seeking():
    eng = _eng()
    # "is pluto a planet?" -> should become an encyclopedic "what is ..." query
    out = eng._rewrite_query_for_web("is pluto a planet?", "pluto planet")
    assert out.startswith("what is "), out
    # "can dogs eat chocolate?" -> encyclopedic recast
    out2 = eng._rewrite_query_for_web("can dogs eat chocolate?", "dogs chocolate")
    assert out2.startswith("what is "), out2


def test_wh_query_unchanged_behavior():
    eng = _eng()
    # wh- queries keep their existing rewrite behavior (definition/meaning).
    out = eng._rewrite_query_for_web("what is trust", "trust")
    assert "definition" in out or "meaning" in out, out


def test_non_factual_untouched():
    eng = _eng()
    # A chitchat / non-factual query is returned as-is (no false yes/no match).
    out = eng._rewrite_query_for_web("tell me a joke", "joke")
    assert out == "tell me a joke", out


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
