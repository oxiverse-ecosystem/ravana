"""Regression tests for M5: identity-query detection (user name recall).

The old detector was an exact-match allowlist plus two endswith() checks, so
natural variants ("do you remember my name?", "can you recall my name?") fell
through to a generic reflective fallback instead of recalling the stored name.
This locks the intent-based detection.

Run from repo root:
    python -m pytest tests/unit/test_identity_query.py -v
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine


def _build_engine():
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               data_dir="/tmp/ravana_m5_test")


# ── 1. identity questions (incl. the report's exact failure case) are caught ─
def test_identity_questions_detected():
    eng = _build_engine()
    eng.user_model.user_name = "Likhith"
    for q in ["do you remember my name?", "can you recall my name?",
              "what is my name?", "what's my name?", "who am i?",
              "do you know my name?", "do you remember my name now?"]:
        # A full turn must be intercepted as user_identity, NOT fall through to
        # reflective_uncertainty / decomposition.
        eng.process_turn(q)
        assert getattr(eng, "_last_strategy", "") == "user_identity", f"{q} -> {eng._last_strategy}"


# ── 2. statements that TELL the name are NOT miscaught as identity questions
def test_name_statement_not_miscaught():
    eng = _build_engine()
    # A statement "my name is X" should be stored, not answered as a question.
    eng.process_turn("my name is Pixel")
    # The user_model should now know the name.
    assert eng.user_model.user_name.lower() == "pixel", "name not stored"


# ── 3. after storing a name, recall works for the report's exact case ────────
def test_recall_after_store():
    eng = _build_engine()
    eng.process_turn("my name is Likhith")
    out = eng.process_turn("do you remember my name?")
    assert "likhith" in out.lower(), f"recall failed: {out}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
