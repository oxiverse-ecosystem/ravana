"""Regression tests for the LOW-severity cleanups L1/L3.

L1: sub_answer_synthesizer._pronominalize only split on ". " so subjects
    repeated after "!", "?" or newline joins were never pronominalized.
L3: RegisterController.compose prepended "i'm not certain, but" to QUESTIONS
    ("do birds fly?" -> "i'm not certain, but do birds fly?"), nonsensical.

Run from repo root:
    python -m pytest tests/unit/test_low_cleanup.py -v
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.core.sub_answer_synthesizer import SubAnswerSynthesizer
from ravana.language.register import RegisterController


def _synthesizer():
    return SubAnswerSynthesizer()


# ── L1: pronominalize handles "!"/"?"/newline sentence joins ─────────────────
def test_pronominalize_handles_non_period_joins():
    syn = _synthesizer()
    # Two sentences joined by ". " — baseline still works.
    out = syn._pronominalize("Gravity is a force. Gravity bends light.", "gravity")
    assert "it" in out, out
    # Joined by "? " — must still pronominalize the second occurrence.
    out2 = syn._pronominalize("Gravity is a force? Gravity bends light.", "gravity")
    assert "it" in out2, out2
    # Joined by "! " — must still pronominalize.
    out3 = syn._pronominalize("Gravity is a force! Gravity bends light.", "gravity")
    assert "it" in out3, out3


def test_pronominalize_leaves_single_occurrence():
    syn = _synthesizer()
    out = syn._pronominalize("Gravity is a force that bends light.", "gravity")
    assert out == "Gravity is a force that bends light.", out


# ── L3: hedge is NOT prepended to questions ─────────────────────────────────
def test_hedge_not_prepended_to_question():
    rc = RegisterController("casual")
    rc.knobs["certainty"] = 0.10  # force low certainty (hedge trigger)
    q = "do birds fly?"
    out = rc.compose(q, 0.5, multi_sentence=False)
    assert not out.lower().startswith("i'm not certain, but"), out
    assert out == q, out  # question passes through untouched


def test_hedge_prepended_to_statement():
    rc = RegisterController("casual")
    rc.knobs["certainty"] = 0.10
    s = "the experiment failed."
    out = rc.compose(s, 0.5, multi_sentence=False)
    assert out.lower().startswith("i'm not certain, but"), out


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
