"""Regression tests for M1: register.compose verbosity truncation.

`RegisterController.compose` truncates to the first sentence when
`verbosity < 0.20 and not multi_sentence`. The engine only passed
`multi_sentence=True` for `decomposed_*` strategies, so Situation-Model
narrative/syntax multi-sentence outputs were silently collapsed to one
sentence once verbosity decayed below 0.20 after a few friendly turns.
The engine now also passes multi_sentence for situation_model_narrative /
situation_model_syntax; this test locks the guard logic that makes that safe.

Run from repo root:
    python -m pytest tests/unit/test_register_truncation.py -v
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.language.register import RegisterController


def _low_verbosity_controller():
    rc = RegisterController("casual")
    rc.knobs["verbosity"] = 0.10  # simulate decayed verbosity after friendly turns
    return rc


_MULTI = "ravana is a cognitive architecture. it learns concepts from the web using Hebbian learning."


def test_low_verbosity_truncates_single_sentence_when_not_multi():
    rc = _low_verbosity_controller()
    out = rc.compose(_MULTI, 0.5, multi_sentence=False)
    # Should keep only the first sentence under low verbosity.
    assert out == "ravana is a cognitive architecture.", out


def test_low_verbosity_preserves_multi_sentence_when_flagged():
    rc = _low_verbosity_controller()
    out = rc.compose(_MULTI, 0.5, multi_sentence=True)
    # The multi_sentence guard must prevent the truncation.
    assert out == _MULTI, out


def test_high_verbosity_preserves_multi_sentence():
    rc = RegisterController("casual")
    rc.knobs["verbosity"] = 0.8
    out = rc.compose(_MULTI, 0.5, multi_sentence=False)
    assert out == _MULTI, out


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
