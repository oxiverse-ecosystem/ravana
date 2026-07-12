"""Tests for the three remaining-brain-failure fixes.

B1  Preamble fragment detector must not eat short answerable queries
    ("explain oxiverse", "define trust") — the hyper-cautious turn-end
    predictor false positive (mirror-image of Wernicke's anosognosia).
B2  Web-learned definitions must not leak non-linguistic code/script
    fragments ("Freestar.config.enabled_slots.push( ); ...") — the vmPFC/
    OFC source/reality-monitoring gap.
B3  `_is_word_salad(subject=None)` must not over-suppress a genuine
    definitional sentence — over-monitoring / false-alarm lesion.
"""
import pytest

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.constants import _is_word_salad
from ravana.chat.web_learning import WebLearningMixin


def _bare_engine():
    return CognitiveChatEngine.__new__(CognitiveChatEngine)


def _bare_web():
    """Bare WebLearningMixin with only the methods under test bound."""
    m = WebLearningMixin.__new__(WebLearningMixin)
    return m


# ── B1: preamble fragment must not eat answerable queries ──────────────────
def test_preamble_holds_fragment():
    eng = _bare_engine()
    assert eng._is_preamble_fragment("so") is True
    assert eng._is_preamble_fragment("by the way") is True


def test_preamble_does_not_eat_answerable_query():
    eng = _bare_engine()
    # Short imperative definition commands are complete speech acts.
    assert eng._is_preamble_fragment("explain oxiverse") is False
    assert eng._is_preamble_fragment("define trust") is False
    assert eng._is_preamble_fragment("what gravity") is False
    # Wh-questions are never fragments.
    assert eng._is_preamble_fragment("what is gravity") is False


def test_answerable_query_marker():
    eng = _bare_engine()
    assert eng._is_answerable_query("explain oxiverse") is True
    assert eng._is_answerable_query("define trust") is True
    assert eng._is_answerable_query("so") is False
    assert eng._is_answerable_query("by the way") is False


# ── B2: code/script fragments must be stripped / rejected ──────────────────
def test_looks_clean_rejects_code_fragment():
    wl = _bare_web()
    junk = "Freestar.config.enabled_slots.push( ); Trust psychology is the study of vulnerability."
    assert wl._definition_looks_clean(junk) is False
    # Pure code, no language.
    assert wl._definition_looks_clean("var x = function(){ return 1; }") is False
    # A clean definition is still clean.
    assert wl._definition_looks_clean(
        "Trust is the belief that others will not exploit your vulnerability.") is True


def test_strip_code_fragments_keeps_clean_part():
    wl = _bare_web()
    dirty = "Freestar.config.enabled_slots.push( ); Trust psychology is the study of how people decide to be vulnerable."
    clean = wl._strip_code_fragments(dirty)
    assert "Freestar" not in clean
    assert "push" not in clean
    assert "Trust psychology is" in clean
    # And the cleaned remainder is now considered clean.
    assert wl._definition_looks_clean(clean) is True


def test_strip_code_fragments_handles_html():
    wl = _bare_web()
    dirty = "<script>console.log('x')</script> Gravity is a fundamental force of attraction."
    clean = wl._strip_code_fragments(dirty)
    assert "<script>" not in clean
    assert "Gravity is a fundamental force" in clean


# ── B3: subject=None must not over-flag real definitions ───────────────────
def test_salad_subject_none_allows_definition():
    # A real definitional sentence must NOT be flagged as salad when no subject
    # is supplied (the over-monitoring / false-alarm lesion).
    real_def = ("The meaning of GRAVITY is the gravitational attraction between "
                "masses with mass.")
    assert _is_word_salad(real_def, subject=None) is False


def test_salad_subject_none_still_flags_garbage():
    # Genuinely degenerate text (no copula, no anchors) is still caught.
    assert _is_word_salad("pet pet pet semantic causal even great", subject=None) is True
