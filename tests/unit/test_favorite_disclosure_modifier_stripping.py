"""Regression: favorite-disclosure keys drop leading/trailing modifier adverbs.

Feature t_ecfba7a1 wired slot_naming.strip_leading_modifiers /
strip_trailing_modifiers into the favorite-disclosure branch of
engine_reasoning.py so mined favorite keys agree with stance/opinion slot
naming (FIX-RV-12). This test pins the helper behavior the branch relies on.

Run:
    pytest tests/unit/test_favorite_disclosure_modifier_stripping.py -v
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.slot_naming import (
    strip_leading_modifiers,
    strip_trailing_modifiers,
)


def _fav_key(phrase):
    toks = phrase.strip(" .!?").split()
    toks = strip_trailing_modifiers(strip_leading_modifiers(toks))
    return " ".join(toks) if toks else phrase.strip(" .!?")


def test_leading_modifier_stripped_from_favorite():
    assert _fav_key("old books") == "books"


def test_trailing_modifier_stripped_from_favorite():
    assert _fav_key("books actually") == "books"


def test_both_sides_stripped():
    assert _fav_key("old books actually") == "books"


def test_single_token_survives():
    # never empties the list
    assert _fav_key("actually") == "actually"


def test_plain_favorite_unchanged():
    assert _fav_key("books") == "books"
