"""D1 contraction stripping in topic extraction (round 2026-10-16T0756Z).

DEFECT: contractions kept as single tokens leaked into stance topics because
``_opinion_topic`` in ``user_model.py`` did not expand contractions before
tokenizing. "you're turning" stayed as the token ``you're`` — a single token
that was not in ``_OPINION_STOP`` (or was, but only as a manual entry,
meaning any contraction not explicitly listed would leak).

The fix applies ``expand_contractions`` before tokenization in
``_opinion_topic``, so "you're" → "you are" and both tokens are caught by
the existing ``_OPINION_STOP`` entries. This is morphological, not a lookup
table — handles any contraction the expander knows.

These tests verify:
  (a) contractions are expanded before tokenization in _opinion_topic
  (b) the expanded tokens are caught by _OPINION_STOP
  (c) no regression: real content words are still extracted
"""
import os
import re
import sys

os.environ["RAVANA_OFFLINE"] = "1"
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ, os.path.join(_PROJ, "ravana", "src"),
           os.path.join(_PROJ, "ravana_ml", "src")):
    sys.path.insert(0, _p)

from ravana.chat.user_model import UserModel


def _um():
    return UserModel()


# ── (a) contraction expansion in _opinion_topic ──────────────────────────

def test_youre_expanded_and_stripped():
    """'you're turning' should extract 'turning', not 'you're turning'."""
    um = _um()
    result = um._opinion_topic("you're turning")
    assert result is not None
    assert "you're" not in result
    assert "you" not in result
    assert "are" not in result
    assert result == "turning"


def test_im_expanded_and_stripped():
    """'i'm happy' should extract 'happy', not 'i'm happy'."""
    um = _um()
    result = um._opinion_topic("i'm happy")
    assert result is not None
    assert "i'm" not in result
    assert "i" not in result
    assert "am" not in result
    assert result == "happy"


def test_ive_expanded_and_stripped():
    """'i've grown' should extract 'grown', not 'i've grown'."""
    um = _um()
    result = um._opinion_topic("i've grown")
    assert result is not None
    assert "i've" not in result
    assert "i" not in result
    assert "have" not in result
    assert result == "grown"


def test_dont_expanded_and_stripped():
    """'don't know' should extract 'know', not 'don't know'."""
    um = _um()
    result = um._opinion_topic("don't know")
    assert result is not None
    assert "don't" not in result
    assert "do" not in result
    assert "not" not in result
    assert result == "know"


def test_cant_expanded_and_stripped():
    """'can't stand' should extract 'stand', not 'can't stand'."""
    um = _um()
    result = um._opinion_topic("can't stand")
    assert result is not None
    assert "can't" not in result
    assert "can" not in result
    assert "not" not in result
    assert result == "stand"


# ── (b) expanded tokens caught by _OPINION_STOP ──────────────────────────

def test_youre_not_in_stop_anymore_but_expanded():
    """'you're' is not in _OPINION_STOP as a literal, but its expansion
    ('you', 'are') IS — so the expanded form is caught."""
    um = _um()
    # The contraction itself is NOT in the stop list
    assert "you're" not in um._OPINION_STOP
    # But its expansion tokens ARE
    assert "you" in um._OPINION_STOP
    assert "are" in um._OPINION_STOP
    # And the topic extraction works correctly
    result = um._opinion_topic("you're turning")
    assert result == "turning"


def test_contractions_not_in_stop_list():
    """No contraction literals should be in _OPINION_STOP — they are handled
    by expansion, not by manual stop-list entries."""
    um = _um()
    contractions = ["you're", "we're", "they're", "i'm", "i've",
                    "don't", "can't", "won't", "isn't", "aren't"]
    for c in contractions:
        assert c not in um._OPINION_STOP, f"{c!r} should not be in _OPINION_STOP"


# ── (c) no regression: real content words still extracted ───────────────

def test_real_topic_unaffected():
    """'running' should still extract as 'running'."""
    um = _um()
    result = um._opinion_topic("running")
    assert result == "running"


def test_multiword_topic_unaffected():
    """'small talk' should still extract as 'small talk'."""
    um = _um()
    result = um._opinion_topic("small talk")
    assert result == "small talk"


def test_contraction_mid_phrase():
    """'the book you're reading' should extract 'book' (mid-phrase stop word
    breaks the loop — pre-existing limitation, not a D1 issue)."""
    um = _um()
    result = um._opinion_topic("the book you're reading")
    assert result is not None
    assert "you're" not in result
    assert "you" not in result
    assert "are" not in result
    # Should contain the real content word
    assert "book" in result


def test_contraction_at_end():
    """'running when you're ready' should extract 'running'."""
    um = _um()
    result = um._opinion_topic("running when you're ready")
    assert result is not None
    assert "you're" not in result
    assert "ready" not in result  # trailing modifier trimmed
    assert result == "running"
