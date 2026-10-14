"""
Regression tests for D1: topic extraction grabs wrong word.

The positional heuristic (last content word) fails for compound nouns and
modifier+noun phrases:
  "migration patterns" -> "patterns" (should be "migration")
  "its undervalued" -> "undervalued" (adjective, no noun head)

The fix uses GloVe embeddings to find the semantic head noun — the word
most central to the phrase (highest average cosine similarity to others).
"""
import os
import sys

os.environ["RAVANA_OFFLINE"] = "1"
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ, os.path.join(_PROJ, "ravana", "src"),
           os.path.join(_PROJ, "ravana_ml", "src")):
    sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine


def _eng(suffix):
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


# -- Case 1: compound noun "migration patterns" -> "migration" not "patterns" --
def test_compound_noun_migration_patterns():
    eng = _eng("test_semantic_migration")
    # The topic extraction should identify "migration" as the head, not "patterns"
    subj, _ = eng._extract_topic("what do you think about migration patterns", [])
    # "patterns" is the modifier, "migration" is the head noun
    assert "migration" in subj, f"head noun 'migration' missing: {subj!r}"
    assert subj != "patterns", f"got modifier 'patterns' instead of head: {subj!r}"


# -- Case 2: "its undervalued" -> single-word adjective, not a compound --
def test_undervalued_single_word():
    eng = _eng("test_semantic_undervalued")
    # "its undervalued" strips to just "undervalued" (single word adjective)
    # This is NOT a compound noun case — _semantic_head_noun returns None for single words
    subj, _ = eng._extract_topic("its undervalued", [])
    # The semantic head noun detection should NOT fire (single word)
    assert subj == "undervalued", f"single word 'undervalued' should pass through: {subj!r}"
    # Verify _semantic_head_noun returns None for single words
    assert eng._semantic_head_noun(["undervalued"]) is None


# -- Case 3: existing working case must not regress --
def test_simple_topic_still_works():
    eng = _eng("test_semantic_simple")
    subj, _ = eng._extract_topic("what is trust", [])
    assert "trust" in subj, f"simple topic 'trust' missing: {subj!r}"


# -- Case 4: "cooking earlier" -> "cooking" (trailing modifier stripped) --
def test_trailing_modifier_stripped():
    eng = _eng("test_semantic_cooking")
    subj, _ = eng._extract_topic("i think cooking earlier is better", [])
    assert "cooking" in subj, f"head noun 'cooking' missing: {subj!r}"
    assert "earlier" not in subj, f"trailing modifier 'earlier' leaked: {subj!r}"


# -- Case 5: _semantic_head_noun returns None for single word --
def test_semantic_head_single_word():
    eng = _eng("test_semantic_single")
    result = eng._semantic_head_noun(["trust"])
    assert result is None, f"single word should return None, got {result!r}"


# -- Case 6: _semantic_head_noun returns None for empty list --
def test_semantic_head_empty():
    eng = _eng("test_semantic_empty")
    result = eng._semantic_head_noun([])
    assert result is None, f"empty list should return None, got {result!r}"


# -- Case 7: _semantic_head_noun finds head in "climate change" --
def test_semantic_head_climate_change():
    eng = _eng("test_semantic_climate")
    result = eng._semantic_head_noun(["climate", "change"])
    # "climate" is the head noun (change modifies it)
    assert result is not None, "should find a semantic head"
    assert result == "climate", f"expected 'climate', got {result!r}"