"""FIX-RV-12 regression: modifier-heavy stance topic keys.

Defect (reproduced at the pre-fix commit c77baeee): the stance miner keyed
opinions on the raw phrase, so leading and trailing modifiers leaked into the
slot:

    "i love cooking earlier in the morning" -> key "cooking earlier"
    "i love dear old jazz clubs"           -> key "dear old jazz clubs"

while a later query resolves the same concept to "cooking" / "jazz clubs" —
so the stance was stored under a key no query could ever reach, and the
recall ("do you think i love cooking?") abstained.

Fix: one shared slot-naming chokepoint (`ravana.chat.slot_naming`) trims
leading and trailing modifiers through the DATA-file seed vocabulary in
`data/functional_lexicon.json` (leading_modifiers / trailing_modifiers), so
the miner and recall name the same slot. No per-topic table, no authored
reply, no retraining.

Run:
    pytest tests/unit/test_slot_naming_fv12.py -v
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.slot_naming import (
    leading_modifiers,
    strip_leading_modifiers,
    strip_trailing_modifiers,
    trailing_modifiers,
)
from ravana.chat.user_model import UserModel


def _topic(phrase):
    return UserModel()._opinion_topic(phrase)


# ── (1) the card's cold cases ────────────────────────────────────────────
def test_trailing_temporal_modifier_stripped():
    """"cooking earlier in the morning" must key on "cooking"."""
    assert _topic("cooking earlier in the morning") == "cooking"


def test_copula_residue_not_left_in_key():
    """"jazz music is relaxing" must key on "jazz music", not "jazz music is"."""
    assert _topic("jazz music is relaxing") == "jazz music"


def test_leading_modifiers_stripped():
    """"dear old jazz clubs" must key on "jazz clubs"."""
    assert _topic("dear old jazz clubs") == "jazz clubs"


# ── (2) regressions: the trim must not eat a real content head ───────────
def test_open_source_software_unchanged():
    assert _topic("open source software") == "open source software"


def test_letterpress_printing_unchanged():
    assert _topic("letterpress printing") == "letterpress printing"


def test_multiword_head_survives():
    assert _topic("cold water swimming") == "cold water swimming"


def test_single_modifier_phrase_not_emptied():
    """A phrase that IS one modifier keeps its token rather than vanishing."""
    assert _topic("earlier") == "earlier"


def test_prepositional_cut_still_applies():
    assert _topic("the solitude of the lighthouse") == "solitude"
    assert _topic("petrichor after a storm") == "petrichor"


def test_relative_clause_bridge_still_applies():
    assert _topic("people who talk in theatres") == "people who talk"


# ── (3) the trim itself: edge-trim only, never empties ───────────────────
def test_strip_trailing_only_touches_the_end():
    assert strip_trailing_modifiers(["cooking", "earlier"]) == ["cooking"]
    # an interior modifier is NOT a modifier — it is content in this position
    assert strip_trailing_modifiers(["earlier", "cooking"]) == ["earlier", "cooking"]


def test_strip_leading_only_touches_the_front():
    assert strip_leading_modifiers(["dear", "old", "jazz"]) == ["jazz"]
    # never empty the list
    assert strip_leading_modifiers(["dear"]) == ["dear"]


def test_modifier_vocabulary_is_data_driven():
    """The vocabulary comes from the functional-lexicon fit file, not from a
    per-topic table baked into the extraction code."""
    assert "earlier" in trailing_modifiers()
    assert "dear" in leading_modifiers()
    assert "old" in leading_modifiers()


# ── (4) end-to-end: mine, then recall the same slot ──────────────────────
def test_mined_stance_is_recallable_under_its_clean_topic():
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_slot_naming_fv12")
    eng.user_model.mine_personal_facts("i love cooking earlier in the morning")
    keys = list(eng.user_model.opinions.stances.keys())
    assert "cooking" in keys, f"expected clean key 'cooking', got {keys}"
    assert "cooking earlier" not in keys, "modifier leaked into the stance key"
    # the follow-up must now REACH the stored stance
    reply = eng._user_stance_reply("do you think i love cooking?")
    assert reply is not None, "stored stance became unrecallable"
    assert "forming a view" not in str(reply).lower(), \
        "recall abstained despite a held stance"
