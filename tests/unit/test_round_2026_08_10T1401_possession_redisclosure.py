"""
Feature regression tests for round 2026-08-10T1401Z: possession
re-disclosure in REVERSE order + owner re-attribution.

Round report limitation #1: a user re-discloses pets the OTHER way round
than the forward miner modelled, so the name was never stored and recall
returned stale/wrong data:

  * reverse-order naming: "the barn owl is mine and she's called wren"
    (THE <species> IS MINE [and (he|she|it)'s called <name>]) must FILE the
    name on the species slot the forward miner already uses.
  * owner re-attribution: "pip is my sister's cat" must MOVE the entity from
    the user (subject "i") to the named third-party owner, so recall keyed by
    the user no longer returns it (self/other boundary).

Both write through the SAME pet_slots resolver the forward miner + recall
sites use, so the stored key agrees by construction. Through the real engine,
recall must reflect the corrected ownership / name.
"""
import os
import sys

os.environ["RAVANA_OFFLINE"] = "1"
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ, os.path.join(_PROJ, "ravana", "src"), os.path.join(_PROJ, "ravana_ml", "src")):
    sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine


def _fresh_engine(suffix):
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


def test_reverse_order_naming_files_name_on_existing_slot():
    eng = _fresh_engine("possfix_a")
    # first disclosure stored the owl under the user via the forward miner
    eng.process_turn("i have an owl called wren")
    # reverse-order re-state with a CORRECTED name
    eng.process_turn("no, actually the owl is mine and she's called briar")
    facts = eng.user_model.personal_facts.facts
    # the stale name is superseded, the corrected one is active
    assert facts[("i", "owl", "briar")].superseded is False
    if ("i", "owl", "wren") in facts:
        assert facts[("i", "owl", "wren")].superseded is True
    recall = eng.process_turn("what is my owl's name?")
    assert "briar" in recall, recall
    assert "wren" not in recall, recall


def test_owner_reattribution_moves_entity_off_user():
    eng = _fresh_engine("possfix_b")
    eng.process_turn("my cat is called pip")
    eng.process_turn("actually pip is my sister's cat")
    facts = eng.user_model.personal_facts.facts
    # pip now lives under subject "sister", not "i"
    assert ("sister", "cat", "pip") in facts
    assert facts[("sister", "cat", "pip")].superseded is False
    # and the user's own cat slot for pip is retired (self/other boundary)
    if ("i", "cat", "pip") in facts:
        assert facts[("i", "cat", "pip")].superseded is True
    # Recall keyed by the USER must NOT claim pip is the user's cat. The
    # correction re-attributed pip to the sister, so an honest recall either
    # says pip now belongs to the sister, or reports no user-cat named pip.
    # It must NEVER say "your cat is pip" / "your cat's name is pip".
    recall = eng.process_turn("what is my cat's name?")
    assert "your cat is pip" not in recall, recall
    assert "your cat's name is pip" not in recall, recall
    # The boundary is enforced: any mention of pip must attribute it to the
    # sister, not to the user.
    if "pip" in recall:
        assert "sister" in recall, recall


def test_reverse_order_naming_stores_name_when_absent():
    eng = _fresh_engine("possfix_c")
    # forward miner does NOT create an owl slot for a bare possession
    eng.process_turn("i keep an owl in the loft")
    eng.process_turn("the owl is mine and she's called wren")
    facts = eng.user_model.personal_facts.facts
    assert ("i", "owl", "wren") in facts
    assert facts[("i", "owl", "wren")].superseded is False
    recall = eng.process_turn("what is my owl's name?")
    assert "wren" in recall, recall
