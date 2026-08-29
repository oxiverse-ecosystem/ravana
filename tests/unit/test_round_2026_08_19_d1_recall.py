"""Regression test for round 2026-08-19T1628Z D1: cross-entity fact recall
contamination.

Root cause (verified): the fact dedup in `_collect_user_model_state` kept only
the LONGEST value per attribute, collapsing all `does`/`event` predicate facts
into one slot. A query like "what does my brother do" then matched the single
surviving `does` fact (ren's birthday) and returned ANOTHER PERSON's fact. The
matcher (`_match_fact`) also bridged on the generic word "does".

Fix: keep every `does`/`event` predicate fact as its own entry, and match on the
VALUE + ENTITY-ATTRIBUTE tokens (not the attribute name "does"). Verifies a
targeted-entity query returns the correct entity's fact and never a sibling's.
"""
import os
os.environ.setdefault("RAVANA_OFFLINE", "1")

import sys
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (_PROJ, f"{_PROJ}/ravana_ml/src", f"{_PROJ}/ravana/src", f"{_PROJ}/ravana-v2/src"):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine


def _eng(suffix):
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


def test_d1_brother_not_contaminated_by_ren():
    """'what did my brother do for work' must return the brother fact, not ren's."""
    eng = _eng("t_d1_brother_ren")
    eng.process_turn("my brother tomas restores vintage motorcycles in a shed")
    eng.process_turn("i found ren's birthday would've been this week")
    eng.process_turn("my sister priya is a marine biologist who studies eels")
    r = eng.process_turn("what did i say my brother does for work?")
    assert "tomas" in r.lower(), f"expected brother tomas, got: {r!r}"
    assert "restores vintage motorcycles" in r.lower(), f"expected the brother's activity, got: {r!r}"
    # The critical regression: ren's birthday must NOT leak into a brother query.
    assert "ren" not in r.lower() or "birthday" not in r.lower(), \
        f"cross-entity contamination: brother query returned ren fact: {r!r}"


def test_d1_sister_not_contaminated_by_brother():
    eng = _eng("t_d1_sister_bro")
    eng.process_turn("my brother tomas restores vintage motorcycles")
    eng.process_turn("my sister priya is a marine biologist who studies eels")
    r = eng.process_turn("what did i say my sister does?")
    assert "priya" in r.lower(), f"expected sister priya, got: {r!r}"
    assert "marine biologist" in r.lower(), f"expected priya's activity, got: {r!r}"
    assert "tomas" not in r.lower(), f"cross-entity contamination: sister query returned brother: {r!r}"


def test_d1_does_facts_all_retrievable():
    """All `does` predicate facts must be individually addressable."""
    eng = _eng("t_d1_does")
    eng.process_turn("i got promoted to head of compost")
    eng.process_turn("i built a tiny garden on my roof")
    eng.process_turn("i found ren's birthday would've been this week")
    facts, _, _ = eng._collect_user_model_state()
    does_vals = [v for a, v, c in facts if a.startswith("does")]
    # three distinct predicate facts, not collapsed to one longest value
    assert len(does_vals) == 3, f"expected 3 distinct 'does' facts, got {len(does_vals)}: {does_vals}"
