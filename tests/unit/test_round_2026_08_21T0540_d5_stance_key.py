"""Regression tests for feature t_46c07b5d — D5 residual limitation.

Round 2026-08-19T1628Z logged D5: a disclosure that names TWO activities in
one object span ("i adore cold water swimming jumping", "nothing beats cold
water swimming jumping", "i care for cold water swimming jumping") mined a
RUN-ON stance key ("cold water swimming jumping"). The resolver + reversal
miner could never bridge a later co-mention ("am i still into cold water
swimming?") to that key, so the stance was unrecallable.

Fix (user_model._opinion_topic): cut the opinion-object head at the FIRST
gerund token that is not the leading content word (a second activity), so the
key lands on the single salient activity. The leading token is always kept,
so a single-activity object ("swimming", "mountain climbing", "fossil
hunting") survives whole and continues to feed the does/event fact stores that
reuse this method.

All assertions read REAL store state / resolution output — no authored reply
strings, no per-topic table, no retraining. The first test below fails on the
pre-fix code (key would be "cold water swimming jumping") and passes after.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.user_model import UserModel
from ravana.chat.engine import CognitiveChatEngine


def _mine_stance_key(text):
    """Return the single stance key an i-love-style disclosure mines."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="d5_key_" + str(abs(hash(text)) % 100000))
    eng.user_model.opinions.stances.clear()
    eng.process_turn(text)
    return list(eng.user_model.opinions.stances.keys())


def test_multi_activity_disclosure_does_not_make_runon_key():
    # The core D5 case: two activities concatenated must NOT become one key.
    for text in (
        "i adore cold water swimming jumping",
        "i care for cold water swimming jumping",
        "nothing beats cold water swimming jumping",
        "i prefer cold water swimming jumping",
        "i love cold water swimming jumping in the lake",
    ):
        keys = _mine_stance_key(text)
        assert keys, f"no stance mined from {text!r}"
        assert keys == ["cold water swimming"], (
            f"{text!r} mined run-on key {keys!r}; expected ['cold water swimming']")


def test_single_activity_stance_key_survives_whole():
    # Single-activity objects must stay whole (regression guard: the cut must
    # not eat a legitimate compound activity).
    for text, expected in (
        ("i love mountain climbing the steep rocks", "mountain climbing"),
        ("i enjoy river kayaking the rapids", "river kayaking"),
        ("i love fossil hunting on the cliffs", "fossil hunting"),
        ("i love small talk at the village market", "small talk"),
        ("i love deep winter silence", "deep winter silence"),
    ):
        keys = _mine_stance_key(text)
        assert keys == [expected], (
            f"{text!r} -> {keys!r}; expected [{expected!r}]")


def test_resolved_stance_is_recallable_from_comention():
    # The whole point of the fix: a co-mention must resolve to the cleaned key.
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="d5_resolve")
    eng.process_turn("i adore cold water swimming jumping")
    assert list(eng.user_model.opinions.stances.keys()) == ["cold water swimming"]
    for q in (
        "am i still into cold water swimming?",
        "do you remember i loved cold water swimming?",
        "have i changed my mind about cold water swimming?",
    ):
        res = eng._match_stance(q)
        assert res is not None, f"co-mention {q!r} did not resolve to the stance"
        assert res[0] == "cold water swimming", (
            f"co-mention {q!r} resolved to wrong topic {res!r}")


def test_does_event_fact_path_unaffected():
    # The shared _opinion_topic chokepoint is also used by the activity/event
    # miners; single-activity gerund facts must remain intact (no over-cut).
    um = UserModel()
    for s, want_substr in (
        ("i build bicycle frames by hand", "build bicycle frames"),
        ("i restore vintage motorcycles", "restore vintage motorcycles"),
        ("i go cold water swimming every dawn", "go cold water swimming"),
    ):
        um.personal_facts.facts.clear()
        um.mine_personal_facts(s, run_correction=True)
        vals = [f.value for (a, b, c), f in um.personal_facts.facts.items()
                if (b.startswith("does") or b.startswith("event")) and not getattr(f, "superseded", False)]
        assert any(want_substr in v for v in vals), (
            f"{s!r} lost activity fact; got {vals!r}")
