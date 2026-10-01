"""FIX-RV-18 — the relation miner must not seat a SUPPORT/AUXILIARY verb (or an
article) in the relation-head slot, and must NOT drop the clause's real content
when it does.

DEFECT (cold-verified in-process, RAVANA_OFFLINE=1, clean suffix):

    process_turn("when i was a teenager i lived in mumbai")

  FACTS MINED BEFORE:
      ('i', 'does:a', 'a teenager')      <-- GARBAGE: 'a' is an ARTICLE, not a head
      (no 'mumbai' anywhere — the disclosed location was DROPPED)

The open-class verb miner took the token after the auxiliary "was" as the
predicate head — "a" — and the match's object span then swallowed the rest of
the sentence, so the real predicate ("lived") and the real disclosed content
("mumbai") never reached the store.

These tests are RED-CAPABLE: on the pre-fix baseline they fail, because
('i', 'does:a', ...) is stored and no mumbai fact exists.
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO_ROOT, "ravana_ml", "src"),
           os.path.join(_REPO_ROOT, "ravana", "src"),
           os.path.join(_REPO_ROOT, "ravana-v2", "src"),
           _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("RAVANA_OFFLINE", "1")


def _engine(tmp_path, suffix):
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)
    return eng


def _facts(eng):
    return {(k[0], k[1]): v.value
            for k, v in eng.user_model.personal_facts.facts.items()}


@pytest.fixture(scope="module")
def disclosed():
    """One cold engine, three canonical disclosures, the RV-18 sentence third."""
    eng = _engine(None, "rv18_aux_head")
    eng.process_turn("my cat is diagnosed with a chronic illness")
    eng.process_turn("my dog had surgery last month")
    eng.process_turn("when i was a teenager i lived in mumbai")
    yield eng
    eng.stop_background_learning()


def test_disclosed_location_is_stored(disclosed):
    """THE assertion: the real disclosed content EXISTS in the store.

    Deliberately asserts presence of 'mumbai', not merely absence of the
    garbage key — a fix that only suppressed ('i', 'does:a', ...) while still
    dropping the place would pass a negative-only test.
    """
    facts = _facts(disclosed)
    values = " | ".join(f"{a}={v}" for (_s, a), v in facts.items())
    assert any("mumbai" in v.lower() for v in facts.values()), (
        "the disclosed location was dropped by the miner; store holds: " + values)


def test_auxiliary_is_not_a_relation_head(disclosed):
    """No stored attribute may be a bare function word ('does:a', 'does:the')."""
    from ravana.chat.pet_slots import base_species, is_function_word
    for (_s, attr), _v in _facts(disclosed).items():
        head = attr.split(":", 1)[-1] if attr.startswith(("does:", "event:")) else attr
        head = base_species(head).split()[0] if head.split() else head
        assert not is_function_word(head), (
            f"function word {head!r} was seated as a relation head (attr={attr!r})")


def test_support_verb_class_is_not_a_frozen_table():
    """The closed-class vocabulary must be runtime-EXTENSIBLE, not a table.

    A frozen table would fail this: an unknown function word would have to be
    added to source. This is what distinguishes a seed from hardcoding.
    """
    from ravana.chat import pet_slots
    novel = "zzyzx"          # not in any seed
    assert not pet_slots.is_function_word(novel)
    pet_slots.learn_function_word(novel)
    assert pet_slots.is_function_word(novel)
    novel_v = "flibbertigibbet"
    assert not pet_slots.is_residence_verb(novel_v)
    pet_slots.learn_residence_verb(novel_v)
    assert pet_slots.is_residence_verb(novel_v)


def test_verb_gate_accepts_real_verbs_and_rejects_support_verbs():
    """The shared gate is the contract: real predicates in, support words out.

    A pure-vocabulary assertion, so it is reliable locally (no GloVe routing).
    The round's real-predicate disclosures ('keep'/'hike'/'grow' as bare
    "i <verb> <obj>" turns) mine zero facts at BASELINE too under a cold GloVe
    cache, so asserting their end-to-end mining here would encode a
    pre-existing routing gap as a FIX-RV-18 regression.
    """
    from ravana.chat.user_model import _activity_verb_ok
    from ravana.chat.pet_slots import is_function_word
    for real in ("keep", "hike", "grow", "lived", "kept", "teach", "build"):
        assert _activity_verb_ok(real), f"{real!r} must stay an acceptable head"
        assert not is_function_word(real)
    # Support verbs / copulas / articles / prepositions may NEVER be a head.
    for support in ("does", "do", "did", "was", "is", "been", "has", "had",
                    "will", "would", "can", "a", "an", "the", "in", "and"):
        assert not _activity_verb_ok(support), (
            f"support/function word {support!r} must not be seated as a head")
        assert is_function_word(support)


def test_auxiliary_led_clauses_keep_their_place_across_verb_stems():
    """Generalisation: the fix is not tuned to the mumbai sentence.

    Each case is an auxiliary/support-word-led temporal clause whose real
    predicate is a DIFFERENT inflected residence verb. Baseline stored
    ('i', 'does:a', ...) for the first two and dropped the place entirely;
    the location miner already handled the third (but double-stored it as
    ('i', 'does:at', 'at university') alongside the place).

    A/B verified against the pre-fix tree: 'i live in berlin' mines nothing on
    BOTH sides (a separate pre-existing gap, deliberately NOT claimed here).
    """
    cases = [
        ("when i was a teenager i lived in mumbai", "mumbai"),
        ("when i was a kid i lived in pune", "pune"),
        ("when i was at university i stayed in barcelona", "barcelona"),
    ]
    for n, (utterance, place) in enumerate(cases):
        eng = _engine(None, "rv18_var%d" % n)
        try:
            eng.process_turn(utterance)
            facts = _facts(eng)
            blob = " | ".join(f"{a}={v}" for (_s, a), v in facts.items())
            assert any(place in v.lower() for v in facts.values()), (
                f"{utterance!r} lost {place!r}; store holds: {blob}")
            # and exactly one slot, not a duplicate verb-phrase half-fact
            assert sum(1 for v in facts.values() if place in v.lower()) == 1, blob
        finally:
            eng.stop_background_learning()


def test_place_disclosure_is_not_double_stored(disclosed):
    """One disclosure, one slot: the residence path owns a locative verb.

    Guards against fixing the loss by storing the place TWICE (once as
    'location', once as an activity verb-phrase half-fact).
    """
    facts = _facts(disclosed)
    hits = [(a, v) for (s, a), v in facts.items()
            if s == "i" and "mumbai" in v.lower()]
    assert len(hits) == 1, f"place stored {len(hits)}x: {hits}"
