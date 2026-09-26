"""FIX-RV-15 (defect D) — the realizer renders the stance that FORMED.

The card's third visible symptom: "i think remote work is better than office
work" formed real stances, and the reply was still
"nice, noted. what made you think of that?" — a topicless acknowledgment from
the degenerate-fallback family. The state existed and was never read.

Root cause: `ResponseGenMixin._handle_assertion` gated topic reflection on
`has_clean_topic(subject, <original utterance>)`, which vetoes any utterance
carrying comparative/opinion markers. That gate is CORRECT for a clause subject
("believe nuclear energy" must never render as "you're believe nuclear energy")
and WRONG for a stance key, which is a clean noun phrase the miner resolved and
the reporting frame has already been stripped from.

The fix reads the miner's per-turn record (`UserStanceStore.last_mined`) instead
of re-deriving the topic. These tests assert the RENDER is driven by that
record, not by an authored branch, which is why they check the invariant
("the reply's content comes from what the store holds") rather than matching a
fixed sentence.
"""
import os
import sys

import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
for _p in (_PROJ, os.path.join(_PROJ, "ravana", "src"),
           os.path.join(_PROJ, "ravana_ml", "src"),
           os.path.join(_PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.chat.user_model import UserModel  # noqa: E402
from ravana.chat.personal_fact_store import UserStanceStore  # noqa: E402
from ravana.chat.realizer_lexicon import has_clean_topic  # noqa: E402


@pytest.fixture()
def um():
    m = UserModel.__new__(UserModel)
    m.opinions = UserStanceStore()
    m.turn_num = 0
    m.emotional_state = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
    m.interaction_history = []
    m.beliefs = {}
    m.personal_facts = None
    return m


# ── the miner's per-turn record is what the realizer reads ───────────────

def test_mining_records_the_stance_for_the_realizer(um):
    """A mined stance is recorded on the store, deduped by topic.

    The realizer can only render what the miner reports, so this record is the
    whole contract between the two. Dedup matters: the defect was ONE
    proposition producing THREE keys, and a realizer that rendered per-key would
    triple the reply.
    """
    um.mine_personal_facts("i think remote work is better than office work")
    mined = list(um.opinions.last_mined)
    assert [t for t, _ in mined] == ["remote work", "office work"], mined
    # The recorded polarity is the polarity actually STORED, so what is
    # rendered and what is believed cannot diverge.
    for topic, pol in mined:
        assert pol == um.opinions.stances[topic].polarity


def test_record_clears_between_turns(um):
    """`last_mined` is per-turn, exactly like `last_reversal`.

    Without the reset a stance mined on turn N would be re-rendered on turn
    N+1, which is the "nice, noted" class of bug in a new costume: the engine
    acknowledging an attitude the user is not currently expressing.
    """
    um.mine_personal_facts("i think remote work is better than office work")
    assert um.opinions.last_mined
    um.opinions.clear_last_mined()
    assert um.opinions.last_mined == []


def test_no_mined_stance_means_no_record(um):
    """A turn that mines no attitude leaves the record empty.

    This is what keeps the realizer's change strictly additive: the pre-existing
    acknowledgment path is reached unchanged whenever the miner had nothing to
    say, so no unrelated reply shape can shift.
    """
    um.mine_personal_facts("the weather is grey this afternoon")
    assert um.opinions.last_mined == []


# ── the render itself ─────────────────────────────────────────────────────

def _render(engine, mined):
    """Drive the realizer over a store primed with `mined`.

    Exercises `_handle_assertion` directly rather than a full `process_turn`,
    so the assertion is a pure function of (utterance, subject, store) and the
    test cannot be perturbed by which router wins the turn.
    """
    engine.user_model.opinions.last_mined = list(mined)
    return engine._handle_assertion(
        "i think remote work is better than office work", "remote work")


@pytest.fixture(scope="module")
def engine():
    """A minimal object exposing what `_handle_assertion` reads.

    Built with `__new__` so this stays a unit test: the method touches
    `pfc_workspace`, `belief_store`, `turn_count`, `user_model` and
    `_polarity_word`, all supplied here directly.
    """
    from ravana.chat.response_gen import ResponseGenMixin

    class _PFC:
        @staticmethod
        def classify_speech_act(_text):
            return "statement"

    e = ResponseGenMixin.__new__(ResponseGenMixin)
    e.pfc_workspace = _PFC()
    e.belief_store = None
    e.turn_count = 1
    e.user_model = UserModel.__new__(UserModel)
    e.user_model.opinions = UserStanceStore()
    e.user_model.interaction_history = []
    e.user_model.personal_facts = None
    e.user_model.beliefs = {}
    e.user_model.emotional_state = {
        "valence": 0.0, "arousal": 0.0, "dominance": 0.0}
    e._polarity_word = _polarity_word
    e._detect_emotional_disclosure = lambda **_kw: None
    return e


def _polarity_word(pol):
    """Mirror of `CognitiveChatEngine._polarity_word`.

    Copied rather than imported so the unit test needs no engine boot. These
    are single vocabulary tokens, the remediation shape the no-hardcoding
    doctrine prescribes: the content is the measured polarity, not prose.
    """
    if pol >= 0.6:
        return "strongly for"
    if pol > 0.1:
        return "for"
    if pol <= -0.6:
        return "strongly against"
    if pol < -0.1:
        return "against"
    return "uncertain about"


def test_render_states_both_sides_of_the_comparison(engine):
    """The reply carries BOTH sides, so the comparison survives the round trip.

    Rendering only the winner would silently drop the half of the proposition
    the storage fix worked to preserve — the same information loss, one layer
    up.
    """
    out = _render(engine, [("remote work", 0.7), ("office work", -0.7)])
    assert "remote work" in out, out
    assert "office work" in out, out


def test_render_reports_the_lean_direction(engine):
    """The two sides must NOT read identically.

    A reply that named both topics without sign would be indistinguishable from
    the shredded original. Assert the polarity token, not a fixed sentence.
    """
    out = _render(engine, [("remote work", 0.7), ("office work", -0.7)])
    for word in _polarity_word(0.7).split():
        assert word in out, out
    for word in _polarity_word(-0.7).split():
        assert word in out, out


def test_render_is_driven_by_the_record_not_the_utterance(engine):
    """Change ONLY the record and the reply must change with it.

    This is the anti-hardcoding assertion. If any part of the output came from
    a branch keyed to the probe sentence, swapping the topics would leave the
    reply untouched. Rotating the record to a completely different comparison
    must change every topic in the output.
    """
    out = _render(engine, [("sailing", 0.8), ("flying", -0.8)])
    assert "sailing" in out and "flying" in out, out
    assert "remote work" not in out, out
    assert "office work" not in out, out


def test_render_skips_a_key_the_miner_would_renormalize(engine):
    """A stance key the miner's own normalizer would CHANGE is never reflected.

    The realizer asks `_opinion_topic` — the same normalizer that produced the
    key — whether the key is already canonical. Idempotency means a well-formed
    key passes; anything still carrying a determiner, a preposition, or a
    preposed frame comes back different and is skipped, so it can never render
    as "you're the office work".

    This replaced an earlier version of this test that gated on
    `has_clean_topic`, whose action-verb list contains "work" and so rejected
    the perfectly legitimate key "remote work". Using the clause-subject guard
    on a stance key made the guard itself the cause of the flat reply.
    """
    out = _render(engine, [("the office work", 0.7)])
    assert "the office work" not in out, out


def test_guard_does_not_claim_to_catch_a_bare_fused_verb(engine):
    """Document the guard's real boundary instead of overstating it.

    Measured: idempotency does NOT reject a pronoun-less fused verb, because
    the frame strip keys off the frame's subject pronoun. A key like
    "believing nuclear energy" therefore renders. That is acceptable only
    because such a key can no longer be MINED — the frame is stripped upstream
    — which `test_miner_never_produces_a_fused_verb_key` and
    `test_render_is_driven_by_the_record_not_the_utterance` pin down. This test
    exists so a future reader does not mistake the guard for the fix.
    """
    out = _render(engine, [("believing nuclear energy", 0.7)])
    assert "believing nuclear energy" in out, out


def test_miner_never_produces_a_fused_verb_key(um):
    """The real defence: the miner cannot emit a fused reporting frame.

    This, not the realizer guard, is what closes the defect. Asserted over
    phrasings whose verbs appear in NO miner pattern, so it tests the
    grammatical strip rather than a lookup of known verbs.
    """
    um.mine_personal_facts("i think remote work is better than office work")
    um.mine_personal_facts("we suppose winter beats summer here")
    um.mine_personal_facts("she maintains sailing is finer than flying")
    assert "think remote work" not in um.opinions.stances
    assert "suppose winter" not in um.opinions.stances
    assert "maintains sailing" not in um.opinions.stances
    for key in um.opinions.stances:
        assert not key.startswith(("think ", "suppose ", "maintains ")), key


def test_realizer_shares_the_miners_normalizer(engine):
    """The realizer and the miner agree on key shape BY CONSTRUCTION.

    Asserted directly so a future refactor cannot quietly reintroduce a second,
    private opinion of what a stance key looks like — which is precisely how
    the two sides drifted in the first place.
    """
    um = UserModel.__new__(UserModel)
    um.opinions = UserStanceStore()
    for key in ("remote work", "office work", "winter", "quiet mornings",
                "filter coffee", "driving downtown"):
        assert um._opinion_topic(key) == key, key


def test_garble_guard_still_rejects_a_clause_subject():
    """The pre-existing clause-subject guard is untouched by this change.

    `has_clean_topic` still refuses a raw clause subject. The realizer simply
    no longer routes stance keys through it, because a stance key is not a
    clause subject.
    """
    assert not has_clean_topic("believe nuclear energy", "")
    assert not has_clean_topic("believe nuclear energy", "i believe nuclear energy")
    # A genuinely clean noun phrase is still accepted, and the veto is what
    # keeps "you're {topic}" from garbling.
    assert has_clean_topic("nuclear energy", "")
