"""Unit tests for ravana.chat.personal_fact_store.

Pure-Python, deterministic tests for the two core user-profile stores:

* ``PersonalFactStore`` -- biographical (subject, attribute, value) facts with
  reinforcement, contradiction/supersession, and decay/prune logic.
* ``UserStanceStore`` -- learned opinions (polarity + confidence) with weighted
  merge, topic resolution, and retraction/reversal.

These stores carry exactly the logic the RAVANA round briefs flagged as
defect-prone (retired facts leaking into recall; stance-flip idempotency), so
they are exercised directly here rather than only indirectly through the slow
engine-level integration suite. No GloVe / engine boot is required, so the
suite is fast and order-independent under xdist.

Run: pytest tests/unit/test_personal_fact_store.py -n 4 --durations=10
"""

import pytest

from ravana.chat.personal_fact_store import (
    PersonalFactStore,
    UserStanceStore,
    PersonalFact,
    Stance,
)


# ───────────────────────── PersonalFactStore: basics ─────────────────────────

def test_assert_fact_stores_and_is_queryable():
    s = PersonalFactStore()
    s.assert_fact("cat", "name", "pixel")
    out = s.query_fact("cat", "name")
    assert len(out) == 1
    assert out[0].value == "pixel"
    assert out[0].confidence == 0.6  # default seed confidence
    assert out[0].source == "seed_regex"


def test_repeat_same_value_reinforces():
    s = PersonalFactStore()
    s.assert_fact("cat", "name", "pixel", confidence=0.6)
    s.assert_fact("cat", "name", "pixel")  # identical value -> reinforce
    out = s.query_fact("cat", "name")
    assert len(out) == 1  # still one record, not two
    assert out[0].confidence == 0.7  # +0.1 reinforcement
    assert out[0].rehearsal_count == 2


def test_different_value_opens_contradiction():
    s = PersonalFactStore()
    s.assert_fact("cat", "name", "pixel")
    s.assert_fact("cat", "name", "milo")  # new value for same (subj,attr)
    # Both records exist as separate entries:
    assert len(s.facts) == 2
    # A contradiction edge was opened:
    assert len(s.contradictions) == 1
    (old, new, _turn) = s.contradictions[0]
    assert old[2] == "pixel"
    assert new[2] == "milo"


# ──────────────── supersession: retired values must NOT leak ────────────────
# Defect class from the round briefs: "Retired values leaking into recall."

def test_contradict_marks_prior_superseded_and_excludes_it_from_recall():
    s = PersonalFactStore()
    s.assert_fact("cat", "name", "pixel")
    s.contradict("cat", "name", "milo")  # user correction: "no, it's milo"

    # The corrected value is the active fact:
    active = s.get("cat", "name")
    assert active is not None
    assert active.value == "milo"
    assert active.source == "correction"
    assert active.superseded is False

    # The retired value is flagged and must NOT surface in queries:
    retired = s.facts[(("cat", "name", "pixel"))]
    assert retired.superseded is True

    recalled = s.query_fact("cat", "name")
    assert [f.value for f in recalled] == ["milo"]  # only the active value


def test_reconcile_resolves_by_recency_recent_wins():
    s = PersonalFactStore()
    s.assert_fact("cat", "name", "pixel")
    # advance several turns so "pixel" becomes older
    for _ in range(5):
        s.advance_turn()
    # A *conflict* (not contradict, which would supersede first) opens a
    # contradiction edge that reconcile() later resolves:
    s.assert_fact("cat", "name", "milo")
    assert len(s.contradictions) == 1
    s.advance_turn()
    resolved = s.reconcile()
    key = ("cat", "name")
    assert key in resolved
    assert resolved[key].value == "milo"  # recent value wins the decay-score battle
    assert s.facts[("cat", "name", "pixel")].superseded is True


# ─────────────────────────────── confirm / reinforce ───────────────────────────────

def test_confirm_boosts_existing_confidence():
    s = PersonalFactStore()
    s.assert_fact("cat", "name", "pixel", confidence=0.6)
    s.confirm("cat", "name", "pixel")  # user: "yes, that's right"
    f = s.get("cat", "name")
    assert f.confidence == 0.85  # 0.6 + 0.25
    assert f.source == "user_confirmation"


def test_confirm_unknown_value_asserts_at_high_confidence():
    s = PersonalFactStore()
    s.confirm("cat", "name", "rex")  # no prior -> take user at their word
    f = s.get("cat", "name")
    assert f is not None
    assert f.value == "rex"
    assert f.confidence == 0.85
    assert f.source == "user_confirmation"


def test_confirm_mismatched_value_triggers_contradiction():
    # NOTE: `confirm` with a value that differs from the held fact behaves
    # differently from `contradict`: it does NOT retire the prior. It asserts
    # the (incorrectly) confirmed value as a high-confidence fact and opens a
    # contradiction edge, leaving the original still active. This is the real
    # behaviour of PersonalFactStore.confirm (verified by probing), recorded
    # here so the divergence is locked in and any future change is deliberate.
    s = PersonalFactStore()
    s.assert_fact("cat", "name", "pixel")
    s.confirm("cat", "name", "milo")  # said pixel, confirms milo -> conflict
    # prior stays active (not superseded) but the new value outranks it:
    assert s.facts[("cat", "name", "pixel")].superseded is False
    assert s.get("cat", "name").value == "milo"
    assert len(s.contradictions) == 1


def test_reinforce_only_touching_best_match():
    s = PersonalFactStore()
    s.assert_fact("cat", "name", "pixel", confidence=0.6)
    s.reinforce("cat", "name", "pixel")
    f = s.get("cat", "name")
    assert f.confidence == 0.7
    assert f.rehearsal_count == 2


# ─────────────────────────────── decay / prune / consolidation ───────────────────────────────

def test_prune_stale_removes_low_conf_old_facts():
    s = PersonalFactStore()
    s.assert_fact("cat", "name", "pixel", confidence=0.3)
    # age it well past the stale window
    for _ in range(12):
        s.advance_turn()
    removed = s.prune_stale(min_confidence=0.4, stale_after=10)
    assert removed == 1
    assert s.query_fact("cat", "name") == []


def test_consolidation_candidates_require_confidence_and_rehearsal():
    s = PersonalFactStore()
    s.assert_fact("cat", "name", "pixel", confidence=0.6)
    # not yet rehearsed enough (rehearsal_count == 1)
    assert s.get_consolidation_candidates() == []
    s.reinforce("cat", "name", "pixel")  # -> rehearsal_count 2, conf 0.7
    cands = s.get_consolidation_candidates()
    assert len(cands) == 1
    assert cands[0].value == "pixel"


# ─────────────────────────────── serialization round-trip ───────────────────────────────

def test_get_set_state_round_trips_including_superseded():
    s = PersonalFactStore()
    s.assert_fact("cat", "name", "pixel")
    s.contradict("cat", "name", "milo")
    state = s.get_state()

    s2 = PersonalFactStore()
    s2.set_state(state)
    # active value survived
    assert s2.get("cat", "name").value == "milo"
    # superseded flag survived (critical for not leaking retired values)
    assert s2.facts[("cat", "name", "pixel")].superseded is True
    # NOTE: contradict() supersedes the prior *before* it asserts the new value,
    # so it opens ZERO contradiction edges (the mechanism is supersession, not
    # the contradiction log). The contradiction log is only populated by plain
    # assert_fact conflicts. Recorded here so the contract is explicit.
    assert len(s2.contradictions) == 0


# ───────────────────────── UserStanceStore: merge / resolve / reverse ─────────────────────────

def test_express_stance_stores_polarity():
    u = UserStanceStore()
    u.express_stance("cats", polarity=1.0, confidence=0.8)
    st = u.query_stance("cats")
    assert st is not None
    assert st.polarity == 1.0
    assert st.confidence == 0.8


def test_express_stance_running_mean_blends_repeats():
    u = UserStanceStore()
    u.express_stance("cats", polarity=1.0, confidence=0.5)   # rehearsal 1
    u.express_stance("cats", polarity=-1.0, confidence=0.5)  # rehearsal 2
    st = u.query_stance("cats")
    # weighted mean: (_w_old*1.0 + _w_new*(-1.0)) / (_w_old + _w_new)
    # _w_old = 0.5*1 = 0.5, _w_new = 0.5 -> (0.5 - 0.5)/1.0 = 0.0
    assert st.polarity == 0.0
    # confidence: min(1, (0.5+0.5)/2 + 0.05) = 0.55
    assert abs(st.confidence - 0.55) < 1e-9
    assert st.rehearsal_count == 2


def test_reverse_stance_flips_pole_and_drops_confidence():
    u = UserStanceStore()
    u.express_stance("cats", polarity=1.0, confidence=0.8)
    # NOTE (behaviour verified by probing): reverse_stance returns the SAME
    # Stance object AFTER mutating it (polarity already flipped to -0.7), NOT
    # the prior stance as its docstring/text claims ("Returns the PRIOR
    # stance"). The prior is instead captured in `last_reversal`. The test
    # asserts the real behaviour so the doc/impl mismatch is explicit.
    prior = u.reverse_stance("cats", reversal_strength=0.85)
    st = u.query_stance("cats")
    assert prior is st  # same object, already post-reversal
    assert st.polarity < 0.0       # flipped to the opposite pole
    assert st.confidence < 0.8     # attitude change injects uncertainty
    # prior read available via last_reversal (topic, old_polarity, new_polarity)
    assert u.last_reversal is not None
    assert u.last_reversal[1] == 1.0  # old polarity preserved here


def test_reverse_stance_is_idempotent_within_a_turn():
    u = UserStanceStore()
    u.express_stance("cats", polarity=1.0, confidence=0.8)
    first = u.reverse_stance("cats", reversal_strength=0.85)
    after_first = u.query_stance("cats").polarity
    # repeated mining of the same utterance in the same turn must NOT re-flip:
    second = u.reverse_stance("cats", reversal_strength=0.85)
    assert second is first  # returned without re-applying
    assert u.query_stance("cats").polarity == after_first


def test_reverse_stance_with_no_prior_returns_none():
    u = UserStanceStore()
    # user says "i take back X" on a topic they never expressed -> benign
    assert u.reverse_stance("nonexistent") is None
    assert u.query_stance("nonexistent") is None


def test_resolve_topic_exact_substring_and_jaccard():
    u = UserStanceStore()
    u.express_stance("plastic bans", polarity=-1.0, confidence=0.7)
    # exact key
    assert u.resolve_topic("plastic bans") == "plastic bans"
    # substring
    assert u.resolve_topic("i support plastic bans") == "plastic bans"
    # disjoint phrase with no overlap -> None (never fabricate a read)
    assert u.resolve_topic("banana bread") is None


def test_stance_serialization_round_trip():
    u = UserStanceStore()
    u.express_stance("cats", polarity=1.0, confidence=0.8, valence=0.3, arousal=0.1)
    state = u.get_state()
    u2 = UserStanceStore()
    u2.set_state(state)
    st = u2.query_stance("cats")
    assert st.polarity == 1.0
    assert st.confidence == 0.8
    assert st.valence == 0.3


def test_prune_stale_drops_low_conf_old_stance():
    u = UserStanceStore()
    u.express_stance("fad", polarity=0.2, confidence=0.2)
    for _ in range(10):
        u.advance_turn()
    removed = u.prune_stale(min_confidence=0.3, stale_after=8)
    assert removed == 1
    assert u.query_stance("fad") is None


# ───────── Residual limitation #1 (round 2026-08-20T0701Z): stance provenance ─────────
# A stance is keyed on a SUBORDINATE concept ("silence") while the user's
# utterance also names the SALIENT broader concept ("winter"). A later query
# about the broader concept must bridge to the held stance instead of returning
# None (which previously forced the "honest i don't have a read" fallback even
# though the user clearly expressed a view). This is the provenance-bridge
# capability: record the salient nouns of the producing utterance and let the
# resolver link a co-mention back to the stance.

def test_stance_provenance_bridges_broader_concept_query():
    u = UserStanceStore()
    # The miner keys the stance on the subordinate head "silence" but records
    # the salient nouns of the whole object phrase as provenance.
    u.express_stance("silence", polarity=1.0, confidence=0.8,
                     provenance=["silence", "deep", "winter"])
    # Exact / substring / Jaccard all MISS: "winter" is not the key nor a token
    # of it. Only the provenance bridge resolves it.
    assert u.resolve_topic("winter") == "silence"
    # A phrase that co-mentions the salient noun also bridges.
    assert u.resolve_topic("am i for or against winter") == "silence"


def test_stance_provenance_empty_seed_does_not_fabricate():
    u = UserStanceStore()
    u.express_stance("silence", polarity=1.0, confidence=0.8)  # no provenance
    # Without provenance there is nothing to bridge -> honest abstention.
    assert u.resolve_topic("winter") is None


def test_stance_provenance_persists_across_serialization():
    u = UserStanceStore()
    u.express_stance("silence", polarity=1.0, confidence=0.8,
                     provenance=["silence", "deep", "winter"])
    state = u.get_state()
    u2 = UserStanceStore()
    u2.set_state(state)
    st = u2.query_stance("silence")
    assert set(st.provenance) == {"silence", "deep", "winter"}
    # The bridge survives the round-trip.
    assert u2.resolve_topic("winter") == "silence"


def test_stance_provenance_merges_across_encounters():
    u = UserStanceStore()
    u.express_stance("silence", polarity=1.0, confidence=0.8,
                     provenance=["silence", "deep", "winter"])
    # A second encounter about the same keyed topic with a different salient
    # noun must union into the held provenance (online growth).
    u.express_stance("silence", polarity=1.0, confidence=0.8,
                     provenance=["silence", "snow"])
    st = u.query_stance("silence")
    assert set(st.provenance) == {"silence", "deep", "winter", "snow"}

