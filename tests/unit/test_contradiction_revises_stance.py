#!/usr/bin/env python3
"""Regression tests — contradiction must REVISE, not average away
(round 2026-10-04T0827Z, defect D1).

Measured defect (fresh rotating probe, this round):

    T3 "i think public libraries are underrated"
       -> stance 'public libraries' +0.80 conf 0.58
    T4 "no actually i think public libraries are overrated, cafes are better"
       -> "got it — you used to say you were uncertain about public libraries,
          and now you're uncertain about. i've updated my read (confidence 0.29)."
       -> stance 'public libraries' -0.07 conf 0.29
    T10 "what do i think about libraries"
       -> "you're uncertain about public libraries."

An announced revision that did not revise, with BOTH sides rendered from the
low-magnitude band.

Root cause, pinned by instrumenting every mutation of that stance:
the merge in `UserStanceStore.express_stance` is a weighted running MEAN, so
folding +0.8 against -0.8 can only produce a near-zero midpoint (+0.0996
measured). The delta rule for an attitude CHANGE already existed in the store
(`recode_stance_toward`) and was already used by `user_model`'s free-form
path; the merge never consulted it.

These tests assert the CLASS property — an opposing-signed expression revises
the stance toward the stated value — and are deliberately written against
topics and phrasings that appear nowhere in the probe above, so they cannot
pass by matching this round's words.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana", "src"), os.path.join(PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.chat.personal_fact_store import UserStanceStore


def _store(polarity=0.8, conf=0.5, topic="tideline marsh"):
    u = UserStanceStore()
    u.express_stance(topic, polarity=polarity, confidence=conf)
    return u


def _revise(u, topic, polarity, confidence=0.5):
    """Express a contradicting value the way the conversational layer does.

    `mark_revision` is what the engine calls when the utterance RESTATES a
    prior view ("no actually i think X is overrated"). Without it an opposing
    expression is just another sample and correctly averages — that boundary
    is pinned by the store's own existing tests.
    """
    u.mark_revision(topic)
    u.express_stance(topic, polarity=polarity, confidence=confidence)


def test_opposing_expression_leaves_the_hemisphere_it_started_in():
    """The core class property: a sign conflict is a CHANGE, not an average.

    A weighted mean of +0.8 and -0.8 is +0.1 — still positive, so the stance
    silently kept the value the user just abandoned.
    """
    u = _store()
    _revise(u, "tideline marsh", -0.8)
    s = u.stances["tideline marsh"]
    assert s.polarity < 0.0, (
        f"contradiction averaged instead of revising: polarity {s.polarity:+.4f} "
        f"still sits in the hemisphere the user abandoned")


def test_revision_lands_close_to_the_stated_value():
    """It must move TOWARD THE STATED VALUE by at least half the distance.

    Guards against the cheap fix of merely negating the sign, which would also
    make the first test pass but is not what the user said.

    The property is the delta rule, not an unconditional flip: the new stance
    must sit at least halfway between the old value and the newly stated one.
    The pre-fix weighted mean landed exactly AT the midpoint (0.8*0.5 +
    -0.8*0.5 = +0.0000), so "halfway or further" is strictly stronger than the
    old behaviour and still admits a weak contrary signal failing to overturn
    a strong held read — which is the store's documented bounded inertia.
    """
    held, stated = 0.8, -0.8
    u = _store(polarity=held)
    _revise(u, "tideline marsh", stated)
    got = u.stances["tideline marsh"].polarity
    halfway = (held + stated) / 2.0
    assert got <= halfway, (
        f"revision stayed on the held side of the midpoint: {got:+.4f} vs "
        f"halfway {halfway:+.4f} (held {held:+.2f}, stated {stated:+.2f})")


def test_same_sign_restatement_still_reinforces():
    """Reinforcement must be untouched — the weighted mean is correct there.

    This is the guard against the merge being replaced outright by a recode:
    a same-sign restatement must still entrench, not snap to its own value.
    """
    u = _store(polarity=0.3, conf=0.5)
    u.express_stance("tideline marsh", polarity=0.8, confidence=0.5)
    s = u.stances["tideline marsh"]
    assert s.polarity > 0.3, (
        f"same-sign reinforcement regressed: {s.polarity:+.4f} did not "
        f"entrench past the prior +0.30")
    assert s.rehearsal_count >= 2, s.rehearsal_count


def test_opposing_sign_predicate_is_class_level():
    """The test is a SIGN comparison, not a topic or a phrase.

    The probe used 'public libraries'; these topics were never mentioned.
    """
    # A weak contrary signal against an entrenched read is NOT required to
    # cross the line — the store's bounded inertia deliberately resists that
    # ("one contrary mention does not overturn an entrenched read"). What must
    # hold for every case is the delta rule: the stance moves toward the newly
    # stated value, and a strong stated value does cross the line.
    # Progress is measured as the fraction of the distance from the held value
    # to the newly stated value that the stance actually covered, which is
    # direction-symmetric (a signed `got <= halfway` comparison only expresses
    # progress for a move toward the negative side).
    for topic, held, stated, must_cross in (
        ("kelp highway", 0.7, -0.6, True),
        ("brass doorknob", -0.75, 0.5, True),
        ("harbour fog", 0.9, -0.3, False),
    ):
        u = _store(polarity=held, conf=0.5, topic=topic)
        _revise(u, topic, stated)
        got = u.stances[topic].polarity
        progress = (got - held) / (stated - held)
        assert progress >= 0.5, (
            f"{topic!r}: held {held:+.2f}, stated {stated:+.2f} landed "
            f"{got:+.4f} — covered only {progress:.0%} of the distance "
            f"(pre-fix the weighted mean cannot exceed the new weight's share)")
        if must_cross:
            assert got * stated > 0.0, (
                f"{topic!r}: a strong stated value {stated:+.2f} against held "
                f"{held:+.2f} landed {got:+.4f} — did not cross the line")


def test_unmarked_opposite_expression_still_averages():
    """The boundary the store must NOT cross.

    An opposing expression with no revision frame is another observation, not a
    change of mind, so the weighted mean still applies and an entrenched read
    still resists it. This is the behaviour the store's own existing tests pin
    (`test_express_stance_running_mean_blends_repeats`,
    `test_entrenchment_has_a_bound_not_a_removal`) and the reason the recode is
    gated on a marker rather than on the sign alone.
    """
    u = _store(polarity=1.0, conf=0.5, topic="tideline marsh")
    u.express_stance("tideline marsh", polarity=-1.0, confidence=0.5)
    got = u.stances["tideline marsh"].polarity
    assert abs(got) < 0.05, (
        f"an unmarked opposite expression stopped averaging: {got:+.4f}")

    # ...and the marker is consumed, so the NEXT turn's bare opinion is not a
    # revision either.
    u2 = _store(polarity=1.0, conf=0.5, topic="tideline marsh")
    _revise(u2, "tideline marsh", -1.0)
    assert u2.stances["tideline marsh"].polarity < 0.0
    u2.advance_turn()
    u2.express_stance("tideline marsh", polarity=-1.0, confidence=0.5)
    got2 = u2.stances["tideline marsh"].polarity
    assert got2 < 0.0, f"post-revision opinion regressed to averaging: {got2:+.4f}"


def test_retraction_still_flips_without_a_stated_value():
    """The retraction path is NOT disabled.

    `reverse_stance` states no new value, so it must still flip toward the
    opposite pole. A fix that simply removed the flip would pass the tests
    above while breaking explicit recants.
    """
    u = _store(polarity=0.8, conf=0.5, topic="brass doorknob")
    u.reverse_stance("brass doorknob", reversal_strength=0.85)
    s = u.stances["brass doorknob"]
    assert s.polarity < 0.0, (
        f"retraction no longer flips: {s.polarity:+.4f}")


def test_end_to_end_engine_revises_the_users_stance():
    """Engine level: the reply must not announce a revision that did not happen.

    Phrasings and topics are deliberately different from the probe that
    exposed the defect.
    """
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_contradiction_0827z")
    eng.process_turn("i think tide pools are underrated")
    before = eng.user_model.opinions.stances.get("tide pools")
    assert before is not None and before.polarity > 0.3, (
        f"setup did not establish a positive stance: {before}")

    eng.process_turn("no actually i think tide pools are overrated")
    after = eng.user_model.opinions.stances.get("tide pools")
    assert after is not None, "contradiction dropped the stance entirely"
    assert after.polarity < 0.0, (
        f"engine-level contradiction averaged away: {after.polarity:+.4f}")

    out = (eng.process_turn("what do i think about tide pools") or "").lower()
    assert "uncertain" not in out, (
        f"engine still reports the revised stance as uncertain: {out!r}")
    try:
        eng.stop_background_learning()
    except Exception:
        pass