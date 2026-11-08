"""Calibration of the two open design choices.

1. Retrieval effort: effort-scaling is right, but the effort term must be
   normalised over the ACTUAL schedule range, and effort must only be rewarded
   when retrieval is actually succeeding (Karpicke 2025 §4.21.2.2, after
   Rowland 2014: the testing effect is robust once initial retrieval success
   exceeds ~75%).

2. Sleep resolution: a replayed challenger is a memory the system itself
   stored, not a fresh web sighting, so repeated consistent replay must
   accumulate toward the corroboration threshold (Gershman et al., eLife
   2017: accumulated prediction error over repeated reactivations is what
   moves even a strong memory).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana_ml", "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ravana.graph.engine import GraphEngine  # noqa: E402
from ravana.learn.intake import (  # noqa: E402
    KnowledgeIntake, Provenance, REVIEW_SCHEDULE,
)
from ravana.web.openie import Fact  # noqa: E402


DIM = 16


def _engine(labels=()):
    ge = GraphEngine(dim=DIM, seed=7)
    for lab in labels:
        node = ge.graph.add_node(vector=np.zeros(DIM, dtype=np.float32),
                                 label=lab)
        ge._all_labels[lab] = node.id
    return ge


def _link(ge, subj, obj, rel="is_a", conf=0.6, weight=0.5):
    a = ge._all_labels[subj]
    b = ge._all_labels[obj]
    return ge.graph.add_edge(source=a, target=b, weight=weight,
                             relation_type=rel, confidence=conf)


def _edge(ge, subj, obj):
    return ge.graph.get_edge(ge._all_labels[subj], ge._all_labels[obj])


def _fact(s, r, o, conf=0.9):
    return Fact(subject=s, relation=r, obj=o, confidence=conf)


def _ki(ge, **kw):
    return KnowledgeIntake(ge=ge, **kw)


def _seed(ki, s="photosynthesis", r="is_a", o="process"):
    ki.intake(_fact(s, r, o), Provenance(source_type="web"))
    return (s, r, o)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Effort must be normalised over the real schedule range.
# ─────────────────────────────────────────────────────────────────────────────

def test_effort_spans_the_whole_schedule():
    """The old log(interval+1)/log(9) saturated at 8 and was nearly flat from
    8 -> 32, so the top of the expanding schedule was worth only ~1.7x the
    bottom. The normaliser must be the schedule maximum."""
    ge = _engine(["photosynthesis", "process"])
    ki = _ki(ge)
    key = _seed(ki)

    def gain(interval):
        g = _engine(["photosynthesis", "process"])
        k = _ki(g)
        k.intake(_fact("photosynthesis", "is_a", "process"),
                 Provenance(source_type="web"))
        e = _edge(g, "photosynthesis", "process")
        before = float(e.confidence)
        k._review_queue[key]["interval"] = interval
        k.record_review(key, recalled=True)
        return float(e.confidence) - before

    shortest = gain(REVIEW_SCHEDULE[0])
    longest = gain(REVIEW_SCHEDULE[-1])
    ratio = longest / max(shortest, 1e-9)
    # The top of the schedule must be worth substantially more than the bottom.
    assert ratio > 1.7, f"effort barely varies across the schedule: ratio={ratio:.2f}"


def test_effort_is_monotonic_in_interval():
    ge = _engine(["photosynthesis", "process"])
    ki = _ki(ge)
    key = _seed(ki)
    gains = []
    for interval in REVIEW_SCHEDULE:
        g = _engine(["photosynthesis", "process"])
        k = _ki(g)
        k.intake(_fact("photosynthesis", "is_a", "process"),
                 Provenance(source_type="web"))
        e = _edge(g, "photosynthesis", "process")
        before = float(e.confidence)
        k._review_queue[key]["interval"] = interval
        k.record_review(key, recalled=True)
        gains.append(float(e.confidence) - before)
    assert gains == sorted(gains), f"gain is not monotonic in interval: {gains}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. The success-rate gate.
# ─────────────────────────────────────────────────────────────────────────────

def test_low_success_rate_holds_the_interval():
    """A fact that keeps failing must not be pushed to a longer interval
    however many times it has been reviewed."""
    ge = _engine(["photosynthesis", "process"])
    ki = _ki(ge)
    key = _seed(ki)
    rec = ki._review_queue[key]

    # A history of failures with occasional successes -> below 75%.
    rec["outcomes"] = [True, False, False, True, False, False, False, False]
    assert ki._success_rate(rec) < ki.min_success_rate

    for _ in range(6):
        ki.record_review(key, recalled=True)

    assert rec["interval"] <= REVIEW_SCHEDULE[0], (
        "a fact below the success floor must be held at the shortest interval")


def test_high_success_rate_expands_the_interval():
    ge = _engine(["photosynthesis", "process"])
    ki = _ki(ge)
    key = _seed(ki)
    rec = ki._review_queue[key]
    rec["outcomes"] = [True] * 8
    assert ki._success_rate(rec) >= ki.min_success_rate

    for _ in range(4):
        ki.record_review(key, recalled=True)
    assert rec["interval"] > REVIEW_SCHEDULE[0], (
        "a reliably-recalled fact must be allowed to space out")


def test_success_rate_defaults_to_permissive():
    """Absence of evidence of failure must not hold a fact back."""
    assert KnowledgeIntake._success_rate({}) == 1.0
    assert KnowledgeIntake._success_rate({"outcomes": []}) == 1.0


def test_outcome_window_is_bounded():
    ge = _engine(["photosynthesis", "process"])
    ki = _ki(ge)
    key = _seed(ki)
    rec = ki._review_queue[key]
    for i in range(20):
        ki._record_outcome(rec, recalled=(i % 2 == 0))
    assert len(rec["outcomes"]) <= 8


def test_min_success_rate_is_configurable():
    ki = KnowledgeIntake(ge=None, min_success_rate=0.5)
    assert ki.min_success_rate == 0.5
    ki2 = KnowledgeIntake(ge=None, min_success_rate=5.0)
    assert ki2.min_success_rate == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sleep resolution: replay accumulates corroboration.
# ─────────────────────────────────────────────────────────────────────────────

def _rehearsed(n):
    """A buffered fact rehearsed n times, as the hippocampal store would."""
    f = Fact(subject="water", relation="is_a", obj="gas", confidence=0.9)
    f.rehearsal_count = n
    return f


def test_replay_corroboration_scales_with_rehearsal():
    """A challenger replayed to the same value repeatedly must accumulate
    evidence; a single replay must not."""
    from ravana.learn.intake import clamp01
    assert clamp01(int(_rehearsed(1).rehearsal_count) / 4.0) < 0.5
    assert clamp01(int(_rehearsed(4).rehearsal_count) / 4.0) >= 0.5


def test_repeated_replay_can_eventually_revise():
    """End-to-end at the gate: with enough accumulated rehearsal, a buffered
    challenger clears the corroboration bar and revises the incumbent."""
    ge = _engine(["water", "liquid", "gas"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.3)
    ki = _ki(ge)

    # One uncorroborated sighting: held.
    v1 = ki.intake(_fact("water", "is_a", "gas"),
                   Provenance(source_type="retrieved", corroboration=0.0))
    assert v1.route == "episodic_buffer"

    # The same value replayed to 4+ times: now corroborated enough to revise.
    v4 = ki.intake(_fact("water", "is_a", "gas"),
                   Provenance(source_type="retrieved", corroboration=0.5))
    assert v4.route == "revise", (
        "accumulated replay must be able to overturn a weakly-held incumbent")


def test_fresh_single_web_sighting_still_cannot_overturn():
    """The guard must still hold for a genuinely fresh, uncorroborated page."""
    ge = _engine(["water", "liquid", "gas"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.3)
    ki = _ki(ge)
    v = ki.intake(_fact("water", "is_a", "gas"), Provenance(source_type="web"))
    assert v.route == "episodic_buffer"


def test_strong_incumbent_still_resists_replay():
    """A well-entrenched incumbent must not be overturned by replay alone --
    the strength boundary condition (Gershman et al. 2017, Fig. 4)."""
    ge = _engine(["water", "liquid", "gas"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.95)
    ki = _ki(ge)
    v = ki.intake(_fact("water", "is_a", "gas"),
                  Provenance(source_type="retrieved", corroboration=1.0))
    assert v.route == "episodic_buffer", (
        "a strongly-entrenched incumbent must resist even a corroborated "
        "replay unless the challenger clears the margin")
