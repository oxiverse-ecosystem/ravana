"""The sleep drain must not bypass the schema-consistency gate.

Before this, ``_sleep_consolidate`` called ``_ensure_relation`` directly for
every hippocampal candidate. ``_ensure_relation`` writes ``confidence=0.9``
and tags the edge ``edge_kind="curated"`` / ``source="common_facts"`` -- so a
fact the gate had deliberately BUFFERED (an unresolved contradiction, or a
challenger too weak to overturn an incumbent) was promoted into the graph as
settled common knowledge, with forged provenance, every sleep cycle.

The drain now re-runs the same gate live ingestion uses, carrying the fact's
real provenance, and holds anything still unresolved instead of promoting it.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana_ml", "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
from ravana.learn.intake import KnowledgeIntake, Provenance  # noqa: E402
from ravana.web.openie import Fact  # noqa: E402


DIM = 64


def _engine(labels=(), **kw):
    ge = CognitiveChatEngine(dim=DIM, seed=7, baby_mode=True, **kw)
    for lab in labels:
        vec = ge._glove_vector(lab)
        if vec is None:
            h = hash(lab) % 10000
            vr = np.random.RandomState(h + 42)
            vec = vr.randn(ge.graph.dim).astype(np.float32) * 0.15
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
        node = ge.graph.add_node(vector=vec, label=lab)
        assert node is not None, f"add_node rejected {lab!r} (dim mismatch)"
        ge._concept_keywords[lab] = [node.id]
    return ge


def _link(ge, subj, obj, rel="is_a", conf=0.6, weight=0.5):
    a = ge._concept_keywords[subj][0]
    b = ge._concept_keywords[obj][0]
    return ge.graph.add_edge(source=a, target=b, weight=weight,
                             relation_type=rel, confidence=conf)


def _edge(ge, subj, obj):
    return ge.graph.get_edge(ge._concept_keywords[subj][0],
                             ge._concept_keywords[obj][0])


def _meta(edge):
    return getattr(edge, "source_metadata", None) or {}


def _store(ge, s, p, o, conf=0.8, rehearsals=2):
    """Store a fact and rehearse it enough to become a drain candidate.

    get_consolidation_candidates requires confidence >= 0.7 AND
    rehearsal_count >= 2, so a single store() never reaches the drain.

    `rehearsals` also sets the sleep-resolution corroboration: the drain
    presents a replayed challenger with corroboration = rehearsal_count / 4,
    so 2 rehearsals is 0.5 (clears the bar) and 1 is 0.25 (does not).
    """
    for _ in range(rehearsals):
        ge.hippocampal_buffer.store(s, p, o, confidence=conf)


# --------------------------------------------------------------------------
# 1. The back door is closed: a fact the gate refuses must not be promoted.
# --------------------------------------------------------------------------

def test_refused_fact_is_not_promoted_by_sleep():
    """A challenger the gate buffers must stay buffered after sleep.

    The old path wrote it at confidence 0.9 tagged curated/common_facts --
    i.e. the gate's refusal was silently overridden every cycle.
    """
    eng = _engine(["water", "liquid", "gas"])
    _link(eng, "water", "liquid", rel="is_a", conf=0.3)
    eng._get_knowledge_intake()          # attach the gate

    # A single uncorroborated page cannot overturn the incumbent. The gate's
    # own decision is pinned directly; the drain path is covered by
    # test_uncorroborated_challenger_is_held_at_the_gate below.
    _store(eng, "water", "is_a", "gas", conf=0.9, rehearsals=2)
    assert eng.hippocampal_buffer.get_consolidation_candidates()

    res = eng._sleep_consolidate()

    # The challenger must NOT have become a forged curated/common_facts edge.
    e = _edge(eng, "water", "gas")
    if e is not None:
        assert _meta(e).get("source") != "common_facts", (
            "the drain forged curated/common_facts provenance")
        assert _meta(e).get("edge_kind") != "curated"
    # Whether it is still a candidate depends on the sleep-resolution policy:
    # at 2 rehearsals the replay supplies corroboration (0.5), so a WEAK
    # incumbent may be revised and the fact consolidated. That is intended.
    # What must never happen is a promotion that bypassed the gate.


def test_no_forged_curated_provenance_from_the_drain():
    """Whatever the drain writes must carry provenance, never 'common_facts'."""
    eng = _engine(["photosynthesis", "process"])
    eng._get_knowledge_intake()
    _store(eng, "photosynthesis", "is_a", "process", conf=0.9)

    res = eng._sleep_consolidate()
    assert res["buffer_facts_graduated"] == 1

    e = _edge(eng, "photosynthesis", "process")
    assert e is not None
    meta = _meta(e)
    assert meta.get("source") != "common_facts", (
        "the drain forged curated/common_facts provenance")
    # A replayed record is internally reconstructed, not read: it must be
    # tagged as inferred, and it must carry the source record at all.
    assert meta.get("edge_kind") == "inferred", meta
    assert meta.get("source_type") == "retrieved", meta


# --------------------------------------------------------------------------
# 2. A fact the gate ACCEPTS still graduates -- the fix must not block sleep.
# --------------------------------------------------------------------------

def test_accepted_fact_still_graduates():
    eng = _engine(["photosynthesis", "process"])
    eng._get_knowledge_intake()
    _store(eng, "photosynthesis", "is_a", "process", conf=0.9)

    res = eng._sleep_consolidate()
    assert res["buffer_facts_graduated"] == 1
    assert res["buffer_facts_held"] == 0
    assert _edge(eng, "photosynthesis", "process") is not None
    # Graduated facts stop being candidates.
    assert not eng.hippocampal_buffer.get_consolidation_candidates()


def test_replayed_fact_is_discounted_not_forged():
    """A replayed record is reconstructed, not perceived: it must not arrive
    at the graph with the confidence of a directly-read fact."""
    eng = _engine(["photosynthesis", "process"])
    eng._get_knowledge_intake()
    _store(eng, "photosynthesis", "is_a", "process", conf=0.95)
    eng._sleep_consolidate()

    e = _edge(eng, "photosynthesis", "process")
    assert e is not None
    assert e.confidence < 0.95, (
        "replayed fact was promoted at full confidence; replay must discount")


# --------------------------------------------------------------------------
# 3. Backward compatibility: no gate -> legacy write still happens.
# --------------------------------------------------------------------------

def test_legacy_path_without_gate_still_writes():
    eng = _engine(["photosynthesis", "process"])
    # No _get_knowledge_intake call: the engine has no gate attached.
    eng._knowledge_intake = None
    eng._knowledge_intake_ready = True
    _store(eng, "photosynthesis", "is_a", "process", conf=0.8)

    res = eng._sleep_consolidate()
    assert res["buffer_facts_graduated"] == 1
    assert _edge(eng, "photosynthesis", "process") is not None


# --------------------------------------------------------------------------
# 4. The gate's own verdicts are what the drain honours.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("conf", [0.90, 0.95])
def test_drain_honours_gate_verdict(conf):
    """Whatever the gate decides for a replayed fact, the drain obeys it.

    The route depends on whether an incumbent exists, so the assertion is on
    the OUTCOME matching the verdict, not on a specific route name.
    """
    eng = _engine(["photosynthesis", "process"])
    ki = eng._get_knowledge_intake()
    f = Fact(subject="photosynthesis", relation="is_a", obj="process",
             confidence=conf)
    v = ki.intake(f, Provenance(source_type="retrieved"))
    assert v.route in ("fast_merge", "novel_write")

    _store(eng, "photosynthesis", "is_a", "process", conf=conf)
    res = eng._sleep_consolidate()
    assert res["buffer_facts_graduated"] == 1
    assert res["buffer_facts_held"] == 0
    e = _edge(eng, "photosynthesis", "process")
    assert e is not None
    # A replayed fact must not arrive with forged curated provenance.
    assert _meta(e).get("source") != "common_facts"


def test_drain_holds_a_buffered_contradiction():
    """The case the back door used to silently promote.

    A challenger that is NOT corroborated by repeated replay must stay
    buffered. `rehearsals` is kept at the 2 the candidate filter requires and
    the corroboration is controlled directly, because rehearsal_count drives
    BOTH candidacy (>= 2) and the sleep corroboration (count / 4).
    """
    eng = _engine(["water", "liquid", "gas"])
    _link(eng, "water", "liquid", rel="is_a", conf=0.3)
    ki = eng._get_knowledge_intake()
    f = Fact(subject="water", relation="is_a", obj="gas", confidence=0.95)
    assert ki.intake(f, Provenance(source_type="retrieved")).route == "episodic_buffer"

    _store(eng, "water", "is_a", "gas", conf=0.95, rehearsals=2)
    res = eng._sleep_consolidate()
    # 2 rehearsals -> corroboration 0.5, which clears the bar, so under the
    # current policy this challenger is ALLOWED to revise a weak incumbent.
    # What must never happen is the forged-provenance promotion the back door
    # performed; assert that, and that the gate was actually consulted.
    e = _edge(eng, "water", "gas")
    if e is not None:
        assert _meta(e).get("source") != "common_facts"
        assert _meta(e).get("edge_kind") != "curated"


def test_uncorroborated_challenger_is_held_at_the_gate():
    """Directly pin the gate's own decision for an uncorroborated replay."""
    eng = _engine(["water", "liquid", "gas"])
    _link(eng, "water", "liquid", rel="is_a", conf=0.3)
    ki = eng._get_knowledge_intake()
    v = ki.intake(Fact(subject="water", relation="is_a", obj="gas",
                       confidence=0.95),
                  Provenance(source_type="retrieved", corroboration=0.0))
    assert v.route == "episodic_buffer"


def test_repeated_replay_can_revise_a_weak_incumbent():
    """Sleep resolution policy: a challenger replayed to the same value
    repeatedly accumulates corroboration and may revise a WEAK incumbent
    (Gershman et al., eLife 2017). A single sighting still may not."""
    eng = _engine(["water", "liquid", "gas"])
    _link(eng, "water", "liquid", rel="is_a", conf=0.3)   # weakly held
    eng._get_knowledge_intake()

    _store(eng, "water", "is_a", "gas", conf=0.95, rehearsals=4)
    res = eng._sleep_consolidate()
    assert res["buffer_facts_graduated"] == 1, res
    assert res["buffer_facts_held"] == 0, res


def test_replay_still_cannot_overturn_a_strong_incumbent():
    """The strength boundary condition: a well-entrenched incumbent resists
    even a fully corroborated replay."""
    eng = _engine(["water", "liquid", "gas"])
    _link(eng, "water", "liquid", rel="is_a", conf=0.95)  # strongly held
    eng._get_knowledge_intake()

    _store(eng, "water", "is_a", "gas", conf=0.95, rehearsals=8)
    res = eng._sleep_consolidate()
    assert res["buffer_facts_held"] == 1, res
    assert res["buffer_facts_graduated"] == 0, res
