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


def _store(ge, s, p, o, conf=0.8):
    """Store a fact and rehearse it enough to become a drain candidate.

    get_consolidation_candidates requires confidence >= 0.7 AND
    rehearsal_count >= 2, so a single store() never reaches the drain.
    """
    for _ in range(2):
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

    # A single uncorroborated page cannot overturn the incumbent.
    _store(eng, "water", "is_a", "gas", conf=0.9)
    assert eng.hippocampal_buffer.get_consolidation_candidates()

    res = eng._sleep_consolidate()

    assert res["buffer_facts_held"] == 1
    assert res["buffer_facts_graduated"] == 0
    # The challenger must NOT have become a curated edge.
    e = _edge(eng, "water", "gas")
    assert e is None or _meta(e).get("edge_kind") != "curated"
    # And it must still be a live candidate, not silently marked consolidated.
    assert eng.hippocampal_buffer.get_consolidation_candidates()


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
    """The case the back door used to silently promote."""
    eng = _engine(["water", "liquid", "gas"])
    _link(eng, "water", "liquid", rel="is_a", conf=0.3)
    ki = eng._get_knowledge_intake()
    f = Fact(subject="water", relation="is_a", obj="gas", confidence=0.95)
    assert ki.intake(f, Provenance(source_type="retrieved")).route == "episodic_buffer"

    _store(eng, "water", "is_a", "gas", conf=0.95)
    res = eng._sleep_consolidate()
    assert res["buffer_facts_held"] == 1
    assert res["buffer_facts_graduated"] == 0
    e = _edge(eng, "water", "gas")
    assert e is None or _meta(e).get("edge_kind") != "curated"
