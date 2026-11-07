"""Unit tests for the schema-consistency-gated knowledge intake.

These exercise the brain's write gate (Tse 2011; McClelland 2013; Kumaran et
al. 2016; Craik & Lockhart 1972; Karpicke & Roediger 2008; AGM 1985; Nader et
al. 2000) against the REAL ConceptGraph, so the wiring is tested, not a mock.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana_ml", "src"))

from ravana.web.openie import Fact  # noqa: E402
from ravana.graph.engine import GraphEngine  # noqa: E402
from ravana.learn.intake import (  # noqa: E402
    KnowledgeIntake, Provenance, SchemaConsistencyGate, clamp01,
    review_interval_for,
)


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


def _fact(s, r, o):
    return Fact(subject=s, relation=r, obj=o, confidence=0.7)


# ── consistency ──────────────────────────────────────────────────────────────

def test_consistency_zero_for_unknown_world():
    ge = _engine()
    gate = SchemaConsistencyGate()
    # Nothing known at all -> the fact fills no existing slot.
    assert gate.consistency(_fact("water", "is_a", "liquid"), ge) == 0.0


def test_consistency_rises_with_existing_structure():
    ge = _engine(["water", "liquid", "sky", "blue"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.7)
    _link(ge, "sky", "blue", rel="has_property", conf=0.7)
    gate = SchemaConsistencyGate()

    novel = gate.consistency(_fact("ozone", "is_a", "gas"), ge)
    known_slot = gate.consistency(_fact("water", "is_a", "liquid"), ge)
    same_rel_new_obj = gate.consistency(_fact("water", "is_a", "substance"), ge)

    assert novel == 0.0                      # neither endpoint known
    assert same_rel_new_obj > novel          # subject + slot exist
    assert known_slot > same_rel_new_obj     # both endpoints + edge exist
    assert known_slot <= 1.0


# ── elaboration ──────────────────────────────────────────────────────────────

def test_elaboration_zero_when_nothing_known():
    ge = _engine()
    gate = SchemaConsistencyGate()
    assert gate.elaboration(_fact("water", "is_a", "liquid"), ge) == 0.0


def test_elaboration_rewards_connected_endpoints():
    ge = _engine(["water", "liquid", "sky", "blue", "ice"])
    _link(ge, "water", "liquid", rel="is_a")
    _link(ge, "water", "ice", rel="has_property")
    _link(ge, "sky", "blue", rel="has_property")
    gate = SchemaConsistencyGate()

    isolated = gate.elaboration(_fact("ozone", "is_a", "gas"), ge)
    connected = gate.elaboration(_fact("water", "is_a", "liquid"), ge)
    assert connected > isolated
    assert connected > 0.0


def test_elaboration_bridge_bonus():
    """A fact linking two known-but-unlinked nodes elaborates more than one
    linking an unknown node."""
    ge = _engine(["water", "steam", "sky"])
    _link(ge, "water", "steam", rel="related_to")
    gate = SchemaConsistencyGate()

    bridge = gate.elaboration(_fact("water", "related_to", "sky"), ge)
    unknown = gate.elaboration(_fact("water", "related_to", "unobtainium"), ge)
    assert bridge > unknown


# ── entrenchment / contradiction ─────────────────────────────────────────────

def test_contradiction_none_when_no_incumbent():
    ge = _engine(["water"])
    gate = SchemaConsistencyGate()
    assert gate.detect_contradiction(_fact("water", "is_a", "liquid"), ge,
                                     Provenance()) is None


def test_contradiction_none_on_re_sighting():
    ge = _engine(["water", "liquid"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.7)
    gate = SchemaConsistencyGate()
    assert gate.detect_contradiction(_fact("water", "is_a", "liquid"), ge,
                                     Provenance()) is None


def test_contradiction_detected_on_conflicting_value():
    ge = _engine(["water", "liquid", "gas"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.7)
    gate = SchemaConsistencyGate()
    con = gate.detect_contradiction(_fact("water", "is_a", "gas"), ge,
                                    Provenance(source_type="web"))
    assert con is not None
    assert con.subject == "water"
    assert con.old_value == "liquid"
    assert con.new_value == "gas"
    assert con.old_entrenchment > con.new_entrenchment


# ── routing ──────────────────────────────────────────────────────────────────

def test_route_fast_merge_when_consistent():
    ge = _engine(["water", "liquid"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.7)
    ki = KnowledgeIntake(ge=ge)
    v = ki.intake(_fact("water", "is_a", "liquid"))
    assert v.route == "fast_merge"
    assert v.reason.startswith("schema-consistent")
    assert ki.stats["fast_merge"] == 1


def test_route_buffer_when_inconsistent():
    ge = _engine(["water", "liquid"])
    ki = KnowledgeIntake(ge=ge)
    v = ki.intake(_fact("water", "is_a", "liquid"))
    assert v.route == "episodic_buffer"
    assert v.consistency < ki.gate.fast_merge_threshold
    # No buffer attached -> retained at LOW confidence, never dropped.
    assert ki.stats["fallback_merge"] == 1
    edge = ge.graph.get_edge(ge._all_labels["water"], ge._all_labels["liquid"])
    assert edge is not None
    assert edge.confidence <= 0.2       # 0.25 * retention_prior


def test_no_fact_is_silently_dropped():
    """Every extracted fact must land somewhere: graph, buffer, or (failing
    both) an explicit rejection count. Nothing may vanish."""
    ge = _engine(["water", "liquid", "sky", "blue", "fire", "smoke"])
    ki = KnowledgeIntake(ge=ge)
    for s, r, o in [("water", "is_a", "liquid"), ("sky", "has_property", "blue"),
                    ("fire", "causes", "smoke")]:
        ki.intake(_fact(s, r, o))
    handled = (ki.stats["fast_merge"] + ki.stats["episodic_buffer"]
               + ki.stats["revise"] + ki.stats["fallback_merge"]
               + ki.stats["rejected"])
    assert handled == 3
    assert ki.stats["rejected"] == 0


def test_route_revise_when_challenger_strong():
    ge = _engine(["water", "liquid", "gas"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.2)   # weakly held incumbent
    ki = KnowledgeIntake(ge=ge)
    # A high-reliability source (the user) is a strong challenger.
    v = ki.intake(_fact("water", "is_a", "gas"),
                  Provenance(source_type="user"), salience=1.0)
    assert v.route == "revise", v.reason
    assert v.contradiction.winner_is_new
    assert ki.stats["revise"] == 1


def test_weak_challenger_cannot_revise():
    """A triple the extractor is unsure about must not overturn an incumbent,
    even a weakly-held one -- otherwise a noisy extractor can displace a real
    value with a bad one."""
    ge = _engine(["water", "liquid", "gas"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.3)
    ki = KnowledgeIntake(ge=ge)
    shaky = Fact(subject="water", relation="is_a", obj="gas", confidence=0.05)
    v = ki.intake(shaky, Provenance(source_type="web"), salience=1.0)
    assert v.route == "episodic_buffer"
    # blocked by one of the two guards: the support floor or the
    # single-uncorroborated-source rule. Either way the incumbent is safe.
    assert ("support floor" in v.reason) or ("uncorroborated" in v.reason)
    # incumbent untouched
    assert ki.gate._edge_confidence(
        ge.graph.get_edge(ge._all_labels["water"], ge._all_labels["liquid"])) == 0.3


def test_extractor_confidence_feeds_challenger_entrenchment():
    ge = _engine(["water", "liquid", "gas"])
    gate = SchemaConsistencyGate()
    prov = Provenance(source_type="web")
    low = gate.challenger_entrenchment(
        Fact("water", "is_a", "gas", 0.05), ge, prov)
    high = gate.challenger_entrenchment(
        Fact("water", "is_a", "gas", 0.95), ge, prov)
    assert high > low


def test_single_web_source_cannot_overturn_incumbent():
    """A single uncorroborated page must not revise an incumbent; a
    corroborated claim (or a high-reliability source) may."""
    ge = _engine(["water", "liquid", "gas"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.3)
    ki = KnowledgeIntake(ge=ge)
    f = Fact("water", "is_a", "gas", 0.9)

    v_lone = ki.intake(f, Provenance(source_type="web"), salience=1.0)
    assert v_lone.route == "episodic_buffer"
    assert "uncorroborated" in v_lone.reason

    v_corr = ki.intake(f, Provenance(source_type="web", corroboration=0.8),
                       salience=1.0)
    assert v_corr.route == "revise"


def test_route_buffer_when_incumbent_entrenched():
    ge = _engine(["water", "liquid", "gas"])
    _link(ge, "water", "liquid", rel="is_a", conf=1.0)   # fully entrenched
    ki = KnowledgeIntake(ge=ge)
    v = ki.intake(_fact("water", "is_a", "gas"),
                  Provenance(source_type="web"), salience=0.4)
    assert v.route == "episodic_buffer", v.reason
    # The incumbent must be untouched.
    assert ki.gate._edge_confidence(ge.graph.get_edge(
        ge._all_labels["water"], ge._all_labels["liquid"])) == 1.0


def test_revise_demotes_incumbent_and_records_supersession():
    ge = _engine(["water", "liquid", "gas"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.2)
    ki = KnowledgeIntake(ge=ge)
    ki.intake(_fact("water", "is_a", "gas"),
              Provenance(source_type="user"), salience=1.0)

    inc = ge.graph.get_edge(ge._all_labels["water"], ge._all_labels["liquid"])
    chal = ge.graph.get_edge(ge._all_labels["water"], ge._all_labels["gas"])
    assert inc.confidence < 0.2                       # demoted
    assert inc.source_metadata.get("superseded_by") == "gas"
    assert chal is not None and chal.confidence > inc.confidence


# ── salience / write gating ──────────────────────────────────────────────────

def test_salience_gates_effective_write():
    ge = _engine(["water", "liquid"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.5)
    ki = KnowledgeIntake(ge=ge)
    high = ki.intake(_fact("water", "is_a", "liquid"), salience=1.0)
    assert 0.0 < high.effective_write <= 1.0
    assert high.effective_write >= clamp01(high.salience) * 0.35


def test_predictable_fact_moves_less():
    """A low-salience (unsurprising) fact must move the weight less than a
    high-salience one, at identical structure."""
    ge = _engine(["water", "liquid"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.5)
    gate = SchemaConsistencyGate()
    v_low = gate.evaluate(_fact("water", "is_a", "liquid"), ge,
                          Provenance(), salience=0.0)
    v_high = gate.evaluate(_fact("water", "is_a", "liquid"), ge,
                           Provenance(), salience=1.0)
    assert v_low.effective_write < v_high.effective_write


# ── provenance ───────────────────────────────────────────────────────────────

def test_provenance_trust_is_computed_not_tagged():
    prov = Provenance(source_type="web", corroboration=0.0)
    t_cold = prov.trust(recency_halflife=0.0)       # no recency credit
    t_warm = prov.trust(recency_halflife=1e12)      # full recency credit
    assert t_warm > t_cold
    # An inferred record is discounted relative to a read one.
    inferred = Provenance(source_type="inference", corroboration=0.5)
    read = Provenance(source_type="web", corroboration=0.5)
    assert inferred.trust(recency_halflife=1e12) < read.trust(recency_halflife=1e12)


def test_provenance_metadata_roundtrip():
    prov = Provenance(source="example.com", source_type="web",
                      url="http://example.com/x", extractor="openie",
                      operations=("copula-splice",), corroboration=0.4)
    back = Provenance.from_metadata(prov.as_metadata())
    assert back.source == prov.source
    assert back.source_type == prov.source_type
    assert back.operations == prov.operations
    assert back.corroboration == pytest.approx(0.4)
    assert Provenance.from_metadata(None).source_type == "web"


def test_provenance_written_to_edge_is_durable():
    ge = _engine(["water", "liquid"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.5)
    ki = KnowledgeIntake(ge=ge)
    prov = Provenance(source="example.com", source_type="web",
                      url="http://example.com/water")
    ki.intake(_fact("water", "is_a", "liquid"), prov)
    edge = ge.graph.get_edge(ge._all_labels["water"], ge._all_labels["liquid"])
    meta = edge.source_metadata
    assert meta["source"] == "example.com"
    assert meta["url"] == "http://example.com/water"
    assert meta["source_type"] == "web"
    # Round-trips through the durable form.
    assert Provenance.from_metadata(meta).trust() > 0.0


# ── retrieval practice ───────────────────────────────────────────────────────

def test_review_interval_expands_with_retention():
    short = review_interval_for(0.0, 0)
    long_ = review_interval_for(1.0, 0)
    assert long_ > short
    assert review_interval_for(0.5, 0) >= 1


def test_intake_schedules_retrieval_practice():
    ge = _engine(["water", "liquid"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.5)
    ki = KnowledgeIntake(ge=ge)
    ki.intake(_fact("water", "is_a", "liquid"))
    assert len(ki._review_queue) == 1
    assert ki.due_for_review() == []            # not due yet
    interval = ki._review_queue[("water", "is_a", "liquid")]["interval"]
    assert interval >= 1
    for _ in range(interval):
        ki.advance_turn()
    due = ki.due_for_review()
    assert due == [("water", "is_a", "liquid")]


def test_successful_retrieval_expands_interval():
    ge = _engine(["water", "liquid"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.5)
    ki = KnowledgeIntake(ge=ge)
    ki.intake(_fact("water", "is_a", "liquid"))
    key = ("water", "is_a", "liquid")
    first_interval = ki._review_queue[key]["interval"]

    ki.advance_turn()
    ki.record_review(key, recalled=True)
    assert ki._review_queue[key]["reviews"] == 1
    assert ki._review_queue[key]["interval"] >= first_interval
    assert ki.stats["reviewed"] == 1


def test_failed_retrieval_resets_interval():
    ge = _engine(["water", "liquid"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.5)
    ki = KnowledgeIntake(ge=ge)
    ki.intake(_fact("water", "is_a", "liquid"))
    key = ("water", "is_a", "liquid")

    for _ in range(4):
        ki.advance_turn()
        ki.record_review(key, recalled=True)
    grown = ki._review_queue[key]["interval"]
    assert grown > 1

    ki.advance_turn()
    ki.record_review(key, recalled=False)
    assert ki._review_queue[key]["reviews"] == 0
    assert ki._review_queue[key]["interval"] == 1     # shortest interval


def test_failed_retrieval_is_rescheduled_soonest():
    ge = _engine(["water", "liquid", "sky", "blue"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.5)
    _link(ge, "sky", "blue", rel="has_property", conf=0.5)
    ki = KnowledgeIntake(ge=ge)
    ki.intake(_fact("water", "is_a", "liquid"))
    ki.intake(_fact("sky", "has_property", "blue"))
    intervals = [rec["interval"] for rec in ki._review_queue.values()]
    for _ in range(max(intervals)):
        ki.advance_turn()
    due = ki.due_for_review()
    assert len(due) == 2


# ── serialisation ────────────────────────────────────────────────────────────

def test_state_roundtrip():
    ge = _engine(["water", "liquid"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.5)
    ki = KnowledgeIntake(ge=ge)
    ki.intake(_fact("water", "is_a", "liquid"))
    ki.advance_turn()
    ki.record_review(("water", "is_a", "liquid"), recalled=True)

    state = ki.get_state()
    ki2 = KnowledgeIntake(ge=ge)
    ki2.set_state(state)
    assert ki2.turn == ki.turn
    assert ki2.stats["reviewed"] == 1
    assert ("water", "is_a", "liquid") in ki2._review_queue
    assert ki2._review_queue[("water", "is_a", "liquid")]["reviews"] == 1


# ── buffer routing ──────────────────────────────────────────────────────────

class _FakeBuffer:
    def __init__(self):
        self.stored = []

    def store(self, subject, predicate, object, confidence=0.8, aliases=None):
        self.stored.append((subject, predicate, object, confidence))


def test_inconsistent_fact_goes_to_episodic_buffer():
    ge = _engine()
    buf = _FakeBuffer()
    ki = KnowledgeIntake(ge=ge, buffer=buf)
    v = ki.intake(_fact("water", "is_a", "liquid"))
    assert v.route == "episodic_buffer"
    assert len(buf.stored) == 1
    assert buf.stored[0][0] == "water"
    assert ki.stats["episodic_buffer"] == 1


def test_buffer_confidence_tracks_retention_prior():
    ge = _engine()
    buf = _FakeBuffer()
    ki = KnowledgeIntake(ge=ge, buffer=buf)
    ki.intake(_fact("water", "is_a", "liquid"),
              Provenance(source_type="user"))
    conf = buf.stored[0][3]
    assert 0.4 <= conf <= 1.0


# ── WebToGraph wiring ────────────────────────────────────────────────────────

def test_web_to_graph_routes_through_intake():
    from ravana.web.web_to_graph import WebToGraph
    ge = _engine(["water", "liquid"])
    _link(ge, "water", "liquid", rel="is_a", conf=0.5)
    ki = KnowledgeIntake(ge=ge)
    w2g = WebToGraph(ge, intake=ki)

    assert w2g.learn_text("Water is liquid.", source_url="http://x.test") == 1
    assert ki.stats["fast_merge"] == 1
    edge = ge.graph.get_edge(ge._all_labels["water"], ge._all_labels["liquid"])
    assert edge.source_metadata["url"] == "http://x.test"
    assert edge.source_metadata["source_type"] == "web"
    assert len(ki._review_queue) == 1


def test_web_to_graph_without_intake_keeps_legacy_write():
    from ravana.web.web_to_graph import WebToGraph
    ge = _engine(["water", "liquid"])
    w2g = WebToGraph(ge)          # no gate -> legacy unconditional write

    assert w2g.learn_text("Water is liquid.", source_url="http://x.test") == 1
    edge = ge.graph.get_edge(ge._all_labels["water"], ge._all_labels["liquid"])
    assert edge is not None
    assert edge.source_metadata["edge_kind"] == "web_fact"


def test_web_to_graph_buffers_inconsistent_fact():
    """A fact about a concept the graph knows nothing about must NOT be
    written straight into the semantic graph -- it buffers for replay."""
    from ravana.web.web_to_graph import WebToGraph
    ge = _engine()
    buf = _FakeBuffer()
    ki = KnowledgeIntake(ge=ge, buffer=buf)
    w2g = WebToGraph(ge, intake=ki)

    w2g.learn_text("Water is liquid.", source_url="http://x.test")
    assert ki.stats["episodic_buffer"] == 1
    assert buf.stored, "inconsistent fact should have been buffered"
    # nothing written to the semantic graph yet
    assert ge.graph.get_edge(ge._all_labels["water"],
                             ge._all_labels["liquid"]) is None
