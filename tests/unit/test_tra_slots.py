"""Unit tests for the default-OFF typed role-slot write/read primitives.

Pure and fake-based: no engine boot, no live store, no network, no benchmark.
The write fake mirrors the production ``KnowledgeIntake.intake`` call shape
(Fact + user Provenance -> a provenance-stamped graph edge); the read tests use
the real ``schema_induction`` functions over fake graph edges. Each test that
enables the flag restores it afterwards.
"""
from __future__ import annotations

import inspect
import os
import pickle
from types import SimpleNamespace

import pytest

from ravana.chat import schema_induction as si
from ravana.chat.response_gen import ResponseGenMixin
from ravana.learn.intake import Fact, Provenance

TYPED_OPTS = ["weather phenomenon", "civic event", "structured whole", "similar form"]
DISJOINT_OPTS = ["apple", "banana", "chair", "table"]
TEACH = [
    "mist is a weather phenomenon.",
    "hail is a weather phenomenon.",
    "drizzle is a weather phenomenon.",
    "election is a civic event.",
    "ceremony is a civic event.",
    "contract is a civic event.",
]


# ── minimal fake graph / intake ───────────────────────────────────────────────
class _Node:
    def __init__(self, nid, label):
        self.id = nid
        self.label = label


class _Edge:
    def __init__(self, relation_type="semantic", confidence=0.9, metadata=None):
        self.relation_type = relation_type
        self.confidence = confidence
        self.source_metadata = dict(metadata or {})


class _Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self._next = 0

    def add_node(self, vector=None, label=""):
        nid = self._next
        self._next += 1
        node = _Node(nid, label)
        self.nodes[nid] = node
        return node

    def get_node(self, nid):
        return self.nodes.get(nid)

    def get_edge(self, sid, tid):
        return self.edges.get((sid, tid))

    def put_edge(self, sid, tid, edge):
        self.edges[(sid, tid)] = edge


class _FakeIntake:
    """Mirrors the intake call shape without booting an engine."""

    def __init__(self, graph):
        self.graph = graph
        self.calls = []

    def intake(self, fact, provenance=None, salience=1.0):
        self.calls.append((fact, provenance, salience))
        nid = {}
        for key, node in self.graph.nodes.items():
            nid[node.label] = key
        sid, oid = nid.get(fact.subject), nid.get(fact.obj)
        if sid is None or oid is None:
            return SimpleNamespace(route="rejected", reason="missing_node")
        md = dict(provenance.as_metadata()) if provenance is not None else {}
        md["edge_kind"] = ("curated" if getattr(provenance, "source_type", "")
                           == "user" else "inferred")
        self.graph.put_edge(sid, oid, _Edge(
            relation_type="semantic", confidence=fact.confidence, metadata=md))
        return SimpleNamespace(route="novel_write", reason="fake_intake")


class _WriteHarness(ResponseGenMixin):
    """Binds the real mixin write/read methods onto fake graph/intake."""

    def __init__(self, graph=None):
        self.graph = graph or _Graph()
        self._intake = _FakeIntake(self.graph)
        self._concept_keywords = {}
        self._concept_labels = set()
        self._trace_enabled = False

    def _get_knowledge_intake(self):
        return self._intake

    def _glove_vector(self, label):
        return None


@pytest.fixture(autouse=True)
def _flags():
    old = {k: os.environ.get(k) for k in ("RAVANA_TRA_SLOTS", "RAVANA_SCHEMA")}
    os.environ.pop("RAVANA_TRA_SLOTS", None)
    os.environ.pop("RAVANA_SCHEMA", None)
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _teach(harness):
    return sum(harness._consolidate_asserted_typed_slots(t) for t in TEACH)


def _views(harness):
    return si.typed_slot_views_from_graph(harness.graph)


# ── write half ────────────────────────────────────────────────────────────────
def test_write_six_is_a_assertions_through_intake_api():
    os.environ["RAVANA_TRA_SLOTS"] = "1"
    h = _WriteHarness()
    assert _teach(h) == 6
    assert len(h._intake.calls) == 6
    for fact, prov, salience in h._intake.calls:
        assert isinstance(fact, Fact) and isinstance(prov, Provenance)
        assert fact.relation == "is_a" and fact.confidence == 0.9
        assert prov.source == "conversation" and prov.source_type == "user"
        assert salience == 1.0
    stamped = [
        e for e in h.graph.edges.values()
        if e.source_metadata.get("slot_writer") == "tra_typed_slot"]
    assert len(stamped) == 6
    for e in stamped:
        assert e.relation_type == "is_a"
        assert e.source_metadata["slot_label"] == "type"
        assert e.source_metadata["type_label"]
        assert e.source_metadata["edge_kind"] == "curated"
        assert e.source_metadata["source"] == "conversation"
        assert e.source_metadata["source_type"] == "user"
        assert not e.source_metadata.get("url")
        assert si._is_trusted_edge(e)


def test_write_flag_off_is_byte_identical():
    h = _WriteHarness()
    assert _teach(h) == 0
    assert h._intake.calls == []
    assert h.graph.edges == {}
    assert h.graph.nodes == {}


def test_write_parser_refuses_hedge_negation_question_and_other_relations():
    os.environ["RAVANA_TRA_SLOTS"] = "1"
    h = _WriteHarness()
    refused = [
        "mist might be a weather phenomenon.",
        "mist is not a weather phenomenon.",
        "is mist a weather phenomenon?",
        "mist causes rain.",
        "mist is part of a storm.",
        "",
    ]
    assert sum(h._consolidate_asserted_typed_slots(t) for t in refused) == 0
    assert h._intake.calls == []
    assert h.graph.edges == {}


def test_cold_reload_simulation_keeps_all_six_slots():
    os.environ["RAVANA_TRA_SLOTS"] = "1"
    h = _WriteHarness()
    assert _teach(h) == 6
    reloaded = pickle.loads(pickle.dumps(h))  # store roundtrip, no engine boot
    out = si.slot_keyed_forced_choice(
        si.typed_slot_views_from_graph(reloaded.graph), "mist", TYPED_OPTS)
    assert out["pick"] == "weather phenomenon"
    stamped = [e for e in reloaded.graph.edges.values()
               if e.source_metadata.get("slot_writer") == "tra_typed_slot"]
    assert len(stamped) == 6


# ── read half ─────────────────────────────────────────────────────────────────
def _positive_harness():
    os.environ["RAVANA_TRA_SLOTS"] = "1"
    h = _WriteHarness()
    _teach(h)
    return h


def test_read_positive_pick_is_permutation_stable():
    views = _views(_positive_harness())
    picks = [
        si.slot_keyed_forced_choice(views, "mist", TYPED_OPTS)["pick"],
        si.slot_keyed_forced_choice(views, "mist", list(reversed(TYPED_OPTS)))["pick"],
        si.slot_keyed_forced_choice(
            views, "mist", ["civic event", "weather phenomenon", "similar form",
                            "structured whole"])["pick"],
    ]
    assert picks == ["weather phenomenon"] * 3
    assert si.slot_keyed_forced_choice(
        views, "election", TYPED_OPTS)["pick"] == "civic event"


def test_read_c10_disjoint_options_fail_closed():
    out = si.slot_keyed_forced_choice(
        _views(_positive_harness()), "mist", DISJOINT_OPTS)
    assert out["pick"] is None
    assert out["reason"] == "no_trusted_typed_slot_above_floor"
    assert out["eligible"] == []


def test_read_unrelated_subject_fails_closed():
    assert si.slot_keyed_forced_choice(
        _views(_positive_harness()), "zeppelin", TYPED_OPTS)["pick"] is None


def test_read_untrusted_typed_edge_cannot_seed():
    h = _positive_harness()
    obj = h._ensure_assertion_node("weather phenomenon")
    zep = h._ensure_assertion_node("zeppelin")
    forged = _Edge(relation_type="is_a", confidence=0.99, metadata={
        "source": "inference", "source_type": "inference",
        "extractor": "forged", "edge_kind": "inferred",
        "type_label": "weather phenomenon", "slot_label": "type",
    })
    h.graph.put_edge(zep, obj, forged)
    views = _views(h)
    assert not si._is_trusted_edge(forged)
    assert si.slot_keyed_forced_choice(views, "zeppelin", TYPED_OPTS)["pick"] is None
    assert si.slot_keyed_forced_choice(views, "hail", TYPED_OPTS)["pick"] is not None


def test_read_tie_and_below_floor_fail_closed():
    h = _positive_harness()
    # Two trusted typed edges for mist -> equal confidences => no strict winner.
    obj = h._ensure_assertion_node("civic event")
    sid = h._ensure_assertion_node("mist")
    second = _Edge(relation_type="is_a", confidence=0.9, metadata={
        "source": "conversation", "source_type": "user", "edge_kind": "curated",
        "type_label": "civic event", "slot_label": "type",
        "slot_writer": "tra_typed_slot",
    })
    h.graph.put_edge(sid, obj, second)
    for e in h.graph.edges.values():
        if e.source_metadata.get("slot_writer") == "tra_typed_slot":
            e.confidence = 0.5
    tie = si.slot_keyed_forced_choice(_views(h), "mist", TYPED_OPTS)
    assert tie["pick"] is None and tie["reason"] == "strict_margin_not_met"
    for e in h.graph.edges.values():
        if e.source_metadata.get("slot_writer") == "tra_typed_slot":
            e.confidence = 0.10
    below = si.slot_keyed_forced_choice(_views(h), "mist", TYPED_OPTS)
    assert below["pick"] is None
    assert below["reason"] == "no_trusted_typed_slot_above_floor"


def test_read_uses_only_stored_confidence_and_no_gold_answer():
    sig = inspect.signature(si.slot_keyed_forced_choice)
    assert not ({"gold", "gold_answer", "answer", "world_model", "item",
                 "item_metadata", "glove"} & set(sig.parameters))
    h = _positive_harness()
    views = _views(h)
    # Stored confidence is the score: raising it does not change the pick,
    # but an unstamped option (however "plausible") can never be selected.
    base = si.slot_keyed_forced_choice(views, "mist", TYPED_OPTS)
    assert base["all_slot_scores"]["weather phenomenon"] > 0.0
    assert si.slot_keyed_forced_choice(
        views, "mist", ["apple", "banana", "chair", "table"])["pick"] is None


# ── integration hook: flag equivalence + duck-typed safety ───────────────────
def test_hook_flag_off_is_byte_identical_and_flag_on_picks():
    h = _positive_harness()
    q = ("Considering the examples above, what is true of mist? "
         "Options: apple, banana, chair, table. Answer briefly.")
    positive_q = ("Considering the examples above, what is true of mist? "
                  "Options: weather phenomenon, civic event, structured whole, "
                  "similar form. Answer briefly.")
    os.environ.pop("RAVANA_TRA_SLOTS", None)
    assert ResponseGenMixin._schema_candidate(h, q) is None
    assert ResponseGenMixin._schema_candidate(h, positive_q) is None
    os.environ["RAVANA_TRA_SLOTS"] = "1"
    # Disjoint options still fail closed with the new flag on (C10).
    assert ResponseGenMixin._schema_candidate(h, q) is None
    assert ResponseGenMixin._schema_candidate(h, positive_q) == "weather phenomenon"


def test_hook_is_independent_of_ravana_schema_flag():
    h = _positive_harness()
    positive_q = ("what is true of mist? Options: weather phenomenon, "
                  "civic event, structured whole, similar form.")
    os.environ["RAVANA_SCHEMA"] = "1"
    assert ResponseGenMixin._schema_candidate(h, positive_q) == "weather phenomenon"
    os.environ.pop("RAVANA_SCHEMA", None)
    assert ResponseGenMixin._schema_candidate(h, positive_q) == "weather phenomenon"


def test_hook_duck_typed_fake_without_hook_stays_inert():
    class _OldEngine:
        graph = None

    os.environ["RAVANA_TRA_SLOTS"] = "1"
    assert ResponseGenMixin._schema_candidate(
        _OldEngine(), "q? Options: a, b", "a") is None


def test_no_cross_turn_state_between_calls():
    h = _positive_harness()
    q = "what is true of mist? Options: weather phenomenon, civic event."
    first = si.slot_keyed_forced_choice(_views(h), "mist", TYPED_OPTS)
    ResponseGenMixin._schema_candidate(h, q)
    second = si.slot_keyed_forced_choice(_views(h), "mist", TYPED_OPTS)
    assert first == second
