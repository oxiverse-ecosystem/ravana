"""Fake-based tests for the RSI schema-hypothesis wiring (H4).

No engine boot, no process_turn, no network, no pytest fixtures that construct
an engine. Every graph/edge here is a plain dataclass stand-in; the pure helpers
live in schema_induction.py and the hook in response_gen.py. Env flags are
restored after each test. The parent runs this module; the subagent does not.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(PROJ, "ravana", "src"), PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat import response_gen as rg  # noqa: E402
import schema_induction as si  # noqa: E402  (flat: pure module, no chat __init__)


QUERY = ("Considering the examples above, what is true of thaw? "
         "Options: sunset, typhoon, mist, hail. Answer briefly.")
CANDS = ["sunset", "typhoon", "mist", "hail"]


@pytest.fixture(autouse=True)
def _flags_off():
    """Both flags unset for every test unless a test opts in explicitly."""
    saved = {k: os.environ.get(k) for k in (si.ENV_FLAG, si.SCHEMA_ENV_FLAG,
                                            rg.EDGE_CHOICE_ENV)}
    for k in saved:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ── fakes ───────────────────────────────────────────────────────────────────

@dataclass
class FakeNode:
    label: str
    vector: Any = None


@dataclass
class FakeGraphEdge:
    relation_type: str = "causal"
    source_metadata: Dict[str, Any] = field(default_factory=dict)


class FakeGraph:
    """Minimal ConceptGraph stand-in: nodes/edges dicts keyed by integer id."""

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def get_node(self, nid):
        return self.nodes.get(nid)


def E(src, tgt, rel="causal", kind="curated", source="conversation"):
    return si.EdgeView(src, rel, tgt, kind, source, 0.9)


class FakeSelf:
    """Bare object exposing only attributes the mixin hooks read."""

    def __init__(self, graph=None, keywords=None):
        self.graph = graph
        self._concept_keywords = keywords or {}


def _v(*xs):
    return np.array(xs, dtype=float)


def demo_graph(with_direct_edge=False, vectors=True):
    """Two converged CAUSE demonstrations; thaw is an UNSEEN subject.

    gale->rainbow and shower->climate are trusted curated/conversation edges.
    Vectors make `mist` the strict, above-floor associate of `thaw`; the other
    candidates are either below the instance floor or unassociated.
    """
    vecs = {
        "gale": _v(1, 0, 0), "rainbow": _v(1, 0.5, 0),
        "shower": _v(0, 1, 0), "climate": _v(0, 1, 0.5),
        "thaw": _v(1, 1, 0), "mist": _v(1, 1, 0.1),
        "sunset": _v(1, 0, 0), "typhoon": _v(0, 1, 0), "hail": _v(0, 0, 1),
    } if vectors else {}
    nodes = {
        1: FakeNode("gale", vecs.get("gale")), 2: FakeNode("rainbow", vecs.get("rainbow")),
        3: FakeNode("shower", vecs.get("shower")), 4: FakeNode("climate", vecs.get("climate")),
        5: FakeNode("thaw", vecs.get("thaw")), 6: FakeNode("mist", vecs.get("mist")),
        7: FakeNode("hail", vecs.get("hail")), 8: FakeNode("sunset", vecs.get("sunset")),
        9: FakeNode("typhoon", vecs.get("typhoon")),
    }
    edges = {
        (1, 2): FakeGraphEdge("causal", {"edge_kind": "curated", "source": "conversation"}),
        (3, 4): FakeGraphEdge("causal", {"edge_kind": "curated", "source": "conversation"}),
    }
    if with_direct_edge:
        edges[(6, 5)] = FakeGraphEdge(
            "causal", {"edge_kind": "curated", "source": "conversation"})
    kw = {name: [nid] for nid, n in nodes.items() for name in [n.label.lower()]}
    return FakeGraph(nodes, edges), kw


# ── 1/2/3: induction thresholds and relation match ─────────────────────────

def test_one_demo_forms_no_hypothesis():
    traces = si.traces_from_edges([E("gale", "rainbow")])
    assert len(traces) == 1
    assert si.induce_schema_hypotheses(traces) == []


def test_two_converged_demos_form_unassertable_hypothesis():
    traces = si.traces_from_edges([E("gale", "rainbow"), E("shower", "climate")])
    hyps = si.induce_schema_hypotheses(traces)
    assert len(hyps) == 1
    assert hyps[0].support == 2
    assert hyps[0].assertable is False
    assert set(hyps[0].instances) == {("gale", "rainbow"), ("shower", "climate")}


def test_duplicate_demo_does_not_inflate_support():
    traces = si.traces_from_edges([E("gale", "rainbow")] * 3)
    assert si.induce_schema_hypotheses(traces) == []


def test_mismatched_relations_form_no_hypothesis():
    traces = si.traces_from_edges([E("gale", "rainbow", rel="causal"),
                                   E("shower", "climate", rel="temporal")])
    assert si.induce_schema_hypotheses(traces) == []


def test_untrusted_edge_cannot_seed_hypothesis():
    traces = si.traces_from_edges([E("gale", "rainbow", kind="co_occurrence", source="")])
    assert traces == []
    assert si.induce_schema_hypotheses(traces) == []


# ── 4: unseen-subject candidate ranking ────────────────────────────────────

def test_unseen_subject_candidate_ranking():
    traces = si.traces_from_edges([E("gale", "rainbow"), E("shower", "climate")])
    hyps = si.induce_schema_hypotheses(traces)
    scores = {"sunset": 0.70, "typhoon": 0.70, "mist": 0.98, "hail": 0.0}
    instance_scores = {("gale", "rainbow"): 0.894, ("shower", "climate"): 0.894}
    assert si.rank_candidates_for_unseen_subject(
        "thaw", CANDS, hyps, traces, scores, instance_scores) == "mist"


def test_ranking_fails_closed_on_tie():
    traces = si.traces_from_edges([E("gale", "rainbow"), E("shower", "climate")])
    hyps = si.induce_schema_hypotheses(traces)
    assert si.rank_candidates_for_unseen_subject(
        "thaw", CANDS, hyps, traces,
        {"mist": 0.9, "hail": 0.9}, {("gale", "rainbow"): 0.5}) is None


def test_ranking_needs_source_admission_not_hypothesis_alone():
    traces = si.traces_from_edges([E("gale", "rainbow"), E("shower", "climate")])
    hyps = si.induce_schema_hypotheses(traces)
    # No observed traces of that relation -> "schemas propose, sources admit".
    assert si.rank_candidates_for_unseen_subject(
        "thaw", CANDS, hyps, [], {"mist": 0.99}) is None


# ── 5: directly taught edge wins over the hypothesis ──────────────────────

def test_directly_taught_edge_wins_hook_defers():
    os.environ[si.SCHEMA_ENV_FLAG] = "1"
    os.environ[rg.EDGE_CHOICE_ENV] = "1"  # H3 must actually decide for this test
    graph, kw = demo_graph(with_direct_edge=True)
    fake = FakeSelf(graph, kw)
    # The direct H3 hook decides; the schema hook must defer, never override.
    assert rg.ResponseGenMixin._trusted_edge_candidate(fake, QUERY, "thaw") == "mist"
    assert rg.ResponseGenMixin._schema_candidate(fake, QUERY, "thaw") is None


# ── 6: foil / negative rejection unaffected ───────────────────────────────

def test_foil_candidate_below_congruence_floor_is_rejected():
    traces = si.traces_from_edges([E("gale", "rainbow"), E("shower", "climate")])
    hyps = si.induce_schema_hypotheses(traces)
    # A demonstrated anchor is a candidate but is NOT congruent with the
    # unseen subject -- a naive surface matcher would wrongly accept it.
    assert si.rank_candidates_for_unseen_subject(
        "thaw", ["gale", "sunset"], hyps, traces,
        {"gale": 0.3, "sunset": 0.2},
        {("gale", "rainbow"): 0.894, ("shower", "climate"): 0.894}) is None


def test_negative_rule_no_trusted_demos_no_hypothesis():
    os.environ[si.SCHEMA_ENV_FLAG] = "1"
    # Negated demonstrations are never consolidated as trusted edges, so the
    # hook sees no traces -> no hypothesis -> no candidate -> hedge/reject.
    graph, kw = demo_graph()
    graph.edges.clear()
    fake = FakeSelf(graph, kw)
    assert rg.ResponseGenMixin._schema_candidate(fake, QUERY, "thaw") is None


# ── 7: flag-off is byte-identical ──────────────────────────────────────────

def test_flag_off_is_inert():
    assert si.schema_enabled() is False
    graph, kw = demo_graph()
    fake = FakeSelf(graph, kw)
    assert rg.ResponseGenMixin._schema_candidate(fake, QUERY, "thaw") is None


def test_old_declaration_flag_does_not_enable_schema_path():
    os.environ[si.ENV_FLAG] = "1"
    assert si.enabled() is True
    assert si.schema_enabled() is False
    graph, kw = demo_graph()
    assert rg.ResponseGenMixin._schema_candidate(
        FakeSelf(graph, kw), QUERY, "thaw") is None


# ── hook-level end-to-end + no cross-turn state ────────────────────────────

def test_hook_ranks_candidate_for_unseen_subject_end_to_end():
    os.environ[si.SCHEMA_ENV_FLAG] = "1"
    graph, kw = demo_graph()
    fake = FakeSelf(graph, kw)
    assert rg.ResponseGenMixin._schema_candidate(fake, QUERY, "thaw") == "mist"


def test_no_cross_turn_state():
    os.environ[si.SCHEMA_ENV_FLAG] = "1"
    graph, kw = demo_graph()
    fake = FakeSelf(graph, kw)
    before = set(vars(fake))
    assert rg.ResponseGenMixin._schema_candidate(fake, QUERY, "thaw") == "mist"
    # Next turn: same object, demonstrations removed -> no hypothesis, no leak.
    fake.graph = FakeGraph({nid: n for nid, n in graph.nodes.items()}, {})
    assert rg.ResponseGenMixin._schema_candidate(fake, QUERY, "thaw") is None
    assert set(vars(fake)) == before


def test_non_forced_choice_query_never_diverted():
    os.environ[si.SCHEMA_ENV_FLAG] = "1"
    graph, kw = demo_graph()
    fake = FakeSelf(graph, kw)
    # No Options: frame -> CAR chain articulation / free text is untouched.
    assert rg.ResponseGenMixin._schema_candidate(
        fake, "why does thaw happen", "thaw") is None
