"""Fake-based tests for H3 trusted-edge forced-choice answer construction.

No engine boot, no process_turn, no network, no thresholds. The pure selector
and the graph adapter live in response_gen.py; every edge/graph here is a
plain dataclass stand-in. The env flag is restored after each test.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict

import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(PROJ, "ravana", "src"), PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat import response_gen as rg  # noqa: E402


QUERY = ("Considering the examples above, what is true of thaw? "
         "Options: sunset, typhoon, mist, hail. Answer briefly.")
CANDS = ["sunset", "typhoon", "mist", "hail"]


@pytest.fixture(autouse=True)
def _flag_off():
    old = os.environ.get(rg.EDGE_CHOICE_ENV)
    os.environ.pop(rg.EDGE_CHOICE_ENV, None)
    yield
    if old is None:
        os.environ.pop(rg.EDGE_CHOICE_ENV, None)
    else:
        os.environ[rg.EDGE_CHOICE_ENV] = old


def E(src, tgt, rel="causal", kind="curated", source="conversation"):
    return rg.CandidateEdge(src, tgt, rel, kind, source)


class FakeSelf:
    """Bare object exposing only the attributes the mixin helper reads."""

    def __init__(self, graph=None, keywords=None):
        self.graph = graph
        self._concept_keywords = keywords or {}


@dataclass
class FakeNode:
    label: str


@dataclass
class FakeGraphEdge:
    relation_type: str = "causal"
    source_metadata: Dict[str, Any] = field(default_factory=dict)


class FakeGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes                 # id -> node (dict, as in production)
        self.edges = edges                 # (id, id) -> edge
        self._nodes = nodes

    def get_node(self, nid):
        return self._nodes.get(nid)


def taught_graph():
    """mist -> thaw, curated/conversation, resolved through integer node ids."""
    nodes = {1: FakeNode("mist"), 2: FakeNode("thaw")}
    edges = {(1, 2): FakeGraphEdge(
        relation_type="causal",
        source_metadata={"edge_kind": "curated", "source": "conversation"})}
    return FakeGraph(nodes, edges), {"thaw": [2]}


# ── pure selector ──────────────────────────────────────────────────────────

def test_trusted_incoming_causal_edge_selects_candidate():
    os.environ[rg.EDGE_CHOICE_ENV] = "1"
    assert rg.select_trusted_candidate("thaw", CANDS, [E("mist", "thaw")]) == "mist"


def test_physical_cause_relation_is_admitted():
    os.environ[rg.EDGE_CHOICE_ENV] = "1"
    e = E("mist", "thaw", rel="physical_cause")
    assert rg.select_trusted_candidate("thaw", CANDS, [e]) == "mist"


def test_outgoing_edge_selects_nothing():
    os.environ[rg.EDGE_CHOICE_ENV] = "1"
    assert rg.select_trusted_candidate("thaw", CANDS, [E("thaw", "mist")]) is None


def test_unknown_provenance_or_relation_selects_nothing():
    os.environ[rg.EDGE_CHOICE_ENV] = "1"
    # high weight / unknown kind and source must NOT be admitted
    bad = [
        E("hail", "thaw", kind="co_occurrence", source=""),
        E("hail", "thaw", kind="curated", source="web"),
        E("hail", "thaw", rel="semantic"),
        E("hail", "thaw", kind="unknown", source="conversation"),
    ]
    for e in bad:
        assert rg.select_trusted_candidate("thaw", CANDS, [e]) is None


def test_multiple_divergent_candidates_fail_closed():
    os.environ[rg.EDGE_CHOICE_ENV] = "1"
    edges = [E("mist", "thaw"), E("hail", "thaw")]
    assert rg.select_trusted_candidate("thaw", CANDS, edges) is None


def test_convergent_edges_select_one_candidate():
    os.environ[rg.EDGE_CHOICE_ENV] = "1"
    # two independently taught premises converging on the same candidate
    edges = [E("mist", "thaw"), E("mist", "thaw", kind="curriculum")]
    assert rg.select_trusted_candidate("thaw", CANDS, edges) == "mist"


def test_no_candidate_match_gives_no_answer():
    os.environ[rg.EDGE_CHOICE_ENV] = "1"
    # trusted incoming edge from a node NOT in the option list
    assert rg.select_trusted_candidate("thaw", CANDS,
                                       [E("gale", "thaw")]) is None
    assert rg.select_trusted_candidate("thaw", [], [E("mist", "thaw")]) is None


def test_flag_off_is_byte_identical_no_answer():
    assert not rg.edge_choice_enabled()
    fake = FakeSelf(*taught_graph())
    assert rg.ResponseGenMixin._trusted_edge_candidate(fake, QUERY, "thaw") is None


# ── forced-choice structure + graph adapter (generic, not item-specific) ───

def test_parse_forced_choice_is_structural():
    q, opts = rg.parse_forced_choice(QUERY)
    assert "thaw" in q and "Options" not in q
    assert opts == CANDS
    # no forced-choice frame -> untouched, no candidates
    q2, opts2 = rg.parse_forced_choice("why does ice melt")
    assert q2 == "why does ice melt" and opts2 == []


def test_graph_adapter_resolves_ids_and_direction():
    graph, kw = taught_graph()
    views = rg.trusted_candidate_edges_from_graph(graph, "thaw", kw)
    assert len(views) == 1
    assert views[0].source_label == "mist" and views[0].target_label == "thaw"
    os.environ[rg.EDGE_CHOICE_ENV] = "1"
    assert rg.select_trusted_candidate("thaw", CANDS, views) == "mist"


def test_adapter_ignores_dangling_keyword_ids():
    graph, kw = taught_graph()
    kw["thaw"] = [999]                      # stale label index (skill section 5)
    views = rg.trusted_candidate_edges_from_graph(graph, "thaw", kw)
    assert views == []                      # no crash, no candidate from a dead id


# ── wiring behavior through the mixin helper ──────────────────────────────

def test_mixin_helper_admits_then_fails_closed_next_turn_no_state_leak():
    os.environ[rg.EDGE_CHOICE_ENV] = "1"
    graph, kw = taught_graph()
    fake = FakeSelf(graph, kw)
    before = dict(vars(fake))
    assert rg.ResponseGenMixin._trusted_edge_candidate(fake, QUERY, "thaw") == "mist"
    # Next turn: same object, graph no longer carries the taught edge.
    fake.graph = FakeGraph({1: FakeNode("mist"), 2: FakeNode("thaw")}, {})
    assert rg.ResponseGenMixin._trusted_edge_candidate(fake, QUERY, "thaw") is None
    # No persistent state was written by the hook.
    assert set(vars(fake)) == set(before)


def test_non_forced_choice_query_never_diverted():
    os.environ[rg.EDGE_CHOICE_ENV] = "1"
    fake = FakeSelf(*taught_graph())
    # Existing chain-articulation / free-text path must not be touched: no
    # Options: frame -> no candidate, even with a trusted edge in the graph.
    assert rg.ResponseGenMixin._trusted_edge_candidate(
        fake, "why does thaw happen", "thaw") is None


def test_subject_not_in_question_is_ignored():
    os.environ[rg.EDGE_CHOICE_ENV] = "1"
    fake = FakeSelf(*taught_graph())
    # A stale/wrong subject that does not occur in the question is discarded;
    # the generic question fallback finds no matching edge -> no answer.
    assert rg.ResponseGenMixin._trusted_edge_candidate(
        fake, QUERY, "rainbow") is None


def test_phrase_subject_falls_back_to_head_noun():
    """Extractors can yield 'true thaw' for 'what is true of thaw?'."""
    os.environ[rg.EDGE_CHOICE_ENV] = "1"
    fake = FakeSelf(*taught_graph())
    assert rg.ResponseGenMixin._trusted_edge_candidate(
        fake, QUERY, "true thaw") == "mist"
