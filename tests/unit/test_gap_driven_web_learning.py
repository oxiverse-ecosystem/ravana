"""Gap -> targeted web read -> typed causal graph -> next-answer chain.

Brain-grounded design (full citations in docs/GAP_DRIVEN_WEB_LEARNING.md):
  Loewenstein (1994, Psychological Bulletin) — a detected information gap drives
    a TARGETED read.
  Schacter & Addis (2007, Phil Trans R Soc B) — a mechanism is >=2 retrieved
    causal hops recombined into a chain.
  Friston (2010, Nat Rev Neurosci) — exploration is bounded/fail-open.
  Gerstenberg (2024, Trends Cogn Sci) — learned causal edges need provenance and
    evidence-scaled strength to support counterfactual simulation.

These tests use a LIGHTWEIGHT host (no ~200 s cold engine boot) with a stub
search engine, so they are deterministic and need no network.
"""
import os
import pickle
import sys

import pytest

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(_PROJ, "ravana", "src"),
           os.path.join(_PROJ, "ravana_ml", "src"),
           os.path.join(_PROJ, "ravana-v2", "src"), _PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.web_learning import WebLearningMixin  # noqa: E402
from ravana.chat.models import FailedQuery  # noqa: E402


CAUSE_A = "Water vapour condenses into clouds."
CAUSE_B = "Clouds release rain."


class StubSearch:
    """Deterministic 'web': returns the given hit dicts, records the query."""

    def __init__(self, hits):
        self.hits = list(hits)
        self.queries = []

    def search(self, query, max_results=3):
        self.queries.append((query, max_results))
        return self.hits[:max_results]


class GapHost(WebLearningMixin):
    """Minimal substrate for WebLearningMixin.close_mechanism_gap."""

    def __init__(self, hits):
        from ravana.graph.engine import GraphEngine
        ge = GraphEngine(dim=64, glove_vecs=None)
        self.graph = ge.graph
        self.dim = 64
        self._glove_vector = lambda w: None
        self._concept_keywords = {}
        self.search_engine = StubSearch(hits)
        self._impossible_queries = []
        self._pending_learning_queue = []
        self._web_to_graph = None
        self._trace_enabled = False
        self._network_available = True
        self.turn_count = 0
        self._gap_read_subjects = set()
        self._gap_read_count = 0
        self._promote_min_sources = 1


def _host_rain(monkeypatch):
    """Two distinct sources, one causal hop each: a 2-hop chain when combined."""
    monkeypatch.delenv("RAVANA_OFFLINE", raising=False)
    return GapHost([
        {"title": "Evaporation", "content": CAUSE_A, "url": "https://example.org/a"},
        {"title": "Precipitation", "content": CAUSE_B, "url": "https://example.org/b"},
    ])


def _causal_edges(host):
    return [(s, t, e) for (s, t), e in host.graph.edges.items()
            if (e.relation_type or "").lower() == "causal"]


class TestGapTriggersLearnableChain:
    def test_gap_yields_two_sourced_hops_and_resolves(self, monkeypatch):
        """The property under test: a recorded mechanism gap drives a targeted
        read that writes >=2 provenanced causal hops forming a chain."""
        host = _host_rain(monkeypatch)
        host._impossible_queries.append(FailedQuery(
            subject="rain", query="why does it rain", resolved=False))
        host._pending_learning_queue.append("rain")

        # RAW before: no causal edges, gap open.
        before = _causal_edges(host)
        print("BEFORE causal edges:", len(before),
              "| gap resolved:", host._impossible_queries[0].resolved)

        out = host.close_mechanism_gap("rain", "why does it rain")

        # RAW after: print every edge with its provenance so the score is
        # auditable against the substrate, not just the returned status.
        print("AFTER result:", out)
        for s, t, e in _causal_edges(host):
            md = getattr(e, "source_metadata", None) or {}
            print("  EDGE", host.graph.get_node(s).label, "->",
                  host.graph.get_node(t).label,
                  "| rel", e.relation_type, "| w", round(e.weight, 3),
                  "| c", round(e.confidence, 3),
                  "| kind", md.get("edge_kind"),
                  "| by", md.get("acquired_by"),
                  "| sources", md.get("converging_sources"))

        assert out["status"] == "learned"
        assert out["sources"] == 2
        assert out["causal_hops"] >= 2
        assert out["chain_hops"] >= 2
        assert host._impossible_queries[0].resolved is True
        assert "rain" not in host._pending_learning_queue

        # The read is TARGETED (Loewenstein 1994): the query names the cause.
        q, cap = host.search_engine.queries[0]
        assert "causes" in q and "mechanism" in q
        assert cap == 2  # Friston 2010: bounded

        # Every new edge carries traceable provenance and evidence-scaled
        # strength (Gerstenberg 2024).
        edges = _causal_edges(host)
        assert len(edges) >= 2
        for _, _, e in edges:
            md = getattr(e, "source_metadata", None) or {}
            assert md.get("edge_kind") == "web_fact"
            assert md.get("acquired_by") == "mechanism_gap"
            assert md.get("method") == "mechanism_gap_read"
            assert md.get("source_url", "").startswith("https://example.org/")
            assert md.get("converging_sources", 0) >= 1
            assert 0.0 < e.confidence <= 1.0

    def test_next_turn_can_resolve_the_learned_chain(self, monkeypatch):
        """Improvement must be due to LEARNING, not a probe change: the chain
        is enumerated from the graph the loop just wrote, with the same
        provenance gate the real emitter uses."""
        host = _host_rain(monkeypatch)
        out = host.close_mechanism_gap("rain", "why does it rain")
        assert out["status"] == "learned"

        from ravana.learn.explanation import (
            enumerate_chains, grade_chain, assertable_chain)
        seeds = host._concept_keywords.get("rain", [])
        assert seeds, "learned 'rain' node was not indexed for the next turn"
        chains = enumerate_chains(seeds, host.graph,
                                  direction="backward", min_steps=2,
                                  max_steps=4, limit=3)
        print("NEXT-TURN chains:",
              [(c.labels, grade_chain(c)["verdict"], assertable_chain(c))
               for c in chains])
        ok = [(c, grade_chain(c)) for c in chains
              if grade_chain(c)["verdict"] == "assert" and assertable_chain(c)]
        assert ok, "learned chain is not assertable on the next turn"
        best = ok[0][0]
        narr = ", and ".join(
            f"{best.labels[i]} leads to {best.labels[i + 1]}"
            for i in range(len(best.labels) - 1))
        print("NEXT-TURN narrative:", narr)
        assert "water vapour" in narr and "clouds" in narr and "rain" in narr

    def test_provenance_survives_pickle_persist_path(self, monkeypatch):
        """eng.save() pickles the graph (authoritative on boot); the stamp must
        survive that round trip (db mirror is written from the same graph)."""
        host = _host_rain(monkeypatch)
        host.close_mechanism_gap("rain", "why does it rain")
        reloaded = pickle.loads(pickle.dumps(host.graph))
        stamped = []
        for e in reloaded.edges.values():
            md = getattr(e, "source_metadata", None) or {}
            if md.get("acquired_by") == "mechanism_gap":
                stamped.append(md)
        print("PERSISTED gap-stamped edges:", stamped)
        assert len(stamped) >= 2
        for md in stamped:
            assert md.get("edge_kind") == "web_fact"
            assert md.get("source_url")
            assert md.get("converging_sources", 0) >= 1


class TestGapLoopIsBoundedAndFailOpen:
    def test_offline_short_circuits_without_reading(self, monkeypatch):
        monkeypatch.setenv("RAVANA_OFFLINE", "1")
        # Build directly: _host_rain() clears RAVANA_OFFLINE for its own tests.
        host = GapHost([
            {"title": "Evaporation", "content": CAUSE_A, "url": "https://example.org/a"},
            {"title": "Precipitation", "content": CAUSE_B, "url": "https://example.org/b"},
        ])
        out = host.close_mechanism_gap("rain", "why does it rain")
        print("OFFLINE result:", out, "| searches:", host.search_engine.queries)
        assert out["status"] == "offline"
        assert out["causal_hops"] == 0
        assert host.search_engine.queries == []
        assert _causal_edges(host) == []

    def test_same_subject_is_not_read_twice(self, monkeypatch):
        host = _host_rain(monkeypatch)
        first = host.close_mechanism_gap("rain", "why does it rain")
        second = host.close_mechanism_gap("rain", "why does it rain")
        print("FIRST:", first["status"], "| SECOND:", second["status"])
        assert first["status"] == "learned"
        assert second["status"] == "already_attempted"
        assert len(host.search_engine.queries) == 1

    def test_session_read_budget_caps_exploration(self, monkeypatch):
        monkeypatch.delenv("RAVANA_OFFLINE", raising=False)
        host = GapHost([])  # no hits -> no learning, but reads still counted
        statuses = [host.close_mechanism_gap(t)["status"]
                    for t in ("a", "b", "c", "d", "e")]
        print("BUDGET statuses:", statuses, "| searches:", len(host.search_engine.queries))
        assert statuses == ["no_chain"] * 4 + ["budget"]
        assert len(host.search_engine.queries) == 4  # capped, no infinite loop

    def test_single_hop_does_not_close_gap(self, monkeypatch):
        monkeypatch.delenv("RAVANA_OFFLINE", raising=False)
        host = GapHost([
            {"title": "One", "content": CAUSE_A, "url": "https://example.org/only"},
        ])
        host._impossible_queries.append(FailedQuery(
            subject="rain", query="why does it rain", resolved=False))
        out = host.close_mechanism_gap("rain", "why does it rain")
        print("SINGLE-HOP result:", out)
        assert out["status"] == "no_chain"
        assert out["causal_hops"] == 1
        assert host._impossible_queries[0].resolved is False

    def test_converging_sources_scale_strength(self, monkeypatch):
        """Gerstenberg 2024: strength tracks convergent evidence, so an edge
        seen in two sources is stronger than one seen in one source."""
        monkeypatch.delenv("RAVANA_OFFLINE", raising=False)
        one = GapHost([
            {"title": "A", "content": CAUSE_A, "url": "https://example.org/a"},
            {"title": "B", "content": CAUSE_B, "url": "https://example.org/b"},
        ])
        two = GapHost([
            {"title": "A1", "content": CAUSE_A + " " + CAUSE_B,
             "url": "https://example.org/a1"},
            {"title": "A2", "content": CAUSE_A + " " + CAUSE_B,
             "url": "https://example.org/a2"},
        ])
        one.close_mechanism_gap("rain")
        two.close_mechanism_gap("rain")

        def conf_of(h, label_from, label_to):
            ids = {nid: (n.label or "").lower() for nid, n in h.graph.nodes.items()}
            for (s, t), e in h.graph.edges.items():
                if (ids.get(s), ids.get(t)) == (label_from, label_to):
                    return e.confidence, (e.source_metadata or {}).get("converging_sources")
            return None, None

        c1, k1 = conf_of(one, "water vapour", "clouds")
        c2, k2 = conf_of(two, "water vapour", "clouds")
        print(f"ONE source: conf={c1} k={k1} | TWO sources: conf={c2} k={k2}")
        assert k1 == 1 and k2 == 2
        assert c2 > c1


class TestNextTurnUsesChain:
    """The loop is not closed until the NEXT answer can use the chain."""

    def test_mechanism_chain_available_flips_after_learning(self, monkeypatch):
        """The internal-knowledge consult must stop shadowing a learned
        mechanism with a stored definition (Schacter & Addis 2007)."""
        import types
        from ravana.chat.engine_self_query import SelfQueryMixin
        host = _host_rain(monkeypatch)
        host._mechanism_chain_available = types.MethodType(
            SelfQueryMixin._mechanism_chain_available, host)
        assert host._mechanism_chain_available("why does it rain", "rain") is False
        host.close_mechanism_gap("rain", "why does it rain")
        assert host._mechanism_chain_available("why does it rain", "rain") is True

    def test_gated_chain_exempt_from_final_emit_guard(self):
        """A provenance-trusted chain is retrieved structure, not self-generated
        salad: the source-monitoring guard must pass it through (Johnson &
        Raye 1981; same rationale as web_direct_answer in the inherited guard).
        """
        from ravana.chat.engine import CognitiveChatEngine
        text = "water vapour leads to clouds, and clouds leads to rain"
        out = CognitiveChatEngine._final_emit_guard(
            object(), text, object(), "gated_explanation_chain")
        print("GUARD out:", repr(out))
        assert out == text
