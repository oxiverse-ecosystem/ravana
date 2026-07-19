"""
Phase 20b — fail-closed degenerate causal-narrative gate (L5).

Reproduces the surfaced garbage from the live engine run:
  - "if sun were different, it could lead to: claims would lead to results"
  - "humans comes before occur, and that in turn comes before known"
These are HRR/GloVe-wired artifacts (none-vector endpoints,
placeholder tails) that previously reached the user as fluent garbage.
The vLPFC->ACC->AI causal-reasoning coherence filter (Operskalski &
Barbey 2016) must WITHHOLD them so the caller falls through to
honest uncertainty.

No network / no full engine boot required.
"""
import sys, os
import pytest

_proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_proj, "ravana", "src"))

from ravana.chat.response_gen import ResponseGenMixin


def _check(text):
    # _causal_narrative_degenerate is a plain instance method that
    # only touches a class-level frozenset + regex, so calling it on a
    # bare instance (no engine boot) works.
    return ResponseGenMixin()._causal_narrative_degenerate(text)


def test_placeholder_chain_withheld():
    text = ("if sun were different, here's what I'd expect to follow: "
             "claims would lead to results; claims would lead to conditions.")
    deg, reason = _check(text)
    assert deg is True, f"placeholder chain must be withheld, got {reason!r}"
    assert "placeholder" in reason or "junk" in reason


def test_hrr_temporal_junk_withheld():
    text = "humans comes before occur, and that in turn comes before known."
    deg, reason = _check(text)
    assert deg is True, f"HRR temporal junk must be withheld, got {reason!r}"
    assert "junk" in reason


def test_real_causal_chain_allowed():
    text = ("if the sun disappeared, here's what I'd expect to follow: "
             "light would lead to no photosynthesis, which would lead to plants dying.")
    deg, reason = _check(text)
    assert deg is False, f"real chain wrongly withheld: {reason!r}"


def test_temporal_junk_withheld():
    # `_structured_fact_answer` temporal guard rejects spurious HRR
    # temporal edges whose terminal endpoint is a generic noun.
    from ravana.chat.response_gen import ResponseGenMixin as R
    inst = R()
    # generic-noun terminal endpoint -> refuse (fall through to web/uncertainty)
    out = inst._structured_fact_answer("humans", "temporal")
    assert out is None, f"temporal junk must be refused, got {out!r}"


def test_temporal_subanswer_junk_detects_ungrounded_endpoint():
    from ravana.chat.response_gen import ResponseGenMixin as R

    # Minimal graph double that returns a single node with a given edge set.
    class _Edge:
        def __init__(self, rt):
            self.relation_type = rt
    class _Node:
        label = "x"
    class _Graph:
        def __init__(self, edges):
            # edges: dict node_id -> list of (target_id, _Edge)
            self._edges = edges
            self.nodes = {0: _Node(), 1: _Node()}
        def get_outgoing(self, nid):
            return list(self._edges.get(nid, []))
    inst = R()
    inst.graph = _Graph({
        0: [(1, _Edge("comes_before"))],   # only a temporal artifact link
    })
    inst._concept_keywords = {"humans": [0], "animal": [1], "thing": [1]}
    # humans->animal only via a "comes_before" edge -> spurious -> junk
    assert inst._temporal_subanswer_junk("humans", "humans comes before animal") is True
    # generic-noun endpoint is always junk regardless of edge
    assert inst._temporal_subanswer_junk("humans", "humans comes before thing") is True
    # non-temporal text must never be flagged
    assert inst._temporal_subanswer_junk("sun", "sun would lead to earth") is False
    # a genuine predicate edge (causal) between the two -> keep
    inst.graph = _Graph({0: [(1, _Edge("causes"))]})
    assert inst._temporal_subanswer_junk("humans", "humans comes before animal") is False
    # unresolved nodes -> fail open (keep)
    inst._concept_keywords = {"humans": [0]}
    assert inst._temporal_subanswer_junk("humans", "humans comes before animal") is False
