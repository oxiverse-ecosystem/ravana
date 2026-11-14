"""Tests for the web->graph relation-type mapping.

MEASURED DEFECT: `WebToGraph.RELATION_TO_EDGE` mapped OpenIE's `"causes"` to
`"causes"`, but `ConceptEdge.RELATION_DEFAULT_WEIGHTS` has no `"causes"` key --
only `"causal"`. Every web-extracted causal fact was written under a relation
type the rest of the system does not recognise, so web learning silently
produced edges causal reasoning could not use.

These tests pin the invariant directly: every mapped target must be a
relation type the graph actually knows.
"""
import os
import sys

import pytest

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (PROJ, os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.web.web_to_graph import WebToGraph  # noqa: E402


def _known_relation_types():
    """Relation types the system recognises.

    There are TWO vocabularies, and conflating them is how the `causes` defect
    survived:

    * `ConceptEdge.RELATION_DEFAULT_WEIGHTS` -- the graph's own weight table
      (`causal`, `semantic`, `temporal`, `analogical`, ...). A type absent here
      gets no default weight, so ranking silently ignores it.
    * `graph_typing.TYPED_RELATION_TYPES` -- the closed set the plan route and
      the relation typers key on (`isa`, `has_property`, `capable_of`,
      `used_for`, `part_of`).

    A web-mapped target must be in at least one; `causal` is in the first.
    """
    types = set()
    try:
        from ravana_ml.graph import ConceptEdge
        types |= set(ConceptEdge.RELATION_DEFAULT_WEIGHTS)
    except Exception:
        pass
    try:
        from ravana.ontology.graph_typing import TYPED_RELATION_TYPES
        types |= set(TYPED_RELATION_TYPES)
    except Exception:
        pass
    if not types:
        pytest.skip("no relation-type vocabulary available")
    return types


class TestWebRelationMapping:
    def test_every_mapped_target_is_a_known_relation_type(self):
        """Scope: the CAUSAL family, which is what the plan route consumes.

        Deliberately NOT asserting the whole mapping. `is_a` and `located_in`
        are also absent from both vocabularies, but that is a separate defect
        (the plan route normalises `is_a` -> `isa` at
        `learn/linearization.py`, and `located_in` has its own construction),
        and folding it into this test would hide which fix closed which gap.
        """
        known = _known_relation_types()
        causal_family = {k: v for k, v in WebToGraph.RELATION_TO_EDGE.items()
                         if k in ("causes", "related_to")}
        unknown = {k: v for k, v in causal_family.items() if v not in known}
        assert not unknown, (
            f"causal-family targets the graph does not recognise: {unknown}")

    def test_causes_maps_to_causal(self):
        """The specific measured defect."""
        assert WebToGraph.RELATION_TO_EDGE["causes"] == "causal"

    def test_related_to_maps_to_semantic(self):
        assert WebToGraph.RELATION_TO_EDGE["related_to"] == "semantic"

    def test_causal_target_carries_a_default_weight(self):
        """A recognised type must also have a weight, or ranking ignores it."""
        known = _known_relation_types()
        from ravana_ml.graph import ConceptEdge
        w = ConceptEdge.RELATION_DEFAULT_WEIGHTS[
            WebToGraph.RELATION_TO_EDGE["causes"]]
        assert w > 0
        assert WebToGraph.RELATION_TO_EDGE["causes"] in known
