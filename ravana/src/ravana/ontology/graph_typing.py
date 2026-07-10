"""Graph typing — inject ConceptNet typed edges into the ravana ConceptGraph.

WHY (brain grounding, see deferred item 1 in the gate audit):
The learned ravana graph encodes only ASSOCIATIVE edges (semantic/temporal/
causal — the Collins & Loftus spreading-activation "thematic" links). It is
missing the taxonomic + componential spokes the ATL hub needs (Binder; Damasio/
Barsalou convergence zones co-deploy BOTH a hierarchy and thematic links).
ConceptNet 5.7 supplies exactly those:
  * /r/IsA            -> taxonomic hierarchy  (relation_type="isa")
  * /r/HasProperty    -> componential feature (relation_type="has_property")
  * /r/CapableOf      -> ability feature      (relation_type="capable_of")
  * /r/UsedFor        -> function feature     (relation_type="used_for")

This module materializes those typed edges in the PERSISTENT ravana graph
(ravana_weights.db via storage/db.py) so chain_walker / DerivedOntology /
abstraction_engine can use them. The old "real graph has zero typed edges"
blocker is thereby removed: Path-2 (inheritance walk over real typed edges)
becomes possible.

DESIGN DECISIONS (faithful to the brief):
* EXPLICIT typed relation_type — never the generic "semantic".
* WEIGHT BY STRENGTH, not just presence: edge.weight = ConceptNet assertion
  weight (mean over assertions for that pair), clamped to (0,1].
* PREFER ISA TO THE NEAREST (most specific) parent, not the root: when several
  ConceptNet parents of a concept exist as ravana nodes, we drop any parent
  that is itself an IsA-ancestor of another existing parent (Rosch basic level
  — "robin is a bird" verified faster than "penguin is a bird"; Collins &
  Loftus 1975). Nearest parents also tend to carry the highest CN weights.
* Direction: ravana edge is concept -> parent/feature (outgoing), matching how
  the other inheritance walkers traverse (child -> ancestor / subject ->
  property holder).

The function is idempotent-ish: it skips edges already present, so re-running
only fills gaps. It is meant to be called once at bootstrap (and optionally
after loading the persisted graph) — see chat/engine.py hook.
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional, Set, Tuple

from ravana_ml.graph import ConceptGraph

# ConceptNet relation -> ravana typed relation_type emitted on the edge.
_FEATURE_REL_TO_RT = {
    "HasProperty": "has_property",
    "CapableOf": "capable_of",
    "UsedFor": "used_for",
}
# Fallback relation_type when the licensed relation is unknown.
_DEFAULT_FEATURE_RT = "has_property"
# relation_types we emit (used by callers / tests for assertions).
TYPED_RELATION_TYPES = {"isa", "has_property", "capable_of", "used_for", "part_of"}


def build_label_index(graph: ConceptGraph) -> Dict[str, List[int]]:
    """Map lowercased node label -> list of node ids in the graph.

    Mirrors the engine's ``_concept_keywords`` but built directly from the
    ConceptGraph so this module has no chat-engine dependency. Multi-word
    labels are kept as-is (ConceptNet terms are single words, so a multi-word
    ravana label simply never matches — which is correct: we only type edges
    for concepts ConceptNet actually knows)."""
    idx: Dict[str, List[int]] = collections.defaultdict(list)
    for nid, node in graph.nodes.items():
        if node.label:
            idx[node.label.lower()].append(nid)
    return dict(idx)


def _nearest_parents(concept: str,
                     parents: Set[str],
                     label_index: Dict[str, List[int]],
                     isa: Dict[str, Set[str]]) -> List[str]:
    """Return the most-specific existing parents of ``concept``.

    A parent P is dropped if it is itself an IsA-ancestor of another existing
    parent Q (i.e. Q is more specific than P). This implements 'prefer the
    NEAREST parent, not the root' from the brief. Cycles are guarded by a
    visited set.
    """
    existing = [p for p in parents if p in label_index]
    if len(existing) <= 1:
        return existing

    # ancestor closure of each existing parent (excluding itself)
    def ancestors(p: str) -> Set[str]:
        seen: Set[str] = set()
        frontier = [p]
        while frontier:
            nxt = []
            for node in frontier:
                for up in isa.get(node, ()):
                    if up not in seen:
                        seen.add(up)
                        nxt.append(up)
            frontier = nxt
        return seen

    anc_sets = {p: ancestors(p) for p in existing}
    # keep p iff no other existing parent q has p in q's ancestor closure
    # (i.e. p is not a strict generalized parent of a more specific existing q)
    kept = []
    for p in existing:
        dominated = False
        for q in existing:
            if q is p:
                continue
            if p in anc_sets[q]:
                dominated = True
                break
        if not dominated:
            kept.append(p)
    return kept


def inject_conceptnet_typed_edges(
    graph: ConceptGraph,
    ontology,
    label_index: Optional[Dict[str, List[int]]] = None,
    default_weight: float = 0.6,
    max_isa_parents: int = 4,
) -> Dict[str, int]:
    """Inject ConceptNet typed edges into ``graph`` for matched nodes.

    Args:
        graph: the ravana ConceptGraph (mutated in place).
        ontology: a ConceptNetOntology (with .isa, .features, .isa_weight,
                  .feature_weight, .feature_rel populated).
        label_index: optional prebuilt label->[nid] map. Built if omitted.
        default_weight: floor weight when an assertion weight is unavailable.
        max_isa_parents: cap on nearest parents emitted per concept (keeps the
                         graph from exploding; nearest-first).

    Returns a counts dict: {"isa": n, "has_property": n, "capable_of": n,
    "used_for": n, "nodes_typed": k, "total": N}.
    """
    if label_index is None:
        label_index = build_label_index(graph)

    counts: Dict[str, int] = {
        "isa": 0, "has_property": 0, "capable_of": 0, "used_for": 0,
        "nodes_typed": 0, "total": 0,
    }

    isa = ontology.isa
    features = ontology.features
    isa_weight = ontology.isa_weight
    feature_weight = ontology.feature_weight
    feature_rel = ontology.feature_rel

    # ── 1. ISA edges (taxonomic hierarchy) ──────────────────────────────
    for label, node_ids in label_index.items():
        parents = isa.get(label)
        if not parents:
            continue
        nearest = _nearest_parents(label, parents, label_index, isa)[:max_isa_parents]
        if not nearest:
            continue
        counts["nodes_typed"] += 1
        for parent in nearest:
            w = isa_weight.get(label, {}).get(parent, default_weight)
            w = max(0.01, min(1.0, float(w)))
            for nid in node_ids:
                for pid in label_index[parent]:
                    if nid == pid:
                        continue
                    graph.add_edge(nid, pid, weight=w,
                                   relation_type="isa", confidence=min(1.0, w + 0.2))
                    counts["isa"] += 1

    # ── 2. Feature edges (has_property / capable_of / used_for) ─────────
    for label, node_ids in label_index.items():
        featmap = features.get(label)
        if not featmap:
            continue
        rels = feature_rel.get(label, {})
        for feat in featmap:
            if feat not in label_index:
                continue  # can't wire an edge to a concept we don't have
            cn_rel = rels.get(feat, "HasProperty")
            rt = _FEATURE_REL_TO_RT.get(cn_rel, _DEFAULT_FEATURE_RT)
            w = feature_weight.get(label, {}).get(feat, default_weight)
            w = max(0.01, min(1.0, float(w)))
            for nid in node_ids:
                for fid in label_index[feat]:
                    if nid == fid:
                        continue
                    graph.add_edge(nid, fid, weight=w,
                                   relation_type=rt, confidence=min(1.0, w + 0.2))
                    counts[rt] += 1

    counts["total"] = counts["isa"] + counts["has_property"] + \
        counts["capable_of"] + counts["used_for"]
    return counts
