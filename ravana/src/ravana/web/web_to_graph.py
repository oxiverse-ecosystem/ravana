"""
C-lite WebToGraph — write extracted facts into the existing typed graph.
=====================================================================
Phase C-lite (web -> facts -> graph). Exercises the infra already shipped:
the ConceptGraph, the HippocampalBuffer (fast store), and sleep consolidation.

Why no dimensionality change (research reframe): facts are associative
structure (McClelland 1995 CLS; Kumaran 2016; Tse 2007). KG embeddings are
low-D by design (TransE/Bordes 2013). RAVANA's typed ConceptGraph + Hebbian
edges IS a KG. HRR only buys what graphs don't — analogy / role-filler binding
/ resonator decode — and that's the deferred N3 reasoning layer, not fact
acquisition (Plate 2003; Eliasmith 2012; Kanerva). So our extracted facts land
as typed edges with a confidence + provenance, NOT as vectors.

Thin EFE / knowledge-gap interface (sketch, per the C-time hook for E):
curiosity-driven web reading is epistemic action minimizing expected free
energy (Friston 2015; Schmidhuber 2010; Oudeyer & Kaplan 2007). A topic whose
neighbourhood in the graph is sparse is a high-EFE (uncertain) region -> a
curiosity target. We emit KnowledgeGap objects now; the full active-inference
control spine (E) is built later once C produces gaps and N3 produces structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ravana.web.openie import Fact, OpenIEExtractor


@dataclass
class KnowledgeGap:
    """Sketch of the EFE interface (wired at C-time, built in E).

    A sparse graph neighbourhood around a topic => high expected free energy
    => worth a curious web read. Partial: the precision-weighted control loop
    that acts on this is deferred to E.
    """
    topic: str
    known_edges: int
    efe: float  # expected free energy proxy: higher = more uncertain

    def is_curiosity_target(self, threshold: float = 1.0) -> bool:
        return self.efe >= threshold


class WebToGraph:
    """Write OpenIE facts into a GraphEngine as typed 'web_fact' edges."""

    RELATION_TO_EDGE = {
        "is_a": "is_a",
        "has_property": "has_property",
        "located_in": "located_in",
        "part_of": "part_of",
        "causes": "causes",
        "related_to": "semantic",
    }

    def __init__(self, graph_engine, openie: Optional[OpenIEExtractor] = None,
                 source: str = "web", xdg: bool = True):
        self.ge = graph_engine
        self.openie = openie or OpenIEExtractor()
        self.source = source
        self.xdg = xdg  # XdG (context-dependent gating, Masse et al. PMC 2018):
        # protect synapses established in ONE context from being overwritten
        # when a DIFFERENT context is ingested. Cheap complement to EWC/SI
        # for the multi-task web-ingest regime.
        # provenance + gap tracking
        self._topic_edges: Dict[str, int] = {}
        self._fact_count = 0

    def _ensure_node(self, label: str) -> Optional[int]:
        """Mint a concept node (reuse glove embedding if available)."""
        label_l = label.lower().strip()
        if not label_l:
            return None
        # reuse engine's label index if the concept already exists
        existing = self.ge._all_labels.get(label_l)
        if existing is not None:
            return existing
        vec = None
        if hasattr(self.ge, "_glove_vector"):
            vec = self.ge._glove_vector(label_l)
        if vec is None:
            # deterministic seed for OOV so the same word maps to one node
            import numpy as np
            h = hash(label_l) % 100000
            rng = np.random.RandomState(h + 7)
            vec = rng.randn(self.ge.dim).astype("float32")
            n = float(np.linalg.norm(vec))
            if n > 0:
                vec /= n
        node = self.ge.graph.add_node(vector=vec, label=label_l)
        self.ge._all_labels[label_l] = node.id
        return node.id

    def learn_text(self, text: str, source_url: str = "",
                 context: Optional[str] = None) -> int:
        """Extract facts from text and write them as typed edges.

        Returns number of NEW facts written (dedup by existing edge).

        context: logical ingest context (e.g. topic/domain). Defaults to
        source_url or self.source. Used by XdG gating: an edge
        established in context A is protected from being overwritten when
        a DIFFERENT context B is ingested (Mas se et al. 2018 context-
        dependent gating — a cheap complement to EWC/SI for the
        multi-task web-ingest regime).
        """
        ctx = (context or source_url or self.source or "web").lower().strip()
        ctx_id = "ctx_" + str(abs(hash(ctx)) % 100000)
        facts = self.openie.extract(text)
        written = 0
        for f in facts:
            subj_id = self._ensure_node(f.subject)
            obj_id = self._ensure_node(f.obj)
            if subj_id is None or obj_id is None or subj_id == obj_id:
                continue
            edge_type = self.RELATION_TO_EDGE.get(f.relation, "semantic")
            existing = self.ge.graph.get_edge(subj_id, obj_id)
            if existing is not None:
                # XdG gating: record the context this synapse was
                # established in. If a NEW context tries to re-bump/overwrite
                # an edge from a DIFFERENT established context, PROTECT it
                # (do not let later contexts erase earlier ones).
                meta = existing.source_metadata if hasattr(existing, "source_metadata") else None
                if meta is not None:
                    ctxs = set(meta.get("contexts", []))
                    if not ctxs:
                        ctxs = {meta.get("context", ctx_id)}
                    if self.xdg and ctx_id not in ctxs and len(ctxs) >= 1:
                        # cross-context: do NOT bump confidence (protect
                        # the established synapse); just record exposure.
                        ctxs.add(ctx_id)
                        meta["contexts"] = sorted(ctxs)
                        continue
                    ctxs.add(ctx_id)
                    meta["contexts"] = sorted(ctxs)
                # same-context (or XdG off): Hebbian confidence bump.
                if existing.confidence is not None:
                    existing.confidence = min(1.0, existing.confidence + 0.05)
                continue
            edge = self.ge.graph.add_edge(
                source=subj_id, target=obj_id, weight=0.5,
                relation_type=edge_type, confidence=f.confidence)
            # provenance: tag on the edge.
            if edge is not None and hasattr(edge, "source_metadata"):
                edge.source_metadata.update({
                    "source": source_url or self.source,
                    "relation": f.relation,
                    "edge_kind": "web_fact",
                    "context": ctx_id,
                    "contexts": [ctx_id],
                })
            # gap tracking (both endpoints are now a bit more known)
            for t in (f.subject, f.obj):
                self._topic_edges[t] = self._topic_edges.get(t, 0) + 1
            self._fact_count += 1
            written += 1
        return written

    def knowledge_gap(self, topic: str, max_known: int = 8) -> KnowledgeGap:
        """EFE proxy for a topic: sparse neighbourhood => high uncertainty.

        Combines two signals of "how much we know about this topic":
          - C-lite facts written about it (self._topic_edges)
          - the node's actual graph edge-degree (a sparse/isolated node in the
            existing graph is genuinely uncertain — the brain-faithful EFE:
            pattern completion is weak where associations are few)
        efe = max(0, max_known - known_edges). Higher = more curious.
        """
        t = topic.lower().strip()
        known = self._topic_edges.get(t, 0)
        # also count the node's real graph neighbourhood (edge degree)
        nid = self.ge._all_labels.get(t)
        if nid is not None:
            node = self.ge.graph.get_node(nid)
            if node is not None:
                # a topic that EXISTS in the graph is at least minimally known;
                # a topic absent from the graph entirely is the MOST uncertain.
                known = max(known, 1)
                deg = len(list(self.ge.graph.get_outgoing(nid))) if hasattr(self.ge.graph, "get_outgoing") else 0
                known = max(known, min(max_known, deg))
        efe = max(0.0, max_known - known)
        return KnowledgeGap(topic=topic, known_edges=known, efe=efe)

    def fact_count(self) -> int:
        return self._fact_count
