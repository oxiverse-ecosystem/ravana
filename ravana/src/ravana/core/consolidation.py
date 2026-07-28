"""Sleep-independent schema consolidation: episodic buffer -> semantic graph.

Brain mechanism: systems consolidation / generative schema extraction
(McClelland, McNaughton & O'Reilly 1995; Lewis & Durrant 2011 —
overlapping hippocampal replay extracts statistical regularities into
neocortical schemas). The hippocampal buffer holds fast, sparse episodic
triples; recurring structure is PROMOTED into the semantic graph (ATL
hub) as probabilistic edges, where Phase-1 spreading activation can use
it for inference the raw episodes never stated.

Two extraction channels (both from the plan, section 6.4):
1. Frequency schemas — a (predicate, object) that recurs for the same
   subject is a stable property, not an episode.
2. Co-occurrence schemas — subjects whose triples share objects
   (Jaccard over object sets) get a co_occurs edge, the graph correlate
   of "fire together, wire together" at the schema level.

Design rules:
- Promotion threshold is DISTRIBUTION-RELATIVE (fraction of the
  subject's own triple count), not a fixed global count.
- Online-only: consumes whatever is in the buffer at call time.
- Idempotent per fact: promoted facts are marked consolidated so the
  same episode is never double-counted into edge weight.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Schema:
    subject: str
    relation: str
    object: str
    support: int          # how many episodes back this edge
    weight: float         # promotion weight for the semantic graph


class Consolidator:
    """Extracts schemas from a HippocampalBuffer into a SemanticGraph.

    Trigger contract (kept by the caller): run after the buffer has grown
    by >= `growth_trigger` facts since the last run, or explicitly before
    an evaluation pass (offline consolidation without a sleep cycle).
    """

    def __init__(self, growth_trigger: int = 50):
        self.growth_trigger = int(growth_trigger)
        self._last_size = 0
        self.total_promoted = 0

    # ── trigger ──────────────────────────────────────────────────────────
    def should_run(self, buffer) -> bool:
        try:
            size = len(getattr(buffer, "_all_facts", []) or [])
        except Exception:
            return False
        return (size - self._last_size) >= self.growth_trigger

    # ── extraction ───────────────────────────────────────────────────────
    def extract_schemas(self, buffer) -> List[Schema]:
        """Frequency + co-occurrence schema mining over the buffer."""
        facts = list(getattr(buffer, "_all_facts", []) or [])
        self._last_size = len(facts)
        if not facts:
            return []
        schemas: List[Schema] = []

        # 1. Frequency schemas: recurring (pred, obj) per subject. The
        # buffer DEDUPES repeated triples into rehearsal_count (store()
        # strengthens instead of appending; a fresh fact starts at 1), so
        # recurrence lives on the fact itself: support = rehearsal_count.
        # Already-consolidated episodes are excluded so a re-run cannot
        # re-promote the same schema and inflate its weight.
        by_subj: Dict[str, List] = defaultdict(list)
        for f in facts:
            if getattr(f, "consolidated", False):
                continue
            s = (getattr(f, "subject", "") or "").strip().lower()
            if s:
                by_subj[s].append(f)
        for subj, sf in by_subj.items():
            pairs: Counter = Counter()
            for f in sf:
                pred = (getattr(f, "predicate", "") or "").strip().lower()
                obj = (getattr(f, "object", "") or "").strip().lower()
                if pred and obj:
                    pairs[(pred, obj)] += int(
                        getattr(f, "rehearsal_count", 1) or 1)
            n = sum(pairs.values())
            for (pred, obj), c in pairs.items():
                # Distribution-relative bar: a pair is schematic when it
                # accounts for a meaningful share of the subject's episodes
                # AND recurred at least twice (a single episode is an
                # event, not a schema).
                if c >= 2 and n and c / n >= 0.2:
                    schemas.append(Schema(
                        subject=subj, relation=pred, object=obj,
                        support=c, weight=min(1.0, c / n + 0.3)))

        # 2. Co-occurrence schemas: subjects sharing objects (Jaccard).
        obj_sets: Dict[str, Set[str]] = {
            s: {(getattr(f, "object", "") or "").strip().lower()
                for f in sf if getattr(f, "object", None)}
            for s, sf in by_subj.items()}
        subjects = [s for s, os_ in obj_sets.items() if os_]
        for i in range(len(subjects)):
            for j in range(i + 1, len(subjects)):
                a, b = subjects[i], subjects[j]
                inter = obj_sets[a] & obj_sets[b]
                union = obj_sets[a] | obj_sets[b]
                if not union or not inter:
                    continue
                jac = len(inter) / len(union)
                if jac > 0.3:
                    schemas.append(Schema(
                        subject=a, relation="co_occurs", object=b,
                        support=len(inter), weight=min(1.0, jac)))
        return schemas

    # ── promotion ────────────────────────────────────────────────────────
    def consolidate(self, buffer, graph) -> int:
        """Extract schemas and promote them into the semantic graph.
        Returns the number of edges promoted. Marks the supporting
        episodic facts consolidated (they remain retrievable — episodic
        and semantic traces coexist; Winocur & Moscovitch 2011)."""
        if graph is None:
            return 0
        schemas = self.extract_schemas(buffer)
        promoted = 0
        for sc in schemas:
            try:
                if sc.relation == "co_occurs":
                    graph.add_relation(sc.subject, "co_occurs", sc.object,
                                       weight=sc.weight, _online=True)
                else:
                    graph.add_triple(sc.subject, sc.relation, sc.object,
                                     confidence=sc.weight)
                promoted += 1
            except Exception:
                continue
        # Mark supporting episodes so future runs don't re-inflate weights.
        if promoted:
            promoted_keys = {(s.subject, s.relation, s.object)
                             for s in schemas if s.relation != "co_occurs"}
            for f in getattr(buffer, "_all_facts", []) or []:
                key = ((getattr(f, "subject", "") or "").strip().lower(),
                       (getattr(f, "predicate", "") or "").strip().lower(),
                       (getattr(f, "object", "") or "").strip().lower())
                if key in promoted_keys:
                    try:
                        buffer.mark_consolidated(f)
                    except Exception:
                        pass
        self.total_promoted += promoted
        return promoted
