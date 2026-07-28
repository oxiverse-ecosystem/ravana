"""Inference operators — structural mechanisms gated by learned scores.

These classes contain NO relational knowledge: they don't know which
predicates are transitive/symmetric. That knowledge lives entirely in
the learned RelationProfile statistics; the operators just pattern-
match structure and consult the profiles' Wilson-bounded gates.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .core import InferenceResult, Triple
from .learning import DECISION_BOUNDARY
from .canonical import canonical_term
from .memory import TripletMemory


class TransitiveChain:
    """A-r->B, B-r->C => candidate A-r->C, gated by learned transitivity."""

    def __init__(self, memory: TripletMemory):
        self.memory = memory

    def infer_all(self, subject: str, predicate: str,
                  max_hops: int = 3) -> List[Tuple[str, float, str]]:
        m = self.memory
        prof = m.profiles.get(m.profile(predicate).predicate)
        if prof is None:
            return []
        gate = prof.transitivity_lower()
        if gate <= DECISION_BOUNDARY:
            return []
        s = canonical_term(subject)
        direct = m.objects_of(s, predicate)
        results: List[Tuple[str, float, str]] = []
        # BFS out to max_hops; confidence decays by the learned score per hop.
        frontier: List[Tuple[str, float, List[str]]] = [
            (b, prof.transitivity_score, [s, b]) for b in direct]
        seen: Set[str] = {s} | set(direct)
        hops = 1
        while frontier and hops < max_hops:
            nxt: List[Tuple[str, float, List[str]]] = []
            for node, conf, path in frontier:
                for c in m.objects_of(node, predicate):
                    if c in seen:
                        continue
                    seen.add(c)
                    new_conf = conf * prof.transitivity_score
                    new_path = path + [c]
                    if not m.has_fact(s, predicate, c):
                        results.append((c, new_conf, "→".join(new_path)))
                    nxt.append((c, new_conf, new_path))
            frontier = nxt
            hops += 1
        results.sort(key=lambda r: r[1], reverse=True)
        return results


class SymmetricClosure:
    """A-r->B => candidate B-r->A, gated by learned symmetry."""

    def __init__(self, memory: TripletMemory):
        self.memory = memory

    def infer(self, subject: str, predicate: str) -> List[Tuple[str, float]]:
        m = self.memory
        prof = m.profiles.get(m.profile(predicate).predicate)
        if prof is None or prof.symmetry_lower() <= DECISION_BOUNDARY:
            return []
        s = canonical_term(subject)
        out = []
        for a in m.subjects_of(s, predicate):
            if not m.has_fact(s, predicate, a):
                out.append((a, prof.symmetry_score * 0.9))
        return out


class InversePredicate:
    """A-r1->B with learned inverse r2 => candidate B-r2->A."""

    def __init__(self, memory: TripletMemory):
        self.memory = memory

    def infer(self, subject: str, predicate: str) -> List[Tuple[str, float, str]]:
        """Answers (subject, predicate, ?) using the learned inverse:
        if r_inv is the inverse of predicate, then (X, r_inv, subject)
        implies (subject, predicate, X)."""
        m = self.memory
        prof = m.profiles.get(m.profile(predicate).predicate)
        if prof is None:
            return []
        inv, share = prof.inverse_predicate()
        # Dominance gate: the top inverse must own the majority of the
        # inverse evidence (distribution-relative, not a fixed count).
        if inv is None or share <= DECISION_BOUNDARY:
            return []
        s = canonical_term(subject)
        out = []
        for x in m.subjects_of(s, inv):
            if not m.has_fact(s, predicate, x):
                out.append((x, share * 0.9, inv))
        return out


class Composition:
    """A-r1->B, B-r2->C => candidate A-r3->C where r3 is the learned
    composed predicate of (r1, r2)."""

    def __init__(self, memory: TripletMemory):
        self.memory = memory

    def infer_all(self, subject: str, predicate: str) -> List[Tuple[str, float, str]]:
        """Answers (subject, predicate, ?) via learned compositions that
        RESULT in `predicate`: find (r1, r2) with composed==predicate."""
        m = self.memory
        s = canonical_term(subject)
        out: List[Tuple[str, float, str]] = []
        target = m.profile(predicate).predicate
        for r1, prof1 in list(m.profiles.items()):
            for r2, bucket in prof1.composition_counts.items():
                total = sum(bucket.values())
                if not total:
                    continue
                cnt = bucket.get(target, 0)
                share = cnt / total
                if share <= DECISION_BOUNDARY or cnt < 2:
                    continue
                for b in m.objects_of(s, r1):
                    for c in m.objects_of(b, r2):
                        if c == s or m.has_fact(s, target, c):
                            continue
                        out.append((c, share * 0.8,
                                    f"{s}-{r1}→{b}-{r2}→{c}"))
        out.sort(key=lambda r: r[1], reverse=True)
        return out


class HierarchicalInference:
    """Property inheritance through the most-hierarchical learned
    predicate: if (s, r_h, parent) where r_h is deep+transitive, and
    (parent, predicate, x), then candidate (s, predicate, x)."""

    def __init__(self, memory: TripletMemory):
        self.memory = memory

    def _hierarchy_predicates(self) -> List[str]:
        """Predicates that are BOTH reliably transitive and deeper-
        chaining than the average predicate (distribution-relative)."""
        profs = [p for p in self.memory.profiles.values() if p.depth_n > 0]
        if not profs:
            return []
        mean_depth = sum(p.hierarchy_depth for p in profs) / len(profs)
        return [p.predicate for p in profs
                if p.transitivity_lower() > DECISION_BOUNDARY
                and p.hierarchy_depth > mean_depth]

    def inherit(self, subject: str, predicate: str) -> List[Tuple[str, float, str]]:
        m = self.memory
        s = canonical_term(subject)
        out: List[Tuple[str, float, str]] = []
        for rh in self._hierarchy_predicates():
            if rh == m.profile(predicate).predicate:
                continue
            hprof = m.profiles[rh]
            for parent in m.objects_of(s, rh):
                for x in m.objects_of(parent, predicate):
                    if m.has_fact(s, predicate, x):
                        continue
                    conf = hprof.transitivity_score * 0.8
                    out.append((x, conf, f"{s}-{rh}→{parent}-{predicate}→{x}"))
        out.sort(key=lambda r: r[1], reverse=True)
        return out
