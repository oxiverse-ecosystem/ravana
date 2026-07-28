"""TripletInferenceOperator — the unified query + ingestion entry point.

Adapters accept all three existing triple shapes:
- core.proposition_parser.Proposition  (fields: subject/predicate/object)
- web.openie.Fact                      (fields: subject/relation/obj)
- core.hippocampal_buffer.FactTriple   (fields: subject/predicate/object)

The optional ``hrr`` argument wires HRRReasoner both as an extra
inference channel (vector chaining) and as a transitivity learning
signal (query_chain success/failure feeds the same statistics).
"""
from __future__ import annotations

from typing import List, Optional

from .abstention import AbstentionGate
from .core import InferenceResult, Triple
from .learning import ProfileLearner
from .canonical import canonical_term
from .memory import TripletMemory
from .operators import (Composition, HierarchicalInference, InversePredicate,
                        SymmetricClosure, TransitiveChain)
from .seed import SEED_TRIPLES


class TripletInferenceOperator:
    def __init__(self, memory: Optional[TripletMemory] = None,
                 hrr=None, seed: bool = True):
        self.memory = memory or TripletMemory()
        self.hrr = hrr
        self.learner = ProfileLearner(self.memory)
        self.chain = TransitiveChain(self.memory)
        self.symmetry = SymmetricClosure(self.memory)
        self.inverse = InversePredicate(self.memory)
        self.composer = Composition(self.memory)
        self.hierarchy = HierarchicalInference(self.memory)
        self.gate = AbstentionGate()
        if seed and not self.memory.triples:
            for t in SEED_TRIPLES:
                # Seeds are stored but NOT observed — they are exemplars
                # for pattern mining, not evidence.
                self.memory.add(t)

    # ── ingestion ─────────────────────────────────────────────────────
    def ingest_triple(self, triple: Triple) -> None:
        """Store a triple AND run the online learning loop."""
        stored = self.memory.add(triple)
        if stored is None:
            return
        self.learner.observe(stored)
        # Mirror into the shared episodic buffer if one is attached.
        if self.memory.episodic is not None:
            try:
                self.memory.episodic.store(
                    stored.subject, stored.predicate, stored.object,
                    confidence=stored.confidence,
                    session_date=stored.session_date,
                    absolute_date=stored.absolute_date)
            except Exception:
                pass
        # Mirror into HRR for vector-side integrative encoding.
        if self.hrr is not None:
            try:
                self.hrr.encode(stored.subject, stored.predicate,
                                stored.object)
            except Exception:
                pass

    def ingest_proposition(self, prop) -> None:
        """Adapter for core.proposition_parser.Proposition."""
        if not getattr(prop, "object", ""):
            return
        self.ingest_triple(Triple(
            subject=prop.subject, predicate=prop.predicate,
            object=prop.object,
            confidence=getattr(prop, "confidence", 0.6),
            source="conversation"))

    def ingest_openie_fact(self, fact) -> None:
        """Adapter for web.openie.Fact (object field is ``obj``)."""
        self.ingest_triple(Triple(
            subject=fact.subject, predicate=fact.relation,
            object=fact.obj,
            confidence=getattr(fact, "confidence", 0.6),
            source="web"))

    # ── query ─────────────────────────────────────────────────────────
    def infer(self, subject: str, predicate: str,
              target: Optional[str] = None,
              max_results: int = 3) -> List[InferenceResult]:
        s = canonical_term(subject)
        results: List[InferenceResult] = []

        def _emit(obj: str, conf: float, path: str, op: str):
            r = InferenceResult(
                triple=Triple(s, predicate, obj, confidence=conf,
                              source="inferred" if op != "lookup"
                              else "conversation"),
                confidence=conf, path=path, operator=op)
            if not self.gate.should_abstain(r, self.memory):
                results.append(r)

        # 1. Direct lookup.
        for obj in self.memory.objects_of(s, predicate):
            _emit(obj, 0.9, "direct", "lookup")

        # 2. Transitive chain (learned gate).
        for obj, conf, path in self.chain.infer_all(s, predicate):
            _emit(obj, conf, path, "transitive")

        # 3. Symmetric closure (learned gate).
        for obj, conf in self.symmetry.infer(s, predicate):
            _emit(obj, conf, f"{obj}→{s} (symmetric)", "symmetric")

        # 4. Learned inverse predicate.
        for obj, conf, inv in self.inverse.infer(s, predicate):
            _emit(obj, conf, f"{obj}-{inv}→{s}", "inverse")

        # 5. Learned composition.
        for obj, conf, path in self.composer.infer_all(s, predicate):
            _emit(obj, conf, path, "composition")

        # 6. Hierarchical inheritance.
        for obj, conf, path in self.hierarchy.inherit(s, predicate):
            _emit(obj, conf, path, "hierarchy")

        # 7. HRR vector chain — extra channel + learning cross-signal.
        if self.hrr is not None:
            try:
                chain = self.hrr.query_chain(s, predicate, max_hops=2)
                success = len(chain) > 1
                self.learner.observe_hrr_chain(predicate, success)
                if success:
                    prof = self.memory.profiles.get(
                        self.memory.profile(predicate).predicate)
                    conf = (prof.transitivity_score * 0.8) if prof else 0.4
                    _emit(chain[-1], conf,
                          "→".join([s] + chain) + " (hrr)", "hrr")
            except Exception:
                pass

        # Dedupe by object, keep highest confidence.
        best = {}
        for r in results:
            k = r.triple.object
            if k not in best or r.confidence > best[k].confidence:
                best[k] = r
        out = sorted(best.values(), key=lambda r: r.confidence, reverse=True)
        if target is not None:
            tgt = canonical_term(target)
            return [r for r in out if r.triple.object == tgt]
        return out[:max_results]

    # ── persistence (call BOTH sides from the host engine) ────────────
    def to_dict(self) -> dict:
        return {"memory": self.memory.to_dict(),
                "gate": self.gate.to_dict()}

    def from_dict(self, d: dict) -> None:
        self.memory.from_dict(d.get("memory", {}))
        self.gate.from_dict(d.get("gate", {}))
