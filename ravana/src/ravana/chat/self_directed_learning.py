"""Self-directed learning: detect own knowledge gaps, form hypotheses, seek
examples, improve through experimentation. No LLM, no retraining, online.

Brain basis: metacognitive monitoring (ACC/FOK) drives curiosity and
hypothesis testing (Koriat 1993; Botvinick 2004). RAVANA already exposes
FOK confidence via metacognition.fok_confidence and a curiosity drive in
web_learning. This module ties them together into one self-directed loop
that runs during normal operation.
"""

from typing import List, Dict, Any, Optional
import math


class KnowledgeGap:
    def __init__(self, topic: str, confidence: float, support: float):
        self.topic = topic
        self.confidence = confidence
        self.support = support

    def to_dict(self) -> Dict[str, Any]:
        return {"topic": self.topic, "confidence": self.confidence,
                "support": self.support}


class SelfDirectedLearner:
    def __init__(self, engine=None):
        self.engine = engine
        self.gaps: List[KnowledgeGap] = []
        self.hypotheses: List[Dict[str, Any]] = []
        self.experiments: List[Dict[str, Any]] = []

    def detect_gaps(self, threshold: float = 0.30) -> List[KnowledgeGap]:
        """Detect concepts whose FOK confidence is below threshold."""
        self.gaps = []
        eng = self.engine
        if eng is None:
            return self.gaps
        try:
            from .metacognition import fok_confidence
        except Exception:
            def fok_confidence(support_count, retrieval_succeeded=False, base=0.0):
                return 1.0 - math.exp(-max(0.0, support_count) / 4.0)
        graph = getattr(eng, "graph", None)
        nodes = getattr(graph, "nodes", {}) if graph is not None else {}
        for nid, node in list(nodes.items()):
            label = getattr(node, "label", None) or str(nid)
            support = float(getattr(node, "prediction_free_energy", 0.0) or 0.0)
            # support proxy: edge count if available
            edges = getattr(graph, "edges", {})
            ec = 0
            for e in edges.values() if hasattr(edges, "values") else []:
                if getattr(e, "source", None) == nid or getattr(e, "target", None) == nid:
                    ec += 1
            conf = fok_confidence(ec, False, 0.0)
            if conf < threshold and len(label) >= 3:
                self.gaps.append(KnowledgeGap(label, conf, float(ec)))
        # de-dupe by topic
        seen = set()
        uniq = []
        for g in self.gaps:
            if g.topic in seen:
                continue
            seen.add(g.topic)
            uniq.append(g)
        self.gaps = uniq
        return self.gaps

    def form_hypothesis(self, gap: KnowledgeGap) -> Dict[str, Any]:
        hyp = {
            "topic": gap.topic,
            "claim": f"i may be missing knowledge about {gap.topic}",
            "confidence": gap.confidence,
            "test": f"seek examples of {gap.topic} and re-check FOK",
        }
        self.hypotheses.append(hyp)
        return hyp

    def seek_examples(self, gap: KnowledgeGap) -> List[str]:
        eng = self.engine
        queries: List[str] = []
        if eng is not None and hasattr(eng, "_auto_select_curiosity_topics"):
            try:
                topics = eng._auto_select_curiosity_topics(max_topics=1)
                if topics:
                    queries.extend(topics)
            except Exception:
                pass
        if gap.topic not in queries:
            queries.append(gap.topic)
        return queries

    def experiment(self, gap: KnowledgeGap) -> Dict[str, Any]:
        before = gap.confidence
        # Re-detect to see if confidence improved (online learning may have occurred)
        new_gaps = self.detect_gaps()
        after_map = {g.topic: g.confidence for g in new_gaps}
        after = after_map.get(gap.topic, before)
        result = {
            "topic": gap.topic,
            "before": before,
            "after": after,
            "improved": after > before,
        }
        self.experiments.append(result)
        return result

    def run_cycle(self) -> Dict[str, Any]:
        gaps = self.detect_gaps()
        hyps = [self.form_hypothesis(g) for g in gaps[:5]]
        queries = []
        for g in gaps[:5]:
            queries.extend(self.seek_examples(g))
        return {"gaps": [g.to_dict() for g in gaps],
                "hypotheses": hyps,
                "queries": queries}
