"""
Synthetic Episode Injector
============================
Feeds structured knowledge into RLMv2's graph via learn().

Supports:
  - Dict-based facts: {"subject": "coffee", "relation": "causes", "object": "alertness", "confidence": 0.9}
  - Tuple-based facts: ("coffee", "causes", "alertness")
  - Batch injection from knowledge bases
  - Confidence-weighted training (repeat high-confidence facts more)
  - Multi-edge support via relation-object hub keys

Usage:
  injector = EpisodeInjector(model, tok)
  injector.inject_facts(facts)
  injector.inject_from_file("knowledge.json")
  injector.inject_from_dict({"coffee": {"causes": ["alertness", "anxiety"], "contains": ["caffeine"]}})
"""
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Fact:
    """A structured knowledge fact."""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    source: str = "manual"
    timestamp: float = 0.0

    def to_triple(self) -> Tuple[str, str, str]:
        return (self.subject, self.relation, self.object)

    def __repr__(self):
        return f"({self.subject}, {self.relation}, {self.object}, conf={self.confidence:.2f})"


class EpisodeInjector:
    """Feeds structured knowledge into RLMv2."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer
        self._injected_facts: List[Fact] = []
        self._injection_stats = {
            "total": 0,
            "successful": 0,
            "skipped": 0,
            "epochs_trained": 0,
        }

    def inject_facts(self, facts, epochs: int = 5, confidence_weighted: bool = True) -> Dict:
        """Inject a list of facts into the model.

        Args:
            facts: List of Fact objects or (subject, relation, object) tuples
            epochs: Number of training epochs per fact
            confidence_weighted: If True, high-confidence facts get more epochs

        Returns:
            Injection statistics
        """
        # Normalize to Fact objects
        normalized = []
        for f in facts:
            if isinstance(f, tuple):
                if len(f) == 3:
                    normalized.append(Fact(subject=f[0], relation=f[1], object=f[2]))
                elif len(f) == 4:
                    normalized.append(Fact(subject=f[0], relation=f[1], object=f[2], confidence=f[3]))
            elif isinstance(f, Fact):
                normalized.append(f)

        self._injection_stats["total"] += len(normalized)

        for fact in normalized:
            # Encode the full triple as tokens
            triple_str = f"{fact.subject} {fact.relation} {fact.object}"
            ids = self.tok.encode(triple_str)

            if len(ids) < 2:
                self._injection_stats["skipped"] += 1
                continue

            # Determine epoch count (confidence-weighted)
            if confidence_weighted:
                fact_epochs = max(1, int(epochs * fact.confidence))
            else:
                fact_epochs = epochs

            # Train
            for epoch in range(fact_epochs):
                ctx = np.array([ids[:-1]], dtype=np.int64)
                tgt = np.array([[ids[-1]]], dtype=np.int64)
                self.model.learn(ctx, tgt)

            self._injected_facts.append(fact)
            self._injection_stats["successful"] += 1
            self._injection_stats["epochs_trained"] += fact_epochs

        return self._injection_stats.copy()

    def inject_from_dict(self, knowledge: Dict[str, Dict[str, List[str]]],
                         confidence: float = 1.0, source: str = "dict") -> Dict:
        """Inject from a nested dict structure.

        Format: {"subject": {"relation": ["object1", "object2"], ...}, ...}

        Example:
            {"coffee": {"causes": ["alertness", "anxiety"], "contains": ["caffeine"]}}
        """
        facts = []
        for subject, relations in knowledge.items():
            for relation, objects in relations.items():
                if isinstance(objects, str):
                    objects = [objects]
                for obj in objects:
                    facts.append(Fact(
                        subject=subject, relation=relation, object=obj,
                        confidence=confidence, source=source
                    ))
        return self.inject_facts(facts)

    def inject_from_file(self, filepath: str, format: str = "auto") -> Dict:
        """Inject facts from a JSON file.

        Supported formats:
          - "triples": [{"subject": "...", "relation": "...", "object": "...", "confidence": 0.9}, ...]
          - "nested": {"subject": {"relation": ["obj1", "obj2"]}}
          - "auto": detect format
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        if format == "auto":
            if isinstance(data, list):
                format = "triples"
            elif isinstance(data, dict):
                format = "nested"

        if format == "triples":
            facts = []
            for item in data:
                facts.append(Fact(
                    subject=item["subject"],
                    relation=item["relation"],
                    object=item["object"],
                    confidence=item.get("confidence", 1.0),
                    source=item.get("source", filepath),
                ))
            return self.inject_facts(facts)
        elif format == "nested":
            return self.inject_from_dict(data, source=filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

    def get_stats(self) -> Dict:
        """Get injection statistics."""
        stats = self._injection_stats.copy()
        stats["unique_facts"] = len(self._injected_facts)
        stats["graph_nodes"] = len(self.model.graph.nodes)
        stats["graph_edges"] = len(self.model.graph.edges)
        return stats

    def get_facts_for_subject(self, subject: str) -> List[Fact]:
        """Get all injected facts for a given subject."""
        return [f for f in self._injected_facts if f.subject == subject]

    def summary(self) -> str:
        """Human-readable summary of injection state."""
        stats = self.get_stats()
        lines = [
            f"EpisodeInjector Summary",
            f"  Facts injected: {stats['successful']}/{stats['total']}",
            f"  Skipped: {stats['skipped']}",
            f"  Total epochs: {stats['epochs_trained']}",
            f"  Graph: {stats['graph_nodes']} nodes, {stats['graph_edges']} edges",
        ]
        return "\n".join(lines)


# ============================================================
# CONVENIENCE: Pre-built knowledge bases
# ============================================================

PHARMACOLOGY_KB = {
    "aspirin": {
        "causes": ["pain_relief", "stomach_irritation", "blood_thinning"],
        "reduces": ["inflammation", "fever", "pain"],
        "contains": ["salicylic_acid"],
        "is_a": ["nsaid", "analgesic"],
    },
    "ibuprofen": {
        "causes": ["pain_relief", "stomach_irritation"],
        "reduces": ["inflammation", "fever", "pain"],
        "contains": ["propionic_acid"],
        "is_a": ["nsaid", "analgesic"],
    },
    "acetaminophen": {
        "causes": ["pain_relief", "liver_damage"],
        "reduces": ["fever", "pain"],
        "is_a": ["analgesic"],
    },
    "caffeine": {
        "increases": ["alertness", "heart_rate", "blood_pressure"],
        "decreases": ["fatigue", "drowsiness"],
        "contains": ["methylxanthine"],
        "is_a": ["stimulant"],
    },
    "nicotine": {
        "increases": ["alertness", "heart_rate", "blood_pressure"],
        "decreases": ["appetite", "anxiety"],
        "causes": ["addiction"],
        "contains": ["pyridine"],
        "is_a": ["stimulant", "alkaloid"],
    },
}

ECOLOGY_KB = {
    "wolf": {
        "hunts": ["deer", "rabbit", "elk"],
        "is_a": ["predator", "mammal"],
        "lives_in": ["forest", "tundra"],
    },
    "deer": {
        "eats": ["grass", "leaves", "shrubs"],
        "is_a": ["herbivore", "mammal"],
        "hunted_by": ["wolf", "bear"],
        "lives_in": ["forest", "meadow"],
    },
    "bear": {
        "hunts": ["deer", "fish", "berries"],
        "is_a": ["predator", "mammal"],
        "lives_in": ["forest", "mountain"],
    },
    "grass": {
        "is_a": ["plant"],
        "needs": ["sunlight", "water", "soil"],
        "eaten_by": ["deer", "rabbit"],
    },
    "sunlight": {
        "enables": ["photosynthesis"],
        "increases": ["plant_growth"],
        "is_a": ["energy_source"],
    },
}


def load_pharmacology_kb():
    """Load the pharmacology knowledge base as facts."""
    facts = []
    for subject, relations in PHARMACOLOGY_KB.items():
        for relation, objects in relations.items():
            for obj in objects:
                facts.append(Fact(
                    subject=subject, relation=relation, object=obj,
                    confidence=0.9, source="pharmacology_kb"
                ))
    return facts


def load_ecology_kb():
    """Load the ecology knowledge base as facts."""
    facts = []
    for subject, relations in ECOLOGY_KB.items():
        for relation, objects in relations.items():
            for obj in objects:
                facts.append(Fact(
                    subject=subject, relation=relation, object=obj,
                    confidence=0.9, source="ecology_kb"
                ))
    return facts
