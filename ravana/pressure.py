import numpy as np
from typing import Dict, List, Optional
from .graph import ConceptGraph
from .tensor import StateTensor, RawTensor


class PressureAccumulator:
    def __init__(self, graph: ConceptGraph):
        self.graph = graph
        self.semantic_pressure = 0.0
        self.linguistic_pressure = 0.0
        self.episodic_pressure = 0.0
        self.contradiction_pressure = 0.0
        self.abstraction_pressure = 0.0  # from co-activated clusters needing compression
        self.history: List[float] = []

    @property
    def total(self) -> float:
        return (self.semantic_pressure + self.linguistic_pressure +
                self.episodic_pressure + self.contradiction_pressure +
                self.abstraction_pressure)

    @property
    def normalized(self) -> float:
        return min(1.0, self.total / 100.0)

    def accumulate_semantic(self, error: float, salience: float = 0.3):
        self.semantic_pressure += error * salience
        self._record()

    def accumulate_linguistic(self, error: float):
        self.linguistic_pressure += error * 0.1
        self._record()

    def accumulate_episodic(self, error: float, recency: float = 0.5):
        self.episodic_pressure += error * recency
        self._record()

    def accumulate_contradiction(self, count: int):
        self.contradiction_pressure += count * 0.5
        self._record()

    def accumulate_abstraction(self, cluster_count: int, mean_coactivation: float):
        """Accumulate pressure from co-activated clusters that haven't been compressed."""
        self.abstraction_pressure += cluster_count * mean_coactivation * 0.3
        self._record()

    def _record(self):
        self.history.append(self.total)

    def decay(self, rate: float = 0.1):
        self.semantic_pressure = max(0.0, self.semantic_pressure - rate * self.semantic_pressure)
        self.linguistic_pressure = max(0.0, self.linguistic_pressure - rate * self.linguistic_pressure)
        self.episodic_pressure = max(0.0, self.episodic_pressure - rate * self.episodic_pressure)
        self.contradiction_pressure = max(0.0, self.contradiction_pressure - rate * self.contradiction_pressure)
        self.abstraction_pressure = max(0.0, self.abstraction_pressure - rate * self.abstraction_pressure)

    def reset(self):
        self.semantic_pressure = 0.0
        self.linguistic_pressure = 0.0
        self.episodic_pressure = 0.0
        self.contradiction_pressure = 0.0
        self.abstraction_pressure = 0.0

    def needs_sleep(self, threshold: float = 10.0) -> bool:
        return self.total > threshold

    def report(self) -> Dict[str, float]:
        return {
            'semantic': self.semantic_pressure,
            'linguistic': self.linguistic_pressure,
            'episodic': self.episodic_pressure,
            'contradiction': self.contradiction_pressure,
            'abstraction': self.abstraction_pressure,
            'total': self.total,
            'normalized': self.normalized,
        }

    def apply_to_graph(self):
        for node in self.graph.nodes.values():
            if node.contradiction_count > 2:
                self.graph.apply_pressure(node.id, node.contradiction_count * 0.1)
