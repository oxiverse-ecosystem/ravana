import numpy as np
from typing import Dict, List, Optional
from .graph import ConceptGraph
from .tensor import StateTensor, RawTensor


class FreeEnergyAccumulator:
    def __init__(self, graph: ConceptGraph):
        self.graph = graph
        self.semantic_free_energy = 0.0
        self.linguistic_free_energy = 0.0
        self.episodic_free_energy = 0.0
        self.contradiction_free_energy = 0.0
        self.abstraction_free_energy = 0.0  # from co-activated clusters needing compression
        self.history: List[float] = []

    @property
    def free_energy(self) -> float:
        """Total free energy: sum of all free energy channels.
        This is the canonical 'distance from coherent state' variable."""
        return (self.semantic_free_energy + self.linguistic_free_energy +
                self.episodic_free_energy + self.contradiction_free_energy +
                self.abstraction_free_energy)

    @property
    def total(self) -> float:
        """Backward-compatible alias for free_energy."""
        return self.free_energy

    @property
    def normalized(self) -> float:
        """Free energy normalized to [0, 1]."""
        return min(1.0, self.free_energy / 100.0)

    def accumulate_semantic(self, error: float, salience: float = 0.3):
        self.semantic_free_energy += error * salience
        self._record()

    def accumulate_linguistic(self, error: float):
        self.linguistic_free_energy += error * 0.1
        self._record()

    def accumulate_episodic(self, error: float, recency: float = 0.5):
        self.episodic_free_energy += error * recency
        self._record()

    def accumulate_contradiction(self, count: int):
        self.contradiction_free_energy += count * 0.5
        self._record()

    def accumulate_abstraction(self, cluster_count: int, mean_coactivation: float):
        """Accumulate free energy from co-activated clusters that haven't been compressed."""
        self.abstraction_free_energy += cluster_count * mean_coactivation * 0.3
        self._record()

    def _record(self):
        self.history.append(self.free_energy)

    def decay(self, rate: float = 0.1):
        self.semantic_free_energy = max(0.0, self.semantic_free_energy - rate * self.semantic_free_energy)
        self.linguistic_free_energy = max(0.0, self.linguistic_free_energy - rate * self.linguistic_free_energy)
        self.episodic_free_energy = max(0.0, self.episodic_free_energy - rate * self.episodic_free_energy)
        self.contradiction_free_energy = max(0.0, self.contradiction_free_energy - rate * self.contradiction_free_energy)
        self.abstraction_free_energy = max(0.0, self.abstraction_free_energy - rate * self.abstraction_free_energy)

    def reset(self):
        self.semantic_free_energy = 0.0
        self.linguistic_free_energy = 0.0
        self.episodic_free_energy = 0.0
        self.contradiction_free_energy = 0.0
        self.abstraction_free_energy = 0.0

    def needs_sleep(self, threshold: float = 10.0) -> bool:
        return self.free_energy > threshold

    def report(self) -> Dict[str, float]:
        return {
            'semantic': self.semantic_free_energy,
            'linguistic': self.linguistic_free_energy,
            'episodic': self.episodic_free_energy,
            'contradiction': self.contradiction_free_energy,
            'abstraction': self.abstraction_free_energy,
            'free_energy': self.free_energy,
            'normalized': self.normalized,
        }

    def apply_to_graph(self):
        for node in self.graph.nodes.values():
            if node.contradiction_count > 2:
                self.graph.apply_free_energy(node.id, node.contradiction_count * 0.1)
