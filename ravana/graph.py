import numpy as np
import time
from typing import Optional, List, Tuple, Dict, Set
from .tensor import StateTensor, RawTensor, tensor


class ConceptNode:
    def __init__(self, node_id: int, vector: np.ndarray, label: str = ""):
        self.id = node_id
        self.vector = vector.copy()
        self.label = label or f"c{node_id}"
        self.activation = 0.0
        self.salience = 0.3
        self.pressure = 0.0
        self.stability = 0.5
        self.confidence = 0.1
        self.timestamp = time.time()
        self.contradiction_count = 0

    def age(self) -> float:
        return time.time() - self.timestamp

    def decay(self, rate=0.01):
        self.activation *= (1.0 - rate * self.age())
        self.timestamp = time.time()

    @property
    def plasticity(self):
        return 1.0 - self.stability

    def __repr__(self):
        return (f"<Node {self.id} '{self.label}' act={self.activation:.3f} "
                f"conf={self.confidence:.3f} stab={self.stability:.3f}>")


class ConceptEdge:
    def __init__(self, source: int, target: int, weight: float = 0.5, shortcut: bool = False):
        self.source = source
        self.target = target
        self.weight = max(0.0, min(1.0, weight))
        self.confidence = 0.1
        self.pressure = 0.0
        self.stability = 0.3
        self.timestamp = time.time()
        self.prediction_count = 0
        self.shortcut = shortcut  # context→target edges are exempt from competition

    @property
    def plasticity(self):
        return 1.0 - self.stability

    def __repr__(self):
        return f"<Edge {self.source}→{self.target} w={self.weight:.3f} conf={self.confidence:.3f} {'[S]' if self.shortcut else ''}>"


class ConceptGraph:
    def __init__(self, dim: int = 64, max_nodes: int = 10000):
        self.dim = dim
        self.max_nodes = max_nodes
        self.nodes: Dict[int, ConceptNode] = {}
        self.edges: Dict[Tuple[int, int], ConceptEdge] = {}
        self.next_id = 0
        self.total_pressure = 0.0
        self.contradiction_hotspots: Set[int] = set()

    # ── node management ──

    def add_node(self, vector: Optional[np.ndarray] = None, label: str = "") -> ConceptNode:
        if len(self.nodes) >= self.max_nodes:
            self._prune_oldest()
        nid = self.next_id
        self.next_id += 1
        v = vector.copy() if vector is not None else np.random.randn(self.dim).astype(np.float32) * 0.1
        node = ConceptNode(nid, v, label)
        self.nodes[nid] = node
        return node

    def get_node(self, nid: int) -> Optional[ConceptNode]:
        return self.nodes.get(nid)

    def remove_node(self, nid: int):
        if nid in self.nodes:
            del self.nodes[nid]
            self.edges = {k: v for k, v in self.edges.items() if k[0] != nid and k[1] != nid}

    # ── edge management ──

    def add_edge(self, source: int, target: int, weight: float = 0.5, shortcut: bool = False) -> ConceptEdge:
        key = (source, target)
        if key in self.edges:
            edge = self.edges[key]
            edge.weight = max(0.0, min(1.0, weight))
            if shortcut:
                edge.shortcut = True
            return edge
        edge = ConceptEdge(source, target, weight, shortcut=shortcut)
        self.edges[key] = edge
        return edge

    def get_edge(self, source: int, target: int) -> Optional[ConceptEdge]:
        return self.edges.get((source, target))

    def remove_edge(self, source: int, target: int):
        self.edges.pop((source, target), None)

    # ── activation ──

    def activate(self, nid: int, amount: float = 1.0):
        node = self.nodes.get(nid)
        if node:
            node.activation = min(1.0, node.activation + amount)

    def spread_activation(self, steps: int = 3, k_active: int = 7, decay: float = 0.5):
        for _ in range(steps):
            new_activations = {}
            for nid, node in self.nodes.items():
                if node.activation > 0.01:
                    outgoing = [(t, e) for (s, t), e in self.edges.items() if s == nid]
                    for target_id, edge in outgoing:
                        act = node.activation * edge.weight * decay
                        new_activations[target_id] = new_activations.get(target_id, 0.0) + act
            for nid, act in new_activations.items():
                if nid in self.nodes:
                    self.nodes[nid].activation = min(1.0, self.nodes[nid].activation + act)
            self._top_k_activation(k_active)

    def _top_k_activation(self, k: int):
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.activation, reverse=True)
        for node in sorted_nodes[k:]:
            node.activation = 0.0

    # ── similarity search ──

    def find_similar(self, vector: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        scores = []
        for nid, node in self.nodes.items():
            sim = np.dot(vector, node.vector) / (np.linalg.norm(vector) * np.linalg.norm(node.vector) + 1e-15)
            scores.append((nid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def bind_input(self, vector: np.ndarray, k: int = 5) -> List[int]:
        matches = self.find_similar(vector, k)
        for nid, sim in matches:
            self.activate(nid, sim)
        return [nid for nid, _ in matches]

    # ── pressure ──

    def apply_pressure(self, nid: int, amount: float):
        node = self.nodes.get(nid)
        if node:
            node.pressure += amount * node.salience * (1.0 - node.confidence)
            node.pressure = min(100.0, node.pressure)
            self.total_pressure += amount
            if node.pressure > 5.0:
                self.contradiction_hotspots.add(nid)

    def apply_prediction_error(self, predicted_nids: List[int], actual_vector: np.ndarray):
        for nid in predicted_nids:
            node = self.nodes.get(nid)
            if node is None:
                continue
            sim = np.dot(node.vector, actual_vector) / (np.linalg.norm(node.vector) * np.linalg.norm(actual_vector) + 1e-15)
            error = max(0.0, 1.0 - sim)
            if error > 0.3:
                node.contradiction_count += 1
            self.apply_pressure(nid, error)

    def adjust_vector(self, nid: int, delta: np.ndarray, lr: float = 0.1):
        node = self.nodes.get(nid)
        if node is None:
            return
        node.vector += delta * lr * node.plasticity
        norm = np.linalg.norm(node.vector)
        if norm > 0:
            node.vector /= norm

    def get_or_create_edge(self, source: int, target: int, weight: float = 0.3, shortcut: bool = False) -> ConceptEdge:
        key = (source, target)
        if key in self.edges:
            edge = self.edges[key]
            if shortcut:
                edge.shortcut = True
            return edge
        return self.add_edge(source, target, weight, shortcut=shortcut)

    # ── plasticity ──

    def hebbian_update(self, source_nid: int, target_nid: int, coactivation: float, lr: float = 0.01):
        edge = self.get_edge(source_nid, target_nid)
        if edge is None:
            if coactivation > 0.3:
                self.add_edge(source_nid, target_nid, coactivation * 0.5)
            return
        source = self.nodes.get(source_nid)
        target = self.nodes.get(target_nid)
        if source is None or target is None:
            return
        pred_error = 1.0 - edge.confidence
        delta = lr * source.activation * target.activation * pred_error * source.salience * target.plasticity
        edge.weight = max(0.0, min(1.0, edge.weight + delta))
        edge.confidence = min(1.0, edge.confidence + abs(delta) * 0.1)
        edge.prediction_count += 1
        edge.stability = min(1.0, edge.stability + 0.001)

    def anti_hebbian_update(self, source_nid: int, target_nid: int, lr: float = 0.01):
        edge = self.get_edge(source_nid, target_nid)
        if edge is None:
            return
        source = self.nodes.get(source_nid)
        if source is None:
            return
        delta = -lr * source.activation * edge.confidence
        edge.weight = max(0.0, min(1.0, edge.weight + delta))
        edge.confidence = max(0.0, edge.confidence - 0.05)
        edge.stability = max(0.0, edge.stability - 0.01)

    # ── structural plasticity ──

    def prune_edges(self, threshold: float = 0.05):
        to_remove = [k for k, e in self.edges.items() if e.confidence < threshold]
        for k in to_remove:
            del self.edges[k]
        return len(to_remove)

    def form_edges(self, coactivation_threshold: float = 0.5):
        formed = 0
        active_nodes = [n for n in self.nodes.values() if n.activation > 0.1]
        for i, a in enumerate(active_nodes):
            for b in active_nodes[i + 1:]:
                coact = a.activation * b.activation
                if coact > coactivation_threshold and self.get_edge(a.id, b.id) is None:
                    self.add_edge(a.id, b.id, coact * 0.3)
                    formed += 1
        return formed

    def _prune_oldest(self):
        oldest = min(self.nodes.values(), key=lambda n: n.timestamp)
        self.remove_node(oldest.id)

    # ── sleep support ──

    def get_hotspots(self, threshold: float = 3.0) -> List[int]:
        return [nid for nid in self.contradiction_hotspots
                if nid in self.nodes and self.nodes[nid].pressure > threshold]

    def clear_hotspots(self):
        self.contradiction_hotspots.clear()

    def reconcile_contradictions(self):
        reconciled = 0
        for nid in list(self.contradiction_hotspots):
            node = self.nodes.get(nid)
            if node is None:
                continue
            neighbors = [(s, e) for (s, t), e in self.edges.items() if t == nid]
            if not neighbors:
                continue
            conflicting_weights = [e.weight for _, e in neighbors]
            mean_w = np.mean(conflicting_weights)
            if node.contradiction_count > 3:
                node.stability = max(0.0, node.stability - 0.1)
                node.pressure *= 0.5
                node.contradiction_count = 0
                reconciled += 1
            else:
                node.pressure = max(0.0, node.pressure - 1.0)
        self.contradiction_hotspots.clear()
        return reconciled

    # ── state ──

    def reset_activation(self):
        for node in self.nodes.values():
            node.activation = 0.0

    def __repr__(self):
        return (f"<ConceptGraph nodes={len(self.nodes)} edges={len(self.edges)} "
                f"dim={self.dim} pressure={self.total_pressure:.2f}>")
