import numpy as np
from typing import Optional
from .graph import ConceptGraph, ConceptNode, ConceptEdge


class HebbianPlasticity:
    def __init__(self, graph: ConceptGraph, lr: float = 0.01):
        self.graph = graph
        self.lr = lr

    def update(self, source_nid: int, target_nid: int):
        edge = self.graph.get_edge(source_nid, target_nid)
        source = self.graph.get_node(source_nid)
        target = self.graph.get_node(target_nid)
        if source is None or target is None:
            return 0.0
        pred_error = 1.0 - (edge.confidence if edge else 0.0)
        salience = source.salience * target.salience
        plasticity = max(source.plasticity, target.plasticity if edge else 0.3)
        coactivation = source.activation * target.activation
        delta = self.lr * coactivation * pred_error * salience * plasticity
        if edge:
            edge.weight = max(0.0, min(1.0, edge.weight + delta))
            edge.confidence = min(1.0, edge.confidence + abs(delta) * 0.1)
            edge.stability = min(1.0, edge.stability + 0.001)
        elif coactivation > 0.3:
            self.graph.add_edge(source_nid, target_nid, coactivation * 0.5)
        return delta


class AntiHebbianPlasticity:
    def __init__(self, graph: ConceptGraph, lr: float = 0.01):
        self.graph = graph
        self.lr = lr

    def update(self, source_nid: int, target_nid: int, persistent_mismatch: float = 1.0):
        edge = self.graph.get_edge(source_nid, target_nid)
        source = self.graph.get_node(source_nid)
        if edge is None or source is None:
            return 0.0
        delta = -self.lr * persistent_mismatch * edge.confidence * source.activation
        edge.weight = max(0.0, min(1.0, edge.weight + delta))
        edge.confidence = max(0.0, edge.confidence - 0.05)
        edge.stability = max(0.0, edge.stability - 0.02)
        # When excitatory edge dies from persistent mismatch, convert to inhibitory
        # instead of deleting — the mismatch itself is information
        if edge.confidence < 0.01:
            if edge.edge_type == "excitatory" and persistent_mismatch > 1.5:
                edge.edge_type = "inhibitory"
                edge.weight = 0.1
                edge.confidence = 0.1
                edge.stability = 0.1
            else:
                self.graph.remove_edge(source_nid, target_nid)
        return delta


class StructuralPlasticity:
    def __init__(self, graph: ConceptGraph,
                 prune_threshold: float = 0.05,
                 form_threshold: float = 0.5):
        self.graph = graph
        self.prune_threshold = prune_threshold
        self.form_threshold = form_threshold

    def step(self):
        pruned = self.graph.prune_edges(self.prune_threshold)
        formed = self.graph.form_edges(self.form_threshold)
        return pruned, formed

    def prune_by_age(self, max_age: float = 3600.0):
        now = __import__('time').time()
        to_remove = [k for k, e in self.graph.edges.items()
                     if now - e.timestamp > max_age and e.prediction_count < 3]
        for k in to_remove:
            del self.graph.edges[k]
        return len(to_remove)
