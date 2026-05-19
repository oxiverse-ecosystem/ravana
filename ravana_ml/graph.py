import numpy as np
import time
from typing import Any, Optional, List, Tuple, Dict, Set
from .tensor import StateTensor, RawTensor, tensor
from collections import defaultdict


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

        # Hierarchical abstraction fields
        self.level: int = 0  # 0 = leaf, higher = more abstract
        self.parent: Optional[int] = None  # parent concept ID
        self.children: Set[int] = set()  # child concept IDs
        self.abstraction_degree: float = 0.0  # 0.0 = raw, 1.0 = fully compressed

    def age(self) -> float:
        return time.time() - self.timestamp

    def decay(self, rate=0.01):
        self.activation *= (1.0 - rate * self.age())
        self.timestamp = time.time()

    @property
    def plasticity(self):
        return 1.0 - self.stability

    def __repr__(self):
        hierarchy = f" L{self.level}" if self.level > 0 else ""
        children = f" [{len(self.children)}ch]" if self.children else ""
        return (f"<Node {self.id} '{self.label}' act={self.activation:.3f} "
                f"conf={self.confidence:.3f} stab={self.stability:.3f}"
                f"{hierarchy}{children}>")


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
            node = self.nodes[nid]
            # Unlink from parent
            if node.parent is not None and node.parent in self.nodes:
                self.nodes[node.parent].children.discard(nid)
            # Orphan children (move them up one level)
            for child_id in node.children:
                child = self.nodes.get(child_id)
                if child:
                    child.parent = node.parent
                    if node.parent is not None and node.parent in self.nodes:
                        self.nodes[node.parent].children.add(child_id)
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
            # Hierarchical upward propagation: children activate parents
            self._propagate_upward(decay=0.3)
            self._top_k_activation(k_active)

    def _propagate_upward(self, decay: float = 0.3):
        """Propagate activation from children to their parent concepts."""
        parent_activations: Dict[int, float] = {}
        for nid, node in self.nodes.items():
            if node.activation > 0.01 and node.parent is not None:
                parent_activations[node.parent] = (
                    parent_activations.get(node.parent, 0.0)
                    + node.activation * decay
                )
        for parent_id, act in parent_activations.items():
            if parent_id in self.nodes:
                self.nodes[parent_id].activation = min(
                    1.0, self.nodes[parent_id].activation + act
                )

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

    # ── hierarchy traversal ──

    def get_children(self, nid: int) -> List[int]:
        """Get direct children of a concept."""
        node = self.nodes.get(nid)
        if node is None:
            return []
        return list(node.children)

    def get_leaves(self, nid: int) -> List[int]:
        """Get all leaf descendants of a concept (recursive)."""
        node = self.nodes.get(nid)
        if node is None:
            return []
        if not node.children:
            return [nid]
        leaves = []
        for child_id in node.children:
            leaves.extend(self.get_leaves(child_id))
        return leaves

    def get_ancestors(self, nid: int) -> List[int]:
        """Get all ancestors from node to root."""
        ancestors = []
        current = self.nodes.get(nid)
        while current and current.parent is not None:
            ancestors.append(current.parent)
            current = self.nodes.get(current.parent)
        return ancestors

    def get_level(self, nid: int) -> int:
        """Get abstraction level of a concept."""
        node = self.nodes.get(nid)
        return node.level if node else 0

    def get_siblings(self, nid: int) -> List[int]:
        """Get siblings (same parent) of a concept."""
        node = self.nodes.get(nid)
        if node is None or node.parent is None:
            return []
        parent = self.nodes.get(node.parent)
        if parent is None:
            return []
        return [c for c in parent.children if c != nid]

    # ── hierarchical abstraction ──

    def merge_concepts(
        self,
        child_ids: List[int],
        label: str = "",
        abstraction_degree: float = 0.5,
    ) -> Optional[int]:
        """
        Create a parent concept by merging child concepts.

        The parent's vector is the centroid of its children. Edges from/to
        children are aggregated to the parent. Children retain their edges
        but gain a parent pointer.

        Args:
            child_ids: Concept IDs to merge (must be >= 2)
            label: Label for the new parent concept
            abstraction_degree: How compressed this abstraction is (0-1)

        Returns:
            Parent concept ID, or None if merge is not possible
        """
        # Validate
        valid_children = [cid for cid in child_ids if cid in self.nodes]
        if len(valid_children) < 2:
            return None

        # Don't merge if any child already has a parent at the same level
        # (prevent double-merging)
        for cid in valid_children:
            child = self.nodes[cid]
            if child.parent is not None:
                # Check if the parent is also being merged — that's fine
                if child.parent not in valid_children:
                    return None

        child_nodes = [self.nodes[cid] for cid in valid_children]

        # Compute parent level: max child level + 1
        max_child_level = max(n.level for n in child_nodes)

        # Compute parent vector: centroid of children
        child_vectors = np.array([n.vector for n in child_nodes])
        parent_vector = np.mean(child_vectors, axis=0).astype(np.float32)
        norm = np.linalg.norm(parent_vector)
        if norm > 0:
            parent_vector /= norm

        # Create parent node
        parent_label = label or f"abs_{'_'.join(str(c) for c in valid_children[:3])}"
        parent = self.add_node(parent_vector, parent_label)
        parent.level = max_child_level + 1
        parent.abstraction_degree = abstraction_degree
        parent.children = set(valid_children)
        parent.salience = np.mean([n.salience for n in child_nodes])
        parent.confidence = np.mean([n.confidence for n in child_nodes])
        parent.stability = np.mean([n.stability for n in child_nodes])

        # Link children to parent
        for child in child_nodes:
            child.parent = parent.id

        # Aggregate edges: collect all external edges from/to children
        # and create weighted edges from/to the parent
        self._aggregate_child_edges(parent.id, valid_children)

        return parent.id

    def _aggregate_child_edges(self, parent_id: int, child_ids: List[int]):
        """
        Aggregate external edges from children to the parent node.

        For each external target connected to any child, create an edge
        from parent to that target with weight = mean child edge weight.
        Similarly for incoming edges.
        """
        child_set = set(child_ids)

        # Outgoing: child → external target
        outgoing: Dict[int, List[float]] = defaultdict(list)
        # Incoming: external source → child
        incoming: Dict[int, List[float]] = defaultdict(list)

        for (src, tgt), edge in self.edges.items():
            if src in child_set and tgt not in child_set:
                outgoing[tgt].append(edge.weight)
            elif tgt in child_set and src not in child_set:
                incoming[src].append(edge.weight)

        # Create aggregated edges to parent
        for target_id, weights in outgoing.items():
            mean_weight = float(np.mean(weights))
            self.get_or_create_edge(parent_id, target_id, mean_weight)

        for source_id, weights in incoming.items():
            mean_weight = float(np.mean(weights))
            self.get_or_create_edge(source_id, parent_id, mean_weight)

    def find_coactivated_clusters(
        self,
        coactivation_threshold: float = 0.5,
        min_cluster_size: int = 3,
        max_cluster_size: int = 8,
    ) -> List[List[int]]:
        """
        Find clusters of frequently co-activated leaf concepts.

        Uses a simple greedy approach: for each highly active node,
        find its co-activated neighbors and group them.

        Returns:
            List of concept ID lists (each list is a merge candidate)
        """
        active_leaves = [
            n for n in self.nodes.values()
            if n.activation > 0.1 and n.level == 0 and n.parent is None
        ]

        if len(active_leaves) < min_cluster_size:
            return []

        # Build co-activation adjacency from edges
        coactive_pairs: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for node in active_leaves:
            outgoing = [(t, e) for (s, t), e in self.edges.items() if s == node.id]
            for target_id, edge in outgoing:
                target = self.nodes.get(target_id)
                if target and target.level == 0 and target.parent is None:
                    coact = node.activation * target.activation * edge.weight
                    if coact > coactivation_threshold:
                        coactive_pairs[node.id].append((target_id, coact))

        # Greedy clustering: seed from highest-activity nodes
        clusters = []
        used = set()
        sorted_nodes = sorted(active_leaves, key=lambda n: n.activation, reverse=True)

        for seed in sorted_nodes:
            if seed.id in used:
                continue

            cluster = [seed.id]
            used.add(seed.id)

            # Add co-activated neighbors
            neighbors = coactive_pairs.get(seed.id, [])
            neighbors.sort(key=lambda x: x[1], reverse=True)

            for neighbor_id, _ in neighbors:
                if len(cluster) >= max_cluster_size:
                    break
                if neighbor_id not in used:
                    # Check that neighbor is co-activated with most of the cluster
                    sim_count = 0
                    for existing_id in cluster:
                        for nid, _ in coactive_pairs.get(existing_id, []):
                            if nid == neighbor_id:
                                sim_count += 1
                                break
                    if sim_count >= max(1, len(cluster) - 1):
                        cluster.append(neighbor_id)
                        used.add(neighbor_id)

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters

    def compute_compression_ratio(self) -> float:
        """
        Compute the abstraction compression ratio.

        Returns the fraction of nodes that are abstract (level > 0).
        """
        if not self.nodes:
            return 0.0
        abstract_count = sum(1 for n in self.nodes.values() if n.level > 0)
        return abstract_count / len(self.nodes)

    def get_abstraction_stats(self) -> Dict[str, Any]:
        """Get full abstraction statistics."""
        levels = [n.level for n in self.nodes.values()]
        abstract_nodes = [n for n in self.nodes.values() if n.level > 0]
        return {
            "total_nodes": len(self.nodes),
            "leaf_nodes": sum(1 for l in levels if l == 0),
            "abstract_nodes": len(abstract_nodes),
            "max_level": max(levels) if levels else 0,
            "compression_ratio": self.compute_compression_ratio(),
            "mean_abstraction_degree": (
                float(np.mean([n.abstraction_degree for n in abstract_nodes]))
                if abstract_nodes else 0.0
            ),
        }

    # ── state ──

    def reset_activation(self):
        for node in self.nodes.values():
            node.activation = 0.0

    def __repr__(self):
        return (f"<ConceptGraph nodes={len(self.nodes)} edges={len(self.edges)} "
                f"dim={self.dim} pressure={self.total_pressure:.2f}>")
