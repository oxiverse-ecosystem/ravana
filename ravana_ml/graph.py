import numpy as np
import time
from typing import Any, Optional, List, Tuple, Dict, Set
from .tensor import StateTensor, RawTensor, tensor
from collections import defaultdict


class ConceptNode:
    def __init__(self, node_id: int, vector: np.ndarray, label: str = ""):
        self.id = node_id
        self.vector = vector.copy()
        self.genesis_vector = vector.copy()  # original vector for drift tracking
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

        # Temporal tracking
        self.last_activated: float = 0.0  # most recent activation timestamp
        self.activation_history: List[float] = []  # last 100 activation timestamps
        self.temporal_context: Optional[np.ndarray] = None  # context at last activation

    @property
    def drift_magnitude(self) -> float:
        """L2 distance between current vector and genesis vector.

        High drift means the concept has shifted significantly from its
        original meaning — it may be time to split or re-evaluate.
        """
        diff = self.vector - self.genesis_vector
        return float(np.linalg.norm(diff))

    def age(self) -> float:
        return time.time() - self.timestamp

    def decay(self, rate=0.01):
        self.activation *= (1.0 - rate * self.age())
        self.timestamp = time.time()

    def record_activation(self, context_vector: Optional[np.ndarray] = None):
        """Record an activation event in temporal history."""
        now = time.time()
        self.last_activated = now
        self.activation_history.append(now)
        # Keep rolling window of 100
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-100:]
        if context_vector is not None:
            self.temporal_context = context_vector.copy()

    def recency_score(self, decay_rate: float = 0.1) -> float:
        """How recently was this concept activated? Exponential decay from last activation.

        Returns 1.0 if just activated, approaches 0.0 over time.
        """
        if self.last_activated <= 0:
            return 0.0
        elapsed = time.time() - self.last_activated
        return float(np.exp(-decay_rate * elapsed / 3600.0))  # decay per hour

    def frequency_score(self, window_seconds: float = 86400.0) -> float:
        """What fraction of activations happened in the recent window?

        Returns 0.0 if never activated, up to 1.0 if all activations are recent.
        """
        if not self.activation_history:
            return 0.0
        cutoff = time.time() - window_seconds
        recent = sum(1 for t in self.activation_history if t > cutoff)
        return recent / len(self.activation_history)

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
    def __init__(self, source: int, target: int, weight: float = 0.5,
                 shortcut: bool = False, edge_type: str = "excitatory"):
        self.source = source
        self.target = target
        self.weight = max(0.0, min(1.0, weight))
        self.confidence = 0.1
        self.pressure = 0.0
        self.stability = 0.3
        self.timestamp = time.time()
        self.prediction_count = 0
        self.shortcut = shortcut  # context→target edges are exempt from competition
        self.edge_type = edge_type  # "excitatory" or "inhibitory"

    @property
    def effective_weight(self) -> float:
        """Weight with sign applied for inhibitory edges."""
        if self.edge_type == "inhibitory":
            return -self.weight
        return self.weight

    @property
    def plasticity(self):
        return 1.0 - self.stability

    def __repr__(self):
        inh = " [I]" if self.edge_type == "inhibitory" else ""
        return f"<Edge {self.source}->{self.target} w={self.weight:.3f} conf={self.confidence:.3f} {'[S]' if self.shortcut else ''}{inh}>"


class ConceptBinding:
    """Probabilistic mapping between tokens, concepts, and memories.

    A single binding links a token (or memory) to a concept node with
    confidence, source tracking, and reinforcement history. Supports
    the unified semantic namespace: language ↔ conceptual ↔ autobiographical.

    Bindings can drift, split, merge, and decay — they are not static
    dictionary entries but living semantic links.
    """

    __slots__ = ('token_id', 'concept_id', 'confidence', 'source',
                 'reinforcement_count', 'last_used', 'created_at',
                 'decay_score', 'ambiguity')

    def __init__(self, token_id: int, concept_id: int,
                 confidence: float = 0.5, source: str = "learned"):
        self.token_id = token_id
        self.concept_id = concept_id
        self.confidence = max(0.0, min(1.0, confidence))
        self.source = source  # "learned", "memory", "manual", "inferred"
        self.reinforcement_count = 0
        self.last_used = time.time()
        self.created_at = time.time()
        self.decay_score = 0.0
        self.ambiguity = 0.0  # 0 = unambiguous, 1 = highly ambiguous

    def reinforce(self, amount: float = 0.05):
        """Strengthen this binding through use."""
        self.confidence = min(1.0, self.confidence + amount)
        self.reinforcement_count += 1
        self.last_used = time.time()
        self.decay_score = max(0.0, self.decay_score - 0.1)
        self.ambiguity = max(0.0, self.ambiguity - 0.02)

    def decay(self, rate: float = 0.01):
        """Apply time-based decay to this binding."""
        age = time.time() - self.last_used
        self.decay_score += rate * (age / 3600.0)  # decay per hour
        if self.decay_score > 5.0:
            self.confidence *= 0.95  # weaken on heavy decay

    @property
    def strength(self) -> float:
        """Effective binding strength = confidence * (1 - decay/10)."""
        return self.confidence * max(0.0, 1.0 - self.decay_score / 10.0)

    def __repr__(self):
        return (f"<Binding tok={self.token_id}→con={self.concept_id} "
                f"conf={self.confidence:.3f} str={self.strength:.3f} "
                f"src={self.source} rein={self.reinforcement_count}>")


class ConceptBindingMap:
    """Manages the full token ↔ concept ↔ memory binding space.

    Supports:
    - Multiple bindings per token (ambiguous meanings)
    - Confidence-weighted lookup
    - Ambiguity detection (multiple high-confidence bindings)
    - Decay and reinforcement
    - Split/merge when concepts evolve
    """

    def __init__(self):
        # token_id -> list of bindings (sorted by confidence)
        self._by_token: Dict[int, List[ConceptBinding]] = {}
        # concept_id -> list of bindings
        self._by_concept: Dict[int, List[ConceptBinding]] = {}
        # (token_id, concept_id) -> binding (fast lookup)
        self._index: Dict[Tuple[int, int], ConceptBinding] = {}

    def bind(self, token_id: int, concept_id: int,
             confidence: float = 0.5, source: str = "learned") -> ConceptBinding:
        """Create or reinforce a binding."""
        key = (token_id, concept_id)
        if key in self._index:
            binding = self._index[key]
            binding.reinforce(confidence * 0.1)
            return binding

        binding = ConceptBinding(token_id, concept_id, confidence, source)
        self._index[key] = binding
        self._by_token.setdefault(token_id, []).append(binding)
        self._by_concept.setdefault(concept_id, []).append(binding)
        # Sort by confidence descending
        self._by_token[token_id].sort(key=lambda b: -b.confidence)
        self._by_concept[concept_id].sort(key=lambda b: -b.confidence)
        return binding

    def get_concepts(self, token_id: int, min_confidence: float = 0.1) -> List[ConceptBinding]:
        """Get all concept bindings for a token, sorted by confidence."""
        bindings = self._by_token.get(token_id, [])
        return [b for b in bindings if b.strength >= min_confidence]

    def get_tokens(self, concept_id: int, min_confidence: float = 0.1) -> List[ConceptBinding]:
        """Get all token bindings for a concept."""
        bindings = self._by_concept.get(concept_id, [])
        return [b for b in bindings if b.strength >= min_confidence]

    def best_concept(self, token_id: int) -> Optional[int]:
        """Get the strongest concept for a token, or None."""
        bindings = self.get_concepts(token_id)
        return bindings[0].concept_id if bindings else None

    def best_token(self, concept_id: int) -> Optional[int]:
        """Get the strongest token for a concept, or None."""
        bindings = self.get_tokens(concept_id)
        return bindings[0].token_id if bindings else None

    def is_ambiguous(self, token_id: int, threshold: float = 0.3) -> bool:
        """Check if a token has multiple strong concept bindings."""
        bindings = self.get_concepts(token_id, min_confidence=threshold)
        return len(bindings) > 1

    def ambiguity_score(self, token_id: int) -> float:
        """How ambiguous is this token? 0 = unambiguous, 1 = highly ambiguous."""
        bindings = self.get_concepts(token_id, min_confidence=0.1)
        if len(bindings) <= 1:
            return 0.0
        # Entropy-like measure over binding strengths
        strengths = [b.strength for b in bindings]
        total = sum(strengths)
        if total == 0:
            return 0.0
        probs = [s / total for s in strengths]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
        max_entropy = np.log2(len(bindings))
        return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0

    def decay_all(self, rate: float = 0.01):
        """Apply decay to all bindings."""
        for binding in self._index.values():
            binding.decay(rate)

    def split_bindings(self, parent_concept_id: int, child_a_id: int, child_b_id: int,
                       criterion_fn) -> None:
        """Redistribute bindings from a parent concept to its split children.

        Args:
            parent_concept_id: The concept that was split
            child_a_id, child_b_id: The two new child concepts
            criterion_fn: Callable that takes a vector and returns True for child_a, False for child_b
        """
        bindings = self._by_concept.get(parent_concept_id, [])
        for binding in bindings:
            # Create new bindings for each child
            token_id = binding.token_id
            # The criterion uses the binding's context or the token's associated vector
            # Since we don't store vectors in bindings, we use confidence as a proxy
            # and let the caller provide the criterion
            self.bind(token_id, child_a_id, confidence=binding.confidence * 0.5, source="split")
            self.bind(token_id, child_b_id, confidence=binding.confidence * 0.5, source="split")

    def disambiguate(self, token_id: int, context_vector: np.ndarray,
                     graph: 'ConceptGraph', suppression_rate: float = 0.1) -> Optional[int]:
        """Resolve ambiguity by suppressing competing bindings based on context.

        When a token maps to multiple concepts (e.g., "python" → snake/programming),
        the current context determines which meaning wins. Losing bindings have
        their confidence decayed faster — like semantic suppression in the brain.

        Args:
            token_id: The ambiguous token
            context_vector: Current context (e.g., surrounding concept activations)
            graph: ConceptGraph to look up concept vectors
            suppression_rate: How much to suppress losing bindings

        Returns:
            The winning concept_id, or None if no bindings
        """
        bindings = self.get_concepts(token_id, min_confidence=0.1)
        if len(bindings) <= 1:
            return bindings[0].concept_id if bindings else None

        # Score each binding by context similarity
        scored = []
        for b in bindings:
            node = graph.get_node(b.concept_id)
            if node is None:
                scored.append((b, -1.0))
                continue
            sim = np.dot(context_vector, node.vector) / (
                np.linalg.norm(context_vector) * np.linalg.norm(node.vector) + 1e-15
            )
            scored.append((b, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        winner = scored[0][0]

        # Suppress losing bindings
        for binding, sim in scored[1:]:
            # More similar to context = less suppression (they're plausible alternatives)
            suppression = suppression_rate * (1.0 - max(0.0, sim))
            binding.confidence = max(0.0, binding.confidence - suppression)
            binding.ambiguity = min(1.0, binding.ambiguity + 0.05)

        # Reinforce winner
        winner.reinforce(0.02)
        winner.ambiguity = max(0.0, winner.ambiguity - 0.05)

        # Re-sort bindings by confidence since values changed
        if token_id in self._by_token:
            self._by_token[token_id].sort(key=lambda b: b.confidence, reverse=True)

        return winner.concept_id

    def prune(self, min_strength: float = 0.05):
        """Remove bindings that have decayed below threshold."""
        to_remove = [(tok, con) for (tok, con), b in self._index.items()
                     if b.strength < min_strength]
        for tok, con in to_remove:
            key = (tok, con)
            binding = self._index.pop(key)
            if tok in self._by_token:
                self._by_token[tok] = [b for b in self._by_token[tok] if b is not binding]
            if con in self._by_concept:
                self._by_concept[con] = [b for b in self._by_concept[con] if b is not binding]
        return len(to_remove)

    def __len__(self):
        return len(self._index)

    def __contains__(self, key: Tuple[int, int]):
        return key in self._index


class ConceptGraph:
    def __init__(self, dim: int = 64, max_nodes: int = 10000):
        self.dim = dim
        self.max_nodes = max_nodes
        self.nodes: Dict[int, ConceptNode] = {}
        self.edges: Dict[Tuple[int, int], ConceptEdge] = {}
        self.next_id = 0
        self.total_pressure = 0.0
        self.contradiction_hotspots: Set[int] = set()

        # Temporal context: a drifting vector that represents "when" we are
        # Slowly shifts toward the centroid of currently active concepts
        self.temporal_context: np.ndarray = np.zeros(dim, dtype=np.float32)
        self.temporal_context_drift_rate: float = 0.05

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

    def add_edge(self, source: int, target: int, weight: float = 0.5,
                 shortcut: bool = False, edge_type: str = "excitatory") -> ConceptEdge:
        key = (source, target)
        if key in self.edges:
            edge = self.edges[key]
            edge.weight = max(0.0, min(1.0, weight))
            if shortcut:
                edge.shortcut = True
            if edge_type == "inhibitory":
                edge.edge_type = "inhibitory"
            return edge
        edge = ConceptEdge(source, target, weight, shortcut=shortcut, edge_type=edge_type)
        self.edges[key] = edge
        return edge

    def get_edge(self, source: int, target: int) -> Optional[ConceptEdge]:
        return self.edges.get((source, target))

    def remove_edge(self, source: int, target: int):
        self.edges.pop((source, target), None)

    # ── activation ──

    def activate(self, nid: int, amount: float = 1.0, context_vector: Optional[np.ndarray] = None):
        node = self.nodes.get(nid)
        if node:
            node.activation = min(1.0, node.activation + amount)
            node.record_activation(context_vector)

    def update_temporal_context(self):
        """Drift temporal context toward the centroid of currently active concepts.

        Called after each cognitive step. The temporal context slowly shifts to
        reflect the current "era" of processing — enabling time-sensitive retrieval.
        """
        active_nodes = [n for n in self.nodes.values() if n.activation > 0.1]
        if not active_nodes:
            return
        centroid = np.mean([n.vector * n.activation for n in active_nodes], axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        # Exponential moving average
        self.temporal_context = (
            (1.0 - self.temporal_context_drift_rate) * self.temporal_context +
            self.temporal_context_drift_rate * centroid
        )
        norm = np.linalg.norm(self.temporal_context)
        if norm > 0:
            self.temporal_context /= norm

    def temporal_context_similarity(self, node: ConceptNode) -> float:
        """How similar is this node's stored context to the current temporal context?

        Enables encoding specificity: memories are easier to recall in the
        context where they were encoded (Tulving, 1973).
        """
        if node.temporal_context is None:
            return 0.0
        sim = np.dot(self.temporal_context, node.temporal_context) / (
            np.linalg.norm(self.temporal_context) * np.linalg.norm(node.temporal_context) + 1e-15
        )
        return max(0.0, sim)

    def spread_activation(self, steps: int = 3, k_active: int = 7, decay: float = 0.5):
        for _ in range(steps):
            new_activations = {}
            # Fan effect: compute in-degree for normalization
            in_degree: Dict[int, int] = defaultdict(int)
            for (s, t) in self.edges:
                in_degree[t] += 1

            for nid, node in self.nodes.items():
                if node.activation > 0.01:
                    outgoing = [(t, e) for (s, t), e in self.edges.items() if s == nid]
                    for target_id, edge in outgoing:
                        # Precision weighting: edge.confidence modulates signal strength
                        act = node.activation * edge.weight * edge.confidence * decay
                        if edge.edge_type == "inhibitory":
                            act = -act
                        # Fan effect: normalize by in-degree (hub nodes receive weaker per-edge activation)
                        fan_factor = 1.0 / (in_degree[target_id] ** 0.5 + 1.0)
                        act *= fan_factor
                        new_activations[target_id] = new_activations.get(target_id, 0.0) + act
            for nid, act in new_activations.items():
                if nid in self.nodes:
                    self.nodes[nid].activation = max(0.0, min(1.0, self.nodes[nid].activation + act))
            # Hierarchical upward propagation: children activate parents
            self._propagate_upward(decay=0.3)
            self._soft_lateral_inhibition(k_active)

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
        """Legacy hard winner-take-all — prefer _soft_lateral_inhibition."""
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.activation, reverse=True)
        for node in sorted_nodes[k:]:
            node.activation = 0.0

    def _soft_lateral_inhibition(self, k: int, inhibition_strength: float = 0.5):
        """Soft lateral inhibition: each active concept suppresses others proportionally to similarity.

        Unlike hard winner-take-all, this preserves near-winners — you can be
        partially aware of alternative meanings. Closer to biological cortical dynamics.
        """
        active_nodes = [(nid, n) for nid, n in self.nodes.items() if n.activation > 0.01]
        if len(active_nodes) <= 1:
            return

        # Sort by activation descending, keep top-k as "winners"
        active_nodes.sort(key=lambda x: x[1].activation, reverse=True)
        winners = active_nodes[:k]

        # Compute suppression for each active node from all other active nodes
        for nid, node in active_nodes:
            suppression = 0.0
            for other_nid, other_node in active_nodes:
                if other_nid == nid:
                    continue
                # Similarity-weighted suppression
                sim = np.dot(node.vector, other_node.vector) / (
                    np.linalg.norm(node.vector) * np.linalg.norm(other_node.vector) + 1e-15
                )
                sim = max(0.0, sim)  # only positive similarity suppresses
                suppression += other_node.activation * sim * inhibition_strength
            # Apply suppression: reduce activation but don't zero it
            node.activation = max(0.0, node.activation / (1.0 + suppression))

        # Ensure at least top-k have meaningful activation (rescue near-winners)
        for nid, node in winners:
            if node.activation < 0.01:
                node.activation = 0.01

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

    def get_or_create_edge(self, source: int, target: int, weight: float = 0.3,
                           shortcut: bool = False, edge_type: str = "excitatory") -> ConceptEdge:
        key = (source, target)
        if key in self.edges:
            edge = self.edges[key]
            if shortcut:
                edge.shortcut = True
            if edge_type == "inhibitory":
                edge.edge_type = "inhibitory"
            return edge
        return self.add_edge(source, target, weight, shortcut=shortcut, edge_type=edge_type)

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
        # Surprise-driven learning rate: high-confidence errors produce bigger updates
        # (like the brain's error-related negativity signal)
        # Surprise = how wrong we were * how confident we were about it
        surprise = abs(pred_error) * edge.confidence
        effective_lr = lr * (1.0 + surprise * 5.0)
        if edge.edge_type == "inhibitory":
            delta = effective_lr * source.activation * target.activation * pred_error * source.salience * target.plasticity
            edge.weight = min(1.0, edge.weight + delta)
            edge.confidence = min(1.0, edge.confidence + abs(delta) * 0.1)
        else:
            delta = effective_lr * source.activation * target.activation * pred_error * source.salience * target.plasticity
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
        # When excitatory edge dies from persistent mismatch, convert to inhibitory
        # instead of deleting — the mismatch itself is information
        if edge.confidence < 0.01 and edge.edge_type == "excitatory":
            edge.edge_type = "inhibitory"
            edge.weight = 0.1  # small initial inhibitory weight
            edge.confidence = 0.1
            edge.stability = 0.1

    def form_inhibitory_edges(self, contradiction_threshold: int = 3):
        """Form inhibitory edges between concepts with persistent contradictions.

        When two concepts repeatedly produce prediction errors together
        (high co-activation but mismatched expectations), they should
        inhibit each other — like semantic suppression in the brain.
        """
        formed = 0
        for nid in list(self.contradiction_hotspots):
            node = self.nodes.get(nid)
            if node is None or node.contradiction_count < contradiction_threshold:
                continue
            # Find co-activated neighbors that contributed to contradiction
            for (src, tgt), edge in self.edges.items():
                if src == nid:
                    target = self.nodes.get(tgt)
                    if target and target.activation > 0.1 and target.contradiction_count > 0:
                        # Check if excitatory edge exists — if so, it's a candidate for inhibition
                        existing = self.get_edge(nid, tgt)
                        if existing and existing.edge_type == "excitatory" and existing.confidence < 0.3:
                            # Convert to inhibitory
                            existing.edge_type = "inhibitory"
                            existing.weight = 0.2
                            existing.confidence = 0.2
                            formed += 1
                        elif existing is None:
                            # Create new inhibitory edge
                            self.add_edge(nid, tgt, 0.2, edge_type="inhibitory")
                            formed += 1
        return formed

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

    # ── concept splitting ──

    def should_split(self, nid: int, contradiction_threshold: int = 3,
                     drift_threshold: float = 0.5, entropy_threshold: float = 2.0) -> bool:
        """Check if a concept has accumulated enough internal contradiction to split.

        A concept should split when:
        - High contradiction count (many prediction errors)
        - High drift from original meaning
        - High edge entropy (edges point to diverse, unrelated targets)
        """
        node = self.nodes.get(nid)
        if node is None or node.level > 0:  # don't split abstract parent nodes
            return False

        reasons = 0

        # Contradiction pressure
        if node.contradiction_count >= contradiction_threshold:
            reasons += 1

        # Drift from genesis
        if node.drift_magnitude >= drift_threshold:
            reasons += 1

        # Edge entropy: how diverse are the targets of outgoing edges?
        outgoing = [t for (s, t), e in self.edges.items() if s == nid and e.edge_type == "excitatory"]
        if len(outgoing) >= 3:
            # Compute pairwise distances between target vectors
            targets = [self.nodes[t] for t in outgoing if t in self.nodes]
            if len(targets) >= 3:
                dists = []
                for i, a in enumerate(targets):
                    for b in targets[i + 1:]:
                        sim = np.dot(a.vector, b.vector) / (
                            np.linalg.norm(a.vector) * np.linalg.norm(b.vector) + 1e-15
                        )
                        dists.append(1.0 - sim)
                if dists:
                    mean_dist = np.mean(dists)
                    if mean_dist > 0.7:  # targets are very diverse
                        reasons += 1

        return reasons >= 2  # need at least 2 signals to split

    def split_concept(self, nid: int, binding_map: Optional['ConceptBindingMap'] = None) -> Tuple[int, int]:
        """Split a concept into two competing sub-concepts.

        Creates two children from the parent, distributes edges based on
        vector alignment, forms an inhibitory edge between them, and
        optionally redistributes bindings.

        Returns:
            (child_a_id, child_b_id)
        """
        parent = self.nodes.get(nid)
        if parent is None:
            raise ValueError(f"Node {nid} not found")

        # Find the two most divergent neighbors to seed the split
        outgoing = [(t, e) for (s, t), e in self.edges.items()
                    if s == nid and e.edge_type == "excitatory"]
        if len(outgoing) < 2:
            # Not enough structure to split meaningfully — create orthogonal children
            vec_a = parent.vector.copy()
            vec_b = -parent.vector.copy()
        else:
            # Find the two most dissimilar target vectors
            targets = [(t, self.nodes[t]) for t, _ in outgoing if t in self.nodes]
            max_dist = 0.0
            best_pair = (targets[0][0], targets[1][0]) if len(targets) >= 2 else (None, None)
            for i, (t1, n1) in enumerate(targets):
                for t2, n2 in targets[i + 1:]:
                    dist = np.linalg.norm(n1.vector - n2.vector)
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (t1, t2)

            # Seed children toward the two divergent targets
            n_a = self.nodes.get(best_pair[0])
            n_b = self.nodes.get(best_pair[1])
            if n_a is not None and n_b is not None:
                vec_a = 0.5 * parent.vector + 0.5 * n_a.vector
                vec_b = 0.5 * parent.vector + 0.5 * n_b.vector
            else:
                vec_a = parent.vector.copy()
                vec_b = -parent.vector.copy()

        # Normalize
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a > 0:
            vec_a /= norm_a
        if norm_b > 0:
            vec_b /= norm_b

        # Create children
        child_a = self.add_node(vec_a, label=f"{parent.label}_a")
        child_b = self.add_node(vec_b, label=f"{parent.label}_b")

        # Copy parent properties to children
        for child in [child_a, child_b]:
            child.level = parent.level  # same level as parent
            child.salience = parent.salience
            child.confidence = parent.confidence * 0.5  # reduced — they're new interpretations
            child.stability = parent.stability * 0.5

        # Make parent abstract
        parent.children = {child_a.id, child_b.id}
        parent.abstraction_degree = 0.5
        parent.level += 1

        # Distribute edges: align each edge to the closer child
        for target_id, edge in list(self.edges.items()):
            if edge.source != nid:
                continue
            target_node = self.nodes.get(target_id)
            if target_node is None:
                continue

            # Which child is closer to the target?
            sim_a = np.dot(vec_a, target_node.vector) / (
                np.linalg.norm(vec_a) * np.linalg.norm(target_node.vector) + 1e-15
            )
            sim_b = np.dot(vec_b, target_node.vector) / (
                np.linalg.norm(vec_b) * np.linalg.norm(target_node.vector) + 1e-15
            )

            chosen_child = child_a if sim_a >= sim_b else child_b
            self.add_edge(chosen_child.id, target_id, edge.weight,
                         shortcut=edge.shortcut, edge_type=edge.edge_type)

        # Inhibitory edge between children (they're competing interpretations)
        self.add_edge(child_a.id, child_b.id, 0.3, edge_type="inhibitory")
        self.add_edge(child_b.id, child_a.id, 0.3, edge_type="inhibitory")

        # Redistribute bindings if binding map provided
        if binding_map is not None:
            binding_map.split_bindings(nid, child_a.id, child_b.id,
                                       lambda vec: np.dot(vec, vec_a) > np.dot(vec, vec_b))

        return child_a.id, child_b.id

    def homeostatic_downscale(self, protection_threshold: float = 0.8,
                               downscale_factor: float = 0.8) -> Tuple[float, float]:
        """Global synaptic homeostasis — downscale all edges, protect the strong.

        During sleep, the brain globally weakens all synapses, but the strongest
        ones survive. This improves signal-to-noise ratio: important connections
        stand out more, noise is washed out.

        Args:
            protection_threshold: edges with stability above this are protected
            downscale_factor: multiply all weights by this

        Returns:
            (total_weight_before, total_weight_after)
        """
        total_before = sum(e.weight for e in self.edges.values())
        protected = {}

        # First pass: identify protected edges
        for key, edge in self.edges.items():
            if edge.stability >= protection_threshold:
                protected[key] = edge.weight

        # Second pass: downscale all
        for edge in self.edges.values():
            edge.weight *= downscale_factor

        # Third pass: restore protected edges
        for key, original_weight in protected.items():
            if key in self.edges:
                self.edges[key].weight = original_weight

        total_after = sum(e.weight for e in self.edges.values())
        return total_before, total_after

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
