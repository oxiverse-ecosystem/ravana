"""
Synaptic Dynamics — continuous sigmoid modulation and short-term plasticity.

Replaces all hardcoded binary thresholds (0.20, 0.35, 0.10, 0.15, 0.40, 0.50, etc.)
with continuous sigmoid gating functions. Adds synaptic depression (STP) for
natural latching dynamics — the system's own activation history determines when
a concept becomes temporarily unavailable, not a fixed threshold.

Neuroscience basis:
- Dayan & Abbott (2001): short-term synaptic depression models vesicle depletion
  at pre-synaptic terminals — each activation consumes a fraction of available
  resources, which recover exponentially.
- Tsodyks & Markram (1997): STD creates natural latching dynamics where the
  network switches between attractors as the current one fatigues.
- Carandini & Heeger (2012): normalization (divisive gain control) replaces
  threshold-based gating in cortical circuits.
"""

import numpy as np
from typing import Dict, Optional, List
from collections import defaultdict


# ─── Continuous Sigmoid Gates ───

def sigmoid_gate(x: float, midpoint: float = 0.5, slope: float = 8.0) -> float:
    """Continuous sigmoid gating function: smooth 0→1 transition.
    
    Replaces: ``if x < threshold: signal *= 0.05 else: signal *= 1.0``
    With:     ``signal *= sigmoid_gate(x, midpoint, slope)``
    
    At x == midpoint, output = 0.5.
    slope=8 gives ~90% transition within ±0.3 of midpoint.
    """
    return 1.0 / (1.0 + np.exp(-slope * (x - midpoint)))


def inverse_sigmoid_gate(x: float, midpoint: float = 0.5, slope: float = 8.0) -> float:
    """Inverse sigmoid: high when x is low (suppression when x exceeds midpoint).
    
    Replaces: ``if x > threshold: signal *= 0.3``
    """
    return 1.0 - sigmoid_gate(x, midpoint, slope)


def relevance_suppression(similarity: float, midpoint: float = 0.20, slope: float = 10.0) -> float:
    """Continuous relevance suppression: concepts with low similarity get suppressed.
    
    Replaces binary::
        if sim < 0.20: signal *= 0.05
        elif sim < 0.35: signal *= 0.15
    
    smooth_below_midpoint gives ~0.05 at sim=0.0, ~0.5 at sim=midpoint, ~1.0 at sim=1.0.
    """
    return sigmoid_gate(similarity, midpoint, slope)


def relevance_suppression_dual(similarity: float) -> float:
    """Dual-regime relevance suppression with two continuous zones.
    
    At sim=0.0 → ~0.05 (strong suppression)
    At sim=0.20 → ~0.15 (weak gate)
    At sim=0.35 → ~0.50 (neutral)
    At sim=0.50 → ~0.85 (pass through)
    At sim=0.70 → ~0.97 (full pass)
    
    Uses two sigmoids blended: one at 0.20 for near-zero suppression,
    one at 0.35 for the main gate.
    """
    low_gate = sigmoid_gate(similarity, 0.20, 8.0)  # 0.05 → 0.5 at sim=0.20
    high_gate = sigmoid_gate(similarity, 0.35, 10.0)  # 0.5 at sim=0.35 → 1.0 at sim=0.70
    return low_gate * 0.5 + high_gate * 0.5


def degree_adaptive_threshold(degree: float, max_degree: float,
                               base_slope: float = 6.0) -> float:
    """Degree-adaptive threshold: high-degree concepts need more similarity.
    
    Replaces: ``threshold = 0.15 + 0.3 * (degree / max_degree)``
    
    Returns a threshold value in [0.10, 0.50] that increases with degree.
    The relationship is sigmoidal rather than linear.
    """
    norm_degree = degree / max(1.0, max_degree)
    # Sigmoid: at norm_degree=0 → ~0.10, at norm_degree=0.5 → ~0.25, at norm_degree=1 → ~0.45
    return 0.10 + 0.40 * sigmoid_gate(norm_degree, 0.5, base_slope)


def degree_suppression(degree: float, max_degree: float, similarity: float) -> float:
    """Continuous suppression based on degree-adaptive threshold.
    
    Replaces::
        threshold = 0.15 + 0.3 * (degree / max_degree)
        if sim < threshold: signal *= 0.05
    """
    thresh = degree_adaptive_threshold(degree, max_degree)
    # How far below threshold? If sim=0.5, threshold=0.3: gap=0.2 → high suppression
    # If sim=0.5, threshold=0.2: gap=-0.03 → low suppression
    gap = thresh - similarity
    # Sigmoid centered: when gap > 0.05 (significantly below), suppression kicks in
    suppress = sigmoid_gate(gap, 0.05, 15.0)
    return 1.0 - 0.95 * suppress  # ranges from ~1.0 (no suppress) to ~0.05 (full suppress)


def recency_modulation(ctx_similarity: float) -> float:
    """Continuous recency modulation: boost related recent concepts, suppress unrelated.
    
    Replaces::
        if ctx_sim > 0.30: signal *= 1.5
        else: signal *= 0.3
    """
    boost = sigmoid_gate(ctx_similarity, 0.30, 10.0)  # ~0.5 at 0.30
    # Map: at boost=0 (sim=0) → 0.3, at boost=1 (sim=1) → 1.5
    return 0.3 + 1.2 * boost


def task_relevance_gate(similarity: float) -> float:
    """Continuous LIFG task relevance gate.
    
    Replaces: ``if task_relevance < 0.10: continue``
    
    Returns a multiplier in [0.01, 1.0] — near zero for completely irrelevant,
    near 1.0 for relevant. No hard cutoff.
    """
    return sigmoid_gate(similarity, 0.10, 12.0)


def post_hoc_relevance_filter(similarity: float) -> float:
    """Continuous post-hoc relevance gate.
    
    Replaces: ``if final_sim < 0.15: continue``
    """
    return sigmoid_gate(similarity, 0.15, 10.0)


def self_penalty_gate(semantic_similarity: float) -> float:
    """Continuous self-penalty for semantically distant concepts.
    
    Replaces::
        if semantic_sim < 0.3 and score > 0.1: score *= 0.5
    
    At sim=0.0 (very distant) → returns ~0.5 (max penalty, like original)
    At sim=0.3 (threshold) → returns ~0.75 (moderate penalty)
    At sim=0.6 (close) → returns ~0.95 (minimal penalty)
    """
    penalty = sigmoid_gate(semantic_similarity, 0.30, 8.0)
    return 0.5 + 0.5 * penalty  # ranges from 0.5 (distant) to 1.0 (close)


def rlm_confidence_modulation(confidence: float) -> float:
    """Continuous RLM confidence modulation.
    
    Replaces::
        if conf < 0.4: signal *= 0.7
        elif conf > 0.75: signal *= 1.15
    """
    base = 1.0
    # Smooth penalty for low confidence
    low_penalty = inverse_sigmoid_gate(confidence, 0.40, 8.0)  # 0→1 when conf<0.4
    base -= 0.3 * low_penalty  # up to -30%
    # Smooth boost for high confidence
    high_boost = sigmoid_gate(confidence, 0.75, 10.0)  # 0→1 when conf>0.75
    base += 0.15 * high_boost  # up to +15%
    return base


def valence_modulation(valence: float, edge_weight: float, max_weight: float) -> float:
    """Continuous VAD valence modulation.
    
    Replaces::
        if valence > 0.10: signal *= 1.0 + 0.15 * (weight / max_weight)
        elif valence < -0.10: signal *= 0.9
    """
    pos_gate = sigmoid_gate(valence, 0.10, 8.0)  # 1 when valence > 0.10
    neg_gate = sigmoid_gate(-valence, 0.10, 8.0)  # 1 when valence < -0.10
    
    boost = 1.0
    boost += 0.15 * (edge_weight / max(0.001, max_weight)) * pos_gate
    boost -= 0.1 * neg_gate
    return boost


def dominance_modulation(dominance: float, edge_weight: float, max_weight: float) -> float:
    """Continuous VAD dominance modulation.
    
    Replaces::
        if dominance < 0.35: signal *= 0.6 + 0.4 * (weight / max_weight)
    """
    low_dom = inverse_sigmoid_gate(dominance, 0.35, 8.0)  # 1 when dom < 0.35
    # Continuous: at weight=0 → 0.6, at weight=max → 1.0
    factor = 0.6 + 0.4 * (edge_weight / max(0.001, max_weight))
    return 1.0 + (factor - 1.0) * low_dom


def edge_strength_suppression(edge_weight: float) -> float:
    """Continuous edge weight suppression instead of binary <0.35 skip.
    
    Replaces: ``if edge.weight < 0.35: continue``
    
    Returns multiplier in [0.01, 1.0] — weak edges still get a small chance.
    """
    return sigmoid_gate(edge_weight, 0.35, 10.0)


def repetition_penalty(distance: float) -> float:
    """Continuous repetition penalty based on recency distance.
    
    Replaces::
        if dist < 10: signal *= 0.3
        elif dist < 20: signal *= 0.6
        else: signal *= 0.8
    """
    # Sigmoid centered around 10 and 20 with smooth transition
    # At dist=0 → 0.3, at dist=10 → 0.5, at dist=20 → 0.8, at dist=30 → 0.95
    recovery = sigmoid_gate(distance, 12.0, 0.25)  # smooth recovery
    return 0.3 + 0.7 * recovery


def dormant_edge_modulation(confidence: float, reactivation_count: int) -> float:
    """Continuous dormant edge awakening.
    
    Replaces::
        if vc > 0 or confidence > 0.15:
            confidence = 0.3
            signal = weight * 0.3
    """
    awakening = sigmoid_gate(confidence, 0.15, 10.0) + sigmoid_gate(reactivation_count, 0.5, 4.0)
    awakening = min(1.0, awakening)
    new_confidence = 0.3 * awakening
    new_signal_factor = 0.3 * awakening
    return new_confidence, new_signal_factor


def causal_bias_activation(signal: float, activation: float) -> float:
    """Continuous activation minimum instead of ``if activation <= 0.01: continue``."""
    return signal * sigmoid_gate(activation, 0.01, 50.0)


# ─── Synaptic Depression (Short-Term Plasticity) ───

class SynapticDepression:
    """Short-term synaptic depression for natural latching dynamics.
    
    Each concept node has a depression trace that increases with each activation
    and decays exponentially during rest. When a concept is highly depressed,
    its effective activation is reduced, forcing the system to latch onto
    alternative concepts — exactly like neural adaptation in the brain.
    
    Parameters:
        depression_per_activation: how much each activation depletes resources (0.0-1.0)
        recovery_rate: exponential recovery per timestep (0.0-1.0)
        min_efficiency: minimum synaptic efficiency when fully depressed (0.0-1.0)
    
    Neuroscience basis:
        Tsodyks & Markram (1997): STD creates competition between synapses —
        the most recently used pathway is temporarily weakened, allowing other
        pathways to dominate.
    """
    
    def __init__(self, depression_per_activation: float = 0.20,
                 recovery_rate: float = 0.10,
                 min_efficiency: float = 0.10):
        self.depression_per_activation = depression_per_activation
        self.recovery_rate = recovery_rate
        self.min_efficiency = min_efficiency
        # Per-node depression traces: node_id -> depression (0.0 = none, 1.0 = fully depressed)
        self._traces: Dict[int, float] = defaultdict(float)
        # Per-edge depression for fine-grained control
        self._edge_traces: Dict[tuple, float] = defaultdict(float)
    
    def activate(self, node_id: int) -> float:
        """Record an activation event and return the effective efficiency.
        
        Returns a multiplier in [min_efficiency, 1.0] for this node's output.
        """
        current = self._traces.get(node_id, 0.0)
        current = min(1.0, current + self.depression_per_activation)
        self._traces[node_id] = current
        return self.get_efficiency(current)
    
    def decay_all(self):
        """Decay all depression traces toward zero (rest / sleep)."""
        for nid in list(self._traces.keys()):
            self._traces[nid] *= (1.0 - self.recovery_rate)
            if self._traces[nid] < 0.01:
                del self._traces[nid]
        for pair in list(self._edge_traces.keys()):
            self._edge_traces[pair] *= (1.0 - self.recovery_rate * 0.5)
            if self._edge_traces[pair] < 0.01:
                del self._edge_traces[pair]
    
    def get_efficiency(self, depression: float) -> float:
        """Convert depression to synaptic efficiency (continuous).
        
        depression=0.0 → efficiency=1.0 (fresh)
        depression=0.5 → efficiency=0.45
        depression=1.0 → efficiency=min_efficiency (depleted)
        
        Uses a non-linear mapping: the first activation hurts more than subsequent ones.
        """
        return self.min_efficiency + (1.0 - self.min_efficiency) * (1.0 - depression) ** 2
    
    def get_node_efficiency(self, node_id: int) -> float:
        """Get the current efficiency for a node without activating it."""
        return self.get_efficiency(self._traces.get(node_id, 0.0))
    
    def get_depression(self, node_id: int) -> float:
        """Get the raw depression value for a node."""
        return self._traces.get(node_id, 0.0)
    
    def activate_edge(self, src_id: int, tgt_id: int) -> float:
        """Depress a specific edge (directed path) and return its efficiency."""
        pair = (src_id, tgt_id)
        current = self._edge_traces.get(pair, 0.0)
        current = min(1.0, current + self.depression_per_activation * 0.5)
        self._edge_traces[pair] = current
        return self.get_efficiency(current)
    
    def get_edge_efficiency(self, src_id: int, tgt_id: int) -> float:
        """Get edge efficiency without activating it."""
        return self.get_efficiency(self._edge_traces.get((src_id, tgt_id), 0.0))
    
    def reset(self):
        """Clear all depression traces (full recovery)."""
        self._traces.clear()
        self._edge_traces.clear()
    
    @property
    def most_depressed(self, top_n: int = 5) -> List:
        """Return the most depressed nodes for diagnostics."""
        sorted_nodes = sorted(self._traces.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]


# ─── Convenience: Combined Signal Modulator ───

class SignalModulator:
    """Aggregates all continuous modulation functions into a clean API.
    
    Creates one instance per engine to replace all threshold-based logic.
    """
    
    def __init__(self):
        self.depression = SynapticDepression()
    
    def modulate_spread_signal(self, signal: float, similarity: float,
                                 degree: float, max_degree: float,
                                 is_causal: bool = False) -> float:
        """Apply all continuous relevance gates to a spreading activation signal."""
        if is_causal:
            # Causal edges skip relevance gates (causality links distant concepts)
            return signal
        
        # Topic relevance gate
        rel_gate = relevance_suppression_dual(similarity)
        signal *= rel_gate
        
        # Degree-adaptive suppression (high-degree concepts need more similarity)
        deg_gate = degree_suppression(degree, max_degree, similarity)
        signal *= deg_gate
        
        return signal
    
    def modulate_recency(self, signal: float, ctx_similarity: float) -> float:
        """Apply continuous recency modulation."""
        return signal * recency_modulation(ctx_similarity)
    
    def modulate_task_relevance(self, signal: float, similarity: float) -> float:
        """Apply continuous LIFG task relevance gate."""
        return signal * task_relevance_gate(similarity)
    
    def apply_depression_to_signal(self, signal: float, node_id: int) -> float:
        """Apply synaptic depression to a signal for a given node."""
        eff = self.depression.get_node_efficiency(node_id)
        return signal * eff


# --- Phase 4: Connector Learner ---
class ConnectorLearner:
    """Learns connector word probabilities from GloVe vector similarity.

    Replaces all hardcoded _EDGE_CONNECTORS, _EDGE_TO_GRAPH_LABEL,
    _EDGE_TO_STARTER maps with vector-similarity-based probabilities.
    """

    PROTOTYPE_CONNECTORS = {
        "causal": ["because", "since", "therefore", "so", "thus", "hence",
                   "consequently", "accordingly", "as", "cause", "leads"],
        "contrastive": ["but", "however", "yet", "although", "though",
                        "nevertheless", "instead", "unlike", "whereas",
                        "contrast", "opposite"],
        "semantic": ["and", "also", "connect", "relate", "refer", "means",
                    "associate", "link", "like", "including"],
        "temporal": ["then", "after", "before", "while", "during",
                    "when", "until", "since", "subsequently", "next"],
        "analogical": ["like", "similar", "analogous", "resemble",
                      "parallel", "metaphor", "compare", "likewise",
                      "correspond", "as"],
        "episodic": ["remember", "recall", "when", "then", "after",
                    "during", "while", "before", "later", "earlier"],
    }

    FALLBACK_CONNECTORS = {
        "causal": "because", "contrastive": "but", "semantic": "and",
        "temporal": "then", "analogical": "like", "episodic": "when",
    }

    GRAPH_LABELS = {
        "causal": "cause", "contrastive": "but", "semantic": "connect",
        "temporal": "change", "analogical": "like", "episodic": "connect",
    }

    STARTER_CONNECTORS = {
        "causal": "because", "contrastive": "but", "semantic": "and",
        "temporal": "then", "analogical": "like", "episodic": "and",
    }

    def __init__(self, glove_fn=None):
        self._glove_fn = glove_fn
        self._prototype_vecs = {}
        self._learned_probs = {}
        self._connector_set = set()
        self._connector_to_rel = {}
        self._is_initialized = False

    def set_glove_fn(self, glove_fn):
        self._glove_fn = glove_fn

    def initialize(self, graph_concepts=None):
        """Precompute prototype vectors and discover connectors from graph."""
        import numpy as np
        if self._glove_fn is None:
            self._is_initialized = True
            return

        for rel_type, seeds in self.PROTOTYPE_CONNECTORS.items():
            vecs = [v for w in seeds if (v := self._glove_fn(w)) is not None]
            if vecs:
                proto = np.mean(vecs, axis=0)
                norm = np.linalg.norm(proto)
                if norm > 0:
                    proto = proto / norm
                self._prototype_vecs[rel_type] = proto

        if graph_concepts:
            for word, vec in graph_concepts:
                if word in self._connector_set or vec is None:
                    continue
                vn = vec / (np.linalg.norm(vec) + 1e-15)
                for rt, pv in self._prototype_vecs.items():
                    sim = float(np.dot(vn, pv))
                    if sim > 0.55:
                        self._connector_set.add(word)
                        self._connector_to_rel[word] = rt
                        self._learned_probs.setdefault(rt, []).append((word, sim))
                        break

        for rt, seeds in self.PROTOTYPE_CONNECTORS.items():
            for w in seeds:
                self._connector_to_rel[w] = rt
                self._connector_set.add(w)

        self._is_initialized = True

    def get_connector(self, rel_type, weight=0.5, confidence=0.5, temperature=0.25, rng=None):
        """Get the best connector word for a relation type."""
        import numpy as np
        if rng is None:
            rng = np.random.RandomState(42)
        if self._learned_probs and rel_type in self._learned_probs:
            candidates = self._learned_probs[rel_type]
            if len(candidates) > 1:
                scores = [s for _, s in candidates]
                words = [w for w, _ in candidates]
                temp = max(0.05, temperature * (1.0 - weight * confidence))
                exp_scores = [np.exp(s / temp) for s in scores]
                total = sum(exp_scores)
                return rng.choice(words, p=[e / total for e in exp_scores])
            elif len(candidates) == 1:
                return candidates[0][0]
        if rel_type in self.PROTOTYPE_CONNECTORS:
            seeds = self.PROTOTYPE_CONNECTORS[rel_type]
            if len(seeds) > 1 and self._glove_fn is not None:
                temp = max(0.05, temperature * (1.0 - weight * confidence) + 0.1)
                scores = []
                for w in seeds:
                    v = self._glove_fn(w)
                    scores.append(float(np.linalg.norm(v)) if v is not None else 1.0)
                if any(s > 0 for s in scores):
                    exp_scores = [np.exp(s / temp) for s in scores]
                    return rng.choice(seeds, p=[e / sum(exp_scores) for e in exp_scores])
            return seeds[0]
        return self.FALLBACK_CONNECTORS.get(rel_type, "and")

    def get_graph_label(self, rel_type):
        return self.GRAPH_LABELS.get(rel_type, "connect")

    def get_starter(self, rel_type):
        return self.STARTER_CONNECTORS.get(rel_type, "and")

    def get_connector_set(self):
        return self._connector_set

    def get_connector_to_rel(self):
        return self._connector_to_rel

    def get_relation_for_connector(self, word):
        """Get the relation type for a connector word."""
        if word in self._connector_to_rel:
            return self._connector_to_rel[word]
        import numpy as np
        if self._glove_fn is not None and self._prototype_vecs:
            v = self._glove_fn(word)
            if v is not None:
                vn = v / (np.linalg.norm(v) + 1e-15)
                best_rel = "semantic"
                best_sim = 0.0
                for rt, pv in self._prototype_vecs.items():
                    sim = float(np.dot(vn, pv))
                    if sim > best_sim:
                        best_sim = sim
                        best_rel = rt
                return best_rel
        return "semantic"
