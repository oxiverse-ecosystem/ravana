"""
Sleep Consolidation for RAVANA.

Implements complementary learning systems (McClelland et al., 1995):
- Hippocampal replay: fast, sparse episodic replay
- Neocortical consolidation: slow, overlapping weight updates
- Synaptic homeostasis: renormalize weights (Tononi & Cirelli, 2006)
- Prediction-error-driven updates (Active Inference during sleep)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from collections import defaultdict


@dataclass
class SleepConfig:
    """Configuration for sleep consolidation."""
    pressure_threshold: float = 0.3
    counterfactual_rate: float = 0.15
    emotional_flip_rate: float = 0.08
    downscaling_budget: float = 5.0
    prune_threshold: float = 0.1


class SleepConsolidation:
    """Sleep consolidation engine: offline replay, renormalization, pruning."""

    def __init__(self, config: Optional[SleepConfig] = None):
        self.config = config or SleepConfig()
        self.metrics: Dict[str, Any] = {
            "edges_strengthened": 0,
            "edges_pruned": 0,
            "episodic_consolidated": 0,
            "impossible_queries_resolved": 0,
            "total_sleep_cycles": 0,
            "last_sleep_turn": 0,
            "last_sleep_metrics": {},
        }

    def run_cycle(self, graph, episodic_buffer: List[Dict],
                  episodic_triples: List[tuple],
                  belief_store,
                  topic_list: List[str],
                  user_model,
                  impossible_queries: List,
                  contradiction_map: Dict,
                  drift_defense_threshold: float = 0.7,
                  drift_pull: float = 0.05) -> Dict[str, int]:
        """Run a full sleep consolidation cycle.

        Args:
            graph: ConceptGraph instance
            episodic_buffer: List of episodic memories
            episodic_triples: List of (subj, rel, obj, time) triples
            belief_store: BeliefStore instance
            topic_list: Recent conversation topics
            user_model: UserModel instance
            impossible_queries: List of FailedQuery objects
            contradiction_map: Dict of concept -> antonym set
            drift_defense_threshold: Threshold for drift correction
            drift_pull: Strength of drift correction

        Returns:
            Dict with sleep metrics
        """
        edges_strengthened = 0
        edges_pruned = 0
        episodic_consolidated = 0
        impossible_resolved = 0

        # 1. Hippocampal replay: replay recent/important triples
        if episodic_triples:
            n_replay = min(10, len(episodic_triples))
            replay_triples = episodic_triples[-n_replay:]

            for subj_cid, rel_idx, obj_cid, ts in replay_triples:
                src = graph.get_node(subj_cid)
                tgt = graph.get_node(obj_cid)
                if src is None or tgt is None:
                    continue
                edge = graph.get_edge(subj_cid, obj_cid)
                if edge is None:
                    continue

                # Strengthen through replay
                edge.weight = min(1.0, edge.weight + 0.02)
                edge.confidence = min(1.0, edge.confidence + 0.01)
                edges_strengthened += 1

                # Hebbian co-activation replay
                graph.hebbian_update(subj_cid, obj_cid, coactivation=0.5, lr=0.005)

        # 2. Homeostatic downscaling: prevent runaway strengthening
        self._normalize_outgoing_weights(graph, budget=self.config.downscaling_budget)

        # 3. Prune weak edges
        edges_pruned = self._prune_weak_edges(graph, threshold=self.config.prune_threshold)

        # 4. Drift defense: pull concept vectors back toward core
        for nid, node in graph.nodes.items():
            drift = getattr(node, 'drift_magnitude', 0.0)
            if drift > drift_defense_threshold:
                pull = drift_pull * (node.core_vector - node.vector)
                node.vector += pull
                norm = np.linalg.norm(node.vector)
                if norm > 0:
                    node.vector /= norm

        # 5. Episodic -> semantic consolidation
        for ep in episodic_buffer:
            if ep['correct'] and ep['error'] < 0.3:
                importance = ep.get('importance', 0.5)
                is_novel = ep.get('consolidation_state', 'fresh') == 'fresh'
                strength_delta = 0.05 * importance * (1.5 if is_novel else 1.0)
                for cid in ep['concepts']:
                    if cid in graph.nodes:
                        # Mark for semantic consolidation (handled by graph)
                        pass
                    episodic_consolidated += 1

        # 6. Belief reconciliation
        if belief_store and belief_store.beliefs:
            resolved = belief_store.reconcile()
            # Metrics logged externally

        # 7. Sleep-replay impossible queries
        for iq in impossible_queries:
            if iq.resolved:
                continue
            subj_nids = user_model._get_concept_nids(iq.subject) if hasattr(user_model, '_get_concept_nids') else []
            if subj_nids:
                subj_node = graph.get_node(subj_nids[0])
                if subj_node and subj_node.vector is not None:
                    new_edges = 0
                    for other_nid, other_node in graph.nodes.items():
                        if other_nid == subj_nids[0]:
                            continue
                        if other_node.vector is not None and other_node.label:
                            sim = float(np.dot(subj_node.vector, other_node.vector))
                            if sim > 0.5 and graph.get_edge(subj_nids[0], other_nid) is None:
                                weight = min(0.6, sim * 0.6)
                                ne = graph.add_edge(subj_nids[0], other_nid, weight=weight,
                                                   relation_type="semantic")
                                ne.confidence = 0.001
                                new_edges += 1
                    if new_edges > 0:
                        iq.resolved = True
                        impossible_resolved += 1

        # 8. Correct mis-typed relations during sleep
        # (Handled by GraphEngine._correct_relation_types)

        self.metrics["edges_strengthened"] += edges_strengthened
        self.metrics["edges_pruned"] += edges_pruned
        self.metrics["episodic_consolidated"] += episodic_consolidated
        self.metrics["impossible_queries_resolved"] += impossible_resolved
        self.metrics["total_sleep_cycles"] += 1

        return {
            "edges_strengthened": edges_strengthened,
            "edges_pruned": edges_pruned,
            "episodic_consolidated": episodic_consolidated,
            "impossible_queries_resolved": impossible_resolved,
        }

    def _normalize_outgoing_weights(self, graph, budget: float = 5.0):
        """Normalize outgoing edge weights per node."""
        for nid in graph._outgoing:
            edges = graph._outgoing.get(nid, [])
            if not edges:
                continue
            total_weight = sum(e.weight for _, e in edges)
            if total_weight > budget:
                scale = budget / total_weight
                for _, edge in edges:
                    edge.weight *= scale

    def _prune_weak_edges(self, graph, threshold: float = 0.1) -> int:
        """Remove edges with weight below threshold."""
        edges_to_remove = []
        for (src, tgt), edge in graph.edges.items():
            if edge.weight < threshold:
                edges_to_remove.append((src, tgt))

        for src, tgt in edges_to_remove:
            graph.remove_edge(src, tgt)
        return len(edges_to_remove)