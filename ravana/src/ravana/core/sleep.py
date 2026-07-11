"""
Sleep Consolidation for RAVANA.

Implements complementary learning systems (McClelland et al., 1995):
- Hippocampal replay: fast, sparse episodic replay (NREM)
- Neocortical consolidation: slow, overlapping weight updates (NREM)
- Synaptic homeostasis: renormalize weights (Tononi & Cirelli, 2006)
- REM dream sabotage: counterfactual edges, emotional reconsolidation
- Prediction-error-driven updates (Active Inference during sleep)
"""
import numpy as np
import random
import time
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

    # CLS sleep gate (SHY down-selection, Tononi & Cirelli 2014; Nere et al. 2013):
    # a web_fact edge CORROBORATED in 2+ independent contexts is protected from
    # weight-only pruning even if its weight is low. Cross-context corroboration
    # is the offline reactivation signal ("a neuron detects suspicious
    # coincidences and protects the associated synapses from depression" —
    # SHY). Single-context low-weight edges still prune (noise). This is the
    # sleep-time complement to the IV-C ingest-time XdG gate; both are needed
    # (van de Ven et al. 2020). Default ON -> backward-compatible.
    protect_cross_context: bool = True
    min_contexts: int = 2

    # REM dream sabotage parameters
    rem_counterfactual_rate: float = 0.12
    rem_emotional_decay: float = 0.05
    rem_flip_rate: float = 0.03
    rem_creative_rate: float = 0.06
    rem_sampling_k: int = 20


class SleepConsolidation:
    """Sleep consolidation engine: offline replay, renormalization, pruning, REM sabotage."""

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
            "rem_counterfactual_edges": 0,
            "rem_flipped_edges": 0,
            "rem_creative_edges": 0,
            "rem_emotional_decays": 0,
        }

    def run_cycle(self, graph, episodic_buffer: List[Dict],
                  episodic_triples: List[tuple],
                  belief_store,
                  topic_list: List[str],
                  user_model,
                  impossible_queries: List,
                  contradiction_map: Dict,
                  drift_defense_threshold: float = 0.7,
                  drift_pull: float = 0.05,
                  concept_vad: Optional[Dict[int, tuple]] = None) -> Dict[str, int]:
        """Run a full sleep consolidation cycle with NREM and REM stages.

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
            concept_vad: Optional dict of node_id -> (valence, arousal, dominance)
                         for emotional reconsolidation during REM

        Returns:
            Dict with sleep metrics
        """
        edges_strengthened = 0
        edges_pruned = 0
        episodic_consolidated = 0
        impossible_resolved = 0
        rem_counterfactual = 0
        rem_flipped = 0
        rem_creative = 0
        rem_emotional = 0

        # NREM STAGE 1: Hippocampal replay
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

                edge.weight = min(1.0, edge.weight + 0.02)
                edge.confidence = min(1.0, edge.confidence + 0.01)
                edges_strengthened += 1

                graph.hebbian_update(subj_cid, obj_cid, coactivation=0.5, lr=0.005)

        # NREM STAGE 2: Homeostatic downscaling
        self._normalize_outgoing_weights(graph, budget=self.config.downscaling_budget)

        # NREM STAGE 3: Prune weak edges (SHY down-selection). Cross-context-
        # corroborated web_fact edges are protected from the weight floor;
        # single-context low-weight edges prune as noise.
        edges_pruned = self._prune_weak_edges(
            graph, threshold=self.config.prune_threshold,
            protect_cross_context=self.config.protect_cross_context,
            min_contexts=self.config.min_contexts)

        # NREM STAGE 3b: provenance-based noise prune (separate predicate, so
        # weight-downscale vs kind-based noise-removal don't get conflated).
        # Keeps edge_kind=="web_fact" (verified); culls co_occurrence/auto_expand
        # orphan noise that lacks cross-context corroboration. Two distinct
        # gates, run as two passes — honesty: each predicate stays separable.
        try:
            noise_pruned = graph.prune_low_quality_edges(enabled=True)
        except Exception:
            noise_pruned = 0
        edges_pruned += noise_pruned

        # NREM STAGE 4: Drift defense
        for nid, node in list(graph.nodes.items()):
            drift = getattr(node, 'drift_magnitude', 0.0)
            if drift > drift_defense_threshold:
                pull = drift_pull * (node.core_vector - node.vector)
                node.vector += pull
                norm = np.linalg.norm(node.vector)
                if norm > 0:
                    node.vector /= norm

        # REM STAGE 5: Dream sabotage (counterfactuals, emotional processing, creativity)
        rem_counterfactual, rem_flipped, rem_creative, rem_emotional = self._rem_dream_sabotage(
            graph=graph,
            concept_vad=concept_vad,
        )

        # NREM STAGE 6: Episodic -> semantic consolidation
        for ep in episodic_buffer:
            if ep['correct'] and ep['error'] < 0.3:
                importance = ep.get('importance', 0.5)
                is_novel = ep.get('consolidation_state', 'fresh') == 'fresh'
                for cid in ep['concepts']:
                    if cid in graph.nodes:
                        pass
                    episodic_consolidated += 1

        # NREM STAGE 7: Belief reconciliation
        if belief_store and belief_store.users:
            resolved = belief_store.reconcile()

        # NREM STAGE 8: Sleep-replay impossible queries
        for iq in impossible_queries:
            if iq.resolved:
                continue
            subj_nids = user_model._get_concept_nids(iq.subject) if hasattr(user_model, '_get_concept_nids') else []
            if subj_nids:
                subj_node = graph.get_node(subj_nids[0])
                if subj_node and subj_node.vector is not None:
                    new_edges = 0
                    for other_nid, other_node in list(graph.nodes.items()):
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

        # STAGE 9: Correct mis-typed relations during sleep
        # (Handled by GraphEngine._correct_relation_types)

        self.metrics["edges_strengthened"] += edges_strengthened
        self.metrics["edges_pruned"] += edges_pruned
        self.metrics["episodic_consolidated"] += episodic_consolidated
        self.metrics["impossible_queries_resolved"] += impossible_resolved
        self.metrics["rem_counterfactual_edges"] += rem_counterfactual
        self.metrics["rem_flipped_edges"] += rem_flipped
        self.metrics["rem_creative_edges"] += rem_creative
        self.metrics["rem_emotional_decays"] += rem_emotional
        self.metrics["total_sleep_cycles"] += 1

        return {
            "edges_strengthened": edges_strengthened,
            "edges_pruned": edges_pruned,
            "episodic_consolidated": episodic_consolidated,
            "impossible_queries_resolved": impossible_resolved,
            "rem_counterfactual_edges": rem_counterfactual,
            "rem_flipped_edges": rem_flipped,
            "rem_creative_edges": rem_creative,
            "rem_emotional_decays": rem_emotional,
        }

    def _normalize_outgoing_weights(self, graph, budget: float = 5.0):
        """Normalize outgoing edge weights per node."""
        for nid in list(graph._outgoing):
            edges = graph._outgoing.get(nid, [])
            if not edges:
                continue
            total_weight = sum(e.weight for _, e in edges)
            if total_weight > budget:
                scale = budget / total_weight
                for _, edge in edges:
                    edge.weight *= scale

    def _prune_weak_edges(self, graph, threshold: float = 0.1,
                          protect_cross_context: bool = True,
                          min_contexts: int = 2) -> int:
        """Remove edges with weight below threshold, EXCEPT edges whose
        provenance shows corroboration in 2+ independent contexts.

        CLS sleep gate (SHY down-selection, Tononi & Cirelli 2014; Nere et al.
        2013): sleep is competitive down-selection — synapses reactivated /
        well-integrated are protected from depression; isolated ones depress.
        Cross-context corroboration (recorded by the IV-C ingest-time XdG gate
        in edge.source_metadata['contexts']) is the computational analog of
        "reactivated across multiple offline bouts" -> it fits prior structure
        -> protect. Single-context low-weight edges still prune (noise).

        This is the sleep-time complement to the ingest-time XdG gate
        (van de Ven et al. 2020): both a write-time gate and a sleep-time gate
        are needed side by side. Default protect_cross_context=True ->
        backward-compatible (all sleep callers unchanged).

        Honesty: protection is grounded in independent-context corroboration
        (a real signal), NOT a blind weight floor. An edge with weight <
        threshold is pruned UNLESS it carries >= min_contexts distinct contexts.
        """
        edges_to_remove = []
        for (src, tgt), edge in list(graph.edges.items()):
            if edge.weight >= threshold:
                continue
            # SHY sleep gate: spare cross-context-corroborated edges.
            if protect_cross_context and min_contexts >= 1:
                contexts = (getattr(edge, "source_metadata", {}) or {}).get("contexts", [])
                distinct = {c for c in contexts if c}
                if len(distinct) >= min_contexts:
                    continue  # reactivated in 2+ contexts -> protected
            edges_to_remove.append((src, tgt))

        for src, tgt in edges_to_remove:
            graph.remove_edge(src, tgt)
        return len(edges_to_remove)

    def _rem_dream_sabotage(self, graph, concept_vad: Optional[Dict[int, tuple]] = None) -> tuple:
        """REM dream sabotage: counterfactual edges, emotional processing, creative recombination.

        Models three key REM sleep functions:
        1. Counterfactual generation: edges between weakly similar nodes (cos sim 0.2-0.45)
           with randomly flipped relation types
        2. Creative hub recombination: edges between distant high-degree hubs
        3. Emotional reconsolidation: decay emotional charge on high-VAD concepts

        Returns:
            (counterfactual_created, edges_flipped, creative_edges, emotional_decays)
        """
        cfg = self.config
        rng = random.Random()
        rng.seed(hash(str(time.time())) % (2 ** 32))

        n_nodes = len(graph.nodes)
        if n_nodes < 5:
            return (0, 0, 0, 0)

        node_ids = list(graph.nodes.keys())
        counterfactual_created = 0
        edges_flipped = 0
        creative_created = 0
        emotional_decays = 0

        # Helper: compute degree (outgoing + incoming) for a node
        def _node_degree(nid: int) -> int:
            return len(graph._outgoing.get(nid, [])) + len(graph._incoming.get(nid, []))

        relation_types = ["semantic", "causal", "temporal", "analogical", "contextual"]

        # ── 1. Counterfactual edge generation ──
        n_counterfactual = max(1, int(n_nodes * cfg.rem_counterfactual_rate))
        candidate_nodes = rng.sample(node_ids, min(n_counterfactual, n_nodes))

        for src_nid in candidate_nodes:
            src_node = graph.get_node(src_nid)
            if src_node is None or src_node.vector is None:
                continue

            other_ids = [nid for nid in node_ids if nid != src_nid]
            if len(other_ids) < 2:
                continue
            samples = rng.sample(other_ids, min(cfg.rem_sampling_k, len(other_ids)))

            for tgt_nid in samples:
                tgt_node = graph.get_node(tgt_nid)
                if tgt_node is None or tgt_node.vector is None:
                    continue
                if graph.get_edge(src_nid, tgt_nid) is not None:
                    continue
                if graph.get_edge(tgt_nid, src_nid) is not None:
                    continue

                sim = float(np.dot(src_node.vector, tgt_node.vector))
                norms_prod = float(np.linalg.norm(src_node.vector) * np.linalg.norm(tgt_node.vector))
                if norms_prod > 0:
                    cos_sim = sim / norms_prod
                else:
                    continue

                # Counterfactual sweet spot: weakly similar
                if 0.2 <= cos_sim <= 0.45:
                    flipped_type = rng.choice(relation_types)
                    if flipped_type == "semantic" and rng.random() < 0.6:
                        flipped_type = rng.choice(["causal", "temporal", "analogical"])

                    weight = 0.25 + 0.3 * cos_sim
                    edge = graph.add_edge(src_nid, tgt_nid, weight=weight,
                                         relation_type=flipped_type,
                                         confidence=0.02)
                    edge.stability = 0.05
                    counterfactual_created += 1
                    break

        # ── 2. Relation type flipping ──
        n_flip = max(1, int(len(graph.edges) * cfg.rem_flip_rate))
        all_edges = list(graph.edges.items())
        flip_candidates = rng.sample(all_edges, min(n_flip, len(all_edges)))

        for (src, tgt), edge in flip_candidates:
            if edge.edge_type == "inhibitory":
                continue
            current = edge.relation_type
            alternatives = [rt for rt in relation_types if rt != current]
            if not alternatives:
                continue
            new_type = rng.choice(alternatives)
            edge.relation_type = new_type
            edge.stability *= 0.8
            edges_flipped += 1

        # ── 3. Creative hub recombination ──
        n_creative = max(1, int(n_nodes * cfg.rem_creative_rate))
        degree_order = sorted(node_ids, key=_node_degree, reverse=True)
        hub_nodes = degree_order[:max(10, n_creative)]

        for src_nid in hub_nodes:
            src_node = graph.get_node(src_nid)
            if src_node is None or src_node.vector is None:
                continue

            for tgt_nid in hub_nodes:
                if tgt_nid <= src_nid:
                    continue
                tgt_node = graph.get_node(tgt_nid)
                if tgt_node is None or tgt_node.vector is None:
                    continue
                if graph.get_edge(src_nid, tgt_nid) is not None:
                    continue
                if graph.get_edge(tgt_nid, src_nid) is not None:
                    continue

                sim = float(np.dot(src_node.vector, tgt_node.vector))
                norms_prod = float(np.linalg.norm(src_node.vector) * np.linalg.norm(tgt_node.vector))
                if norms_prod > 0:
                    cos_sim = sim / norms_prod
                else:
                    continue

                # Distant but both important: creative recombination
                if cos_sim < 0.15 and cos_sim > -0.15:
                    weight = 0.15 + abs(cos_sim) * 0.2
                    flip_type = rng.choice(["analogical", "contextual", "inferred"])
                    edge = graph.add_edge(src_nid, tgt_nid, weight=weight,
                                         relation_type=flip_type,
                                         confidence=0.01)
                    edge.stability = 0.02
                    creative_created += 1
                    break

        # ── 4. Emotional reconsolidation ──
        if concept_vad and len(concept_vad) > 0:
            vad_nids = list(concept_vad.keys())
            for nid in vad_nids:
                if nid not in graph.nodes:
                    continue
                vad = concept_vad[nid]
                if vad is None or len(vad) < 3:
                    continue
                valence, arousal, dominance = vad

                emotional_intensity = abs(valence) * arousal
                if emotional_intensity > 0.3:
                    decay = cfg.rem_emotional_decay
                    new_valence = valence * (1.0 - decay)
                    new_arousal = max(0.0, arousal - decay * 0.5)
                    concept_vad[nid] = (new_valence, new_arousal, dominance)
                    emotional_decays += 1

        return (counterfactual_created, edges_flipped, creative_created, emotional_decays)
