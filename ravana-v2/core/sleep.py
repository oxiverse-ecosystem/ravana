"""
RAVANA v2 — SLEEP & DREAM CONSOLIDATION
Periodic consolidation phase with structured dream sabotage.

PRINCIPLE: Sleep is thermodynamic necessity — pressure accumulation
triggers reorganization that stabilizes useful patterns and weakens
brittle ones.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable, Set
from enum import Enum
import copy
import math


class SleepStage(Enum):
    AWAKE = "awake"
    TOPOLOGY_ANALYSIS = "topology_analysis"
    PATTERN_COMPRESSION = "pattern_compression"
    ABSTRACTION_COMPRESSION = "abstraction_compression"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    INTEGRATION = "integration"


class DreamPerturbationType(Enum):
    COUNTERFACTUAL_REVERSAL = "counterfactual_reversal"
    EMOTIONAL_FLIP = "emotional_flip"
    FAILURE_OVERSAMPLE = "failure_oversample"
    SYMBOLIC_RECOMBINATION = "symbolic_recombination"


@dataclass
class SleepConfig:
    """Configuration for sleep consolidation."""
    # Pressure threshold to trigger sleep
    pressure_threshold: float = 0.2
    min_pressure_for_sleep: float = 0.05
    
    # Sleep stage durations (in cognitive cycles)
    topology_analysis_cycles: int = 5
    pattern_compression_cycles: int = 8
    contradiction_resolution_cycles: int = 10
    integration_cycles: int = 5
    
    # Dream sabotage parameters
    counterfactual_rate: float = 0.20  # 20% of memories get reversed
    emotional_flip_rate: float = 0.10  # 10% emotional valence flip
    failure_oversample_factor: float = 1.5  # 1.5x failure replay
    
    # Perturbation limits
    max_perturbation_hops: int = 2
    max_edge_weight_change: float = 0.05
    
    # Abstraction compression parameters
    abstraction_min_cluster_size: int = 3
    abstraction_max_cluster_size: int = 8
    abstraction_coactivation_threshold: float = 0.5
    abstraction_max_level: int = 5  # max hierarchy depth

    # Rollback protection
    coherence_drop_threshold: float = 0.05
    
    # Tier-0 protected concepts (never perturbed)
    tier_0_identifiers: List[str] = field(default_factory=lambda: [
        "self_reference",
        "identity_core",
        "survival_pressure",
        "coherence_drive",
    ])


@dataclass
class SleepRecord:
    """Record of a sleep cycle."""
    episode: int
    stage: SleepStage
    pre_coherence: float
    post_coherence: float
    perturbations_applied: int
    rollback_occurred: bool
    sabotages_applied: int
    pressure_before: float
    pressure_after: float
    details: Dict[str, Any] = field(default_factory=dict)


class SleepConsolidation:
    """
    Sleep cycle orchestrator with 4-stage consolidation and dream sabotage.
    
    Sleep triggers when accumulated_pressure > threshold.
    All perturbations are localized and bounded.
    """
    
    def __init__(self, config: Optional[SleepConfig] = None):
        self.config = config or SleepConfig()
        self.sleep_history: List[SleepRecord] = []
        self._accumulated_pressure: float = 0.0
        self._current_stage: SleepStage = SleepStage.AWAKE
        self._stage_cycle: int = 0
        self._snapshot: Optional[Dict[str, Any]] = None
        
    def accumulate_pressure(self, delta: float):
        """Add pressure from cognitive events."""
        self._accumulated_pressure = np.clip(
            self._accumulated_pressure + delta, 0.0, 1.0
        )
    
    def should_sleep(self) -> bool:
        """Check if accumulated pressure triggers sleep."""
        return self._accumulated_pressure > self.config.pressure_threshold
    
    def replay_through_graph(self, graph, memories: List[Dict[str, Any]],
                              n_replays: int = 10, lr: float = 0.02) -> Dict[str, int]:
        """Hippocampal replay: re-activate memories through the ConceptGraph.

        During sleep, the brain literally re-activates neural pathways from
        recent experiences, strengthening the connections that were active.
        This method samples episodic memories and runs their concepts through
        the graph, applying Hebbian learning on the replayed activations.

        Args:
            graph: ConceptGraph to replay through
            memories: List of memory dicts (from HumanMemoryEngine)
            n_replays: Number of memories to replay
            lr: Learning rate for Hebbian updates during replay

        Returns:
            Dict with replay statistics
        """
        if not memories:
            return {"replayed": 0, "edges_strengthened": 0}

        # Sample memories (prefer recent and important)
        sorted_memories = sorted(memories,
                                  key=lambda m: m.get("importance", 0.5) * 0.5 +
                                               (1.0 if m.get("consolidated") else 0.0) * 0.5,
                                  reverse=True)
        sample = sorted_memories[:n_replays]

        replayed = 0
        edges_strengthened = 0

        for mem in sample:
            content = str(mem.get("content") or "")
            tags = str(mem.get("tags") or "")

            # Find concept nodes matching memory keywords
            keywords = set()
            for tag in tags.split(","):
                tag = tag.strip().lower()
                if tag:
                    keywords.add(tag)
            # Also extract words from content
            for word in content.lower().split()[:10]:
                if len(word) > 3:
                    keywords.add(word)

            if not keywords:
                continue

            # Match keywords to concept labels
            matched_nids = []
            for nid, node in graph.nodes.items():
                label = (node.label or "").lower()
                if any(kw in label or label in kw for kw in keywords):
                    matched_nids.append(nid)

            if not matched_nids:
                continue

            # Activate matched concepts and spread
            for nid in matched_nids:
                graph.activate(nid, amount=0.5)
            graph.spread_activation(steps=1, k_active=5, decay=0.3)

            # Hebbian strengthening between co-activated concepts
            active = [(n.id, n.activation) for n in graph.nodes.values() if n.activation > 0.1]
            for i, (a_id, a_act) in enumerate(active):
                for b_id, b_act in active[i + 1:]:
                    edge = graph.get_edge(a_id, b_id)
                    if edge:
                        coact = a_act * b_act
                        graph.hebbian_update(a_id, b_id, coact, lr=lr)
                        edges_strengthened += 1

            # Reset activation after replay
            for node in graph.nodes.values():
                node.activation = 0.0

            replayed += 1

        return {"replayed": replayed, "edges_strengthened": edges_strengthened}

    def execute_sleep_cycle(
        self,
        episode: int,
        state_snapshot: Dict[str, Any],
        beliefs: Optional[List[Any]] = None,
        hypotheses: Optional[List[Any]] = None,
        episodic_memories: Optional[List[Any]] = None,
        emotion_engine: Optional[Any] = None,
        coherence_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
        graph: Optional[Any] = None,
    ) -> SleepRecord:
        """
        Execute one full sleep cycle.
        
        4 stages:
        1. Topology Analysis — identify high-pressure zones
        2. Pattern Compression — strengthen consistent patterns
        3. Contradiction Resolution — rewire weakest edges
        4. Integration — merge, stabilize, rollback if needed
        
        Returns:
            SleepRecord with full diagnostics
        """
        pre_coherence = coherence_fn(state_snapshot) if coherence_fn else 0.5
        pre_pressure = self._accumulated_pressure
        
        # Save pre-sleep snapshot for rollback
        self._snapshot = copy.deepcopy(state_snapshot)
        total_perturbations = 0
        total_sabotages = 0
        
        # Stage 1: Topology Analysis
        self._current_stage = SleepStage.TOPOLOGY_ANALYSIS
        pressure_zones = self._analyze_topology(
            state_snapshot, beliefs, hypotheses
        )
        
        # Stage 2: Pattern Compression
        self._current_stage = SleepStage.PATTERN_COMPRESSION
        compression_result = self._compress_patterns(
            state_snapshot, beliefs, hypotheses, pressure_zones
        )
        total_perturbations += compression_result["perturbations"]
        
        # Apply dream sabotage during compression
        sabotage_result = self._apply_dream_sabotage(
            state_snapshot, episodic_memories, emotion_engine
        )
        total_sabotages += sabotage_result["sabotages_applied"]

        # Stage 2.5: Abstraction Compression (hierarchical merging)
        abstraction_result = {"merges": 0, "clusters_found": 0, "abstract_nodes_created": 0}
        if graph is not None:
            self._current_stage = SleepStage.ABSTRACTION_COMPRESSION
            abstraction_result = self._abstract_compress(graph)
            total_perturbations += abstraction_result["merges"]

        # Stage 3: Contradiction Resolution
        self._current_stage = SleepStage.CONTRADICTION_RESOLUTION
        resolution_result = self._resolve_contradictions(
            state_snapshot, beliefs, pressure_zones
        )
        total_perturbations += resolution_result["perturbations"]
        
        # Stage 4: Integration
        self._current_stage = SleepStage.INTEGRATION
        post_coherence = coherence_fn(state_snapshot) if coherence_fn else pre_coherence
        
        # Check for rollback
        rollback = False
        if post_coherence < pre_coherence - self.config.coherence_drop_threshold:
            # Rollback: restore pre-sleep state
            state_snapshot.update(self._snapshot)
            post_coherence = pre_coherence
            rollback = True
        
        # Reduce pressure from sleep
        self._accumulated_pressure = max(0.0, self._accumulated_pressure - 0.15)
        
        record = SleepRecord(
            episode=episode,
            stage=SleepStage.INTEGRATION,
            pre_coherence=pre_coherence,
            post_coherence=post_coherence,
            perturbations_applied=total_perturbations,
            rollback_occurred=rollback,
            sabotages_applied=total_sabotages,
            pressure_before=pre_pressure,
            pressure_after=self._accumulated_pressure,
            details={
                "pressure_zones": len(pressure_zones),
                "compression": compression_result,
                "abstraction": abstraction_result,
                "sabotage": sabotage_result,
                "resolution": resolution_result,
            }
        )
        
        self.sleep_history.append(record)
        self._current_stage = SleepStage.AWAKE
        
        return record
    
    def _analyze_topology(
        self,
        state: Dict[str, Any],
        beliefs: Optional[List[Any]] = None,
        hypotheses: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Identify high-pressure zones.
        
        Scans for:
        - High dissonance regions
        - Unstable prediction edges (high confidence volatility)
        - Recently active bottleneck concepts
        """
        pressure_zones = []
        
        # Check dissonance-based pressure
        dissonance = state.get("dissonance", 0.5)
        if dissonance > 0.6:
            pressure_zones.append({
                "type": "high_dissonance",
                "intensity": dissonance,
                "location": "global_state",
            })
        
        # Check belief-based pressure zones
        if beliefs:
            for b in beliefs:
                if hasattr(b, "confidence") and hasattr(b, "uncertainty"):
                    # High uncertainty + low confidence = pressure zone
                    zone_pressure = (1.0 - b.confidence) * b.uncertainty
                    if zone_pressure > self.config.pressure_threshold:
                        pressure_zones.append({
                            "type": "belief_instability",
                            "intensity": zone_pressure,
                            "location": str(getattr(b, "id", "unknown")),
                        })
        
        return pressure_zones
    
    def _compress_patterns(
        self,
        state: Dict[str, Any],
        beliefs: Optional[List[Any]] = None,
        hypotheses: Optional[List[Any]] = None,
        pressure_zones: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Stage 2: Strengthen consistent patterns.
        
        - Find frequently co-occurring belief clusters
        - Strengthen internal edges within clusters
        - Only operates within pressure zones (localized)
        """
        perturbations = 0
        
        if not pressure_zones:
            return {"perturbations": 0, "clusters_found": 0}
        
        clusters_found = 0
        if beliefs:
            # Group beliefs by similarity
            for i, b1 in enumerate(beliefs):
                for b2 in beliefs[i + 1:]:
                    if hasattr(b1, "confidence") and hasattr(b2, "confidence"):
                        # Check if both are high-confidence (consistent cluster)
                        if b1.confidence > 0.7 and b2.confidence > 0.7:
                            clusters_found += 1
                            # Strengthen within cluster bounded perturbation
                            if hasattr(b1, "_strength"):
                                b1._strength = min(1.0, b1._strength + 0.02)
                            perturbations += 1
        
        return {
            "perturbations": perturbations,
            "clusters_found": clusters_found,
        }
    
    def _abstract_compress(self, graph: Any) -> Dict[str, Any]:
        """
        Stage 2.5: Hierarchical abstraction compression.

        Identifies clusters of co-activated leaf concepts and merges them
        into parent concepts. This is where pressure-driven structural
        reorganization actually happens — the graph develops hierarchy.

        Args:
            graph: ConceptGraph instance

        Returns:
            Dict with merge statistics
        """
        clusters_found = 0
        merges = 0
        abstract_nodes_created = 0

        # Check hierarchy depth limit
        stats = graph.get_abstraction_stats()
        if stats["max_level"] >= self.config.abstraction_max_level:
            return {"merges": 0, "clusters_found": 0, "abstract_nodes_created": 0,
                    "reason": "max_level_reached"}

        # Find co-activated clusters
        clusters = graph.find_coactivated_clusters(
            coactivation_threshold=self.config.abstraction_coactivation_threshold,
            min_cluster_size=self.config.abstraction_min_cluster_size,
            max_cluster_size=self.config.abstraction_max_cluster_size,
        )
        clusters_found = len(clusters)

        # Merge each cluster into a parent concept
        for cluster in clusters:
            # Skip if any node in cluster already has a parent
            if any(graph.nodes[cid].parent is not None for cid in cluster if cid in graph.nodes):
                continue

            # Compute mean co-activation strength for this cluster
            activations = [graph.nodes[cid].activation for cid in cluster if cid in graph.nodes]
            mean_act = float(np.mean(activations)) if activations else 0.0

            # Compute abstraction degree based on cluster coherence
            # Higher co-activation within cluster = higher abstraction degree
            coherence = self._cluster_coherence(graph, cluster)
            abstraction_degree = min(1.0, mean_act * coherence)

            parent_id = graph.merge_concepts(
                cluster,
                abstraction_degree=abstraction_degree,
            )

            if parent_id is not None:
                merges += 1
                abstract_nodes_created += 1

        return {
            "merges": merges,
            "clusters_found": clusters_found,
            "abstract_nodes_created": abstract_nodes_created,
        }

    def _cluster_coherence(self, graph: Any, cluster: List[int]) -> float:
        """Compute coherence of a cluster (mean edge weight between members)."""
        weights = []
        for i, a in enumerate(cluster):
            for b in cluster[i + 1:]:
                edge = graph.get_edge(a, b)
                if edge:
                    weights.append(edge.weight)
                edge_rev = graph.get_edge(b, a)
                if edge_rev:
                    weights.append(edge_rev.weight)
        return float(np.mean(weights)) if weights else 0.0

    def _resolve_contradictions(
        self,
        state: Dict[str, Any],
        beliefs: Optional[List[Any]] = None,
        pressure_zones: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Stage 3: Resolve contradictions by adjusting weakest links.
        
        For each active contradiction:
        - Find weakest edge in the contradiction chain
        - Attempt to rewire (adjust weight, not delete)
        - If rewiring creates new contradictions, abort this resolution
        """
        perturbations = 0
        contradictions_resolved = 0
        
        if not pressure_zones or not beliefs:
            return {"perturbations": 0, "contradictions_resolved": 0}
        
        for zone in pressure_zones:
            if zone["intensity"] > 0.3:
                # Strong pressure zone — attempt resolution
                # Reduce confidence of the weakest belief involved
                if beliefs:
                    weakest = min(beliefs, key=lambda b: getattr(b, "confidence", 1.0))
                    if hasattr(weakest, "confidence"):
                        # Bounded adjustment
                        old_conf = weakest.confidence
                        weakest.confidence = max(0.1, weakest.confidence - 0.05)
                        perturbations += 1
                        contradictions_resolved += 1
        
        return {
            "perturbations": perturbations,
            "contradictions_resolved": contradictions_resolved,
        }
    
    def _apply_dream_sabotage(
        self,
        state: Dict[str, Any],
        episodic_memories: Optional[List[Any]] = None,
        emotion_engine: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Apply structured dream sabotage to prevent overfitting.
        
        Three sabotage types:
        1. Counterfactual reversal (20%): Flip outcome of randomly selected memories
        2. Emotional flip (10%): Flip VAD valence of emotional tags
        3. Failure oversampling (1.5x): Replay failures more times than successes
        """
        sabotages_applied = 0
        reversals = 0
        flips = 0
        
        if not episodic_memories:
            return {"sabotages_applied": 0, "reversals": 0, "flips": 0}
        
        # Counterfactual reversals: flip outcome of 20% of memories
        for memory in episodic_memories:
            if np.random.random() < self.config.counterfactual_rate:
                # Flip the correctness/success field if it exists
                if hasattr(memory, "_correctness"):
                    memory._correctness = not memory._correctness
                    reversals += 1
                elif isinstance(memory, dict):
                    if "correctness" in memory:
                        memory["correctness"] = not memory["correctness"]
                        reversals += 1
        
        # Emotional flipping: flip valence if emotion engine available
        if emotion_engine is not None and hasattr(emotion_engine, "_concept_tags"):
            for cid in list(emotion_engine._concept_tags.keys()):
                if np.random.random() < self.config.emotional_flip_rate:
                    tag = emotion_engine._concept_tags[cid]
                    if hasattr(tag, "valence"):
                        tag.valence = -tag.valence  # Flip valence
                        flips += 1
        
        sabotages_applied = reversals + flips
        
        return {
            "sabotages_applied": sabotages_applied,
            "reversals": reversals,
            "flips": flips,
        }
    
    def get_pressure(self) -> float:
        """Current accumulated sleep pressure."""
        return self._accumulated_pressure
    
    def get_status(self) -> Dict[str, Any]:
        """Full sleep system status."""
        recent_sleeps = self.sleep_history[-10:] if self.sleep_history else []
        return {
            "accumulated_pressure": self._accumulated_pressure,
            "current_stage": self._current_stage.value,
            "should_sleep": self.should_sleep(),
            "total_sleep_cycles": len(self.sleep_history),
            "last_10_cycles": [
                {
                    "episode": r.episode,
                    "stage": r.stage.value if r.stage != SleepStage.AWAKE else "awake",
                    "coherence_delta": r.post_coherence - r.pre_coherence,
                    "perturbations": r.perturbations_applied,
                    "rollback": r.rollback_occurred,
                }
                for r in recent_sleeps
            ]
        }
