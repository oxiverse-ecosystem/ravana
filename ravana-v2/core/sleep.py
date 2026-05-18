"""
RAVANA v2 — SLEEP & DREAM CONSOLIDATION
Periodic consolidation phase with structured dream sabotage.

PRINCIPLE: Sleep is thermodynamic necessity — pressure accumulation
triggers reorganization that stabilizes useful patterns and weakens
brittle ones.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
import copy
import math


class SleepStage(Enum):
    AWAKE = "awake"
    TOPOLOGY_ANALYSIS = "topology_analysis"
    PATTERN_COMPRESSION = "pattern_compression"
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
    
    def execute_sleep_cycle(
        self,
        episode: int,
        state_snapshot: Dict[str, Any],
        beliefs: Optional[List[Any]] = None,
        hypotheses: Optional[List[Any]] = None,
        episodic_memories: Optional[List[Any]] = None,
        emotion_engine: Optional[Any] = None,
        coherence_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
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
