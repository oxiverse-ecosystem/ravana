"""
RAVANA v2 — PHASE F.5: Belief Reasoner
From stabilizer to reasoner:

- Competing hypotheses (H1, H2, ...) with confidence weights
- Full evidence history with context
- Confidence decay over time (without confirmation)
- Structural consistency checks
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from enum import Enum


class EvidenceType(Enum):
    """Types of evidence."""
    BOUNDARY_PROXIMITY = "boundary_proximity"
    CLAMP_EVENT = "clamp_event"
    DISSONANCE_SPIKE = "dissonance_spike"
    PATTERN_CONSISTENCY = "pattern_consistency"


@dataclass
class Hypothesis:
    """
    A competing hypothesis about the world.
    
    Not just a boundary estimate, but a confidence-weighted belief
    with temporal tracking.
    """
    id: int
    boundary_estimate: float = 0.95
    confidence: float = 0.5  # 0-1, separate from estimate
    uncertainty: float = 0.1  # Standard deviation
    
    # Evidence tracking
    evidence_count: int = 0
    confirming_evidence: int = 0
    contradicting_evidence: int = 0
    
    # Temporal
    created_episode: int = 0
    last_updated: int = 0
    
    # History for structural checks
    prediction_history: List[Dict] = field(default_factory=list)
    
    def update_confidence(self, confirmation: float, episode: int):
        """
        Update confidence based on confirmation.
        confirmation: 1.0 = strong confirm, -1.0 = strong contradict
        """
        if confirmation > 0:
            # Confirming evidence boosts confidence
            self.confidence = min(1.0, self.confidence + confirmation * 0.15)
            self.confirming_evidence += 1
        else:
            # Contradicting evidence reduces confidence
            self.confidence = max(0.1, self.confidence + confirmation * 0.2)
            self.contradicting_evidence += 1
        
        self.last_updated = episode
        self.evidence_count += 1
    
    def decay_confidence(self, decay_rate: float):
        """Decay confidence without fresh confirmation."""
        self.confidence = max(0.1, self.confidence * (1 - decay_rate))
    
    def compute_weight(self) -> float:
        """Compute hypothesis weight for evidence distribution."""
        return self.confidence * (1 - self.uncertainty)


@dataclass
class EvidenceEvent:
    """A single piece of evidence with full context."""
    episode: int
    predicted_d: float
    actual_d: float
    observed_boundary: float
    mode: int
    clamp_occurred: bool
    context_snapshot: Dict[str, float]
    
    # Derived
    surprise: float = 0.0
    evidence_type: Optional[EvidenceType] = None


@dataclass
class BeliefConfig:
    """Configuration for belief reasoner."""
    max_hypotheses: int = 3
    confidence_decay_rate: float = 0.02  # Per episode
    structural_consistency_threshold: float = 0.7
    
    # Evidence thresholds
    strong_confirm_threshold: float = 0.8
    strong_contradict_threshold: float = 0.3
    
    # Hypothesis management
    spawn_new_hypothesis_threshold: float = 0.4  # If all hyps < this confidence
    prune_low_confidence_threshold: float = 0.15


class BeliefReasoner:
    """
    Maintains multiple competing hypotheses about the world.
    
    Key innovation:
    - Not single belief, but distribution over hypotheses
    - Confidence decays without confirmation (skepticism)
    - Structural consistency checks ("would past make sense?")
    """
    
    def __init__(self, config: Optional[BeliefConfig] = None):
        self.config = config or BeliefConfig()
        
        # Hypothesis ID counter (MUST be before spawning)
        self._next_hypothesis_id: int = 1
        
        # Multiple hypotheses
        self.hypotheses: List[Hypothesis] = []
        self._spawn_initial_hypothesis()
        
        # Full evidence history
        self.evidence_history: List[EvidenceEvent] = []
        
        # Rejection log (for analysis)
        self.rejected_hypotheses: List[Dict] = []
        self.structural_rejections: int = 0
    
    def _spawn_initial_hypothesis(self):
        """Create initial hypothesis (conservative default)."""
        h = Hypothesis(
            id=self._next_hypothesis_id,
            boundary_estimate=0.95,
            confidence=0.5,
            uncertainty=0.1,
            created_episode=0
        )
        self.hypotheses.append(h)
        self._next_hypothesis_id += 1
    
    def _spawn_hypothesis_from_evidence(self, evidence: EvidenceEvent) -> Optional[Hypothesis]:
        """
        Spawn new hypothesis from surprising evidence.
        
        If evidence contradicts all current hypotheses, consider new hypothesis.
        """
        # Infer boundary from evidence
        inferred_boundary = evidence.observed_boundary
        
        # Check if this is genuinely novel (not close to existing hypotheses)
        for h in self.hypotheses:
            if abs(h.boundary_estimate - inferred_boundary) < 0.05:
                return None  # Too close to existing
        
        # Spawn new hypothesis
        h = Hypothesis(
            id=self._next_hypothesis_id,
            boundary_estimate=inferred_boundary,
            confidence=0.3,  # Start with low confidence
            uncertainty=0.2,
            created_episode=evidence.episode,
            last_updated=evidence.episode
        )
        self._next_hypothesis_id += 1
        return h
    
    def _check_structural_consistency(self, hypothesis: Hypothesis, 
                                      new_boundary: float) -> Tuple[bool, float]:
        """
        CRITICAL: Would the past make sense under new belief?
        
        Simulates: "If this new boundary were true, would my past predictions hold?"
        
        Returns: (is_consistent, consistency_score)
        """
        if len(hypothesis.prediction_history) < 5:
            return True, 1.0  # Not enough history
        
        # Simulate past episodes under new boundary
        errors_under_new_belief = []
        errors_under_old_belief = []
        
        for pred in hypothesis.prediction_history[-10:]:  # Last 10 predictions
            # Error under current belief
            old_error = abs(pred['actual'] - pred['predicted_under_current'])
            errors_under_old_belief.append(old_error)
            
            # Simulate what error would be under new boundary
            simulated_prediction = self._simulate_prediction(
                pred['context'], new_boundary
            )
            new_error = abs(pred['actual'] - simulated_prediction)
            errors_under_new_belief.append(new_error)
        
        # Compare: is new belief better at explaining past?
        mean_old = np.mean(errors_under_old_belief)
        mean_new = np.mean(errors_under_new_belief)
        
        # Consistency: new belief should not be MUCH worse
        consistency_score = mean_old / (mean_new + 0.001)  # > 1 means new is better
        
        # Reject if new belief makes past incomprehensible
        is_consistent = consistency_score > self.config.structural_consistency_threshold
        
        return is_consistent, consistency_score
    
    def _simulate_prediction(self, context: Dict, boundary: float) -> float:
        """Simulate what would be predicted given context and boundary."""
        # Simplified: closer to boundary → higher dissonance risk
        base_d = context.get('dissonance', 0.5)
        proximity = base_d / boundary if boundary > 0 else 0
        
        if proximity > 0.9:
            return base_d + 0.05  # Expect increase near boundary
        return base_d
    
    def observe_evidence(self, evidence: EvidenceEvent, true_boundary: float):
        """
        Process new evidence.
        
        1. Distribute evidence to hypotheses
        2. Update confidences
        3. Apply decay
        4. Check structural consistency
        5. Spawn/prune hypotheses
        """
        # Compute surprise
        evidence.surprise = abs(evidence.actual_d - evidence.predicted_d)
        
        # Classify evidence type
        if evidence.clamp_occurred:
            evidence.evidence_type = EvidenceType.CLAMP_EVENT
        elif evidence.surprise > 0.1:
            evidence.evidence_type = EvidenceType.DISSONANCE_SPIKE
        elif evidence.observed_boundary > 0.8:
            evidence.evidence_type = EvidenceType.BOUNDARY_PROXIMITY
        else:
            evidence.evidence_type = EvidenceType.PATTERN_CONSISTENCY
        
        # Store evidence
        self.evidence_history.append(evidence)
        
        # Distribute to hypotheses
        for hypothesis in self.hypotheses:
            # Each hypothesis "explains" evidence differently
            predicted_under_hyp = self._predict_dissonance(
                evidence.context_snapshot, 
                hypothesis.boundary_estimate
            )
            
            # Store prediction for structural checks
            hypothesis.prediction_history.append({
                'episode': evidence.episode,
                'predicted_under_current': predicted_under_hyp,
                'actual': evidence.actual_d,
                'context': evidence.context_snapshot
            })
            
            # Compute confirmation
            error = abs(evidence.actual_d - predicted_under_hyp)
            confirmation = 1.0 - min(1.0, error / 0.2)  # Normalize
            
            # Strong boundary signal?
            if evidence.observed_boundary > 0.8:
                # Hypotheses close to observed boundary get boosted
                boundary_agreement = 1.0 - abs(hypothesis.boundary_estimate - evidence.observed_boundary)
                confirmation *= (0.5 + 0.5 * boundary_agreement)
            
            # Update hypothesis confidence
            hypothesis.update_confidence(confirmation - 0.5, evidence.episode)  # Center at 0
        
        # Apply confidence decay to all hypotheses
        for hypothesis in self.hypotheses:
            hypothesis.decay_confidence(self.config.confidence_decay_rate)
        
        # Check for new hypothesis spawn
        max_conf = max((h.confidence for h in self.hypotheses), default=0)
        if max_conf < self.config.spawn_new_hypothesis_threshold:
            new_hyp = self._spawn_hypothesis_from_evidence(evidence)
            if new_hyp and len(self.hypotheses) < self.config.max_hypotheses:
                self.hypotheses.append(new_hyp)
        
        # Prune low-confidence hypotheses
        self._prune_hypotheses()
    
    def _predict_dissonance(self, context: Dict, boundary: float) -> float:
        """Predict dissonance given context and boundary hypothesis."""
        current_d = context.get('dissonance', 0.5)
        clamp_rate = context.get('clamp_rate', 0.0)
        
        # Near boundary → expect pressure
        proximity = current_d / boundary if boundary > 0 else 0
        if proximity > 0.85:
            return current_d + 0.03 + clamp_rate * 0.05
        return current_d
    
    def _prune_hypotheses(self):
        """Remove low-confidence hypotheses."""
        kept = []
        for h in self.hypotheses:
            if h.confidence > self.config.prune_low_confidence_threshold:
                kept.append(h)
            else:
                self.rejected_hypotheses.append({
                    'hypothesis': h,
                    'reason': 'low_confidence',
                    'final_confidence': h.confidence
                })
        self.hypotheses = kept
    
    def get_dominant_hypothesis(self) -> Optional[Any]:
        """Get highest-confidence hypothesis."""
        if not self.hypotheses:
            return None
        # Handle both Hypothesis objects and dicts
        def get_confidence(h):
            if isinstance(h, dict):
                return h.get('confidence', 0.5)
            return getattr(h, 'confidence', 0.5)
        return max(self.hypotheses, key=get_confidence)
    
    def get_mode_recommendation(self) -> str:
        """Recommend exploration mode based on belief uncertainty."""
        dominant = self.get_dominant_hypothesis()
        if not dominant:
            return "explore_safe"
        
        # High uncertainty → cautious exploration
        if dominant.uncertainty > 0.15 or dominant.confidence < 0.4:
            return "explore_safe"
        
        # Confident near boundary → stabilize
        if dominant.confidence > 0.7 and dominant.boundary_estimate < 0.8:
            return "stabilize"
        
        return "explore_safe"
    
    def get_reasoning_status(self) -> Dict[str, Any]:
        """
        Return full reasoning status.
        """
        total_confidence = sum(h.confidence for h in self.hypotheses)
        
        return {
            'num_hypotheses': len(self.hypotheses),
            'hypothesis_weights': {h.id: h.compute_weight() for h in self.hypotheses},
            'total_evidence': len(self.evidence_history),
            'structural_rejections': self.structural_rejections,
            'total_confidence_decay': sum(
                h.confidence * (1 - 0.98) for h in self.hypotheses  # Approximate total decay
            ),
            'rejected_hypothesis_count': len(self.rejected_hypotheses),
            'dominant_boundary': self.get_dominant_hypothesis().boundary_estimate if self.hypotheses else None,
            'dominant_confidence': self.get_dominant_hypothesis().confidence if self.hypotheses else None,
        }
    
    def get_belief_state(self) -> List[Hypothesis]:
        """
        Return the current set of hypotheses.
        """
        return self.hypotheses

    @property
    def current_belief(self) -> float:
        """Current best belief estimate."""
        dominant = self.get_dominant_hypothesis()
        if not dominant:
            return 0.5
        if isinstance(dominant, dict):
            return dominant.get('boundary_estimate', 0.5)
        return getattr(dominant, 'boundary_estimate', 0.5)
    
    @property  
    def current_uncertainty(self) -> float:
        """Current uncertainty estimate."""
        dominant = self.get_dominant_hypothesis()
        if not dominant:
            return 0.5
        if isinstance(dominant, dict):
            return 1.0 - dominant.get('confidence', 0.5)
        return 1.0 - getattr(dominant, 'confidence', 0.5)
