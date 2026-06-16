"""
RAVANA v2 — PHASE J: Hypothesis Generation
From "which hypothesis is correct?" → "what hypothesis should I consider?"

PRINCIPLE: Constraint-guided hypothesis generation when KL probing plateaus.

TRIGGER:
    if KL plateau AND uncertainty persists AND rising dissonance:
        spawn_new_hypothesis()

HYPOTHESIS TYPES (incremental complexity):
    1. Parametric variations: boundary = f(time), boundary = f(state)
    2. Structural mutations: dual-boundary, hidden constraints, asymmetric
    3. Causal hypotheses: "what causes boundary change?"

LIFECYCLE MANAGEMENT:
    - Confidence: belief strength
    - Complexity: Occam penalty (prevent overfitting)
    - Explanatory power: fit to data
    - Survival time: prune weak hypotheses
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum, auto
import numpy as np
from collections import deque


class HypothesisType(Enum):
    """Types of hypothesis generation, ordered by complexity/risk."""
    PARAMETRIC_TIME = auto()      # boundary = f(time) - SAFEST
    PARAMETRIC_STATE = auto()     # boundary = f(state)
    STRUCTURAL_DUAL = auto()      # dual-boundary zones
    STRUCTURAL_ASYMMETRIC = auto()  # asymmetric boundaries
    CAUSAL_CORRELATE = auto()     # boundary linked to X
    CAUSAL_MECHANISM = auto()     # "what causes boundary change?"


@dataclass
class GeneratedHypothesis:
    """A generated hypothesis with full provenance."""
    id: int
    hypothesis_type: HypothesisType
    
    # Core belief
    boundary_model: Callable[[int, Dict], float]  # f(episode, context) -> boundary
    
    # Metadata
    birth_episode: int
    parent_hypothesis_id: Optional[int]  # What triggered this generation?
    generation_trigger: str  # Why was this created?
    
    # Quality metrics (updated over time)
    confidence: float = 0.5
    complexity_score: float = 0.0  # Occam penalty
    explanatory_power: float = 0.0   # How well it fits data
    survival_score: float = 1.0    # Time-discounted quality
    
    # Evidence tracking
    evidence_count: int = 0
    prediction_errors: List[float] = field(default_factory=list)
    
    def predict_boundary(self, episode: int, context: Dict) -> float:
        """Predict boundary for this episode under this hypothesis."""
        return self.boundary_model(episode, context)
    
    def update_quality(self, prediction_error: float, episode: int):
        """Update hypothesis quality based on new evidence."""
        self.evidence_count += 1
        self.prediction_errors.append(prediction_error)
        
        # Keep only recent errors
        if len(self.prediction_errors) > 50:
            self.prediction_errors.pop(0)
        
        # Update explanatory power (inverse of mean error)
        if self.prediction_errors:
            mean_error = np.mean(self.prediction_errors[-20:])
            self.explanatory_power = max(0, 1.0 - mean_error * 2)
        
        # Update confidence with exponential decay
        age_factor = np.exp(-0.001 * (episode - self.birth_episode))
        self.survival_score = self.explanatory_power * age_factor


@dataclass
class GenerationConfig:
    """Configuration for hypothesis generation."""
    # Trigger thresholds
    kl_plateau_threshold: float = 0.05    # KL gain below this = plateau
    uncertainty_persistence_window: int = 30  # Episodes of sustained uncertainty
    dissonance_rise_threshold: float = 0.05   # D must rise by this much
    
    # Generation limits
    max_hypotheses: int = 6               # Hard cap to prevent explosion
    min_episodes_between_generations: int = 50  # Don't spam
    
    # Complexity penalties (Occam factors)
    complexity_penalty: Dict[HypothesisType, float] = field(default_factory=lambda: {
        HypothesisType.PARAMETRIC_TIME: 0.1,
        HypothesisType.PARAMETRIC_STATE: 0.15,
        HypothesisType.STRUCTURAL_DUAL: 0.25,
        HypothesisType.STRUCTURAL_ASYMMETRIC: 0.30,
        HypothesisType.CAUSAL_CORRELATE: 0.35,
        HypothesisType.CAUSAL_MECHANISM: 0.40
    })
    
    # Survival thresholds
    min_survival_score: float = 0.2       # Prune below this
    survival_window: int = 100            # Episodes to consider
    
    # Generation strategy
    preferred_types_order: List[HypothesisType] = field(default_factory=lambda: [
        HypothesisType.PARAMETRIC_TIME,
        HypothesisType.PARAMETRIC_STATE,
        HypothesisType.STRUCTURAL_DUAL,
        HypothesisType.STRUCTURAL_ASYMMETRIC,
        HypothesisType.CAUSAL_CORRELATE,
        HypothesisType.CAUSAL_MECHANISM
    ])


class HypothesisGenerator:
    """
    Generate new hypotheses when existing ones fail to explain the world.
    
    CORE LOOP:
        1. Monitor: Track KL gain, uncertainty, dissonance
        2. Detect: Identify when probing plateaus
        3. Generate: Create new hypothesis from next complexity tier
        4. Evaluate: Track quality, prune weak hypotheses
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        
        # Hypothesis management
        self.hypotheses: Dict[int, GeneratedHypothesis] = {}
        self._next_hypothesis_id: int = 1
        
        # Monitoring
        self.kl_history: deque = deque(maxlen=50)
        self.uncertainty_history: deque = deque(maxlen=50)
        self.dissonance_history: deque = deque(maxlen=50)
        
        # Generation tracking
        self.last_generation_episode: int = -100
        self.generation_count: int = 0
        
        # Analytics
        self.generation_history: List[Dict] = []
        self.pruned_hypotheses: List[int] = []
    
    def monitor_state(
        self,
        episode: int,
        kl_gain: float,
        uncertainty: float,
        dissonance: float,
        hypotheses: List[Any]
    ) -> Dict[str, Any]:
        """
        Monitor epistemic state. Returns detection signals.
        """
        self.kl_history.append(kl_gain)
        self.uncertainty_history.append(uncertainty)
        self.dissonance_history.append(dissonance)
        
        # Update all hypothesis qualities
        self._update_hypothesis_qualities(episode, hypotheses)
        
        # Run lifecycle management (pruning)
        pruned = self._prune_weak_hypotheses(episode)
        
        # Detect generation triggers
        triggers = self._detect_triggers(episode)
        
        return {
            "triggers_detected": triggers,
            "pruned_hypotheses": pruned,
            "active_hypotheses": len(self.hypotheses),
            "should_generate": len(triggers) > 0 and self._can_generate(episode)
        }
    
    def _detect_triggers(self, episode: int) -> List[str]:
        """Detect which generation conditions are met."""
        triggers = []
        
        # 1. KL Plateau: Low information gain across recent probes
        if len(self.kl_history) >= 10:
            recent_kl = list(self.kl_history)[-10:]
            mean_kl = np.mean([kl for kl in recent_kl if kl > 0])
            if mean_kl < self.config.kl_plateau_threshold:
                triggers.append(f"kl_plateau({mean_kl:.3f})")
        
        # 2. Persistent Uncertainty: Sustained high uncertainty
        if len(self.uncertainty_history) >= self.config.uncertainty_persistence_window:
            recent_unc = list(self.uncertainty_history)[-self.config.uncertainty_persistence_window:]
            if np.mean(recent_unc) > 0.3:  # Sustained uncertainty
                triggers.append(f"persistent_uncertainty({np.mean(recent_unc):.3f})")
        
        # 3. Rising Dissonance: Epistemic stress increasing
        if len(self.dissonance_history) >= 20:
            early = np.mean(list(self.dissonance_history)[:10])
            late = np.mean(list(self.dissonance_history)[-10:])
            if late - early > self.config.dissonance_rise_threshold:
                triggers.append(f"rising_dissonance({late-early:.3f})")
        
        return triggers
    
    def _can_generate(self, episode: int) -> bool:
        """Check if generation is allowed (rate limiting, capacity)."""
        # Rate limit
        if episode - self.last_generation_episode < self.config.min_episodes_between_generations:
            return False
        
        # Capacity limit
        if len(self.hypotheses) >= self.config.max_hypotheses:
            return False
        
        return True
    
    def generate_hypothesis(
        self,
        episode: int,
        current_hypotheses: List[Any],
        triggers: List[str]
    ) -> Optional[GeneratedHypothesis]:
        """
        Generate new hypothesis when triggers fire.
        
        STRATEGY: Start with simplest unexplored hypothesis type.
        """
        if not self._can_generate(episode):
            return None
        
        # Determine what types already exist
        existing_types = {h.hypothesis_type for h in self.hypotheses.values()}
        
        # Find next unexplored type in order of complexity
        selected_type = None
        for htype in self.config.preferred_types_order:
            if htype not in existing_types:
                selected_type = htype
                break
        
        # If all types exist, pick the one with lowest current confidence
        if selected_type is None:
            lowest_conf_type = min(
                existing_types,
                key=lambda t: np.mean([h.confidence for h in self.hypotheses.values() if h.hypothesis_type == t]) if any(h.hypothesis_type == t for h in self.hypotheses.values()) else 1.0
            )
            selected_type = lowest_conf_type
        
        # Generate the hypothesis model
        hypothesis = self._create_hypothesis_model(selected_type, episode, current_hypotheses)
        
        if hypothesis:
            self.last_generation_episode = episode
            self.generation_count += 1
            self.hypotheses[hypothesis.id] = hypothesis
            
            self.generation_history.append({
                "episode": episode,
                "hypothesis_id": hypothesis.id,
                "type": selected_type.name,
                "triggers": triggers.copy(),
                "parent_ids": [h.id for h in current_hypotheses] if current_hypotheses else []
            })
            
            return hypothesis
        
        return None
    
    def _create_hypothesis_model(
        self,
        htype: HypothesisType,
        episode: int,
        parents: List[Any]
    ) -> Optional[GeneratedHypothesis]:
        """Create hypothesis model based on type."""
        
        if htype == HypothesisType.PARAMETRIC_TIME:
            # boundary = base + amplitude * sin(omega * time)
            # Estimate from recent boundary observations
            omega = 2 * np.pi / 200  # Slow oscillation
            amplitude = 0.1
            base = 0.75  # Midpoint estimate
            
            def boundary_model(ep, ctx):
                return base + amplitude * np.sin(omega * ep)
            
            complexity = self.config.complexity_penalty[htype]
            
        elif htype == HypothesisType.PARAMETRIC_STATE:
            # boundary = f(current_dissonance)
            # Higher dissonance → lower effective boundary (stress compresses range)
            def boundary_model(ep, ctx):
                d = ctx.get('dissonance', 0.5)
                # Stress-adaptive: high D → lower ceiling
                return 0.8 - 0.2 * max(0, d - 0.5)
            
            complexity = self.config.complexity_penalty[htype]
            
        elif htype == HypothesisType.STRUCTURAL_DUAL:
            # Two zones: exploration zone and resolution zone
            def boundary_model(ep, ctx):
                i = ctx.get('identity', 0.5)
                # Strong identity → higher boundary (confident exploration)
                if i > 0.6:
                    return 0.90  # High zone
                else:
                    return 0.70  # Conservative zone
            
            complexity = self.config.complexity_penalty[htype]
            
        elif htype == HypothesisType.STRUCTURAL_ASYMMETRIC:
            # Different boundaries for increasing vs decreasing
            def boundary_model(ep, ctx):
                trend = ctx.get('dissonance_trend', 0)
                if trend > 0:  # Rising
                    return 0.75  # Tighter when rising
                else:
                    return 0.85  # Looser when falling
            
            complexity = self.config.complexity_penalty[htype]
            
        else:
            # Default: simple constant boundary
            def boundary_model(ep, ctx):
                return 0.75
            
            complexity = 0.5
        
        hypothesis = GeneratedHypothesis(
            id=self._next_hypothesis_id,
            hypothesis_type=htype,
            boundary_model=boundary_model,
            birth_episode=episode,
            parent_hypothesis_id=parents[0].id if parents else None,
            generation_trigger="kl_plateau_and_uncertainty",
            complexity_score=complexity
        )
        
        self._next_hypothesis_id += 1
        return hypothesis
    
    def _update_hypothesis_qualities(self, episode: int, current_hypotheses: List[Any]):
        """Update quality metrics for all hypotheses."""
        # This would compare predictions to actual outcomes
        # For now, placeholder
        pass
    
    def _prune_weak_hypotheses(self, episode: int) -> List[int]:
        """Remove hypotheses with low survival scores."""
        pruned = []
        to_remove = []
        
        for hid, hyp in self.hypotheses.items():
            # Prune if survival score too low
            if hyp.survival_score < self.config.min_survival_score and hyp.evidence_count > 20:
                to_remove.append(hid)
                pruned.append({
                    "id": hid,
                    "type": hyp.hypothesis_type.name,
                    "survival_score": hyp.survival_score,
                    "evidence_count": hyp.evidence_count,
                    "age": episode - hyp.birth_episode
                })
        
        for hid in to_remove:
            del self.hypotheses[hid]
            self.pruned_hypotheses.append(hid)
        
        return pruned
    
    def get_active_hypotheses(self) -> List[GeneratedHypothesis]:
        """Return current active hypotheses sorted by survival score."""
        return sorted(
            self.hypotheses.values(),
            key=lambda h: h.survival_score,
            reverse=True
        )
    
    def get_generation_status(self) -> Dict[str, Any]:
        """Full status of hypothesis generation system."""
        type_distribution = {}
        for h in self.hypotheses.values():
            t = h.hypothesis_type.name
            type_distribution[t] = type_distribution.get(t, 0) + 1
        
        return {
            "total_generated": self.generation_count,
            "currently_active": len(self.hypotheses),
            "total_pruned": len(self.pruned_hypotheses),
            "type_distribution": type_distribution,
            "generation_history": self.generation_history[-5:],
            "recent_triggers": list(self.kl_history)[-10:] if self.kl_history else [],
            "avg_uncertainty": np.mean(self.uncertainty_history) if self.uncertainty_history else 0,
            "avg_dissonance": np.mean(self.dissonance_history) if self.dissonance_history else 0
        }


# Convenience alias
HypothesisGeneration = HypothesisGenerator
