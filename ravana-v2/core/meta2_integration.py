"""
RAVANA v2 — PHASE I²: Meta²-Hypothesis Integration
Wires Meta² cognition into live hypothesis generation.

PRINCIPLE: When systematic failure is detected, expand hypothesis space automatically.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import deque

from .meta2_cognition import Meta2CognitionEngine, Meta2Config
from .hypothesis_generation import HypothesisGenerator, GenerationConfig, HypothesisType, GeneratedHypothesis


@dataclass
class Meta2GenerationConfig:
    """Configuration for Meta²-aware hypothesis generation."""
    # Meta² trigger thresholds
    min_epiphanies_for_expansion: int = 1  # Need at least 1 epiphany
    systematic_failure_rate_threshold: float = 0.3  # 30% failure = systematic
    
    # Expansion strategy
    expand_to_causal: bool = True  # Allow causal hypotheses
    expand_to_unconventional: bool = True  # Allow non-standard types
    
    # Integration weights
    meta2_weight: float = 0.5  # How much Meta² overrides standard generation


class Meta2IntegratedGenerator:
    """
    Hypothesis generator that listens to Meta² critiques.
    
    This is the critical integration: when Meta² detects the hypothesis space
    is inadequate, this generator EXPANDS the space rather than optimizing
    within the inadequate space.
    """
    
    def __init__(
        self,
        base_generator: HypothesisGenerator,
        meta2_engine: Meta2CognitionEngine,
        config: Optional[Meta2GenerationConfig] = None
    ):
        self.base_generator = base_generator
        self.meta2 = meta2_engine
        self.config = config or Meta2GenerationConfig()
        
        # Track epistemic crises
        self.epistemic_crises: List[Dict] = []
        self.expansion_events: List[Dict] = []
        
        # Track hypothesis space evolution
        self.allowed_hypothesis_types: set = {
            HypothesisType.PARAMETRIC_TIME,
            HypothesisType.PARAMETRIC_STATE,
            HypothesisType.STRUCTURAL_DUAL,
            HypothesisType.STRUCTURAL_ASYMMETRIC,
        }
        self.causal_types_unlocked: bool = False
        self.unconventional_types_unlocked: bool = False
        
        # Success tracking
        self.prediction_history: deque = deque(maxlen=100)
        self.failure_streak: int = 0
    
    def generate_with_meta2(
        self,
        episode: int,
        current_hypotheses: List[Any],
        kl_gain: float,
        uncertainty: float,
        dissonance: float
    ) -> Optional[GeneratedHypothesis]:
        """
        Generate hypothesis with Meta² awareness.
        
        CRITICAL: If Meta² detects systematic failure, expand hypothesis space
        before generating, not just generate within inadequate space.
        """
        # First, run Meta² monitoring
        meta2_state = self.meta2.step(episode, dissonance, uncertainty)
        
        # Check if we have an epiphany about hypothesis space inadequacy
        space_inadequate = any(
            audit.critique_type == 'space_inadequate'
            for audit in meta2_state['hypothesis_audits']
        )
        
        # Also check for systematic failure in recent predictions
        recent_failure_rate = self._compute_failure_rate(window=20)
        systematic_failure = recent_failure_rate > self.config.systematic_failure_rate_threshold
        
        # EXPAND HYPOTHESIS SPACE if needed
        if space_inadequate or systematic_failure:
            expansion = self._expand_hypothesis_space(episode, meta2_state)
            if expansion:
                self.expansion_events.append(expansion)
        
        # Now generate with potentially expanded space
        hypothesis = self._generate_in_expanded_space(
            episode, current_hypotheses, kl_gain, uncertainty, dissonance
        )
        
        return hypothesis
    
    def _compute_failure_rate(self, window: int = 20) -> float:
        """Compute recent prediction failure rate."""
        if len(self.prediction_history) < window:
            return 0.0
        
        recent = list(self.prediction_history)[-window:]
        failures = sum(1 for r in recent if r['error'] > 0.1)
        return failures / len(recent)
    
    def _expand_hypothesis_space(self, episode: int, meta2_state: Dict) -> Optional[Dict]:
        """
        EXPAND the hypothesis space based on Meta² critique.
        
        This is the key Meta² intervention: when the space is inadequate,
        we don't just try harder within the space — we change the space.
        """
        expansion_triggered = False
        expansion_types = []
        
        # Check for bias-related critiques
        bias_critiques = [c for c in meta2_state['bias_critiques'] if c['severity'] > 0.6]
        
        for critique in bias_critiques:
            bias_type = critique.get('bias_type')
            
            if bias_type == 'occam_bias' and not self.causal_types_unlocked:
                # We're over-preferring simple hypotheses — unlock causal types
                self.allowed_hypothesis_types.add(HypothesisType.CAUSAL_CORRELATE)
                self.allowed_hypothesis_types.add(HypothesisType.CAUSAL_MECHANISM)
                self.causal_types_unlocked = True
                expansion_triggered = True
                expansion_types.append('causal')
            
            elif bias_type == 'exploration_bias' and not self.unconventional_types_unlocked:
                # We're stuck exploring the wrong space — unlock unconventional
                self.unconventional_types_unlocked = True
                expansion_triggered = True
                expansion_types.append('unconventional')
        
        # Also expand if space inadequacy detected
        if any(a['critique_type'] == 'space_inadequate' for a in meta2_state['hypothesis_audits']):
            if not self.causal_types_unlocked:
                self.allowed_hypothesis_types.add(HypothesisType.CAUSAL_CORRELATE)
                self.allowed_hypothesis_types.add(HypothesisType.CAUSAL_MECHANISM)
                self.causal_types_unlocked = True
                expansion_triggered = True
                expansion_types.append('causal_for_space')
        
        if expansion_triggered:
            return {
                'episode': episode,
                'expansion_types': expansion_types,
                'new_space_size': len(self.allowed_hypothesis_types),
                'triggered_by': 'meta2_critique'
            }
        
        return None
    
    def _generate_in_expanded_space(
        self,
        episode: int,
        current_hypotheses: List[Any],
        kl_gain: float,
        uncertainty: float,
        dissonance: float
    ) -> Optional[GeneratedHypothesis]:
        """
        Generate hypothesis within the (possibly expanded) hypothesis space.
        """
        # Update base generator with current monitoring
        monitor_result = self.base_generator.monitor_state(
            episode=episode,
            kl_gain=kl_gain,
            uncertainty=uncertainty,
            dissonance=dissonance,
            hypotheses=current_hypotheses
        )
        
        # Check if we should generate
        if not monitor_result['should_generate']:
            return None
        
        # OVERRIDE: If Meta² expanded the space, generate from new types first
        existing_types = {h.hypothesis_type for h in self.base_generator.hypotheses.values()}
        available_new_types = self.allowed_hypothesis_types - existing_types
        
        if available_new_types:
            # Prioritize unexplored types from expanded space
            selected_type = min(available_new_types, key=lambda t: t.value)
            
            # Create hypothesis of this type
            hypothesis = self._create_meta2_hypothesis(
                selected_type, episode, current_hypotheses
            )
            
            if hypothesis:
                # Register with base generator
                self.base_generator.hypotheses[hypothesis.id] = hypothesis
                self.base_generator.last_generation_episode = episode
                self.base_generator.generation_count += 1
                
                return hypothesis
        
        # Fall back to base generator
        return self.base_generator.generate_hypothesis(
            episode, current_hypotheses, monitor_result['triggers_detected']
        )
    
    def _create_meta2_hypothesis(
        self,
        htype: HypothesisType,
        episode: int,
        parents: List[Any]
    ) -> Optional[GeneratedHypothesis]:
        """Create hypothesis from Meta²-expanded space."""
        # Use base generator's creation method
        hypothesis = self.base_generator._create_hypothesis_model(htype, episode, parents)
        
        if hypothesis:
            # Mark as Meta²-generated
            hypothesis.generation_trigger = 'meta2_expansion'
        
        return hypothesis
    
    def record_prediction(self, predicted_boundary: float, actual_boundary: float):
        """Record prediction for failure tracking."""
        error = abs(predicted_boundary - actual_boundary)
        self.prediction_history.append({
            'predicted': predicted_boundary,
            'actual': actual_boundary,
            'error': error
        })
        
        # Update failure streak
        if error > 0.1:
            self.failure_streak += 1
        else:
            self.failure_streak = max(0, self.failure_streak - 1)
    
    def get_meta2_status(self) -> Dict[str, Any]:
        """Get full Meta² integration status."""
        return {
            'epistemic_crises': len(self.epistemic_crises),
            'expansion_events': len(self.expansion_events),
            'hypothesis_space_size': len(self.allowed_hypothesis_types),
            'causal_unlocked': self.causal_types_unlocked,
            'unconventional_unlocked': self.unconventional_types_unlocked,
            'recent_failure_rate': self._compute_failure_rate(window=20),
            'failure_streak': self.failure_streak,
            'expansion_history': self.expansion_events[-5:] if self.expansion_events else []
        }


# Convenience alias
Meta2Integration = Meta2IntegratedGenerator
