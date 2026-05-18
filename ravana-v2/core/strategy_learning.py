"""
RAVANA v2 — PHASE C: Strategy Learning v0
Self-evaluating mode effectiveness: rules → learned preferences.

PRINCIPLE: Don't just select modes. Learn which modes work when.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from .strategy import ExplorationMode, BehavioralContext


@dataclass
class ModeOutcome:
    """Result of being in a mode for some duration."""
    mode: ExplorationMode
    episode_start: int
    episode_end: int
    duration: int
    
    # Pre-mode state
    start_clamp_rate: float
    start_dissonance: float
    start_identity: float
    
    # Post-mode outcomes
    end_clamp_rate: float
    end_dissonance: float
    end_identity: float
    
    # Derived score
    score: float = 0.0


@dataclass
class LearningConfig:
    """Configuration for strategy learning."""
    learning_rate: float = 0.05
    outcome_window: int = 5  # Episodes to evaluate mode effect
    
    # Score weights
    clamp_improvement_weight: float = 2.0  # Reducing clamps is good
    dissonance_quality_weight: float = 1.0  # Healthy D range is good
    identity_preservation_weight: float = 1.5  # Preserving identity is good
    
    # Exploration bonus (to prevent mode collapse)
    exploration_bonus: float = 0.1
    
    # Softmax temperature for selection
    temperature: float = 0.5


class StrategyLearningLayer:
    """
    Minimal strategy learning: accumulate mode effectiveness.
    
    Adds to rule-based selection:
    final_score = rule_score + learned_score * learned_weight
    """
    
    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        
        # Learned effectiveness scores for each mode
        self.mode_scores: Dict[ExplorationMode, float] = {
            mode: 0.0 for mode in ExplorationMode
        }
        
        # Experience counters
        self.mode_experiences: Dict[ExplorationMode, int] = {
            mode: 0 for mode in ExplorationMode
        }
        
        # Recent outcomes for analysis
        self.outcome_history: deque = deque(maxlen=100)
        
        # Current mode tracking (for outcome calculation)
        self._current_mode_episode: Optional[int] = None
        self._current_mode_start_state: Optional[Dict] = None
        
    def start_mode_tracking(self, mode: ExplorationMode, episode: int, 
                           context: BehavioralContext):
        """Record that we entered a mode at this episode."""
        self._current_mode_episode = episode
        self._current_mode_start_state = {
            'clamp_rate': context.clamp_rate,
            'dissonance': context.dissonance,
            'identity': context.identity
        }
    
    def end_mode_tracking(self, mode: ExplorationMode, episode: int,
                         context: BehavioralContext) -> Optional[ModeOutcome]:
        """
        Calculate outcome of mode that just ended.
        Returns ModeOutcome if tracking was active.
        """
        if self._current_mode_episode is None or self._current_mode_start_state is None:
            return None
        
        duration = episode - self._current_mode_episode
        if duration < 1:
            return None  # Too short to evaluate
        
        outcome = ModeOutcome(
            mode=mode,
            episode_start=self._current_mode_episode,
            episode_end=episode,
            duration=duration,
            start_clamp_rate=self._current_mode_start_state['clamp_rate'],
            start_dissonance=self._current_mode_start_state['dissonance'],
            start_identity=self._current_mode_start_state['identity'],
            end_clamp_rate=context.clamp_rate,
            end_dissonance=context.dissonance,
            end_identity=context.identity
        )
        
        # Calculate score
        outcome.score = self._compute_outcome_score(outcome)
        
        # Store and learn
        self.outcome_history.append(outcome)
        self._learn_from_outcome(outcome)
        
        # Reset tracking
        self._current_mode_episode = None
        self._current_mode_start_state = None
        
        return outcome
    
    def record_mode_usage(
        self,
        mode: ExplorationMode,
        episode: int,
        pre_state: Dict[str, float],
        post_state: Dict[str, float],
        clamp_events: List[Dict]
    ):
        """
        Simplified interface for recording mode usage.
        Wraps start_mode_tracking and end_mode_tracking.
        """
        from .strategy import BehavioralContext
        
        # Create context from pre_state
        pre_context = BehavioralContext(
            clamp_rate=len(clamp_events) / max(1, episode) if episode > 0 else 0.0,
            dissonance=pre_state.get('dissonance', 0.5),
            identity=pre_state.get('identity', 0.5),
            dissonance_trend=0.0,
            identity_drift=0.0,
            stability=0.1,
            dissonance_variance=0.1
        )
        
        # Start tracking
        self.start_mode_tracking(mode, episode, pre_context)
        
        # Create context from post_state
        post_context = BehavioralContext(
            clamp_rate=len(clamp_events) / max(1, episode + 1),
            dissonance=post_state.get('dissonance', 0.5),
            identity=post_state.get('identity', 0.5),
            dissonance_trend=post_state.get('dissonance', 0.5) - pre_state.get('dissonance', 0.5),
            identity_drift=post_state.get('identity', 0.5) - pre_state.get('identity', 0.5),
            stability=0.1,
            dissonance_variance=0.1
        )
        
        # End tracking and get outcome
        outcome = self.end_mode_tracking(mode, episode + 1, post_context)
        
        return outcome
    
    def _compute_outcome_score(self, outcome: ModeOutcome) -> float:
        """
        Score mode effectiveness.
        
        Higher = better mode performance
        """
        # Clamp improvement (reducing = good)
        clamp_delta = outcome.end_clamp_rate - outcome.start_clamp_rate
        clamp_score = -clamp_delta * self.config.clamp_improvement_weight
        
        # Dissonance quality (healthy range = good)
        # Ideal: 0.3-0.7, penalize extremes
        d_quality_start = self._dissonance_quality(outcome.start_dissonance)
        d_quality_end = self._dissonance_quality(outcome.end_dissonance)
        d_quality_score = (d_quality_end - d_quality_start) * self.config.dissonance_quality_weight
        
        # Identity preservation (maintaining = good)
        identity_score = (
            (outcome.end_identity - outcome.start_identity) * 
            self.config.identity_preservation_weight
        )
        
        total = clamp_score + d_quality_score + identity_score
        
        # Exploration bonus (to prevent mode collapse)
        total += self.config.exploration_bonus
        
        return total
    
    def _dissonance_quality(self, d: float) -> float:
        """
        Score dissonance quality: 1.0 at 0.5, falls off toward extremes.
        """
        # Gaussian-ish: peak at 0.5, width 0.3
        return np.exp(-((d - 0.5) ** 2) / 0.09)
    
    def _learn_from_outcome(self, outcome: ModeOutcome):
        """
        Update mode effectiveness score based on outcome.
        """
        mode = outcome.mode
        old_score = self.mode_scores[mode]
        
        # Moving average update
        n = self.mode_experiences[mode] + 1
        new_score = (old_score * (n - 1) + outcome.score) / n
        
        self.mode_scores[mode] = new_score
        self.mode_experiences[mode] = n
    
    def get_mode_weights(self, context: BehavioralContext) -> Dict[ExplorationMode, float]:
        """
        Get learned weights for each mode.
        
        Returns softmax-normalized weights that sum to 1.
        """
        # Get raw scores
        raw_scores = {mode: self.mode_scores[mode] for mode in ExplorationMode}
        
        # Add exploration bonus based on recency (prevent collapse)
        for mode in ExplorationMode:
            experiences = self.mode_experiences[mode]
            if experiences < 5:
                # Boost underexplored modes
                raw_scores[mode] += 0.5 * (5 - experiences) / 5
        
        # Softmax normalization
        scores_array = np.array(list(raw_scores.values()))
        
        # Subtract max for numerical stability
        scores_array -= np.max(scores_array)
        
        exp_scores = np.exp(scores_array / self.config.temperature)
        weights = exp_scores / np.sum(exp_scores)
        
        return {mode: weights[i] for i, mode in enumerate(ExplorationMode)}
    
    def combine_with_rule_scores(
        self,
        rule_scores: Dict[ExplorationMode, float],
        context: BehavioralContext,
        learned_weight: float = 0.3
    ) -> Dict[ExplorationMode, float]:
        """
        Combine rule-based scores with learned scores.
        
        final_score = rule_score + learned_weight * learned_score
        """
        learned_weights = self.get_mode_weights(context)
        
        combined = {}
        for mode in ExplorationMode:
            rule_component = rule_scores[mode]
            learned_component = learned_weights[mode] * learned_weight
            combined[mode] = rule_component + learned_component
        
        return combined
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Return learning status summary."""
        return {
            'mode_scores': {m.value: round(s, 3) for m, s in self.mode_scores.items()},
            'mode_experiences': {m.value: n for m, n in self.mode_experiences.items()},
            'outcomes_recorded': len(self.outcome_history),
            'recent_outcomes': [
                {
                    'mode': o.mode.value,
                    'score': round(o.score, 3),
                    'duration': o.duration,
                    'clamp_improvement': round(o.start_clamp_rate - o.end_clamp_rate, 3)
                }
                for o in list(self.outcome_history)[-5:]
            ]
        }


class StrategyWithLearning:
    """
    Combined strategy + learning layer.
    
    Use this instead of standalone StrategyLayer for Phase C.
    """
    
    def __init__(
        self,
        strategy_layer,
        learning_layer: Optional[StrategyLearningLayer] = None
    ):
        self.strategy = strategy_layer
        self.learning = learning_layer or StrategyLearningLayer()
        
        self.learned_weight = 0.3  # How much to trust learned scores vs rules
    
    def select_mode(self, context: BehavioralContext, episode: int) -> Any:
        """
        Select mode using both rules and learned preferences.
        """
        # Get rule-based scores
        d_trend = self.strategy._compute_d_trend()
        rule_scores = self.strategy._evaluate_mode_scores(context, d_trend)
        
        # Combine with learned scores
        combined_scores = self.learning.combine_with_rule_scores(
            rule_scores, context, self.learned_weight
        )
        
        # Select best mode
        best_mode = max(combined_scores, key=combined_scores.get)
        
        # Handle mode transition
        if best_mode != self.strategy.current_mode:
            # End tracking for old mode
            self.learning.end_mode_tracking(
                self.strategy.current_mode, episode, context
            )
            # Start tracking for new mode
            self.learning.start_mode_tracking(best_mode, episode, context)
            
            # Update strategy layer
            self.strategy.current_mode = best_mode
            self.strategy.mode_switches += 1
        
        return best_mode
    
    def record_mode_usage(
        self,
        mode: ExplorationMode,
        episode: int,
        pre_state: Dict[str, Any],
        post_state: Dict[str, Any],
        clamp_events: List[Dict]
    ):
        """Delegate to learning layer to record mode usage and outcome."""
        # Convert pre_state/post_state to BehavioralContext-like format
        # for the learning layer
        from core.strategy import BehavioralContext
        
        context = BehavioralContext(
            dissonance=post_state.get('dissonance', 0.5),
            identity=post_state.get('identity', 0.5),
            clamp_rate=len(clamp_events) / max(1, episode) if episode > 0 else 0.0,
            dissonance_trend=post_state.get('dissonance', 0.5) - pre_state.get('dissonance', 0.5),
            identity_drift=0.0,
            stability=0.5,
            dissonance_variance=0.0,
            recent_resolution_success=0.5
        )
        
        self.learning.record_mode_usage(mode, episode, pre_state, post_state, clamp_events)
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Return learning status summary."""
        return self.learning.get_learning_status()
