"""
RAVANA v2 — PHASE D: Intent Engine v0
Dynamic objectives that evolve based on outcomes.

PRINCIPLE: From "what works" to "what I want."
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from enum import Enum

from .strategy import ExplorationMode


class SystemObjective(Enum):
    """High-level objectives the system can pursue."""
    EXPLORE = "explore"           # Maximize learning through dissonance
    STABILIZE = "stabilize"       # Lock in gains, preserve state
    OPTIMIZE_IDENTITY = "optimize_identity"  # Strengthen self-concept
    MINIMIZE_CLAMPS = "minimize_clamps"      # Reduce constitutional violations


@dataclass
class ObjectiveState:
    """Current state of an objective."""
    weight: float = 1.0           # Current importance (0-2 range)
    activation: float = 0.0       # Current drive strength
    satisfaction_history: List[float] = field(default_factory=list)
    last_updated: int = 0
    
    def update_satisfaction(self, value: float, episode: int):
        """Record how well this objective was met."""
        self.satisfaction_history.append(value)
        if len(self.satisfaction_history) > 50:
            self.satisfaction_history.pop(0)
        self.last_updated = episode
    
    def get_trend(self) -> float:
        """Positive = improving, negative = degrading."""
        if len(self.satisfaction_history) < 10:
            return 0.0
        recent = np.mean(self.satisfaction_history[-5:])
        older = np.mean(self.satisfaction_history[-10:-5])
        return recent - older


@dataclass
class IntentConfig:
    """Configuration for intent formation."""
    # Weight evolution
    learning_rate: float = 0.02
    max_weight: float = 2.0
    min_weight: float = 0.3
    
    # Objective satisfaction thresholds
    explore_satisfaction_d: Tuple[float, float] = (0.3, 0.7)  # Sweet spot
    stabilize_satisfaction_variance: float = 0.02  # Low variance = stable
    identity_target: float = 0.7
    
    # Mode-objective alignment (fixed biases)
    mode_objective_alignment: Dict[ExplorationMode, Dict[SystemObjective, float]] = field(
        default_factory=lambda: {
            ExplorationMode.EXPLORE_AGGRESSIVE: {
                SystemObjective.EXPLORE: 1.0,
                SystemObjective.STABILIZE: -0.3,
                SystemObjective.OPTIMIZE_IDENTITY: 0.1,
                SystemObjective.MINIMIZE_CLAMPS: -0.2,
            },
            ExplorationMode.EXPLORE_SAFE: {
                SystemObjective.EXPLORE: 0.5,
                SystemObjective.STABILIZE: 0.2,
                SystemObjective.OPTIMIZE_IDENTITY: 0.3,
                SystemObjective.MINIMIZE_CLAMPS: 0.1,
            },
            ExplorationMode.STABILIZE: {
                SystemObjective.EXPLORE: -0.5,
                SystemObjective.STABILIZE: 1.0,
                SystemObjective.OPTIMIZE_IDENTITY: 0.4,
                SystemObjective.MINIMIZE_CLAMPS: 0.3,
            },
            ExplorationMode.RECOVER: {
                SystemObjective.EXPLORE: -0.8,
                SystemObjective.STABILIZE: 0.5,
                SystemObjective.OPTIMIZE_IDENTITY: -0.3,
                SystemObjective.MINIMIZE_CLAMPS: 1.0,
            },
        }
    )


class IntentEngine:
    """
    🎯 Intent Formation Layer
    
    Manages dynamic objectives that evolve based on outcomes.
    The system begins to behave "for a reason" — its own reason.
    """
    
    def __init__(self, config: Optional[IntentConfig] = None):
        self.config = config or IntentConfig()
        
        # Initialize objective states
        self.objectives: Dict[SystemObjective, ObjectiveState] = {
            obj: ObjectiveState(weight=1.0) for obj in SystemObjective
        }
        
        # Default: balanced priorities
        self.objectives[SystemObjective.MINIMIZE_CLAMPS].weight = 1.5  # Safety first
        self.objectives[SystemObjective.EXPLORE].weight = 1.0
        
        # History for introspection
        self.weight_history: List[Dict[str, float]] = []
        self.episode_count: int = 0
        
        # Current state tracking
        self.current_context: Optional[Dict[str, Any]] = None
        self.current_clamp_events: List[Dict] = []
    
    def update_state(self, context: Dict[str, Any], clamp_events: List[Dict]):
        """
        Store current state for intent computation.
        Called before mode selection to capture context.
        """
        self.current_context = context
        self.current_clamp_events = clamp_events
    
    def evaluate_outcomes(
        self,
        episode: int,
        pre_state: Dict[str, float],
        post_state: Dict[str, float],
        mode_used: ExplorationMode,
        clamp_count: int
    ):
        """
        Evaluate how well each objective was satisfied.
        Update objective weights based on trends.
        """
        self.episode_count = episode
        
        # Calculate satisfaction for each objective
        explore_sat = self._compute_explore_satisfaction(
            pre_state, post_state, mode_used
        )
        stabilize_sat = self._compute_stabilize_satisfaction(
            pre_state, post_state
        )
        identity_sat = self._compute_identity_satisfaction(
            pre_state, post_state
        )
        clamp_sat = self._compute_clamp_satisfaction(clamp_count)
        
        # Update satisfaction histories
        self.objectives[SystemObjective.EXPLORE].update_satisfaction(explore_sat, episode)
        self.objectives[SystemObjective.STABILIZE].update_satisfaction(stabilize_sat, episode)
        self.objectives[SystemObjective.OPTIMIZE_IDENTITY].update_satisfaction(identity_sat, episode)
        self.objectives[SystemObjective.MINIMIZE_CLAMPS].update_satisfaction(clamp_sat, episode)
        
        # 🧠 SELF-ADJUSTMENT: Evolve weights based on trends
        self._evolve_weights()
        
        # Record history
        self.weight_history.append({
            "episode": episode,
            **{obj.value: state.weight for obj, state in self.objectives.items()}
        })
    
    def compute_mode_bias(
        self,
        mode: ExplorationMode
    ) -> float:
        """
        Compute objective-driven bias for a mode.
        
        Returns: bias score (-1 to +1) based on how well this mode
        serves the system's current objectives.
        """
        alignment = self.config.mode_objective_alignment.get(mode, {})
        
        total_bias = 0.0
        total_weight = 0.0
        
        for objective, mode_alignment in alignment.items():
            obj_state = self.objectives[objective]
            # Weight by both objective importance AND alignment
            contribution = obj_state.weight * mode_alignment
            total_bias += contribution
            total_weight += abs(obj_state.weight)
        
        # Normalize
        if total_weight > 0:
            bias = total_bias / total_weight
        else:
            bias = 0.0
        
        return np.clip(bias, -1.0, 1.0)
    
    def get_current_intent(self) -> Dict[str, Any]:
        """
        Return current intent state for introspection.
        
        Shows: what the system currently 'wants' and how strongly.
        """
        # Find dominant objective
        dominant = max(self.objectives.items(), key=lambda x: x[1].weight)
        
        return {
            "dominant_objective": dominant[0].value,
            "dominant_weight": dominant[1].weight,
            "objective_weights": {obj.value: state.weight for obj, state in self.objectives.items()},
            "objective_trends": {obj.value: state.get_trend() for obj, state in self.objectives.items()},
            "episode": self.episode_count,
        }
    
    def _compute_explore_satisfaction(
        self,
        pre: Dict[str, float],
        post: Dict[str, float],
        mode: ExplorationMode
    ) -> float:
        """
        Exploration satisfied when:
        - Dissonance moves toward sweet spot (0.3-0.7)
        - If already in sweet spot, staying there is good
        """
        d_post = post.get('dissonance', 0.5)
        low, high = self.config.explore_satisfaction_d
        
        # Distance from sweet spot center
        center = (low + high) / 2
        distance = abs(d_post - center)
        
        # 1.0 = dead center, 0.0 = at edge
        satisfaction = 1.0 - (distance / (high - low))
        
        # Bonus for aggressive mode that succeeded
        if mode == ExplorationMode.EXPLORE_AGGRESSIVE and satisfaction > 0.7:
            satisfaction = min(1.0, satisfaction + 0.1)
        
        return np.clip(satisfaction, 0.0, 1.0)
    
    def _compute_stabilize_satisfaction(
        self,
        pre: Dict[str, float],
        post: Dict[str, float]
    ) -> float:
        """
        Stabilization satisfied when:
        - Low variance in recent dissonance
        - High identity maintained
        """
        d_pre = pre.get('dissonance', 0.5)
        d_post = post.get('dissonance', 0.5)
        
        # Small change = stable
        change = abs(d_post - d_pre)
        stability = 1.0 - min(1.0, change / 0.1)
        
        return stability
    
    def _compute_identity_satisfaction(
        self,
        pre: Dict[str, float],
        post: Dict[str, float]
    ) -> float:
        """
        Identity optimization satisfied when:
        - Identity strength increases or stays high
        """
        i_pre = pre.get('identity', 0.5)
        i_post = post.get('identity', 0.5)
        target = self.config.identity_target
        
        # Improvement toward target
        if i_post > i_pre:
            return 0.7 + 0.3 * (i_post / target)
        # Maintained high identity
        elif i_post > target * 0.8:
            return 0.8
        # Declining
        else:
            return max(0.0, i_post / target)
    
    def _compute_clamp_satisfaction(self, clamp_count: int) -> float:
        """
        Clamp minimization satisfied when:
        - Few or zero clamps
        """
        if clamp_count == 0:
            return 1.0
        elif clamp_count == 1:
            return 0.7
        elif clamp_count <= 3:
            return 0.4
        else:
            return 0.1
    
    def _evolve_weights(self):
        """
        🧠 CORE: Self-adjust objective weights based on trends.
        
        If an objective is consistently unsatisfied, increase its weight.
        If an objective is consistently satisfied, can reduce its weight.
        """
        for objective, state in self.objectives.items():
            trend = state.get_trend()
            
            # Declining satisfaction → increase weight (need it more)
            if trend < -0.1:
                state.weight = min(
                    self.config.max_weight,
                    state.weight + self.config.learning_rate
                )
            # Improving satisfaction → can decrease weight (need it less)
            elif trend > 0.1:
                state.weight = max(
                    self.config.min_weight,
                    state.weight - self.config.learning_rate * 0.5
                )
        
        # 🚨 DEVELOPMENTAL PHASES
        # Early learning: prioritize safety
        # Later: allow exploration to grow
        if self.episode_count > 5000:
            # Reduce clamp minimization weight if already doing well
            clamp_obj = self.objectives[SystemObjective.MINIMIZE_CLAMPS]
            if np.mean(clamp_obj.satisfaction_history[-20:]) > 0.8:
                clamp_obj.weight = max(0.8, clamp_obj.weight - 0.01)
                # Redistribute to exploration
                self.objectives[SystemObjective.EXPLORE].weight = min(
                    2.0, self.objectives[SystemObjective.EXPLORE].weight + 0.005
                )


class IntentAwareStrategy:
    """
    🎯 Bridge: Strategy + Intent
    
    Combines rule-based strategy, learned preferences, and intent-driven bias.
    """
    
    def __init__(
        self,
        strategy_layer,
        learning_layer,
        intent_engine: IntentEngine
    ):
        self.strategy = strategy_layer
        self.learning = learning_layer
        self.intent = intent_engine
    
    def select_mode(
        self,
        context: Dict[str, Any],
        clamp_events: List[Dict]
    ) -> Tuple[ExplorationMode, Dict[str, Any]]:
        """
        Select mode using intent-weighted strategy.
        
        Process:
        1. Get base strategy selection
        2. Adjust based on intent weights
        3. Return final mode with full context
        """
        # Update intent engine with current state
        self.intent.update_state(context, clamp_events)
        
        # Get base strategy selection (rule + learned scores)
        selection = self.strategy.select_mode(context)
        mode = selection.mode
        
        # Get intent bias for this mode
        intent_bias = self.intent.compute_mode_bias(mode)
        
        # Build info dict from selection
        info = {
            'score': selection.confidence,
            'reason': selection.reason,
            'context': selection.context,
        }
        
        # Combine: add intent as final modulation
        final_score = info.get('score', 0.5) + 0.2 * intent_bias
        
        # Record what we did
        info['intent_bias'] = intent_bias
        info['final_score'] = final_score
        info['dominant_objective'] = self.intent.get_current_intent()['dominant_objective']
        
        return mode, info
    
    def update_after_step(
        self,
        episode: int,
        pre_state: Dict[str, float],
        post_state: Dict[str, float],
        mode: ExplorationMode,
        clamp_events: List[Dict]
    ):
        """Update all layers: strategy, learning, AND intent."""
        # Update strategy analytics
        self.strategy.update_mode_analytics(mode, episode)
        
        # Update learning layer
        self.learning.record_mode_usage(mode, episode, pre_state, post_state, clamp_events)
        
        # 🎯 Update intent engine (the new part)
        self.intent.evaluate_outcomes(
            episode, pre_state, post_state, mode, len(clamp_events)
        )
    
    def get_full_status(self) -> Dict[str, Any]:
        """Complete system status including intent."""
        return {
            "strategy": self.strategy.get_mode_analytics(),
            "learning": self.learning.get_learning_status(),
            "intent": self.intent.get_current_intent(),
        }
