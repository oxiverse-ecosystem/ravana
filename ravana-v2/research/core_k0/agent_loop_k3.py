"""
RAVANA K3 — Trajectory-Aware Strategy Adaptation
"Where is this heading?"

K3 is a thin layer on top of K2:
- K2 decides: "What works here?"
- K3 decides: "Where should I steer?"

Core principle: Anticipate trajectories, not just react to states.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set
from enum import Enum, auto
import numpy as np
import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research.experiments_k0.resource_env import AgentAction
from research.core_k0.agent_loop_k2 import K2_Agent, AgentState, ActionOutcome, ExplorationMode


class StrategyMode(Enum):
    """
    3 strategy modes (minimal, interpretable).
    
    RECOVERY: Prioritize stability, avoid risk
    EXPLOIT: Maximize gains when safe  
    PROBE: Test unknowns when uncertain but not critical
    """
    RECOVERY = "recovery"
    EXPLOIT = "exploit"
    PROBE = "probe"


@dataclass
class ContextTransition:
    """
    Log of what happened when taking action A in context C.
    
    This is the core memory structure for trajectory learning.
    """
    from_context: str  # C_t: "low_high_falling_3+"
    action: AgentAction  # A_t
    to_context: str  # C_t+1
    delta_energy: float  # ΔE (outcome)
    survived: bool
    mode_active: StrategyMode  # Which mode was active
    episode: int


@dataclass
class TrajectoryPattern:
    """
    Learned pattern: (C_t, A_t) → likely outcomes.
    
    Aggregates multiple transitions to predict where actions lead.
    """
    trigger_context: str  # C_t
    trigger_action: AgentAction  # A_t (optional, can be None for any action)
    
    # Outcome distribution (learned from transitions)
    next_context_counts: Dict[str, int] = field(default_factory=dict)
    avg_delta_energy: float = 0.0
    survival_rate: float = 0.5
    transition_count: int = 0
    
    # Pattern metadata
    first_seen: int = 0
    last_seen: int = 0
    
    def update(self, transition: ContextTransition):
        """Incorporate new transition into pattern statistics."""
        self.next_context_counts[transition.to_context] = \
            self.next_context_counts.get(transition.to_context, 0) + 1
        
        # Running average of energy change
        n = self.transition_count
        self.avg_delta_energy = (self.avg_delta_energy * n + transition.delta_energy) / (n + 1)
        
        # Survival rate
        self.survival_rate = (self.survival_rate * n + (1.0 if transition.survived else 0.0)) / (n + 1)
        
        self.transition_count += 1
        self.last_seen = transition.episode
    
    def predict_next_context(self) -> Optional[str]:
        """Most likely next context based on history."""
        if not self.next_context_counts:
            return None
        return max(self.next_context_counts, key=self.next_context_counts.get)
    
    def is_collapse_trajectory(self, threshold: float = -0.1) -> bool:
        """Does this pattern usually lead to energy loss?"""
        return self.avg_delta_energy < threshold and self.survival_rate < 0.8
    
    def is_recovery_trajectory(self, threshold: float = 0.05) -> bool:
        """Does this pattern usually lead to energy gain?"""
        return self.avg_delta_energy > threshold and self.survival_rate > 0.9


class K3_Agent:
    """
    K3: Trajectory-Aware Strategy Adaptation.
    
    K3 sits on top of K2:
    - K3 selects STRATEGY MODE based on trajectory patterns
    - K2 selects ACTIONS within that mode
    
    This preserves K2's proven decision engine while adding
    anticipatory intelligence.
    """
    
    def __init__(self, learning_rate: float = 0.1):
        # K2 engine (proven, robust)
        self.k2 = K2_Agent(learning_rate=learning_rate, survival_boost=2.0)
        
        # K3: Transition memory
        self.transition_history: List[ContextTransition] = []
        self.patterns: Dict[Tuple[str, Optional[AgentAction]], TrajectoryPattern] = {}
        
        # Current mode
        self.current_mode: StrategyMode = StrategyMode.RECOVERY
        self.mode_history: List[Tuple[int, StrategyMode, str]] = []  # (episode, mode, reason)
        
        # Pattern recognition thresholds
        self.collapse_pattern_threshold: int = 5  # INCREASED: Require more evidence
        self.recovery_pattern_threshold: int = 5  # INCREASED: Require more evidence
        self.mode_switch_cooldown: int = 15  # INCREASED: Longer phases
        self.episodes_since_switch: int = 0
        
        # Tracking
        self.episode: int = 0
        self.prev_context: Optional[str] = None
        self.prev_action: Optional[AgentAction] = None
    
    def _get_current_context(self) -> str:
        """Get current context key from K2's state."""
        E = self.k2.state.energy_estimate
        U = self.k2.state.uncertainty
        trend = self.k2.state.get_energy_trend(5)
        failures = self.k2.consecutive_exploration_failures
        return self.k2.state.get_context_key(E, U, trend, failures)
    
    def _log_transition(self, from_context: str, action: AgentAction, 
                      delta_energy: float, survived: bool):
        """Record that taking action A in context C led to outcome O."""
        to_context = self._get_current_context()
        
        transition = ContextTransition(
            from_context=from_context,
            action=action,
            to_context=to_context,
            delta_energy=delta_energy,
            survived=survived,
            mode_active=self.current_mode,
            episode=self.episode
        )
        
        self.transition_history.append(transition)
        
        # Keep last 200 transitions
        if len(self.transition_history) > 200:
            self.transition_history = self.transition_history[-200:]
        
        # Update pattern for (context, action)
        pattern_key = (from_context, action)
        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = TrajectoryPattern(
                trigger_context=from_context,
                trigger_action=action,
                first_seen=self.episode
            )
        
        self.patterns[pattern_key].update(transition)
    
    def _check_collapse_trajectory(self, context: str) -> bool:
        """
        Check if current context matches known collapse patterns.
        
        Looks at:
        - (context, any_action) patterns
        - Falling energy trend
        - Recent survival rate
        """
        # Check patterns for this context with any action
        collapse_signals = 0
        
        for action in [AgentAction.EXPLORE, AgentAction.EXPLOIT, AgentAction.CONSERVE]:
            pattern_key = (context, action)
            if pattern_key in self.patterns:
                pattern = self.patterns[pattern_key]
                if pattern.transition_count >= self.collapse_pattern_threshold:
                    if pattern.is_collapse_trajectory():
                        collapse_signals += 1
        
        # Also check trend-based signal
        trend = self.k2.state.get_energy_trend(5)
        E = self.k2.state.energy_estimate
        
        # Critical: falling energy + already low = likely collapse
        if trend < -0.03 and E < 0.3:
            collapse_signals += 2
        
        return collapse_signals >= 2  # Multiple signals = pattern match
    
    def _check_recovery_opportunity(self, context: str) -> bool:
        """
        Check if current context matches known recovery patterns.
        
        Conditions:
        - Stable or rising energy trend
        - High uncertainty (opportunity for learning)
        - Not in critical zone
        """
        E = self.k2.state.energy_estimate
        trend = self.k2.state.get_energy_trend(5)
        U = self.k2.state.uncertainty
        
        # Basic conditions
        if E < 0.25:  # Too critical to exploit
            return False
        if trend < 0:  # Falling energy
            return False
        
        # Check for positive patterns
        positive_signals = 0
        
        for action in [AgentAction.EXPLORE, AgentAction.EXPLOIT, AgentAction.CONSERVE]:
            pattern_key = (context, action)
            if pattern_key in self.patterns:
                pattern = self.patterns[pattern_key]
                if pattern.transition_count >= self.recovery_pattern_threshold:
                    if pattern.is_recovery_trajectory():
                        positive_signals += 1
        
        # High uncertainty with positive trend = probing opportunity
        if U > 0.4 and trend > 0.01:
            positive_signals += 1
        
        return positive_signals >= 1
    
    def _select_mode(self) -> StrategyMode:
        """
        K3's core intelligence: select strategy mode based on patterns.
        
        Modes:
        - RECOVERY: When collapse trajectory detected or critical energy
        - EXPLOIT: When recovery patterns detected and energy stable
        - PROBE: When uncertainty high but not critical
        """
        self.episodes_since_switch += 1
        
        # Cooldown: don't switch modes too frequently
        if self.episodes_since_switch < self.mode_switch_cooldown:
            return self.current_mode
        
        context = self._get_current_context()
        E = self.k2.state.energy_estimate
        mode_before = self.current_mode
        reason = ""
        
        # Priority 1: CRITICAL → RECOVERY
        if E < 0.2 or self._check_collapse_trajectory(context):
            self.current_mode = StrategyMode.RECOVERY
            reason = f"collapse_trajectory (E={E:.2f})" if E >= 0.2 else f"critical_energy (E={E:.2f})"
        
        # Priority 2: STABLE + PATTERNS → EXPLOIT
        elif self._check_recovery_opportunity(context):
            self.current_mode = StrategyMode.EXPLOIT
            reason = "recovery_opportunity"
        
        # Priority 3: UNCERTAIN + SAFE → PROBE
        elif self.k2.state.uncertainty > 0.35 and E > 0.3:
            self.current_mode = StrategyMode.PROBE
            reason = "uncertainty_probe"
        
        # Default: maintain current or RECOVERY if unsure
        else:
            if self.current_mode != StrategyMode.RECOVERY and E < 0.35:
                self.current_mode = StrategyMode.RECOVERY
                reason = "fallback_recovery"
            else:
                reason = "maintain"
        
        # Log mode change
        if self.current_mode != mode_before:
            self.mode_history.append((self.episode, self.current_mode, reason))
            self.episodes_since_switch = 0
        
        return self.current_mode
    
    def _apply_mode_biases(self, mode: StrategyMode) -> Dict[str, Any]:
        """
        Apply mode-specific biases to K2's decision parameters.
        
        This is how K3 steers K2 without replacing it.
        """
        biases = {
            "exploration_boost": 0.0,
            "risk_tolerance": 0.5,  # 0=conservative, 1=aggressive
            "learning_rate_multiplier": 1.0,
        }
        
        if mode == StrategyMode.RECOVERY:
            # Prioritize safety, reduce exploration
            biases["exploration_boost"] = -0.2  # Reduce exploration probability
            biases["risk_tolerance"] = 0.2  # Conservative
            biases["learning_rate_multiplier"] = 1.5  # Learn fast from survival signals
            
        elif mode == StrategyMode.EXPLOIT:
            # Prioritize gains, increase exploitation
            biases["exploration_boost"] = 0.1  # Slight boost if profitable
            biases["risk_tolerance"] = 0.7  # Moderate risk acceptable
            biases["learning_rate_multiplier"] = 0.8  # Stable learning
            
        elif mode == StrategyMode.PROBE:
            # Prioritize information gain
            biases["exploration_boost"] = 0.3  # Boost exploration
            biases["risk_tolerance"] = 0.4  # Controlled risk
            biases["learning_rate_multiplier"] = 2.0  # Learn fast from probes
        
        return biases
    
    def select_action(self, obs: Dict[str, float]) -> AgentAction:
        """
        K3 action selection: Mode selection → K2 action selection with biases.
        """
        self.episode += 1
        
        # K3: Select strategy mode
        mode = self._select_mode()
        biases = self._apply_mode_biases(mode)
        
        # Store context before K2 processes observation
        context_before = self._get_current_context()
        
        # Apply biases to K2 (steering without replacement)
        original_lr = self.k2.learning_rate
        self.k2.learning_rate = original_lr * biases["learning_rate_multiplier"]
        
        # Get K2's action (proven decision engine)
        action = self.k2.select_action(obs)
        
        # K3 can override in critical cases
        E = self.k2.state.energy_estimate
        if mode == StrategyMode.RECOVERY and E < 0.15:
            # Force conservative action regardless of K2's choice
            action = AgentAction.CONSERVE
        
        # Restore K2's learning rate
        self.k2.learning_rate = original_lr
        
        return action, mode, context_before, biases
    
    def step(self, env) -> Dict[str, Any]:
        """Execute one step with K3 trajectory awareness."""
        # Capture state before action
        energy_before = env.true_energy
        
        # K3 selects action (with mode awareness)
        obs = env._generate_observation()
        action, mode, context_before, biases = self.select_action(obs)
        
        # Execute
        result = env.execute_action(action)
        
        # Calculate outcomes
        energy_after = env.true_energy
        delta_energy = energy_after - energy_before
        survived = result["alive"]
        
        # Log transition for pattern learning
        if self.prev_context is not None and self.prev_action is not None:
            self._log_transition(
                from_context=self.prev_context,
                action=self.prev_action,
                delta_energy=delta_energy,
                survived=survived
            )
        
        # Update previous state for next transition
        self.prev_context = context_before
        self.prev_action = action
        
        # Update K2's internal state (for compatibility)
        self.k2.episode = self.episode
        self.k2.cumulative_reward += result["utility"]
        self.k2.state.action_history.append((self.episode, action, result["utility"]))
        
        if survived:
            self.k2.survival_count += 1
        else:
            self.k2.death_count += 1
        
        return {
            "alive": survived,
            "observation": obs,
            "action": action,
            "episode": self.episode,
            "mode": mode.value,
            "mode_reason": self.mode_history[-1][2] if self.mode_history else "initial",
            "energy_delta": delta_energy,
            "pattern_count": len(self.patterns),
            "transition_count": len(self.transition_history),
            "biases_applied": biases
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Report including K3 trajectory awareness."""
        k2_status = self.k2.get_status()
        
        # Calculate K3-specific metrics
        mode_switches = len(self.mode_history)
        mode_durations = []
        if self.mode_history:
            for i in range(len(self.mode_history) - 1):
                duration = self.mode_history[i+1][0] - self.mode_history[i][0]
                mode_durations.append((self.mode_history[i][1].value, duration))
            # Last mode duration
            mode_durations.append((self.mode_history[-1][1].value, self.episode - self.mode_history[-1][0]))
        
        # Pattern statistics
        collapse_patterns = sum(1 for p in self.patterns.values() if p.is_collapse_trajectory())
        recovery_patterns = sum(1 for p in self.patterns.values() if p.is_recovery_trajectory())
        
        return {
            **k2_status,
            "k3_mode": self.current_mode.value,
            "mode_switches": mode_switches,
            "mode_durations": mode_durations,
            "patterns_learned": len(self.patterns),
            "collapse_patterns": collapse_patterns,
            "recovery_patterns": recovery_patterns,
            "transitions_logged": len(self.transition_history),
            "trajectory_awareness": True
        }
