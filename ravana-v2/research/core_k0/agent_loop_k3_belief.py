"""
RAVANA K3 (Redefined) — Belief Layer

Not modes. Not switching. Just: state understanding.

K3 adds to K2:
    belief_good ∈ [0, 1]  — probability we're in GOOD regime
    
Update rule:
    belief_good = f(history of signals from exploration)
    
Decision:
    expected_value(action) = 
        belief_good * reward_good(action) +
        belief_bad * reward_bad(action)
        
K2 still handles action optimization.
K3 only supplies missing information (inferred regime).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k2 import K2_Agent, ExplorationMode
from research.experiments_k0.resource_env import AgentAction


@dataclass
class SignalHistory:
    """Track exploration signals for belief estimation."""
    signals: List[str] = field(default_factory=list)  # "positive" or "negative"
    episodes: List[int] = field(default_factory=list)  # when each signal was received
    
    def add_signal(self, signal: str, episode: int):
        """Record a new signal from exploration."""
        self.signals.append(signal)
        self.episodes.append(episode)
        # Keep only recent signals (sliding window)
        if len(self.signals) > 10:
            self.signals.pop(0)
            self.episodes.pop(0)
    
    def get_recent(self, window: int = 5) -> List[str]:
        """Get last N signals."""
        return self.signals[-window:]


class K3_Belief_Agent(K2_Agent):
    """
    K3 = K2 + Belief Layer
    
    Extends K2 with regime inference from exploration signals.
    No interference with K2's proven decision engine.
    """
    
    def __init__(self, signal_accuracy: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        
        # Belief state
        self.belief_good: float = 0.5  # Start neutral
        self.signal_accuracy = signal_accuracy
        
        # Track signals from exploration
        self.signal_history = SignalHistory()
        
        # Prior for Bayesian update (assume regimes are equally likely)
        self.prior_good = 0.5
        
    def _get_prefs_as_dict(self, context_key: str) -> Dict[str, float]:
        """Safely get preferences as dictionary, handling PolicyWeights or dict."""
        raw = self.context_weights.get(context_key, None)
        
        if raw is None:
            return {"explore": 0.5, "exploit": 0.5, "conserve": 0.5}
        
        if isinstance(raw, dict):
            return {
                "explore": raw.get("explore", 0.5),
                "exploit": raw.get("exploit", 0.5),
                "conserve": raw.get("conserve", 0.5)
            }
        
        # PolicyWeights object
        return {
            "explore": getattr(raw, "explore", 0.5),
            "exploit": getattr(raw, "exploit", 0.5),
            "conserve": getattr(raw, "conserve", 0.5)
        }
    
    def select_action(self, obs: Dict[str, float]) -> AgentAction:
        """
        K3: Belief-informed action selection (minimal interference with K2).
        
        Strategy:
        1. Compute K2's normal preferences
        2. Adjust preferences using belief (information layer)
        3. Let K2's proven decision logic run with adjusted prefs
        4. Only override in critical moments
        """
        self.episode += 1
        self.state.update_from_observation(obs, self.episode)
        
        E = self.state.energy_estimate
        U = self.state.uncertainty
        R = self.state.resource_estimate
        trend = self.state.get_energy_trend(5)
        
        # Update resource tracking
        if R > self.last_resource_estimate + 0.05:
            self.steps_without_resource_gain = 0
        else:
            self.steps_without_resource_gain += 1
        self.last_resource_estimate = R
        
        # Update exploration tracking
        self.steps_since_explore += 1
        
        # === K3: BELIEF-ADJUSTED DECISION ===
        
        # Get K2's context key
        context_key = self.state.get_context_key(E, U, trend)
        
        # Get K2's learned preferences for this context (safely as dict)
        base_prefs = self._get_prefs_as_dict(context_key)
        
        # Compute belief-based adjustment
        # GOOD regime: boost exploit, penalize conserve
        # BAD regime: penalize exploit, boost explore
        belief_bad = 1.0 - self.belief_good
        
        # Adjustment factors (subtle, not replacement)
        if self.belief_good > 0.6:
            # Likely GOOD: exploit is safer, conserve less needed
            adjusted_prefs = {
                "explore": base_prefs["explore"] * 0.9,  # Less need to explore
                "exploit": base_prefs["exploit"] * 1.2,  # Safer to exploit
                "conserve": base_prefs["conserve"] * 0.8  # Less need to conserve
            }
        elif self.belief_good < 0.4:
            # Likely BAD: exploit is dangerous, need info
            adjusted_prefs = {
                "explore": base_prefs["explore"] * 1.3,  # More need to explore
                "exploit": base_prefs["exploit"] * 0.5,  # Dangerous to exploit
                "conserve": base_prefs["conserve"] * 1.0  # Neutral
            }
        else:
            # Uncertain: trust K2's learned preferences
            adjusted_prefs = base_prefs
        
        # Normalize to prevent extreme values
        for k in adjusted_prefs:
            adjusted_prefs[k] = np.clip(adjusted_prefs[k], 0.1, 0.9)
        
        # Temporarily override context_weights for this decision
        original_weights = self.context_weights.get(context_key, {})
        self.context_weights[context_key] = adjusted_prefs
        
        # === CRITICAL MOMENT OVERRIDES ===
        
        # Near death + confident about BAD regime → never exploit
        if E < self.energy_critical and self.belief_good < 0.3:
            self.context_weights[context_key] = original_weights  # Restore
            return AgentAction.EXPLORE  # Must get info
        
        # Starving + uncertain → gather info
        if self.steps_without_resource_gain > 15 and abs(self.belief_good - 0.5) < 0.2:
            mode = self._get_exploration_mode()
            if mode != ExplorationMode.DISABLED:
                self.steps_since_explore = 0
                self.context_weights[context_key] = original_weights  # Restore
                return AgentAction.EXPLORE
        
        # === NORMAL POLICY: K2 WITH ADJUSTED PREFERENCES ===
        
        # Use K2's proven decision logic with adjusted preferences
        if U > self.uncertainty_high and E > self.energy_low:
            action = self._get_action_by_expected_utility(context_key)
        elif E < self.energy_low:
            action = AgentAction.CONSERVE
        else:
            action = self._get_action_by_expected_utility(context_key)
        
        # Restore original weights after decision
        if original_weights:
            self.context_weights[context_key] = original_weights
        
        # Track if this was exploration
        if action == AgentAction.EXPLORE:
            self.steps_since_explore = 0
        
        return action
    
    def step(self, env) -> Dict[str, Any]:
        """Execute step with belief updating from exploration signals."""
        # Capture state before action
        energy_before = env.true_energy
        
        # Select and execute action
        obs = env._generate_observation()
        action = self.select_action(obs)
        result = env.execute_action(action)
        
        # Track if this was exploration
        if action == AgentAction.EXPLORE:
            self.steps_since_explore = 0
        
        # === K3: UPDATE BELIEF FROM SIGNAL ===
        
        signal = result.get('signal')
        if signal:
            # Received a signal from exploration
            self.signal_history.add_signal(signal, self.episode)
            self._update_belief()
        
        # Learn from outcome (K2's learning)
        energy_after = env.true_energy
        outcome = self._record_outcome(env, action, result, energy_before, energy_after)
        self._learn_from_outcome(outcome)
        
        # Track survival
        if result['alive']:
            self.survival_count += 1
        else:
            self.death_count += 1
        
        self.cumulative_reward += result['utility']
        
        return {
            'alive': result['alive'],
            'observation': obs,
            'action': action,
            'episode': self.episode,
            'belief_good': self.belief_good,
            'signal': signal
        }
    
    def _update_belief(self):
        """
        Update belief_good from signal history.
        
        Simple Bayesian approach (or just weighted counting).
        """
        recent_signals = self.signal_history.get_recent(5)
        
        if not recent_signals:
            return  # No data, keep prior
        
        # Count signal types
        positive_count = sum(1 for s in recent_signals if s == "positive")
        negative_count = sum(1 for s in recent_signals if s == "negative")
        total = len(recent_signals)
        
        if total == 0:
            return
        
        # Likelihood calculation
        p_signals_given_good = (
            (self.signal_accuracy ** positive_count) *
            ((1 - self.signal_accuracy) ** negative_count)
        )
        
        p_signals_given_bad = (
            ((1 - self.signal_accuracy) ** positive_count) *
            (self.signal_accuracy ** negative_count)
        )
        
        prior_good = self.prior_good
        prior_bad = 1.0 - prior_good
        
        unnorm_good = p_signals_given_good * prior_good
        unnorm_bad = p_signals_given_bad * prior_bad
        
        total_prob = unnorm_good + unnorm_bad
        if total_prob > 0:
            self.belief_good = unnorm_good / total_prob
        
        self.belief_good = np.clip(self.belief_good, 0.1, 0.9)
    
    def _record_outcome(self, env, action, result, energy_before, energy_after) -> Any:
        """Create outcome record for learning (compatible with K2)."""
        from dataclasses import dataclass
        from typing import Dict
        
        @dataclass
        class SimpleOutcome:
            episode: int
            context: Dict[str, float]
            action: Any
            delta_energy: float
            survived: bool
            exploration_success: bool
            
        # Get current context from state
        trend = self.state.get_energy_trend(5) if self.state.energy_history else 0.0
        context = {
            "energy": self.state.energy_estimate,
            "uncertainty": self.state.uncertainty,
            "trend": trend,
            "regime": getattr(env, 'current_regime', 'unknown')
        }
            
        return SimpleOutcome(
            episode=self.episode,
            context=context,
            action=action,
            delta_energy=energy_after - energy_before,
            survived=result['alive'],
            exploration_success=energy_after > energy_before if action.value == 'explore' else False
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Extended status with belief information."""
        base_status = super().get_status()
        base_status.update({
            "belief_good": self.belief_good,
            "belief_bad": 1.0 - self.belief_good,
            "n_signals": len(self.signal_history.signals),
            "recent_signals": self.signal_history.get_recent(3),
            "k3_type": "belief_layer"
        })
        return base_status
