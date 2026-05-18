"""
RAVANA K1.3 — Context-Aware Exploration (Judgment, Not Panic)
Fixes the EP24 deterministic death from K1.2's exploration_floor.

Key insight: Exploration is a COSTLY gamble. Only take it when:
1. You can AFFORD the cost (energy > survival buffer)
2. The TREND supports it (not bleeding energy)
3. You have MARGIN for failure (not at the edge)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np

# Import AgentAction from environment (don't define our own)
import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research.experiments_k0.resource_env import AgentAction


class ExplorationMode(Enum):
    """K1.3: Context-aware exploration states."""
    DISABLED = "disabled"      # Energy bleeding — conserve only
    GUARDED = "guarded"          # Stable margin — limited exploration
    ENABLED = "enabled"          # Healthy — full exploration allowed


@dataclass
class AgentState:
    energy_estimate: float = 0.5
    resource_estimate: float = 0.5
    risk_estimate: float = 0.3
    uncertainty: float = 0.3
    action_history: List[Tuple[int, AgentAction, float]] = field(default_factory=list)
    energy_history: List[float] = field(default_factory=list)
    exploration_success_history: List[bool] = field(default_factory=list)
    
    def update_from_observation(self, obs: Dict[str, float], episode: int):
        self.energy_estimate = obs.get("energy_obs", self.energy_estimate)
        self.resource_estimate = obs.get("resource_obs", self.resource_estimate)
        noise = obs.get("noise", 0.0)
        self.risk_estimate = 0.2 + noise * 0.5
        self.uncertainty = obs.get("observation_quality", 0.3)
        self.energy_history.append(self.energy_estimate)
        if len(self.energy_history) > 20:
            self.energy_history = self.energy_history[-20:]
    
    def get_energy_trend(self, window: int = 5) -> float:
        """Calculate energy trend over recent window. Positive = gaining, negative = bleeding."""
        if len(self.energy_history) < window:
            return 0.0
        recent = self.energy_history[-window:]
        # Linear regression slope approximation
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0] if len(set(recent)) > 1 else 0.0
        return float(slope)
    
    def record_action(self, episode: int, action: AgentAction, utility: float, success: bool = True):
        self.action_history.append((episode, action, utility))
        if action == AgentAction.EXPLORE:
            self.exploration_success_history.append(success)
        if len(self.action_history) > 50:
            self.action_history = self.action_history[-50:]
        if len(self.exploration_success_history) > 20:
            self.exploration_success_history = self.exploration_success_history[-20:]
    
    def get_exploration_success_rate(self) -> float:
        """Track if exploration is actually working."""
        if not self.exploration_success_history:
            return 0.5
        return sum(self.exploration_success_history) / len(self.exploration_success_history)


class K1_3_Agent:
    """
    K1.3: The birth of judgment.
    
    K1.2 taught: "Don't sit still and die"
    K1.3 teaches: "Don't jump off a cliff just because you're scared"
    """
    
    def __init__(self):
        self.state = AgentState()
        self.episode: int = 0
        self.survival_count: int = 0
        self.death_count: int = 0
        self.cumulative_reward: float = 0.0
        
        # K1.3: Survival thresholds (tighter than K1.2)
        self.energy_critical: float = 0.15
        self.energy_low: float = 0.35
        self.uncertainty_high: float = 0.4
        
        # K1.3: Exploration abort tracking
        self.consecutive_exploration_failures: int = 0
        self.max_consecutive_failures: int = 3
        
        # Track starvation (for trend analysis)
        self.steps_without_resource_gain: int = 0
        self.last_resource_estimate: float = 0.5
        
        # Exploration timing
        self.steps_since_explore: int = 0
        
        # Metabolism estimate (for buffer calculation)
        self.base_metabolism: float = 0.02
        
    def _get_exploration_mode(self) -> ExplorationMode:
        """
        K1.3 CORE: Determine if exploration is safe based on energy state and trend.
        
        DISABLED: Energy bleeding or critically low — DO NOT EXPLORE
        GUARDED: Stable but marginal — cautious exploration only
        ENABLED: Healthy margin — exploration is safe
        """
        E = self.state.energy_estimate
        trend = self.state.get_energy_trend(window=5)
        
        # Survival buffer: 3 steps of metabolism reserve
        survival_buffer = self.base_metabolism * 3  # 0.06
        
        # 🔴 DISABLED: Bleeding energy or below survival buffer
        # Trend analysis: negative slope means losing energy
        if E < survival_buffer * 2:  # < 0.12 (was critical 0.15 in K1.2)
            return ExplorationMode.DISABLED
        
        if trend < -0.05:  # Bleeding energy fast
            return ExplorationMode.DISABLED
            
        # 🟡 GUARDED: Marginal but stable
        # Energy is above critical but not comfortable, trend is neutral or positive
        if E < survival_buffer * 4:  # < 0.24
            return ExplorationMode.GUARDED
        
        if trend < 0:  # Slight bleeding but not critical
            return ExplorationMode.GUARDED
        
        # 🟢 ENABLED: Healthy and gaining or stable
        return ExplorationMode.ENABLED
    
    def _exploration_is_feasible(self) -> Tuple[bool, str]:
        """
        K1.3: Feasibility check before allowing exploration.
        
        Returns (should_explore, reason)
        """
        mode = self._get_exploration_mode()
        E = self.state.energy_estimate
        trend = self.state.get_energy_trend(window=5)
        survival_buffer = self.base_metabolism * 3
        
        if mode == ExplorationMode.DISABLED:
            return False, f"EXPLORATION DISABLED: E={E:.3f} < buffer={survival_buffer*2:.3f} or trend={trend:+.3f} < -0.05"
        
        if mode == ExplorationMode.GUARDED:
            # In guarded mode, only explore if we have consecutive successes
            success_rate = self.state.get_exploration_success_rate()
            if success_rate < 0.5:
                return False, f"EXPLORATION ABORT: Guarded mode + low success rate ({success_rate:.2f})"
            # Also require trend >= 0 (not losing energy)
            if trend < 0:
                return False, f"EXPLORATION ABORT: Guarded mode + negative trend ({trend:+.3f})"
        
        # Check for too many consecutive failures (exploration not working here)
        if self.consecutive_exploration_failures >= self.max_consecutive_failures:
            return False, f"EXPLORATION ABORT: {self.consecutive_exploration_failures} consecutive failures"
        
        return True, f"EXPLORATION APPROVED: mode={mode.value}, E={E:.3f}, trend={trend:+.3f}"
    
    def select_action(self, obs: Dict[str, float]) -> AgentAction:
        self.episode += 1
        self.state.update_from_observation(obs, self.episode)
        
        E = self.state.energy_estimate
        U = self.state.uncertainty
        R = self.state.resource_estimate
        
        # Detect resource gain (for starvation tracking)
        if R > self.last_resource_estimate + 0.05:
            self.steps_without_resource_gain = 0
        else:
            self.steps_without_resource_gain += 1
        self.last_resource_estimate = R
        
        # Update exploration tracking
        self.steps_since_explore += 1
        
        # 🔥 K1.3: CRITICAL SURVIVAL (unchanged from K1.2)
        # When energy is critically low, we MUST try something
        if E < self.energy_critical:
            # But even here, check if exploration has been failing
            if self.consecutive_exploration_failures < self.max_consecutive_failures:
                return AgentAction.EXPLORE
            else:
                # Exploration is killing us, try conserve instead
                return AgentAction.CONSERVE
        
        # 🔥 K1.3: STARVATION TRIGGERS (with feasibility check)
        if self.steps_without_resource_gain > 15:
            feasible, reason = self._exploration_is_feasible()
            if feasible:
                return AgentAction.EXPLORE
            else:
                # Can't explore safely — conserve and hope
                return AgentAction.CONSERVE
        
        # 🔥 K1.3: EXPLORATION FLOOR (THE FIX)
        # K1.2 BUG: This triggered regardless of energy state
        # K1.3 FIX: Only trigger if exploration is feasible
        if self.steps_since_explore > 10:
            feasible, reason = self._exploration_is_feasible()
            if feasible:
                self.steps_since_explore = 0
                return AgentAction.EXPLORE
            else:
                # Exploration would be dangerous — reset timer but don't explore
                self.steps_since_explore = 0
                # Fall through to normal policy
        
        # Normal policy (unchanged from K1.2)
        if U > self.uncertainty_high and E > self.energy_low:
            # High uncertainty + sufficient energy = explore
            feasible, _ = self._exploration_is_feasible()
            if feasible:
                return AgentAction.EXPLORE
            else:
                return AgentAction.CONSERVE  # Don't gamble
        elif E < self.energy_low:
            return AgentAction.CONSERVE
        else:
            return AgentAction.EXPLOIT
    
    def step(self, env) -> Dict[str, Any]:
        obs = env._generate_observation()
        action = self.select_action(obs)
        result = env.execute_action(action)
        
        # Track exploration outcomes
        if action == AgentAction.EXPLORE:
            self.steps_since_explore = 0
            # Check if exploration succeeded (positive utility)
            exploration_success = result["utility"] > 0
            if not exploration_success:
                self.consecutive_exploration_failures += 1
            else:
                self.consecutive_exploration_failures = 0
        
        self.cumulative_reward += result["utility"]
        self.state.record_action(
            self.episode, 
            action, 
            result["utility"],
            success=(result["utility"] > 0)
        )
        
        if result["alive"]:
            self.survival_count += 1
        else:
            self.death_count += 1
        
        return {
            "alive": result["alive"], 
            "observation": obs, 
            "action": action, 
            "episode": self.episode,
            "mode": self._get_exploration_mode().value,
            "energy_trend": self.state.get_energy_trend(5),
            "consecutive_failures": self.consecutive_exploration_failures
        }
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "episode": self.episode,
            "survival_count": self.survival_count,
            "death_count": self.death_count,
            "survival_rate": self.survival_count / max(1, self.episode),
            "cumulative_reward": self.cumulative_reward,
            "exploration_success_rate": self.state.get_exploration_success_rate(),
            "current_state": {
                "energy": self.state.energy_estimate,
                "resources": self.state.resource_estimate,
                "risk": self.state.risk_estimate,
                "uncertainty": self.state.uncertainty,
                "energy_trend": self.state.get_energy_trend(5),
                "exploration_mode": self._get_exploration_mode().value,
                "consecutive_exploration_failures": self.consecutive_exploration_failures
            }
        }
