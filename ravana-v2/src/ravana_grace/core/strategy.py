"""
RAVANA v2 — STRATEGY LAYER v0 (Phase B.5 Minimal)
Deliberate mode selection: choosing how to explore.

PRINCIPLE: Don't just react to clamps. Choose exploration mode based on context.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum, auto
from collections import deque


class ExplorationMode(Enum):
    """Deliberate exploration modes."""
    EXPLORE_AGGRESSIVE = "explore_aggressive"  # High uncertainty, high potential
    EXPLORE_SAFE = "explore_safe"              # Near boundary, need caution
    STABILIZE = "stabilize"                     # Lock in gains
    RECOVER = "recover"                         # Crisis response


@dataclass
class ModeSelection:
    """Result of mode selection decision."""
    mode: ExplorationMode
    confidence: float  # 0-1, how clear the decision is
    reason: str
    context: Dict[str, float]


@dataclass
class StrategyConfig:
    """Configuration for strategy layer."""
    # Mode thresholds
    crisis_clamp_rate: float = 0.15      # >15% clamps = crisis
    high_exploration_threshold: float = 0.25  # D < 0.25 = room to explore
    boundary_proximity: float = 0.75   # D > 0.75 = near boundary
    stability_threshold: float = 0.02    # Var < 0.02 = stable
    
    # Mode switching
    hysteresis_bonus: float = 0.1  # Preference for current mode
    
    # Mode → policy bias
    aggressive_delta_scale: float = 1.3
    aggressive_noise: float = 0.03
    aggressive_dampening: float = 0.3
    
    safe_delta_scale: float = 0.7
    safe_noise: float = 0.01
    safe_dampening: float = 0.7
    
    stabilize_delta_scale: float = 0.4
    stabilize_noise: float = 0.005
    stabilize_dampening: float = 0.9
    
    recover_delta_scale: float = 0.3
    recover_noise: float = 0.0
    recover_dampening: float = 1.0


@dataclass
class BehavioralContext:
    """Snapshot of current system state for mode selection."""
    clamp_rate: float = 0.0
    dissonance: float = 0.5
    identity: float = 0.5
    dissonance_trend: float = 0.0  # Positive = rising
    identity_drift: float = 0.0    # Negative = falling
    stability: float = 0.5         # Low variance = stable
    dissonance_variance: float = 0.0  # Recent dissonance variance
    recent_resolution_success: float = 0.5  # Resolution success rate
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for policy input."""
        return np.array([
            self.clamp_rate,
            self.dissonance,
            self.identity,
            self.dissonance_trend,
            self.identity_drift,
            self.stability
        ])


class StrategyLayer:
    """
    Minimal strategy layer: deliberate mode selection.
    
    Replaces: "react to clamps"
    With: "choose exploration mode based on context"
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.current_mode = ExplorationMode.EXPLORE_SAFE
        self.history: List[ModeSelection] = []
        
        # Track dissonance for trend computation
        self._dissonance_window: deque = deque(maxlen=10)
        
        # 🔴 EXIT INTELLIGENCE: Recover mode tracking
        self._recover_intensity: float = 0.0
        
        # Analytics
        self.mode_distribution: Dict[ExplorationMode, int] = {mode: 0 for mode in ExplorationMode}
        self.mode_history: List[ExplorationMode] = []  # Track mode sequence
        self.mode_durations: Dict[ExplorationMode, List[int]] = {mode: [] for mode in ExplorationMode}
        self.mode_switches: int = 0
        self.current_mode_start: int = 0
    
    def _compute_d_trend(self) -> float:
        """
        🔮 Compute dissonance trend for anticipatory switching.
        
        Positive = rising (toward boundary)
        Negative = falling (away from boundary)
        """
        if len(self._dissonance_window) < 4:
            return 0.0
        
        # Current vs 3 steps ago
        current = self._dissonance_window[-1]
        past = self._dissonance_window[-4]
        
        return current - past
    
    def select_mode(self, context: BehavioralContext) -> ModeSelection:
        """
        Select mode with:
        - Soft scoring (no hard thresholds)
        - Trend injection (anticipatory switching)
        - Hysteresis (preference for current mode)
        - Exit intelligence for RECOVER
        """
        # Compute trend for anticipatory switching
        d_trend = self._compute_d_trend()
        
        # Get soft scores
        scores = self._evaluate_mode_scores(context, d_trend)
        
        # 🔴 EXIT INTELLIGENCE: If in RECOVER, check if we can decay
        if self.current_mode == ExplorationMode.RECOVER:
            if context.clamp_rate < 0.08 and d_trend < 0:  # Improving
                # Decay recover intensity instead of hard exit
                self._recover_intensity *= 0.9  # Gradual exit
                if self._recover_intensity < 0.3:
                    # Safe to exit RECOVER
                    pass  # Let normal selection take over
            else:
                # Still in crisis, maintain intensity
                self._recover_intensity = min(1.0, self._recover_intensity * 1.1)
                # Force stay in recover
                return ModeSelection(
                    mode=ExplorationMode.RECOVER,
                    confidence=self._recover_intensity,
                    reason=f"RECOVER_INTENSITY: {self._recover_intensity:.2f}, clamp={context.clamp_rate:.2%}"
                )
        
        # Apply hysteresis (preference for current mode)
        if self.current_mode in scores:
            scores[self.current_mode] += self.config.hysteresis_bonus
        
        # Select mode with highest score
        best_mode = max(scores, key=scores.get)
        best_score = scores[best_mode]
        
        # Build reason string with trend info
        reason_parts = [f"score={best_score:.2f}"]
        if abs(d_trend) > 0.01:
            direction = "→" if d_trend > 0 else "←"
            reason_parts.append(f"trend={d_trend:+.3f}{direction}")
        
        return ModeSelection(
            mode=best_mode,
            confidence=best_score,
            reason=" | ".join(reason_parts),
            context={
                'clamp_rate': context.clamp_rate,
                'dissonance': context.dissonance,
                'identity': context.identity,
                'trend': d_trend,
                'score': best_score
            }
        )
    
    def _evaluate_mode_scores(
        self,
        context: BehavioralContext,
        d_trend: float
    ) -> Dict[ExplorationMode, float]:
        """
        🎯 SOFT SCORING: Convert hard thresholds to smooth sigmoid curves.
        
        Prevents mode cliffs and oscillations near boundaries.
        """
        D = context.dissonance
        I = context.identity
        clamp_rate = context.clamp_rate
        
        k = 15.0  # Steepness for sigmoid curves
        
        # RECOVER: Crisis detection (always highest priority if triggered)
        # Use sharp sigmoid for crisis — we want decisive response
        recover_score = 1.0 / (1.0 + np.exp(-k * (clamp_rate - 0.15)))
        
        # STABILIZE: Stable plateau with high identity
        # Smooth activation based on dissonance variance AND identity
        stability_score = (
            (1.0 / (1.0 + np.exp(-k * (0.05 - context.dissonance_variance)))) *  # Low variance
            (1.0 / (1.0 + np.exp(-k * (I - 0.70))))  # High identity
        )
        
        # EXPLORE_AGGRESSIVE: Room to grow (low D)
        # Soft threshold: stronger as D gets lower
        aggressive_score = 1.0 / (1.0 + np.exp(-k * (0.30 - D)))
        
        # EXPLORE_SAFE: Near boundary but not in crisis
        # Soft activation as D approaches limit
        safe_score = (
            (1.0 / (1.0 + np.exp(-k * (D - 0.65)))) *  # D is high
            (1.0 - recover_score)  # But NOT in crisis
        )
        
        # 🔮 TREND INJECTION: Anticipatory adjustment
        # If trending toward boundary, boost safe modes early
        if d_trend > 0.02:  # Drifting toward high dissonance
            # Boost safe mode before we hit the hard limit
            safe_score = max(safe_score, 0.3 + d_trend * 5.0)
            # Reduce aggressive to prevent drift
            aggressive_score *= 0.5
        elif d_trend < -0.02:  # Drifting toward center
            # Slight boost to aggressive (room opening up)
            aggressive_score = min(1.0, aggressive_score * 1.2)
        
        return {
            ExplorationMode.RECOVER: recover_score,
            ExplorationMode.STABILIZE: stability_score,
            ExplorationMode.EXPLORE_SAFE: safe_score,
            ExplorationMode.EXPLORE_AGGRESSIVE: aggressive_score,
        }
    
    def _select(self, mode: ExplorationMode, confidence: float, 
                reason: str, context: BehavioralContext) -> ModeSelection:
        """Record mode selection and return result."""
        # Track mode switches
        if mode != self.current_mode:
            duration = len(self.mode_history) - self.current_mode_start
            self.mode_durations[self.current_mode].append(duration)
            self.mode_switches += 1
            self.current_mode = mode
            self.current_mode_start = len(self.mode_history)
        
        self.mode_history.append(mode)
        
        return ModeSelection(
            mode=mode,
            confidence=confidence,
            reason=reason,
            context={
                'clamp_rate': context.clamp_rate,
                'dissonance': context.dissonance,
                'identity': context.identity,
                'trend': context.dissonance_trend
            }
        )
    
    def apply_policy_bias(self, raw_deltas: Tuple[float, float], 
                         mode: ExplorationMode) -> Tuple[float, float, Dict[str, Any]]:
        """
        Apply mode-specific policy bias to raw deltas.
        
        Returns: (modified_d_delta, modified_i_delta, bias_info)
        """
        dd, di = raw_deltas
        
        if mode == ExplorationMode.EXPLORE_AGGRESSIVE:
            dd *= self.config.aggressive_delta_scale
            noise_dd = np.random.normal(0, self.config.aggressive_noise)
            noise_di = np.random.normal(0, self.config.aggressive_noise * 0.5)
            dd += noise_dd
            di += noise_di
            dampening = self.config.aggressive_dampening
            
        elif mode == ExplorationMode.EXPLORE_SAFE:
            dd *= self.config.safe_delta_scale
            noise_dd = np.random.normal(0, self.config.safe_noise)
            dd += noise_dd
            dampening = self.config.safe_dampening
            
        elif mode == ExplorationMode.STABILIZE:
            dd *= self.config.stabilize_delta_scale
            # Reduce identity noise to preserve gains
            noise_di = np.random.normal(0, self.config.stabilize_noise)
            di += noise_di
            dampening = self.config.stabilize_dampening
            
        elif mode == ExplorationMode.RECOVER:
            dd *= self.config.recover_delta_scale
            # Force dissonance reduction
            if dd > 0:
                dd = -abs(dd) * 0.5  # Reverse and dampen
            dampening = self.config.recover_dampening
            
        else:
            dampening = 0.5  # Default
        
        return dd, di, {
            'mode': mode.value,
            'delta_scale': getattr(self.config, f'{mode.value}_delta_scale', 1.0),
            'noise_injected': noise_dd if 'noise_dd' in dir() else 0.0,
            'dampening': dampening
        }
    
    def compute_context(self, governor, state_manager, window: int = 20) -> BehavioralContext:
        """
        Compute behavioral context from recent history.
        
        Extracts trends and patterns for mode selection.
        """
        history = state_manager.history[-window:] if len(state_manager.history) >= window else state_manager.history
        
        if len(history) < 5:
            return BehavioralContext()  # Not enough data
        
        # Recent clamp rate
        recent_caps = sum(1 for h in history if h.get('constraint_activated', False))
        clamp_rate = recent_caps / len(history)
        
        # Current state
        current = state_manager.state
        
        # Trend computation (simple linear trend)
        if len(history) >= 10:
            recent_d = [h['post_dissonance'] for h in history[-10:]]
            early_mean = np.mean(recent_d[:5])
            late_mean = np.mean(recent_d[5:])
            dissonance_trend = late_mean - early_mean
        else:
            dissonance_trend = 0.0
        
        # Identity drift
        if len(history) >= 10:
            recent_i = [h['post_identity'] for h in history[-10:]]
            early_i = np.mean(recent_i[:5])
            late_i = np.mean(recent_i[5:])
            identity_drift = late_i - early_i
        else:
            identity_drift = 0.0
        
        # Stability (variance)
        if len(history) >= 5:
            recent_states = [h['post_dissonance'] for h in history[-5:]]
            stability = np.var(recent_states)
            dissonance_variance = stability  # Same as stability for recent window
        else:
            stability = 0.5
            dissonance_variance = 0.1
        
        return BehavioralContext(
            clamp_rate=clamp_rate,
            dissonance=current.dissonance,
            identity=current.identity,
            dissonance_trend=dissonance_trend,
            identity_drift=identity_drift,
            stability=stability,
            dissonance_variance=dissonance_variance
        )
    
    def get_mode_analytics(self) -> Dict[str, Any]:
        """
        Return analytics about mode usage and effectiveness.
        """
        if not self.mode_history:
            return {"error": "No mode history"}
        
        # Mode distribution
        mode_counts = {}
        for mode in self.mode_history:
            mode_counts[mode.value] = mode_counts.get(mode.value, 0) + 1
        
        # Average durations
        avg_durations = {}
        for mode, durations in self.mode_durations.items():
            avg_durations[mode.value] = np.mean(durations) if durations else 0
        
        # Switch frequency
        total_steps = len(self.mode_history)
        switch_rate = self.mode_switches / total_steps if total_steps > 0 else 0
        
        return {
            "mode_distribution": mode_counts,
            "avg_mode_duration": avg_durations,
            "mode_switches": self.mode_switches,
            "switch_rate": switch_rate,
            "current_mode": self.current_mode.value,
            "current_mode_duration": len(self.mode_history) - self.current_mode_start
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Quick status summary."""
        return {
            "current_mode": self.current_mode.value,
            "mode_history_length": len(self.mode_history),
            "mode_switches": self.mode_switches,
            "recent_modes": [m.value for m in list(self.mode_history)[-5:]]
        }
    
    def update_mode_analytics(self, mode: ExplorationMode, episode: int):
        """Update analytics when a mode is selected."""
        # Track distribution
        self.mode_distribution[mode] = self.mode_distribution.get(mode, 0) + 1
        
        # Track switches
        if mode != self.current_mode:
            self.mode_switches += 1
            self.current_mode = mode
            self.current_mode_start = episode
        
        # Track dissonance for trend computation
        # (This would be called from outside with current dissonance)
