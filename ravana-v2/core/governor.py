"""
RAVANA v2 — GOVERNOR (First-Class Citizen)
Central control system for cognitive state regulation.

PRINCIPLE: No state modification without governor passage.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import math


class RegulationMode(Enum):
    """Governor regulation modes."""
    NORMAL = "normal"          # Standard operation
    EXPLORATION = "exploration"  # High uncertainty, seek novelty
    RESOLUTION = "resolution"    # Active conflict resolution
    RECOVERY = "recovery"        # Crisis recovery mode
    PLATEAU = "plateau"          # Stagnation detected


@dataclass
class GovernorConfig:
    """Immutable governor configuration."""
    # Hard constraints (non-negotiable)
    max_dissonance: float = 0.95
    min_dissonance: float = 0.15
    target_dissonance: float = 0.30  # Paper target
    max_identity: float = 0.95
    soft_limit: float = 0.70  # Start pressure here
    boundary_k: float = 12.0  # Slightly steeper
    min_pressure: float = 0.2  # Minimum allowed pressure
    min_identity: float = 0.10
    
    # Regulation parameters
    dissonance_target: float = 0.35  # Closer to paper's 0.3
    identity_target: float = 0.85   # Target paper's 0.85
    exploration_threshold: float = 0.25
    resolution_threshold: float = 0.60  # Trigger sooner
    
    # Smoothing
    use_smoothed_dissonance: bool = True  # Enable EMA-based regulation
    smoothing_alpha: float = 0.2          # EMA alpha (lower = more smoothing)
    
    # Recovery parameters
    recovery_boost: float = 0.15
    crisis_threshold: float = 0.90
    
    # Plateau detection
    plateau_window: int = 50
    plateau_tolerance: float = 0.02
    
    # Clamp diagnostics
    clamp_alert_threshold: float = 0.05  # Log alert if correction > 5%


@dataclass
class ClampEvent:
    """Single clamp correction event."""
    episode: int
    variable: str  # 'dissonance' or 'identity'
    before: float
    after: float
    correction: float
    layer: str  # 'hard_constraint' or 'final_clamp'
    reason: str


@dataclass
class ClampDiagnostics:
    """
    Comprehensive clamp diagnostics.
    
    Tracks how often the governor's hard constraints override
    upstream suggestions, and measures controller/clamp alignment.
    """
    # Raw event log
    events: List[ClampEvent] = field(default_factory=list)
    
    # Counters
    total_upstream_suggestions: int = 0
    
    # Dissonance clamp stats
    d_clamp_activations: int = 0
    d_clamp_corrections_total: float = 0.0
    d_clamp_by_layer: Dict[str, int] = field(default_factory=lambda: {"hard_constraint": 0, "final_clamp": 0})
    
    # Identity clamp stats  
    i_clamp_activations: int = 0
    i_clamp_corrections_total: float = 0.0
    i_clamp_by_layer: Dict[str, int] = field(default_factory=lambda: {"hard_constraint": 0, "final_clamp": 0})
    
    # Significant corrections (above alert threshold)
    significant_corrections: int = 0
    
    def record_event(self, event: ClampEvent):
        """Record a clamp event and update counters."""
        self.events.append(event)
        
        if event.variable == 'dissonance':
            self.d_clamp_activations += 1
            self.d_clamp_corrections_total += event.correction
            self.d_clamp_by_layer[event.layer] = self.d_clamp_by_layer.get(event.layer, 0) + 1
        else:
            self.i_clamp_activations += 1
            self.i_clamp_corrections_total += event.correction
            self.i_clamp_by_layer[event.layer] = self.i_clamp_by_layer.get(event.layer, 0) + 1
        
        # Check if significant
        if event.correction > 0.05:  # 5% threshold
            self.significant_corrections += 1
    
    def record_upstream_suggestion(self):
        """Count a suggestion from upstream (for alignment ratio)."""
        self.total_upstream_suggestions += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Compute diagnostic metrics."""
        if self.total_upstream_suggestions == 0:
            return {"status": "no_data"}
        
        total_clamps = self.d_clamp_activations + self.i_clamp_activations
        total_correction = self.d_clamp_corrections_total + self.i_clamp_corrections_total
        
        return {
            # Alignment: how often does upstream need correction?
            "clamp_rate": total_clamps / self.total_upstream_suggestions,
            "alignment_score": 1.0 - (total_clamps / self.total_upstream_suggestions),
            
            # By variable
            "d_clamp_rate": self.d_clamp_activations / max(1, self.total_upstream_suggestions),
            "i_clamp_rate": self.i_clamp_activations / max(1, self.total_upstream_suggestions),
            
            # Magnitude
            "mean_correction": total_correction / max(1, total_clamps) if total_clamps > 0 else 0,
            "total_correction": total_correction,
            
            # By layer
            "hard_constraint_clamps": self.d_clamp_by_layer.get("hard_constraint", 0) + self.i_clamp_by_layer.get("hard_constraint", 0),
            "final_clamp_clamps": self.d_clamp_by_layer.get("final_clamp", 0) + self.i_clamp_by_layer.get("final_clamp", 0),
            
            # Alerts
            "significant_corrections": self.significant_corrections,
            "significant_rate": self.significant_corrections / max(1, total_clamps) if total_clamps > 0 else 0,
            
            # Raw counts
            "total_clamps": total_clamps,
            "d_clamps": self.d_clamp_activations,
            "i_clamps": self.i_clamp_activations,
            "upstream_suggestions": self.total_upstream_suggestions,
        }
    
    def get_summary_report(self) -> str:
        """Generate human-readable summary."""
        m = self.get_metrics()
        if m.get("status") == "no_data":
            return "[ClampDiagnostics] No data collected yet."
        
        lines = [
            "=" * 50,
            "📊 CLAMP DIAGNOSTICS REPORT",
            "=" * 50,
            f"Alignment Score: {m['alignment_score']:.1%} (higher = better)",
            f"  └─ Upstream suggestions: {m['upstream_suggestions']:,}",
            f"  └─ Total clamps applied: {m['total_clamps']:,}",
            "",
            "By Variable:",
            f"  Dissonance: {m['d_clamps']:,} clamps ({m['d_clamp_rate']:.1%} rate)",
            f"  Identity:   {m['i_clamps']:,} clamps ({m['i_clamp_rate']:.1%} rate)",
            "",
            "By Layer:",
            f"  Hard constraints: {m['hard_constraint_clamps']:,}",
            f"  Final clamp:      {m['final_clamp_clamps']:,}",
            "",
            f"Mean correction: {m['mean_correction']:.4f}",
            f"Significant corrections (>5%): {m['significant_corrections']:,} ({m['significant_rate']:.1%})",
            "=" * 50,
        ]
        return "\n".join(lines)


@dataclass
class CognitiveSignals:
    """Raw signals from cognitive modules (pre-governor)."""
    dissonance_delta: float = 0.0
    identity_delta: float = 0.0
    exploration_drive: float = 0.0
    resolution_potential: float = 0.0
    
    # 🔮 Grace Layer: Predictive fields
    trend: float = 0.0  # dD/dt (emotional salience proxy)
    predicted_dissonance: float = 0.0  # Look-ahead prediction
    horizon: int = 3  # Prediction steps ahead
    
    # Metadata
    source: str = "unknown"
    confidence: float = 0.5


@dataclass
class RegulatedOutput:
    """Governor-regulated output (post-governor)."""
    dissonance_delta: float = 0.0
    identity_delta: float = 0.0
    mode: RegulationMode = RegulationMode.NORMAL
    
    # Governor decisions
    dampened: bool = False
    boosted: bool = False
    capped: bool = False
    
    # Debug info
    reason: str = ""
    raw_input: Optional[CognitiveSignals] = None


class Governor:
    """
    Central governor for RAVANA v2.
    
    NON-NEGOTIABLE PRINCIPLE:
    All state changes flow through here. No exceptions.
    """
    
    def __init__(self, config: Optional[GovernorConfig] = None):
        self.config = config or GovernorConfig()
        self.history: List[RegulatedOutput] = []
        self.mode_history: List[RegulationMode] = []
        
        # Plateau tracking
        self.recent_dissonance: List[float] = []
        self.recent_identity: List[float] = []
        
        # 🔮 Grace Layer tracking
        self.predictions_made: int = 0
        self.predictions_correct: int = 0
        self.boundary_pressure_history: List[float] = []
        self.center_force_history: List[float] = []
        self.dampening_activations: int = 0
        self.overshoot_events: int = 0

        # 🔴 CLAMP DIAGNOSTICS: New comprehensive system
        self.clamp_diagnostics = ClampDiagnostics()
        
    def regulate(
        self,
        current_dissonance: float,
        current_identity: float,
        signals: CognitiveSignals,
        episode: int = 0
    ) -> RegulatedOutput:
        """
        Central regulation point. ALL state changes pass through here.
        
        Returns regulated deltas that keep system in healthy bounds.
        """
        # Count this as an upstream suggestion
        self.clamp_diagnostics.record_upstream_suggestion()
        
        # Detect current mode
        mode = self._detect_mode(
            current_dissonance,
            current_identity,
            signals
        )
        
        # Apply hard constraints FIRST
        dissonance_delta, identity_delta, constraints = self._apply_hard_constraints(
            current_dissonance,
            current_identity,
            signals,
            episode
        )
        
        # 🔮 PHASE B.0: Grace Layer — Predictive & Soft Regulation
        # 1. Look ahead: dampen based on predicted future state
        dissonance_delta = self._predictive_dampening(current_dissonance, dissonance_delta, signals)
        
        # 2. Feel the boundary: air resistance near limits
        dissonance_delta = self._boundary_pressure(current_dissonance, dissonance_delta)
        
        # 3. Return to center: homeostatic pull
        dissonance_delta = self._center_seeking_force(current_dissonance, dissonance_delta)
        
        # Apply mode-specific regulation
        dissonance_delta, identity_delta = self._apply_mode_regulation(
            mode,
            dissonance_delta,
            identity_delta,
            current_dissonance,
            current_identity,
            episode
        )
        
        # NOTE: The old universal suppression (dd < 0 and current_dissonance > 0.35)
        # has been removed. It was preventing correct episodes from reducing
        # dissonance even when appropriate. Boundary pressure and mode-specific
        # regulation in RESOLUTION mode handle high-D cases properly.
        # 
        # The boundary_pressure() function disables itself when D >= resolution_threshold
        # (_apply_mode_regulation RESOLUTION case) so dissonance can grow naturally.
        # The _apply_mode_regulation RESOLUTION case also suppresses reduction when D > 0.35.
        
        # Build output
        output = RegulatedOutput(
            dissonance_delta=dissonance_delta,
            identity_delta=identity_delta,
            mode=mode,
            dampened=constraints.get('dampened', False),
            boosted=constraints.get('boosted', False),
            capped=constraints.get('capped', False),
            reason=constraints.get('reason', ''),
            raw_input=signals
        )
        
        # Track history
        self.history.append(output)
        self.mode_history.append(mode)
        self._update_tracking(current_dissonance, current_identity)
        
        return output
    
    def get_clamp_report(self) -> str:
        """Get human-readable clamp diagnostics report."""
        return self.clamp_diagnostics.get_summary_report()
    
    def get_clamp_metrics(self) -> Dict[str, Any]:
        """Get clamp diagnostics as dict."""
        return self.clamp_diagnostics.get_metrics()

    def get_health_metrics(self) -> Dict[str, Any]:
        """
        Return health metrics for the Grace Layer.
        """
        recent = self.history[-20:] if len(self.history) >= 20 else self.history
        
        return {
            'predictions_made': self.predictions_made,
            'boundary_pressure_avg': sum(self.boundary_pressure_history[-10:]) / len(self.boundary_pressure_history[-10:]) if self.boundary_pressure_history else 0,
            'center_force_avg': sum(self.center_force_history[-10:]) / len(self.center_force_history[-10:]) if self.center_force_history else 0,
            'overshoot_count': sum(1 for r in recent if r.capped),
            'mean_approach_velocity': np.mean([abs(r.dissonance_delta) for r in recent]) if recent else 0.0,
            'total_regulation_events': len(self.history),
            'predictions_made': self.predictions_made,
            'prediction_accuracy': self.predictions_correct / max(1, self.predictions_made),
            # Include clamp metrics
            'clamp_metrics': self.clamp_diagnostics.get_metrics(),
        }

    def _detect_mode(
        self,
        dissonance: float,
        identity: float,
        signals: CognitiveSignals
    ) -> RegulationMode:
        """Detect which regulation mode to use."""
        
        # CRISIS: Near catastrophic bounds
        if dissonance > self.config.crisis_threshold or identity < self.config.min_identity:
            return RegulationMode.RECOVERY
        
        # PLATEAU: Stagnation detected
        if self._is_plateau():
            return RegulationMode.PLATEAU
        
        # RESOLUTION: High dissonance, ready to resolve
        if dissonance > self.config.resolution_threshold:
            return RegulationMode.RESOLUTION
        
        # EXPLORATION: Low dissonance, seek novelty
        if dissonance < self.config.exploration_threshold:
            return RegulationMode.EXPLORATION
        
        # Normal operation
        return RegulationMode.NORMAL
    
    def _apply_hard_constraints(
        self,
        current_dissonance: float,
        current_identity: float,
        signals: CognitiveSignals,
        episode: int = 0
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Apply hard constraints. These are ABSOLUTE and cannot be overridden.
        """
        constraints = {'dampened': False, 'boosted': False, 'capped': False, 'reason': ''}
        
        dd = signals.dissonance_delta
        id_val = signals.identity_delta
        
        # === DISSONANCE HARD CONSTRAINTS ===
        
        # CEILING: Prevent dissonance > max_dissonance
        projected_d = current_dissonance + dd
        if projected_d > self.config.max_dissonance:
            dd_before = dd  # Always save before modification
            
            # FIX (high_dissonance_pressure): When D is already HIGH (> 0.35),
            # don't force a reduction. Allow D to stabilize at ceiling or even
            # decrease naturally. The ceiling clamp should NOT override failures'
            # intended D increase when D > 0.35.
            if current_dissonance > 0.35 and dd > 0:
                # D is high and upstream wants to increase it - let it pass
                # but don't actually increase past ceiling, just cap at ceiling
                dd = min(dd, 0.0)  # No increase allowed, but don't force decrease
                constraints['reason'] = f"dissonance_ceiling_highD (proj={projected_d:.3f})"
            else:
                dd = self.config.max_dissonance - current_dissonance - 0.01  # Stay just below
                constraints['capped'] = True
                constraints['reason'] = f"dissonance_ceiling (proj={projected_d:.3f})"
            
            # 🔴 Record clamp event
            self.clamp_diagnostics.record_event(ClampEvent(
                episode=episode,
                variable='dissonance',
                before=dd_before,
                after=dd,
                correction=abs(dd_before - dd),
                layer='hard_constraint',
                reason='dissonance_ceiling'
            ))
        
        # FLOOR: Prevent dissonance < min_dissonance
        if projected_d < self.config.min_dissonance:
            dd_before = dd
            dd = self.config.min_dissonance - current_dissonance + 0.01  # Stay just above
            constraints['capped'] = True
            constraints['reason'] += f" dissonance_floor (proj={projected_d:.3f})"
            
            # 🔴 Record clamp event
            self.clamp_diagnostics.record_event(ClampEvent(
                episode=episode,
                variable='dissonance',
                before=dd_before,
                after=dd,
                correction=abs(dd_before - dd),
                layer='hard_constraint',
                reason='dissonance_floor'
            ))
        
        # === IDENTITY HARD CONSTRAINTS ===
        
        # IDENTITY FLOOR: Prevent identity collapse
        projected_i = current_identity + id_val
        if projected_i < self.config.min_identity:
            id_before = id_val
            id_val = self.config.min_identity - current_identity + 0.01
            constraints['boosted'] = True
            constraints['reason'] += f" identity_floor (proj={projected_i:.3f})"
            
            # 🔴 Record clamp event
            self.clamp_diagnostics.record_event(ClampEvent(
                episode=episode,
                variable='identity',
                before=id_before,
                after=id_val,
                correction=abs(id_before - id_val),
                layer='hard_constraint',
                reason='identity_floor'
            ))
        
        # IDENTITY CEILING: Prevent identity inflation
        if projected_i > self.config.max_identity:
            id_before = id_val
            id_val = self.config.max_identity - current_identity - 0.01
            constraints['capped'] = True
            constraints['reason'] += f" identity_ceiling (proj={projected_i:.3f})"
            
            # 🔴 Record clamp event
            self.clamp_diagnostics.record_event(ClampEvent(
                episode=episode,
                variable='identity',
                before=id_before,
                after=id_val,
                correction=abs(id_before - id_val),
                layer='hard_constraint',
                reason='identity_ceiling'
            ))
        
        # FIX: Governor's identity_delta zeroing should ONLY apply to FAILURE cases.
        # When correctness=False, the identity engine already computed the failure
        # penalty in its first pass. Zeroing negative identity_delta is correct.
        # But for correctness=True, identity_delta > 0 is the resolution bonus
        # that should NOT be zeroed - it represents legitimate identity growth.
        # Only zero when identity_delta is negative (failure penalty already applied).
        if signals.source == "state_step" and id_val < 0:
            id_val = 0.0
        
        return dd, id_val, constraints
    


    def _predictive_dampening(
        self,
        current_d: float,
        dd: float,
        signals: CognitiveSignals
    ) -> float:
        """
        🔮 Look-ahead regulation: dampen based on FUTURE state, not current.
        
        Principle: "Slow down before you see the wall"
        """
        # Predict where we'll be after horizon steps
        predicted_d = current_d + dd * signals.horizon
        signals.predicted_dissonance = predicted_d
        
        # If prediction exceeds threshold, apply early dampening
        threshold = self.config.max_dissonance * 0.85  # Start early
        if predicted_d > threshold:
            # Progressive reduction: stronger as we approach limit
            overshoot = predicted_d - threshold
            reduction = 1.0 / (1.0 + overshoot * 2.0)  # Smooth decay
            dd *= reduction
            self.predictions_made += 1
            if hasattr(self, '_last_log') and self._last_log:
                print(f"  [PREDICTIVE] D={current_d:.3f} → predicted={predicted_d:.3f}, reducing dd by {reduction:.2f}x")
        
        return dd

    def _boundary_pressure(
        self,
        current_d: float,
        dd: float
    ) -> float:
        """
        🌊 Soft boundary pressure: air resistance, not brick wall.
        
        Starts subtle, becomes dominant near boundary.
        Returns dampened dd.
        
        FIX (high_dissonance_pressure): When D is already high (>= resolution_threshold),
        we're in RESOLUTION mode — the system needs D to grow naturally to drive conflict.
        Disable boundary pressure when D is high so failures CAN increase D.
        """
        # Below soft limit: no pressure
        if current_d < self.config.soft_limit:
            if hasattr(self, 'boundary_pressure_history'):
                self.boundary_pressure_history.append(0.0)
                if len(self.boundary_pressure_history) > 100:
                    self.boundary_pressure_history.pop(0)
            return dd
        
        # FIX: When D >= resolution_threshold, we're in RESOLUTION mode.
        # Boundary pressure fights against the natural D increase needed for conflict.
        # Disable it here so failures can increase D even when D is high.
        if current_d >= self.config.resolution_threshold:
            if hasattr(self, 'boundary_pressure_history'):
                self.boundary_pressure_history.append(0.0)
                if len(self.boundary_pressure_history) > 100:
                    self.boundary_pressure_history.pop(0)
            return dd
        
        # Sigmoid pressure curve for moderate D
        excess = current_d - self.config.soft_limit
        k = getattr(self.config, 'boundary_k', 10.0)
        
        import math
        pressure = 1.0 / (1.0 + math.exp(-k * (excess - 0.05)))
        
        dampened_dd = dd * (1.0 - pressure * 0.8)
        
        if hasattr(self, 'boundary_pressure_history'):
            self.boundary_pressure_history.append(pressure)
            if len(self.boundary_pressure_history) > 100:
                self.boundary_pressure_history.pop(0)
        
        return dampened_dd

    def _apply_mode_regulation(
        self,
        mode: RegulationMode,
        dd: float,
        id_val: float,
        current_d: float,
        current_i: float,
        episode: int = 0
    ) -> Tuple[float, float]:
        """Apply mode-specific regulation."""
        
        if mode == RegulationMode.RECOVERY:
            # Recovery: aggressive stabilization
            # FIX (high_dissonance_pressure): When D is at ceiling (max_dissonance),
            # don't force reduction - maintain high D so failures keep pressure up.
            # Only force reduction when D is below the ceiling.
            if current_d > self.config.crisis_threshold and current_d < self.config.max_dissonance - 0.01:
                dd = -0.05  # Force reduction only when below ceiling
            
            # NOTE: Identity boost in RECOVERY mode has been REMOVED.
            # The identity engine handles failure penalties independently via
            # the correctness flag. Adding governor-side boosts here caused
            # oscillation between identity_floor clamping and RECOVERY boost.
            # Let identity handle its own recovery through normal mechanics.
            
        elif mode == RegulationMode.NORMAL:
            # Normal operation: standard regulation
            # FIX (wisdom_stagnation): Changed threshold from 0.35 to 0.65.
            # The 0.35 threshold was too conservative — it blocked D reduction
            # when D was in the normal operating band (0.35-0.65), preventing
            # the resolution engine from computing meaningful delta events,
            # which stopped wisdom accumulation entirely. The new 0.65 threshold
            # allows D to reduce in normal operation while still protecting
            # against over-correction near crisis levels (> 0.65).
            if dd < 0 and current_d > 0.65:
                dd = 0.0  # Suppress reduction only when D is genuinely elevated
                
        elif mode == RegulationMode.PLATEAU:
            # Plateau: controlled perturbation
            # FIX (wisdom_stagnation): Changed threshold from 0.35 to 0.65.
            # Match NORMAL mode — allow perturbations to reduce D in the
            # normal operating band, only suppress near crisis.
            if dd < 0 and current_d > 0.65:
                dd = 0.0  # Suppress reduction only when D is elevated
            else:
                dd += np.random.normal(0, 0.03)
            id_val += np.random.normal(0, 0.01)
            
        elif mode == RegulationMode.RESOLUTION:
            # Resolution: adaptive amplification based on safety
            # Safe (low D): amplify freely
            # Near boundary: dampen to avoid overshoot
            safety_factor = 1.0 - current_d  # 1.0 at D=0, 0.0 at D=1.0
            
            # FIX (wisdom_stagnation): Changed threshold from 0.35 to 0.65.
            # When D is moderately high (0.35-0.65), allow natural reduction
            # as part of the resolution process. Only suppress when D is
            # genuinely elevated (> 0.65), where over-correction risk is real.
            if dd < 0 and current_d > 0.65:
                # SUPPRESS reduction only when D is elevated — allow natural
                # resolution to proceed when D is in the normal band
                dd = 0.0  # Suppression only above 0.65
            elif dd < 0:
                # Normal reduction when D is not yet high — amplify it
                amplification = 1.0 + (0.2 * safety_factor)
                dd *= amplification
            else:
                # Amplify increases when D is low (normal resolution growth)
                amplification = 1.0 + (0.2 * safety_factor)
                dd *= amplification
            
            if dd < 0:  # Reducing dissonance
                # Governor doesn't know correctness — don't boost identity here.
                # Identity engine has the final say with full context (correctness flag).
                # Governor should only dampen/govern, not reward.
                # Id val stays as regulated. Pass 0 to identity to avoid double-counting.
                pass
                
        elif mode == RegulationMode.EXPLORATION:
            # Exploration: maintain curiosity
            if abs(dd) < 0.01:  # Too stable
                dd = np.random.choice([-0.02, 0.02])  # Induce small variation
        
        # === FINAL HARD CLAMP: Absolute enforcement after all processing ===
        
        # Clamp dissonance delta
        max_allowed_d = self.config.max_dissonance - current_d
        min_allowed_d = self.config.min_dissonance - current_d
        dd_before = dd
        dd = np.clip(dd, min_allowed_d - 0.01, max_allowed_d - 0.01)
        
        # Record if clamp was applied
        if abs(dd - dd_before) > 0.0001:
            self.clamp_diagnostics.record_event(ClampEvent(
                episode=episode,
                variable='dissonance',
                before=dd_before,
                after=dd,
                correction=abs(dd_before - dd),
                layer='final_clamp',
                reason='final_safety_clamp'
            ))
        
        # Clamp identity delta
        max_allowed_i = self.config.max_identity - current_i
        min_allowed_i = self.config.min_identity - current_i
        id_before = id_val
        id_val = np.clip(id_val, min_allowed_i + 0.01, max_allowed_i - 0.01)
        correction = abs(id_before - id_val)
        
        if correction > 0.0001:  # Clamp actually changed the value
            self.clamp_diagnostics.record_event(ClampEvent(
                episode=episode,
                variable='identity',
                before=id_before,
                after=id_val,
                correction=correction,
                layer='final_clamp',
                reason='final_safety_clamp'
            ))
            
            # Log if clamp is fighting upstream significantly
            if correction > self.config.clamp_alert_threshold:
                print(f"  [CLAMP ALERT] Upstream suggested {id_before:+.3f}, clamped to {id_val:+.3f} (Δ{correction:.3f})")
        
        return dd, id_val
    
    def _is_plateau(self) -> bool:
        """Detect if system is in plateau (stagnation)."""
        if len(self.recent_dissonance) < self.config.plateau_window:
            return False
        
        recent = self.recent_dissonance[-self.config.plateau_window:]
        variance = np.var(recent)
        
        return variance < self.config.plateau_tolerance
    

    def _center_seeking_force(
        self,
        current_d: float,
        dd: float
    ) -> float:
        """
        🎯 Anti-overshoot term: pull toward center when far from it.
        
        Prevents drift accumulation, creates homeostasis.
        
        FIX (high_dissonance_pressure): When D is high (> 0.35),
        the system should be in resolution mode, NOT auto-correcting downward.
        Center-seeking is DISABLED when D > 0.35 so failures can increase D naturally.
        """
        target = self.config.target_dissonance
        distance_from_center = current_d - target
        
        # FIX: Only apply center-seeking when D is below 0.35 (moderate range).
        # When D > 0.35, we're in high-dissonance territory where failures
        # should naturally increase D. Disable center-seeking here.
        if abs(distance_from_center) > 0.12 and current_d < 0.35:
            k_center = 0.06
            center_force = -distance_from_center * k_center
            dd += center_force
            
            if hasattr(self, '_last_log') and self._last_log:
                direction = "→" if center_force > 0 else "←"
                print(f"  [CENTER] D={current_d:.3f} {direction} target={target:.3f} (force={center_force:.4f})")
        
        return dd

    def _update_tracking(self, dissonance: float, identity: float):
        """Update tracking buffers."""
        self.recent_dissonance.append(dissonance)
        self.recent_identity.append(identity)
        
        # Keep only necessary history
        max_len = self.config.plateau_window + 10
        if len(self.recent_dissonance) > max_len:
            self.recent_dissonance = self.recent_dissonance[-max_len:]
            self.recent_identity = self.recent_identity[-max_len:]
    
    def get_status(self) -> Dict[str, Any]:
        """Return governor status for monitoring."""
        if not self.history:
            return {"mode": "unknown", "cycles": 0}
        
        recent_modes = self.mode_history[-20:]
        mode_counts = {}
        for m in recent_modes:
            mode_counts[m.value] = mode_counts.get(m.value, 0) + 1
        
        return {
            "current_mode": self.mode_history[-1].value if self.mode_history else "unknown",
            "mode_distribution": mode_counts,
            "total_cycles": len(self.history),
            "constraint_activations": sum(1 for h in self.history if h.capped or h.dampened),
            "recent_dissonance_variance": np.var(self.recent_dissonance) if self.recent_dissonance else 0,
        }
