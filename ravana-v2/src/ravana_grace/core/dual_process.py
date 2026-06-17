"""
RAVANA v2 — DUAL-PROCESS CONTROLLER
System 1 (fast/intuitive) vs System 2 (slow/deliberate) with override logic.

PRINCIPLE: Cognition balances speed and accuracy.
Most decisions use System 1. System 2 engages when stakes are high,
confidence is low, or novelty is detected.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum


class ProcessingRoute(Enum):
    SYSTEM1_FAST = "system1_fast"
    SYSTEM2_SLOW = "system2_slow"


@dataclass
class DualProcessConfig:
    """Configuration for dual-process controller."""
    # Confidence thresholds
    system2_confidence_threshold: float = 0.3
    system2_novelty_threshold: float = 0.6
    system2_stakes_threshold: float = 0.5
    
    # Cognitive load
    max_consecutive_system2: int = 5
    system2_cooldown_cycles: int = 3
    
    # Fluency heuristic
    fluency_confidence_threshold: float = 0.8
    fluency_speed_threshold: float = 0.3  # seconds equivalent
    
    # System 1 parameters
    system1_noise: float = 0.05
    system1_hysteresis: float = 0.1  # bias toward previous route


@dataclass
class RouteDecision:
    """Record of a route selection decision."""
    route: ProcessingRoute
    confidence: float
    novelty: float
    stakes: float
    reason: str
    cognitive_load: float
    fluency: float
    override_activated: bool = False


class DualProcessController:
    """
    Manages the System 1 / System 2 tradeoff.
    
    System 1 (fast): Hebbian concept activation, pattern completion
    System 2 (slow): MCTS planning, belief reasoning, argument construction
    
    Decides which route to take based on:
    - Confidence (low → System 2)
    - Novelty (high → System 2)
    - Stakes (high → System 2)
    - Cognitive load (high → bias toward System 1)
    - Fluency heuristic (fast + confident → skip System 2)
    """
    
    def __init__(self, config: Optional[DualProcessConfig] = None):
        self.config = config or DualProcessConfig()
        self._last_route: ProcessingRoute = ProcessingRoute.SYSTEM1_FAST
        self._consecutive_system2: int = 0
        self._cooldown_counter: int = 0
        self._decisions: List[RouteDecision] = []
        
    def decide_route(
        self,
        confidence: float,
        novelty: float = 0.0,
        stakes: float = 0.0,
        fluency: float = 0.5,
    ) -> RouteDecision:
        """
        Decide whether to use System 1 (fast) or System 2 (slow).
        
        Args:
            confidence: Current epistemic confidence (0-1)
            novelty: How novel the current situation is (0-1)
            stakes: How high the stakes are (0-1)
            fluency: How fluent/intuitive the current processing is (0-1)
        
        Returns:
            RouteDecision with the selected route and rationale
        """
        c = self.config
        
        # Cognitive load check: if too many consecutive System 2, force System 1
        if self._consecutive_system2 >= c.max_consecutive_system2:
            self._cooldown_counter = c.system2_cooldown_cycles
            decision = RouteDecision(
                route=ProcessingRoute.SYSTEM1_FAST,
                confidence=confidence,
                novelty=novelty,
                stakes=stakes,
                reason="system2_cooldown",
                cognitive_load=self._consecutive_system2 / c.max_consecutive_system2,
                fluency=fluency,
            )
            self._record_decision(decision)
            return decision
        
        # Cooldown from previous System 2 burst
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            decision = RouteDecision(
                route=ProcessingRoute.SYSTEM1_FAST,
                confidence=confidence,
                novelty=novelty,
                stakes=stakes,
                reason="system2_cooldown",
                cognitive_load=self._cooldown_counter / c.system2_cooldown_cycles,
                fluency=fluency,
            )
            self._record_decision(decision)
            return decision
        
        # Fluency heuristic: if fast and confident, skip System 2
        if fluency > c.fluency_confidence_threshold and confidence > c.fluency_confidence_threshold:
            decision = RouteDecision(
                route=ProcessingRoute.SYSTEM1_FAST,
                confidence=confidence,
                novelty=novelty,
                stakes=stakes,
                reason="fluency_heuristic",
                cognitive_load=0.0,
                fluency=fluency,
            )
            self._record_decision(decision)
            return decision
        
        # Check override conditions for System 2
        system2_reasons = []
        
        if confidence < c.system2_confidence_threshold:
            system2_reasons.append("low_confidence")
        
        if novelty > c.system2_novelty_threshold:
            system2_reasons.append("high_novelty")
        
        if stakes > c.system2_stakes_threshold:
            system2_reasons.append("high_stakes")
        
        # Hysteresis: bias toward previous route
        prev_bias = c.system1_hysteresis if self._last_route == ProcessingRoute.SYSTEM1_FAST else 0.0
        
        if system2_reasons and np.random.random() > prev_bias:
            route = ProcessingRoute.SYSTEM2_SLOW
            self._consecutive_system2 += 1
        else:
            route = ProcessingRoute.SYSTEM1_FAST
            self._consecutive_system2 = max(0, self._consecutive_system2 - 1)
        
        decision = RouteDecision(
            route=route,
            confidence=confidence,
            novelty=novelty,
            stakes=stakes,
            reason="+".join(system2_reasons) if system2_reasons else "default_system1",
            cognitive_load=self._consecutive_system2 / c.max_consecutive_system2,
            fluency=fluency,
            override_activated=bool(system2_reasons),
        )
        
        self._record_decision(decision)
        return decision
    
    def _record_decision(self, decision: RouteDecision):
        """Record and maintain decision history."""
        self._last_route = decision.route
        self._decisions.append(decision)
        if len(self._decisions) > 100:
            self._decisions = self._decisions[-100:]
    
    def get_system2_rate(self, window: int = 50) -> float:
        """Proportion of recent decisions using System 2."""
        recent = self._decisions[-window:] if len(self._decisions) >= window else self._decisions
        if not recent:
            return 0.0
        return sum(1 for d in recent if d.route == ProcessingRoute.SYSTEM2_SLOW) / len(recent)
    
    def get_status(self) -> Dict[str, Any]:
        """Full dual-process status."""
        return {
            "current_route": self._last_route.value,
            "consecutive_system2": self._consecutive_system2,
            "system2_rate": self.get_system2_rate(),
            "cooldown_remaining": self._cooldown_counter,
            "recent_decisions": [
                {
                    "route": d.route.value,
                    "reason": d.reason,
                    "stakes": d.stakes,
                    "override": d.override_activated,
                }
                for d in self._decisions[-10:]
            ]
        }
