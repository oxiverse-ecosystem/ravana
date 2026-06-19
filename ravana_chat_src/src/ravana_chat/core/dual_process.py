"""
Dual Process Controller for RAVANA.

System 1: Fast, automatic, heuristic-based processing (default for teens).
System 2: Slow, deliberate, analytical processing (engaged for novelty/complexity).

Based on: Kahneman's dual process theory, Evans & Stanovich (2013).
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Route(Enum):
    """Processing route."""
    SYSTEM1_FAST = "system1_fast"
    SYSTEM2_SLOW = "system2_slow"


@dataclass
class DualProcessConfig:
    """Configuration for dual process routing."""
    system2_confidence_threshold: float = 0.25  # engage System 2 if confidence below
    system2_novelty_threshold: float = 0.4      # engage System 2 if novelty above
    max_consecutive_system2: int = 5            # prevent System 2 lockup


@dataclass
class RouteDecision:
    """Decision output from dual process controller."""
    route: Route
    reason: str
    confidence: float
    novelty: float


class DualProcessController:
    """Routes cognition between fast (System 1) and slow (System 2) processing."""

    def __init__(self, config: Optional[DualProcessConfig] = None):
        self.config = config or DualProcessConfig()
        self.consecutive_system2 = 0
        self.last_route = Route.SYSTEM1_FAST

    def decide_route(self, confidence: float, novelty: float, stakes: float = 0.1) -> RouteDecision:
        """Decide processing route based on confidence, novelty, and stakes."""
        cfg = self.config

        # Force System 1 if too many consecutive System 2
        if self.consecutive_system2 >= cfg.max_consecutive_system2:
            decision = RouteDecision(
                route=Route.SYSTEM1_FAST,
                reason="max_consecutive_system2_reached",
                confidence=confidence,
                novelty=novelty)
            self.consecutive_system2 = 0
            self.last_route = decision.route
            return decision

        # Low confidence -> System 2
        if confidence < cfg.system2_confidence_threshold:
            decision = RouteDecision(
                route=Route.SYSTEM2_SLOW,
                reason="low_confidence",
                confidence=confidence,
                novelty=novelty)
            self.consecutive_system2 += 1
            self.last_route = decision.route
            return decision

        # High novelty -> System 2
        if novelty > cfg.system2_novelty_threshold:
            decision = RouteDecision(
                route=Route.SYSTEM2_SLOW,
                reason="high_novelty",
                confidence=confidence,
                novelty=novelty)
            self.consecutive_system2 += 1
            self.last_route = decision.route
            return decision

        # High stakes -> System 2
        if stakes > 0.5:
            decision = RouteDecision(
                route=Route.SYSTEM2_SLOW,
                reason="high_stakes",
                confidence=confidence,
                novelty=novelty)
            self.consecutive_system2 += 1
            self.last_route = decision.route
            return decision

        # Default: System 1
        decision = RouteDecision(
            route=Route.SYSTEM1_FAST,
            reason="default_fast",
            confidence=confidence,
            novelty=novelty)
        self.consecutive_system2 = 0
        self.last_route = decision.route
        return decision

    def reset(self):
        """Reset consecutive System 2 counter."""
        self.consecutive_system2 = 0