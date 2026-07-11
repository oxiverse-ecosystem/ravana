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
    """Processing route.

    Three-route switch unifying the two dual-process literatures (per the
    RAVANA brain-grounding plan, Section I key clarification):
      - SYSTEM1_FAST  : Kahneman S1 — fast, automatic familiarity (HRR decode).
      - RECOLLECT     : Yonelinas recollection — exact retrieval via the graph
                        (the graph-override). Engaged by the Botvinick ACC
                        conflict signal when HRR proposes no graph-coherent
                        candidate under genuine uncertainty.
      - SYSTEM2_SLOW  : Kahneman S2 — slow, deliberate explicit reasoning.
    """
    SYSTEM1_FAST = "system1_fast"
    RECOLLECT = "recollect"
    SYSTEM2_SLOW = "system2_slow"


@dataclass
class ConflictSignal:
    """Botvinick (2001) ACC-style conflict monitor output.

    Raised when HRR (familiarity) is uncertain — small top-1 vs top-2 gap AND
    the graph has an on-verb edge available to recollect. This recruits the
    RECOLLECT route (graph exact-edge correction) instead of wholesale System-2
    reasoning, mirroring conflict-driven control recruitment under genuine
    uncertainty only.
    """
    conflict: bool
    top1_conf: float
    top2_conf: float
    top1_top2_gap: float
    graph_has_edge: bool
    reason: str


@dataclass
class DualProcessConfig:
    """Configuration for dual process routing."""
    system2_confidence_threshold: float = 0.25  # engage System 2 if confidence below
    system2_novelty_threshold: float = 0.4      # engage System 2 if novelty above
    max_consecutive_system2: int = 5            # prevent System 2 lockup
    # Botvinick ACC conflict gate (recruits RECOLLECT, not S2):
    # HRR is "uncertain" only when the top-1 vs top-2 gap is below this
    # threshold AND the graph actually has an on-verb edge to recollect.
    conflict_gap_threshold: float = 0.06


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

    def conflict_monitor(self, top1_conf: float, top2_conf: float,
                         graph_has_edge: bool) -> ConflictSignal:
        """Botvinick (2001) ACC-style conflict detection for HRR familiarity.

        HRR decode confs are ~uniform (~0.58, uncalibrated), so a raw
        `conf < threshold` gate would fire on EVERY hop and wholesale-replace
        vector composition (the calibration trap). Instead we gate on the
        *gap* between the top-1 and top-2 HRR candidates (a local competition
        signal) AND require that the graph actually has an on-verb edge to
        recollect. Conflict is genuine only when HRR is a near-tie AND the
        graph can disambiguate. This recruits the RECOLLECT route under
        genuine uncertainty — never as a blanket fallback.

        Returns a ConflictSignal; callers map `conflict==True` to Route.RECOLLECT.
        """
        gap = float(top1_conf - top2_conf)
        uncertain = gap < self.config.conflict_gap_threshold
        conflict = bool(uncertain and graph_has_edge)
        reason = ("recollect: small top1-top2 gap (%.3f) + graph edge"
                  % gap) if conflict else (
            "no_conflict: gap=%.3f" % gap if not uncertain
            else "no_graph_edge: HRR uncertain but nothing to recollect")
        return ConflictSignal(
            conflict=conflict,
            top1_conf=float(top1_conf),
            top2_conf=float(top2_conf),
            top1_top2_gap=gap,
            graph_has_edge=bool(graph_has_edge),
            reason=reason)