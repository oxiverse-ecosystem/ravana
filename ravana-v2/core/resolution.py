"""
RAVANA v2 — RESOLUTION ENGINE
Computes and tracks cognitive resolution with partial credit accumulation.

Key innovation: Resolution is continuous, not binary.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from collections import deque


@dataclass
class ResolutionEvent:
    """A single resolution event with full context."""
    episode: int
    delta_dissonance: float  # Change in dissonance
    partial_credit: float     # Accumulated partial credit
    source: str               # What triggered resolution
    
    # Context
    difficulty: float = 0.5
    correctness: bool = False
    
    # Computed
    full_resolution: bool = False  # Threshold crossing
    wisdom_generated: float = 0.0


@dataclass
class ResolutionMemory:
    """
    Accumulated resolution memory.
    """
    total_partial_credit: float = 0.0
    events: List[ResolutionEvent] = field(default_factory=list)
    
    # Thresholds — NOTE: class default was 1.0 which was WRONG
    # 0.15 is the intended threshold (matching __init__ param)
    partial_credit_threshold: float = 0.20  # FIX (wisdom_gain): Raised to prevent spurious wisdom on easy episodes. Interview-quality deltas (0.08-0.10) now accumulate partial credit.
    
    def add_event(self, event: ResolutionEvent) -> Optional[float]:
        """
        Add resolution event. Returns wisdom amount if threshold crossed.
        """
        self.events.append(event)
        self.total_partial_credit += event.partial_credit
        
        # Check for threshold crossing
        if self.total_partial_credit >= self.partial_credit_threshold:
            # Convert accumulated partial credit to wisdom
            wisdom = self._convert_to_wisdom()
            self.total_partial_credit = 0.0  # Reset accumulator
            return wisdom
        
        return None
    
    def _convert_to_wisdom(self) -> float:
        """Convert accumulated partial credit to wisdom."""
        # FIX (wisdom_interview): Fixed from 0.8*threshold. Now uses full threshold as base.
        # With threshold at 0.10 and delta guard at 0.08, we get 2 resolution events per wisdom
        # (alternating ~0.085 and ~0.095 per event). Each crossing generates exactly 0.10.
        base = self.partial_credit_threshold
        
        # Streak bonus: capped at 0.02
        recent_successes = sum(1 for e in self.events[-10:] if e.correctness)
        streak_bonus = min(0.02, 0.005 * recent_successes)
        
        return base + streak_bonus
    
    def get_recent_resolution_rate(self, window: int = 20) -> float:
        """Compute recent resolution rate."""
        if len(self.events) < window:
            return 0.0
        
        recent = self.events[-window:]
        resolutions = sum(1 for e in recent if e.full_resolution)
        return resolutions / window


class ResolutionEngine:
    """
    Resolution computation with partial credit and streak tracking.
    """
    
    def __init__(self, partial_threshold: float = 0.20):  # FIX (wisdom_gain): Raised to 0.20 to prevent spurious wisdom on easy episodes
        self.memory = ResolutionMemory(partial_credit_threshold=partial_threshold)
        self.streak_counter: int = 0
        self.last_outcome: Optional[bool] = None
        
    def compute(
        self,
        episode: int,
        prev_dissonance: float,
        current_dissonance: float,
        correctness: bool,
        difficulty: float = 0.5,
        source: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute resolution event and update memory.
        
        Returns:
            dict with resolution_delta, partial_credit, wisdom_generated, streak
        """
        # Raw resolution: change in dissonance
        delta = prev_dissonance - current_dissonance
        
        # Streak tracking
        if correctness == self.last_outcome and correctness:
            self.streak_counter += 1
        else:
            self.streak_counter = 1 if correctness else 0
        self.last_outcome = correctness
        
        # Partial credit calculation
        # More credit for:
        # - Successful correctness
        # - Higher difficulty
        # - Breaking negative streaks
        
        base_credit = abs(delta) * 0.5  # Base from dissonance change
        
        if correctness:
            base_credit *= 1.5  # Bonus for success
            # Preserve the intent of the old guard — tiny changes should count less,
            # but they should not be erased entirely. Use a smooth ramp from 0 to 1
            # over the first 0.08 of dissonance reduction.
            signal_scale = min(1.0, max(0.0, abs(delta) / 0.08))
            base_credit *= signal_scale
            if delta < 0 and correctness:
                base_credit *= 0.5
        else:
            base_credit = 0.0
        
        difficulty_multiplier = 1.0 + (difficulty - 0.5) * 0.5  # Less generous scaling
        
        streak_bonus = 0.0
        if abs(delta) > 0.0 and correctness:
            streak_bonus = min(0.02, self.streak_counter * 0.005)  # 0.005 per streak, max 0.02
        
        partial_credit = base_credit * difficulty_multiplier + streak_bonus
        # FIX (wisdom_gain): Lowered cap from 0.20 to 0.12 so single events
        # generate less partial credit, preventing spurious wisdom generation
        partial_credit = np.clip(partial_credit, 0.0, 0.12)
        
        # Determine if full resolution achieved
        # Full resolution = meaningful dissonance reduction
        full_resolution = delta > 0.05 and correctness
        
        # Create event
        event = ResolutionEvent(
            episode=episode,
            delta_dissonance=delta,
            partial_credit=partial_credit,
            source=source,
            difficulty=difficulty,
            correctness=correctness,
            full_resolution=full_resolution
        )
        
        # Add to memory and check for wisdom generation
        wisdom = self.memory.add_event(event)
        
        return {
            "delta": delta,
            "partial_credit": partial_credit,
            "full_resolution": full_resolution,
            "wisdom_generated": wisdom if wisdom else 0.0,
            "streak": self.streak_counter,
            "total_partial_credit": self.memory.total_partial_credit,
            "resolution_rate": self.memory.get_recent_resolution_rate(),
        }
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Return resolution memory status."""
        return {
            "total_events": len(self.memory.events),
            "accumulated_partial": self.memory.total_partial_credit,
            "threshold": self.memory.partial_credit_threshold,
            "progress_to_wisdom": self.memory.total_partial_credit / self.memory.partial_credit_threshold,
            "current_streak": self.streak_counter,
            "recent_resolution_rate": self.memory.get_recent_resolution_rate(),
        }
