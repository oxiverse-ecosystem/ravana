"""
RAVANA v2 — GLOBAL WORKSPACE
Consciousness bottleneck: competitive broadcast system for inter-module coordination.

PRINCIPLE: Modules compete for attention. The winner broadcasts to all.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import random


@dataclass
class GWContent:
    """Content broadcast by the Global Workspace."""
    source: str             # which module produced this (e.g., "emotion", "belief")
    payload: dict           # arbitrary content
    urgency: float          # 0-1, how important this is
    valence: float          # emotional coloring
    timestamp: float
    episode: int


@dataclass
class GWConfig:
    """Configuration for the Global Workspace."""
    capacity: int = 7               # working memory buffer size
    broadcast_threshold: float = 0.3  # min urgency to broadcast
    decay_rate: float = 0.1         # how fast old broadcasts fade
    competition_noise: float = 0.05  # stochasticity in bid selection
    max_history: int = 500          # max bid history entries
    sleep_pressure_threshold: float = 1.5


class GlobalWorkspace:
    """
    Consciousness bottleneck — competitive broadcast system.

    Modules submit bids (content + urgency). The GW selects the
    highest-urgency bid, broadcasts it, and maintains a temporal
    buffer of recent broadcasts for context priming.

    This is the sole shared communication channel between modules.
    Modules still don't call each other directly.
    """

    def __init__(self, config: Optional[GWConfig] = None):
        self.config = config or GWConfig()

        # Current cycle's bids
        self._bids: List[GWContent] = []

        # Temporal buffer of recent broadcasts (working memory)
        self._buffer: List[GWContent] = []

        # Pressure tracking for sleep integration
        self._pressure: float = 0.0
        self._pressure_history: List[float] = []

        # Bid history for analysis
        self._bid_history: List[Dict[str, Any]] = []

        # Broadcast counter
        self._broadcast_count: int = 0

    # --- Bid submission ---

    def submit_bid(
        self,
        source: str,
        payload: dict,
        urgency: float,
        valence: float = 0.0,
        episode: int = 0,
    ) -> None:
        """
        Submit a bid to the Global Workspace.

        Args:
            source: Module name (e.g., "emotion", "belief", "meaning")
            payload: Arbitrary content to broadcast if this bid wins
            urgency: How important this is (0-1)
            valence: Emotional coloring (-1 to 1)
            episode: Current episode number
        """
        urgency = float(np.clip(urgency, 0.0, 1.0))
        valence = float(np.clip(valence, -1.0, 1.0))

        content = GWContent(
            source=source,
            payload=payload,
            urgency=urgency,
            valence=valence,
            timestamp=time.time(),
            episode=episode,
        )
        self._bids.append(content)

        # Record in history
        self._bid_history.append({
            "source": source,
            "urgency": urgency,
            "valence": valence,
            "episode": episode,
            "timestamp": content.timestamp,
        })
        if len(self._bid_history) > self.config.max_history:
            self._bid_history = self._bid_history[-self.config.max_history:]

    def clear_bids(self) -> None:
        """Clear current cycle's bids."""
        self._bids.clear()

    # --- Competition & broadcast ---

    def compete(self) -> Optional[GWContent]:
        """
        Select winning bid (highest urgency + noise).

        Returns the broadcast content, or None if no bid exceeds threshold.
        The winning bid is added to the temporal buffer.
        """
        if not self._bids:
            return None

        # Add noise to prevent deterministic selection
        noisy_bids = []
        for bid in self._bids:
            noise = random.gauss(0, self.config.competition_noise)
            noisy_urgency = bid.urgency + noise
            noisy_bids.append((bid, noisy_urgency))

        # Select winner
        ranked = sorted(noisy_bids, key=lambda x: x[1], reverse=True)
        winner, winner_score = ranked[0]

        # Check threshold
        if winner.urgency < self.config.broadcast_threshold:
            self.clear_bids()
            return None

        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
        competition_gap = max(0.0, winner_score - runner_up_score)
        competing_bids = max(0, len(self._bids) - 1)

        pressure_delta = (
            0.25
            + 0.35 * winner.urgency
            + 0.30 * competition_gap
            + 0.05 * competing_bids
        )
        pressure_delta = float(np.clip(pressure_delta, 0.0, 1.0))
        self.accumulate_pressure(pressure_delta)

        # Add to buffer (most recent first)
        self._buffer.insert(0, winner)
        if len(self._buffer) > self.config.capacity:
            self._buffer = self._buffer[:self.config.capacity]

        self._broadcast_count += 1

        # Clear bids for next cycle
        self.clear_bids()

        return winner

    # --- Buffer access ---

    def get_recent(self, k: int = 3) -> List[GWContent]:
        """Get k most recent broadcasts from the buffer."""
        return self._buffer[:k]

    def get_context_vector(self) -> np.ndarray:
        """
        Weighted average of recent broadcasts for context priming.

        Returns a 3D vector [valence, arousal_proxy, dominance_proxy]
        that can be used to prime other modules.
        """
        if not self._buffer:
            return np.array([0.0, 0.3, 0.5])  # neutral baseline

        # Weight by recency and urgency
        weights = []
        values = []
        for i, content in enumerate(self._buffer):
            recency_weight = np.exp(-self.config.decay_rate * i)
            weight = content.urgency * recency_weight
            weights.append(weight)
            values.append([
                content.valence,
                content.urgency,  # arousal proxy
                0.5,  # dominance neutral
            ])

        weights = np.array(weights)
        values = np.array(values)

        if weights.sum() < 1e-8:
            return np.array([0.0, 0.3, 0.5])

        weights = weights / weights.sum()
        return (values * weights[:, np.newaxis]).sum(axis=0)

    def get_active_sources(self) -> Dict[str, int]:
        """Count how many times each source has broadcast recently."""
        counts: Dict[str, int] = {}
        for content in self._buffer:
            counts[content.source] = counts.get(content.source, 0) + 1
        return counts

    # --- Pressure integration ---

    def accumulate_pressure(self, delta: float) -> None:
        """Accumulate pressure from competition intensity."""
        self._pressure = min(100.0, self._pressure + delta)
        self._pressure_history.append(self._pressure)
        if len(self._pressure_history) > self.config.max_history:
            self._pressure_history = self._pressure_history[-self.config.max_history:]

    def get_pressure(self) -> float:
        """Return the current accumulated pressure."""
        return self._pressure

    def should_sleep(self) -> bool:
        """Check if workspace pressure warrants consolidation."""
        return self._pressure >= self.config.sleep_pressure_threshold

    # --- Introspection ---

    def get_status(self) -> Dict[str, Any]:
        """Full workspace status."""
        return {
            "buffer_size": len(self._buffer),
            "buffer_capacity": self.config.capacity,
            "pending_bids": len(self._bids),
            "broadcast_count": self._broadcast_count,
            "pressure": self._pressure,
            "pressure_history": self._pressure_history[-3:],
            "sleep_pressure_threshold": self.config.sleep_pressure_threshold,
            "active_sources": self.get_active_sources(),
            "recent_urgencies": [b.urgency for b in self._buffer[:3]],
        }

    def get_bid_history(self) -> List[Dict[str, Any]]:
        """Return bid history for analysis."""
        return self._bid_history

    def get_buffer_snapshot(self) -> List[Dict[str, Any]]:
        """Serializable snapshot of current buffer."""
        return [
            {
                "source": c.source,
                "urgency": c.urgency,
                "valence": c.valence,
                "episode": c.episode,
                "payload_keys": list(c.payload.keys()),
            }
            for c in self._buffer
        ]
