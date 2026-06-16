"""
Global Workspace for RAVANA.

The Global Workspace Theory (Baars, 1988; Dehaene, 2014) posits a
"theater of consciousness" where information is broadcast to all
specialized modules. Only high-priority/salient items enter the workspace.

Capacity-limited: ~7 items (Miller's working memory limit).
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from collections import deque


@dataclass
class GWConfig:
    """Configuration for global workspace."""
    capacity: int = 7
    broadcast_threshold: float = 0.3
    decay_rate: float = 0.1


@dataclass
class WorkspaceItem:
    """Item in the global workspace."""
    source: str
    payload: Dict[str, Any]
    urgency: float
    valence: float
    episode: int
    timestamp: float = field(default_factory=lambda: __import__('time').time())


class GlobalWorkspace:
    """Global Workspace: limited-capacity broadcast hub for cognitive content."""

    def __init__(self, config: Optional[GWConfig] = None):
        self.config = config or GWConfig()
        self.contents: List[WorkspaceItem] = []
        self.broadcast_log: deque = deque(maxlen=100)

    def submit_bid(self, source: str, payload: Dict[str, Any],
                   urgency: float, valence: float, episode: int):
        """Submit a bid for workspace entry."""
        item = WorkspaceItem(
            source=source,
            payload=payload,
            urgency=urgency,
            valence=valence,
            episode=episode)
        self.contents.append(item)

    def compete(self) -> List[WorkspaceItem]:
        """Run competition: select top items by urgency*valence for broadcast."""
        if not self.contents:
            return []

        # Score items
        for item in self.contents:
            item.score = item.urgency * (1.0 + abs(item.valence))

        # Sort by score descending
        self.contents.sort(key=lambda x: x.score, reverse=True)

        # Select winners above threshold
        winners = [item for item in self.contents[:self.config.capacity]
                   if item.score >= self.config.broadcast_threshold]

        # Log broadcast
        for item in winners:
            self.broadcast_log.append({
                'source': item.source,
                'episode': item.episode,
                'score': item.score,
                'payload_keys': list(item.payload.keys())})

        # Decay non-winners and remove old
        self.contents = [item for item in self.contents
                         if item.score >= self.config.broadcast_threshold * 0.5]

        return winners

    def get_recent_broadcasts(self, n: int = 10) -> List[Dict]:
        """Get recent broadcast log."""
        return list(self.broadcast_log)[-n:]