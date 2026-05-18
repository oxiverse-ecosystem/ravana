"""
RAVANA v2 — Human-like Memory Architecture
Implements Episodic, Semantic, and Working Memory for Level 3 AGI.

Based on LIDA cognitive cycles and Bayesian belief updating.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

@dataclass
class MemoryTrace:
    """A single episodic memory unit."""
    timestamp: float
    episode: int
    content: Dict[str, Any]
    dissonance_at_time: float
    identity_at_time: float
    tags: List[str] = field(default_factory=list)
    salience: float = 0.5  # Initial emotional/importance weight

class EpisodicMemory:
    """
    Short-to-medium term storage for cognitive events.
    Used for 'Dream Sabotage' and 'Reflective Learning'.
    """
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.traces: List[MemoryTrace] = []
        
    def record(self, trace: MemoryTrace):
        """Add a new trace, maintaining capacity via salience-weighted decay."""
        self.traces.append(trace)
        if len(self.traces) > self.capacity:
            # Drop lowest salience trace
            self.traces.sort(key=lambda x: x.salience, reverse=True)
            self.traces = self.traces[:self.capacity]

    def retrieve_by_dissonance(self, threshold: float = 0.7) -> List[MemoryTrace]:
        """Find past events where dissonance was high (prime candidates for reflection)."""
        return [t for t in self.traces if t.dissonance_at_time >= threshold]

class SemanticMemory:
    """
    Long-term storage for generalized knowledge and social norms.
    Represented as a Bayesian Knowledge Graph.
    """
    def __init__(self):
        # Nodes: concepts/norms, Edges: conditional probabilities/confidences
        self.knowledge_graph: Dict[str, Dict[str, float]] = {
            "fairness": {"weight": 0.8, "confidence": 0.9},
            "honesty": {"weight": 0.9, "confidence": 0.85},
            "growth": {"weight": 0.7, "confidence": 0.5}
        }
        self.history: List[Dict] = []

    def update_norm(self, norm_name: str, delta: float, confidence_shift: float):
        """Slowly update a long-term commitment (Identity Strength)."""
        if norm_name in self.knowledge_graph:
            current = self.knowledge_graph[norm_name]
            current["weight"] = np.clip(current["weight"] + delta, 0.0, 1.0)
            current["confidence"] = np.clip(current["confidence"] + confidence_shift, 0.0, 1.0)
            self.history.append({"norm": norm_name, "delta": delta, "time": time.time()})

class WorkingMemory:
    """
    The 'Global Workspace' (GW) buffer.
    Limited capacity (System 1/2 threshold).
    """
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.current_focus: List[Dict[str, Any]] = []

    def broadcast(self, signals: List[Dict[str, Any]]):
        """Select top-k signals based on bid (salience + dissonance + novelty)."""
        # Sort by bid (assumed to be in signal dict)
        sorted_signals = sorted(signals, key=lambda x: x.get('bid', 0.0), reverse=True)
        self.current_focus = sorted_signals[:self.capacity]
        return self.current_focus

class RavanaMemorySystem:
    """Integrated Memory Controller."""
    def __init__(self):
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.working = WorkingMemory()
        
    def process_step(self, episode_data: Dict[str, Any], state_snapshot: Dict[str, float]):
        """Integrate new data into memory layers."""
        # 1. Create episodic trace
        trace = MemoryTrace(
            timestamp=time.time(),
            episode=episode_data.get('episode', 0),
            content=episode_data,
            dissonance_at_time=state_snapshot.get('dissonance', 0.0),
            identity_at_time=state_snapshot.get('identity', 0.0),
            salience=state_snapshot.get('dissonance', 0.5) * 1.5 # Dissonance drives salience
        )
        self.episodic.record(trace)
        
        # 2. Update semantic norms if dissonance is low (reinforcement) or high (revision)
        # Placeholder for complex Bayesian update logic
        if trace.dissonance_at_time < 0.2:
            self.semantic.update_norm("growth", 0.01, 0.005)
            
    def get_context_for_decision(self) -> Dict[str, Any]:
        """Retrieve relevant past experiences for current deliberation (System 2)."""
        high_d_events = self.episodic.retrieve_by_dissonance(0.6)
        return {
            "past_failures": high_d_events[:3],
            "core_norms": self.semantic.knowledge_graph,
            "working_focus": self.working.current_focus
        }
