import numpy as np
import random
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass, field


@dataclass
class ReplayExperience:
    pair: Tuple[str, str]
    context: str
    weight: float
    timestamp: float
    priority: float = 1.0


class HippocampalReplay:
    """Hippocampal replay for offline consolidation (roadmap #10).

    Neuroscience basis:
    - NREM sleep: sharp-wave ripples replay recent experiences forward
    - REM sleep: recombines experiences into new generalizations
    - Reverse replay enables causal inference
    - Priority sampling: higher error = more replay

    Three-phase sleep cycle:
    1. NREM: forward replay of recent experiences
    2. Interleaved: mix new with old consolidated memories
    3. Pruning: weaken connections below threshold
    """

    def __init__(self, capacity: int = 200):
        self.buffer: deque = deque(maxlen=capacity)
        self.consolidated: Dict[str, float] = {}
        self.total_replays: int = 0

    def add_experience(self, pair: Tuple[str, str], context: str = "",
                       weight: float = 0.5, priority: float = 1.0):
        exp = ReplayExperience(
            pair=pair, context=context, weight=weight,
            timestamp=time.time(), priority=priority,
        )
        self.buffer.append(exp)

    def _sample_by_priority(self, count: int) -> List[ReplayExperience]:
        if not self.buffer:
            return []
        priorities = np.array([e.priority for e in self.buffer])
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), size=min(count, len(self.buffer)),
                                   replace=False, p=probs)
        return [self.buffer[i] for i in indices]

    def run_nrem_cycle(self, replay_count: int = 100) -> List[ReplayExperience]:
        recent = list(self.buffer)[-replay_count:]
        for exp in recent:
            key = f"{exp.pair[0]}->{exp.pair[1]}"
            self.consolidated[key] = self.consolidated.get(key, 0) + exp.weight * 0.1
        self.total_replays += len(recent)
        return recent

    def run_interleaved_cycle(self, sample_count: int = 50, mix_ratio: float = 0.3) -> List[ReplayExperience]:
        if not self.consolidated:
            return []

        new_samples = self._sample_by_priority(sample_count)
        old_keys = list(self.consolidated.keys())
        old_count = int(sample_count * mix_ratio)
        old_samples = random.sample(old_keys, min(old_count, len(old_keys)))
        from .curiosity import CuriosityEngine

        result: List[ReplayExperience] = list(new_samples)
        for key in old_samples:
            parts = key.split("->")
            if len(parts) == 2:
                exp = ReplayExperience(
                    pair=(parts[0], parts[1]),
                    context="consolidated",
                    weight=self.consolidated[key],
                    timestamp=time.time(),
                    priority=self.consolidated[key],
                )
                result.append(exp)
                self.consolidated[key] += 0.05
        self.total_replays += len(result)
        return result

    def prune(self, threshold: float = 0.1) -> int:
        before = len(self.consolidated)
        self.consolidated = {k: v for k, v in self.consolidated.items() if v >= threshold}
        return before - len(self.consolidated)

    def sleep_cycle(self, replay_count: int = 100,
                    interleave_count: int = 50,
                    prune_threshold: float = 0.1) -> Dict[str, int]:
        nrem = self.run_nrem_cycle(replay_count)
        interleaved = self.run_interleaved_cycle(interleave_count)
        pruned = self.prune(prune_threshold)
        return {
            'nrem_replays': len(nrem),
            'interleaved_replays': len(interleaved),
            'pruned': pruned,
            'total_consolidated': len(self.consolidated),
        }

    def get_state(self) -> dict:
        return {
            'buffer': [(e.pair, e.context, e.weight, e.timestamp, e.priority) for e in self.buffer],
            'consolidated': dict(self.consolidated),
            'total_replays': self.total_replays,
        }

    def set_state(self, state: dict):
        self.buffer = deque(
            [ReplayExperience(pair=tuple(p), context=c, weight=w,
                              timestamp=t, priority=pr)
             for p, c, w, t, pr in state.get('buffer', [])],
            maxlen=self.buffer.maxlen if hasattr(self, 'buffer') else 200,
        )
        self.consolidated = dict(state.get('consolidated', {}))
        self.total_replays = state.get('total_replays', 0)
