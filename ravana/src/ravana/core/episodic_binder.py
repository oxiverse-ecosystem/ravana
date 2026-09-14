"""
EpisodicBinder — CA3-style fast concept-pair binding
=====================================================

Neuroscience grounding:
- Hippocampal CA3 is an autoassociative network that binds two arbitrary
  concepts into a single episodic trace after ONE co-occurrence (complementary
  learning systems: McClelland et al. 1995; O'Reilly & Norman 2002).
- Pattern completion: a partial cue (one concept) retrieves the bound partner.
- Fast learning: no weight update needed — a single exposure writes the trace.

Design:
- Each bound pair is an EpisodicPair (concept_a, concept_b, context_vector,
  turn_number, confidence, rehearsal_count, consolidated).
- The binding is a composite key (a,b) so either concept can cue retrieval.
- Retrieval is content-addressable: given a fragment (one concept), return
  the bound partner with the strongest association.
- Sleep consolidation replays high-confidence pairs into the ConceptGraph as
  typed edges (episodic → semantic transfer).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import numpy as np


@dataclass
class EpisodicPair:
    """A single bound concept-pair from one exposure."""
    concept_a: str
    concept_b: str
    context: str = ""  # the original utterance that created the binding
    turn_number: int = 0
    confidence: float = 0.8
    rehearsal_count: int = 1
    timestamp: float = field(default_factory=time.time)
    consolidated: bool = False  # True after sleep replay → graph edge
    # VAD of the encoding context (for emotional tagging of the trace)
    valence: float = 0.0
    arousal: float = 0.0
    # Source monitoring: True when the pair is a USER self-disclosure
    # ("my cat is pixel" → cat↔pixel). User pairs stay episodic; they do
    # NOT drain into the world graph as entity-keyed edges.
    user_fact: bool = False


@dataclass
class EpisodicBinderConfig:
    """Configuration for the episodic binder."""
    max_pairs: int = 200  # maximum bound pairs before oldest decay
    decay_turns: int = 80  # turns before unrehearsed pairs decay
    confidence_threshold: float = 0.5  # min confidence for retrieval
    replay_batch_size: int = 15  # how many pairs to replay per sleep cycle
    min_rehearsal_for_consolidate: int = 2  # min rehearsals before sleep promotes


class EpisodicBinder:
    """CA3-style fast concept-pair binding with pattern-completion retrieval.

    Stores concept-pair associations from a single exposure and supports
    content-addressable retrieval: given one concept, return the bound partner.
    """

    def __init__(self, config: Optional[EpisodicBinderConfig] = None):
        self.config = config or EpisodicBinderConfig()
        # concept → list of EpisodicPair (each pair indexed under BOTH concepts)
        self.pairs: Dict[str, List[EpisodicPair]] = defaultdict(list)
        self._all_pairs: List[EpisodicPair] = []  # flat list for decay/consolidation
        self._turn_counter: int = 0

    def advance_turn(self):
        """Advance turn counter and apply decay to unrehearsed pairs."""
        self._turn_counter += 1
        self._apply_decay()

    def bind(self, concept_a: str, concept_b: str, context: str = "",
             confidence: float = 0.8, user_fact: bool = False,
             valence: float = 0.0, arousal: float = 0.0) -> Optional[EpisodicPair]:
        """Bind two concepts into a single episodic trace after ONE exposure.

        If the same pair already exists (in either direction), strengthen it.
        Otherwise create a new trace. This is the fast-learning property of
        CA3: a single co-occurrence writes the binding.

        Args:
            concept_a: first concept (order-independent)
            concept_b: second concept
            context: the utterance that created the binding
            confidence: initial confidence (0-1)
            user_fact: True if this is a user self-disclosure
            valence: emotional valence of the encoding context
            arousal: emotional arousal of the encoding context
        """
        a = concept_a.lower().strip()
        b = concept_b.lower().strip()
        if not a or not b or a == b:
            return None

        # Check if this pair already exists (either direction)
        existing = self._find_pair(a, b)
        if existing:
            # Strengthen existing trace (rehearsal)
            existing.confidence = min(1.0, existing.confidence + 0.1)
            existing.rehearsal_count += 1
            existing.turn_number = self._turn_counter
            existing.timestamp = time.time()
            existing.user_fact = existing.user_fact or user_fact
            return existing

        # New episodic trace — one-shot learning
        pair = EpisodicPair(
            concept_a=a,
            concept_b=b,
            context=context[:300],
            turn_number=self._turn_counter,
            confidence=confidence,
            valence=valence,
            arousal=arousal,
            user_fact=user_fact,
        )
        # Index under both concepts for bidirectional retrieval
        self.pairs[a].append(pair)
        self.pairs[b].append(pair)
        self._all_pairs.append(pair)

        # Enforce capacity
        if len(self._all_pairs) > self.config.max_pairs * 2:
            self._trim_oldest()
        return pair

    def _find_pair(self, a: str, b: str) -> Optional[EpisodicPair]:
        """Find an existing pair binding a and b (either direction)."""
        for pair in self.pairs.get(a, []):
            if (pair.concept_a == b or pair.concept_b == b):
                return pair
        return None

    def retrieve(self, cue: str) -> List[Tuple[str, EpisodicPair]]:
        """Pattern completion: given a cue concept, return bound partners.

        Returns list of (partner_concept, pair) sorted by confidence.
        Strengthens retrieved pairs (retrieval practice effect).
        """
        cue_lower = cue.lower().strip()
        if not cue_lower:
            return []

        # Direct match
        matches = self.pairs.get(cue_lower, [])

        # Also try partial/substring match for robustness
        if not matches:
            for key, pairs in self.pairs.items():
                if cue_lower in key or key in cue_lower:
                    matches = pairs
                    break

        if not matches:
            return []

        # Build (partner, pair) list, filtering by confidence
        results = []
        seen = set()
        for pair in matches:
            if pair.confidence < self.config.confidence_threshold:
                continue
            # Determine the partner (the side that is NOT the cue)
            if pair.concept_a == cue_lower:
                partner = pair.concept_b
            elif pair.concept_b == cue_lower:
                partner = pair.concept_a
            else:
                # partial match — pick the side closer to the cue
                partner = pair.concept_a if pair.concept_b == cue_lower else pair.concept_b

            if partner in seen:
                continue
            seen.add(partner)
            results.append((partner, pair))

            # Retrieval practice: strengthen
            pair.confidence = min(1.0, pair.confidence + 0.05)
            pair.rehearsal_count += 1

        # Sort by confidence (highest first)
        results.sort(key=lambda x: x[1].confidence, reverse=True)
        return results

    def retrieve_with_context(self, cue: str) -> List[Dict]:
        """Retrieve bound partners with full context for reconstruction.

        Returns list of dicts with partner, confidence, context, turn_number.
        """
        raw = self.retrieve(cue)
        return [
            {
                "partner": partner,
                "confidence": pair.confidence,
                "context": pair.context,
                "turn_number": pair.turn_number,
                "rehearsal_count": pair.rehearsal_count,
                "consolidated": pair.consolidated,
                "user_fact": pair.user_fact,
            }
            for partner, pair in raw
        ]

    def get_consolidation_candidates(self) -> List[EpisodicPair]:
        """Get high-confidence pairs ready for sleep consolidation."""
        candidates = []
        for pair in self._all_pairs:
            if (pair.confidence >= 0.6
                    and pair.rehearsal_count >= self.config.min_rehearsal_for_consolidate
                    and not pair.consolidated):
                candidates.append(pair)
        candidates.sort(key=lambda p: p.confidence * p.rehearsal_count, reverse=True)
        return candidates[:self.config.replay_batch_size]

    def mark_consolidated(self, pair: EpisodicPair):
        """Mark a pair as consolidated (replayed into the concept graph)."""
        pair.consolidated = True

    def _apply_decay(self):
        """Decay unrehearsed pairs that haven't been retrieved recently."""
        to_remove = []
        for i, pair in enumerate(self._all_pairs):
            turns_ago = self._turn_counter - pair.turn_number
            if turns_ago > self.config.decay_turns and pair.rehearsal_count <= 2:
                pair.confidence *= 0.9
                if pair.confidence < 0.15:
                    to_remove.append(i)

        for idx in sorted(to_remove, reverse=True):
            pair = self._all_pairs[idx]
            # Remove from both concept indexes
            for key in [pair.concept_a, pair.concept_b]:
                bucket = self.pairs.get(key, [])
                if pair in bucket:
                    bucket.remove(pair)
            self._all_pairs.pop(idx)

    def _trim_oldest(self):
        """Remove oldest pairs when capacity exceeded."""
        if len(self._all_pairs) <= self.config.max_pairs:
            return
        self._all_pairs.sort(key=lambda p: (p.turn_number, p.confidence))
        to_remove = self._all_pairs[:len(self._all_pairs) - self.config.max_pairs]
        for pair in to_remove:
            for key in [pair.concept_a, pair.concept_b]:
                bucket = self.pairs.get(key, [])
                if pair in bucket:
                    bucket.remove(pair)
            self._all_pairs.remove(pair)

    def get_state(self) -> Dict:
        """Serialize state for saving/loading."""
        return {
            'pairs': [
                {
                    'concept_a': p.concept_a,
                    'concept_b': p.concept_b,
                    'context': p.context,
                    'turn_number': p.turn_number,
                    'confidence': p.confidence,
                    'rehearsal_count': p.rehearsal_count,
                    'consolidated': p.consolidated,
                    'valence': p.valence,
                    'arousal': p.arousal,
                    'user_fact': p.user_fact,
                }
                for p in self._all_pairs
            ],
            'turn_counter': self._turn_counter,
        }

    def set_state(self, state: Dict):
        """Restore state from serialized data."""
        self._all_pairs = []
        self.pairs.clear()
        for pd in state.get('pairs', []):
            pair = EpisodicPair(
                concept_a=pd['concept_a'],
                concept_b=pd['concept_b'],
                context=pd.get('context', ''),
                turn_number=pd['turn_number'],
                confidence=pd.get('confidence', 0.8),
                rehearsal_count=pd.get('rehearsal_count', 1),
                consolidated=pd.get('consolidated', False),
                valence=pd.get('valence', 0.0),
                arousal=pd.get('arousal', 0.0),
                user_fact=pd.get('user_fact', False),
            )
            self._all_pairs.append(pair)
            self.pairs[pair.concept_a].append(pair)
            self.pairs[pair.concept_b].append(pair)
        self._turn_counter = state.get('turn_counter', 0)
