"""
Hippocampal Buffer — Episodic Fact Memory with Pattern Completion
=================================================================
Implements a fast-learning, high-capacity, temporary buffer for episodic
factual assertions, inspired by hippocampal pattern completion.

Neuroscience grounding:
- Hippocampus performs pattern completion: from partial cue → full memory trace
- Fast learning, temporary storage, turn-gated decay
- Sleep consolidation transfers high-confidence facts to neocortical graph

Design:
- Stores (subject, predicate, object, turn_number, confidence) triples
- Pattern completion retrieval: query by subject cue
- Turn-gated decay: facts older than N turns naturally fade
- Reinforcement: facts referenced in user queries get strengthened
- Recall triggers route to this buffer instead of the graph
"""
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import time


@dataclass
class FactTriple:
    """A single episodic fact with metadata."""
    subject: str
    predicate: str
    object: str
    turn_number: int
    confidence: float = 0.8
    rehearsal_count: int = 1
    timestamp: float = field(default_factory=time.time)
    consolidated: bool = False  # True after sleep consolidation → graph edges
    # Temporal grounding (Phase 1): the real-world date of the session in which
    # the fact was stated, and the absolute date the fact REFERS to (resolved
    # from relative phrases like "yesterday"/"3 years ago" against session_date).
    session_date: Optional[datetime] = None
    absolute_date: Optional[datetime] = None
    # Knowledge-update bookkeeping (Phase 4): superseded facts are kept for
    # "you said X before but now Y" queries; the active one is returned first.
    superseded: bool = False
    # Source monitoring: True when the fact is a USER self-disclosure ("my cat
    # is pixel"). User facts stay in the buffer for episodic recall but are
    # NEVER drained into the neocortical world graph — the entity-keyed edge
    # would conflate "this user's cat" with "cats in general". They graduate
    # through UserModel.personal_facts instead.
    user_fact: bool = False


@dataclass
class HippocampalConfig:
    """Configuration for hippocampal buffer."""
    max_facts: int = 50  # Maximum episodes before oldest decay
    decay_turns: int = 50  # Turns before unrehearsed facts decay
    confidence_threshold: float = 0.6  # Min confidence for direct retrieval
    replay_batch_size: int = 10  # How many facts to replay during sleep


class HippocampalBuffer:
    """Fast-learning episodic fact buffer with pattern completion.

    Stores recent factual assertions and supports pattern-completion retrieval.
    """

    def __init__(self, config: Optional[HippocampalConfig] = None):
        self.config = config or HippocampalConfig()
        self.facts: Dict[str, List[FactTriple]] = defaultdict(list)  # subject → [facts]
        self._all_facts: List[FactTriple] = []  # Flat list for decay/consolidation
        self._turn_counter: int = 0
        self._recent_retrievals: Set[str] = set()

    def advance_turn(self):
        """Advance turn counter and apply decay to unrehearsed facts."""
        self._turn_counter += 1
        self._apply_decay()

    def store(self, subject: str, predicate: str, object: str,
              confidence: float = 0.8, aliases: Optional[List[str]] = None,
              session_date: Optional["datetime"] = None,
              absolute_date: Optional["datetime"] = None,
              user_fact: bool = False):
        """Store a factual assertion in the hippocampal buffer.

        If the same (subject, predicate, object) triple already exists,
        strengthen its confidence and rehearsal count.

        Args:
            subject: Primary subject key
            predicate: Relation predicate
            object: Object value
            confidence: Confidence score (0-1)
            aliases: Additional subject aliases to index this fact under
                     (e.g., content words from the user's statement)
            session_date: Real-world date of the session (Phase 1 temporal)
            absolute_date: Resolved absolute date the fact refers to
        """
        subj_lower = subject.lower().strip()
        obj_lower = object.lower().strip()
        pred_lower = predicate.lower().strip()

        # Build all keys: primary subject + any aliases
        all_keys = [subj_lower]
        if aliases:
            for alias in aliases:
                a = alias.lower().strip()
                if a and a != subj_lower and len(a) >= 2:
                    all_keys.append(a)

        # Check if this exact triple already exists under any key
        existing = None
        for key in all_keys:
            for fact in self.facts.get(key, []):
                if fact.predicate == pred_lower and fact.object == obj_lower:
                    existing = fact
                    break
            if existing:
                break

        if existing:
            # Strengthen existing memory
            existing.confidence = min(1.0, existing.confidence + 0.1)
            existing.rehearsal_count += 1
            existing.turn_number = self._turn_counter
            existing.timestamp = time.time()
            existing.user_fact = existing.user_fact or user_fact
            if session_date is not None:
                existing.session_date = session_date
            if absolute_date is not None:
                existing.absolute_date = absolute_date
            # Ensure it's indexed under all keys
            for key in all_keys:
                if existing not in self.facts.get(key, []):
                    self.facts[key].append(existing)
        else:
            # New episodic memory
            fact = FactTriple(
                subject=subj_lower,
                predicate=pred_lower,
                object=obj_lower,
                turn_number=self._turn_counter,
                confidence=confidence,
                session_date=session_date,
                absolute_date=absolute_date,
                user_fact=user_fact,
            )
            # Index under all keys for multi-alias lookup
            for key in all_keys:
                self.facts[key].append(fact)
            self._all_facts.append(fact)

        # Enforce capacity limit (LRU-like)
        if len(self._all_facts) > self.config.max_facts * 2:
            self._trim_oldest()

    def retrieve(self, subject: str) -> Optional[List[FactTriple]]:
        """Pattern completion: retrieve facts by subject cue.

        Returns all facts matching the subject with confidence > threshold.
        Strengthens retrieved facts (retrieval practice effect).
        """
        subj_lower = subject.lower().strip()
        subj_facts = self.facts.get(subj_lower, [])

        if not subj_facts:
            # Fuzzy match: try to find partial or substring matches
            for s, facts in self.facts.items():
                if subj_lower in s or s in subj_lower:
                    subj_facts = facts
                    break

        if not subj_facts:
            return None

        # Filter by confidence threshold and reinforce
        valid = []
        for fact in subj_facts:
            if fact.confidence >= self.config.confidence_threshold:
                # Deduplicate by checking if already in valid list
                if not any(v.subject == fact.subject and v.predicate == fact.predicate and v.object == fact.object for v in valid):
                    valid.append(fact)
                # Retrieval practice: strengthen
                fact.confidence = min(1.0, fact.confidence + 0.05)
                fact.rehearsal_count += 1
                self._recent_retrievals.add(subj_lower)

        return valid if valid else None

    def retrieve_any(self, cues: List[str]) -> Optional[List[FactTriple]]:
        """Cross-subject retrieval: search across ALL stored facts using
        multiple cue words. Returns facts where ANY cue word matches ANY
        part of the fact (subject, predicate, or object).

        This enables recall like "remember when you said my name is Pixel"
        where the cues are ["remember", "name", "pixel", "said"].
        """
        if not cues:
            return None

        matched_facts = []
        seen = set()  # Track (subject, predicate, object) to deduplicate
        reinforced = set()  # Track facts already strengthened this call

        for cue in cues:
            c = cue.lower().strip()
            if len(c) < 2:
                continue

            # Check direct key match
            if c in self.facts:
                for fact in self.facts[c]:
                    key = (fact.subject, fact.predicate, fact.object)
                    if key not in seen and fact.confidence >= self.config.confidence_threshold:
                        seen.add(key)
                        matched_facts.append(fact)

            # Check partial key match
            for stored_key, facts in self.facts.items():
                if c in stored_key or stored_key in c:
                    for fact in facts:
                        key = (fact.subject, fact.predicate, fact.object)
                        if key not in seen and fact.confidence >= self.config.confidence_threshold:
                            seen.add(key)
                            matched_facts.append(fact)

            # Check content match in stored facts' object field
            for fact in self._all_facts:
                if fact.confidence < self.config.confidence_threshold:
                    continue
                key = (fact.subject, fact.predicate, fact.object)
                if key in seen:
                    continue
                # Check if cue matches fact's object content
                if c in fact.object or fact.object in c:
                    seen.add(key)
                    matched_facts.append(fact)

        # Apply retrieval practice ONCE per matched fact (after dedup)
        for fact in matched_facts:
            if id(fact) not in reinforced:
                reinforced.add(id(fact))
                fact.confidence = min(1.0, fact.confidence + 0.05)
                fact.rehearsal_count += 1

        # Sort by confidence (highest first), then by recency
        matched_facts.sort(key=lambda f: (f.confidence, f.turn_number), reverse=True)
        return matched_facts[:5] if matched_facts else None

    # ── Phase 1: temporal / recency-aware retrieval ─────────────────────────
    def retrieve_dated(self, subject: str) -> Optional[List[FactTriple]]:
        """Retrieve facts for a subject that carry an absolute_date, sorted
        chronologically (oldest first). Used by temporal 'when did X' queries."""
        facts = self.retrieve(subject) or []
        dated = [f for f in facts if f.absolute_date is not None]
        dated.sort(key=lambda f: f.absolute_date)
        return dated or None

    def retrieve_between(self, subject: str, start, end) -> List[FactTriple]:
        """Facts about subject whose absolute_date falls in [start, end]."""
        out = []
        for f in (self.retrieve(subject) or []):
            if f.absolute_date is not None and start <= f.absolute_date <= end:
                out.append(f)
        out.sort(key=lambda f: f.absolute_date)
        return out

    def retrieve_latest(self, subject: str,
                        predicate: Optional[str] = None) -> Optional[FactTriple]:
        """Return the most recent (by absolute_date, then turn_number) ACTIVE
        fact for a subject — Phase 4 recency-wins for knowledge updates."""
        cands = [f for f in (self.retrieve(subject) or []) if not f.superseded]
        if predicate:
            cands = [f for f in cands if f.predicate == predicate.lower().strip()]
        if not cands:
            return None
        return max(cands, key=lambda f: (
            f.absolute_date or datetime.min, f.turn_number))

    def query(self, subject: str, predicate: str) -> Optional[str]:
        """Direct query: get the object for a (subject, predicate) pair."""
        subj_lower = subject.lower().strip()
        pred_lower = predicate.lower().strip()
        subj_facts = self.facts.get(subj_lower, [])

        for fact in subj_facts:
            if fact.predicate == pred_lower:
                # Retrieval practice
                fact.confidence = min(1.0, fact.confidence + 0.05)
                fact.rehearsal_count += 1
                return fact.object

        return None

    def get_facts_for_subject(self, subject: str) -> List[FactTriple]:
        """Get all facts about a subject without reinforcement."""
        subj_lower = subject.lower().strip()
        return self.facts.get(subj_lower, [])

    def get_consolidation_candidates(self) -> List[FactTriple]:
        """Get high-confidence facts ready for sleep consolidation."""
        candidates = []
        for fact in self._all_facts:
            if (fact.confidence >= 0.7 and fact.rehearsal_count >= 2
                    and not fact.consolidated):
                candidates.append(fact)
        # Sort by confidence * rehearsal, take top N
        candidates.sort(key=lambda f: f.confidence * f.rehearsal_count, reverse=True)
        return candidates[:self.config.replay_batch_size]

    def mark_consolidated(self, fact: FactTriple):
        """Mark a fact as consolidated (transferred to long-term memory)."""
        fact.consolidated = True

    def _apply_decay(self):
        """Decay unrehearsed facts that haven't been retrieved recently."""
        to_remove = []
        for i, fact in enumerate(self._all_facts):
            turns_ago = self._turn_counter - fact.turn_number
            if turns_ago > self.config.decay_turns and fact.rehearsal_count <= 2:
                fact.confidence *= 0.9  # Gradual decay
                if fact.confidence < 0.2:
                    to_remove.append(i)

        # Remove decayed facts (reverse order to maintain indices)
        for idx in sorted(to_remove, reverse=True):
            fact = self._all_facts[idx]
            subj_facts = self.facts.get(fact.subject, [])
            if fact in subj_facts:
                subj_facts.remove(fact)
            self._all_facts.pop(idx)

    def _trim_oldest(self):
        """Remove oldest facts when capacity is exceeded."""
        if len(self._all_facts) <= self.config.max_facts:
            return
        # Sort by turn_number (oldest first) and confidence (lowest first)
        self._all_facts.sort(key=lambda f: (f.turn_number, f.confidence))
        to_remove = self._all_facts[:len(self._all_facts) - self.config.max_facts]
        for fact in to_remove:
            subj_facts = self.facts.get(fact.subject, [])
            if fact in subj_facts:
                subj_facts.remove(fact)
            self._all_facts.remove(fact)

    def get_state(self) -> Dict:
        """Serialize state for saving/loading."""
        return {
            'facts': [
                {
                    'subject': f.subject,
                    'predicate': f.predicate,
                    'object': f.object,
                    'turn_number': f.turn_number,
                    'confidence': f.confidence,
                    'rehearsal_count': f.rehearsal_count,
                    'consolidated': f.consolidated,
                    'session_date': (f.session_date.isoformat()
                                     if f.session_date else None),
                    'absolute_date': (f.absolute_date.isoformat()
                                      if f.absolute_date else None),
                    'superseded': f.superseded,
                    'user_fact': f.user_fact,
                }
                for f in self._all_facts
            ],
            'turn_counter': self._turn_counter,
        }

    def set_state(self, state: Dict):
        """Restore state from serialized data."""
        self._all_facts = []
        self.facts.clear()
        for fd in state.get('facts', []):
            fact = FactTriple(
                subject=fd['subject'],
                predicate=fd['predicate'],
                object=fd['object'],
                turn_number=fd['turn_number'],
                confidence=fd.get('confidence', 0.8),
                rehearsal_count=fd.get('rehearsal_count', 1),
                timestamp=time.time(),
                consolidated=fd.get('consolidated', False),
                session_date=self._parse_iso(fd.get('session_date')),
                absolute_date=self._parse_iso(fd.get('absolute_date')),
                superseded=fd.get('superseded', False),
                user_fact=fd.get('user_fact', False),
            )
            self._all_facts.append(fact)
            self.facts[fact.subject].append(fact)
        self._turn_counter = state.get('turn_counter', 0)

    @staticmethod
    def _parse_iso(val) -> Optional["datetime"]:
        if not val:
            return None
        try:
            return datetime.fromisoformat(val)
        except Exception:
            return None
