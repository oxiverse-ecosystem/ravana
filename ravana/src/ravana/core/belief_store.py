"""
Belief Store - Tracks asserted beliefs and contradictions with multi-user support.
"""
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class UserBeliefProfile:
    """Belief profile for a specific user."""
    user_id: str
    beliefs: Dict[Tuple[str, str], Tuple[str, float, int]] = field(default_factory=dict)
    # (subject, predicate) -> (value, confidence, turn)
    contradictions: List[Tuple[Tuple, Tuple, int]] = field(default_factory=list)
    # [(old_triple, new_triple, turn)]
    resolution_history: Dict[str, str] = field(default_factory=dict)
    # triple_str -> "accept_new" | "reject_new" | "both"
    turn_num: int = 0


@dataclass
class BeliefConfig:
    """Configuration for belief store."""
    recency_decay: float = 0.1
    min_confidence_threshold: float = 0.1


class BeliefStore:
    """
    Multi-user Belief Store.

    Tracks asserted belief triples per user and detects contradictions.
    Supports cross-referencing beliefs when users discuss the same topic.
    """

    def __init__(self, config: Optional[BeliefConfig] = None):
        self.config = config or BeliefConfig()
        self.users: Dict[str, UserBeliefProfile] = {}
        self.current_user: Optional[str] = None
        self.global_turn = 0

    def set_user(self, user_id: str):
        """Set the active user for belief operations."""
        if user_id not in self.users:
            self.users[user_id] = UserBeliefProfile(user_id=user_id)
        self.current_user = user_id

    def get_user_profile(self, user_id: str) -> UserBeliefProfile:
        """Get or create a user's belief profile."""
        if user_id not in self.users:
            self.users[user_id] = UserBeliefProfile(user_id=user_id)
        return self.users[user_id]

    def advance_turn(self):
        """Advance turn counter for all users."""
        self.global_turn += 1
        for profile in self.users.values():
            profile.turn_num += 1

    def assert_belief(self, subject_id: str, predicate: str,
                      value: str, confidence: float = 0.8,
                      user_id: Optional[str] = None):
        """Assert a belief for the current or specified user."""
        uid = user_id or self.current_user or "default"
        profile = self.get_user_profile(uid)
        key = (subject_id, predicate)
        profile.beliefs[key] = (value, confidence, profile.turn_num)
        profile.turn_num += 1

    def query_belief(self, subject_id: str, predicate: str,
                     user_id: Optional[str] = None) -> Optional[Tuple[str, float, int]]:
        """Query a belief for the current or specified user."""
        uid = user_id or self.current_user or "default"
        if uid in self.users:
            return self.users[uid].beliefs.get((subject_id, predicate))
        return None

    def detect_contradiction(self, subject_id: str, predicate: str,
                              new_value: str, user_id: Optional[str] = None) -> Optional[Tuple]:
        """Detect if a new belief contradicts an existing one."""
        uid = user_id or self.current_user or "default"
        if uid not in self.users:
            return None
        profile = self.users[uid]
        existing = profile.beliefs.get((subject_id, predicate))
        if existing and existing[0] != new_value:
            old_triple = (subject_id, predicate, existing[0])
            new_triple = (subject_id, predicate, new_value)
            profile.contradictions.append((old_triple, new_triple, profile.turn_num))
            return (existing, new_value, existing[2])
        return None

    def resolve_contradiction(self, old_triple: Tuple, new_triple: Tuple,
                               choice: str, user_id: Optional[str] = None):
        """Record user's resolution choice for a contradiction."""
        uid = user_id or self.current_user or "default"
        if uid in self.users:
            profile = self.users[uid]
            profile.contradictions.append((old_triple, new_triple, profile.turn_num))
            key = str(new_triple)
            profile.resolution_history[key] = choice

    def reconcile(self, user_id: Optional[str] = None) -> Dict[Tuple[str, str], Tuple[str, float, int]]:
        """Resolve contradictions: pick winner by confidence * recency decay."""
        uid = user_id or self.current_user or "default"
        if uid not in self.users:
            return {}
        profile = self.users[uid]

        resolved: Dict[Tuple[str, str], Tuple[str, float, int]] = {}
        groups: Dict[Tuple[str, str], List[Tuple[str, float, int]]] = {}

        for old_triple, new_triple, c_turn in profile.contradictions:
            subj, pred = old_triple[0], old_triple[1]
            key = (subj, pred)
            old_val, new_val = old_triple[2], new_triple[2]
            old_conf = new_conf = 0.0
            cur = profile.beliefs.get(key)
            if cur:
                if cur[0] == old_val:
                    old_conf = cur[1]
                elif cur[0] == new_val:
                    new_conf = cur[1]
            groups.setdefault(key, []).append((old_val, old_conf if old_conf > 0 else 0.5, c_turn))
            groups.setdefault(key, []).append((new_val, new_conf if new_conf > 0 else 0.5, c_turn))

        for key, candidates in groups.items():
            seen = set()
            unique = []
            for c in candidates:
                if c[0] not in seen:
                    seen.add(c[0])
                    unique.append(c)
            if len(unique) < 2:
                continue

            def decay_score(vc: Tuple) -> float:
                _, conf, turn = vc
                recency = 1.0 / (1.0 + (profile.turn_num - turn) * self.config.recency_decay)
                return conf * recency

            winner = max(unique, key=decay_score)
            profile.beliefs[key] = (winner[0], winner[1], profile.turn_num)
            resolved[key] = profile.beliefs[key]

        return resolved

    def cross_reference_users(self, subject_id: str, predicate: str,
                               users: List[str]) -> Dict[str, Tuple[str, float, int]]:
        """
        Cross-reference beliefs across multiple users for the same subject/predicate.
        Returns dict of user_id -> (value, confidence, turn).
        """
        results = {}
        for uid in users:
            if uid in self.users:
                belief = self.users[uid].beliefs.get((subject_id, predicate))
                if belief:
                    results[uid] = belief
        return results

    def find_agreement(self, subject_id: str, predicate: str,
                        users: List[str]) -> Optional[str]:
        """Find if multiple users agree on a belief (majority vote)."""
        beliefs = self.cross_reference_users(subject_id, predicate, users)
        if not beliefs:
            return None
        # Count values
        counts = defaultdict(int)
        for uid, (value, conf, turn) in beliefs.items():
            counts[value] += 1
        # Return majority (or highest confidence if tied)
        if counts:
            return max(counts.items(), key=lambda x: x[1])[0]
        return None

    def find_disagreement(self, subject_id: str, predicate: str,
                           users: List[str]) -> List[Tuple[str, str]]:
        """Find pairs of users who disagree on a belief."""
        beliefs = self.cross_reference_users(subject_id, predicate, users)
        disagreements = []
        user_list = list(beliefs.keys())
        for i, u1 in enumerate(user_list):
            for u2 in user_list[i+1:]:
                if beliefs[u1][0] != beliefs[u2][0]:
                    disagreements.append((u1, u2))
        return disagreements

    def get_state(self, user_id: Optional[str] = None) -> Dict:
        """Get serialized state for a specific user or all users."""
        if user_id:
            if user_id in self.users:
                p = self.users[user_id]
                return {
                    'user_id': p.user_id,
                    'beliefs': p.beliefs,
                    'contradictions': p.contradictions,
                    'resolution_history': p.resolution_history,
                    'turn_num': p.turn_num,
                }
            return {}
        # All users
        return {
            'global_turn': self.global_turn,
            'users': {
                uid: {
                    'user_id': p.user_id,
                    'beliefs': p.beliefs,
                    'contradictions': p.contradictions,
                    'resolution_history': p.resolution_history,
                    'turn_num': p.turn_num,
                }
                for uid, p in self.users.items()
            }
        }

    def set_state(self, state: Dict):
        """Load state for all users."""
        self.global_turn = state.get('global_turn', 0)
        for uid, pdata in state.get('users', {}).items():
            profile = UserBeliefProfile(user_id=uid)
            profile.beliefs = pdata.get('beliefs', {})
            profile.contradictions = pdata.get('contradictions', [])
            profile.resolution_history = pdata.get('resolution_history', {})
            profile.turn_num = pdata.get('turn_num', 0)
            self.users[uid] = profile

    def get_all_beliefs_for_subject(self, subject_id: str) -> Dict[str, List[Tuple[str, str, float, int]]]:
        """Get all beliefs about a subject across all users.
        Returns: {user_id: [(predicate, value, confidence, turn), ...]}
        """
        results = {}
        for uid, profile in self.users.items():
            subject_beliefs = []
            for (subj, pred), (val, conf, turn) in profile.beliefs.items():
                if subj == subject_id:
                    subject_beliefs.append((pred, val, conf, turn))
            if subject_beliefs:
                results[uid] = subject_beliefs
        return results