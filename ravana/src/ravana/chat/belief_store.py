from typing import Dict, List, Optional, Tuple


class BeliefStore:
    def __init__(self):
        self.beliefs: Dict[Tuple[str, str], Tuple[str, float, int]] = {}
        self.contradictions: List[Tuple[Tuple, Tuple, int]] = []
        self.resolution_history: Dict[str, str] = {}
        self.turn_num = 0

    def advance_turn(self):
        self.turn_num += 1

    def assert_belief(self, subject_id: str, predicate: str,
                       value: str, confidence: float = 0.8):
        key = (subject_id, predicate)
        self.beliefs[key] = (value, confidence, self.turn_num)

    def query_belief(self, subject_id: str,
                     predicate: str) -> Optional[Tuple[str, float, int]]:
        return self.beliefs.get((subject_id, predicate))

    def detect_contradiction(self, subject_id: str, predicate: str,
                              new_value: str) -> Optional[Tuple]:
        existing = self.query_belief(subject_id, predicate)
        if existing and existing[0] != new_value:
            old_triple = (subject_id, predicate, existing[0])
            new_triple = (subject_id, predicate, new_value)
            self.contradictions.append((old_triple, new_triple, self.turn_num))
            return (existing, new_value, existing[2])
        return None

    def resolve_contradiction(self, old_triple: Tuple, new_triple: Tuple,
                               choice: str):
        self.contradictions.append((old_triple, new_triple, self.turn_num))
        key = str(new_triple)
        self.resolution_history[key] = choice

    def reconcile(self) -> Dict[Tuple[str, str], Tuple[str, float, int]]:
        resolved: Dict[Tuple[str, str], Tuple[str, float, int]] = {}
        groups: Dict[Tuple[str, str], List[Tuple[str, float, int]]] = {}
        for old_triple, new_triple, c_turn in self.contradictions:
            subj, pred = old_triple[0], old_triple[1]
            key = (subj, pred)
            old_val, old_conf = old_triple[2], 0.0
            new_val, new_conf = new_triple[2], 0.0
            cur = self.beliefs.get(key)
            if cur:
                if cur[0] == old_val:
                    old_conf = cur[1]
                elif cur[0] == new_val:
                    new_conf = cur[1]
            groups.setdefault(key, []).append(
                (old_val, old_conf if old_conf > 0 else 0.5, c_turn))
            groups.setdefault(key, []).append(
                (new_val, new_conf if new_conf > 0 else 0.5, c_turn))
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
                recency = 1.0 / (1.0 + (self.turn_num - turn) * 0.1)
                return conf * recency
            winner = max(unique, key=decay_score)
            self.beliefs[key] = (winner[0], winner[1], self.turn_num)
            resolved[key] = self.beliefs[key]
        return resolved

    def get_state(self) -> Dict:
        return {
            'beliefs': self.beliefs,
            'contradictions': self.contradictions,
            'resolution_history': self.resolution_history,
            'turn_num': self.turn_num,
        }

    def set_state(self, state: Dict):
        self.beliefs = state.get('beliefs', {})
        self.contradictions = state.get('contradictions', [])
        self.resolution_history = state.get('resolution_history', {})
        self.turn_num = state.get('turn_num', 0)
