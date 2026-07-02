import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

class CuriosityEngine:
    """Epistemic curiosity engine — drives autonomous exploration.

    Neuroscience basis (roadmap #13):
    - Anterior cingulate monitors prediction error -> norepinephrine -> exploration
    - Friston free energy principle: minimize surprise through active inference
    - Oudeyer & Kaplan: optimal learning progress drives curiosity

    Five curiosity signals:
    1. Prediction Error (Friston): high surprise -> explore
    2. Information Gap (Loewenstein): unresolved queries
    3. Cognitive Dissonance: contradiction pairs
    4. Novelty: least-visited or dormant-edge concepts
    5. Learning Progress (Oudeyer): PE delta tracking
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.prediction_errors: Dict[str, float] = defaultdict(lambda: 1.0)
        self.visit_counts: Dict[str, int] = defaultdict(int)
        self.learning_progress: Dict[str, float] = defaultdict(float)
        self.pe_deltas: Dict[str, float] = defaultdict(float)
        self.contradiction_map: Dict[str, Set[str]] = defaultdict(set)
        self.impossible_queries: List[str] = []
        self.explored_contradictions: Set[Tuple[str, str]] = set()
        self.recent_selections: List[Tuple[str, int]] = []

        self.last_urgency: float = 0.0
        self._rng = rng or np.random.default_rng()
        self.selection_cooldown: int = 10
        self.cycle_count: int = 0

    def uncertainty_for(self, concept: str) -> float:
        pe = self.prediction_errors.get(concept, 1.0)
        visits = self.visit_counts.get(concept, 0)
        return pe * np.exp(-visits * 0.1)

    def update_prediction_error(self, concept: str, error: float):
        old_pe = self.prediction_errors.get(concept, 1.0)
        alpha = 0.3
        smoothed = (1 - alpha) * old_pe + alpha * error
        self.prediction_errors[concept] = smoothed
        self.pe_deltas[concept] = old_pe - smoothed
        self.learning_progress[concept] = self.pe_deltas[concept]

    def record_visit(self, concept: str):
        self.visit_counts[concept] += 1

    def add_contradiction(self, concept_a: str, concept_b: str):
        self.contradiction_map[concept_a].add(concept_b)
        self.contradiction_map[concept_b].add(concept_a)

    def add_impossible_query(self, query: str):
        if query not in self.impossible_queries:
            self.impossible_queries.append(query)

    def resolve_impossible_query(self, query: str):
        if query in self.impossible_queries:
            self.impossible_queries.remove(query)

    def compute_urgency(self, arousal: float = 0.5, identity_strength: float = 0.5,
                        low_conf_count: int = 0) -> float:
        urgency = 0.0
        if self._prediction_error_count() > 5:
            urgency += min(0.4, self._mean_pe() * 2.0)
        unresolved = len([q for q in self.impossible_queries])
        if unresolved > 0:
            urgency += min(0.5, unresolved * 0.05)
        arousal_gain = 0.5 + 0.5 * arousal
        urgency *= arousal_gain
        urgency += (1.0 - identity_strength) * 0.15
        if low_conf_count > 0:
            urgency += min(0.2, low_conf_count * 0.01)
        self.last_urgency = min(1.0, urgency)
        return self.last_urgency

    def select_topics(self, max_topics: int = 3, all_labels: Optional[Set[str]] = None,
                      dormant_ratios: Optional[Dict[str, float]] = None,
                      user_topics: Optional[List[str]] = None,
                      concept_keywords: Optional[Dict] = None,
                      graph=None) -> List[str]:
        candidates: List[Tuple[str, float]] = []
        seen: Set[str] = set()

        for iq in self.impossible_queries:
            topic = iq.strip().lower()
            if topic and topic not in seen and len(topic) >= 3:
                candidates.append((topic, 5.0))
                seen.add(topic)

        high_pe = [(c, pe) for c, pe in self.prediction_errors.items()
                   if pe > 0.3 and c not in seen and len(c) >= 3]
        high_pe.sort(key=lambda x: -x[1])
        for label, pe in high_pe[:5]:
            candidates.append((label, 3.0 * min(1.0, pe)))
            seen.add(label)

        for concept, antonyms in self.contradiction_map.items():
            if concept not in seen and len(concept) >= 3:
                candidates.append((concept, 3.0))
                seen.add(concept)
            for ant in antonyms:
                if ant not in seen and len(ant) >= 3:
                    candidates.append((ant, 2.5))
                    seen.add(ant)

        if all_labels:
            unvisited = [l for l in all_labels if l not in seen and len(l) >= 3]
            if dormant_ratios:
                unvisited.sort(key=lambda l: dormant_ratios.get(l, 0), reverse=True)
            for label in unvisited[:3]:
                dr = dormant_ratios.get(label, 0) if dormant_ratios else 0
                weight = 0.5 + dr * 0.5
                candidates.append((label, weight))
                seen.add(label)

        epsilon = 0.35
        if all_labels and self._rng.random() < epsilon:
            available = [l for l in all_labels if l not in seen and len(l) >= 3]
            if available:
                random_topic = str(self._rng.choice(list(available)))
                candidates.append((random_topic, 0.5))
                seen.add(random_topic)

        self._apply_diversity_penalty(candidates)

        candidates.sort(key=lambda x: -x[1])
        selected = [topic for topic, _ in candidates[:max_topics]]

        for topic in selected:
            self.recent_selections.append((topic, self.cycle_count))
        self.cycle_count += 1
        self.recent_selections = [(t, c) for t, c in self.recent_selections
                                  if self.cycle_count - c < self.selection_cooldown]

        return selected

    def suggest_exploration(self) -> Optional[str]:
        if not self.prediction_errors:
            return None
        return max(self.prediction_errors, key=lambda c: self.uncertainty_for(c))

    def generate_query(self, topic: str, source_type: str = "general", antonym: str = "") -> str:
        clean = topic.lower().strip()
        if source_type == "contradiction" and antonym:
            ant = antonym.lower().strip()
            return f"{clean} versus {ant} comparison explained"
        elif source_type == "prediction_error":
            return f"{clean} what is it explained"
        elif source_type == "unknown":
            return f"{clean} definition meaning"
        return f"{clean} explained overview"

    def _apply_diversity_penalty(self, candidates: List[Tuple[str, float]]):
        recently = {t for t, _ in self.recent_selections}
        for i, (topic, weight) in enumerate(candidates):
            if topic in recently:
                for t, turn in self.recent_selections:
                    if t == topic:
                        recency = self.cycle_count - turn
                        penalty = max(0.01, 0.2 ** recency)
                        candidates[i] = (topic, weight * penalty)
                        break

    def _prediction_error_count(self) -> int:
        return sum(1 for v in self.prediction_errors.values() if v > 0.3)

    def _mean_pe(self) -> float:
        if not self.prediction_errors:
            return 0.0
        return sum(self.prediction_errors.values()) / len(self.prediction_errors)

    def get_state(self) -> dict:
        return {
            'prediction_errors': dict(self.prediction_errors),
            'visit_counts': dict(self.visit_counts),
            'learning_progress': dict(self.learning_progress),
            'pe_deltas': dict(self.pe_deltas),
            'contradiction_map': {k: list(v) for k, v in self.contradiction_map.items()},
            'impossible_queries': list(self.impossible_queries),
            'explored_contradictions': [list(p) for p in self.explored_contradictions],
            'recent_selections': list(self.recent_selections),
            'last_urgency': self.last_urgency,
            'cycle_count': self.cycle_count,
        }

    def set_state(self, state: dict):
        self.prediction_errors = defaultdict(float, state.get('prediction_errors', {}))
        self.visit_counts = defaultdict(int, state.get('visit_counts', {}))
        self.learning_progress = defaultdict(float, state.get('learning_progress', {}))
        self.pe_deltas = defaultdict(float, state.get('pe_deltas', {}))
        self.contradiction_map = defaultdict(
            set, {k: set(v) for k, v in state.get('contradiction_map', {}).items()})
        self.impossible_queries = list(state.get('impossible_queries', []))
        self.explored_contradictions = {tuple(p) for p in state.get('explored_contradictions', [])}
        self.recent_selections = [tuple(t) for t in state.get('recent_selections', [])]
        self.last_urgency = state.get('last_urgency', 0.0)
        self.cycle_count = state.get('cycle_count', 0)
