"""
Causal Schema — State-Transition Learning and Causal Simulation
================================================================
Learns triple patterns (state_A, action/condition, state_B) and generalizes
to new instances for causal reasoning without hardcoded physics rules.

Neuroscience grounding:
- Dorsal fronto-parietal pathway implements a "mental physics engine"
- Dopaminergic prediction error drives schema learning

Design:
- Learns (state_A, condition, state_B) triples from co-occurrence
- Generalizes: solid + heat → liquid extends to any solid
- Free-energy-driven: prediction failures drive new schema learning
- Used during discourse planning for "hypothetical" question types
"""
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class CausalSchema:
    """A learned state-transition pattern."""
    state_a: str  # Initial state/concept
    condition: str  # Action or condition
    state_b: str  # Resulting state/concept
    confidence: float = 0.5
    generalization_count: int = 1  # How many instances support this schema
    properties: Set[str] = field(default_factory=set)

    def matches(self, state: str) -> Tuple[bool, float]:
        """Check if a concept matches this schema's state_A, with similarity score."""
        s = state.lower()
        a = self.state_a.lower()
        if s == a:
            return (True, 1.0)
        # Check if state has a property matching this schema
        if any(prop.lower() in s for prop in self.properties):
            return (True, 0.7)
        return (False, 0.0)


@dataclass
class CausalSchemaConfig:
    """Configuration for causal schema learning."""
    learning_rate: float = 0.15
    min_confidence_for_generalization: float = 0.6
    schema_decay: float = 0.005
    max_schemas: int = 200


class CausalSchemaLearner:
    """Learns and applies causal state-transition schemas.

    Schema patterns:
        (state_A, condition, state_B) → e.g., (ice, heat, water)
        Generalization: (solid, heat, liquid) from (ice, heat, water)

    Free-energy-driven learning:
        When causal prediction fails, prediction error drives new schema formation.
    """

    # Innate property categories — NOT hardcoded physics, generic property
    # categories that bootstrap schema learning (analogous to infant core knowledge)
    PROPERTY_CATEGORIES = {
        "solid": {"ice", "stone", "rock", "wood", "metal", "bone", "brick", "glass"},
        "liquid": {"water", "oil", "juice", "milk", "wine", "blood", "lava"},
        "gas": {"air", "steam", "vapor", "smoke", "fog", "cloud", "wind"},
        "hot": {"sun", "fire", "flame", "lava", "heat", "oven", "stove"},
        "cold": {"ice", "snow", "frost", "freeze", "winter", "arctic"},
        "living": {"plant", "tree", "flower", "animal", "bird", "fish", "human"},
        "alive": {"person", "dog", "cat", "bird", "fish", "tree", "flower"},
        "dead": {"corpse", "fossil", "skeleton", "ash"},
    }

    # Innate schema priors — bootstrapping hypotheses that get refined
    # through experience. These are NOT hardcoded rules — they are weak
    # priors (confidence 0.3-0.4) that compete with learned schemas.
    INNATE_PRIORS = [
        ("solid", "heat", "liquid", 0.3, {"solid", "heat"}),
        ("liquid", "cold", "solid", 0.3, {"liquid", "cold"}),
        ("liquid", "heat", "gas", 0.3, {"liquid", "heat"}),
        ("alive", "death", "dead", 0.4, {"alive", "death"}),
        ("living", "water", "growth", 0.3, {"living", "water"}),
        ("object", "heat", "hot", 0.35, {"heat"}),
        ("object", "cold", "cold", 0.35, {"cold"}),
    ]

    def __init__(self, config: Optional[CausalSchemaConfig] = None):
        self.config = config or CausalSchemaConfig()
        self.schemas: List[CausalSchema] = []
        self._prediction_history: List[Tuple[str, str, str, bool]] = []  # (state, cond, predicted, success)
        self._total_errors: int = 0

        # Seed with innate priors
        for state_a, cond, state_b, conf, props in self.INNATE_PRIORS:
            self.schemas.append(CausalSchema(
                state_a=state_a,
                condition=cond,
                state_b=state_b,
                confidence=conf,
                properties=props,
            ))

    def learn(self, state_a: str, condition: str, state_b: str, success: bool = True):
        """Learn a state-transition pattern from experience.

        If a similar schema exists, strengthen it.
        If not, create a new schema.
        If success is False, prediction error drives new schema formation.
        """
        state_a_l = state_a.lower()
        condition_l = condition.lower()
        state_b_l = state_b.lower()

        existing = self._find_matching_schema(state_a_l, condition_l, state_b_l)

        if existing:
            # Strengthen existing schema (Hebbian)
            existing.confidence = min(1.0, existing.confidence + self.config.learning_rate)
            existing.generalization_count += 1
            if success:
                existing.confidence = min(1.0, existing.confidence + 0.05)
        else:
            # Create new schema
            # Learn property associations
            props = set()
            for category, members in self.PROPERTY_CATEGORIES.items():
                if state_a_l in members:
                    props.add(category)
                    props.add(state_a_l)
                if condition_l in members:
                    props.add(category)
                    props.add(condition_l)

            new_schema = CausalSchema(
                state_a=state_a_l,
                condition=condition_l,
                state_b=state_b_l,
                confidence=0.4,  # Start low, strengthen through repetition
                properties=props,
            )
            self.schemas.append(new_schema)

        # If prediction failed, increase error counter
        if not success:
            self._total_errors += 1

    def predict(self, state: str, condition: str) -> Tuple[Optional[str], float]:
        """Predict the result of applying a condition to a state.

        Uses schema generalization: if no exact match, try property-based matching.

        Returns:
            (predicted_state, confidence) or (None, 0.0) if no prediction possible
        """
        state_l = state.lower()
        condition_l = condition.lower()

        # 1. Try exact match first
        for schema in self.schemas:
            if schema.state_a == state_l and schema.condition == condition_l:
                return (schema.state_b, schema.confidence)

        # 2. Try property/generalization matching
        best_pred = None
        best_conf = 0.0
        for schema in self.schemas:
            matches, sim = schema.matches(state_l)
            if matches and schema.condition == condition_l:
                # Generalize: "ice + heat → water" → "stone + heat → ?"
                generalized_conf = schema.confidence * sim * 0.8
                if generalized_conf > best_conf:
                    best_conf = generalized_conf
                    best_pred = schema.state_b

        # 3. Try category-level generalization
        if best_conf < 0.4:
            state_categories = []
            for cat, members in self.PROPERTY_CATEGORIES.items():
                if state_l in members:
                    state_categories.append(cat)

            for schema in self.schemas:
                if schema.state_a in state_categories and schema.condition == condition_l:
                    generalized_conf = schema.confidence * 0.6
                    if generalized_conf > best_conf:
                        best_conf = generalized_conf
                        best_pred = schema.state_b

        return (best_pred, best_conf)

    def get_causal_path(self, start: str, end: str, max_hops: int = 3) -> List[Tuple[str, str, str, float]]:
        """Find a causal path from start to end through schemas.

        Returns: [(state, condition, result, confidence), ...]
        """
        if max_hops <= 0:
            return []

        # BFS through schemas
        visited = set()
        from collections import deque
        queue = deque()
        queue.append((start, []))  # (current_state, path)

        while queue:
            current, path = queue.popleft()
            if current == end:
                return path

            if current in visited:
                continue
            visited.add(current)

            for schema in self.schemas:
                matches, _ = schema.matches(current)
                if matches:
                    new_path = path + [(schema.state_a, schema.condition, schema.state_b, schema.confidence)]
                    if len(new_path) <= max_hops:
                        queue.append((schema.state_b, new_path))

        return []

    def explain_causal_chain(self, start: str, end: str) -> Tuple[Optional[str], List[str]]:
        """Generate a causal explanation chain from start to end.

        Returns: (final_state, list_of_explanations)
            e.g., ("water", ["ice + heat → water", "water + heat → steam"])
        """
        path = self.get_causal_path(start, end)
        if not path:
            return (None, [])

        explanations = []
        for state_a, cond, state_b, conf in path:
            explanations.append(f"{state_a} {cond} → {state_b}")
            final_state = state_b

        return (final_state, explanations)

    def record_prediction(self, state: str, condition: str, predicted: str, success: bool):
        """Record a prediction outcome for free-energy tracking."""
        self._prediction_history.append((state, condition, predicted, success))
        # Trim history
        if len(self._prediction_history) > 100:
            self._prediction_history = self._prediction_history[-100:]

    def get_prediction_error_rate(self) -> float:
        """Get recent prediction error rate (free energy signal)."""
        if not self._prediction_history:
            return 0.3
        recent = self._prediction_history[-20:]
        errors = sum(1 for _, _, _, success in recent if not success)
        return errors / len(recent)

    def get_state(self) -> Dict:
        """Serialize state."""
        return {
            'schemas': [
                {
                    'state_a': s.state_a,
                    'condition': s.condition,
                    'state_b': s.state_b,
                    'confidence': s.confidence,
                    'generalization_count': s.generalization_count,
                    'properties': list(s.properties),
                }
                for s in self.schemas
            ],
            'total_errors': self._total_errors,
        }

    def set_state(self, state: Dict):
        """Restore state."""
        self.schemas = []
        for sd in state.get('schemas', []):
            self.schemas.append(CausalSchema(
                state_a=sd['state_a'],
                condition=sd['condition'],
                state_b=sd['state_b'],
                confidence=sd.get('confidence', 0.5),
                generalization_count=sd.get('generalization_count', 1),
                properties=set(sd.get('properties', [])),
            ))
        self._total_errors = state.get('total_errors', 0)
