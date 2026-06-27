"""
Relation Memory — Transitive Inference and Relational Reasoning
================================================================
Stores comparative relations as first-class entities and computes
transitive closure for relational reasoning.

Neuroscience grounding:
- Hippocampus performs relational encoding (linking items into structure)
- RLPFC performs relational integration (combining relations for inference)

Design:
- Separate RelationGraph where nodes are entities, edges are comparative relations
- Transitive inference operator: A > B AND B > C → A > C
- Relation-as-entity encoding for complex relational reasoning
"""
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import re


@dataclass
class ComparativeRelation:
    """A directional comparative relation between two entities."""
    entity_a: str
    entity_b: str
    relation_type: str  # "taller", "heavier", "faster", "bigger", "smaller", etc.
    strength: float = 1.0  # How certain we are
    is_directional: bool = True  # True: A > B, False: A = B


@dataclass
class RelationMemoryConfig:
    """Configuration for relation memory."""
    max_relations: int = 200
    inference_confidence: float = 0.7  # Confidence multiplier for inferred relations
    max_path_length: int = 10


class RelationMemory:
    """Stores and infers comparative relations using transitive closure.

    Grammar for relations:
        - Comparative: A is taller than B → (A, B, "taller", True)
        - Equality: A is as tall as B → (A, B, "taller", False)
    """

    def __init__(self, config: Optional[RelationMemoryConfig] = None):
        self.config = config or RelationMemoryConfig()
        self.relations: List[ComparativeRelation] = []
        self._inferred_relations: Set[Tuple[str, str, str]] = set()  # (A, B, type)

        # Index: entity → list of relations involving that entity
        self._entity_index: Dict[str, List[int]] = defaultdict(list)

        # Known relation antonyms for inversion
        self.RELATION_ANTONYMS = {
            "taller": "shorter", "shorter": "taller",
            "heavier": "lighter", "lighter": "heavier",
            "faster": "slower", "slower": "faster",
            "bigger": "smaller", "smaller": "bigger",
            "larger": "smaller", "smaller": "larger",
            "longer": "shorter", "shorter": "longer",
            "older": "younger", "younger": "older",
            "greater": "lesser", "lesser": "greater",
            "more": "less", "less": "more",
            "better": "worse", "worse": "better",
            "higher": "lower", "lower": "higher",
            "stronger": "weaker", "weaker": "stronger",
            "richer": "poorer", "poorer": "richer",
            "smarter": "dumber", "dumber": "smarter",
            "happier": "sadder", "sadder": "happier",
        }

    def store(self, entity_a: str, entity_b: str, relation_type: str,
              is_directional: bool = True, strength: float = 1.0):
        """Store a comparative relation."""
        a_l, b_l = entity_a.lower().strip(), entity_b.lower().strip()
        rel_type = relation_type.lower().strip()

        # Avoid duplicates
        for rel in self.relations:
            if (rel.entity_a == a_l and rel.entity_b == b_l
                    and rel.relation_type == rel_type):
                return  # Already stored

        relation = ComparativeRelation(
            entity_a=a_l,
            entity_b=b_l,
            relation_type=rel_type,
            is_directional=is_directional,
            strength=strength,
        )
        self.relations.append(relation)
        self._entity_index[a_l].append(len(self.relations) - 1)
        self._entity_index[b_l].append(len(self.relations) - 1)

        # Enforce capacity
        if len(self.relations) > self.config.max_relations:
            self.relations = self.relations[-self.config.max_relations:]

    def compare(self, entity_a: str, entity_b: str, relation_type: str
                ) -> Tuple[Optional[str], float]:
        """Compare two entities using a specific relation type.

        Returns: ("a_greater" | "b_greater" | "equal" | None, confidence)
        """
        a_l, b_l = entity_a.lower().strip(), entity_b.lower().strip()
        rel_type = relation_type.lower().strip()

        # 1. Direct lookup
        for rel in self.relations:
            if rel.relation_type != rel_type:
                continue
            if rel.entity_a == a_l and rel.entity_b == b_l:
                return ("a_greater", rel.strength * self.config.inference_confidence)
            if rel.is_directional and rel.entity_a == b_l and rel.entity_b == a_l:
                return ("b_greater", rel.strength * self.config.inference_confidence)

        # 2. Check antonym relation
        antonym = self.RELATION_ANTONYMS.get(rel_type)
        if antonym:
            for rel in self.relations:
                if rel.relation_type != antonym:
                    continue
                if rel.entity_a == a_l and rel.entity_b == b_l:
                    return ("b_greater", rel.strength * 0.8 * self.config.inference_confidence)
                if rel.is_directional and rel.entity_a == b_l and rel.entity_b == a_l:
                    return ("a_greater", rel.strength * 0.8 * self.config.inference_confidence)

        # 3. Transitive inference: A > B AND B > C → A > C
        result, confidence = self._transitive_infer(a_l, b_l, rel_type)
        if result:
            return (result, confidence)

        return (None, 0.0)

    def transitive_query(self, entity: str, relation_type: str
                         ) -> List[Tuple[str, str, float]]:
        """Find all entities that have a relation with the given entity.

        Returns: [(other_entity, "a_greater" | "b_greater", confidence), ...]
        """
        e_l = entity.lower().strip()
        rel_type = relation_type.lower().strip()
        results = []

        # Direct relations
        for idx in self._entity_index.get(e_l, []):
            rel = self.relations[idx]
            if rel.relation_type != rel_type:
                continue
            if rel.entity_a == e_l:
                results.append((rel.entity_b, "a_greater", rel.strength))
            elif rel.is_directional:
                results.append((rel.entity_a, "b_greater", rel.strength))

        # Transitive inference
        for other_rel in self.relations:
            if other_rel.relation_type != rel_type:
                continue
            if other_rel.entity_a == e_l:
                # Find further relations from entity_b
                for deeper_idx in self._entity_index.get(other_rel.entity_b, []):
                    deeper = self.relations[deeper_idx]
                    if (deeper.relation_type == rel_type
                            and deeper.entity_a == other_rel.entity_b
                            and deeper.entity_b != e_l):
                        # Found chain: e_l > B > C
                        combined_conf = other_rel.strength * deeper.strength * 0.8
                        results.append((deeper.entity_b, "a_greater", combined_conf))
            elif other_rel.is_directional and other_rel.entity_b == e_l:
                # A > e_l, so we know A > e_l already
                pass

        return results

    def _transitive_infer(self, entity_a: str, entity_b: str, relation_type: str
                          ) -> Tuple[Optional[str], float]:
        """Transitive inference: find if A > B through a chain.

        BFS through relations to find a path from A to B.
        """
        visited = set()
        # Stack: (current_entity, accumulated_confidence)
        stack = [(entity_a, 1.0)]

        while stack:
            current, conf = stack.pop()
            if current == entity_b and conf < 1.0:  # Found through chain
                return ("a_greater", conf)
            if current in visited:
                continue
            visited.add(current)

            # Explore all relations from current entity
            for idx in self._entity_index.get(current, []):
                rel = self.relations[idx]
                if rel.relation_type != relation_type or not rel.is_directional:
                    continue
                if rel.entity_a == current and rel.entity_b not in visited:
                    stack.append((rel.entity_b, conf * rel.strength * 0.9))

        return (None, 0.0)

    def extract_from_text(self, text: str):
        """Extract comparative relations from natural language text.

        Parses patterns like:
            - "Tom is taller than Sarah"
            - "A is heavier than B"
            - "X runs faster than Y"
        """
        text_lower = text.lower().strip()

        # Pattern: "X is [comparative] than Y"
        comp_pattern = re.compile(
            r"(.+?)\s+is\s+(taller|shorter|heavier|lighter|faster|slower|"
            r"bigger|smaller|larger|longer|older|younger|greater|lesser|"
            r"better|worse|higher|lower|stronger|weaker|richer|poorer|"
            r"smarter|happier|sadder|more|less)\s+than\s+(.+)",
            re.IGNORECASE
        )
        m = comp_pattern.match(text_lower)
        if m:
            a = m.group(1).strip()
            rel = m.group(2).strip().lower()
            b = m.group(3).strip()
            # Clean up articles
            a = a.lstrip("the ").lstrip("a ").lstrip("an ")
            b = b.lstrip("the ").lstrip("a ").lstrip("an ")
            if a and b and a != b:
                self.store(a, b, rel, is_directional=True, strength=0.8)

        # Pattern: "X and Y are [comparative]"
        equal_pattern = re.compile(
            r"(.+?)\s+(?:and|&)\s+(.+?)\s+are\s+(?:both\s+)?(?:the\s+)?"
            r"(same|equal|identical|similar)\s+",
            re.IGNORECASE
        )
        m = equal_pattern.match(text_lower)
        if m:
            a = m.group(1).strip()
            b = m.group(2).strip()
            if a and b and a != b:
                self.store(a, b, "equal", is_directional=False, strength=0.7)

    def get_state(self) -> Dict:
        """Serialize state."""
        return {
            'relations': [
                {
                    'entity_a': r.entity_a,
                    'entity_b': r.entity_b,
                    'relation_type': r.relation_type,
                    'strength': r.strength,
                    'is_directional': r.is_directional,
                }
                for r in self.relations
            ],
            'inferred_relations': list(self._inferred_relations),
        }

    def set_state(self, state: Dict):
        """Restore state."""
        self.relations = []
        self._entity_index.clear()
        for rd in state.get('relations', []):
            rel = ComparativeRelation(
                entity_a=rd['entity_a'],
                entity_b=rd['entity_b'],
                relation_type=rd['relation_type'],
                strength=rd.get('strength', 1.0),
                is_directional=rd.get('is_directional', True),
            )
            self.relations.append(rel)
            self._entity_index[rel.entity_a].append(len(self.relations) - 1)
            self._entity_index[rel.entity_b].append(len(self.relations) - 1)
        self._inferred_relations = set(state.get('inferred_relations', []))
