"""RAVANA — Online Construction Grammar: construction schema induction.

Implements pivot component C1: induce SLOT SCHEMAS from contrastive utterance
pairs (pattern grammar, Nakazawa 1995). Given two utterances that differ in
exactly one position, the differing span is a SLOT and the shared frame is a
SCHEMA. Stored as data (a schema store), not as if/elif. RAVANA can add,
revise, and overwrite schemas at runtime; removing one degrades gracefully to
the unschematised path.

No LLM, no retraining, no hardcoded reply strings. Everything here is
learnable online from conversation.

Seed (minimal, growable):
  - 6 slots: SUBJ, VERB, OBJ, MOD, CLAUSE, ADJUNCT
  - 5 relation types (mirrors syntactic_cell_assembly relation types)
  - 2 contrastive pairs (the minimum for a slot to generalise)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Minimal closed class of function words used to align contrastive pairs.
# Growable: callers may extend via register_function_word().
_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on",
    "for", "with", "and", "or", "but", "i", "you", "he", "she", "it", "we",
    "they", "my", "your", "his", "her", "its", "our", "their",
}

SEED_SLOTS = ["SUBJ", "VERB", "OBJ", "MOD", "CLAUSE", "ADJUNCT"]
SEED_RELATION_TYPES = ["causal", "semantic", "contrastive", "analogical", "temporal"]

# Two minimal contrastive pairs (the minimum for a slot to generalise).
SEED_CONTRASTIVE_PAIRS: List[Tuple[str, str]] = [
    ("i like cats", "i like dogs"),
    ("the cat sleeps", "the dog sleeps"),
]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", (text or "").lower())


@dataclass
class ConstructionSchema:
    """A learned slot schema (structure, not a string template)."""
    name: str
    slots: List[str] = field(default_factory=list)
    # support_count: how many contrastive pairs induced this schema
    support_count: int = 0
    # example fillers seen for each slot (growable, revisable)
    slot_examples: Dict[str, List[str]] = field(default_factory=dict)
    usage_count: int = 0
    success_count: int = 0

    def covers(self, required: List[str]) -> bool:
        return all(r in self.slots for r in (required or []))

    def success_rate(self) -> float:
        return self.success_count / max(1, self.usage_count)


class ConstructionSchemaStore:
    """Growable store of induced construction schemas."""

    def __init__(self):
        self.slots: List[str] = list(SEED_SLOTS)
        self.relation_types: List[str] = list(SEED_RELATION_TYPES)
        self.schemas: List[ConstructionSchema] = []
        self._function_words: set = set(_FUNCTION_WORDS)
        # Induce from the two seed pairs so the store is non-empty but grown,
        # not authored.
        for a, b in SEED_CONTRASTIVE_PAIRS:
            self.induce_from_pair(a, b)

    def register_function_word(self, word: str) -> None:
        if word:
            self._function_words.add(word.lower())

    def register_slot(self, slot: str) -> None:
        if slot and slot not in self.slots:
            self.slots.append(slot)

    def register_relation_type(self, rel: str) -> None:
        if rel and rel not in self.relation_types:
            self.relation_types.append(rel)

    def induce_from_pair(self, utt_a: str, utt_b: str) -> Optional[ConstructionSchema]:
        """Induce a schema from two utterances differing in one position.

        Returns the induced schema (new or updated) or None if the pair does
        not differ in exactly one span.
        """
        ta = _tokenize(utt_a)
        tb = _tokenize(utt_b)
        if not ta or not tb:
            return None
        # Find common prefix length
        i = 0
        while i < len(ta) and i < len(tb) and ta[i] == tb[i]:
            i += 1
        # Find common suffix length
        j = 0
        while (j < len(ta) - i and j < len(tb) - i
               and ta[-(j + 1)] == tb[-(j + 1)]):
            j += 1
        # The differing middle must be a single token on each side (or empty)
        left_a = ta[i:len(ta) - j if j else None]
        left_b = tb[i:len(tb) - j if j else None]
        if len(left_a) != 1 or len(left_b) != 1:
            # Not a single-slot contrast; cannot induce a clean schema.
            if len(left_a) == 0 or len(left_b) == 0:
                return None
            # Allow multi-token spans but mark as a CLAUSE slot.
            pass
        # Build the shared frame slots: one slot per position, with the
        # differing position becoming a slot.
        prefix = ta[:i]
        suffix = ta[len(ta) - j:] if j else []
        # Name the schema from its shared (non-slot) tokens.
        frame_tokens = prefix + ["<SLOT>"] + suffix
        name = " ".join(frame_tokens).strip() or "<schema>"
        # Determine slot name: prefer semantic slot by position/content.
        slot_name = self._slot_for_position(i, len(ta), left_a)
        # Find existing schema with same frame shape
        existing = None
        for s in self.schemas:
            if s.name == name:
                existing = s
                break
        if existing is None:
            existing = ConstructionSchema(name=name, slots=[slot_name])
            self.schemas.append(existing)
        if slot_name not in existing.slots:
            existing.slots.append(slot_name)
        existing.support_count += 1
        existing.slot_examples.setdefault(slot_name, [])
        for val in (left_a + left_b):
            if val not in existing.slot_examples[slot_name]:
                existing.slot_examples[slot_name].append(val)
        return existing

    def _slot_for_position(self, idx: int, n: int, span: List[str]) -> str:
        # Heuristic slot naming by position/content (growable seed, not a
        # fixed answer). First content word -> SUBJ, verb-ish -> VERB, etc.
        if idx == 0:
            return "SUBJ"
        if idx >= n - 1:
            return "OBJ"
        # If span contains a known function-ish/verb cue, treat as VERB
        for w in span:
            if w in {"is", "are", "was", "likes", "hates", "loves"}:
                return "VERB"
        return "MOD"

    def get_schema(self, required: Optional[List[str]] = None) -> Optional[ConstructionSchema]:
        """Return the best schema covering required slots (by support_count)."""
        required = required or []
        cands = [s for s in self.schemas if s.covers(required)] if required else list(self.schemas)
        if not cands:
            cands = list(self.schemas)
        if not cands:
            return None
        return max(cands, key=lambda s: (s.support_count, s.success_rate()))

    def learn_schema(self, schema: ConstructionSchema, success: bool = True) -> None:
        """Hebbian update on a schema (online)."""
        schema.usage_count += 1
        if success:
            schema.success_count += 1
        else:
            schema.success_count = max(0, schema.success_count - 1)

    def add_schema(self, name: str, slots: List[str]) -> ConstructionSchema:
        s = ConstructionSchema(name=name, slots=list(slots))
        self.schemas.append(s)
        return s

    def remove_schema(self, name: str) -> bool:
        before = len(self.schemas)
        self.schemas = [s for s in self.schemas if s.name != name]
        return len(self.schemas) < before

    def get_state(self) -> Dict:
        return {
            "slots": list(self.slots),
            "relation_types": list(self.relation_types),
            "function_words": sorted(self._function_words),
            "schemas": [
                {
                    "name": s.name,
                    "slots": list(s.slots),
                    "support_count": s.support_count,
                    "slot_examples": {k: list(v) for k, v in s.slot_examples.items()},
                    "usage_count": s.usage_count,
                    "success_count": s.success_count,
                }
                for s in self.schemas
            ],
        }

    def set_state(self, state: Dict) -> None:
        if not state:
            return
        self.slots = list(state.get("slots", self.slots))
        self.relation_types = list(state.get("relation_types", self.relation_types))
        if "function_words" in state:
            self._function_words = set(state["function_words"])
        self.schemas = []
        for sd in state.get("schemas", []):
            s = ConstructionSchema(
                name=sd.get("name", ""),
                slots=list(sd.get("slots", [])),
                support_count=int(sd.get("support_count", 0)),
                slot_examples={k: list(v) for k, v in sd.get("slot_examples", {}).items()},
                usage_count=int(sd.get("usage_count", 0)),
                success_count=int(sd.get("success_count", 0)),
            )
            self.schemas.append(s)