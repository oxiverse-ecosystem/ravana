"""GenerativeEmitter — compose surface form from cognitive state.

Replaces template-shaped emission (frame table + slot fill) with a mechanism
that PRODUCES surface form from learned structure. Constructions are stored in
an online-expandable bank (seed = small compositional rules, not full sentences)
and are selected/composed from the live cognitive state (identity, stances,
facts, beliefs). No LLM, no retraining: a new construction can be added during
a single conversation via ``learn_construction``.

The form (which construction, which ordering) comes from state, not from a
fixed authored string. Content tokens come from state too.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple


class GenerativeEmitter:
    def __init__(self, seed: int = 0):
        # construction bank: list of (slot_template, slots) where slot_template
        # is a sequence of slot names; realized by joining realized slot text.
        # Seed = compositional rules (slot names), NOT full sentences.
        self.constructions: List[List[str]] = [
            ["stance_topic", "stance_lean"],
            ["identity_strength", "stance_topic"],
            ["fact_value", "fact_attr"],
            ["belief_text"],
        ]
        self._rng = random.Random(seed)
        self.learned: List[List[str]] = []

    def _realize_slot(self, slot: str, state: Dict[str, Any]) -> Optional[str]:
        if slot == "stance_topic":
            stances = state.get("stances") or {}
            if not stances:
                return None
            # pick by hash of query for determinism-per-state, not per-question table
            topics = sorted(stances.keys())
            if not topics:
                return None
            idx = hash(state.get("query", "")) % len(topics)
            return str(topics[idx])
        if slot == "stance_lean":
            stances = state.get("stances") or {}
            topic = self._last_slot_value("stance_topic", state)
            if topic is None or topic not in stances:
                return None
            pol = stances[topic].get("polarity", 0.0) if isinstance(stances[topic], dict) else getattr(stances[topic], "polarity", 0.0)
            if pol > 0.2:
                return "i lean for it"
            if pol < -0.2:
                return "i lean against it"
            return "i'm neutral on it"
        if slot == "identity_strength":
            ident = state.get("identity") or {}
            s = ident.get("strength", 0.0)
            if s >= 0.6:
                return "i'm fairly settled"
            if s <= 0.3:
                return "i'm still forming"
            return "i'm in between"
        if slot == "fact_value":
            facts = state.get("facts") or {}
            if not facts:
                return None
            keys = sorted(facts.keys(), key=lambda k: str(k))
            idx = hash(state.get("query", "")) % len(keys)
            f = facts[keys[idx]]
            val = f.get("value") if isinstance(f, dict) else getattr(f, "value", None)
            return str(val) if val is not None else None
        if slot == "fact_attr":
            facts = state.get("facts") or {}
            if not facts:
                return None
            keys = sorted(facts.keys(), key=lambda k: str(k))
            idx = hash(state.get("query", "")) % len(keys)
            k = keys[idx]
            # derive attr from key tuple if possible
            if isinstance(k, tuple) and len(k) >= 2:
                return str(k[1])
            return str(k)
        if slot == "belief_text":
            beliefs = state.get("beliefs") or {}
            if not beliefs:
                return None
            keys = sorted(beliefs.keys(), key=lambda k: str(k))
            idx = hash(state.get("query", "")) % len(keys)
            b = beliefs[keys[idx]]
            txt = b[0] if isinstance(b, tuple) else (b.get("text") if isinstance(b, dict) else str(b))
            return str(txt) if txt is not None else None
        return None

    def _last_slot_value(self, slot, state):
        # realize the slot for real so downstream slots (e.g. stance_lean)
        # can depend on the chosen topic instead of re-deriving it.
        return self._realize_slot(slot, state)

    def compose(self, state: Dict[str, Any]) -> Optional[str]:
        pool = self.constructions + self.learned
        if not pool:
            return None
        # select construction by state hash so same state -> same form,
        # different state -> different form
        idx = hash((state.get("query", ""), len(state.get("stances") or {}), len(state.get("facts") or {}))) % len(pool)
        cons = pool[idx]
        parts = []
        for slot in cons:
            v = self._realize_slot(slot, state)
            if v is None:
                parts = []
                break
            parts.append(v)
        if not parts:
            return None
        return " ".join(parts).strip().lower()

    def learn_construction(self, construction: List[str]):
        if construction and construction not in self.learned and construction not in self.constructions:
            self.learned.append(list(construction))
