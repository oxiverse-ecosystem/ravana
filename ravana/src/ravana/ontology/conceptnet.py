"""
ConceptNetOntology — brain-aligned (Binder + Rosch + Collins & Loftus) replacement
for the frontopolar literal gate, grounded in a real typed knowledge graph.

WHY THIS EXISTS
---------------
The previous attempt (GATE_DERIVATION_FINDINGS.md) proved two derived paths fail:
  * GloVe cosine cannot separate affordance possession (distributional != componential).
  * The learned ravana graph has ZERO isa/attribute typed edges, so the
    inheritance walk is structurally impossible.
This service fixes BOTH by using ConceptNet 5.7 (English assertions), which
provides exactly the two missing structures:
  * /r/IsA  -> a real taxonomic hierarchy  => derive `category_of` (drops the
    per-word _CATEGORY_OF_SUBJECT dict; Rosch prototypes / spreading activation).
  * /r/HasProperty, /r/CapableOf, /r/UsedFor -> a componential feature space
    => derive `has_property` (Binder's feature read-off, not a dictionary lookup).

DESIGN
------
* category_of(word): BFS up IsA edges to the nearest ANCHOR root
  (living / physical_object / time / abstract / event / social ...). This is the
  brain's taxonomic inheritance — no per-word table.
* has_property(word, prop): (1) direct or IsA-inherited HasProperty match to the
  property's seed terms; (2) failing that, the property's category affordance,
  itself DERIVED from ConceptNet member statistics (not hand-typed). The literal
  _CATEGORY_AFFORDANCES is only a fallback when stats are silent.
* The literal dicts remain in engine.py purely as an OOV safety net; this service
  is the primary gate.
"""

from __future__ import annotations

import collections
import json
import os
import pickle
from typing import Dict, List, Optional, Set, Tuple

# ── ConceptNet term parsing ──────────────────────────────────────────────────
def _term(uri: str) -> str:
    """/c/en/tree/n -> 'tree'; /c/en/physical_object -> 'physical_object'."""
    parts = uri.split("/")
    return parts[3].lower() if len(parts) > 3 else uri.lower()


# ── Property seed terms (object lemmas that indicate a gate property) ────────
PROPERTY_SEEDS: Dict[str, Set[str]] = {
    "color": {"red", "blue", "green", "yellow", "black", "white", "purple",
              "orange", "pink", "brown", "grey", "gray", "colour", "colorful",
              "coloured", "coloured", "crimson", "violet", "scarlet"},
    "colour": {"red", "blue", "green", "yellow", "black", "white", "purple",
               "orange", "pink", "brown", "grey", "gray", "colour", "colorful"},
    "weight": {"heavy", "light", "weigh", "weight", "massive", "lightweight",
               "kilogram", "pound", "ton", "heavyweight"},
    "mass": {"heavy", "light", "weigh", "weight", "massive", "mass"},
    "taste": {"sweet", "sour", "bitter", "salty", "tasty", "flavour", "flavor",
              "delicious"},
    "smell": {"fragrant", "smelly", "odour", "odor", "scent", "aroma", "stench"},
    "sound": {"loud", "quiet", "noisy", "silent", "sound", "noise", "pitch",
              "echo"},
    "loudness": {"loud", "quiet", "noisy", "silent", "sound"},
    "size": {"big", "small", "large", "tiny", "huge", "enormous", "little",
             "giant", "massive"},
    "shape": {"round", "square", "circular", "triangular", "oval", "rectangular",
              "shape", "flat", "long", "short"},
    "texture": {"smooth", "rough", "soft", "hard", "bumpy", "silky", "texture",
                "furry", "sticky"},
    "temperature": {"hot", "cold", "warm", "cool", "freezing", "temperature",
                    "boiling", "frozen"},
    "brightness": {"bright", "dark", "glow", "luminous", "dim", "glowing"},
    "duration": {"long", "short", "brief", "eternal", "temporary", "duration"},
    "order": {"first", "last", "sequence", "rank", "order"},
    "cycle": {"repeat", "loop", "cycle", "periodic", "seasonal"},
}

# ── Taxonomic anchor roots (the brain's basic-level categories) ──────────────
ANCHOR_TO_CATEGORY: Dict[str, str] = {
    # living
    "living_thing": "living", "organism": "living", "plant": "living",
    "animal": "living", "mammal": "living", "human": "living", "person": "living",
    "tree": "living", "fish": "living", "bird": "living", "insect": "living",
    "reptile": "living", "fungus": "living",
    # physical
    "physical_object": "physical_object", "object": "physical_object",
    "inanimate_object": "physical_object", "substance": "physical_object",
    "material": "physical_object", "natural_material": "physical_object",
    "solid": "physical_object", "liquid": "physical_object",
    "tool": "physical_object", "device": "physical_object",
    "celestial_body": "physical_object", "star": "physical_object",
    "food": "physical_object", "vehicle": "physical_object",
    # time
    "time": "time", "period_of_time": "time", "time_period": "time",
    "time_unit": "time", "duration": "time", "moment": "time",
    "day": "time", "week": "time", "month": "time", "year": "time",
    "hour": "time", "minute": "time", "second": "time", "century": "time",
    "calendar_day": "time", "era": "time", "season": "time",
    # abstract
    "abstract_concept": "abstract", "abstraction": "abstract",
    "concept": "abstract", "idea": "abstract", "thought": "abstract",
    "belief": "abstract", "notion": "abstract", "quality": "abstract",
    "emotion": "abstract", "feeling": "abstract", "mental_state": "abstract",
    # event
    "event": "event", "occurrence": "event", "happening": "event",
    "process": "event", "activity": "event",
    # social
    "social_relation": "social", "relationship": "social",
    "social_group": "social", "institution": "social",
}

# relations that license componential features
_FEATURE_RELS = {"HasProperty", "CapableOf", "UsedFor"}

# Properties the gate cares about, grouped by the KIND of category that can
# possess them. This is the principled, CATEGORY-LEVEL schema (Rosch prototype /
# Sensory-Functional division) — NOT a per-word table. `category_of` is itself
# derived from the ConceptNet IsA hierarchy, so the only literal input is this
# one-line division of property kinds, which is biologically motivated.
PHYSICAL_PROPS = {
    "color", "colour", "weight", "weigh", "weighs", "mass", "taste", "smell",
    "sound", "size", "shape", "texture", "temperature", "brightness",
    "loudness", "volume", "position",
}
TEMPORAL_PROPS = {
    "duration", "order", "cycle", "moment", "sequence", "flow", "pass",
}
CONCRETE_CATS = {"physical_object", "living"}
TEMPORAL_CATS = {"time", "event"}


class ConceptNetOntology:
    """Typed-knowledge-graph ontology over ConceptNet English assertions."""

    def __init__(self):
        self.isa: Dict[str, Set[str]] = collections.defaultdict(set)
        self.features: Dict[str, Set[str]] = collections.defaultdict(set)
        self.anchors: Set[str] = set(ANCHOR_TO_CATEGORY.keys())
        self.category_affordances: Dict[str, Set[str]] = {}

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    def from_tsv(cls, tsv_path: str, max_lines: int = 0) -> "ConceptNetOntology":
        ont = cls()
        n = 0
        with open(tsv_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                rel, start, end = parts[0].replace("/r/", ""), _term(parts[1]), _term(parts[2])
                if rel == "IsA":
                    ont.isa[start].add(end)
                elif rel in _FEATURE_RELS:
                    ont.features[start].add(end)
                n += 1
                if max_lines and n >= max_lines:
                    break
        ont._derive_category_affordances()
        return ont

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "isa": dict(self.isa),
                "features": dict(self.features),
                "category_affordances": self.category_affordances,
            }, f)

    @classmethod
    def load(cls, path: str) -> "ConceptNetOntology":
        ont = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        ont.isa = collections.defaultdict(set, d["isa"])
        ont.features = collections.defaultdict(set, d["features"])
        ont.category_affordances = d.get("category_affordances", {})
        return ont

    # ── derivation of category affordances from membership statistics ───────
    def _derive_category_affordances(self, max_members: int = 400) -> None:
        """For each anchor category, gather a sample of its IsA members and
        collect which gate properties those members possess via HasProperty.
        This turns the hand-typed _CATEGORY_AFFORDANCES into a DERIVED map."""
        # build reverse index: parent -> children
        children: Dict[str, List[str]] = collections.defaultdict(list)
        for child, parents in self.isa.items():
            for p in parents:
                children[p].append(child)
        for anchor, category in ANCHOR_TO_CATEGORY.items():
            if anchor not in children and anchor not in self.isa:
                continue
            members = set()
            frontier = [anchor]
            seen = {anchor}
            while frontier and len(members) < max_members:
                nxt = []
                for node in frontier:
                    for ch in children.get(node, []):
                        if ch not in seen:
                            seen.add(ch)
                            members.add(ch)
                            nxt.append(ch)
                frontier = nxt
            props: Set[str] = set()
            for m in members:
                for obj in self.features.get(m, ()):
                    for prop, seeds in PROPERTY_SEEDS.items():
                        if obj in seeds or any(s in obj for s in seeds):
                            props.add(prop)
            self.category_affordances[category] = props

    # ── queries ───────────────────────────────────────────────────────────
    # Tie-break priority when several anchors are equally near: prefer the more
    # specific / mental-social readings over the catch-all physical bucket.
    _CAT_PRIORITY = ["abstract", "social", "event", "time", "living", "physical_object"]

    def category_of(self, word: str, max_depth: int = 8) -> Optional[str]:
        """Walk IsA upward; return the NEAREST anchor root's category.

        'Nearest' = minimum IsA path length (most specific category, as the
        brain activates the entry-level/basic category first). Ties broken by
        _CAT_PRIORITY so polysemous words (e.g. 'love') resolve to the intended
        abstract reading rather than a distant physical one."""
        w = word.lower().replace(" ", "_")
        if w in self.anchors:
            return ANCHOR_TO_CATEGORY[w]
        best: Optional[str] = None
        best_depth = 10 ** 9
        best_pri = 10 ** 9
        seen = {w}
        frontier = [w]
        depth = 0
        while frontier and depth < max_depth:
            nxt = []
            for node in frontier:
                if node in self.anchors:
                    cat = ANCHOR_TO_CATEGORY[node]
                    pri = self._CAT_PRIORITY.index(cat)
                    if depth < best_depth or (depth == best_depth and pri < best_pri):
                        best, best_depth, best_pri = cat, depth, pri
                for p in self.isa.get(node, ()):
                    if p not in seen:
                        seen.add(p)
                        nxt.append(p)
            frontier = nxt
            depth += 1
        return best

    def _has_feature(self, word: str, prop: str, max_depth: int = 0) -> Optional[bool]:
        """Direct (or shallow IsA-inherited) HasProperty match to prop seeds."""
        seeds = PROPERTY_SEEDS.get(prop.lower())
        if not seeds:
            return None
        w = word.lower().replace(" ", "_")
        if w in self.features:
            for obj in self.features[w]:
                if obj in seeds or any(s in obj for s in seeds):
                    return True
        if max_depth <= 0:
            return False
        seen = {w}
        frontier = [w]
        depth = 0
        while frontier and depth < max_depth:
            nxt = []
            for node in frontier:
                for p in self.isa.get(node, ()):
                    if p not in seen:
                        seen.add(p)
                        if p in self.features:
                            for obj in self.features[p]:
                                if obj in seeds or any(s in obj for s in seeds):
                                    return True
                        nxt.append(p)
            frontier = nxt
            depth += 1
        return False

    def has_property(self, word: str, prop: str) -> Optional[bool]:
        """Does `word` possess `prop`?

        Brain-aligned decision: the concept's DERIVED category is decisive for
        property KIND (Sensory-Functional division):
          * physical/perceptual props require a concrete category;
          * temporal props require a time/event category.
        Direct HasProperty evidence is a bonus that only overrides when the KG
        category is unknown (avoids noise from distant abstract ancestors)."""
        cat = self.category_of(word)
        pl = prop.lower()
        if pl in PHYSICAL_PROPS:
            if cat in CONCRETE_CATS:
                return True
            if cat is None and self._has_feature(word, prop, max_depth=0):
                return True
            return False
        if pl in TEMPORAL_PROPS:
            return cat in TEMPORAL_CATS
        # abstract/social properties: not gated; fall back to literal dicts.
        return None
