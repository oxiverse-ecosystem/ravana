"""Entity-scoped possession-attribute mining for the personal-fact store.

Why this module exists
----------------------
A disclosure like "the cabin is a hand-hewn pine lodge with a sod roof" or
"my sword is made of meteorite iron" states a PROPERTY of a possession the user
owns or describes. The main fact miner (user_model.mine_personal_facts) only
captured explicit "my X is Y" self-facts and pet names, so material/attribute
facts about a possession were never stored as a recallable, correctable fact --
and "what's my cabin made of" could not be answered from the structured store
(round 2026-08-15T0830Z, Bug 4).

The fix mines those disclosures into the PersonalFactStore under the ENTITY
(e.g. cabin / sword), not the user's own "i" subject, exactly like the pet
slots do for animals. From there the store LEARNS: a later "no, my cabin is
oak-framed" contradicts via the existing contradict() path; confirm/contradict
work unchanged.

Seed vocabulary, not an answer table
------------------------------------
The MATERIALS set is SEED data mirroring pet_slots._SPECIES_SEED: a closed-core
list of material nouns RAVANA is born understanding, plus a runtime-grown
extension via learn_material() so a word it has never heard ("hempcrete",
"rammed earth") becomes addressable for later recall without a code change.
Removing an entry degrades gracefully (the material is simply not mined until
re-learned). The material sense is DATA, never reply text -- nothing here is
ever rendered to the user; recall rendering lives in engine_memory._reconstruct_entity.
"""

from typing import Dict, Optional, Set

# Seed material vocabulary: noun forms RAVANA recognises as a building/material.
# Extended at runtime via learn_material(); never a source of reply text.
_MATERIALS_SEED: Set[str] = {
    # wood + wood products
    "wood", "timber", "pine", "oak", "cedar", "maple", "birch", "walnut",
    "mahogany", "teak", "ash", "elm", "spruce", "fir", "log", "logs",
    "plywood", "particleboard", "mdf",
    # earth + stone
    "stone", "brick", "bricks", "adobe", "clay", "mud", "cob", "rammed",
    "earth", "concrete", "cement", "granite", "marble", "slate", "limestone",
    "sandstone", "tuff", "basalt", "coral", "sod", "turf", "thatch", "straw",
    "bamboo", "reed", "wattle",
    # metal
    "iron", "steel", "bronze", "copper", "brass", "aluminium", "aluminum",
    "tin", "lead", "gold", "silver", "meteorite", "cast", "wrought",
    # glass + modern
    "glass", "crystal", "plastic", "fiberglass", "composite", "carbon",
    "resin", "epoxy", "ceramic", "tile", "tiles", "linoleum", "vinyl",
    # fabric / soft
    "canvas", "linen", "cotton", "wool", "silk", "leather", "hemp",
 "insulation", "felt",
 }

# Runtime-grown extension of the seed material table.
_MATERIALS_LEARNED: Dict[str, str] = {}

# Feature nouns a material can modify -- "sod roof", "brick wall". When a
# material is immediately followed by one of these in the description, the fact
# is stored under that feature (cabin.roof = sod) rather than the generic
# material of the whole entity.
_FEATURE_NOUNS: Set[str] = {
    "roof", "wall", "walls", "door", "floor", "floors", "ceiling", "window",
    "windows", "foundation", "frame", "frames", "beam", "beams", "pillar",
    "deck", "porch", "chimney", "fence", "panel", "panels", "handle", "counter",
}

# Built-structure / possession kind nouns that signal a possessive material
# description ("a pine lodge", "a brick house"). Used as a precision gate: a
# clause is only mined as a possession description when it names a kind noun,
# so "the river is a fast mountain stream" (no material, no kind noun) is
# correctly ignored.
_KIND_NOUNS: Set[str] = {
    "lodge", "cabin", "house", "home", "hut", "shack", "shed", "barn", "tower",
    "castle", "cottage", "bungalow", "villa", "manor", "fort", "temple",
    "shrine", "building", "structure", "warehouse", "garage", "stable",
    "boathouse", "mill", "workshop", "studio", "office", "school", "church",
    "roof", "wall", "door", "floor", "floorboard", "ceiling", "window",
    "table", "chair", "desk", "bed", "bench", "shelf", "shelves", "fence",
    "gate", "bridge", "boat", "ship", "canoe", "kayak", "raft", "dock",
    "pier", "deck", "stairs", "step", "steps", "altar", "throne", "statue",
    "monument", "sword", "blade", "knife", "axe", "shield", "armor", "helmet",
    "ring", "crown", "goblet", "bowl", "pot", "vase", "lamp", "lantern",
    "wagon", "cart", "sled", "wheel", "frame", "box", "chest", "crate",
}


def learn_material(word: str) -> str:
    """Register a material word seen in a live disclosure and return its canon.

    Growth path for the seed vocabulary: a material RAVANA has never heard of
    ("hempcrete", "rammed earth") becomes addressable for later recall without
    any code change. A trailing plural "s" is folded onto the singular so the
    plural form of the same word resolves to one entry.
    """
    w = (word or "").strip().lower()
    if not w:
        return ""
    known = is_material(w)
    if known:
        return w
    canon = w[:-1] if len(w) > 3 and w.endswith("s") else w
    _MATERIALS_LEARNED[w] = canon
    _MATERIALS_LEARNED[canon] = canon
    if not canon.endswith("s"):
        _MATERIALS_LEARNED[canon + "s"] = canon
    return canon


def is_material(word: str) -> bool:
    """True when a surface word is a known material noun."""
    w = (word or "").strip().lower()
    if not w:
        return False
    return w in _MATERIALS_SEED or w in _MATERIALS_LEARNED


def is_feature_noun(word: str) -> bool:
    """True when a surface word is a possession feature (roof/wall/...)."""
    w = (word or "").strip().lower()
    return w in _FEATURE_NOUNS


def is_kind_noun(word: str) -> bool:
    """True when a surface word marks a built-structure / possession kind."""
    w = (word or "").strip().lower()
    return w in _KIND_NOUNS


def render(ent: str, attr: str, val: str) -> str:
    """Render a stored possession fact as a natural clause for a recall reply.

    'madeof' -> "your cabin is made of pine"; a feature attr -> "your cabin's
    roof is sod". Kept here so the miner and every recall site agree on one
    phrasing by construction (mirrors pet_slots.render_pair).
    """
    if attr == "madeof":
        return f"your {ent} is made of {val}"
    if is_feature_noun(attr):
        return f"your {ent}'s {attr} is {val}"
    return f"your {ent}'s {attr} is {val}"
