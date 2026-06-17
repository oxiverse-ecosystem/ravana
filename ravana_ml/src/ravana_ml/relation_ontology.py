"""
Relation Ontology
==================
Multi-level relation hierarchy for typed traversal.

Hierarchy:
  Family > Sub-family > Predicate

Traversal can operate at any granularity:
  PREDICATE: "causes" only
  SUB-FAMILY: "causal-strong" (causes, produces, triggers)
  FAMILY: "causal" (all causal sub-families)
  SUPER-FAMILY: "causal + contributory"

Structured Candidate:
  (word, predicate, family, sub_family, depth, confidence, path)
"""
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# RELATION ONTOLOGY
# ============================================================

# Sub-families group predicates by semantic similarity
SUB_FAMILIES = {
    # --- CAUSAL family ---
    "causal-strong": {
        "family": "causal",
        "predicates": ["causes", "cause", "produces", "produce", "creates", "create",
                       "triggers", "trigger", "results_in", "generates", "generate"],
        "description": "Direct, strong causation",
        "confidence_range": (0.85, 0.95),
    },
    "causal-moderate": {
        "family": "causal",
        "predicates": ["leads_to", "leads", "makes", "make", "forces", "force",
                       "drives", "drive", "powers", "power"],
        "description": "Indirect or moderate causation",
        "confidence_range": (0.70, 0.85),
    },
    "causal-weak": {
        "family": "causal",
        "predicates": ["may_cause", "can_cause", "might_cause",
                       "contributes_to", "contribute",
                       "associated_with", "associated",
                       "linked_to", "linked",
                       "correlated_with", "correlated"],
        "description": "Weak, probabilistic, or correlational causation",
        "confidence_range": (0.35, 0.60),
    },

    # --- DIRECTIONAL family ---
    "directional-positive": {
        "family": "directional",
        "predicates": ["increases", "increase", "amplifies", "amplify",
                       "boosts", "boost", "enhances", "enhance",
                       "improves", "improve", "strengthens", "strengthen",
                       "accelerates", "accelerate"],
        "description": "Positive directional change",
        "confidence_range": (0.70, 0.85),
    },
    "directional-negative": {
        "family": "directional",
        "predicates": ["decreases", "decrease", "reduces", "reduce",
                       "inhibits", "inhibit", "suppresses", "suppress",
                       "weakens", "weaken", "diminishes", "diminish",
                       "worsens", "worsen", "impairs", "impair",
                       "prevents", "prevent", "blocks", "block"],
        "description": "Negative directional change",
        "confidence_range": (0.70, 0.85),
    },

    # --- COMPOSITIONAL family ---
    "compositional": {
        "family": "compositional",
        "predicates": ["contains", "contain", "includes", "include",
                       "has", "have", "comprises", "comprise",
                       "consists_of", "composed_of", "made_of"],
        "description": "Part-whole, containment",
        "confidence_range": (0.85, 0.95),
    },

    # --- TAXONOMIC family ---
    "taxonomic": {
        "family": "taxonomic",
        "predicates": ["is_a", "is_type_of", "type_of", "subclass_of",
                       "kind_of", "example_of", "instance_of",
                       "means", "mean", "defines", "define",
                       "equivalent_to", "same_as"],
        "description": "Category membership, definition",
        "confidence_range": (0.80, 0.90),
    },

    # --- TEMPORAL family ---
    "temporal": {
        "family": "temporal",
        "predicates": ["precedes", "precede", "follows", "follow",
                       "before", "after", "during", "while",
                       "then", "next", "later", "until", "since"],
        "description": "Temporal ordering",
        "confidence_range": (0.70, 0.80),
    },

    # --- CAPABILITY family ---
    "capability": {
        "family": "capability",
        "predicates": ["can", "could", "able_to", "capable_of",
                       "enables", "enable", "allows", "allow",
                       "supports", "support", "facilitates", "facilitate"],
        "description": "Ability, permission, enablement",
        "confidence_range": (0.60, 0.80),
    },
}

# Reverse lookup: predicate -> sub_family
_PREDICATE_TO_SUB = {}
for sub_name, sub_info in SUB_FAMILIES.items():
    for pred in sub_info["predicates"]:
        _PREDICATE_TO_SUB[pred] = sub_name

# Super-families: groups of families that can compose
SUPER_FAMILIES = {
    "causal_all": ["causal-strong", "causal-moderate", "causal-weak"],
    "directional_all": ["directional-positive", "directional-negative"],
    "causal_directional": ["causal-strong", "causal-moderate", "causal-weak",
                           "directional-positive", "directional-negative"],
}


@dataclass
class Candidate:
    """Structured reasoning result."""
    word: str
    predicate: str          # original predicate word (e.g., "contributes_to")
    family: str             # family name (e.g., "causal")
    sub_family: str         # sub-family name (e.g., "causal-weak")
    depth: int              # traversal depth
    confidence: float       # predicate confidence score
    path: list = field(default_factory=list)  # chain of (node, predicate) tuples


@dataclass
class TraversalConfig:
    """Controls traversal granularity."""
    mode: str = "family"    # "predicate", "sub_family", "family", "super_family", "relaxed"
    family: Optional[str] = None
    sub_family: Optional[str] = None
    super_family: Optional[str] = None


def get_sub_family(predicate: str) -> Optional[str]:
    """Look up which sub-family a predicate belongs to."""
    return _PREDICATE_TO_SUB.get(predicate)


def get_family(predicate: str) -> Optional[str]:
    """Look up which family a predicate belongs to."""
    sub = get_sub_family(predicate)
    if sub:
        return SUB_FAMILIES[sub]["family"]
    return None


def get_confidence(predicate: str) -> float:
    """Get default confidence for a predicate."""
    sub = get_sub_family(predicate)
    if sub:
        lo, hi = SUB_FAMILIES[sub]["confidence_range"]
        return (lo + hi) / 2
    return 0.5


def matches_config(predicate: str, config: TraversalConfig) -> bool:
    """Check if a predicate matches the traversal config."""
    if config.mode == "relaxed":
        return True

    pred_sub = get_sub_family(predicate)
    pred_family = get_family(predicate)

    if config.mode == "predicate":
        return predicate == config.sub_family  # sub_family field holds the specific predicate
    elif config.mode == "sub_family":
        return pred_sub == config.sub_family
    elif config.mode == "family":
        return pred_family == config.family
    elif config.mode == "super_family":
        allowed_subs = SUPER_FAMILIES.get(config.super_family, [])
        return pred_sub in allowed_subs
    return False


def print_ontology():
    """Pretty-print the full relation ontology."""
    print("=" * 70)
    print("RELATION ONTOLOGY")
    print("=" * 70)

    families = {}
    for sub_name, sub_info in SUB_FAMILIES.items():
        fam = sub_info["family"]
        if fam not in families:
            families[fam] = []
        families[fam].append((sub_name, sub_info))

    for fam_name, subs in sorted(families.items()):
        print(f"\n  {fam_name.upper()}")
        for sub_name, sub_info in subs:
            lo, hi = sub_info["confidence_range"]
            preds = ", ".join(sub_info["predicates"][:5])
            if len(sub_info["predicates"]) > 5:
                preds += f" ... (+{len(sub_info['predicates'])-5})"
            print(f"    {sub_name:25s} [{lo:.2f}-{hi:.2f}] {preds}")

    print(f"\n  SUPER-FAMILIES")
    for sf_name, sf_subs in SUPER_FAMILIES.items():
        print(f"    {sf_name:25s} = {' + '.join(sf_subs)}")


if __name__ == "__main__":
    print_ontology()

    # Test lookups
    print("\n\nLookup tests:")
    for pred in ["causes", "contributes_to", "increases", "contains", "is_a"]:
        sub = get_sub_family(pred)
        fam = get_family(pred)
        conf = get_confidence(pred)
        print(f"  {pred:20s} -> sub={sub}, family={fam}, conf={conf:.2f}")
