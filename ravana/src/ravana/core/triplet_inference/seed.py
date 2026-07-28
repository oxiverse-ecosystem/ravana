"""Seed triples — minimal exemplars, NOT inference rules.

These bare facts give the pattern miners something to notice on day
one. They carry low confidence and the "seed" source tag; they are
stored WITHOUT being counted as learning evidence (see
TripletInferenceOperator.__init__), so no profile score moves until
real experience arrives. There is no RelationOntology in this repo:
predicates and their profiles are discovered lazily from experience.
"""
from __future__ import annotations

from .core import Triple

SEED_TRIPLES = [
    # Taxonomic scaffold
    Triple("entity", "has", "property", source="seed", confidence=0.4),
    Triple("category", "contains", "instance", source="seed", confidence=0.4),
    # Temporal ordering
    Triple("past", "precedes", "present", source="seed", confidence=0.3),
    Triple("present", "precedes", "future", source="seed", confidence=0.3),
    # Mereological
    Triple("whole", "contains", "part", source="seed", confidence=0.4),
]
