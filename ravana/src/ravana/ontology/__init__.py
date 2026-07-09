"""Derived ontology package — brain-inspired, compute-on-demand category and
attribute knowledge over the GloVe manifold + learned concept graph.

Replaces hand-edited literal tables (the old frontopolar gate dicts) with
geometric + graph-derived inference.
"""

from .derived import (
    DerivedOntology,
    _PROPERTY_PROBE_SEEDS,
    _KNOWN_PROPERTIES,
)

__all__ = [
    "DerivedOntology",
    "_PROPERTY_PROBE_SEEDS",
    "_KNOWN_PROPERTIES",
]
