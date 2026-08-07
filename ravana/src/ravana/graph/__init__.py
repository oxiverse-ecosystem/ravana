"""
Graph Engine Package for RAVANA.
"""
from .engine import GraphEngine, TEEN_CONCEPTS, DOMAIN_CONCEPTS, CONTRASTIVE_PAIRS, CAUSAL_PAIRS, IS_A_PAIRS, STOP_WORDS

# Backward compatibility - re-export from ravana_ml.graph (optional)
try:
    from ravana_ml.graph import ConceptGraph, ConceptNode, ConceptEdge, ConceptBindingMap
except ImportError:
    from ravana_ml._import_guard_shim import report_missing
    report_missing("ravana_ml.graph", "ConceptGraph world-model re-export", kind="internal")
    ConceptGraph = None
    ConceptNode = None
    ConceptEdge = None
    ConceptBindingMap = None

__all__ = [
    'GraphEngine',
    'TEEN_CONCEPTS',
    'DOMAIN_CONCEPTS',
    'CONTRASTIVE_PAIRS',
    'CAUSAL_PAIRS',
    'IS_A_PAIRS',
    'STOP_WORDS',
    # Backward compat
    'ConceptGraph',
    'ConceptNode',
    'ConceptEdge',
    'ConceptBindingMap',
]